#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <functional>
#include <utility>
#include <stdexcept>

namespace mini_trt_llm {
namespace {

struct DimRange {
    int32_t min = 0;
    int32_t opt = 0;
    int32_t max = 0;
};

// 一个动态维无法确定取值范围时的哨兵值，与真实范围（min >= 1）区分开。
constexpr DimRange kUnsupportedRange{-1, -1, -1};

// 按 range_for_dim 给出的范围填充 profile 的 min / opt / max。
//
// 只处理动态维（d < 0）：静态维无需出现在 profile 里，TRT 要求的是每个动态输入维
// 都必须有 min/opt/max。整张量全静态时直接跳过，不创建空 profile。
bool ApplyProfile(nvinfer1::IOptimizationProfile* profile,
                  const nvinfer1::ITensor* tensor,
                  const std::function<DimRange(int32_t)>& range_for_dim) {
    nvinfer1::Dims min_dims = tensor->getDimensions();
    nvinfer1::Dims opt_dims = min_dims;
    nvinfer1::Dims max_dims = min_dims;

    bool has_dynamic_dim = false;
    for (int32_t dim = 0; dim < min_dims.nbDims; ++dim) {
        if (min_dims.d[dim] >= 0) {
            continue;
        }
        const DimRange range = range_for_dim(dim);
        if (range.min <= 0 || range.opt < range.min || range.max < range.opt) {
            MINI_TRT_LOG_ERROR("Cannot determine optimization range for dynamic dim "
                               << dim << " of tensor " << tensor->getName());
            return false;
        }
        min_dims.d[dim] = range.min;
        opt_dims.d[dim] = range.opt;
        max_dims.d[dim] = range.max;
        has_dynamic_dim = true;
    }

    if (!has_dynamic_dim) {
        return true;
    }
    return profile->setDimensions(tensor->getName(),
                                  nvinfer1::OptProfileSelector::kMIN, min_dims) &&
           profile->setDimensions(tensor->getName(),
                                  nvinfer1::OptProfileSelector::kOPT, opt_dims) &&
           profile->setDimensions(tensor->getName(),
                                  nvinfer1::OptProfileSelector::kMAX, max_dims);
}

bool HasDynamicInputDim(nvinfer1::INetworkDefinition* network) {
    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        const nvinfer1::Dims dims = network->getInput(i)->getDimensions();
        for (int32_t dim = 0; dim < dims.nbDims; ++dim) {
            if (dims.d[dim] < 0) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

EngineBuilder::EngineBuilder(Logger& logger, const Config& config)
    : logger_(logger), config_(config), registry_(std::make_unique<ModelRegistry>()) {}

EngineBuilder::~EngineBuilder() = default;

void EngineBuilder::RegisterModelBuilder(const std::string& name,
                                         std::shared_ptr<IModelBuilder> builder) {
    registry_->Register(name, builder);
}

bool EngineBuilder::SetupBuilder(
    std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    std::unique_ptr<nvinfer1::IBuilderConfig>& config) {
    builder.reset(nvinfer1::createInferBuilder(logger_));
    NVINFER_CHECK(builder);

    network.reset(builder->createNetworkV2(0U));
    NVINFER_CHECK(network);

    config.reset(builder->createBuilderConfig());
    NVINFER_CHECK(config);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               config_.workspace_bytes);

    if (config_.precision == Precision::FP16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // INT8 后续迭代实现

    return true;
}

bool EngineBuilder::SerializeAndSave(nvinfer1::IHostMemory* serialized,
                                     const std::string& engine_path) {
    if (!serialized) {
        MINI_TRT_LOG_ERROR("serialize engine failed");
        return false;
    }
    WriteFile(engine_path, serialized->data(), serialized->size());
    MINI_TRT_LOG_INFO("Engine saved: " << engine_path
                    << " (" << serialized->size() / 1024 / 1024 << " MB)");
    return true;
}

bool EngineBuilder::AddCvOptimizationProfile(nvinfer1::IBuilder* builder,
                                             nvinfer1::IBuilderConfig* config,
                                             nvinfer1::INetworkDefinition* network) {
    if (!HasDynamicInputDim(network)) {
        // 全静态网络不需要任何 profile，TRT 也不会接受空 profile
        return true;
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    NVINFER_CHECK(profile);

    // CV Phase 1 只支持动态 batch：只允许第 0 维是动态的。
    // 动态分辨率属于后续迭代（见 docs/future_iterations.md §2.4），此处明确拒绝而非猜测。
    auto range_for_dim = [this](int32_t dim) -> DimRange {
        if (dim == 0) {
            return DimRange{config_.min_batch, config_.opt_batch, config_.max_batch};
        }
        return kUnsupportedRange;
    };

    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        if (!ApplyProfile(profile, network->getInput(i), range_for_dim)) {
            return false;
        }
    }
    return config->addOptimizationProfile(profile) >= 0;
}

bool EngineBuilder::AddLlmOptimizationProfiles(nvinfer1::IBuilder* builder,
                                               nvinfer1::IBuilderConfig* config,
                                               nvinfer1::INetworkDefinition* network) {
    if (!HasDynamicInputDim(network)) {
        return true;
    }

    // 布局约定：第 0 维是 batch，其余动态维一律视为序列相关维。
    // 这样对 [B, S, H] 与 RoPE 的 [B, H, S, D] 都成立，无需为每种布局写一份规则。
    //
    // 已知限制：Decode profile 把非 batch 动态维固定为 1（对应"每步只喂 1 个 token"），
    // 因此 KV Cache 长度维目前不能声明为动态。接入真实 LLM 网络（Phase 2）时需细化，
    // 届时该维应单独取 max_decode_seq_len。
    auto make_range = [this](const DimRange& batch_range, const DimRange& seq_range) {
        return [batch_range, seq_range](int32_t dim) -> DimRange {
            return dim == 0 ? batch_range : seq_range;
        };
    };

    const DimRange prefill_batch{config_.min_prefill_batch, config_.opt_prefill_batch,
                                 config_.max_prefill_batch};
    const DimRange prefill_seq{config_.min_prefill_seq_len, config_.opt_prefill_seq_len,
                               config_.max_prefill_seq_len};
    const DimRange decode_batch{config_.min_decode_batch, config_.opt_decode_batch,
                                config_.max_decode_batch};
    const DimRange decode_seq{1, 1, 1};

    const std::pair<DimRange, DimRange> profiles[] = {
        {prefill_batch, prefill_seq}, {decode_batch, decode_seq}};

    for (const auto& [batch_range, seq_range] : profiles) {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        NVINFER_CHECK(profile);
        const auto range_for_dim = make_range(batch_range, seq_range);
        for (int32_t i = 0; i < network->getNbInputs(); ++i) {
            if (!ApplyProfile(profile, network->getInput(i), range_for_dim)) {
                return false;
            }
        }
        if (config->addOptimizationProfile(profile) < 0) {
            return false;
        }
    }
    return true;
}

bool EngineBuilder::BuildFromConfig(const std::string& model_dir,
                                    const std::string& engine_path) {
    ModelConfig model_config;
    try {
        model_config = ModelConfig::Load(model_dir);
    } catch (const std::exception& e) {
        MINI_TRT_LOG_ERROR("Failed to load model config: " << e.what());
        return false;
    }

    auto builder_impl = registry_->Get(model_config.model_type);
    if (!builder_impl) {
        MINI_TRT_LOG_ERROR("No registered builder for model type: "
                         << model_config.model_type);
        return false;
    }

    // 先做纯数据校验（配置 + 权重文件），再初始化 TensorRT。
    // 这样"权重缺失 / 文件名写错"这类问题不必等到创建 IBuilder 才暴露——
    // createInferBuilder 既慢又依赖 GPU 驱动，把它排在纯文件检查之后能显著降低失败路径的成本。
    WeightLoader weights;
    if (!weights.Load(model_dir)) {
        return false;
    }
    weights.SetWeightMap(model_config.weight_map);
    weights.SetDefaultPrecision(ToTrtDataType(config_.precision));

    std::unique_ptr<nvinfer1::IBuilder> builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    std::unique_ptr<nvinfer1::IBuilderConfig> trt_config;
    if (!SetupBuilder(builder, network, trt_config)) {
        return false;
    }

    if (!builder_impl->Build(network.get(), weights, model_config)) {
        MINI_TRT_LOG_ERROR("Model builder failed: " << builder_impl->Name());
        return false;
    }

    // 按 architecture 选择 profile 组合：CNN 只动态 batch，decoder-only 用 Prefill/Decode 双 profile。
    // 全静态网络会在上面的 check 里直接跳过，不会创建空 profile。
    if (model_config.architecture == "cnn") {
        if (!AddCvOptimizationProfile(builder.get(), trt_config.get(), network.get())) {
            MINI_TRT_LOG_ERROR("Failed to set up CV optimization profile");
            return false;
        }
    } else if (model_config.architecture == "decoder_only" ||
               model_config.architecture == "encoder_decoder") {
        if (!AddLlmOptimizationProfiles(builder.get(), trt_config.get(), network.get())) {
            MINI_TRT_LOG_ERROR("Failed to set up LLM optimization profiles");
            return false;
        }
    } else if (HasDynamicInputDim(network.get())) {
        // 架构未知但输入含动态维：宁可直接失败，也不要建出一个运行期形状对不上的 engine
        MINI_TRT_LOG_ERROR("Network has dynamic inputs but architecture '"
                           << model_config.architecture
                           << "' has no optimization profile rule");
        return false;
    }

    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *trt_config));
    return SerializeAndSave(serialized.get(), engine_path);
}

bool EngineBuilder::BuildFromOnnx(const std::string& onnx_path,
                                  const std::string& engine_path,
                                  const std::vector<std::string>& plugin_ops) {
    std::unique_ptr<nvinfer1::IBuilder> builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    std::unique_ptr<nvinfer1::IBuilderConfig> trt_config;
    if (!SetupBuilder(builder, network, trt_config)) {
        return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    NVINFER_CHECK(parser);

    if (!parser->parseFromFile(
            onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            MINI_TRT_LOG_ERROR("ONNX parse error: "
                             << parser->getError(i)->desc());
        }
        return false;
    }

    // TODO(Phase 3): 根据 plugin_ops 进行子图替换
    (void)plugin_ops;

    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *trt_config));
    return SerializeAndSave(serialized.get(), engine_path);
}

}  // namespace mini_trt_llm
