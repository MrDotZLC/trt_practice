#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/gpt2_model_builder.hpp"
#include "mini_trt_llm/core/resnet18_model_builder.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <NvOnnxParser.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

namespace mini_trt_llm {

namespace {

// **手工维护的"图版本"**：任何改动建图 / 精度 / 插件行为的代码变更都要 +1。
//
// 为什么手工而不是自动：运行期无法察觉"图代码变了"。自动方案只有把编译时间戳写进指纹，
// 但那会让任何一次无关重编（例如改了注释）都强制重建引擎（分钟级），代价不成比例。
// 于是采用"约定 + 指纹里的其他项兜底"：配置、精度、源文件身份、TRT/CUDA 版本都是自动的，
// 只有"图代码本身的代次"需要人手动声明。忘记 +1 的后果是复用旧引擎——这与本功能之前的行为等价，
// 不会比现状更差。
//
// **1 → 2（2026-09-27）**：`PagedAttentionPlugin` 的 `getWorkspaceSize` 从 0 变成正数
// （decode 注意力改成上下文维切开 + 两阶段归约），插件版本也提到 "2"。
// 旧引擎是在"这个插件不需要 workspace"的前提下建的，复用它会**往 0 字节缓冲里写**——
// 而指纹默认看不见插件源码的变化（它只看模型/配置的 size+mtime），所以这里必须手工声明代次
// （开发计划 §12.8 第 1 条）。副作用是下次真机第一次会重建全部引擎（分钟级），那是预期的。
// 建图代码代次。任何改变图行为的改动都必须 +1，否则引擎缓存会判定"未过期"而沿用旧图
// （指纹不含建图代码本身）。
// 3: prefill 图新增 padding_bias 输入（REQ-016 S3，支持批内 prompt 长度不齐）
constexpr int32_t kEngineGraphVersion = 3;
// **4: S4 的 packed 混合批 prefill 图**（一个 packed 张量装两相、attention 按段分派）。
// 它与 version 3 的 padding prefill 图**并存**（由 `Config::packed_mixed_prefill` 选），
// 两套图的 I/O 契约不同（多 6 个输入、K/V 输出换成 token-major 的 [T,NH,D]、去掉 padding_bias），
// 所以用各自代次进指纹 —— 复用错引擎的后果是"按旧契约绑张量"，那是静默错而不是报错。
// **4 → 5（2026-10-04，S5）**：packed 图的 context 段从"自包含 varlen 注意力"变成
// **chunked 分页因果注意力**（K/V = 分页缓存前缀 ++ 本 chunk，见 `packed_attention_plugin.cu`）。
// I/O 契约、profile 区间与 `getWorkspaceSize` 都没变，但**插件对同一绑定的计算语义变了** ——
// 指纹看不见插件源码的变化，复用按旧语义建的引擎会让分块后的结果**静默算错**，所以手工 +1。
// 依据：本文件开头的 1 → 2（`PagedAttentionPlugin::getWorkspaceSize` 从 0 变正数）同一条规矩。
// **5 → 6（2026-10-04，S5 收口）**：`cu_seqlens_ctx` 的**行维 profile 上界**从 `max_prefill_batch`
// 改成 `max_prefill_batch + 1`（它的长度是 B_ctx + 1，最后一个元素是段内 token 总数；用行维范围
// 会让"整批都是 context 行"的首步 `setInputShape` 直接失败）。profile 区间也是指纹看不见的
// 建图产物 → 同样手工 +1，否则缓存里的旧引擎会带着过小的上界被复用。
constexpr int32_t kPackedPrefillGraphVersion = 6;

const char* StageName(BuildStage stage) {
    switch (stage) {
        case BuildStage::kSingle: return "single";
        case BuildStage::kPrefill: return "prefill";
        case BuildStage::kDecode: return "decode";
    }
    return "unknown";
}

}  // namespace

OnnxIoContract OnnxIoContractFor(const std::string& architecture) {
    // 只有 cnn 是"像素质进、logits 出"的形态；其余（decoder_only / encoder_decoder）
    // 都是 token 进、logits 出。默认走 LLM 契约：它是既有行为，改默认会静默影响 Phase 3 的用例。
    if (architecture == "cnn") {
        return {"input", "output"};
    }
    return {"input_ids", "logits"};
}

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
                  const std::function<DimRange(const std::string&, int32_t)>& range_for_dim) {
    nvinfer1::Dims min_dims = tensor->getDimensions();
    nvinfer1::Dims opt_dims = min_dims;
    nvinfer1::Dims max_dims = min_dims;

    bool has_dynamic_dim = false;
    for (int32_t dim = 0; dim < min_dims.nbDims; ++dim) {
        if (min_dims.d[dim] >= 0) {
            continue;
        }
        const DimRange range = range_for_dim(tensor->getName(), dim);
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
    : logger_(logger), config_(config), registry_(std::make_unique<ModelRegistry>()) {
    // 内置模型随 EngineBuilder 一起可用，调用方不必先知道有哪些模型；
    // 外部/测试用模型仍可通过 RegisterModelBuilder 覆盖或追加。
    registry_->Register("gpt2", std::make_shared<GPT2ModelBuilder>());
    registry_->Register("resnet18", std::make_shared<ResNet18ModelBuilder>());
}

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
    // INT8 不需要任何 builder flag：`kINT8` 自 TRT 10.12 起废弃（由 Q/DQ 显式量化取代），
    // 引擎的精度由网络里的 Q/DQ 节点决定。实测（手搓对称 Q/DQ 最小图）：弱类型网络下
    // Q/DQ 被正常接受，且 Q/DQ 与 Conv 会融合。见 docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md + PH4-INT8-TOOLCHAIN 的 S1-b。

    if (config_.detailed_profiling) {
        // 逐层精度只在 DETAILED 下写进引擎；默认（kLAYER_NAMES_ONLY）读不出精度，
        // 也就无法自证"引擎真的在跑 INT8"。见 Config 里的注释与该配置项的出处。
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    }

    return true;
}

bool EngineBuilder::SerializeAndSave(nvinfer1::IHostMemory* serialized,
                                     const std::string& engine_path,
                                     const std::string& fingerprint,
                                     const EngineFingerprintInputs& fingerprint_inputs) {
    if (!serialized) {
        MINI_TRT_LOG_ERROR("serialize engine failed");
        return false;
    }
    WriteFile(engine_path, serialized->data(), serialized->size());
    MINI_TRT_LOG_INFO("Engine saved: " << engine_path
                    << " (" << serialized->size() / 1024 / 1024 << " MB)");
    // 指纹与引擎一起落盘：下次运行要能判断"这份引擎是不是当前配置/当前代码的产物"。
    // 写失败不算致命（引擎本身可用），但要把话说清楚，否则下次会静默重建、让人以为是别的问题。
    if (!WriteEngineFingerprint(engine_path, fingerprint, fingerprint_inputs)) {
        MINI_TRT_LOG_WARN("引擎指纹写入失败：" << EngineFingerprintPath(engine_path)
                        << "（下次运行会重建这份引擎）");
    }
    return true;
}

EngineFingerprintInputs EngineBuilder::MakeFingerprintInputs(const std::string& model_dir,
                                                            const std::string& onnx_path,
                                                            BuildStage stage) const {
    EngineFingerprintInputs inputs;
    inputs.stage = StageName(stage);
    inputs.precision = PrecisionString(config_.precision);
    inputs.source_kind = onnx_path.empty() ? "config" : "onnx";
    // packed 混合批图与 padding prefill 图是两套契约，各用各的代次（见上面两个常量的说明）。
    inputs.graph_version = config_.packed_mixed_prefill ? kPackedPrefillGraphVersion
                                                        : kEngineGraphVersion;
    inputs.trt_version = std::to_string(getInferLibVersion());
    int cuda_version = 0;
    if (cudaRuntimeGetVersion(&cuda_version) == cudaSuccess) {
        inputs.cuda_runtime_version = cuda_version;
    }

    // 源文件身份：配置 + 权重（方案 A）/ ONNX 图（方案 B）。缺文件的项也会进入指纹（identity=missing），
    // 于是"文件从无到有"同样会让指纹变化。
    inputs.source_files = {model_dir + "/config.json"};
    if (onnx_path.empty()) {
        inputs.source_files.push_back(model_dir + "/model.safetensors");
    } else {
        inputs.source_files.push_back(onnx_path);
    }

    // 所有影响建图的数值参数都要进来：漏掉任何一个都会让"改了范围却复用旧引擎"重新变成一个坑。
    inputs.numeric_params = {
        {"workspace_bytes", static_cast<int64_t>(config_.workspace_bytes)},
        {"cv.min_batch", config_.min_batch},
        {"cv.opt_batch", config_.opt_batch},
        {"cv.max_batch", config_.max_batch},
        {"prefill.min_batch", config_.min_prefill_batch},
        {"prefill.opt_batch", config_.opt_prefill_batch},
        {"prefill.max_batch", config_.max_prefill_batch},
        {"prefill.min_seq", config_.min_prefill_seq_len},
        {"prefill.opt_seq", config_.opt_prefill_seq_len},
        {"prefill.max_seq", config_.max_prefill_seq_len},
        {"decode.min_batch", config_.min_decode_batch},
        {"decode.opt_batch", config_.opt_decode_batch},
        {"decode.max_batch", config_.max_decode_batch},
        {"decode.min_seq", config_.min_decode_seq_len},
        {"decode.opt_seq", config_.opt_decode_seq_len},
        {"decode.max_seq", config_.max_decode_seq_len},
    };
    inputs.flags = {
        {"export_diagnostics", config_.export_diagnostics},
        {"detailed_profiling", config_.detailed_profiling},
    };
    return inputs;
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
    // 动态分辨率属于后续迭代（见 docs/future_iterations.md + OI-CV-DYNAMIC-RES），此处明确拒绝而非猜测。
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
                                               nvinfer1::INetworkDefinition* network,
                                               BuildStage stage) {
    if (!HasDynamicInputDim(network)) {
        return true;
    }

    // 布局约定（非 packed 路径）：第 0 维是 batch，其余动态维一律视为序列相关维。
    // 这样对 [B, S, H] 与 RoPE 的 [B, H, S, D] 都成立，无需为每种布局写一份规则。
    //
    // 已知限制：Decode profile 把非 batch 动态维固定为 1（对应"每步只喂 1 个 token"），
    // 因此 KV Cache 长度维目前不能声明为动态。接入真实 LLM 网络（Phase 2）时需细化，
    // 届时该维应单独取 max_decode_seq_len。
    // （packed 图不适用这条约定：它有两条独立动态轴且落在不同输入上 —— 见下面的 packed_* 说明。）
    const DimRange prefill_batch{config_.min_prefill_batch, config_.opt_prefill_batch,
                                 config_.max_prefill_batch};
    const DimRange prefill_seq{config_.min_prefill_seq_len, config_.opt_prefill_seq_len,
                               config_.max_prefill_seq_len};
    const DimRange decode_batch{config_.min_decode_batch, config_.opt_decode_batch,
                                config_.max_decode_batch};
    const DimRange decode_seq{1, 1, 1};

    // **S4 packed 图有两条独立动态轴，且落在不同输入上**（见 p5_s4_interface_spec.md §7）：
    //   * `input_ids` / `position_ids`：dim0 固定 1、dim1 是 **token 维 T**；
    //   * `block_tables` / `context_lens`：dim0 是 **行维 B_total**；
    //   * `cu_seqlens_ctx`：dim0 是 **段内前缀和的长度 B_ctx + 1**（最后一个元素是段内 token 总数）
    //     —— 比行维多一格，所以它单独一组范围（`packed_rows` 的上界会让"整批都是 context 行"
    //     的首步越界：B_ctx = B 时长度为 B + 1）。
    // 所以不能沿用"按 dim 下标对所有输入套同一组范围"——那会把 block_tables 的行维钉成 1，
    // `B_total > 1` 直接越界（这正是 ApplyProfile 现在收**输入名**的原因）。
    // 两个 opt 都**先取保守值并标注待实测**（作者 2026-10-04 定：opt 等 P4 实测后定）。
    const DimRange packed_tokens{1, config_.opt_prefill_seq_len,
                                 config_.max_prefill_batch * config_.max_prefill_seq_len};
    const DimRange packed_rows{1, config_.max_prefill_batch, config_.max_prefill_batch};
    const DimRange packed_ctx_prefix{1, config_.max_prefill_batch + 1,
                                     config_.max_prefill_batch + 1};

    // 按 stage 过滤要挂哪些 profile：双引擎方案下每个 engine 只该有自己那一组，
    // 多挂一组不会报错，但会让 TRT 为用不到的形状多编译一份 kernel（GPT-2 上是分钟级开销）。
    std::vector<std::pair<DimRange, DimRange>> profiles;
    if (stage == BuildStage::kSingle || stage == BuildStage::kPrefill) {
        profiles.emplace_back(prefill_batch, prefill_seq);
    }
    if (stage == BuildStage::kSingle || stage == BuildStage::kDecode) {
        profiles.emplace_back(decode_batch, decode_seq);
    }

    for (const auto& [batch_range, seq_range] : profiles) {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        NVINFER_CHECK(profile);
        // 按 stage 的**语义**选范围映射：packed prefill 图用"按输入名"的那套，
        // 其余（padding prefill / decode / single）保持"按 dim 下标"的原行为。
        const bool packed_profile =
            config_.packed_mixed_prefill && (stage == BuildStage::kPrefill);
        const auto range_for_dim =
            [&](const std::string& name, int32_t dim) -> DimRange {
            if (!packed_profile) {
                return dim == 0 ? batch_range : seq_range;
            }
            if (name == "input_ids" || name == "position_ids") {
                return dim == 0 ? DimRange{1, 1, 1} : packed_tokens;
            }
            if (name == "cu_seqlens_ctx") {
                return dim == 0 ? packed_ctx_prefix : DimRange{1, 1, 1};
            }
            return dim == 0 ? packed_rows : DimRange{1, 1, 1};
        };
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
                                    const std::string& engine_path,
                                    BuildStage stage) {
    // 缓存检查放在最前面（在任何昂贵动作之前）：指纹一致就直接复用，
    // 指纹不一致**必须重建**——这正是"缓存不随代码失效"那个老坑的堵法（见 core/engine_cache.hpp）。
    const EngineFingerprintInputs fingerprint_inputs =
        MakeFingerprintInputs(model_dir, /*onnx_path=*/{}, stage);
    const std::string fingerprint = ComputeEngineFingerprint(fingerprint_inputs);
    if (EngineCacheIsFresh(engine_path, fingerprint)) {
        MINI_TRT_LOG_INFO("Engine cache hit: " << engine_path << "（指纹一致，跳过重建）");
        return true;
    }
    if (std::filesystem::exists(engine_path)) {
        // 说清楚"为什么不复用"：否则下一个人只会看到构建耗时，却不知道是配置变了还是代码变了。
        MINI_TRT_LOG_WARN("Engine cache stale: " << engine_path
                         << "（指纹与当前配置/代码不一致 → 重建；旧引擎将被覆盖）");
    }

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

    BuildOptions build_options;
    build_options.stage = stage;
    build_options.weight_dtype = ToTrtDataType(config_.precision);
    build_options.export_diagnostics = config_.export_diagnostics;
    // S4：把 packed 混合批图的开关透给模型构建器（只有 prefill 阶段有意义，构建器会自行校验）。
    build_options.packed_mixed = config_.packed_mixed_prefill;

    if (!builder_impl->Build(network.get(), weights, model_config, build_options)) {
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
        if (!AddLlmOptimizationProfiles(builder.get(), trt_config.get(), network.get(),
                                        stage)) {
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
    return SerializeAndSave(serialized.get(), engine_path, fingerprint, fingerprint_inputs);
}

bool EngineBuilder::BuildFromOnnx(const std::string& model_dir,
                                  const std::string& onnx_path,
                                  const std::string& engine_path,
                                  const std::vector<std::string>& subgraph_names) {
    // 与 BuildFromConfig 同一套语义：指纹一致即复用，不一致即重建。
    // ONNX 路径的指纹里含 ONNX 文件身份——重新导出过图必须重建。
    const EngineFingerprintInputs fingerprint_inputs =
        MakeFingerprintInputs(model_dir, onnx_path, BuildStage::kSingle);
    const std::string fingerprint = ComputeEngineFingerprint(fingerprint_inputs);
    if (EngineCacheIsFresh(engine_path, fingerprint)) {
        MINI_TRT_LOG_INFO("Engine cache hit: " << engine_path << "（指纹一致，跳过重建）");
        return true;
    }
    if (std::filesystem::exists(engine_path)) {
        MINI_TRT_LOG_WARN("Engine cache stale: " << engine_path
                         << "（指纹与当前配置/代码/ONNX 不一致 → 重建；旧引擎将被覆盖）");
    }

    // 先做纯数据校验（与方案 A 同样的顺序理由：createInferBuilder 既慢又依赖驱动，
    // 把它排在文件检查之后能显著降低失败路径的成本）。
    ModelConfig model_config;
    try {
        model_config = ModelConfig::Load(model_dir);
    } catch (const std::exception& e) {
        MINI_TRT_LOG_ERROR("Failed to load model config: " << e.what());
        return false;
    }
    {
        // 只做存在性检查：读整个 652MB 图来验证"文件能打开"没有意义，
        // 真正的解析交给下面的 nvonnxparser（它自己会报逐条错误）。
        std::ifstream probe(onnx_path, std::ios::binary);
        if (!probe.good()) {
            MINI_TRT_LOG_ERROR("ONNX file not found or unreadable: " << onnx_path);
            return false;
        }
    }

    // 已支持核对的子图名。当前阶段（D1=C）不做替换，所以这里只声明"能核对哪些"，
    // 真正的结构识别在 tools/inspect_onnx.py；写错名字即失败，
    // 避免出现"声明了子图却没核对"的假安全感。
    const std::vector<std::string> known_subgraphs = {"attention", "layernorm",
                                                      "position_embedding"};
    for (const std::string& name : subgraph_names) {
        if (std::find(known_subgraphs.begin(), known_subgraphs.end(), name) ==
            known_subgraphs.end()) {
            MINI_TRT_LOG_ERROR("Unknown subgraph name: " << name
                                << " (supported: attention / layernorm / position_embedding)");
            return false;
        }
    }
    if (!subgraph_names.empty()) {
        MINI_TRT_LOG_INFO("ONNX 构建将核对子图: " << subgraph_names.size()
                        << " 个（识别由 tools/inspect_onnx.py --check 落地）");
    }

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

    // ONNX 侧的输入输出契约：方案 B 的图必须与方案 A 同名同义，
    // 否则"两条路对齐"无从谈起；名字对不上就直接失败，而不是等到绑定时才报错。
    //
    // 契约随 architecture 变（LLM: input_ids/logits；CV: input/output）——见 OnnxIoContractFor。
    const OnnxIoContract io = OnnxIoContractFor(model_config.architecture);
    if (network->getNbInputs() < 1 ||
        std::string(network->getInput(0)->getName()) != io.input_name) {
        MINI_TRT_LOG_ERROR("ONNX graph: expected an input named '" << io.input_name << "', got '"
                           << (network->getNbInputs() > 0 ? network->getInput(0)->getName()
                                                          : "<none>")
                           << "' (architecture=" << model_config.architecture << ")");
        return false;
    }
    bool has_expected_output = false;
    for (int32_t i = 0; i < network->getNbOutputs(); ++i) {
        if (std::string(network->getOutput(i)->getName()) == io.output_name) {
            has_expected_output = true;
        }
    }
    if (!has_expected_output) {
        MINI_TRT_LOG_ERROR("ONNX graph: no output named '" << io.output_name
                           << "' (architecture=" << model_config.architecture << ")");
        return false;
    }

    // profile：ONNX 图是整段前向（无 KV cache 输入），因此只挂 **prefill** 那一组。
    // 依据（D4 的历史工程核对）：`1_gpt2_onnx/src/builder.cpp` 当年也是**单个** profile，
    // 取值 min[1,1] / opt[1,64] / max[4,512]，与 EngineBuilder::Config 的 prefill 默认值一致。
    // 该历史工程已于 Phase 5 下线；要复核原始实现见删除前的提交
    // `git show ba3ea7a:1_gpt2_onnx/src/builder.cpp`（`docs/dev/REQ-009-retire-legacy/phase5_development_plan.md`）。
    // 多挂一组 decode profile 不会错，但会让 TRT 白编译一份用不到的形状。
    if (model_config.architecture == "cnn") {
        if (!AddCvOptimizationProfile(builder.get(), trt_config.get(), network.get())) {
            MINI_TRT_LOG_ERROR("Failed to set up CV optimization profile for ONNX");
            return false;
        }
    } else {
        if (!AddLlmOptimizationProfiles(builder.get(), trt_config.get(), network.get(),
                                        BuildStage::kPrefill)) {
            MINI_TRT_LOG_ERROR("Failed to set up LLM prefill profile for ONNX");
            return false;
        }
    }

    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *trt_config));
    return SerializeAndSave(serialized.get(), engine_path, fingerprint, fingerprint_inputs);
}

}  // namespace mini_trt_llm
