#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <stdexcept>

namespace mini_trt_llm {

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

    std::unique_ptr<nvinfer1::IBuilder> builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    std::unique_ptr<nvinfer1::IBuilderConfig> trt_config;
    if (!SetupBuilder(builder, network, trt_config)) {
        return false;
    }

    WeightLoader weights;
    if (!weights.Load(model_dir)) {
        return false;
    }
    weights.SetWeightMap(model_config.weight_map);
    weights.SetDefaultPrecision(ToTrtDataType(config_.precision));

    if (!builder_impl->Build(network.get(), weights, model_config)) {
        MINI_TRT_LOG_ERROR("Model builder failed: " << builder_impl->Name());
        return false;
    }

    // TODO: 根据 architecture 添加 optimization profile
    // Phase 0 先不设置动态 profile，由 Phase 2/4 具体 builder 扩展。

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
