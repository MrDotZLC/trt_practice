#pragma once

#include "mini_trt_llm/core/model_registry.hpp"
#include "mini_trt_llm/core/precision.hpp"
#include "logger.hpp"
#include <NvInfer.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {

// 统一 Engine 构建入口。
// 支持两种模式：
//   1. BuildFromConfig：从 Safetensors + JSON config 原生构建。
//   2. BuildFromOnnx：从 ONNX + 自定义 Plugin 替换构建。
class EngineBuilder {
 public:
    struct Config {
        // 默认 FP16：Turing sm_75 对 FP16 支持较好，且为本项目目标精度。
        Precision precision = Precision::FP16;

        // TensorRT builder 工作区上限，默认 1GB（1UL << 30）。
        // 复杂 Plugin（如 PagedAttention）可能需要更大空间，后续可配置。
        size_t workspace_bytes = 1UL << 30;

        // CV 动态 batch 范围，覆盖 ResNet18 等 CNN 模型常见 batch。
        int min_batch = 1;
        int opt_batch = 1;
        int max_batch = 16;

        // LLM Prefill 阶段动态 shape 范围。
        // 默认值基于 GTX 1660 Ti 6GB 显存与常见 prompt 长度设定。
        int min_prefill_batch = 1;
        int opt_prefill_batch = 1;
        int max_prefill_batch = 4;
        int min_prefill_seq_len = 1;
        int opt_prefill_seq_len = 64;
        int max_prefill_seq_len = 512;

        // LLM Decode 阶段动态 shape 范围。
        // decode 阶段 seq_len 固定为 1，max_decode_seq_len 指已生成序列上限。
        int min_decode_batch = 1;
        int opt_decode_batch = 1;
        int max_decode_batch = 4;
        int min_decode_seq_len = 1;
        int opt_decode_seq_len = 1;
        int max_decode_seq_len = 512;
    };

    explicit EngineBuilder(Logger& logger, const Config& config);
    ~EngineBuilder();

    // 注册模型构建器。EngineBuilder 持有 registry，但具体 builder 可外部注入。
    void RegisterModelBuilder(const std::string& name,
                              std::shared_ptr<IModelBuilder> builder);

    // 方案 A：从配置构建
    bool BuildFromConfig(const std::string& model_dir,
                         const std::string& engine_path);

    // 方案 B：从 ONNX + Plugin 替换构建
    bool BuildFromOnnx(const std::string& onnx_path,
                       const std::string& engine_path,
                       const std::vector<std::string>& plugin_ops);

    ModelRegistry* GetRegistry() { return registry_.get(); }

 private:
    bool SetupBuilder(std::unique_ptr<nvinfer1::IBuilder>& builder,
                      std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                      std::unique_ptr<nvinfer1::IBuilderConfig>& config);

    bool AddCvOptimizationProfile(nvinfer1::IOptimizationProfile* profile,
                                  nvinfer1::INetworkDefinition* network);

    bool AddLlmOptimizationProfiles(nvinfer1::IBuilderConfig* config,
                                    nvinfer1::INetworkDefinition* network);

    bool SerializeAndSave(nvinfer1::IHostMemory* serialized,
                          const std::string& engine_path);

    Logger& logger_;
    Config config_;
    std::unique_ptr<ModelRegistry> registry_;
};

}  // namespace mini_trt_llm
