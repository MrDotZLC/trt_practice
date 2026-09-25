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

        // 诊断输出开关，默认关。只有"定位某个张量为何出 NaN"这类仪器用例才该打开，
        // 因为它会改变网络的 I/O 契约（多出 4 个中间张量，消费方必须逐个绑定），
        // 语义与后果见 BuildOptions::export_diagnostics。
        bool export_diagnostics = false;
    };

    explicit EngineBuilder(Logger& logger, const Config& config);
    ~EngineBuilder();

    // 注册模型构建器。EngineBuilder 持有 registry，但具体 builder 可外部注入。
    void RegisterModelBuilder(const std::string& name,
                              std::shared_ptr<IModelBuilder> builder);

    // 方案 A：从配置构建
    // stage 决定这次建的是哪一种网络切面（见 BuildStage）：
    //   kSingle  —— 单引擎，挂 Prefill/Decode 两组 profile；
    //   kPrefill —— 只挂 Prefill profile（整段序列前向 + 导出每层 K/V）；
    //   kDecode  —— 只挂 Decode profile（单 token 增量前向）。
    bool BuildFromConfig(const std::string& model_dir,
                         const std::string& engine_path,
                         BuildStage stage = BuildStage::kSingle);

    // 方案 B：从 ONNX 构建。
    //
    // model_dir 提供 `config.json`——profile 规则按 `architecture` 选择，与方案 A **同一套**
    // 语义（否则"ONNX 引擎与原生引擎对齐"这个判据就没有意义）。
    //
    // subgraph_names 是**要识别并核对**的子图名（如 "attention" / "layernorm" /
    // "position_embedding"）。当前阶段（D1=C）不做替换，只做识别声明与校验：
    // 结构识别本身落在 `tools/inspect_onnx.py`（Python 侧不需要 CUDA，C++ 侧的
    // nvonnxparser 需要 `createInferBuilder`，在无 GPU 环境跑不了）。
    // 名字写错即失败——"声明了却没核对"比不声明更危险。
    bool BuildFromOnnx(const std::string& model_dir,
                       const std::string& onnx_path,
                       const std::string& engine_path,
                       const std::vector<std::string>& subgraph_names = {});

    ModelRegistry* GetRegistry() { return registry_.get(); }

 private:
    bool SetupBuilder(std::unique_ptr<nvinfer1::IBuilder>& builder,
                      std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                      std::unique_ptr<nvinfer1::IBuilderConfig>& config);

    // OptimizationProfile 只能由 IBuilder 创建，因此这几个入口都需要 builder。
    bool AddCvOptimizationProfile(nvinfer1::IBuilder* builder,
                                  nvinfer1::IBuilderConfig* config,
                                  nvinfer1::INetworkDefinition* network);

    bool AddLlmOptimizationProfiles(nvinfer1::IBuilder* builder,
                                    nvinfer1::IBuilderConfig* config,
                                    nvinfer1::INetworkDefinition* network,
                                    BuildStage stage);

    bool SerializeAndSave(nvinfer1::IHostMemory* serialized,
                          const std::string& engine_path);

    Logger& logger_;
    Config config_;
    std::unique_ptr<ModelRegistry> registry_;
};

}  // namespace mini_trt_llm
