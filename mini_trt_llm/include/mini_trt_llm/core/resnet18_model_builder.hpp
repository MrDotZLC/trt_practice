#pragma once

#include "mini_trt_llm/core/imodel_builder.hpp"

#include <string>

namespace mini_trt_llm {

// ResNet18 的原生建图（Phase 4 / P4-5）。
//
// **与 ONNX 路径的关系**：两条路必须建出同名同义的 I/O（`input` / `output`，见
// `OnnxIoContractFor`）并消费**同一份权重**——权重由
// `tools/convert/onnx_to_mini_trt_llm.py` 从那份 ONNX 导出（BN 已折叠），
// 因此"原生 vs ONNX"的对拍差异只剩实现差异，不掺权重差异。
//
// **为什么不需要任何 Plugin**：ResNet18 的算子（Conv / Relu / Add / MaxPool /
// GlobalAveragePool / Flatten / Gemm）TRT 全有原生实现，且 BatchNorm 已在导出时折叠进 Conv。
//
// **stage 语义**：CV 没有 prefill/decode 之分，只接受 `BuildStage::kSingle`；
// 传别的 stage 会**显式失败**（静默忽略会建出一个语义不明的引擎）。
class ResNet18ModelBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "resnet18"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader& weights,
               const ModelConfig& config, const BuildOptions& options) override;
};

}  // namespace mini_trt_llm
