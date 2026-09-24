#pragma once

#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include <NvInfer.h>
#include <memory>
#include <string>

namespace mini_trt_llm {

// 同一份权重可以建出不同的网络切面。
// 自回归推理天然分两段：整段 prompt 一次前向（Prefill）与单 token 增量前向（Decode），
// 两者的注意力结构不同；双引擎方案下每个 engine 还只应挂自己那一组 optimization profile，
// 否则会为用不到的形状白编译一份。
enum class BuildStage {
    // 一个引擎承载整段序列前向，同时挂 Prefill / Decode 两组 profile（Phase 1.5 的既有行为）。
    kSingle,
    // 只挂 Prefill profile：整段序列前向，并把每层 K/V 导出为网络输出，供写入 KV Cache。
    kPrefill,
    // 只挂 Decode profile：单 token 增量前向，注意力走 PagedAttention。
    kDecode,
};

struct BuildOptions {
    BuildStage stage = BuildStage::kSingle;

    // 权重常量要建成的精度。由 EngineBuilder 从 Precision 映射而来：
    // builder 必须显式知道目标 dtype 才能建出正确精度的常量层，
    // 而 TensorRT 的 builder flag 只表达"整网倾向"，无法回答"这个常量该是什么类型"。
    nvinfer1::DataType weight_dtype = nvinfer1::DataType::kFLOAT;
};

// 模型构建器抽象接口。
// 每种模型（GPT2, ResNet18 等）实现一个具体子类并注册到 ModelRegistry。
class IModelBuilder {
 public:
    virtual ~IModelBuilder() = default;

    // 模型类型名，如 "gpt2", "resnet18"
    virtual std::string Name() const = 0;

    // 根据配置、权重与构建选项构建 TRT network。
    // options 描述的是"这次要建哪个切面"，属于构建期参数，因此不放进 ModelConfig。
    virtual bool Build(nvinfer1::INetworkDefinition* network,
                       const WeightLoader& weights,
                       const ModelConfig& config,
                       const BuildOptions& options) = 0;
};

}  // namespace mini_trt_llm
