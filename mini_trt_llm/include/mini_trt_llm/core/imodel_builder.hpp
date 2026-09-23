#pragma once

#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include <NvInfer.h>
#include <memory>
#include <string>

namespace mini_trt_llm {

// 模型构建器抽象接口。
// 每种模型（GPT2, ResNet18 等）实现一个具体子类并注册到 ModelRegistry。
class IModelBuilder {
 public:
    virtual ~IModelBuilder() = default;

    // 模型类型名，如 "gpt2", "resnet18"
    virtual std::string Name() const = 0;

    // 根据配置与权重构建 TRT network
    virtual bool Build(nvinfer1::INetworkDefinition* network,
                       const WeightLoader& weights,
                       const ModelConfig& config) = 0;
};

}  // namespace mini_trt_llm
