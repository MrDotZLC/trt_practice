#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <string>
#include <vector>

namespace mini_trt_llm {

// IPluginV3 插件基类封装。
// Phase 0 仅提供基础接口声明，Phase 1 由各具体 Plugin 实现。
//
// TensorRT 10.x IPluginV3 接口包括：
//   - getMetadata()
//   - supportsFormatCombination()
//   - getOutputDataTypes()
//   - getOutputShapes()
//   - configurePlugin()
//   - getWorkspaceSize()
//   - enqueue()
//   - getFieldsToSerialize()
class IPluginV3Base : public nvinfer1::IPluginV3 {
 public:
    virtual ~IPluginV3Base() = default;

    // 子类必须实现：插件类型名
    virtual const char* getPluginType() const noexcept = 0;

    // 子类必须实现：插件版本
    virtual const char* getPluginVersion() const noexcept = 0;

 protected:
    std::string plugin_namespace_;
};

}  // namespace mini_trt_llm
