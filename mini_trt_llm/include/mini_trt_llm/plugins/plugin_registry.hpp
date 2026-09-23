#pragma once

#include <NvInfer.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {

// Plugin 注册表。
// 统一管理自定义 Plugin 的 creator，支持 ONNX + Plugin 方案中的命名查找。
class PluginRegistry {
 public:
    static PluginRegistry& Instance();

    void RegisterCreator(nvinfer1::IPluginCreatorV3One* creator);

    nvinfer1::IPluginCreatorV3One* GetCreator(const std::string& name) const;

    std::vector<std::string> ListCreators() const;

    // 初始化并注册所有 mini_trt_llm 内置 plugin
    void RegisterAllPlugins();

 private:
    PluginRegistry() = default;
    std::map<std::string, nvinfer1::IPluginCreatorV3One*> creators_;
};

}  // namespace mini_trt_llm
