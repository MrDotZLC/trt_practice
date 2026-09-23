#include "mini_trt_llm/plugins/plugin_registry.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <NvInferPlugin.h>

namespace mini_trt_llm {

PluginRegistry& PluginRegistry::Instance() {
    static PluginRegistry registry;
    return registry;
}

void PluginRegistry::RegisterCreator(nvinfer1::IPluginCreatorV3One* creator) {
    if (!creator) return;
    std::string name = creator->getPluginName();
    creators_[name] = creator;
    MINI_TRT_LOG_INFO("Registered plugin creator: " << name);
}

nvinfer1::IPluginCreatorV3One* PluginRegistry::GetCreator(
    const std::string& name) const {
    auto it = creators_.find(name);
    if (it == creators_.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<std::string> PluginRegistry::ListCreators() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : creators_) {
        names.push_back(name);
    }
    return names;
}

void PluginRegistry::RegisterAllPlugins() {
    // Phase 1 注册 RoPE / RMSNorm / PagedAttention / Sampler 等 creator
    MINI_TRT_LOG_INFO("PluginRegistry::RegisterAllPlugins stub in Phase 0");
}

}  // namespace mini_trt_llm
