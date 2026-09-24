#include "mini_trt_llm/plugins/plugin_registry.hpp"
#include "mini_trt_llm/plugins/paged_attention_plugin.hpp"
#include "mini_trt_llm/plugins/rmsnorm_plugin.hpp"
#include "mini_trt_llm/plugins/rope_plugin.hpp"
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
    // 只登记指针，creator 的生命周期由各自的静态实例持有，注册表不负责释放。
    RegisterCreator(&GetRmsNormPluginCreator());
    RegisterCreator(&GetRoPEPluginCreator());
    RegisterCreator(&GetPagedAttentionPluginCreator());
}

}  // namespace mini_trt_llm
