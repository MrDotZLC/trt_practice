#include "mini_trt_llm/core/model_registry.hpp"
#include "mini_trt_llm/utils/logger.hpp"

namespace mini_trt_llm {

ModelRegistry::ModelRegistry() = default;
ModelRegistry::~ModelRegistry() = default;

void ModelRegistry::Register(const std::string& name,
                             std::shared_ptr<IModelBuilder> builder) {
    builders_[name] = builder;
    MINI_TRT_LOG_INFO("Registered model builder: " << name);
}

std::shared_ptr<IModelBuilder> ModelRegistry::Get(const std::string& name) const {
    auto it = builders_.find(name);
    if (it == builders_.end()) {
        return nullptr;
    }
    return it->second;
}

bool ModelRegistry::Has(const std::string& name) const {
    return builders_.find(name) != builders_.end();
}

std::vector<std::string> ModelRegistry::List() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : builders_) {
        names.push_back(name);
    }
    return names;
}

}  // namespace mini_trt_llm
