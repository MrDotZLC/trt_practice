#pragma once

#include "mini_trt_llm/core/imodel_builder.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {

// 模型构建器注册表。
// 新模型通过 Register 接入，无需修改 EngineBuilder。
class ModelRegistry {
 public:
    ModelRegistry();
    ~ModelRegistry();

    void Register(const std::string& name,
                  std::shared_ptr<IModelBuilder> builder);

    std::shared_ptr<IModelBuilder> Get(const std::string& name) const;

    bool Has(const std::string& name) const;

    std::vector<std::string> List() const;

 private:
    std::map<std::string, std::shared_ptr<IModelBuilder>> builders_;
};

}  // namespace mini_trt_llm
