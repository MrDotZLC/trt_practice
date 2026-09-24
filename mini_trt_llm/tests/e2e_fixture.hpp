#pragma once

#include "e2e_safetensors_writer.hpp"

#include <map>
#include <string>

namespace mini_trt_llm {
namespace test_support {

// 一个临时的模型目录：{dir}/config.json + {dir}/model.safetensors。
//
// 用 mkdtemp 建目录、析构时递归删除，保证端到端测试不留残留文件，
// 也不依赖固定路径（ctest 可能并行执行多个用例）。
class ModelDirectory {
 public:
    ModelDirectory() = default;
    ~ModelDirectory();

    ModelDirectory(const ModelDirectory&) = delete;
    ModelDirectory& operator=(const ModelDirectory&) = delete;
    ModelDirectory(ModelDirectory&& other) noexcept;
    ModelDirectory& operator=(ModelDirectory&& other) noexcept;

    // 创建临时目录；失败时返回一个 path() 为空的实例。
    static ModelDirectory Create(const std::string& tag);

    bool valid() const { return !path_.empty(); }
    const std::string& path() const { return path_; }

    // 写 config.json
    bool WriteConfig(const std::string& json) const;
    // 写 model.safetensors
    bool WriteWeights(const std::map<std::string, TensorSpec>& tensors) const;
    // 删除 model.safetensors，用于覆盖"权重文件缺失"这类场景
    bool RemoveWeights() const;

    // engine 输出路径也放在同一个临时目录下，便于断言"失败时不应产出 engine"
    std::string EnginePath(const std::string& name = "model.engine") const;

 private:
    std::string path_;
};

}  // namespace test_support
}  // namespace mini_trt_llm
