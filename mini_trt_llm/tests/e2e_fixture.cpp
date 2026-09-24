#include "e2e_fixture.hpp"

#include "mini_trt_llm/utils/io.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <utility>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

ModelDirectory ModelDirectory::Create(const std::string& tag) {
    ModelDirectory directory;
    std::string pattern = "/tmp/mini_trt_llm_" + tag + "_XXXXXX";
    // mkdtemp 会就地替换 XXXXXX，因此必须传入可写的 char 缓冲区
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (::mkdtemp(buffer.data()) == nullptr) {
        return directory;
    }
    directory.path_ = buffer.data();
    return directory;
}

ModelDirectory::~ModelDirectory() {
    if (!path_.empty()) {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
}

ModelDirectory::ModelDirectory(ModelDirectory&& other) noexcept
    : path_(std::move(other.path_)) {
    other.path_.clear();
}

ModelDirectory& ModelDirectory::operator=(ModelDirectory&& other) noexcept {
    if (this != &other) {
        if (!path_.empty()) {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }
        path_ = std::move(other.path_);
        other.path_.clear();
    }
    return *this;
}

bool ModelDirectory::WriteConfig(const std::string& json) const {
    if (!valid()) {
        return false;
    }
    WriteFile(path_ + "/config.json", json.data(), json.size());
    return true;
}

bool ModelDirectory::WriteWeights(
    const std::map<std::string, TensorSpec>& tensors) const {
    if (!valid()) {
        return false;
    }
    return WriteSafetensorsFile(path_ + "/model.safetensors", tensors);
}

bool ModelDirectory::RemoveWeights() const {
    if (!valid()) {
        return false;
    }
    std::error_code error;
    return std::filesystem::remove(path_ + "/model.safetensors", error);
}

std::string ModelDirectory::EnginePath(const std::string& name) const {
    return path_ + "/" + name;
}

}  // namespace test_support
}  // namespace mini_trt_llm
