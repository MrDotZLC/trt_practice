#pragma once

#include "mini_trt_llm/utils/json.hpp"
#include <string>
#include <vector>

namespace mini_trt_llm {

// 读取二进制文件到 vector<char>
std::vector<char> ReadFile(const std::string& path);

// 写二进制文件
void WriteFile(const std::string& path, const void* data, size_t bytes);

// 加载 JSON 文件
JsonValue LoadJson(const std::string& path);

}  // namespace mini_trt_llm
