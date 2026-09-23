#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mini_trt_llm {

std::vector<char> ReadFile(const std::string& path) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    size_t size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!fin.read(buffer.data(), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return buffer;
}

void WriteFile(const std::string& path, const void* data, size_t bytes) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Failed to create file: " + path);
    }
    fout.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(bytes));
    if (!fout) {
        throw std::runtime_error("Failed to write file: " + path);
    }
}

JsonValue LoadJson(const std::string& path) {
    auto buffer = ReadFile(path);
    std::string text(buffer.begin(), buffer.end());
    JsonParser parser;
    return parser.Parse(text);
}

}  // namespace mini_trt_llm
