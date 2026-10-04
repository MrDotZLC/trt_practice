#include "mini_trt_llm/core/engine_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace mini_trt_llm {
namespace {

// 文件身份：size + mtime（纳秒）。不用哈希内容——缓存键只需"变更敏感"，
// 而给 GB 级 safetensors/onnx 做哈希会让每次构建都多花几秒到几十秒。
std::string FileIdentity(const std::string& path) {
    std::ostringstream oss;
    // 路径**先规范化**再进身份：同一个文件用不同写法（`models/x` / `../../../models/x` /
    // 带 `./` 的绝对路径）必须给出同一个身份。否则指纹会随"从哪个工作目录调用"变化——
    // 2026-09-27 实测：`ctest`（CWD = 构建目录，`FindFile` 返回 `../../../models/…`）与
    // 手动从仓库根跑（返回 `models/…`）会各把对方的引擎判成过期、交替重建一次
    // （见 `docs/TROUBLESHOOTING.md` + `TS-048`）。
    // `weakly_canonical` 对**不存在**的路径也能工作（只要求前缀存在）；万一失败就退回原字符串——
    // 宁可"该变而变"（多花一次构建），也不要漏掉真的要失效的情况。
    std::error_code canonical_error;
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(path, canonical_error);
    oss << (canonical_error ? path : canonical.string()) << '|';
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        // 文件不存在也是身份的一部分：**不能**把"缺失"和"空文件"混为一谈。
        return oss.str() + "missing";
    }
    oss << size << '|';
    const auto mtime = std::filesystem::last_write_time(path, error);
    if (error) {
        return oss.str() + "no-mtime";
    }
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch()).count();
    oss << nanos;
    return oss.str();
}

// FNV-1a 64：作为缓存键足够，且实现只有几行（要防篡改请另用 sha256）。
std::string Fnv1a64Hex(const std::string& text) {
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << hash;
    return oss.str();
}

}  // namespace

std::string CanonicalFingerprintText(const EngineFingerprintInputs& inputs) {
    std::ostringstream oss;
    // 排序后再拼：调用方给参数的顺序不该影响指纹（否则同一个配置会算出两个键）。
    oss << "stage=" << inputs.stage << '\n';
    oss << "source_kind=" << inputs.source_kind << '\n';
    oss << "precision=" << inputs.precision << '\n';
    oss << "trt_version=" << inputs.trt_version << '\n';
    oss << "cuda_runtime_version=" << inputs.cuda_runtime_version << '\n';
    oss << "graph_version=" << inputs.graph_version << '\n';

    std::vector<std::string> files;
    files.reserve(inputs.source_files.size());
    for (const std::string& file : inputs.source_files) {
        files.push_back(FileIdentity(file));
    }
    std::sort(files.begin(), files.end());
    for (const std::string& identity : files) {
        oss << "file=" << identity << '\n';
    }

    std::vector<std::pair<std::string, int64_t>> numbers = inputs.numeric_params;
    std::sort(numbers.begin(), numbers.end());
    for (const auto& [name, value] : numbers) {
        oss << "num." << name << '=' << value << '\n';
    }

    std::vector<std::pair<std::string, bool>> flags = inputs.flags;
    std::sort(flags.begin(), flags.end());
    for (const auto& [name, value] : flags) {
        oss << "flag." << name << '=' << (value ? 1 : 0) << '\n';
    }
    return oss.str();
}

std::string ComputeEngineFingerprint(const EngineFingerprintInputs& inputs) {
    return Fnv1a64Hex(CanonicalFingerprintText(inputs));
}

std::string EngineFingerprintPath(const std::string& engine_path) {
    return engine_path + ".fingerprint";
}

bool WriteEngineFingerprint(const std::string& engine_path, const std::string& fingerprint,
                            const EngineFingerprintInputs& inputs) {
    std::ofstream out(EngineFingerprintPath(engine_path), std::ios::trunc);
    if (!out) return false;
    // 两段：第一行机器读；`---` 之后是给人看的规范化文本，排错时能直接看出哪一项变了。
    out << "fingerprint=" << fingerprint << "\n---\n" << CanonicalFingerprintText(inputs);
    return static_cast<bool>(out);
}

std::string ReadEngineFingerprint(const std::string& engine_path) {
    std::ifstream in(EngineFingerprintPath(engine_path));
    if (!in) return {};
    std::string line;
    std::getline(in, line);
    const std::string prefix = "fingerprint=";
    if (line.rfind(prefix, 0) != 0) return {};
    return line.substr(prefix.size());
}

std::string ReadEngineSidecarField(const std::string& engine_path, const std::string& key) {
    if (key.empty()) return {};
    std::ifstream in(EngineFingerprintPath(engine_path));
    if (!in) return {};
    // 与 `ReadEngineFingerprint` 的解析风格一致：逐行比对**完整行键**（`num.` 前缀也是键的一部分，
    // 见 CanonicalFingerprintText），不做前缀模糊匹配 —— 模糊匹配会让 `prefill.max_seq` 命中
    // `num.prefill.max_seq_extra` 这类未来的键。`---` 之前是机器读的那一段，正文只从它之后开始。
    std::string line;
    bool in_body = false;
    while (std::getline(in, line)) {
        // 行尾的 `\r` 要去掉：文本模式写出的 CRLF 会让键比较失配（Linux 上通常看不到，但别依赖）。
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!in_body) {
            if (line == "---") {
                in_body = true;
            }
            continue;
        }
        const std::string prefix = key + '=';
        if (line.rfind(prefix, 0) != 0) {
            continue;
        }
        return line.substr(prefix.size());
    }
    // 文件不存在 / 缺 `---` / 缺该键：一律返回空串（调用方按"不可信"处理，见头文件注释）。
    return {};
}

bool EngineCacheIsFresh(const std::string& engine_path, const std::string& fingerprint) {
    std::error_code error;
    if (!std::filesystem::exists(engine_path, error)) return false;
    const std::string stored = ReadEngineFingerprint(engine_path);
    // 空串 = 没写过指纹（旧引擎）或写坏了 → 一律重建，不默认信任。
    return !stored.empty() && stored == fingerprint;
}

}  // namespace mini_trt_llm
