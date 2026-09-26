#pragma once

// BPE Tokenizer 用例的共享夹具：定位 tokenizer 目录与参考数据。
//
// 为什么要单独一个头：两个用例文件都要用（tokenizer 自己的用例、以及
// Gpt2GenerateTest 里"文本 → prompt token"的桥接用例）。路径规则写两份必然漂移——
// 这个项目已经吃过"两份参考各写一遍"的亏（PROGRESS.md §2.13）。

#include <cstdlib>
#include <filesystem>
#include <string>

namespace mini_trt_llm {
namespace test_support {

// 本机 HF 缓存里的 gpt2 快照（2026-09-26 实测存在）。换机器请设
// MINI_TRT_GPT2_TOKENIZER_DIR，或把 vocab.json + merges.txt 放进 models/gpt2/。
constexpr const char* kDefaultGpt2Snapshot =
    "/home/mr_zlc/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e";

inline bool HasTokenizerFiles(const std::string& dir) {
    if (dir.empty()) return false;
    return std::filesystem::exists(dir + "/vocab.json") &&
           std::filesystem::exists(dir + "/merges.txt");
}

// 返回**仓库根目录**（名字说实话：返回目录就是目录——TROUBLESHOOTING #25 的教训）。
// 首选 __FILE__ 反推：测试源码路径在编译期固定，比"猜相对层数"稳。
inline std::string FindRepoRootDir() {
    const std::filesystem::path source(__FILE__);
    if (source.is_absolute()) {
        // __FILE__ = <repo>/mini_trt_llm/tests/tokenizer_test_support.hpp
        const std::filesystem::path root = source.parent_path().parent_path().parent_path();
        if (std::filesystem::exists(root / "mini_trt_llm")) return root.string();
    }
    for (const char* prefix : {"", "../", "../../", "../../../"}) {
        const std::filesystem::path root = std::filesystem::path(prefix) / "mini_trt_llm";
        if (std::filesystem::exists(root)) {
            return std::filesystem::absolute(root).parent_path().string();
        }
    }
    return {};
}

// 返回**参考数据文件路径**（不是目录）。
inline std::string FindTokenizerGoldenPath() {
    const std::string root = FindRepoRootDir();
    if (root.empty()) return {};
    const std::filesystem::path golden =
        std::filesystem::path(root) / "mini_trt_llm/tests/data/gpt2_tokenizer_golden.json";
    if (std::filesystem::exists(golden)) return golden.string();
    return {};
}

// 返回**tokenizer 目录**（不是文件）。优先级：环境变量 → 仓库内 models/gpt2 → HF 缓存。
inline std::string FindTokenizerDir() {
    if (const char* from_env = std::getenv("MINI_TRT_GPT2_TOKENIZER_DIR")) {
        if (HasTokenizerFiles(from_env)) return from_env;
    }
    const std::string root = FindRepoRootDir();
    if (!root.empty()) {
        const std::string in_repo = root + "/models/gpt2";
        if (HasTokenizerFiles(in_repo)) return in_repo;
    }
    if (HasTokenizerFiles(kDefaultGpt2Snapshot)) return kDefaultGpt2Snapshot;
    return {};
}

// 跳过时必须**打印探测结果**（PROGRESS.md §2.13：静默跳过比失败更贵）。
inline std::string DescribeTokenizerProbe() {
    const std::string dir = FindTokenizerDir();
    const std::string golden = FindTokenizerGoldenPath();
    return "探测结果：tokenizer_dir=" + (dir.empty() ? std::string("<缺失>") : dir) +
           "，golden=" + (golden.empty() ? std::string("<缺失>") : golden) +
           "（可用 MINI_TRT_GPT2_TOKENIZER_DIR 指定 tokenizer 目录）";
}

}  // namespace test_support
}  // namespace mini_trt_llm
