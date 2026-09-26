// 引擎缓存指纹的 host 用例（`core/engine_cache.hpp`）。
//
// 为什么要有它：指纹的用途是"决定能不能复用一份 .engine"。它的失效方向有两种，代价完全不同：
//   · **该变而不变** → 复用旧引擎，测出假结果（这正是 TROUBLESHOOTING #34 被牵连的那个坑）；
//   · 不该变而变 → 多花一次构建时间，属于可接受的浪费。
// 所以这里逐项验证"改任何一个字段都会改变指纹"，并验证"缺指纹 = 不可信"。
// 全部是纯 host 逻辑，不需要 GPU 与 TRT 运行时。

#include "mini_trt_llm/core/engine_cache.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace mini_trt_llm {
namespace {

EngineFingerprintInputs BaseInputs() {
    EngineFingerprintInputs inputs;
    inputs.stage = "single";
    inputs.precision = "fp32";
    inputs.source_kind = "config";
    inputs.source_files = {"models/gpt2/config.json"};
    inputs.numeric_params = {{"prefill.max_seq", 512}, {"decode.max_batch", 4}};
    inputs.flags = {{"export_diagnostics", false}, {"detailed_profiling", false}};
    inputs.trt_version = "101501";
    inputs.cuda_runtime_version = 12060;
    inputs.graph_version = 1;
    return inputs;
}

std::string MakeTempEnginePath(const std::string& name) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("mini_trt_llm_cache_test_" + name + ".engine");
    std::filesystem::remove(path);
    std::filesystem::remove(EngineFingerprintPath(path.string()));
    return path.string();
}

void Touch(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::trunc);
    out << content;
}

}  // namespace

TEST(EngineCacheTest, FingerprintIsDeterministicAndOrderInsensitive) {
    const EngineFingerprintInputs a = BaseInputs();
    EngineFingerprintInputs b = BaseInputs();
    std::reverse(b.numeric_params.begin(), b.numeric_params.end());
    std::reverse(b.flags.begin(), b.flags.end());
    // 参数给入顺序不该影响指纹：否则同一个配置会算出两个键，缓存命中率会莫名掉下去。
    EXPECT_EQ(ComputeEngineFingerprint(a), ComputeEngineFingerprint(b));
    EXPECT_EQ(ComputeEngineFingerprint(a), ComputeEngineFingerprint(a));
}

TEST(EngineCacheTest, EveryFieldChangeMovesTheFingerprint) {
    const std::string base = ComputeEngineFingerprint(BaseInputs());
    const auto changed = [&base](const EngineFingerprintInputs& mutated) {
        return ComputeEngineFingerprint(mutated) != base;
    };

    EngineFingerprintInputs inputs = BaseInputs();
    inputs.stage = "prefill";
    EXPECT_TRUE(changed(inputs)) << "stage（不同切面 = 不同图）必须影响指纹";

    inputs = BaseInputs();
    inputs.precision = "fp16";
    EXPECT_TRUE(changed(inputs)) << "精度必须影响指纹";

    inputs = BaseInputs();
    inputs.source_kind = "onnx";
    EXPECT_TRUE(changed(inputs)) << "来源（config/onnx）必须影响指纹";

    inputs = BaseInputs();
    inputs.numeric_params[0].second = 1024;
    EXPECT_TRUE(changed(inputs)) << "seq/batch 范围必须影响指纹（改了范围却复用旧引擎是个真坑）";

    inputs = BaseInputs();
    inputs.flags[1].second = true;
    EXPECT_TRUE(changed(inputs)) << "建图开关必须影响指纹";

    inputs = BaseInputs();
    inputs.trt_version = "101502";
    EXPECT_TRUE(changed(inputs)) << "TRT 版本必须影响指纹";

    inputs = BaseInputs();
    inputs.cuda_runtime_version = 12070;
    EXPECT_TRUE(changed(inputs)) << "CUDA 运行时版本必须影响指纹";

    inputs = BaseInputs();
    inputs.graph_version = 2;
    EXPECT_TRUE(changed(inputs)) << "手工图版本必须影响指纹（它是'图代码变了'的唯一人工信号）";
}

TEST(EngineCacheTest, MissingFingerprintIsNeverTrusted) {
    const std::string engine = MakeTempEnginePath("missing_fp");
    Touch(engine, "engine-bytes");
    const std::string fingerprint = ComputeEngineFingerprint(BaseInputs());

    // 只有引擎、没有 sidecar（本功能之前建的旧引擎就是这种形态）→ **不可复用**。
    EXPECT_FALSE(EngineCacheIsFresh(engine, fingerprint))
        << "缺指纹必须视为不可信，否则旧引擎会被静默复用";

    ASSERT_TRUE(WriteEngineFingerprint(engine, fingerprint, BaseInputs()));
    EXPECT_TRUE(EngineCacheIsFresh(engine, fingerprint));

    // 指纹不同（配置或代码变了）→ 不可复用。
    EXPECT_FALSE(EngineCacheIsFresh(engine, "0123456789abcdef"));

    std::filesystem::remove(engine);
    std::filesystem::remove(EngineFingerprintPath(engine));
}

TEST(EngineCacheTest, FingerprintRoundTripKeepsCanonicalText) {
    const std::string engine = MakeTempEnginePath("round_trip");
    Touch(engine, "engine-bytes");
    const EngineFingerprintInputs inputs = BaseInputs();
    const std::string fingerprint = ComputeEngineFingerprint(inputs);
    ASSERT_TRUE(WriteEngineFingerprint(engine, fingerprint, inputs));

    EXPECT_EQ(ReadEngineFingerprint(engine), fingerprint);
    // sidecar 里带规范化文本：排错时要能直接看出"到底哪一项变了"。
    std::ifstream in(EngineFingerprintPath(engine));
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("num.prefill.max_seq=512"), std::string::npos);
    EXPECT_NE(content.find("flag.detailed_profiling=0"), std::string::npos);

    std::filesystem::remove(engine);
    std::filesystem::remove(EngineFingerprintPath(engine));
}

TEST(EngineCacheTest, SourceFileIdentityChangesWithMtime) {
    const std::string probe = MakeTempEnginePath("source_file") + ".bin";
    Touch(probe, "v1");
    EngineFingerprintInputs inputs = BaseInputs();
    inputs.source_files = {probe};
    const std::string first = ComputeEngineFingerprint(inputs);

    // 改内容 → size 变 → 指纹必须变（模型换了却复用旧引擎是最坏的一种失效）。
    Touch(probe, "v2-longer");
    EXPECT_NE(first, ComputeEngineFingerprint(inputs));
    std::filesystem::remove(probe);

    // 文件不存在也进指纹：不能把"缺文件"与"空文件"混为一谈。
    EngineFingerprintInputs missing = BaseInputs();
    missing.source_files = {probe};
    EXPECT_NE(first, ComputeEngineFingerprint(missing));
}

}  // namespace mini_trt_llm
