// 引擎缓存指纹的 host 用例（`core/engine_cache.hpp`）。
//
// 为什么要有它：指纹的用途是"决定能不能复用一份 .engine"。它的失效方向有两种，代价完全不同：
//   · **该变而不变** → 复用旧引擎，测出假结果（这正是 TROUBLESHOOTING + TS-034 被牵连的那个坑）；
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

// 按**原始字节**写（不做文本模式的换行翻译）：用来构造"Windows 上写出的 sidecar"那种 CRLF 文件，
// 否则在 Windows 上跑这组用例时 `\r\n` 会被再翻译一次（`\r\r\n`），把用例变成平台相关的假红。
void TouchRaw(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
}

// 带图属性项的输入：`model.n_positions` 是 A1 写进 sidecar 的那一项（2026-10-05）。
EngineFingerprintInputs InputsWithPositions() {
    EngineFingerprintInputs inputs = BaseInputs();
    inputs.numeric_params = {{"prefill.max_seq", 32}, {"model.n_positions", 16}};
    return inputs;
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

// 同一个文件的不同**写法**必须给同一个身份。
// 为什么单列一条：模型路径由调用方的 `FindFile({"models/…", "../models/…", …})` 按**当前工作目录**
// 解析，于是"同一个模型"会得到 `models/…` 或 `../../../models/…` 两种字符串。指纹里存的是路径，
// 所以两种写法会算出两个指纹、把对方的引擎判成过期——2026-09-27 真机上实测过这种交替重建
// （见 `docs/TROUBLESHOOTING.md` + `TS-048`）。
TEST(EngineCacheTest, SourceFileIdentityIgnoresPathSpelling) {
    const std::string probe = MakeTempEnginePath("path_spelling") + ".bin";
    Touch(probe, "same-bytes");
    const std::filesystem::path p(probe);

    EngineFingerprintInputs plain = BaseInputs();
    plain.source_files = {probe};
    const std::string baseline = ComputeEngineFingerprint(plain);

    // `…/tmp/../tmp/x.bin`：与真机上 `../../../models/…` 同类的"绕一圈还是同一个文件"。
    EngineFingerprintInputs via_dotdot = BaseInputs();
    via_dotdot.source_files = {(p.parent_path() / ".." / p.parent_path().filename() / p.filename()).string()};
    EXPECT_EQ(baseline, ComputeEngineFingerprint(via_dotdot));

    // `…/./x.bin`：多余的当前目录分量。
    EngineFingerprintInputs via_dot = BaseInputs();
    via_dot.source_files = {(p.parent_path() / "." / p.filename()).string()};
    EXPECT_EQ(baseline, ComputeEngineFingerprint(via_dot));

    // 反面：真的换了文件（不同路径、内容也不同）必须仍然不同——规范化不能变成"忽略路径"。
    const std::string other = probe + ".other";
    Touch(other, "other-bytes");
    EngineFingerprintInputs other_inputs = BaseInputs();
    other_inputs.source_files = {other};
    EXPECT_NE(baseline, ComputeEngineFingerprint(other_inputs));

    std::filesystem::remove(probe);
    std::filesystem::remove(other);
}

// ---------------------------------------------------------------------------
// A1 的侧车读回：`ReadEngineSidecarField`（2026-10-05，B1+A1）
// ---------------------------------------------------------------------------
// 这组与上面几条一样是**纯 host**（不需要 GPU / TRT 运行时），所以它比
// `LlmRunnerChunkedTest` 的三条 B1+A1 用例**更早能验**：后者要真机窗口，这组有编译器就能跑。
//
// 覆盖的坑都属于"读错了不报错，只是读不到"这一类 —— 而"读不到"的后果由 `LLMRunner` 兜住：
// **拒绝启动**（不猜默认值）。所以这里的每条都同时锁住"读得到"和"读不到时确实读不到"。

TEST(EngineCacheTest, SidecarFieldReadsNumericParamByFullLineKey) {
    const std::string engine = MakeTempEnginePath("sidecar_key");
    Touch(engine, "engine-bytes");
    const EngineFingerprintInputs inputs = InputsWithPositions();
    ASSERT_TRUE(WriteEngineFingerprint(engine, ComputeEngineFingerprint(inputs), inputs));

    // 规范化文本里的字面量是 `num.<名字>=<值>` —— key 必须是**完整行键**。
    EXPECT_EQ(ReadEngineSidecarField(engine, "num.model.n_positions"), "16");
    // 反向自证：只写名字（漏掉 `num.`）必须读不到。这条是给"照文档示例抄半个键"上的锁：
    // 漏了前缀会**静默**失效（返回空串 → 拒绝启动），而不是读到错的值。
    EXPECT_EQ(ReadEngineSidecarField(engine, "model.n_positions"), "")
        << "半个键（漏 `num.` 前缀）不该命中 —— 详见 engine_cache.hpp 的注释";

    std::filesystem::remove(engine);
    std::filesystem::remove(EngineFingerprintPath(engine));
}

TEST(EngineCacheTest, SidecarFieldDoesNotMatchByPrefix) {
    const std::string engine = MakeTempEnginePath("sidecar_prefix");
    Touch(engine, "engine-bytes");
    EngineFingerprintInputs inputs = BaseInputs();
    // 故意造一对前缀关系（`prefill.max_seq` vs `prefill.max_seq_extra`）：
    // 模糊匹配在这里会读出另一个键的值，且**不会报错**。
    inputs.numeric_params = {{"prefill.max_seq", 512}, {"prefill.max_seq_extra", 1024}};
    ASSERT_TRUE(WriteEngineFingerprint(engine, ComputeEngineFingerprint(inputs), inputs));

    EXPECT_EQ(ReadEngineSidecarField(engine, "num.prefill.max_seq"), "512")
        << "读到 1024 说明解析做成了前缀匹配（撞上了 num.prefill.max_seq_extra）";
    EXPECT_EQ(ReadEngineSidecarField(engine, "num.prefill.max_seq_extra"), "1024");

    std::filesystem::remove(engine);
    std::filesystem::remove(EngineFingerprintPath(engine));
}

TEST(EngineCacheTest, SidecarFieldMissingKeyOrFileIsEmpty) {
    const std::string engine = MakeTempEnginePath("sidecar_missing");
    Touch(engine, "engine-bytes");
    const EngineFingerprintInputs inputs = InputsWithPositions();
    ASSERT_TRUE(WriteEngineFingerprint(engine, ComputeEngineFingerprint(inputs), inputs));

    // 缺该键 / 空 key / 侧车不存在：一律空串（= 不可信 → 调用方拒绝启动），不猜默认值。
    EXPECT_EQ(ReadEngineSidecarField(engine, "num.decode.max_batch"), "");
    EXPECT_EQ(ReadEngineSidecarField(engine, ""), "");
    EXPECT_EQ(ReadEngineSidecarField(MakeTempEnginePath("sidecar_absent"),
                                     "num.model.n_positions"), "");

    std::filesystem::remove(engine);
    std::filesystem::remove(EngineFingerprintPath(engine));
}

TEST(EngineCacheTest, SidecarFieldReadsBodyOnly) {
    const std::string engine = MakeTempEnginePath("sidecar_body");
    Touch(engine, "engine-bytes");
    const EngineFingerprintInputs inputs = InputsWithPositions();
    ASSERT_TRUE(WriteEngineFingerprint(engine, ComputeEngineFingerprint(inputs), inputs));

    // 第一行是机器读的 `fingerprint=<hash>`：它**不属于**正文项，用同一个 key 必须读不到
    // （否则"读正文"的起点就错了，任何键都可能命中第一行）。
    EXPECT_EQ(ReadEngineSidecarField(engine, "fingerprint"), "");

    // 只有第一行、没有 `---`：没有正文 → 空串。
    Touch(engine, "fingerprint=0123456789abcdef\n");
    EXPECT_EQ(ReadEngineSidecarField(engine, "num.model.n_positions"), "");

    std::filesystem::remove(engine);
    std::filesystem::remove(EngineFingerprintPath(engine));
}

TEST(EngineCacheTest, SidecarFieldToleratesCrlfLineEndings) {
    const std::string engine = MakeTempEnginePath("sidecar_crlf");
    Touch(engine, "engine-bytes");
    // 文本模式下写出的 sidecar 在 Windows 上是 CRLF：**正文行与 `---` 分隔行**的行尾 `\r` 都必须
    // 先剥掉再比较，否则键比较失配、连"正文起点"都认不出来，表现为"文件里明明有这一项却读不到"
    // （跨平台排查时很容易被当成"文件没写成功"）。所以这里整份文件都用 CRLF，`---` 也用。
    TouchRaw(EngineFingerprintPath(engine),
             "fingerprint=0123456789abcdef\r\n---\r\nstage=single\r\n"
             "num.model.n_positions=16\r\n");
    EXPECT_EQ(ReadEngineSidecarField(engine, "num.model.n_positions"), "16");

    std::filesystem::remove(engine);
    std::filesystem::remove(EngineFingerprintPath(engine));
}

}  // namespace mini_trt_llm
