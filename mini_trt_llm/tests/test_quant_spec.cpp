// REQ-017（路线 C）的清单解析用例。
//
// 为什么这些用例全是 host 侧：清单是**纯文本契约**（Python 产出、C++ 读取），它的每一道
// 拒绝面都不需要 GPU 或 TensorRT —— 而正是这些拒绝面在保护"声明了量化、其实没量化"这件事
// （清单缺项 / 粒度不支持 / scale 不合法 都会让引擎静默退回 FP32）。放在 host 侧才能进沙箱 CI。

#include "e2e_fixture.hpp"
#include "mini_trt_llm/core/quant_spec.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace mini_trt_llm {
namespace {

// 把清单 JSON 写进临时目录并载入。返回 LoadFromFile 的结果。
// 用 `ModelDirectory` 复用既有的"临时目录 + 析构清理"，不再造第二套。
bool LoadManifest(const test_support::ModelDirectory& directory, const std::string& json,
                  QuantSpec* out) {
    const std::string path = directory.path() + "/quant_int8.json";
    std::ofstream handle(path, std::ios::binary);
    if (!handle) {
        return false;
    }
    handle << json;
    handle.close();
    return QuantSpec::LoadFromFile(path, out);
}

// 一份合法清单：两个 per_tensor 条目，路径都写成**相对清单目录**（用来验证解析）。
std::string ValidManifestJson() {
    return R"({
  "format_version": 1,
  "generator": {"tool": "tools/convert/quantize_gpt2.py", "command": "..."},
  "weights_source": {"path": "model.safetensors", "sha256": "aa"},
  "int8_weights": {"path": "model_int8.safetensors", "sha256": "bb"},
  "scheme": "symmetric_per_tensor",
  "zero_point": 0,
  "entries": [
    {"tensor": "wte.weight", "source_key": "transformer.wte.weight",
     "granularity": "per_tensor", "axis": null, "scales": [0.125], "saturate_ratio": 0.0},
    {"tensor": "h.0.attn.c_attn.weight", "source_key": "transformer.h.0.attn.c_attn.weight",
     "granularity": "per_tensor", "axis": null, "scales": [0.25], "saturate_ratio": 0.0}
  ]
})";
}

// 只替换第一处。
//
// **不要用 `std::string::replace(pos, len, ...)` 直接改**：那个 `len` 是"删除多少个字符"，
// 与替换串长度是两回事；传错会造出**非法 JSON**，于是用例"因为解析失败而通过"——看起来绿，
// 实际那条拒绝面根本没被覆盖（本项目最忌讳的假绿）。这里找不到目标串就原样返回，用例会因此
// 变红而不是变绿。
std::string ReplaceOnce(const std::string& text, const std::string& from, const std::string& to) {
    const size_t pos = text.find(from);
    if (pos == std::string::npos) {
        return text;
    }
    return text.substr(0, pos) + to + text.substr(pos + from.size());
}

using QuantSpecTest = ::testing::Test;

TEST(QuantSpecTest, LoadsValidManifestAndResolvesBothNames) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_ok");
    ASSERT_TRUE(directory.valid());

    QuantSpec spec;
    ASSERT_TRUE(LoadManifest(directory, ValidManifestJson(), &spec));
    ASSERT_EQ(spec.entries().size(), 2u);

    // 双向命中：建图侧给 TRT 名，产物侧给文件 key，两个方向都要能找到同一条目。
    const QuantEntry* by_tensor = spec.Find("wte.weight");
    ASSERT_NE(by_tensor, nullptr);
    EXPECT_EQ(by_tensor->source_key, "transformer.wte.weight");
    EXPECT_FLOAT_EQ(by_tensor->scale(), 0.125f);

    const QuantEntry* by_source = spec.Find("transformer.h.0.attn.c_attn.weight");
    ASSERT_NE(by_source, nullptr);
    EXPECT_EQ(by_source->tensor, "h.0.attn.c_attn.weight");
    EXPECT_FLOAT_EQ(by_source->scale(), 0.25f);

    // 清单里没有的名字必须返回 nullptr（调用方据此走原有 FP32/FP16 路径）。
    EXPECT_EQ(spec.Find("h.0.ln_1.weight"), nullptr);
    EXPECT_EQ(spec.Find("transformer.wpe.weight"), nullptr);

    // 相对路径按清单所在目录解析 —— 否则从别的 cwd 跑就找不到 int8 权重。
    EXPECT_EQ(spec.int8_weights_path(), directory.path() + "/model_int8.safetensors");
    EXPECT_EQ(spec.weights_source_path(), directory.path() + "/model.safetensors");
}

TEST(QuantSpecTest, RejectsUnsupportedFormatVersion) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_version");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;
    EXPECT_FALSE(LoadManifest(directory, ReplaceOnce(ValidManifestJson(),
                                                     "\"format_version\": 1",
                                                     "\"format_version\": 2"), &spec));
}

TEST(QuantSpecTest, RejectsNonPerTensorGranularity) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_granularity");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;
    // 注意 `from` 里带上引号与 key：清单里还有 `symmetric_per_tensor`，
    // 只按 `per_tensor` 替换会误伤 scheme（那会让用例测错东西）。
    EXPECT_FALSE(LoadManifest(directory, ReplaceOnce(ValidManifestJson(),
                                                     "\"granularity\": \"per_tensor\"",
                                                     "\"granularity\": \"per_channel\""), &spec));
}

TEST(QuantSpecTest, RejectsNonSymmetricScheme) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_scheme");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;
    EXPECT_FALSE(LoadManifest(directory, ReplaceOnce(ValidManifestJson(),
                                                     "\"symmetric_per_tensor\"",
                                                     "\"asymmetric\""), &spec));
}

TEST(QuantSpecTest, RejectsNonZeroZeroPoint) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_zero_point");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;
    EXPECT_FALSE(LoadManifest(directory, ReplaceOnce(ValidManifestJson(),
                                                     "\"zero_point\": 0",
                                                     "\"zero_point\": -3"), &spec));
}

TEST(QuantSpecTest, RejectsEmptyOrNonArrayEntries) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_entries");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;

    // 空数组
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "weights_source": {"path": "m"}, "int8_weights": {"path": "i"},
        "scheme": "symmetric_per_tensor", "zero_point": 0, "entries": []})", &spec));

    // entries 不是数组
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "weights_source": {"path": "m"}, "int8_weights": {"path": "i"},
        "scheme": "symmetric_per_tensor", "zero_point": 0, "entries": {}})", &spec));
}

TEST(QuantSpecTest, RejectsBadScales) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_scales");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;

    const std::string prefix = R"({"format_version": 1,
        "weights_source": {"path": "m"}, "int8_weights": {"path": "i"},
        "scheme": "symmetric_per_tensor", "zero_point": 0, "entries": [
        {"tensor": "t", "source_key": "s", "granularity": "per_tensor", )";
    const std::string suffix = R"(}]})";

    // 长度必须恰好 1（per_tensor）
    EXPECT_FALSE(LoadManifest(directory, prefix + R"("scales": [0.5, 0.5])" + suffix, &spec));
    EXPECT_FALSE(LoadManifest(directory, prefix + R"("scales": [])" + suffix, &spec));
    // 必须有限且为正：0 会把整层静默清零，负数会让符号翻转
    EXPECT_FALSE(LoadManifest(directory, prefix + R"("scales": [0.0])" + suffix, &spec));
    EXPECT_FALSE(LoadManifest(directory, prefix + R"("scales": [-1.0])" + suffix, &spec));
    // 合法值必须通过（否则上面四条可能因为别的原因失败而"假绿"）
    EXPECT_TRUE(LoadManifest(directory, prefix + R"("scales": [0.5])" + suffix, &spec));
}

TEST(QuantSpecTest, RejectsMissingRequiredFields) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_missing");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;

    // 缺 int8_weights.path
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "weights_source": {"path": "m"}, "scheme": "symmetric_per_tensor",
        "zero_point": 0, "entries": [{"tensor": "t", "source_key": "s",
        "granularity": "per_tensor", "scales": [0.5]}]})", &spec));

    // 缺 weights_source.path
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "int8_weights": {"path": "i"}, "scheme": "symmetric_per_tensor",
        "zero_point": 0, "entries": [{"tensor": "t", "source_key": "s",
        "granularity": "per_tensor", "scales": [0.5]}]})", &spec));

    // 条目缺 tensor
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "weights_source": {"path": "m"}, "int8_weights": {"path": "i"},
        "scheme": "symmetric_per_tensor", "zero_point": 0,
        "entries": [{"source_key": "s", "granularity": "per_tensor", "scales": [0.5]}]})",
        &spec));

    // 条目缺 source_key
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "weights_source": {"path": "m"}, "int8_weights": {"path": "i"},
        "scheme": "symmetric_per_tensor", "zero_point": 0,
        "entries": [{"tensor": "t", "granularity": "per_tensor", "scales": [0.5]}]})", &spec));
}

TEST(QuantSpecTest, RejectsDuplicateEntries) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_dup");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;

    // 同一个 tensor 出现两次
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "weights_source": {"path": "m"}, "int8_weights": {"path": "i"},
        "scheme": "symmetric_per_tensor", "zero_point": 0, "entries": [
        {"tensor": "t", "source_key": "s1", "granularity": "per_tensor", "scales": [0.5]},
        {"tensor": "t", "source_key": "s2", "granularity": "per_tensor", "scales": [0.5]}]})",
        &spec));

    // 同一个 source_key 出现两次
    EXPECT_FALSE(LoadManifest(directory, R"({"format_version": 1,
        "weights_source": {"path": "m"}, "int8_weights": {"path": "i"},
        "scheme": "symmetric_per_tensor", "zero_point": 0, "entries": [
        {"tensor": "t1", "source_key": "s", "granularity": "per_tensor", "scales": [0.5]},
        {"tensor": "t2", "source_key": "s", "granularity": "per_tensor", "scales": [0.5]}]})",
        &spec));
}

TEST(QuantSpecTest, FailsOnMissingManifestFile) {
    const auto directory = test_support::ModelDirectory::Create("quant_spec_absent");
    ASSERT_TRUE(directory.valid());
    QuantSpec spec;
    EXPECT_FALSE(QuantSpec::LoadFromFile(directory.path() + "/no_such_manifest.json", &spec));
}

}  // namespace
}  // namespace mini_trt_llm
