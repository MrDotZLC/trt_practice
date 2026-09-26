// GPT-2 BPE Tokenizer 的 host 用例（future_iterations A1）。
//
// 判据：与 HF `transformers.GPT2TokenizerFast` **逐 token 全等**，参考数据在
// tests/data/gpt2_tokenizer_golden.json（由 tools/make_tokenizer_golden.py 产出）。
//
// 环境语义：缺 tokenizer 文件或参考文件 → **显式跳过**（打印探测结果）。
// "缺环境"不等于"实现有问题"，但"资产在、加载失败"必须判红——两者的区别正是
// docs/TROUBLESHOOTING.md #19 留下的教训。

#include "mini_trt_llm/tokenizer/bpe_tokenizer.hpp"

#include "tokenizer_test_support.hpp"

#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/json.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::DescribeTokenizerProbe;
using test_support::FindTokenizerDir;
using test_support::FindTokenizerGoldenPath;

std::string JoinIds(const std::vector<JsonValue>& ids) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out += ", ";
        out += std::to_string(ids[i].AsInt());
    }
    return out + "]";
}

std::string JoinIds(const std::vector<int64_t>& ids) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out += ", ";
        out += std::to_string(ids[i]);
    }
    return out + "]";
}

std::string JoinStrings(const std::vector<std::string>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ", ";
        out += values[i];
    }
    return out + "]";
}

// 每个用例都走这个入口：资产缺失 → 跳过（并打印探测）；资产在但加载失败 → 判红。
struct Fixture {
    std::string tokenizer_dir;
    std::string golden_path;
    JsonValue golden;
    bool ready = false;
};

Fixture PrepareFixture() {
    Fixture fixture;
    fixture.tokenizer_dir = FindTokenizerDir();
    fixture.golden_path = FindTokenizerGoldenPath();
    if (fixture.tokenizer_dir.empty() || fixture.golden_path.empty()) return fixture;
    fixture.golden = LoadJson(fixture.golden_path);
    fixture.ready = true;
    return fixture;
}

// 逐样本比对 Encode。kind 过滤让"哪条用例盯哪类样本"写在测试名里，而不是靠注释。
void ExpectEncodeMatchesGolden(BpeTokenizer* tokenizer, const JsonValue& golden, const std::string& kind) {
    int checked = 0;
    for (size_t i = 0; i < golden["samples"].Size(); ++i) {
        const JsonValue& sample = golden["samples"][i];
        if (sample["kind"].AsString() != kind) continue;

        const std::string text = sample["text"].AsString();
        std::vector<JsonValue> expected_ids;
        for (size_t k = 0; k < sample["ids"].Size(); ++k) expected_ids.push_back(sample["ids"][k]);
        std::vector<std::string> expected_pieces;
        for (size_t k = 0; k < sample["pieces"].Size(); ++k) expected_pieces.push_back(sample["pieces"][k].AsString());

        const std::vector<int64_t> actual_ids = tokenizer->Encode(text);
        const std::vector<std::string> actual_pieces = tokenizer->ByteEncodedPiecesForTesting(text);
        // 预切分与 merge 分开比：不一致时这张诊断表直接指出该往哪边查。
        EXPECT_EQ(actual_pieces, expected_pieces)
            << "样本[" << i << "] kind=" << kind << " 的**预切分/byte 映射**与 HF 不一致\n"
            << "  text     : " << text << "\n"
            << "  HF pieces: " << JoinStrings(expected_pieces) << "\n"
            << "  我们的   : " << JoinStrings(actual_pieces);
        ASSERT_EQ(actual_ids.size(), expected_ids.size())
            << "样本[" << i << "] 的 token 数不一致：HF " << JoinIds(expected_ids) << " vs 我们 " << JoinIds(actual_ids);
        for (size_t k = 0; k < actual_ids.size(); ++k) {
            EXPECT_EQ(actual_ids[k], expected_ids[k].AsInt())
                << "样本[" << i << "] 第 " << k << " 个 token 不一致\n"
                << "  text     : " << text << "\n"
                << "  HF ids   : " << JoinIds(expected_ids) << "\n"
                << "  我们的    : " << JoinIds(actual_ids);
        }
        ++checked;
    }
    EXPECT_GT(checked, 0) << "参考数据里没有 kind=" << kind << " 的样本（参考文件被改过？）";
}

}  // namespace

TEST(BpeTokenizerTest, LoadsFromDirectory) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();

    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(fixture.tokenizer_dir)) << "资产存在，但 Load 失败：" << fixture.tokenizer_dir;
    EXPECT_EQ(tokenizer.VocabSize(), static_cast<size_t>(fixture.golden["vocab_size"].AsInt()));
    std::cout << "[BPE] 已加载 " << fixture.tokenizer_dir << "，vocab_size=" << tokenizer.VocabSize() << "\n";
}

TEST(BpeTokenizerTest, RejectsMissingOrMalformed) {
    BpeTokenizer tokenizer;
    // 目录不存在
    EXPECT_FALSE(tokenizer.Load("/tmp/definitely_missing_bpe_dir"));

    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "mini_trt_llm_bpe_negative_test";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    // 只有 vocab.json、缺 merges.txt
    {
        std::ofstream(tmp / "vocab.json") << "{\"a\": 0}";
        EXPECT_FALSE(tokenizer.Load(tmp.string()));
    }
    // 坏 JSON
    {
        std::ofstream(tmp / "vocab.json") << "{not json";
        std::ofstream(tmp / "merges.txt") << "#version: 0.2\na b\n";
        EXPECT_FALSE(tokenizer.Load(tmp.string()));
    }
    // merges 行没有空格分隔
    {
        std::ofstream(tmp / "vocab.json") << "{\"a\": 0, \"b\": 1, \"ab\": 2}";
        std::ofstream(tmp / "merges.txt") << "#version: 0.2\nab\n";
        EXPECT_FALSE(tokenizer.Load(tmp.string()));
    }
    std::filesystem::remove_all(tmp);
}

TEST(BpeTokenizerTest, EncodeMatchesGoldenBasic) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(fixture.tokenizer_dir));
    ExpectEncodeMatchesGolden(&tokenizer, fixture.golden, "basic");
}

TEST(BpeTokenizerTest, EncodeMatchesGoldenWhitespace) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(fixture.tokenizer_dir));
    // 前导空格 / 连续空格 / 制表符 / 换行：GPT-2 的 `Ġ` 与 `\s+(?!\S)` 语义全在这几条上。
    ExpectEncodeMatchesGolden(&tokenizer, fixture.golden, "whitespace");
}

TEST(BpeTokenizerTest, EncodeMatchesGoldenUtf8) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(fixture.tokenizer_dir));
    // 中文 / emoji / 重音拉丁：走 byte-level 回退路径，也是"非 ASCII 算不算字母"的判据所在。
    ExpectEncodeMatchesGolden(&tokenizer, fixture.golden, "utf8");
}

TEST(BpeTokenizerTest, EncodeMatchesGoldenLongText) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(fixture.tokenizer_dir));
    ExpectEncodeMatchesGolden(&tokenizer, fixture.golden, "long");
}

TEST(BpeTokenizerTest, DecodeMatchesGolden) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(fixture.tokenizer_dir));

    for (size_t i = 0; i < fixture.golden["samples"].Size(); ++i) {
        const JsonValue& sample = fixture.golden["samples"][i];
        std::vector<int64_t> ids;
        for (size_t k = 0; k < sample["ids"].Size(); ++k) ids.push_back(sample["ids"][k].AsInt());
        EXPECT_EQ(tokenizer.Decode(ids), sample["decoded"].AsString())
            << "样本[" << i << "] 的解码与 HF 不一致（text=" << sample["text"].AsString() << "）";
        // 顺带确认 HF 自己的解码是可逆的——这条不满足时"解码判据"本身就要重审。
        EXPECT_EQ(sample["decoded"].AsString(), sample["text"].AsString())
            << "参考数据里 decoded != text，样本[" << i << "]";
    }
}

TEST(BpeTokenizerTest, EmptyStringAndEdgeCases) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(fixture.tokenizer_dir));

    EXPECT_TRUE(tokenizer.Encode("").empty()) << "空串必须编码成空序列";
    EXPECT_EQ(tokenizer.Decode({}), "");
    ExpectEncodeMatchesGolden(&tokenizer, fixture.golden, "edge");
}

// 参考数据自证（PROGRESS.md §2.13：参考实现必须唯一、必须自带断言）。
// 这里检查结构与"内容自洽"；**来源可复现性**（重新用 HF 算一遍再比对）由 ctest 项
// tokenizer_golden_check 负责——那一步需要 Python 与本地 tokenizer 文件，在 CI 里可能被跳过。
TEST(BpeTokenizerReferenceTest, GoldenIsSelfConsistent) {
    const Fixture fixture = PrepareFixture();
    if (!fixture.ready) GTEST_SKIP() << DescribeTokenizerProbe();
    const JsonValue& golden = fixture.golden;

    ASSERT_TRUE(golden.Has("reference"));
    const JsonValue& reference = golden["reference"];
    EXPECT_EQ(reference["implementation"].AsString(), "transformers.GPT2TokenizerFast");
    EXPECT_TRUE(reference["local_files_only"].AsBool()) << "参考不许来自联网加载";
    EXPECT_FALSE(reference["vocab_json_sha256"].AsString().empty());
    EXPECT_FALSE(reference["merges_txt_sha256"].AsString().empty());
    EXPECT_FALSE(golden["samples"].AsArray().empty());

    const int vocab_size = golden["vocab_size"].AsInt();
    EXPECT_EQ(vocab_size, 50257) << "GPT-2 的 vocab_size 是固定的；变了说明参考换了模型";

    for (size_t i = 0; i < golden["samples"].Size(); ++i) {
        const JsonValue& sample = golden["samples"][i];
        EXPECT_FALSE(sample["kind"].AsString().empty());
        EXPECT_FALSE(sample["pieces"].AsArray().empty() && !sample["ids"].AsArray().empty())
            << "样本[" << i << "] 有 token 却没有 pieces（或反之）——参考记录不完整";
        for (size_t k = 0; k < sample["ids"].Size(); ++k) {
            const int id = sample["ids"][k].AsInt();
            EXPECT_GE(id, 0);
            EXPECT_LT(id, vocab_size) << "样本[" << i << "] 的 id 超出词表：参考文件被改过？";
        }
    }
}

}  // namespace mini_trt_llm
