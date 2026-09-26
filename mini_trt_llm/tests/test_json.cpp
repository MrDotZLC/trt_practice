// `utils/json.hpp` 的解析用例。
//
// 为什么单独补这一份：为了读 GPT-2 的 `vocab.json`（5 万个 `"\u0120xxx"` 形式的 key，
// 见 future_iterations §5.1），给这个公共解析器加了 `\uXXXX` 支持。**改公共件就要有覆盖**——
// 否则下一个人只看得到"解析偶尔抛异常"，看不到"哪些转义是承诺支持的"。

#include "mini_trt_llm/utils/json.hpp"

#include <gtest/gtest.h>

#include <string>

namespace mini_trt_llm {
namespace {

JsonValue ParseEscaped(const std::string& literal) {
    JsonParser parser;
    return parser.Parse("{\"k\": \"" + literal + "\"}");
}

}  // namespace

TEST(JsonParserTest, ParsesUnicodeEscape) {
    // GPT-2 词表的 key 就是这个形态：\u0120 = 'Ġ'（空格被映射后的可见字符）。
    EXPECT_EQ(ParseEscaped("\\u0120the")["k"].AsString(), "\xC4\xA0the");
}

TEST(JsonParserTest, ParsesSurrogatePair) {
    // \uD83D\uDE42 是一对代理，必须还原成单个码点 U+1F642（🙂），而不是两个替换字符。
    EXPECT_EQ(ParseEscaped("\\uD83D\\uDE42")["k"].AsString(), "\xF0\x9F\x99\x82");
}

TEST(JsonParserTest, KeepsAsciiEscapesWorking) {
    EXPECT_EQ(ParseEscaped("a\\nb\\t\\\"c\\\\d\\/e")["k"].AsString(), "a\nb\t\"c\\d/e");
}

TEST(JsonParserTest, ReadsSurrogatePairAsOneCodePoint) {
    // 反证"拼成两个码点也算过"：字符串长度必须是 4 字节（UTF-8 的 U+1F642），不是 6。
    EXPECT_EQ(ParseEscaped("\\uD83D\\uDE42")["k"].AsString().size(), 4u);
}

TEST(JsonParserTest, RejectsLoneSurrogate) {
    EXPECT_THROW(ParseEscaped("\\uD83D"), std::runtime_error);
    EXPECT_THROW(ParseEscaped("\\uDE42"), std::runtime_error);
}

TEST(JsonParserTest, RejectsInvalidUnicodeEscape) {
    EXPECT_THROW(ParseEscaped("\\uZZZZ"), std::runtime_error);
    EXPECT_THROW(ParseEscaped("\\u12"), std::runtime_error);
}

}  // namespace mini_trt_llm
