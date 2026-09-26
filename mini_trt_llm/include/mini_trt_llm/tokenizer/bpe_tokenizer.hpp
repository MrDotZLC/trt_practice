#pragma once

#include "mini_trt_llm/tokenizer/base_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_trt_llm {

// GPT-2 的 byte-level BPE tokenizer。
//
// 为什么不用已有的 SentencePiece：GPT-2 的词表与切分规则都不同（byte-level 回退 + BPE merge），
// 而 SentencePiece 的空白与子词语义与之不一致——混用会让"分词对不对"跟"数值对不对"纠缠在一起
// （背景见 docs/future_iterations.md §5.1）。
//
// **判据是"与 HF 逐 token 全等"**：参考实现 = `transformers.GPT2TokenizerFast`，
// 参考数据 = tests/data/gpt2_tokenizer_golden.json（由 tools/make_tokenizer_golden.py 产出，
// 含 samples 的 text / ids / decoded 与来源文件的 SHA256）。
//
// 两处必须写在明面上的取舍：
//
// 1. **Unicode 分类是近似的**。预切分要区分"字母 / 数字 / 空白 / 其它"，而 C++ 标准库不带
//    Unicode 属性表。这里的规则是：ASCII 精确判断；非 ASCII **默认算字母**，只有落在空白表、
//    数字表与标点/符号区间（见 .cpp 的符号区间表）里的算"其它"。
//    **已知限制**：极少数非常用符号可能被归错类，表现为与 HF 的预切分不同。
//    处置方式明确：命中就补区间并加样本，判据以 golden 对拍为准（不是"看起来对"）。
//
// 2. **`Load` 的入参是"目录"而不是单个文件**：byte-level BPE 需要 `vocab.json` 与
//    `merges.txt` 两个文件，而 `BaseTokenizer::Load` 只有一个路径参数——基类注释原话是
//    "具体路径含义由子类决定"，所以这里把入参解释为目录，避免改动已交付的基类接口。
class BpeTokenizer : public BaseTokenizer {
 public:
    // 失败即返回 false（缺文件 / 坏 JSON / 词表与 merges 不自洽都算失败），**不抛异常**：
    // 调用方（测试与将来的文本入口）按返回值判断，与 SentencePieceTokenizer 的约定一致。
    bool Load(const std::string& vocab_dir) override;

    // 文本 -> token ids。词表里查不到的符号属于真故障：打日志并返回空 vector
    // （空 vector 在本项目里已经约定为"失败"，见 LLMRunner::Generate 的约定）。
    std::vector<int64_t> Encode(const std::string& text) const override;

    // token ids -> 文本。GPT-2 的 byte-level 解码是"把每个 token 的码点还原成字节"再按 UTF-8 组装；
    // 非法 UTF-8 字节按 HF 的口径替换成 U+FFFD（而不是抛异常）。
    std::string Decode(const std::vector<int64_t>& ids) const override;

    size_t VocabSize() const override { return id_to_token_.size(); }

    // 预切分（不做 BPE merge），返回**原始文本**切出的片段。抽出来是为了排错：
    // 与 HF 不一致时，第一步要分清是"预切分错了"还是"merge 错了"——两者修法完全不同。
    std::vector<std::string> PreTokenizeForTesting(const std::string& text) const;

    // 预切分 + byte 映射后的片段，**与 HF 的 `pre_tokenize_str` 输出同口径**（都是 byte-encoded）。
    // 参考数据里存的就是这一份：不一致时可以直接逐条比对，定位到是切分还是 merge 的问题。
    std::vector<std::string> ByteEncodedPiecesForTesting(const std::string& text) const;

 private:
    // 预切分（GPT-2 的 `'s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` 语义）。
    std::vector<std::string> PreTokenize(const std::string& text) const;

    // 原始字节 -> 可打印码点（GPT-2 的 bytes_to_unicode）。
    std::string ByteEncode(const std::string& raw_bytes) const;

    // 一段（已 byte-encode 的）文本按 merges 做 BPE，返回最终符号串。
    std::vector<std::string> ApplyBpe(const std::string& piece) const;

    // 已加载的词表：token 字符串 <-> id。
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;

    // merge 优先级：key = first + '\x01' + second。用分隔符拼 key 是因为 token 里不会出现
    // 原始控制字节（它们已被映射到 U+0100+ 的码点），所以 '\x01' 不会与内容混淆。
    std::unordered_map<std::string, int32_t> merge_ranks_;

    // byte <-> 码点 的双向映射（GPT-2 的 bytes_to_unicode 表，加载时构建一次）。
    std::unordered_map<uint32_t, uint8_t> codepoint_to_byte_;
    std::unordered_map<uint8_t, uint32_t> byte_to_codepoint_;
};

}  // namespace mini_trt_llm
