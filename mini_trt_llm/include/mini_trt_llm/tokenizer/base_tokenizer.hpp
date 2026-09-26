#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

// 多模态 tokenizer 抽象接口。
// 后续可扩展 SentencePiece、BPE、Tiktoken、CLIP 等实现。
class BaseTokenizer {
 public:
    virtual ~BaseTokenizer() = default;

    // 加载词表资源。**入参的含义由子类决定**——这不是随手放宽，而是刻意的接口分工：
    // 不同 tokenizer 的资产形态本来就不同，SentencePiece 收单个 `.model` 文件，
    // BpeTokenizer 收含 `vocab.json` + `merges.txt` 的**目录**（见 bpe_tokenizer.hpp）。
    // 若要统一语义，先按 AGENTS.md §5 出计划：这属于改已交付接口，不是顺手能改的事。
    virtual bool Load(const std::string& vocab_path) = 0;

    // 文本 -> token ids
    virtual std::vector<int64_t> Encode(const std::string& text) const = 0;

    // token ids -> 文本
    virtual std::string Decode(const std::vector<int64_t>& ids) const = 0;

    // 词表大小
    virtual size_t VocabSize() const = 0;
};

}  // namespace mini_trt_llm
