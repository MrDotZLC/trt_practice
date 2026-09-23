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

    // 加载词表文件（具体路径含义由子类决定）
    virtual bool Load(const std::string& vocab_path) = 0;

    // 文本 -> token ids
    virtual std::vector<int64_t> Encode(const std::string& text) const = 0;

    // token ids -> 文本
    virtual std::string Decode(const std::vector<int64_t>& ids) const = 0;

    // 词表大小
    virtual size_t VocabSize() const = 0;
};

}  // namespace mini_trt_llm
