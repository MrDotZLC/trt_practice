#pragma once

#include "mini_trt_llm/tokenizer/base_tokenizer.hpp"
#include <memory>
#include <sentencepiece_processor.h>
#include <string>
#include <vector>

namespace mini_trt_llm {

// 基于 SentencePiece 的 tokenizer 实现。
class SentencePieceTokenizer : public BaseTokenizer {
 public:
    SentencePieceTokenizer();
    ~SentencePieceTokenizer() override;

    bool Load(const std::string& vocab_path) override;

    std::vector<int64_t> Encode(const std::string& text) const override;

    std::string Decode(const std::vector<int64_t>& ids) const override;

    size_t VocabSize() const override;

 private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
};

}  // namespace mini_trt_llm
