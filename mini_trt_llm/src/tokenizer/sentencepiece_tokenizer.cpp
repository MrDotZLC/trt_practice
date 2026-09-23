#include "mini_trt_llm/tokenizer/sentencepiece_tokenizer.hpp"
#include "mini_trt_llm/utils/logger.hpp"

namespace mini_trt_llm {

SentencePieceTokenizer::SentencePieceTokenizer()
    : processor_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

bool SentencePieceTokenizer::Load(const std::string& vocab_path) {
    auto status = processor_->Load(vocab_path);
    if (!status.ok()) {
        MINI_TRT_LOG_ERROR("Failed to load SentencePiece model: " << vocab_path
                         << ", error: " << status.ToString());
        return false;
    }
    MINI_TRT_LOG_INFO("Loaded SentencePiece model, vocab size: "
                    << processor_->GetPieceSize());
    return true;
}

std::vector<int64_t> SentencePieceTokenizer::Encode(
    const std::string& text) const {
    std::vector<int> ids;
    processor_->Encode(text, &ids);
    return std::vector<int64_t>(ids.begin(), ids.end());
}

std::string SentencePieceTokenizer::Decode(
    const std::vector<int64_t>& ids) const {
    std::vector<int> int_ids(ids.begin(), ids.end());
    std::string text;
    processor_->Decode(int_ids, &text);
    return text;
}

size_t SentencePieceTokenizer::VocabSize() const {
    return processor_->GetPieceSize();
}

}  // namespace mini_trt_llm
