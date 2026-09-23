#include "mini_trt_llm/core/llm_runner.hpp"
#include <stdexcept>

namespace mini_trt_llm {

LLMRunner::LLMRunner(std::shared_ptr<Engine> prefill_engine,
                     std::shared_ptr<Engine> decode_engine,
                     std::shared_ptr<BaseTokenizer> tokenizer)
    : prefill_engine_(prefill_engine),
      decode_engine_(decode_engine),
      tokenizer_(tokenizer) {}

LLMRunner::~LLMRunner() = default;

std::vector<int64_t> LLMRunner::Generate(const std::vector<int64_t>& input_ids,
                                         const GenerateOptions& options) {
    // Phase 2 实现
    (void)input_ids;
    (void)options;
    throw std::runtime_error("LLMRunner::Generate not implemented in Phase 0");
}

}  // namespace mini_trt_llm
