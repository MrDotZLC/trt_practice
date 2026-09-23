#pragma once

#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/tokenizer/base_tokenizer.hpp"
#include <memory>
#include <vector>

namespace mini_trt_llm {

// LLM 自回归生成 Runner。
// Phase 0 仅声明，Phase 2 实现。
class LLMRunner {
 public:
    struct GenerateOptions {
        int max_new_tokens = 20;
        float temperature = 1.0f;
        int top_k = 1;
        float top_p = 1.0f;
    };

    LLMRunner(std::shared_ptr<Engine> prefill_engine,
              std::shared_ptr<Engine> decode_engine,
              std::shared_ptr<BaseTokenizer> tokenizer);
    ~LLMRunner();

    std::vector<int64_t> Generate(const std::vector<int64_t>& input_ids,
                                  const GenerateOptions& options);

 private:
    std::shared_ptr<Engine> prefill_engine_;
    std::shared_ptr<Engine> decode_engine_;
    std::shared_ptr<BaseTokenizer> tokenizer_;
};

}  // namespace mini_trt_llm
