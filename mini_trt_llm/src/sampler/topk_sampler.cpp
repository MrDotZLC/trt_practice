#include "mini_trt_llm/sampler/topk_sampler.hpp"
#include <stdexcept>

namespace mini_trt_llm {

void TopKSample(const float* logits, int batch_size, int vocab_size, int top_k,
                std::vector<int>* output_ids, cudaStream_t stream) {
    // Phase 1 实现
    (void)logits;
    (void)batch_size;
    (void)vocab_size;
    (void)top_k;
    (void)output_ids;
    (void)stream;
    throw std::runtime_error("TopKSample not implemented in Phase 0");
}

}  // namespace mini_trt_llm
