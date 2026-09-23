#include "mini_trt_llm/sampler/greedy_sampler.hpp"
#include <stdexcept>

namespace mini_trt_llm {

void GreedySample(const float* logits, int batch_size, int vocab_size,
                  std::vector<int>* output_ids, cudaStream_t stream) {
    // Phase 1 实现
    (void)logits;
    (void)batch_size;
    (void)vocab_size;
    (void)output_ids;
    (void)stream;
    throw std::runtime_error("GreedySample not implemented in Phase 0");
}

}  // namespace mini_trt_llm
