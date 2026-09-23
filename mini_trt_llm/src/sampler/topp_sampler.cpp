#include "mini_trt_llm/sampler/topp_sampler.hpp"
#include <stdexcept>

namespace mini_trt_llm {

void TopPSample(const float* logits, int batch_size, int vocab_size,
                float top_p, std::vector<int>* output_ids,
                cudaStream_t stream) {
    // Phase 1 实现
    (void)logits;
    (void)batch_size;
    (void)vocab_size;
    (void)top_p;
    (void)output_ids;
    (void)stream;
    throw std::runtime_error("TopPSample not implemented in Phase 0");
}

}  // namespace mini_trt_llm
