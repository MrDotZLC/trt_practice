#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace mini_trt_llm {

// Top-K 采样 CUDA kernel声明。
// Phase 0 仅声明，Phase 1 实现。
void TopKSample(const float* logits, int batch_size, int vocab_size, int top_k,
                std::vector<int>* output_ids, cudaStream_t stream);

}  // namespace mini_trt_llm
