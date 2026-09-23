#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace mini_trt_llm {

// Greedy 采样 CUDA kernel声明。
// Phase 0 仅声明，Phase 1 实现。
void GreedySample(const float* logits, int batch_size, int vocab_size,
                  std::vector<int>* output_ids, cudaStream_t stream);

}  // namespace mini_trt_llm
