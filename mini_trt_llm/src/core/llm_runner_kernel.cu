#include "mini_trt_llm/core/llm_runner_kernel.hpp"

#include <cuda_runtime.h>

namespace mini_trt_llm {
namespace {

constexpr int32_t kThreadsPerBlock = 128;

// position_ids 是 [batch, 1]，所以每个序列只写一个元素；batch 通常很小（本版为 1），
// 一个 block 足够，不需要 grid-stride。
__global__ void FillPositionIdsKernel(const int32_t* __restrict__ context_lens,
                                      int32_t* __restrict__ position_ids,
                                      int32_t batch_size) {
    const int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        position_ids[b] = context_lens[b];
    }
}

}  // namespace

cudaError_t LaunchFillPositionIds(const int32_t* context_lens, int32_t* position_ids,
                                  int32_t batch_size, cudaStream_t stream) {
    if (context_lens == nullptr || position_ids == nullptr || batch_size <= 0) {
        return cudaErrorInvalidValue;
    }
    // CUDA 的 last-error 是粘性的：先清掉入口处可能残留的旧错误，
    // 后面 cudaGetLastError() 的结果才只反映本次 launch（见 TROUBLESHOOTING #13）。
    (void)cudaGetLastError();
    const int32_t blocks = (batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    FillPositionIdsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(context_lens,
                                                                  position_ids, batch_size);
    return cudaGetLastError();
}

}  // namespace mini_trt_llm
