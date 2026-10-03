#pragma once

// 仅可在 .cu 中 include：本文件包含 __device__ 函数。

#include <cuda_runtime.h>

namespace mini_trt_llm {
namespace cuda {

// Block 内求和归约，结果通过 shared memory 广播给所有线程。
//
// scratch 至少需要 (blockDim.x / 32) 个 float。首尾各放一次 __syncthreads()，
// 使连续调用天然安全——否则下一轮的 scratch 写入可能与本轮仍在读 scratch[0] 的线程竞争。
//
// 用 __shfl_down_sync 而非共享内存树形归约：warp 内 shuffle 无需同步且没有 bank 冲突。
// 前提是 blockDim.x 为 32 的整数倍，这样每个 warp 都是满编的。
__device__ __forceinline__ float BlockReduceSum(float val, float* scratch) {
    constexpr unsigned int kFullWarpMask = 0xffffffffu;

    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullWarpMask, val, offset);
    }
    if (lane == 0) {
        scratch[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float total = (lane < num_warps) ? scratch[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(kFullWarpMask, total, offset);
        }
        if (lane == 0) {
            scratch[0] = total;
        }
    }
    __syncthreads();
    return scratch[0];
}

// Block 内取最大值归约，结果通过 shared memory 广播给所有线程。
//
// 与 BlockReduceSum 同一套同步纪律与前提（scratch 至少 (blockDim.x/32) 个 float、
// blockDim.x 为 32 的整数倍、首尾各一次 __syncthreads()）—— 存在的理由：
// S4 的 packed context kernel 要按**段内 key 维**求 softmax 的最大值（每条序列长度不同，
// 不能在寄存器里串行求），必须有一次 block 级归约。
__device__ __forceinline__ float BlockReduceMax(float val, float* scratch) {
    constexpr unsigned int kFullWarpMask = 0xffffffffu;

    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(kFullWarpMask, val, offset));
    }
    if (lane == 0) {
        scratch[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float best = (lane < num_warps) ? scratch[lane] : -CUDART_INF_F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            best = fmaxf(best, __shfl_down_sync(kFullWarpMask, best, offset));
        }
        if (lane == 0) {
            scratch[0] = best;
        }
    }
    __syncthreads();
    return scratch[0];
}

}  // namespace cuda
}  // namespace mini_trt_llm
