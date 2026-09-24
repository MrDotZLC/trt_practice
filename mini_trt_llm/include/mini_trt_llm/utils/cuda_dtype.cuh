#pragma once

// 仅可在 .cu 中 include：本文件包含 __device__ 函数。

#include <cuda_fp16.h>

namespace mini_trt_llm {
namespace cuda {

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }

// 累加统一在 FP32 上做，只在写回时降精度：FP16 累加会显著放大误差。
template <typename T>
__device__ __forceinline__ T FromFloat(float v);
template <>
__device__ __forceinline__ float FromFloat<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __half FromFloat<__half>(float v) {
    return __float2half_rn(v);
}

}  // namespace cuda
}  // namespace mini_trt_llm
