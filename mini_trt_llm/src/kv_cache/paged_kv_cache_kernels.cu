#include "mini_trt_llm/kv_cache/paged_kv_cache_kernels.hpp"

#include "mini_trt_llm/utils/cuda_dtype.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mini_trt_llm {
namespace {

constexpr int32_t kThreadsPerBlock = 256;

// 把源张量 [batch, kv_heads, tokens, head_size] 的每个元素搬到分页 cache。
//
// 一个线程负责一个元素：搬运动作没有归约、没有跨元素依赖，
// 用"一元素一线程 + grid-stride"换取最简单且无分支错误的索引推导。
// 维度分解从最后一维往前推，避免出现多个维度的整数除法混在一起。
template <typename SrcT, typename DstT>
__global__ void WriteKVKernel(const SrcT* __restrict__ key, const SrcT* __restrict__ value,
                              DstT* __restrict__ key_cache, DstT* __restrict__ value_cache,
                              const int32_t* __restrict__ block_tables,
                              const int32_t* __restrict__ context_lens,
                              int32_t kv_heads, int32_t tokens, int32_t head_size,
                              int32_t block_size, int32_t max_blocks_per_seq,
                              bool append, int64_t elements) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t d = static_cast<int32_t>(i % head_size);
        int64_t rest = i / head_size;
        const int32_t t = static_cast<int32_t>(rest % tokens);
        rest /= tokens;
        const int32_t h = static_cast<int32_t>(rest % kv_heads);
        const int32_t b = static_cast<int32_t>(rest / kv_heads);

        // prefill 从 0 开始覆盖写；decode 从当前语境长度处追加。
        const int32_t position = (append ? context_lens[b] : 0) + t;
        const int32_t physical_block = block_tables[b * max_blocks_per_seq +
                                                   position / block_size];
        const int32_t slot = position % block_size;
        const int64_t cache_offset =
            ((static_cast<int64_t>(physical_block) * block_size + slot) * kv_heads + h) *
                head_size +
            d;
        // 源与目标精度可能不同（见头文件说明）：统一经 float 中转，避免直接
        // static_cast 在半精度/单精度之间踩隐式取整规则的坑。
        key_cache[cache_offset] = cuda::FromFloat<DstT>(cuda::ToFloat(key[i]));
        value_cache[cache_offset] = cuda::FromFloat<DstT>(cuda::ToFloat(value[i]));
    }
}

__global__ void AdvanceContextLensKernel(int32_t* __restrict__ context_lens, int32_t tokens,
                                         int32_t batch_size) {
    const int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        context_lens[b] += tokens;
    }
}

}  // namespace

cudaError_t LaunchWriteKV(const PagedKVWriteArgs& args, cudaStream_t stream) {
    if (args.key == nullptr || args.value == nullptr || args.key_cache == nullptr ||
        args.value_cache == nullptr || args.block_tables == nullptr ||
        args.context_lens == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (args.batch_size <= 0 || args.tokens <= 0 || args.num_kv_heads <= 0 ||
        args.head_size <= 0 || args.block_size <= 0 || args.max_blocks_per_seq <= 0) {
        return cudaErrorInvalidValue;
    }

    const int64_t elements = static_cast<int64_t>(args.batch_size) * args.tokens *
                             args.num_kv_heads * args.head_size;
    const int64_t blocks = (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > 0x7fffffffLL) {
        return cudaErrorInvalidValue;
    }

    // CUDA 的 last-error 是粘性的：先清掉入口处可能残留的旧错误，
    // 后面 cudaGetLastError() 的结果才只反映本次 launch（见 TROUBLESHOOTING #13）。
    (void)cudaGetLastError();

    // 四种组合（FP32/FP16 × FP32/FP16）：源由引擎决定、目标由 cache 决定，
    // 两者独立，所以必须显式分发而不是假定一致。
    const dim3 grid(static_cast<unsigned int>(blocks));
    const auto launch = [&](auto src_tag, auto dst_tag) {
        using SrcT = decltype(src_tag);
        using DstT = decltype(dst_tag);
        WriteKVKernel<SrcT, DstT><<<grid, kThreadsPerBlock, 0, stream>>>(
            static_cast<const SrcT*>(args.key), static_cast<const SrcT*>(args.value),
            static_cast<DstT*>(args.key_cache), static_cast<DstT*>(args.value_cache),
            args.block_tables, args.context_lens, args.num_kv_heads, args.tokens,
            args.head_size, args.block_size, args.max_blocks_per_seq, args.append, elements);
    };
    if (args.source_is_half && args.is_half) {
        launch(__half{}, __half{});
    } else if (args.source_is_half && !args.is_half) {
        launch(__half{}, float{});
    } else if (!args.source_is_half && args.is_half) {
        launch(float{}, __half{});
    } else {
        launch(float{}, float{});
    }
    return cudaGetLastError();
}

cudaError_t LaunchAdvanceContextLens(int32_t* context_lens, int32_t batch_size,
                                     int32_t tokens, cudaStream_t stream) {
    if (context_lens == nullptr || batch_size <= 0 || tokens <= 0) {
        return cudaErrorInvalidValue;
    }
    (void)cudaGetLastError();
    const int32_t blocks = (batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    AdvanceContextLensKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(context_lens, tokens,
                                                                     batch_size);
    return cudaGetLastError();
}

}  // namespace mini_trt_llm
