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
                              const int32_t* __restrict__ rows,
                              int32_t kv_heads, int32_t tokens, int32_t head_size,
                              int32_t block_size, int32_t max_blocks_per_seq,
                              bool append, int64_t elements,
                              const int32_t* __restrict__ cu_seqlens_ctx,
                              int32_t context_seq_count) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t d = static_cast<int32_t>(i % head_size);
        int64_t rest = i / head_size;
        const int32_t t = static_cast<int32_t>(rest % tokens);
        rest /= tokens;
        const int32_t h = static_cast<int32_t>(rest % kv_heads);
        const int32_t engine_row = static_cast<int32_t>(rest / kv_heads);

        // 引擎行 → 缓存批内行：源张量的行序与块表的行序不同源时（S3 的活跃批）必须按映射
        // 寻址，否则会写进**别的序列**自己的块（静默算错，见 PagedKVWriteArgs::rows）。
        const int32_t row = rows[engine_row];
        // S4：源张量是 packed 的（context 段在前、generation 段在后），源行基址因此不是
        // `engine_row * tokens`。两个分支的公式见 PagedKVWriteArgs::cu_seqlens_ctx 的说明；
        // `cu_seqlens_ctx` 为空时保持既有语义（每行等长）。
        const int32_t source_row =
            (cu_seqlens_ctx == nullptr)
                ? engine_row
                : (append ? (cu_seqlens_ctx[context_seq_count] + engine_row)
                          : cu_seqlens_ctx[engine_row]);
        const int64_t source_index =
            ((static_cast<int64_t>(source_row) * kv_heads + h) * tokens + t) * head_size + d;
        // prefill 从 0 开始覆盖写；decode 从当前语境长度处追加。
        const int32_t position = (append ? context_lens[row] : 0) + t;
        const int32_t physical_block = block_tables[row * max_blocks_per_seq +
                                                   position / block_size];
        const int32_t slot = position % block_size;
        const int64_t cache_offset =
            ((static_cast<int64_t>(physical_block) * block_size + slot) * kv_heads + h) *
                head_size +
            d;
        // 源与目标精度可能不同（见头文件说明）：统一经 float 中转，避免直接
        // static_cast 在半精度/单精度之间踩隐式取整规则的坑。
        key_cache[cache_offset] = cuda::FromFloat<DstT>(cuda::ToFloat(key[source_index]));
        value_cache[cache_offset] = cuda::FromFloat<DstT>(cuda::ToFloat(value[source_index]));
    }
}

// **S4 的 packed 写回**：源是打包张量（context 段的全部 token 在前），**每行的长度不同**，
// 所以不能用"统一 stride × 行号"去分解线性下标（那正是上面那个通用 kernel 的做法）。
// 这里改成"**一个 block 负责一行**"：行的区间由 `cu_seqlens_ctx` 给出（段内下标），
// 目标缓存行由 `rows` 给出，块内位置从 0 覆盖写（prefill 语义）。
//
// 越界保护：`cu_seqlens_ctx` 是调用方（runner）与自己构造的 `row_lengths` 同源给出的，
// 预留量的校验在 host 侧（`WritePrefillKV` 的逐行检查）已经做过 —— kernel 里不再重复查，
// 但因此**必须**保证两者同源（这也是为什么它只由 PagedKVCache 自己调用）。
template <typename SrcT, typename DstT>
__global__ void WriteKVPackedPrefillKernel(const SrcT* __restrict__ key,
                                           const SrcT* __restrict__ value,
                                           DstT* __restrict__ key_cache,
                                           DstT* __restrict__ value_cache,
                                           const int32_t* __restrict__ block_tables,
                                           const int32_t* __restrict__ rows,
                                           const int32_t* __restrict__ cu_seqlens_ctx,
                                           int32_t kv_heads, int32_t head_size,
                                           int32_t block_size, int32_t max_blocks_per_seq,
                                           const int32_t* __restrict__ row_starts) {
    const int32_t engine_row = blockIdx.x;
    const int32_t begin = cu_seqlens_ctx[engine_row];
    const int32_t len = cu_seqlens_ctx[engine_row + 1] - begin;
    const int32_t row = rows[engine_row];
    const int64_t elements = static_cast<int64_t>(len) * kv_heads * head_size;
    for (int64_t i = threadIdx.x; i < elements; i += blockDim.x) {
        const int32_t d = static_cast<int32_t>(i % head_size);
        int64_t rest = i / head_size;
        const int32_t h = static_cast<int32_t>(rest % kv_heads);
        const int32_t t = static_cast<int32_t>(rest / kv_heads);
        const int64_t source_index =
            ((static_cast<int64_t>(begin + t) * kv_heads + h) * head_size) + d;
        // S5：写回起点（分块 prefill 的第 2 块起必须从 `prompt_done` 续写，不能从 0 覆盖）。
        // nullptr = 0 → 与 S3/S4 的"从 0 覆盖写"逐位相同。
        const int32_t start = (row_starts != nullptr) ? row_starts[engine_row] : 0;
        const int32_t position = start + t;
        const int32_t physical_block =
            block_tables[static_cast<size_t>(row) * max_blocks_per_seq + position / block_size];
        const int32_t slot = position % block_size;
        const int64_t cache_offset =
            ((static_cast<int64_t>(physical_block) * block_size + slot) * kv_heads + h) *
                head_size +
            d;
        key_cache[cache_offset] = cuda::FromFloat<DstT>(cuda::ToFloat(key[source_index]));
        value_cache[cache_offset] = cuda::FromFloat<DstT>(cuda::ToFloat(value[source_index]));
    }
}

// 推进语境长度。`rows == nullptr` 时按恒等映射推进**前 batch_size 行**（S3/S4 的行为，
// 因为那时"本步追加的行"就是活跃前缀 = cache 前缀）；S5 分块后两者可能不再重合，
// 所以给一张**显式行列表**（每步重建、行号指向 cache 内部行空间）。
__global__ void AdvanceContextLensKernel(int32_t* __restrict__ context_lens, int32_t tokens,
                                         int32_t batch_size,
                                         const int32_t* __restrict__ rows) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        const int32_t b = (rows != nullptr) ? rows[i] : i;
        context_lens[b] += tokens;
    }
}

}  // namespace

cudaError_t LaunchWriteKV(const PagedKVWriteArgs& args, cudaStream_t stream) {
    if (args.key == nullptr || args.value == nullptr || args.key_cache == nullptr ||
        args.value_cache == nullptr || args.block_tables == nullptr ||
        args.context_lens == nullptr || args.rows == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (args.row_count <= 0 || args.tokens <= 0 || args.num_kv_heads <= 0 ||
        args.head_size <= 0 || args.block_size <= 0 || args.max_blocks_per_seq <= 0) {
        return cudaErrorInvalidValue;
    }

    // 元素总数按**本次参与的行数**算：源张量只有 row_count 行是本次算出来的，
    // 用批内已登记序列数会多写那些残留行（见 PagedKVWriteArgs::rows）。
    const int64_t elements = static_cast<int64_t>(args.row_count) * args.tokens *
                             args.num_kv_heads * args.head_size;
    const int64_t blocks = (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > 0x7fffffffLL) {
        return cudaErrorInvalidValue;
    }

    // CUDA 的 last-error 是粘性的：先清掉入口处可能残留的旧错误，
    // 后面 cudaGetLastError() 的结果才只反映本次 launch（见 TROUBLESHOOTING + TS-013）。
    (void)cudaGetLastError();

    // **S4 的 packed prefill**：行长不等，走"一个 block 一行"的专用 kernel
    // （通用 kernel 按 `tokens` 统一 stride 分解，packed 下不成立）。
    // 启动的 grid 是行数，因此这里不按元素数算块数。
    if (args.cu_seqlens_ctx != nullptr && !args.append && args.rows != nullptr) {
        const dim3 packed_grid(static_cast<unsigned int>(args.row_count));
        const auto launch_packed = [&](auto src_tag, auto dst_tag) {
            using SrcT = decltype(src_tag);
            using DstT = decltype(dst_tag);
            WriteKVPackedPrefillKernel<SrcT, DstT><<<packed_grid, kThreadsPerBlock, 0, stream>>>(
                static_cast<const SrcT*>(args.key), static_cast<const SrcT*>(args.value),
                static_cast<DstT*>(args.key_cache), static_cast<DstT*>(args.value_cache),
                args.block_tables, args.rows, args.cu_seqlens_ctx, args.num_kv_heads,
                args.head_size, args.block_size, args.max_blocks_per_seq, args.row_starts);
        };
        if (args.source_is_half && args.is_half) {
            launch_packed(__half{}, __half{});
        } else if (args.source_is_half && !args.is_half) {
            launch_packed(__half{}, float{});
        } else if (!args.source_is_half && args.is_half) {
            launch_packed(float{}, __half{});
        } else {
            launch_packed(float{}, float{});
        }
        return cudaGetLastError();
    }

    // 四种组合（FP32/FP16 × FP32/FP16）：源由引擎决定、目标由 cache 决定，
    // 两者独立，所以必须显式分发而不是假定一致。
    const dim3 grid(static_cast<unsigned int>(blocks));
    const auto launch = [&](auto src_tag, auto dst_tag) {
        using SrcT = decltype(src_tag);
        using DstT = decltype(dst_tag);
        WriteKVKernel<SrcT, DstT><<<grid, kThreadsPerBlock, 0, stream>>>(
            static_cast<const SrcT*>(args.key), static_cast<const SrcT*>(args.value),
            static_cast<DstT*>(args.key_cache), static_cast<DstT*>(args.value_cache),
            args.block_tables, args.context_lens, args.rows, args.num_kv_heads, args.tokens,
            args.head_size, args.block_size, args.max_blocks_per_seq, args.append, elements,
            args.cu_seqlens_ctx, args.context_seq_count);
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
                                     int32_t tokens, cudaStream_t stream, const int32_t* rows) {
    if (context_lens == nullptr || batch_size <= 0 || tokens <= 0) {
        return cudaErrorInvalidValue;
    }
    (void)cudaGetLastError();
    const int32_t blocks = (batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    AdvanceContextLensKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(context_lens, tokens,
                                                                     batch_size, rows);
    return cudaGetLastError();
}

}  // namespace mini_trt_llm
