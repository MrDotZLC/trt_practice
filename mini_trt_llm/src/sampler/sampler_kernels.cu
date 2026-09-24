#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_dtype.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace mini_trt_llm {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr size_t kWorkspaceAlign = 256;

using cuda::ToFloat;

__device__ __forceinline__ uint32_t MultiplyHigh(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}

// Philox4x32-10 单轮。
__device__ __forceinline__ void PhiloxRound(uint32_t* counter, uint32_t key0,
                                            uint32_t key1) {
    constexpr uint32_t kMultiplier0 = 0xD2511F53u;
    constexpr uint32_t kMultiplier1 = 0xCD9E8D57u;
    const uint32_t hi0 = MultiplyHigh(kMultiplier0, counter[0]);
    const uint32_t lo0 = kMultiplier0 * counter[0];
    const uint32_t hi1 = MultiplyHigh(kMultiplier1, counter[2]);
    const uint32_t lo1 = kMultiplier1 * counter[2];
    counter[0] = hi1 ^ counter[1] ^ key0;
    counter[1] = lo1;
    counter[2] = hi0 ^ counter[3] ^ key1;
    counter[3] = lo0;
}

// 基于 (seed, offset, index) 的确定性 [0,1) 均匀随机数。
//
// 不用 curand：省掉一个链接依赖，且采样只需一个均匀数；自实现 Philox 让测试用固定
// seed 即可完全复现（Q8 / Q12）。
__device__ __forceinline__ float Uniform01(uint64_t seed, uint64_t offset, uint32_t index) {
    uint32_t counter[4] = {static_cast<uint32_t>(offset),
                           static_cast<uint32_t>(offset >> 32),
                           index,
                           0u};
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

#pragma unroll
    for (int round = 0; round < 10; ++round) {
        PhiloxRound(counter, key0, key1);
        key0 += 0x9E3779B9u;
        key1 += 0xBB67AE85u;
    }
    // 取高 24 位构造尾数，保证落在 [0, 1)
    return static_cast<float>(counter[0] >> 8) * (1.0f / 16777216.0f);
}

// 并列时取最小下标，与 torch.argmax 语义一致。
__device__ __forceinline__ bool IsBetter(float value, int32_t index, float best_value,
                                         int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

template <typename T>
__global__ void GreedyKernel(const T* __restrict__ logits, int32_t* __restrict__ token_ids,
                             int32_t vocab_size) {
    __shared__ float shared_value[kThreadsPerBlock / 32];
    __shared__ int32_t shared_index[kThreadsPerBlock / 32];

    const int32_t row = blockIdx.x;
    const T* row_logits = logits + static_cast<size_t>(row) * vocab_size;

    float best_value = -CUDART_INF_F;
    int32_t best_index = 0;
    for (int32_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        const float value = ToFloat(row_logits[i]);
        if (IsBetter(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
        const int32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (IsBetter(other_value, other_index, best_value, best_index)) {
            best_value = other_value;
            best_index = other_index;
        }
    }
    if (lane == 0) {
        shared_value[warp] = best_value;
        shared_index[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        best_value = (lane < num_warps) ? shared_value[lane] : -CUDART_INF_F;
        best_index = (lane < num_warps) ? shared_index[lane] : 0x7fffffff;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
            const int32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
            if (IsBetter(other_value, other_index, best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            token_ids[row] = best_index;
        }
    }
}

// 为分段排序准备输入：把 logits 转成 FP32 key，并生成 0..vocab-1 的下标。
// 下标逐行重复，排序时随 key 一起置换，从而把"排序值"还原成"词表下标"。
template <typename T>
__global__ void PrepareSortInputKernel(const T* __restrict__ logits,
                                       float* __restrict__ keys,
                                       int32_t* __restrict__ indices,
                                       int32_t* __restrict__ offsets, int32_t batch_size,
                                       int32_t vocab_size) {
    const int64_t total = static_cast<int64_t>(batch_size) * vocab_size;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        keys[i] = ToFloat(logits[i]);
        indices[i] = static_cast<int32_t>(i % vocab_size);
    }

    // 分段边界只由一个 block 生成，避免多个 block 重复写
    if (blockIdx.x == 0) {
        for (int32_t segment = threadIdx.x; segment <= batch_size; segment += blockDim.x) {
            offsets[segment] = segment * vocab_size;
        }
    }
}

__global__ void TopKSampleKernel(const float* __restrict__ sorted_logits,
                                 const int32_t* __restrict__ sorted_indices,
                                 const int32_t* __restrict__ top_k,
                                 int32_t* __restrict__ token_ids, int32_t batch_size,
                                 int32_t vocab_size, uint64_t seed, uint64_t offset) {
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    // 最后一个 block 可能不满，必须挡住越界的 row
    if (row >= batch_size) {
        return;
    }

    int32_t k = top_k[row];
    if (k < 1) {
        k = 1;
    }
    if (k > vocab_size) {
        k = vocab_size;
    }

    const float* row_logits = sorted_logits + static_cast<size_t>(row) * vocab_size;
    const int32_t* row_indices = sorted_indices + static_cast<size_t>(row) * vocab_size;

    // 已按降序排好，第 0 个就是 top-k 内的最大值，可直接作为 softmax 的稳定项
    const float max_value = row_logits[0];
    float total = 0.0f;
    for (int32_t i = 0; i < k; ++i) {
        total += __expf(row_logits[i] - max_value);
    }

    const float target = Uniform01(seed, offset, static_cast<uint32_t>(row)) * total;
    float cumulative = 0.0f;
    int32_t chosen = row_indices[0];
    for (int32_t i = 0; i < k; ++i) {
        cumulative += __expf(row_logits[i] - max_value);
        if (cumulative >= target) {
            chosen = row_indices[i];
            break;
        }
    }
    token_ids[row] = chosen;
}

__global__ void TopPSampleKernel(const float* __restrict__ sorted_logits,
                                 const int32_t* __restrict__ sorted_indices,
                                 const float* __restrict__ top_p,
                                 int32_t* __restrict__ token_ids, int32_t batch_size,
                                 int32_t vocab_size, uint64_t seed, uint64_t offset) {
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    // 最后一个 block 可能不满，必须挡住越界的 row
    if (row >= batch_size) {
        return;
    }

    const float* row_logits = sorted_logits + static_cast<size_t>(row) * vocab_size;
    const int32_t* row_indices = sorted_indices + static_cast<size_t>(row) * vocab_size;

    float p = top_p[row];
    if (!(p > 0.0f)) {
        p = 1.0f;  // NaN / 非正值一律退化为不截断，避免 kernel 内出现未定义行为
    }
    if (p > 1.0f) {
        p = 1.0f;
    }

    const float max_value = row_logits[0];
    float total = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        total += __expf(row_logits[i] - max_value);
    }

    // 取满足"累计概率 >= p"的最短前缀；至少保留 1 个 token
    float cumulative = 0.0f;
    int32_t cutoff = vocab_size;
    for (int32_t i = 0; i < vocab_size; ++i) {
        cumulative += __expf(row_logits[i] - max_value) / total;
        if (cumulative >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // 在被保留的前缀内重新归一化后采样
    float kept_total = 0.0f;
    for (int32_t i = 0; i < cutoff; ++i) {
        kept_total += __expf(row_logits[i] - max_value);
    }

    const float target = Uniform01(seed, offset, static_cast<uint32_t>(row)) * kept_total;
    cumulative = 0.0f;
    int32_t chosen = row_indices[0];
    for (int32_t i = 0; i < cutoff; ++i) {
        cumulative += __expf(row_logits[i] - max_value);
        if (cumulative >= target) {
            chosen = row_indices[i];
            break;
        }
    }
    token_ids[row] = chosen;
}

// 分段排序的 workspace 布局。
//
// Top-K / Top-P 都需要"按行降序排序"，用 CUB 的分段排序实现（Q15 选定 CUB）：
// 正确性优先，性能优化留到后续迭代（见 plan §9）。
struct SortWorkspace {
    float* keys_in = nullptr;
    int32_t* indices_in = nullptr;
    float* keys_out = nullptr;
    int32_t* indices_out = nullptr;
    int32_t* offsets = nullptr;
    void* cub_temp = nullptr;
};

char* AlignCursor(char* cursor) {
    const auto address = reinterpret_cast<uintptr_t>(cursor);
    const size_t misalignment = address % kWorkspaceAlign;
    return misalignment == 0 ? cursor : cursor + (kWorkspaceAlign - misalignment);
}

size_t SortTempBytes(int32_t batch_size, int32_t vocab_size) {
    // CUB 的 temp_storage_bytes 是 size_t 引用，传 int 会匹配不到重载
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_bytes, static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr), static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr), batch_size * vocab_size, batch_size,
        static_cast<const int32_t*>(nullptr), static_cast<const int32_t*>(nullptr), 0,
        static_cast<int>(sizeof(float) * 8), static_cast<cudaStream_t>(nullptr));
    return temp_bytes;
}

size_t SortWorkspaceBytes(int32_t batch_size, int32_t vocab_size) {
    const size_t elements = static_cast<size_t>(batch_size) * vocab_size;
    // 每段前各预留一次对齐余量，保证 PartitionSortWorkspace 一定能切分成功
    return 6 * kWorkspaceAlign +
           elements *
               (sizeof(float) + sizeof(int32_t) + sizeof(float) + sizeof(int32_t)) +
           (static_cast<size_t>(batch_size) + 1) * sizeof(int32_t) +
           SortTempBytes(batch_size, vocab_size);
}

bool PartitionSortWorkspace(void* workspace, size_t workspace_bytes, int32_t batch_size,
                            int32_t vocab_size, SortWorkspace* out) {
    const size_t elements = static_cast<size_t>(batch_size) * vocab_size;
    char* cursor = static_cast<char*>(workspace);
    char* end = cursor + workspace_bytes;

    auto take = [&cursor, end](size_t size) -> void* {
        cursor = AlignCursor(cursor);
        if (cursor + size > end) {
            return nullptr;
        }
        void* segment = cursor;
        cursor += size;
        return segment;
    };

    out->keys_in = static_cast<float*>(take(elements * sizeof(float)));
    out->indices_in = static_cast<int32_t*>(take(elements * sizeof(int32_t)));
    out->keys_out = static_cast<float*>(take(elements * sizeof(float)));
    out->indices_out = static_cast<int32_t*>(take(elements * sizeof(int32_t)));
    out->offsets =
        static_cast<int32_t*>(take((static_cast<size_t>(batch_size) + 1) * sizeof(int32_t)));
    out->cub_temp = take(SortTempBytes(batch_size, vocab_size));

    return out->keys_in != nullptr && out->indices_in != nullptr &&
           out->keys_out != nullptr && out->indices_out != nullptr &&
           out->offsets != nullptr && out->cub_temp != nullptr;
}

// Top-K / Top-P 共用的"排序 + 采样"流程。
template <typename SampleKernel>
cudaError_t SortThenSample(const SamplerArgs& args, cudaStream_t stream, void* workspace,
                           size_t workspace_bytes, SampleKernel sample_kernel) {
    if (workspace == nullptr || workspace_bytes == 0) {
        return cudaErrorInvalidValue;
    }

    SortWorkspace layout;
    if (!PartitionSortWorkspace(workspace, workspace_bytes, args.batch_size,
                                args.vocab_size, &layout)) {
        return cudaErrorInvalidValue;
    }

    const int32_t total_items = args.batch_size * args.vocab_size;
    const int32_t blocks = (total_items + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // CUDA 的 last-error 是粘性的：先清掉入口处可能残留的旧错误，
    // 后面 cudaGetLastError() 的结果才只反映本次调用里的 launch。
    (void)cudaGetLastError();
    if (args.is_half) {
        PrepareSortInputKernel<__half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __half*>(args.logits), layout.keys_in, layout.indices_in,
            layout.offsets, args.batch_size, args.vocab_size);
    } else {
        PrepareSortInputKernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const float*>(args.logits), layout.keys_in, layout.indices_in,
            layout.offsets, args.batch_size, args.vocab_size);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    size_t temp_bytes = SortTempBytes(args.batch_size, args.vocab_size);
    err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        layout.cub_temp, temp_bytes, layout.keys_in, layout.keys_out, layout.indices_in,
        layout.indices_out, total_items, args.batch_size, layout.offsets,
        layout.offsets + 1, 0, static_cast<int>(sizeof(float) * 8), stream);
    if (err != cudaSuccess) {
        return err;
    }

    sample_kernel(layout.keys_out, layout.indices_out, stream);
    return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchGreedySampler(const SamplerArgs& args, cudaStream_t stream) {
    if (args.logits == nullptr || args.token_ids == nullptr || args.batch_size <= 0 ||
        args.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(static_cast<unsigned int>(args.batch_size));
    const dim3 block(kThreadsPerBlock);
    // CUDA 的 last-error 是粘性的：先清掉入口处可能残留的旧错误，
    // 后面 cudaGetLastError() 的结果才只反映本次 launch。
    (void)cudaGetLastError();
    if (args.is_half) {
        GreedyKernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(args.logits), args.token_ids, args.vocab_size);
    } else {
        GreedyKernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(args.logits), args.token_ids, args.vocab_size);
    }
    return cudaGetLastError();
}

size_t TopKSamplerWorkspaceBytes(int32_t batch_size, int32_t vocab_size) {
    if (batch_size <= 0 || vocab_size <= 0) {
        return 0;
    }
    return SortWorkspaceBytes(batch_size, vocab_size);
}

cudaError_t LaunchTopKSampler(const TopKSamplerArgs& args, cudaStream_t stream,
                              void* workspace, size_t workspace_bytes) {
    if (args.logits == nullptr || args.token_ids == nullptr || args.top_k == nullptr ||
        args.batch_size <= 0 || args.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    const int32_t blocks =
        (args.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return SortThenSample(
        args, stream, workspace, workspace_bytes,
        [&](const float* sorted_logits, const int32_t* sorted_indices, cudaStream_t s) {
            TopKSampleKernel<<<blocks, kThreadsPerBlock, 0, s>>>(
                sorted_logits, sorted_indices, args.top_k, args.token_ids,
                args.batch_size, args.vocab_size, args.seed, args.offset);
        });
}

size_t TopPSamplerWorkspaceBytes(int32_t batch_size, int32_t vocab_size) {
    if (batch_size <= 0 || vocab_size <= 0) {
        return 0;
    }
    return SortWorkspaceBytes(batch_size, vocab_size);
}

cudaError_t LaunchTopPSampler(const TopPSamplerArgs& args, cudaStream_t stream,
                              void* workspace, size_t workspace_bytes) {
    if (args.logits == nullptr || args.token_ids == nullptr || args.top_p == nullptr ||
        args.batch_size <= 0 || args.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    const int32_t blocks =
        (args.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return SortThenSample(
        args, stream, workspace, workspace_bytes,
        [&](const float* sorted_logits, const int32_t* sorted_indices, cudaStream_t s) {
            TopPSampleKernel<<<blocks, kThreadsPerBlock, 0, s>>>(
                sorted_logits, sorted_indices, args.top_p, args.token_ids,
                args.batch_size, args.vocab_size, args.seed, args.offset);
        });
}

}  // namespace mini_trt_llm
