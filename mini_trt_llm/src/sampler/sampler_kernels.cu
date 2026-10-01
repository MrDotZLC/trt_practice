#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/sampler/nucleus_cutoff.hpp"
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
// 快速路径：一行一个 warp。见 sampler_common.hpp 里 kTopKFastMaxK 的说明。
constexpr int32_t kFastTopKThreads = 32;
// Top-P 并行路径：一行一个 block。线程数就是"分块数"——线程越多每段越短（行内并行度更高），
// 但 thread 0 要串行合并的分块和也越多，256 是两者的折中，与 legacy 的 block 尺寸一致。
constexpr int32_t kTopPThreadsPerRow = 256;
// 每个线程把自己那段再切成多少个子块（P9_2-5b）。收尾段的串行重扫长度 = ceil(每段元素数 / 本值)：
// 50257 词表下 ≤13 个元素、128000 下 ≤32 个（改动前分别是 ≤197 / ≤500，且只有 thread 0 在做）。
// 取 16 是"重扫长度"与"共享内存"的折中：子块和占 256 × 16 × 4B = 16 KB，仍远低于 48 KB 上限。
constexpr int32_t kTopPSubChunks = 16;

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

// 把 (value, index) 插进"降序、并列取小下标"的有序数组（长度上限 k）。
// 只在它比当前第 k 名更好时才做搬移，因此扫描一趟的均摊成本很低。
__device__ __forceinline__ void InsertCandidate(float value, int32_t index, int32_t k,
                                                float* values, int32_t* indices,
                                                int32_t* count) {
    if (*count == k && !IsBetter(value, index, values[k - 1], indices[k - 1])) {
        return;
    }
    int32_t position = (*count < k) ? *count : k - 1;
    while (position > 0 && IsBetter(value, index, values[position - 1], indices[position - 1])) {
        values[position] = values[position - 1];
        indices[position] = indices[position - 1];
        --position;
    }
    values[position] = value;
    indices[position] = index;
    if (*count < k) {
        ++(*count);
    }
}

// Top-K 快速路径：一行一个 warp。
//
// ① 每线程在自己的 strided 切片上维护**线程本地的有序 top-k**（`InsertCandidate`）；
// ② k 轮 warp 归并：每轮在 32 个"队首"里取最优（值大者优先、并列取小下标）→ 得到全行 top-k；
// ③ 采样数学与 `TopKSampleKernel` **逐字一致**（同样的 max 稳定项、同样的 `Uniform01(seed, offset, row)`、
//    同样的逆变换 CDF 与 `>=` 比较）→ 同一 seed 下与旧路径逐 token 相同。
//
// 正确性依据：某元素若属于全局 top-k，则它必然属于"它所在切片"的本地 top-k，因此 32 份本地 top-k
// 的并集一定包含全局 top-k（k ≤ 每线程本地容量时成立——这正是 `kTopKFastMaxK` 的来源）。
template <typename T>
__global__ void FastTopKSampleKernel(const T* __restrict__ logits,
                                     const int32_t* __restrict__ top_k,
                                     int32_t* __restrict__ token_ids, int32_t vocab_size,
                                     uint64_t seed, uint64_t offset) {
    const int32_t row = blockIdx.x;
    const int32_t lane = threadIdx.x;

    int32_t k = top_k[row];
    if (k < 1) k = 1;
    if (k > vocab_size) k = vocab_size;
    // 契约被破坏时不静默给错答案：整组 lane 走同一条（uniform）分支，避免 warp 内分叉。
    const bool violates_contract = k > kTopKFastMaxK;
    if (violates_contract) {
        if (lane == 0) token_ids[row] = -1;
        return;
    }

    // 每线程一份本地候选（放在 shared 而不是寄存器里：k=64 时 64×8B 线程本地会吃掉太多寄存器）。
    __shared__ float s_values[kFastTopKThreads * kTopKFastMaxK];
    __shared__ int32_t s_indices[kFastTopKThreads * kTopKFastMaxK];
    __shared__ int32_t s_count[kFastTopKThreads];
    __shared__ int32_t s_head[kFastTopKThreads];
    __shared__ float s_picked_value[kTopKFastMaxK];
    __shared__ int32_t s_picked_index[kTopKFastMaxK];

    float* my_values = s_values + lane * kTopKFastMaxK;
    int32_t* my_indices = s_indices + lane * kTopKFastMaxK;
    int32_t count = 0;

    const T* row_logits = logits + static_cast<size_t>(row) * vocab_size;
    for (int32_t v = lane; v < vocab_size; v += kFastTopKThreads) {
        InsertCandidate(ToFloat(row_logits[v]), v, k, my_values, my_indices, &count);
    }
    s_count[lane] = count;
    s_head[lane] = 0;
    __syncthreads();

    for (int32_t pick = 0; pick < k; ++pick) {
        float best_value = -CUDART_INF_F;
        int32_t best_index = 0x7FFFFFFF;
        int32_t best_owner = -1;
        const int32_t head = s_head[lane];
        if (head < s_count[lane]) {
            best_value = my_values[head];
            best_index = my_indices[head];
            best_owner = lane;
        }
#pragma unroll
        for (int32_t delta = kFastTopKThreads / 2; delta > 0; delta >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffffu, best_value, delta);
            const int32_t other_index = __shfl_down_sync(0xffffffffu, best_index, delta);
            const int32_t other_owner = __shfl_down_sync(0xffffffffu, best_owner, delta);
            if (other_owner >= 0 &&
                (best_owner < 0 || IsBetter(other_value, other_index, best_value, best_index))) {
                best_value = other_value;
                best_index = other_index;
                best_owner = other_owner;
            }
        }
        // 广播胜者（lane 0 持有归约结果）
        best_value = __shfl_sync(0xffffffffu, best_value, 0);
        best_index = __shfl_sync(0xffffffffu, best_index, 0);
        best_owner = __shfl_sync(0xffffffffu, best_owner, 0);
        if (lane == 0) {
            s_picked_value[pick] = best_value;
            s_picked_index[pick] = best_index;
        }
        if (lane == best_owner) {
            s_head[lane] = head + 1;
        }
        __syncwarp();
    }

    if (lane == 0) {
        // 与 TopKSampleKernel 完全同构：稳定项取 top-1、`__expf`、逆变换 CDF。
        const float max_value = s_picked_value[0];
        float total = 0.0f;
        for (int32_t i = 0; i < k; ++i) {
            total += __expf(s_picked_value[i] - max_value);
        }
        const float target = Uniform01(seed, offset, static_cast<uint32_t>(row)) * total;
        float cumulative = 0.0f;
        int32_t chosen = s_picked_index[0];
        for (int32_t i = 0; i < k; ++i) {
            cumulative += __expf(s_picked_value[i] - max_value);
            if (cumulative >= target) {
                chosen = s_picked_index[i];
                break;
            }
        }
        token_ids[row] = chosen;
    }
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

// Top-P 的 legacy 采样 kernel：整行由**一个线程**处理（4 趟 O(vocab) 串行扫描，每趟每元素
// 一次 `__expf`）。性能实测见 future_iterations_development_plan.md §10.5——它是 top-p 端到端
// 耗时的大头，已被下面的并行版本取代；**保留它是为了对照与回归**（`LaunchTopPSamplerLegacy`），
// 不再接生产路径。
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

// Top-P 的并行采样（P9_2-5 + P9_2-5b）：一行一个 block。
//
// 分工：① 每线程把自己那段**连续**元素（块）串行求和，**并顺带切出 16 个子块和**，一起入 shared；
// ② thread 0 在"块和 → 子块和 → 元素"三级上用 `FindCrossingByLevels` 定位 cutoff 与采样点。
//
// 为什么这样切：legacy 的瓶颈是"整行只由一个线程扫 4 趟、每趟每元素一次 `__expf`"。
// 摊到 256 个线程后（P9_2-5），每行只剩"常数个块和的合并 + 一次块内重扫"留给单线程；
// 而那次重扫最多要 197（V=50257）/ 500（V=128000）个**相互依赖**的 global load，
// 实测它（两次重扫之和）就是收尾段 67~129 µs 的来源——**延迟受限，不是吞吐受限**
// （`TROUBLESHOOTING.md` + TS-035）。P9_2-5b 在块和之下加一级子块和，把重扫长度按
// kTopPSubChunks 收窄到 ≤13 / ≤32 个元素，且两级定位都只在 shared 上做。
//
// 连续分块（而非 strided）是刻意的：它让"块内累加"本身就是一段连续前缀，三级定位因此退化成
// "块和 + 子块和 + 一次短重扫"，不需要 block scan；代价是 warp 内访存不连续，但每线程顺序走
// 自己那段，相邻访问命中同一 cache line。
//
// 数值口径与 legacy 的唯一差异：legacy 逐元素累加 `exp/total` 再与 `p` 比，这里累加 `exp`
// 再与 `p * Σexp` 比（先除后加 vs 先加后除）。随机数消费（同一个 `Uniform01(seed, offset, row)`）、
// `>=` 比较、稳定项取 top-1、前缀内重新归一化都保持一致；差异登记在
// future_iterations_test_plan.md §9.4。
//
// **模板参数 `kSubChunked`**：`true` = 生产版本（块和 + 子块和 + 元素，P9_2-5b）；
// `false` = P9_2-5b 之前的两级版本（块和 + 元素），**只用于 A/B 对照**（`LaunchTopPSamplerTwoLevel`）。
// 为什么要把它编译进同一个二进制：P9_2-5b 的效果此前一直判不了——分段口径没有判别力（`docs/TROUBLESHOOTING.md` + TS-037）、
// 配对口径又跨 session/跨协议（`docs/TROUBLESHOOTING.md` + TS-038）。同二进制内两版**同轮配对**，噪声对二者同向，差值才是干净答案。
template <bool kSubChunked>
__global__ void TopPParallelSampleKernel(const float* __restrict__ sorted_logits,
                                         const int32_t* __restrict__ sorted_indices,
                                         const float* __restrict__ top_p,
                                         int32_t* __restrict__ token_ids, int32_t vocab_size,
                                         uint64_t seed, uint64_t offset) {
    const int32_t row = blockIdx.x;
    const int32_t tid = threadIdx.x;

    const float* row_logits = sorted_logits + static_cast<size_t>(row) * vocab_size;
    const int32_t* row_indices = sorted_indices + static_cast<size_t>(row) * vocab_size;

    float p = top_p[row];
    if (!(p > 0.0f)) {
        p = 1.0f;  // NaN / 非正值一律退化为不截断，避免 kernel 内出现未定义行为
    }
    if (p > 1.0f) {
        p = 1.0f;
    }

    // 两级版不需要子块和：用 1 个占位元素替代整块 shared，避免为对照版本付出 16 KB 的代价
    // （否则"对照"就不再是改动前的形态了）。
    constexpr int32_t kSubSumStorage = kSubChunked ? kTopPThreadsPerRow * kTopPSubChunks : 1;
    __shared__ float s_chunk_sum[kTopPThreadsPerRow];
    __shared__ float s_sub_sum[kSubSumStorage];

    // 已按降序排好，第 0 个就是全行最大值，可直接作为 softmax 的稳定项（与 legacy 同）
    const float max_value = row_logits[0];
    const int32_t chunk_size = (vocab_size + kTopPThreadsPerRow - 1) / kTopPThreadsPerRow;
    const int32_t sub_size = (chunk_size + kTopPSubChunks - 1) / kTopPSubChunks;
    const int32_t begin = tid * chunk_size;
    const int32_t end = begin + chunk_size < vocab_size ? begin + chunk_size : vocab_size;

    // 一趟同时产出块和与子块和。子块和必须与块和同口径（都是块内按下标升序的串行累加），
    // 否则第三级重扫看到的累计和与第二级判定用的和不是同一个数——1 ulp 边界上会提前/滞后命中。
    float chunk_sum = 0.0f;
    if constexpr (kSubChunked) {
        int32_t cursor = begin;
        for (int32_t sub = 0; sub < kTopPSubChunks; ++sub) {
            const int32_t sub_end = cursor + sub_size < end ? cursor + sub_size : end;
            float sub_sum = 0.0f;
            for (int32_t i = cursor; i < sub_end; ++i) {
                sub_sum += __expf(row_logits[i] - max_value);
            }
            s_sub_sum[tid * kTopPSubChunks + sub] = sub_sum;
            chunk_sum += sub_sum;
            cursor = sub_end;
        }
    } else {
        for (int32_t i = begin; i < end; ++i) {
            chunk_sum += __expf(row_logits[i] - max_value);
        }
    }
    s_chunk_sum[tid] = chunk_sum;
    __syncthreads();

    // 单线程收尾：行内唯一的串行段，量级是"分块数 + 一个子块"（两级定位都在 shared 上做），
    // 不是 vocab、也不是一整块。
    if (tid != 0) {
        return;
    }

    const auto exp_at = [&](int32_t index) { return __expf(row_logits[index] - max_value); };
    const int32_t chunk_count = (vocab_size + chunk_size - 1) / chunk_size;

    // total 用"块和按块序串行相加"求得，而不是树形归约：阈值 `p * total` 与下面所有累计和
    // 必须走同一口径，否则两者各带一份舍入，边界处会随机差一格。
    float total = 0.0f;
    for (int32_t c = 0; c < chunk_count; ++c) {
        total += s_chunk_sum[c];
    }

    float kept_total = 0.0f;
    const float* sub_sums = kSubChunked ? s_sub_sum : nullptr;
    const int32_t sub_chunks = kSubChunked ? kTopPSubChunks : 0;
    const int32_t cutoff = FindCrossingByLevels(s_chunk_sum, chunk_count, sub_sums, sub_chunks,
                                                chunk_size, sub_size, vocab_size, p * total,
                                                exp_at, &kept_total);

    // 在被保留的前缀内重新归一化后采样。阈值 `u * kept_total < kept_total`，因此交叉点必然
    // 落在 [0, cutoff) 内；第二次调用把 size 传成 cutoff，是为了不越过截断边界去找交叉点。
    const float target = Uniform01(seed, offset, static_cast<uint32_t>(row)) * kept_total;
    float ignored_prefix = 0.0f;
    const int32_t chosen_prefix = FindCrossingByLevels(s_chunk_sum, chunk_count, sub_sums,
                                                       sub_chunks, chunk_size, sub_size, cutoff,
                                                       target, exp_at, &ignored_prefix);
    token_ids[row] = row_indices[chosen_prefix > 0 ? chosen_prefix - 1 : 0];
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

cudaError_t LaunchTopKSamplerFast(const TopKSamplerArgs& args, cudaStream_t stream) {
    if (args.logits == nullptr || args.token_ids == nullptr || args.top_k == nullptr ||
        args.batch_size <= 0 || args.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    // 一行一个 warp：grid 的行数 = batch，block = 32。
    const dim3 grid(static_cast<unsigned int>(args.batch_size));
    const dim3 block(kFastTopKThreads);
    // CUDA 的 last-error 是粘性的：先清掉入口处可能残留的旧错误（与其它 Launch* 同一纪律）。
    (void)cudaGetLastError();
    if (args.is_half) {
        FastTopKSampleKernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(args.logits), args.top_k, args.token_ids,
            args.vocab_size, args.seed, args.offset);
    } else {
        FastTopKSampleKernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(args.logits), args.top_k, args.token_ids,
            args.vocab_size, args.seed, args.offset);
    }
    return cudaGetLastError();
}

cudaError_t LaunchTopPSampler(const TopPSamplerArgs& args, cudaStream_t stream,
                              void* workspace, size_t workspace_bytes) {
    if (args.logits == nullptr || args.token_ids == nullptr || args.top_p == nullptr ||
        args.batch_size <= 0 || args.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    // 排序（CUB）保持不变，只把采样 kernel 换成行内并行版本；workspace 需求因此不变。
    const dim3 grid(static_cast<unsigned int>(args.batch_size));
    const dim3 block(kTopPThreadsPerRow);
    return SortThenSample(
        args, stream, workspace, workspace_bytes,
        [&](const float* sorted_logits, const int32_t* sorted_indices, cudaStream_t s) {
            TopPParallelSampleKernel</*kSubChunked=*/true><<<grid, block, 0, s>>>(
                sorted_logits, sorted_indices, args.top_p, args.token_ids, args.vocab_size,
                args.seed, args.offset);
        });
}

// P9_2-5b 的 A/B 对照：同一条排序 + **不带子块**的两级采样 kernel（即改动前的形态）。
// 只用于性能对照与回归，**不接生产路径**（生产走上面那个 `kSubChunked = true` 的实例）。
cudaError_t LaunchTopPSamplerTwoLevel(const TopPSamplerArgs& args, cudaStream_t stream,
                                      void* workspace, size_t workspace_bytes) {
    if (args.logits == nullptr || args.token_ids == nullptr || args.top_p == nullptr ||
        args.batch_size <= 0 || args.vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(static_cast<unsigned int>(args.batch_size));
    const dim3 block(kTopPThreadsPerRow);
    return SortThenSample(
        args, stream, workspace, workspace_bytes,
        [&](const float* sorted_logits, const int32_t* sorted_indices, cudaStream_t s) {
            TopPParallelSampleKernel</*kSubChunked=*/false><<<grid, block, 0, s>>>(
                sorted_logits, sorted_indices, args.top_p, args.token_ids, args.vocab_size,
                args.seed, args.offset);
        });
}

cudaError_t LaunchTopPSamplerLegacy(const TopPSamplerArgs& args, cudaStream_t stream,
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
