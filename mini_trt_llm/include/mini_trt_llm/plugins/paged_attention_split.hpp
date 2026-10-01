#pragma once

// decode 阶段 PagedAttention 的**上下文维切分（FlashDecoding 式 split-K）**纯逻辑。
//
// 为什么单独抽出来：分片边界（不整除、分片数超过位置数、`context_len == 0` 且带当前 token）
// 全是纯逻辑，而 kernel 在沙箱里跑不了（无 GPU，见 `PROGRESS.md` §5.10）。做成
// `__host__ __device__` 之后，**同一份实现**既给 kernel 用、也能被 host 用例直接裁决
// （`tests/test_paged_attention_split.cpp`），避免"参考与被测各写一份、然后各自漂移"
// （`PROGRESS.md` §2.13）。
//
// 本文件同时是 **workspace 布局的唯一实现**：`getWorkspaceSize()` 报多少字节、kernel 往哪写，
// 两边都必须走这里的函数。各写一份算法是最容易在"改了片数 / 改了 head_size"时静默错位的地方
// （开发计划 §12.3 D3）。
//
// 只用到 cuda_runtime.h 提供的 `__host__` / `__device__` 宏：host 编译器（g++）下这两个宏为空，
// 因此本头文件可以被 .cpp 测试直接包含（同 `sampler/nucleus_cutoff.hpp` 的做法）。

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mini_trt_llm {

// 分片并行度上限。
//
// 为什么是 8：改动前 GPT-2 decode 每层只有 `num_heads(12) × batch(1) = 12` 个 block，而本机
// 有 24 个 SM —— 一半闲置，且每块要串行扫完整段上下文（976 次迭代）。取 8 让每层 block 数
// 变成 12 × 8 = 96（24 SM 的 4 波），再往上加会被"每位置一次 BlockReduceSum"与归并开销吃掉
// 收益（开发计划 §12.3 D1/D2 与 §12.9 F2=A）。
constexpr int32_t kPagedAttentionMaxSplits = 8;

// 每个分片的目标位置数。
//
// 为什么是 128：让短上下文（PF-9 的 ctx≈20 档）落在 1 片、长上下文（ctx≈976）用满 8 片
// （976 / 8 = 122 位置/片）。这个值只影响"怎么切"，不影响正确性。
constexpr int32_t kPagedAttentionTargetChunk = 128;

// 一个 (batch, head) 在 `total_len` 个位置上的**有效分片数**。
//
// `override_splits > 0` 时强制使用该值（钳到上限，保证不越过 workspace 契约）——那是给 A/B
// 与边界用例用的测试开关（`SetPagedAttentionNumSplitsOverride`），生产路径不设置它。
//
// 返回 0 表示"没有位置可看"（`total_len <= 0`），调用方必须据此走兜底（输出 0），
// 不能让下游出现 `exp(-inf - -inf)` 这类 NaN（见 Merge 侧的 `l <= 0` 判据）。
__host__ __device__ inline int32_t PagedAttentionResolveSplits(int32_t total_len,
                                                              int32_t override_splits) {
    if (total_len <= 0) {
        return 0;
    }
    if (override_splits > 0) {
        return override_splits > kPagedAttentionMaxSplits ? kPagedAttentionMaxSplits
                                                         : override_splits;
    }
    const int32_t by_length =
        (total_len + kPagedAttentionTargetChunk - 1) / kPagedAttentionTargetChunk;
    return by_length > kPagedAttentionMaxSplits ? kPagedAttentionMaxSplits : by_length;
}

// 第 `split_idx` 片覆盖的逻辑位置区间 `[begin, end)`（左闭右开）。
//
// 切法：均分，余数摊给**前** `total_len % effective_splits` 片。
// **为什么把切法写死成"确定"的**：同样的输入必须切出同样的区间，否则两次运行的归并顺序不同、
// 结果不可复现（`AGENTS.md` §7 的"可复现"要求）。
//
// **空片是合法的**：`effective_splits == 0`、`split_idx` 越界、或 `total_len < effective_splits`
// 时返回 `begin == end`。调用方必须显式处理它（写哨兵 `m = -inf, l = 0, acc = 0`），
// 否则归并会读到未初始化的显存（`PROGRESS.md` §2.12 / `TROUBLESHOOTING` + TS-004）。
__host__ __device__ inline void PagedAttentionSplitRange(int32_t total_len, int32_t split_idx,
                                                        int32_t effective_splits,
                                                        int32_t* begin, int32_t* end) {
    if (total_len <= 0 || effective_splits <= 0 || split_idx < 0 ||
        split_idx >= effective_splits) {
        const int32_t clamped = total_len > 0 ? total_len : 0;
        *begin = clamped;
        *end = clamped;
        return;
    }
    const int32_t base = total_len / effective_splits;
    const int32_t remainder = total_len % effective_splits;
    const int32_t extra = split_idx < remainder ? split_idx : remainder;
    const int32_t b = split_idx * base + extra;
    *begin = b;
    *end = b + base + (split_idx < remainder ? 1 : 0);
}

// 每个分片槽位占多少个 float：`[m, l, acc[0..head_size)]`。
// m / l 是局部 max 与局部 exp 和（都按 float 存，FP16 源也如此——只读写走 FP16）。
__host__ __device__ inline int32_t PagedAttentionWorkspaceStride(int32_t head_size) {
    return 2 + head_size;
}

// 分片槽位的**唯一布局**：`split` 最外层 → `batch` → `head` → `stride`。
// 返回以 float 为单位的偏移；调用方自行乘 `sizeof(float)`。
__host__ __device__ inline size_t PagedAttentionWorkspaceSlotOffset(
    int32_t split, int32_t batch, int32_t head, int32_t batch_size, int32_t num_heads,
    int32_t head_size) {
    const size_t stride = static_cast<size_t>(PagedAttentionWorkspaceStride(head_size));
    const size_t per_split =
        static_cast<size_t>(batch_size) * static_cast<size_t>(num_heads) * stride;
    const size_t within = (static_cast<size_t>(batch) * static_cast<size_t>(num_heads) +
                           static_cast<size_t>(head)) *
                          stride;
    return static_cast<size_t>(split) * per_split + within;
}

// 分片缓冲需要的字节数（0 = 入参非法，调用方必须据此放弃 split 路径）。
//
// 契约：`getWorkspaceSize()` 报的值必须 ≥ kernel 实际用到的最大偏移 + stride——
// 这是 workspace 版的"按对方查询、不按配置假定"（`PROGRESS.md` §2.15），
// 而两边用的是同一份布局函数（本文件）。
inline size_t PagedAttentionWorkspaceBytes(int32_t batch_size, int32_t num_heads,
                                           int32_t head_size, int32_t splits) {
    if (batch_size <= 0 || num_heads <= 0 || head_size <= 0 || splits <= 0) {
        return 0;
    }
    const size_t stride = static_cast<size_t>(PagedAttentionWorkspaceStride(head_size));
    return static_cast<size_t>(batch_size) * static_cast<size_t>(num_heads) *
           static_cast<size_t>(splits) * stride * sizeof(float);
}

}  // namespace mini_trt_llm
