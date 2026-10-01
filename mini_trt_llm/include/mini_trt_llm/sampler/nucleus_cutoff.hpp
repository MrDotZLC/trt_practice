#pragma once

// Top-P 的"分块求和 → 定位 cutoff / 采样点"这一段纯逻辑。
//
// 为什么单独抽出来：内核本身在沙箱里跑不了（无 GPU，见 PROGRESS.md §5.10），而
// "第一个累计 >= 阈值"的前缀定位是整段代码里最容易出 off-by-one 的地方。做成
// __host__ __device__ 的模板后，**同一份实现**既给 kernel 用、也能被 host 用例直接裁决
// （tests/test_sampler.cpp 的 NucleusCutoffTest.*）——避免"参考实现与被测实现各写一份、
// 然后各自漂移"（PROGRESS.md §2.13）。
//
// 只用到 cuda_runtime.h 提供的 __host__/__device__ 宏：host 编译器（g++）下这两个宏为空，
// 因此本头文件可以被 .cpp 测试直接包含。

#include <cuda_runtime.h>

#include <cstdint>

namespace mini_trt_llm {

// 在"分段和"数组上找第一个满足"累计 >= threshold"的段，返回段下标（-1 = 都不够），
// 并通过 `out_running` 给出**进入该段之前**的累计和（返回 -1 时为全数组的累计和）。
//
// 为什么抽出来：内核要按"块 → 子块 → 元素"三级逼近交叉点，而每一级的判定规则完全一样
// （同序串行累加 + 包含等号的比较）。规则只留一份，才不会出现"上一级和下一级口径不一致"
// 这种最难查的偏差（`PROGRESS.md` §2.13）。
__host__ __device__ inline int32_t FindCrossingSegment(const float* segment_sums, int32_t count,
                                                      float threshold, float* out_running) {
    float running = 0.0f;
    for (int32_t c = 0; c < count; ++c) {
        const float next = running + segment_sums[c];
        if (next >= threshold) {
            *out_running = running;
            return c;
        }
        running = next;
    }
    *out_running = running;
    return -1;
}

// 在区间 [begin, end) 内逐元素定位交叉点：从 `running_before` 起按序累加 `exp_at(i)`，
// 返回第一个使累计 >= threshold 的前缀长度（i+1），并通过 `out_prefix_sum` 给出该处的累计和。
// 区间内都没跨过时返回 `end`（并把区间末的累计和写回）——调用方据此走退化路径（见调用点注释）。
template <typename ExpAt>
__host__ __device__ inline int32_t ScanSegmentForCrossing(ExpAt exp_at, int32_t begin, int32_t end,
                                                          float running_before, float threshold,
                                                          float* out_prefix_sum) {
    float running = running_before;
    for (int32_t i = begin; i < end; ++i) {
        running += exp_at(i);
        if (running >= threshold) {
            *out_prefix_sum = running;
            return i + 1;
        }
    }
    *out_prefix_sum = running;
    return end;
}

// 三级定位：**块和 → 子块和 → 元素**，每级都是同一个规则（`FindCrossingSegment` / `ScanSegmentForCrossing`）。
//
// 为什么要有第三级：内核的粗分块让每个线程负责一大段（128000 词表下 500 个元素），两级实现里
// 最后那次"块内逐元素重扫"只能由一个线程串行做——那是**延迟受限**的依赖读链（`TROUBLESHOOTING.md` + TS-035）。
// 把每段再切成 `sub_chunks` 个子块（子块和同样按块内串行口径求出），最后的重扫就从 ≤500 个元素
// 降到 ≤ ceil(chunk_size / sub_chunks) 个，而且这两级定位全在共享内存里做。
//
// `sub_sums` 为 nullptr 时退化为两级（只用块和 + 元素重扫），这正是 `FindFirstPrefixCrossing` 的老语义。
// `sub_sums` 的布局是 [chunk][sub]（行主序）：第 c 段的第 s 个子块在 `sub_sums[c * sub_chunks + s]`。
template <typename ExpAt>
__host__ __device__ inline int32_t FindCrossingByLevels(const float* chunk_sums, int32_t chunk_count,
                                                       const float* sub_sums, int32_t sub_chunks,
                                                       int32_t chunk_size, int32_t sub_size,
                                                       int32_t size, float threshold,
                                                       ExpAt exp_at, float* out_prefix_sum) {
    // 第一级：定位交叉发生在哪一块，同时攒出进入该块之前的累计和。
    float chunk_running = 0.0f;
    const int32_t crossing_chunk =
        FindCrossingSegment(chunk_sums, chunk_count, threshold, &chunk_running);
    if (crossing_chunk < 0) {
        *out_prefix_sum = chunk_running;
        return size;  // 整行都不够（threshold > 全行和）
    }

    int32_t begin = crossing_chunk * chunk_size;
    int32_t end = begin + chunk_size < size ? begin + chunk_size : size;
    float running_before = chunk_running;

    // 第二级（可选）：在该块的子块和上再逼近一次。命中则把重扫范围收窄到一个子块。
    if (sub_sums != nullptr && sub_chunks > 0) {
        float sub_running = 0.0f;
        const float* my_sub_sums = sub_sums + static_cast<size_t>(crossing_chunk) * sub_chunks;
        const int32_t crossing_sub =
            FindCrossingSegment(my_sub_sums, sub_chunks, threshold, &sub_running);
        if (crossing_sub >= 0) {
            begin += crossing_sub * sub_size;
            const int32_t sub_end = begin + sub_size;
            end = sub_end < end ? sub_end : end;
            running_before = chunk_running + sub_running;
        }
        // crossing_sub < 0：块级命中、子块级却没跨过（两级加法结合序不同，只在 1 ulp 边界发生）。
        // 此时**必须**退化为"整块重扫"，且 running_before 保持 chunk_running——
        // 加上 sub_running 等于把整块的和算两遍，会让阈值提前命中、cutoff 偏小。
    }

    // 第三级：在收窄后的区间（最多一个子块）里逐元素定位精确下标，同时给出交叉点的累计和。
    return ScanSegmentForCrossing(exp_at, begin, end, running_before, threshold, out_prefix_sum);
}

// 两级版（块和 → 元素）：返回第一个满足 `Σ_{i < cutoff} exp_i >= threshold` 的 cutoff
// （即前缀长度），并把该处的前缀和写回 `out_prefix_sum`。
//
// 参数：
//   chunk_sums     [chunk_count]，每块元素的**串行**和（块内按下标升序累加）。必须与
//                  exp_at 逐元素重算的口径同序，否则块级判定与块内扫描各持一份舍入结果。
//   size           有效元素个数；第二次调用传 cutoff，把搜索限制在已保留的前缀内。
//   chunk_size     每块元素个数，块 c 覆盖 [c*chunk_size, min((c+1)*chunk_size, size))。
//   exp_at(i)      第 i 个元素（已降序排列）的 exp 值，调用方保证非负。
//   out_prefix_sum 交叉点的前缀和（= Top-P 重新归一化用的 kept_total），不可为 nullptr。
// 返回：前缀长度 ∈ [1, size]；返回 size 表示"整行都不够"（threshold > 全行和）。
//
// 判据是**包含等号**的 `>=`，与 legacy 的 `cumulative >= p` 同口径：阈值恰好落在某个前缀
// 和上时要截到该元素——少截一个会让 nucleus 边界处的 token 永远采不到（S-9 锁的就是这类）。
//
// 本函数是上面三级版的退化调用（不传子块和），数值行为与拆分前**逐位一致**；保留它是因为
// 两级场景的调用方与 host 回归用例锁的都是这套语义。
template <typename ExpAt>
__host__ __device__ inline int32_t FindFirstPrefixCrossing(const float* chunk_sums,
                                                          int32_t chunk_count, int32_t size,
                                                          int32_t chunk_size, float threshold,
                                                          ExpAt exp_at,
                                                          float* out_prefix_sum) {
    return FindCrossingByLevels(chunk_sums, chunk_count, /*sub_sums=*/nullptr,
                                /*sub_chunks=*/0, chunk_size, /*sub_size=*/0, size, threshold,
                                exp_at, out_prefix_sum);
}

}  // namespace mini_trt_llm
