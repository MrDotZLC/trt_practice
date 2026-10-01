#pragma once

// 采样器用例的**唯一**参考实现（Top-P 的解析口径，纯 host、double 计算）。
//
// 为什么单独一份：`test_sampler.cpp` 与 `test_fp16_paths.cpp` 都要裁决"采样结果落在哪个集合 /
// 词频收敛到哪里"，参考实现分成两份就会各自漂移（PROGRESS.md §2.13 的教训）。
// 本文件不进任何产品路径，也不持有设备状态，因此沙箱内可直接跑。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// 词表顺序：按 (值降序, 下标升序)。与实现依赖的 CUB **稳定**分段排序同序——
// 并列时必须取小下标（开发计划 D3 的隐式契约）。
inline std::vector<int32_t> SortedDescendingIndices(const std::vector<float>& row) {
    std::vector<int32_t> indices(row.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int32_t>(i);
    }
    std::stable_sort(indices.begin(), indices.end(),
                     [&row](int32_t a, int32_t b) { return row[a] > row[b]; });
    return indices;
}

// 解析口径的 nucleus（= 允许被采到的 token 集合）。
//
// 边界刻意多放宽一个元素：参考是 double 串行口径、实现是 float 分块口径，恰好在阈值上的
// 元素可能差一格——那是 `future_iterations_test_plan.md` §9.4 登记的**允许差异**。
// 放宽的是参考的边界，不是断言：集合成员关系本身不含阈值（所以不存在"阈值出处"问题）。
inline std::vector<int32_t> AnalyticNucleus(const std::vector<float>& row, float p) {
    const std::vector<int32_t> order = SortedDescendingIndices(row);
    if (order.empty()) {
        return {};
    }
    // 与 kernel 同样的 p 归一化规则：NaN / 非正 → 不截断，>1 → 1
    if (!(p > 0.0f)) {
        p = 1.0f;
    }
    p = std::min(p, 1.0f);

    const double max_value = row[order[0]];
    double total = 0.0;
    for (int32_t index : order) {
        total += std::exp(static_cast<double>(row[index]) - max_value);
    }

    size_t cutoff = order.size();
    double cumulative = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        cumulative += std::exp(static_cast<double>(row[order[i]]) - max_value) / total;
        if (cumulative >= static_cast<double>(p)) {
            cutoff = i + 1;
            break;
        }
    }
    cutoff = std::min(cutoff + 1, order.size());
    return std::vector<int32_t>(order.begin(), order.begin() + static_cast<ptrdiff_t>(cutoff));
}

// 解析口径的"截断 + 前缀内重新归一化"分布，用于分布级判据（3σ）。
// p 的归一化规则与 kernel 一致（见上）。
inline std::vector<double> TruncatedSoftmaxProbabilities(const std::vector<float>& row, float p) {
    const std::vector<int32_t> order = SortedDescendingIndices(row);
    std::vector<double> probabilities(row.size(), 0.0);
    if (order.empty()) {
        return probabilities;
    }
    if (!(p > 0.0f)) {
        p = 1.0f;
    }
    p = std::min(p, 1.0f);

    const double max_value = row[order[0]];
    double total = 0.0;
    for (int32_t index : order) {
        total += std::exp(static_cast<double>(row[index]) - max_value);
    }

    size_t cutoff = order.size();
    double cumulative = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        cumulative += std::exp(static_cast<double>(row[order[i]]) - max_value) / total;
        if (cumulative >= static_cast<double>(p)) {
            cutoff = i + 1;
            break;
        }
    }

    double kept_total = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        kept_total += std::exp(static_cast<double>(row[order[i]]) - max_value);
    }
    for (size_t i = 0; i < cutoff; ++i) {
        probabilities[order[i]] =
            std::exp(static_cast<double>(row[order[i]]) - max_value) / kept_total;
    }
    return probabilities;
}

// 解析口径的"top-K 集合"，**并列安全**：返回所有满足 `value >= 第 k 大值` 的 token。
//
// 为什么不能用"排序后的前 k 个"：**数值并列时"前 k 个"不是良定义的**——它取决于排序算法
// 在并列组里挑谁。实测（`TROUBLESHOOTING.md` + TS-036）：128000 词表的 sin 造数据上，第 64 名有
// **4 个 token 精确并列**（全行 1622 对相邻并列），于是 `std::partial_sort` 取前 k 个时
// 可能把合法的那一个排掉，判据就会在**实现完全正确**时报红。
// 取值口径的判据与 tie-break 无关：只要 kernel 在"值 ≥ 第 k 大值"的元素里选，它就没选错。
//
// k >= 词表大小时返回全词表；k <= 0 返回空。
inline std::vector<int32_t> TopKSetByValue(const std::vector<float>& row, int32_t k) {
    std::vector<int32_t> allowed;
    if (row.empty() || k <= 0) {
        return allowed;
    }
    if (static_cast<size_t>(k) >= row.size()) {
        allowed.resize(row.size());
        for (size_t i = 0; i < allowed.size(); ++i) {
            allowed[i] = static_cast<int32_t>(i);
        }
        return allowed;
    }
    // 第 k 大值 = 降序 0-based 第 (k-1) 个。用 nth_element 只为拿这个**阈值**，
    // 不依赖它给出唯一的下标划分（并列组会跨过 nth 位置，这正是不用下标集合的原因）。
    std::vector<float> values = row;
    const auto nth = values.begin() + static_cast<ptrdiff_t>(k - 1);
    std::nth_element(values.begin(), nth, values.end(), std::greater<float>());
    const float kth_largest = *nth;
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i] >= kth_largest) {
            allowed.push_back(static_cast<int32_t>(i));
        }
    }
    return allowed;
}

// 解析口径的 Top-K 分布：softmax 限制在"按 (值降序, 下标升序) 排出的前 k 个"上后重新归一化。
//
// 与 `TopKSampleKernel` 同口径——它的 `total` 只在 top-k 内累加、逆变换也只在 top-k 内走，
// 等价于"先在 top-k 内重新归一化再采样"。k 超过词表时退化为全词表 softmax（S-8 的用法）。
inline std::vector<double> TopKSoftmaxProbabilities(const std::vector<float>& row, int32_t k) {
    const std::vector<int32_t> order = SortedDescendingIndices(row);
    std::vector<double> probabilities(row.size(), 0.0);
    if (order.empty() || k <= 0) {
        return probabilities;
    }
    const size_t kept = std::min(static_cast<size_t>(k), order.size());
    const double max_value = row[order[0]];  // 降序 → 第 0 个就是稳定项（与 kernel 同）
    double kept_total = 0.0;
    for (size_t i = 0; i < kept; ++i) {
        kept_total += std::exp(static_cast<double>(row[order[i]]) - max_value);
    }
    for (size_t i = 0; i < kept; ++i) {
        probabilities[order[i]] = std::exp(static_cast<double>(row[order[i]]) - max_value) /
                                  kept_total;
    }
    return probabilities;
}

}  // namespace test_support
}  // namespace mini_trt_llm
