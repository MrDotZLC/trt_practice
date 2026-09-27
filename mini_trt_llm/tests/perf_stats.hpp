#pragma once

// 性能统计量的唯一实现（H 层：纯 host，沙箱可跑、可进 CI）。
//
// **为什么要有它**：`future_iterations_development_plan.md` §11.3 把 G6 的落地口径
// 定成"报中位数 + 分位 + 极差，同二进制 A/B 另用斜率 (T4−T1)/3"。这几个量若每个用例
// 各写一份，口径就会漂移——同一个"中位数"可能一个取平均、另一个取低中位，于是两次
// 测量"不可比"却没人看得出来（这正是 §9.2 测量事故的同一类根因，见
// `docs/TROUBLESHOOTING.md` + TS-037 / TS-038）。
//
// 本文件是这些统计量的唯一来源，且每一条都被 `tests/test_decode_perf.cpp` 的
// `PerfStatsTest.*` 用已知输入锁住语义（`PROGRESS.md` §2.13：参考实现必须唯一且自带断言）。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// 中位数：排序后取中间元素；偶数个取中间两个的算术平均。
// 空输入返回 0.0（哨兵）——调用方必须先保证非空，这条由用例锁住。
inline double Median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// 最近秩（nearest-rank）分位数：返回排序后第 ceil(q/100 · n) 个元素（1-based），
// **不做线性插值**；q ≤ 0 取最小，q ≥ 100 取最大；空输入返回 0.0（哨兵）。
//
// **为什么用最近秩而不是插值**：报告里要写的是"实测到的最坏值落在哪个样本上"。
// 插值会造出一个从未被观测到的数，那违反 `AGENTS.md` §7"每个数都要有出处"。
inline double Percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    if (q <= 0.0) {
        return values.front();
    }
    if (q >= 100.0) {
        return values.back();
    }
    const size_t n = values.size();
    size_t rank = static_cast<size_t>(std::ceil(q / 100.0 * static_cast<double>(n)));
    if (rank == 0) {
        rank = 1;
    }
    if (rank > n) {
        rank = n;
    }
    return values[rank - 1];
}

// 极差的两端；空输入返回 {0.0, 0.0}（哨兵）。
inline std::pair<double, double> MinMax(const std::vector<double>& values) {
    if (values.empty()) {
        return {0.0, 0.0};
    }
    const auto mm = std::minmax_element(values.begin(), values.end());
    return {*mm.first, *mm.second};
}

// 斜率口径（`TROUBLESHOOTING.md` #38）：窗口耗时 T(k) = fixed + k · net，
// 于是 net = (T(m) − T(1)) / (m − 1)，自动扣掉每窗口的固定开销
// （事件记录 + 同步 + 首次发射，本环境可达几十 µs，与待测信号同量级）。
//
// 入参 times_by_window[i] 对应窗口大小 i+1，即 {T(1), …, T(m)}。
// 少于 2 个窗口无法相减 → 返回 0.0（哨兵，用例锁住）。
inline double SlopePerExtraLaunch(const std::vector<double>& times_by_window) {
    if (times_by_window.size() < 2) {
        return 0.0;
    }
    const double span = static_cast<double>(times_by_window.size() - 1);
    return (times_by_window.back() - times_by_window.front()) / span;
}

}  // namespace test_support
}  // namespace mini_trt_llm
