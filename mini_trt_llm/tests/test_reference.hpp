#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// 确定性伪随机输入：用 sin 生成，避免 rand() 带来的跨环境差异，
// 否则 CPU 参考实现与 GPU kernel 之间无法做可复现的对比。
inline float DeterministicValue(int64_t index) {
    return std::sin(0.37f * static_cast<float>(index)) * 1.5f;
}

// RMSNorm 的 CPU 参考实现。用 double 累加平方和，避免参考侧自身的精度误差
// 掩盖被测 kernel 的误差。
inline void CpuRmsNorm(const std::vector<float>& input, const std::vector<float>& weight,
                       int64_t rows, int32_t hidden_size, float eps,
                       std::vector<float>* output) {
    output->assign(input.size(), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(hidden_size);
        double sum_sq = 0.0;
        for (int32_t i = 0; i < hidden_size; ++i) {
            const double v = input[base + i];
            sum_sq += v * v;
        }
        const float inv_rms =
            static_cast<float>(1.0 / std::sqrt(sum_sq / hidden_size + eps));
        for (int32_t i = 0; i < hidden_size; ++i) {
            (*output)[base + i] = input[base + i] * inv_rms * weight[i];
        }
    }
}

// 精度判定沿用 Phase 1 已确认标准：相对误差阈值 + 小值时的绝对误差 Guardrail。
// 参考值接近 0 时相对误差会被放大，必须改判绝对误差，否则会误报。
inline bool WithinTolerance(float reference, float actual, float rel_tol, float abs_tol) {
    const float diff = std::fabs(reference - actual);
    if (std::fabs(reference) < 1e-4f) {
        return diff < abs_tol;
    }
    return diff / std::fabs(reference) < rel_tol;
}

}  // namespace test_support
}  // namespace mini_trt_llm
