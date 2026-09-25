#pragma once

// 数值差异诊断的**唯一来源**（跨模型共用）。
//
// 为什么单独提出来：这些口径原本在 `gpt2_test_support.hpp` 里，但 Phase 4 的 CV 用例同样要用。
// 若各写一份，"绝对差 / 相对差怎么算、分母取什么"就会各自漂移，而阈值口径漂移是 AGENTS.md §7
// 明令禁止的（见 PROGRESS.md §2.13「唯一来源」）。GPT-2 侧的头文件改为包含本文件，
// 既有用例不受影响。

#include "mini_trt_llm/utils/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// 诊断用的差异汇总。
//
// **绝对差与相对差都要看**：相对差在参考值接近 0 时会被放大（logits 里本来就有
// 接近 0 的分量），只报相对差会把"绝对差 1e-4 作用在 0.003 上"读成 3% 的误差，
// 从而把结论引向错误的方向。因此两个都算，判据用绝对差为主。
struct DiffStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;  // 以 max|reference| 为分母，避免小值放大
};

inline DiffStats ComputeDiffStats(const std::vector<float>& reference,
                                  const std::vector<float>& actual) {
    DiffStats stats;
    float scale = 0.0f;
    for (float v : reference) {
        scale = std::max(scale, std::fabs(v));
    }
    const float denominator = scale > 0.0f ? scale : 1.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float diff = std::fabs(reference[i] - actual[i]);
        stats.max_abs = std::max(stats.max_abs, diff);
        stats.max_rel = std::max(stats.max_rel, diff / denominator);
    }
    return stats;
}

// 绝对差判据（参考值的幅度已知时最可靠），相对差另附在诊断行里。
inline bool WithinAbs(float reference, float actual, float abs_tol) {
    return std::fabs(reference - actual) < abs_tol;
}

// 把某个输出张量从设备读回主机。
inline std::vector<float> ReadFloats(const void* device_ptr, size_t count) {
    std::vector<float> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device_ptr, count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace test_support
}  // namespace mini_trt_llm
