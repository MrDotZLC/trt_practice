// "跨实现 argmax 比较"判据的 host 用例（方案 B，TROUBLESHOOTING.md #34）。
//
// 为什么单独一处：判据本身（哪些行算可判、翻转怎么归因）是纯 host 逻辑，
// 必须能在 CI / 沙箱里被裁决——否则它只能在真机上"看起来对"。
// 本文件锁三件事：① 语义（可判行 / 不可判行的划分）；② 边界（严格 `>`）；
// ③ #34 那次事故的最小复现（真实数字：余量 1.53e-05 vs 两侧差 1.14e-04）。

#include "gpt2_test_support.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::ArgmaxAgreement;
using test_support::CompareArgmaxByDecidability;

constexpr int32_t kVocab = 4;

// 造一行：native 的 top1 在 class 0、top2 在 class 1，余量为 margin，其余类压到 -100。
std::vector<float> NativeRow(float margin) {
    return {0.0f, -margin, -100.0f, -100.0f};
}

}  // namespace

TEST(ArgmaxCriterionTest, DecidableRowAgreeingIsClean) {
    // margin = 1.0，两侧差 = 0.05 → 2d = 0.1 < margin → 可判；两边都选 class 0。
    const std::vector<float> native = NativeRow(1.0f);
    const std::vector<float> onnx = {0.05f, -0.95f, -100.0f, -100.0f};
    const ArgmaxAgreement result =
        CompareArgmaxByDecidability(onnx.data(), native.data(), 1, kVocab);
    EXPECT_EQ(result.undecidable_rows, 0);
    EXPECT_EQ(result.violations, 0);
}

TEST(ArgmaxCriterionTest, NearTieFlipIsUndecidableNotViolation) {
    // margin = 1e-4，两侧差 = 0.2 → 2d = 0.4 > margin → 不可判；ONNX 翻到 class 1。
    // **这正是 #34 的形态**：并列低于两侧差异时，翻转不该再被算成缺陷。
    const std::vector<float> native = NativeRow(1e-4f);
    const std::vector<float> onnx = {-0.1f, 0.1f, -100.0f, -100.0f};
    const ArgmaxAgreement result =
        CompareArgmaxByDecidability(onnx.data(), native.data(), 1, kVocab);
    EXPECT_EQ(result.undecidable_rows, 1);
    EXPECT_EQ(result.violations, 0) << "不可判行的翻转被误判成缺陷";
    ASSERT_EQ(result.undecidable_row_indices.size(), 1u);
    EXPECT_EQ(result.undecidable_row_indices[0], 0);
}

TEST(ArgmaxCriterionTest, NearTieAgreeingStillCountsAsUndecidable) {
    // 同样不可判，但两边恰好同侧：**仍要计数**——否则"不可判行数"会在运气好的运行里凭空变小，
    // 上界这条护栏就失去意义。
    const std::vector<float> native = NativeRow(1e-4f);
    const std::vector<float> onnx = {0.05f, -0.95f, -100.0f, -100.0f};
    const ArgmaxAgreement result =
        CompareArgmaxByDecidability(onnx.data(), native.data(), 1, kVocab);
    EXPECT_EQ(result.undecidable_rows, 1);
    EXPECT_EQ(result.violations, 0);
}

TEST(ArgmaxCriterionTest, DecidabilityBoundaryUsesStrictComparison) {
    // 边界：margin 恰好 = 2d（2×0.5 = 1.0）→ 判为**不可判**（判据用严格 `>`）。
    // 理由：m = 2d 时仍存在恰好把 argmax 翻过去的扰动（δ_ia − δ_ib = 2d = m），
    // 所以"可判"必须要求严格大于。
    const std::vector<float> native = NativeRow(1.0f);
    const std::vector<float> onnx = {-0.5f, -1.5f, -100.0f, -100.0f};  // 两侧差 0.5
    const ArgmaxAgreement result =
        CompareArgmaxByDecidability(onnx.data(), native.data(), 1, kVocab);
    EXPECT_EQ(result.undecidable_rows, 1) << "m = 2d 应当是边界外侧（不可判）";
    EXPECT_EQ(result.violations, 0);
}

TEST(ArgmaxCriterionTest, CountsAndIndicesAcrossMixedRows) {
    // 3 行混合：第 0 行可判且一致、第 1 行不可判且翻转、第 2 行不可判且一致。
    const std::vector<float> native = {
        0.0f, -1.0f, -100.0f, -100.0f,     // 行 0：margin 1.0
        0.0f, -1e-4f, -100.0f, -100.0f,    // 行 1：margin 1e-4（不可判）
        0.0f, -1e-4f, -100.0f, -100.0f,    // 行 2：margin 1e-4（不可判）
    };
    const std::vector<float> onnx = {
        0.05f, -0.95f, -100.0f, -100.0f,   // 行 0：一致、两侧差 0.05 → 可判
        -0.1f, 0.1f, -100.0f, -100.0f,     // 行 1：翻转、两侧差 0.2 → 不可判
        0.05f, -0.95f, -100.0f, -100.0f,   // 行 2：一致、两侧差 0.05 → 不可判（margin 太小）
    };
    const ArgmaxAgreement result =
        CompareArgmaxByDecidability(onnx.data(), native.data(), 3, kVocab);
    EXPECT_EQ(result.rows, 3);
    EXPECT_EQ(result.undecidable_rows, 2);
    EXPECT_EQ(result.violations, 0);
    ASSERT_EQ(result.undecidable_row_indices.size(), 2u);
    EXPECT_EQ(result.undecidable_row_indices[0], 1);
    EXPECT_EQ(result.undecidable_row_indices[1], 2);
}

// #34 事故的最小复现：用真机实测的那组数字，确认它被判为"不可判"而不是"缺陷"。
// 数字出处：TROUBLESHOOTING.md #34.6（native 余量 1.52588e-05；两侧最大差 1.14441e-04）。
TEST(ArgmaxCriterionTest, IncidentRow118IsClassifiedUndecidable) {
    constexpr float kMargin = 1.52588e-05f;         // native top1 − top2（实测）
    constexpr float kInterEngineDiff = 1.14441e-04f;  // 该行两侧最大逐元素差（实测）
    const std::vector<float> native = {0.0f, -kMargin, -100.0f, -100.0f};
    // ONNX 侧：把差距反号（模拟翻到 class 1），并让逐元素最大差等于实测值。
    const std::vector<float> onnx = {-kInterEngineDiff, -kMargin + kInterEngineDiff, -100.0f, -100.0f};
    const ArgmaxAgreement result =
        CompareArgmaxByDecidability(onnx.data(), native.data(), 1, kVocab);
    EXPECT_EQ(result.undecidable_rows, 1);
    EXPECT_EQ(result.violations, 0)
        << "若这条报红，说明判据把 #34 那种并列当成缺陷了（方案 B 的核心行为被破坏）";
}

}  // namespace mini_trt_llm
