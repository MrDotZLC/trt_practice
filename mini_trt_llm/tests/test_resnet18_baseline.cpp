#include "cv_test_support.hpp"
#include "mini_trt_llm/utils/io.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::ArgmaxOfRow;
using test_support::FindBaselineDir;
using test_support::kCvChannels;
using test_support::kCvClasses;
using test_support::kCvSize;
using test_support::MakeRampInput;
using test_support::NormalizePixels;
using test_support::ReadF32File;

constexpr int32_t kBatch = 8;
const size_t kInputElements =
    static_cast<size_t>(kBatch) * kCvChannels * kCvSize * kCvSize;
const size_t kLogitsElements = static_cast<size_t>(kBatch) * kCvClasses;

// P4-1 的基线**不是**一次性临时文件：它是后续所有"对齐"结论的标尺（AGENTS.md §7）。
// 所以它自己也得有人守——下面三条放在沙箱里跑，不依赖 GPU，也不用重跑 Python。

// 基线的 ramp 输入必须与公式**逐位一致**。
//
// 存在理由：公式在 Python 脚本与 C++ 测试里各有一份，一旦漂移，"引擎与基线对不上"就会
// 伪装成实现 bug。拿 Python 落盘的产物去比 C++ 的计算，就把这个假设变成了可执行的检查。
TEST(ResNet18BaselineTest, RampInputMatchesFormulaBitExact) {
    const std::string dir = FindBaselineDir();
    if (dir.empty()) {
        GTEST_SKIP() << "缺少 models/resnet18 基线产物（先跑 scripts/ref_resnet18.py）";
    }
    const std::vector<float> stored =
        ReadF32File(dir + "/inputs/ref_ramp_b8.contract_input.f32.bin", kInputElements);
    ASSERT_EQ(stored.size(), kInputElements) << "契约输入张量大小不符";

    const std::vector<float> expected = MakeRampInput(kBatch);
    size_t mismatches = 0;
    size_t first_bad_index = 0;
    float first_bad_stored = 0.0f;
    float first_bad_expected = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (stored[i] != expected[i]) {  // 同一公式 → 要求逐位相等，不留容差
            if (mismatches == 0) {
                first_bad_index = i;
                first_bad_stored = stored[i];
                first_bad_expected = expected[i];
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "首个不符下标 " << first_bad_index << "：存储值 "
                              << first_bad_stored << " vs C++ 公式 " << first_bad_expected;
}

// 像素质基线与归一化公式一致：`normalized` 必须等于对 `contract_input` 施加
// ImageNet mean/std 的结果（C++ 侧独立算一遍）。
//
// 这条是 CVRunner 预处理用例（R0.3）的前置：先证明"Python 落盘的归一化结果 == 公式"，
// 等 CVRunner 实现出来后再证"CVRunner == 同一公式"。
TEST(ResNet18BaselineTest, PixelsNormalizationMatchesFormula) {
    const std::string dir = FindBaselineDir();
    if (dir.empty()) {
        GTEST_SKIP() << "缺少 models/resnet18 基线产物";
    }
    const std::vector<float> pixels =
        ReadF32File(dir + "/inputs/ref_pixels_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> stored_normalized =
        ReadF32File(dir + "/inputs/ref_pixels_b8.normalized.f32.bin", kInputElements);
    ASSERT_EQ(pixels.size(), kInputElements);
    ASSERT_EQ(stored_normalized.size(), kInputElements);

    const std::vector<float> expected = NormalizePixels(pixels);
    float max_abs = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(expected[i] - stored_normalized[i]));
    }
    std::cout << "[基线] 归一化公式 C++ vs Python: max_abs = " << max_abs << "\n";
    // 实测（2026-09-26）两侧**逐位相同**（max_abs = 0）。阈值仍留 1e-6 作为回归护栏：
    // 未来若有人改公式顺序（例如先除 std 再减 mean），误差会从这个量级开始出现，
    // 而它足以让对拍结论失真。出处：本条用例的实测记录 + TROUBLESHOOTING #21 的同类差异量级。
    EXPECT_LT(max_abs, 1e-6f);
    for (float v : pixels) {
        ASSERT_GE(v, 0.0f);
        ASSERT_LE(v, 255.0f);
    }
}

// 元数据必须与数据自洽：`argmax_per_sample` 要能从 logits 现算出来。
// 元数据是"这份基线哪来的"的唯一记录，它一旦失真，后面引用它的人都会被带偏。
TEST(ResNet18BaselineTest, MetaArgmaxMatchesLogits) {
    const std::string dir = FindBaselineDir();
    if (dir.empty()) {
        GTEST_SKIP() << "缺少 models/resnet18 基线产物";
    }
    for (const std::string& stem : {"ref_ramp_b8", "ref_pixels_b8"}) {
        const std::vector<float> logits =
            ReadF32File(dir + "/" + stem + ".bin", kLogitsElements);
        ASSERT_EQ(logits.size(), kLogitsElements) << stem;

        const JsonValue meta = LoadJson(dir + "/" + stem + ".meta.json");
        ASSERT_TRUE(meta.IsObject()) << stem;
        ASSERT_TRUE(meta.Has("logits")) << stem;
        const JsonValue& logits_meta = meta["logits"];
        ASSERT_TRUE(logits_meta.Has("argmax_per_sample")) << stem;
        const JsonValue& argmax = logits_meta["argmax_per_sample"];
        ASSERT_EQ(static_cast<int32_t>(argmax.Size()), kBatch) << stem;
        for (int32_t b = 0; b < kBatch; ++b) {
            EXPECT_EQ(argmax[static_cast<size_t>(b)].AsInt(), ArgmaxOfRow(logits, b))
                << stem << " 第 " << b << " 行 argmax 与 logits 不符";
        }
    }
}

}  // namespace
}  // namespace mini_trt_llm
