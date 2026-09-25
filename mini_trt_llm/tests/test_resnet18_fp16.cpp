#include "cv_test_support.hpp"
#include "diff_stats.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/cv_runner.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::ArgmaxOfRow;
using test_support::ComputeDiffStats;
using test_support::DiffStats;
using test_support::FindBaselineDir;
using test_support::FindFile;
using test_support::kCvClasses;
using test_support::ReadF32File;

constexpr int32_t kBatch = 8;
const size_t kInputElements = static_cast<size_t>(kBatch) * 3 * 224 * 224;
const size_t kLogitsElements = static_cast<size_t>(kBatch) * kCvClasses;

const char* const kOnnxFp16Engine = "/tmp/mini_trt_llm_resnet18_onnx_fp16.engine";
const char* const kNativeFp16Engine = "/tmp/mini_trt_llm_resnet18_native_fp16.engine";
const char* const kNativeFp32Engine = "/tmp/mini_trt_llm_resnet18_native_fp32.engine";

// FP16 对拍阈值：`max_abs < 0.1`（配 argmax 逐样本一致）。
//
// **出处（2026-09-26 真机实测，不是拍的）**——先量"与正确性无关的差异"，而且这里的差异
// 主要不是"实现不同"，而是 **FP16 本身的舍入**：
//   · torchvision FP32 vs FP16（同一台 GPU、同一份权重、同一输入）：
//     ramp 输入 `max_abs = 0.01814`（rel 2.00e-3）、pixels 输入 `max_abs = 0.02697`（rel 1.08e-3）
//   · 两者 argmax 均 8/8 一致
// 即：**光是把权重和计算换成 FP16，logits 就会差 0.02 上下**，与该网络的激活幅度
// （logits 约 ±8..25）相比约 1e-3 相对量级。阈值取实测上界（0.027）的约 4 倍留 TRT
// tactic 差异的余量，而**语义正确性另由 argmax 判据保证**。
//
// ⚠️ **不要**把 P4-2/P4-5 的 FP32 阈值（1e-4 / 1e-5）套到这里，也不要反着把这条放宽到
// 掩盖真问题：`PROGRESS.md` §7 明令"阈值不跨精度复用"。D4 的 `rel < 1e-3` 同理——它是给
// 单算子/同精度比较定的（实测本次纯舍入的 rel 已达 1.08e-3~2.0e-3，本来就超）。
constexpr float kFp16MaxAbs = 0.1f;

// 两条 FP16 路径之间的上界：`max_abs < 0.05`。
//
// **第一版这里写的是 1e-4，是错的**（实测被打回：原生-FP16 vs ONNX-FP16 = 0.00757）。
// 错在把 FP32 的尺子拿到 FP16 上用：要求两个**不同实现**的 FP16 引擎一致到 1e-4，
// 等于要求它们比 FP16 本身还准——纯 FP16 舍入一项实测就有 0.018~0.027（见 kFp16MaxAbs），
// 而两个 FP16 引擎各自离 FP32 是 0.031（ONNX）与 0.062（CVRunner 那条），
// 它们**互相**只差 0.0076，本来就该是这个量级。
//
// 阈值取实测值的约 6.6 倍，理由与 kFp16MaxAbs 同源：给 TRT 的 tactic 选择留余量；
// **语义正确性仍由 argmax 逐样本一致来保证**，而两条 FP16 路径也各自与 FP32 基线对过
// （R2.5a 与本文件对 native 那侧）。见 TROUBLESHOOTING #26。
constexpr float kFp16CrossPathMaxAbs = 0.05f;

std::string FindModelDir() {
    const std::string config = FindFile({"models/resnet18/config.json",
                                         "../models/resnet18/config.json",
                                         "../../models/resnet18/config.json",
                                         "../../../models/resnet18/config.json"});
    if (config.empty()) {
        return {};
    }
    return std::filesystem::path(config).parent_path().string();
}

std::string FindOnnxPath() {
    return FindFile({"0_resnet18_onnx/resnet18.onnx", "../0_resnet18_onnx/resnet18.onnx",
                     "../../0_resnet18_onnx/resnet18.onnx",
                     "../../../0_resnet18_onnx/resnet18.onnx"});
}

EngineBuilder::Config Fp16Config() {
    EngineBuilder::Config config;
    // 显式写出目标精度：`Config{}` 的默认值恰好也是 FP16，但"恰好"不是证据——
    // 第一版 ResNet18 对拍把默认值当 FP32 用，白查了一轮（TROUBLESHOOTING #21）。
    config.precision = Precision::FP16;
    return config;
}

EngineBuilder::Config Fp32Config() {
    EngineBuilder::Config config;
    config.precision = Precision::FP32;  // 三角证据里的 FP32 参照
    return config;
}

std::vector<float> RunEngine(Engine* engine, const std::vector<float>& input) {
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        return {};
    }
    const size_t input_bytes = input.size() * sizeof(float);
    DeviceBuffer d_input(input_bytes);
    DeviceBuffer d_output(kLogitsElements * sizeof(float));
    if (!d_input.Allocate(input_bytes) || !d_output.Allocate(kLogitsElements * sizeof(float))) {
        return {};
    }
    CUDA_CHECK(cudaMemcpy(d_input.data(), input.data(), input_bytes, cudaMemcpyHostToDevice));
    if (!engine->SetOptimizationProfile(0, nullptr) ||
        !engine->SetInputShape("input", nvinfer1::Dims4{kBatch, 3, 224, 224})) {
        return {};
    }
    // 按**引擎声明的** dtype 绑定输入：FP16 引擎的 I/O 未必是 FP16（弱类型网络下由 TRT 决定，
    // 实测通常是 FP32）——猜错会得到"数值全错"或直接失败。见 TROUBLESHOOTING #18。
    const nvinfer1::DataType input_dtype = cuda->getTensorDataType("input");
    if (input_dtype == nvinfer1::DataType::kHALF) {
        std::vector<__half> half_input(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            half_input[i] = __float2half(input[i]);
        }
        DeviceBuffer d_half(input.size() * sizeof(__half));
        if (!d_half.Allocate(input.size() * sizeof(__half))) {
            return {};
        }
        CUDA_CHECK(cudaMemcpy(d_half.data(), half_input.data(), d_half.size(),
                              cudaMemcpyHostToDevice));
        if (!engine->SetTensorAddress("input", d_half.data())) {
            return {};
        }
    } else if (!engine->SetTensorAddress("input", d_input.data())) {
        return {};
    }
    if (!engine->SetTensorAddress("output", d_output.data())) {
        return {};
    }
    if (!engine->Enqueue(nullptr)) {
        MINI_TRT_LOG_ERROR("ResNet18 FP16 enqueue failed");
        return {};
    }
    engine->Synchronize(nullptr);
    return test_support::ReadFloats(d_output.data(), kLogitsElements);
}

// 统计一次对拍，并按 argmax 与数值双判据断言。
void ExpectMatches(const std::string& label, const std::vector<float>& reference,
                   const std::vector<float>& actual, float abs_tol) {
    ASSERT_EQ(actual.size(), reference.size()) << label;
    const DiffStats stats = ComputeDiffStats(reference, actual);
    int32_t mismatches = 0;
    for (int32_t b = 0; b < kBatch; ++b) {
        if (ArgmaxOfRow(reference, b) != ArgmaxOfRow(actual, b)) {
            ++mismatches;
        }
    }
    std::cout << "[ResNet18 FP16] " << label << ": max_abs = " << stats.max_abs
              << "  max_rel = " << stats.max_rel << "  argmax 不一致 = " << mismatches << "/"
              << kBatch << "\n";
    EXPECT_EQ(mismatches, 0) << label << "：FP16 下逐个样本的分类结果必须与 FP32 一致";
    EXPECT_LT(stats.max_abs, abs_tol) << label;
}

}  // namespace

// R2.5a：FP16 引擎（ONNX 路径）vs FP32 基线。
//
// 判据是**两条并列**的：语义上 argmax 必须逐样本一致；数值上按实测的"纯 FP16 舍入"量级
// 取上界（见 kFp16MaxAbs 的推导）。只报数值、或只比 argmax，都会漏掉一类问题。
TEST(ResNet18Fp16PathTest, OnnxEngineMatchesFp32Baseline) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    const std::string baseline_dir = FindBaselineDir();
    if (dir.empty() || onnx.empty() || baseline_dir.empty()) {
        GTEST_SKIP() << "需要 models/resnet18（含 P4-1 基线）与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder builder(logger, Fp16Config());
    if (!std::filesystem::exists(kOnnxFp16Engine)) {
        ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, kOnnxFp16Engine, {}));
    }
    Engine engine(kOnnxFp16Engine, logger);

    const std::vector<float> input = ReadF32File(
        baseline_dir + "/inputs/ref_ramp_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_ramp_b8.bin", kLogitsElements);
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(reference.empty());
    ExpectMatches("ONNX-FP16 vs torchvision-FP32 (ramp)", reference, RunEngine(&engine, input),
                  kFp16MaxAbs);
}

// R2.5b：原生 FP16 vs ONNX FP16。
//
// 这条**不涉精度换算**（两边同精度、同一份权重），所以阈值按实测取紧值——
// 它验的是"两条路在 FP16 下也一致"，与 R2.5a 的"FP16 与 FP32 的固有差距"是两回事，
// 阈值自然不能共用。
TEST(ResNet18Fp16PathTest, NativeMatchesOnnxInFp16) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    if (dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18 与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder builder(logger, Fp16Config());
    if (!std::filesystem::exists(kOnnxFp16Engine)) {
        ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, kOnnxFp16Engine, {}));
    }
    if (!std::filesystem::exists(kNativeFp16Engine)) {
        ASSERT_TRUE(builder.BuildFromConfig(dir, kNativeFp16Engine, BuildStage::kSingle));
    }
    Engine onnx_engine(kOnnxFp16Engine, logger);
    Engine native_engine(kNativeFp16Engine, logger);

    // 先把**I/O 契约**比一遍：名字与声明精度必须一致。
    // 为什么值得断言：FP16 引擎的 I/O 未必声明成 FP16（弱类型网络下由 TRT 决定，实测常为 FP32），
    // 调用方要靠声明精度决定喂什么——两条路若不一致，"同一份调用代码"就不成立了。
    // 见 TROUBLESHOOTING #18。
    {
        nvinfer1::ICudaEngine* a = onnx_engine.GetCudaEngine();
        nvinfer1::ICudaEngine* b = native_engine.GetCudaEngine();
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        for (const char* name : {"input", "output"}) {
            EXPECT_EQ(a->getTensorDataType(name), b->getTensorDataType(name))
                << name << " 的声明精度在两条路之间不一致";
            EXPECT_EQ(a->getTensorIOMode(name), b->getTensorIOMode(name)) << name;
        }
        std::cout << "[ResNet18 FP16] I/O 声明精度: input="
                  << (a->getTensorDataType("input") == nvinfer1::DataType::kHALF ? "FP16" : "FP32")
                  << ", output="
                  << (a->getTensorDataType("output") == nvinfer1::DataType::kHALF ? "FP16" : "FP32")
                  << "\n";
    }

    const std::vector<float> input = test_support::MakeRampInput(kBatch);
    const std::vector<float> onnx_logits = RunEngine(&onnx_engine, input);
    const std::vector<float> native_logits = RunEngine(&native_engine, input);

    // 把"三角"打出来当证据：两条 FP16 各自离 FP32 多远、它们互相差多少。
    // 没有这几行，"阈值为什么是这个数"就只能靠嘴说——而那正是 AGENTS.md §7 禁止的。
    EngineBuilder fp32_builder(logger, Fp32Config());
    if (!std::filesystem::exists(kNativeFp32Engine)) {
        ASSERT_TRUE(fp32_builder.BuildFromConfig(dir, kNativeFp32Engine, BuildStage::kSingle));
    }
    Engine fp32_engine(kNativeFp32Engine, logger);
    const std::vector<float> fp32_logits = RunEngine(&fp32_engine, input);
    ASSERT_EQ(fp32_logits.size(), kLogitsElements);
    std::cout << "[ResNet18 FP16 三角] 原生-FP16 vs 原生-FP32 = "
              << ComputeDiffStats(fp32_logits, native_logits).max_abs
              << "；ONNX-FP16 vs 原生-FP32 = "
              << ComputeDiffStats(fp32_logits, onnx_logits).max_abs << "\n";

    ExpectMatches("原生-FP16 vs ONNX-FP16", onnx_logits, native_logits,
                  kFp16CrossPathMaxAbs);
}

// R2.5d：原生 FP16 vs **外部基线**（torchvision FP32）。
//
// 为什么单列：R2.5b 只证"两条 FP16 路径互相一致"，R2.5c 走的是 ONNX FP16 引擎。
// 原生 FP16 与外部真值之间此前只有**传递**推断（0.0076 + 0.031），没有直接测量——
// 而"两条路共享同一个 bug"这种情况恰好能躲过互拍、躲不过外部基线。
TEST(ResNet18Fp16PathTest, NativeEngineMatchesFp32Baseline) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string baseline_dir = FindBaselineDir();
    if (dir.empty() || baseline_dir.empty()) {
        GTEST_SKIP() << "需要 models/resnet18（含 P4-1 基线）";
    }
    Logger logger;
    EngineBuilder builder(logger, Fp16Config());
    if (!std::filesystem::exists(kNativeFp16Engine)) {
        ASSERT_TRUE(builder.BuildFromConfig(dir, kNativeFp16Engine, BuildStage::kSingle));
    }
    Engine engine(kNativeFp16Engine, logger);

    const std::vector<float> input = ReadF32File(
        baseline_dir + "/inputs/ref_ramp_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_ramp_b8.bin", kLogitsElements);
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(reference.empty());
    ExpectMatches("原生-FP16 vs torchvision-FP32 (ramp)", reference,
                  RunEngine(&engine, input), kFp16MaxAbs);
}

// R2.5c：CVRunner 驱动 FP16 引擎（端到端）。
//
// 为什么单列：Runner 的前处理与绑定都按 FP32 像素质设计，而 FP16 引擎的内部精度不同。
// 这条验的是"调用方接口在 FP16 下不变"——输入仍喂 `[0,255]` 的 FP32 NCHW。
TEST(ResNet18Fp16PathTest, CvRunnerOnFp16EngineMatchesFp32Baseline) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    const std::string baseline_dir = FindBaselineDir();
    if (dir.empty() || onnx.empty() || baseline_dir.empty()) {
        GTEST_SKIP() << "需要 models/resnet18（含 P4-1 基线）与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder builder(logger, Fp16Config());
    if (!std::filesystem::exists(kOnnxFp16Engine)) {
        ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, kOnnxFp16Engine, {}));
    }
    const std::vector<float> mean = {test_support::kImageNetMean[0],
                                     test_support::kImageNetMean[1],
                                     test_support::kImageNetMean[2]};
    const std::vector<float> std = {test_support::kImageNetStd[0], test_support::kImageNetStd[1],
                                    test_support::kImageNetStd[2]};
    CVRunner runner(std::make_shared<Engine>(kOnnxFp16Engine, logger), mean, std);
    ASSERT_TRUE(runner.ok()) << "CVRunner 必须能驱动 FP16 引擎（I/O 声明由引擎决定）";

    const std::vector<float> pixels =
        ReadF32File(baseline_dir + "/inputs/ref_pixels_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_pixels_b8.bin", kLogitsElements);
    ASSERT_FALSE(pixels.empty());
    ASSERT_FALSE(reference.empty());
    ExpectMatches("CVRunner+ONNX-FP16 vs torchvision-FP32 (pixels)", reference,
                  runner.Infer(pixels, kBatch), kFp16MaxAbs);
}

}  // namespace mini_trt_llm
