#include "cv_test_support.hpp"
#include "diff_stats.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/cv_runner.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "test_gpu_guard.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::ComputeDiffStats;
using test_support::DiffStats;
using test_support::FindBaselineDir;
using test_support::FindFile;
using test_support::kCvChannels;
using test_support::kCvClasses;
using test_support::kCvSize;
using test_support::ReadF32File;

constexpr int32_t kBatch = 8;
const size_t kInputElements =
    static_cast<size_t>(kBatch) * kCvChannels * kCvSize * kCvSize;
const size_t kLogitsElements = static_cast<size_t>(kBatch) * kCvClasses;

// P4-2 建好的 FP32 引擎（CVRunner 需要一个真正的引擎才能验证端到端）。
const char* const kFp32EnginePath = "/tmp/mini_trt_llm_resnet18_onnx_fp32.engine";

// 返回**模型目录**（不是 config.json 的路径）。
//
// 踩过的坑（#25）：第一版返回的是 config.json 的文件路径，于是调用方拼出的
// `<...>/config.json/model.safetensors` 永远不存在 → 用例**静默 skip**。
// 与"路径类 helper 返回什么"这种细节相比，静默跳过才是最贵的错误：
// 它让一条本该验证的用例看起来"跑过了"。
std::string FindResNet18ModelDir() {
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

// ImageNet 的 mean/std，与 cv_test_support.hpp / P4-1 脚本一致（同一份数值，不各写一套）。
const std::vector<float> kMean = {test_support::kImageNetMean[0], test_support::kImageNetMean[1],
                                  test_support::kImageNetMean[2]};
const std::vector<float> kStd = {test_support::kImageNetStd[0], test_support::kImageNetStd[1],
                                 test_support::kImageNetStd[2]};

}  // namespace

// ---------------------------------------------------------------------------
// L0：前处理（纯 host，沙箱可跑）
//
// 前处理是"数值差会被误判成引擎错"的那一层：两侧公式一旦不一致，后面所有对拍结论都失真。
// P4-1 已经把 Python 侧的产物落盘（`ref_pixels_b8.normalized.f32.bin`），这里拿它当对照物。
// ---------------------------------------------------------------------------

TEST(CvRunnerPreprocessTest, MatchesBaselineNormalization) {
    const std::string dir = FindBaselineDir();
    if (dir.empty()) {
        GTEST_SKIP() << "缺少 models/resnet18 基线产物（先跑 scripts/ref_resnet18.py）";
    }
    const std::vector<float> pixels =
        ReadF32File(dir + "/inputs/ref_pixels_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> expected =
        ReadF32File(dir + "/inputs/ref_pixels_b8.normalized.f32.bin", kInputElements);
    ASSERT_EQ(pixels.size(), kInputElements);
    ASSERT_EQ(expected.size(), kInputElements);

    // 刻意用 batch=8 来测：通道下标公式在 batch=1 时会**凑巧对**（#23），
    // 只有多 batch 才能发现分母用错。
    const std::vector<float> actual = NormalizePixelsToNchw(
        pixels, kCvChannels, kCvSize * kCvSize, kMean, kStd);
    ASSERT_EQ(actual.size(), expected.size());

    const DiffStats stats = ComputeDiffStats(expected, actual);
    std::cout << "[CVRunner] 前处理 vs P4-1 基线: max_abs = " << stats.max_abs
              << "  max_rel = " << stats.max_rel << "\n";
    // 两侧是同一公式的两种语言实现（都是 float32 运算）→ 期望逐位相同；
    // 阈值 1e-6 只作为回归护栏（改运算顺序会从这里开始偏离）。
    EXPECT_LT(stats.max_abs, 1e-6f);
}

// 逐通道 mean/std 与像素边界：**写反通道顺序**是最容易犯、又最难从数值上发现的错，
// 所以这里刻意让三通道的 mean/std 互不相同；边界值 0 与 255 单独钉一下。
TEST(CvRunnerPreprocessTest, HandlesPerChannelAndBoundaries) {
    // 构造 1×3×1×1：三个通道各取一个可以区分的值
    const std::vector<float> pixels = {0.0f, 127.5f, 255.0f};
    const std::vector<float> mean = {1.0f, 2.0f, 3.0f};   // 刻意互不相同
    const std::vector<float> std = {0.5f, 4.0f, 10.0f};   // 同上

    const std::vector<float> got = NormalizePixelsToNchw(pixels, 3, /*pixels_per_channel=*/1, mean, std);
    ASSERT_EQ(got.size(), 3u);
    // (0/255 - 1.0)/0.5、 (127.5/255 - 2.0)/4.0、 (255/255 - 3.0)/10.0
    EXPECT_FLOAT_EQ(got[0], (0.0f / 255.0f - mean[0]) / std[0]);
    EXPECT_FLOAT_EQ(got[1], (127.5f / 255.0f - mean[1]) / std[1]);
    EXPECT_FLOAT_EQ(got[2], (255.0f / 255.0f - mean[2]) / std[2]);

    // 通道顺序写反（把第 3 个通道的 mean/std 用在第 1 个通道上）必须产生不同的结果——
    // 若两者相同，说明实现把通道参数用错了（或压根没用）。
    const std::vector<float> swapped = NormalizePixelsToNchw(
        pixels, 3, /*pixels_per_channel=*/1, {3.0f, 2.0f, 1.0f}, {10.0f, 4.0f, 0.5f});
    ASSERT_EQ(swapped.size(), 3u);
    EXPECT_NE(got[0], swapped[0]) << "通道参数没有被逐通道使用";

    // 边界：全 0 与全 255 都应落在可表示的有限值上（且方向正确：0 更小、255 更大）
    const std::vector<float> black =
        NormalizePixelsToNchw({0.0f}, 1, /*pixels_per_channel=*/1, {0.5f}, {0.5f});
    const std::vector<float> white =
        NormalizePixelsToNchw({255.0f}, 1, /*pixels_per_channel=*/1, {0.5f}, {0.5f});
    ASSERT_EQ(black.size(), 1u);
    ASSERT_EQ(white.size(), 1u);
    EXPECT_FLOAT_EQ(black[0], -1.0f);
    EXPECT_FLOAT_EQ(white[0], 1.0f);
    EXPECT_TRUE(std::isfinite(black[0]) && std::isfinite(white[0]));

    // 参数不合法必须返回空（而不是算出一堆垃圾）
    EXPECT_TRUE(NormalizePixelsToNchw(pixels, 3, 1, {1.0f}, {1.0f, 1.0f, 1.0f}).empty());
    EXPECT_TRUE(NormalizePixelsToNchw({1.0f, 2.0f, 3.0f, 4.0f}, 3, 1, mean, std).empty());
    EXPECT_TRUE(NormalizePixelsToNchw({}, 3, 1, mean, std).empty());
    EXPECT_TRUE(NormalizePixelsToNchw(pixels, 3, /*pixels_per_channel=*/0, mean, std).empty());
}

// ---------------------------------------------------------------------------
// L3：端到端（真机）
// ---------------------------------------------------------------------------

// 构造期的失败必须能被 `ok()` 查出来，而不是等到 `Infer` 才崩——这是头文件里写明的契约。
//
// 空引擎这一支**不需要 GPU**（构造函数在碰 CUDA 之前就返回了），所以它能在沙箱里跑。
TEST(CvRunnerTest, RejectsNullEngineWithoutCuda) {
    CVRunner runner(nullptr, kMean, kStd);
    EXPECT_FALSE(runner.ok());
    const std::vector<float> one_image(
        static_cast<size_t>(kCvChannels) * kCvSize * kCvSize, 0.0f);
    EXPECT_TRUE(runner.Infer(one_image, 1).empty()) << "无效 runner 必须返回空，而不是崩";
    const Engine::BenchResult bench = runner.Benchmark(1, 1, 1);
    EXPECT_EQ(bench.mean_ms, 0.0f) << "无效 runner 的 benchmark 必须返回零值统计";
}

TEST(CvRunnerTest, InferMatchesBaseline) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindBaselineDir();
    const std::string model_dir = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (dir.empty() || model_dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18（基线 + config）与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;  // 与基线对拍必须显式指定（#21）
    EngineBuilder builder(logger, builder_config);
    ASSERT_TRUE(builder.BuildFromOnnx(model_dir,
                                      onnx, kFp32EnginePath, {}));

    CVRunner runner(std::make_shared<Engine>(kFp32EnginePath, logger), kMean, kStd);
    ASSERT_TRUE(runner.ok());

    // 契约输入是 `[0,255]` 的 NCHW 像素质：CVRunner 自己归一化（D4）
    const std::vector<float> pixels =
        ReadF32File(dir + "/inputs/ref_pixels_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> reference =
        ReadF32File(dir + "/ref_pixels_b8.bin", kLogitsElements);
    ASSERT_EQ(pixels.size(), kInputElements);
    ASSERT_EQ(reference.size(), kLogitsElements);

    for (int32_t batch : {1, kBatch}) {
        std::vector<float> batch_pixels(pixels.begin(),
                                        pixels.begin() + static_cast<ptrdiff_t>(
                                            static_cast<size_t>(batch) * kCvChannels * kCvSize * kCvSize));
        const std::vector<float> got = runner.Infer(batch_pixels, batch);
        ASSERT_EQ(got.size(), static_cast<size_t>(batch) * kCvClasses) << "batch=" << batch;
        const std::vector<float> ref_rows(reference.begin(),
                                          reference.begin() + static_cast<ptrdiff_t>(
                                              static_cast<size_t>(batch) * kCvClasses));
        const DiffStats stats = ComputeDiffStats(ref_rows, got);
        std::cout << "[CVRunner] batch=" << batch << " vs 基线: max_abs = " << stats.max_abs
                  << "  max_rel = " << stats.max_rel << "\n";
        // 阈值与 P4-2 的引擎级对拍同一口径（1e-4 = 最大无关差异的 5 倍，见
        // tests/test_resnet18_onnx.cpp 里 kRampFp32MaxAbs 的推导）：CVRunner 只是把
        // 同一个引擎包了一层前处理，不该引入额外的数值差异。
        EXPECT_LT(stats.max_abs, 1e-4f) << "batch=" << batch;
    }
}

TEST(CvRunnerTest, RejectsInvalidBatchAndSize) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (model_dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/config.json 与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;
    EngineBuilder builder(logger, builder_config);
    ASSERT_TRUE(builder.BuildFromOnnx(model_dir,
                                      onnx, kFp32EnginePath, {}));
    CVRunner runner(std::make_shared<Engine>(kFp32EnginePath, logger), kMean, kStd);
    ASSERT_TRUE(runner.ok());

    const size_t one_image = static_cast<size_t>(kCvChannels) * kCvSize * kCvSize;
    std::vector<float> valid(one_image, 0.0f);

    // 超出 profile 上界（引擎的 max_batch 是 16）
    EXPECT_TRUE(runner.Infer(std::vector<float>(17 * one_image, 0.0f), 17).empty())
        << "batch=17 超出 profile，必须返回空而不是静默用错形状";
    // batch=0
    EXPECT_TRUE(runner.Infer({}, 0).empty());
    // 长度与 batch 不匹配
    EXPECT_TRUE(runner.Infer(valid, 2).empty()) << "元素数不足 batch=2 时必须失败";
    // 合法的那次必须成功（否则上面的"失败"没有对照意义）
    EXPECT_EQ(runner.Infer(valid, 1).size(), static_cast<size_t>(kCvClasses));
}

// mean/std 长度与输入通道数不符：必须在**构造期**就被拒（`ok() == false`），
// 而不是拿着错的常量算出一堆垃圾数值。这一支需要真机引擎（要知道通道数）。
TEST(CvRunnerTest, RejectsMeanStdSizeMismatch) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (model_dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/config.json 与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;
    EngineBuilder builder(logger, builder_config);
    ASSERT_TRUE(builder.BuildFromOnnx(model_dir,
                                      onnx, kFp32EnginePath, {}));
    auto engine = std::make_shared<Engine>(kFp32EnginePath, logger);

    const std::vector<float> one_image(
        static_cast<size_t>(kCvChannels) * kCvSize * kCvSize, 0.0f);
    for (const std::vector<float>& bad : {std::vector<float>{0.5f},         // 太短
                                          std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}}) {
        CVRunner runner(engine, bad, kStd);
        EXPECT_FALSE(runner.ok()) << "mean 长度 " << bad.size() << " 应当被拒";
        EXPECT_TRUE(runner.Infer(one_image, 1).empty());
    }
    CVRunner runner(engine, kMean, std::vector<float>{0.5f});  // std 太短
    EXPECT_FALSE(runner.ok());
}

TEST(CvRunnerTest, BenchmarkReportsFiniteStats) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (model_dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/config.json 与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;
    EngineBuilder builder(logger, builder_config);
    ASSERT_TRUE(builder.BuildFromOnnx(model_dir,
                                      onnx, kFp32EnginePath, {}));
    CVRunner runner(std::make_shared<Engine>(kFp32EnginePath, logger), kMean, kStd);
    ASSERT_TRUE(runner.ok());

    const Engine::BenchResult result = runner.Benchmark(/*batch_size=*/8, /*n_warmup=*/2,
                                                        /*n_run=*/5);
    std::cout << "[CVRunner] benchmark batch=8: mean=" << result.mean_ms
              << "ms p50=" << result.p50_ms << "ms p99=" << result.p99_ms
              << "ms throughput=" << result.throughput << " img/s\n";
    // 只验统计自洽——**不把性能数字当判据**（性能结论要按 future_iterations §11 的 G6 口径专门测）。
    EXPECT_GT(result.mean_ms, 0.0f);
    EXPECT_GT(result.p50_ms, 0.0f);
    EXPECT_GE(result.p99_ms, result.p50_ms);
    EXPECT_GT(result.throughput, 0.0f);

    // 非法参数必须返回零值统计而不是抛异常/给假数字
    const Engine::BenchResult bad = runner.Benchmark(/*batch_size=*/99, 1, 1);
    EXPECT_EQ(bad.mean_ms, 0.0f);
}

// CVRunner 也要能驱动**原生引擎**。
//
// 为什么单列：Runner 是按**引擎声明的** I/O 契约工作的（输入用 getProfileShape、输出用
// getTensorShape），而原生引擎与 ONNX 引擎的维度来自**不同来源**（手写建图 vs 解析图）。
// 同名 ≠ 同契约——这条把"两套引擎都能被同一个 runner 驱动"钉住。
TEST(CvRunnerTest, InferMatchesBaselineOnNativeEngine) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string baseline_dir = FindBaselineDir();
    const std::string model_dir = FindResNet18ModelDir();
    if (baseline_dir.empty() || model_dir.empty() ||
        !std::filesystem::exists(model_dir + "/model.safetensors")) {
        GTEST_SKIP() << "需要 models/resnet18（基线 + 转换产物）";
    }
    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;
    EngineBuilder builder(logger, builder_config);
    const std::string native_engine = "/tmp/mini_trt_llm_resnet18_native_fp32.engine";
    ASSERT_TRUE(builder.BuildFromConfig(model_dir, native_engine, BuildStage::kSingle));
    CVRunner runner(std::make_shared<Engine>(native_engine, logger), kMean, kStd);
    ASSERT_TRUE(runner.ok());

    const std::vector<float> pixels =
        ReadF32File(baseline_dir + "/inputs/ref_pixels_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_pixels_b8.bin", kLogitsElements);
    ASSERT_FALSE(pixels.empty());
    ASSERT_FALSE(reference.empty());

    const std::vector<float> got = runner.Infer(pixels, kBatch);
    ASSERT_EQ(got.size(), kLogitsElements);
    const DiffStats stats = ComputeDiffStats(reference, got);
    std::cout << "[CVRunner + 原生引擎] batch=" << kBatch
              << " vs 基线: max_abs = " << stats.max_abs << "  max_rel = " << stats.max_rel
              << "\n";
    // 与 R3.1 同口径（1e-4 = 最大无关差异的 5 倍，见 tests/test_resnet18_onnx.cpp 的推导）
    EXPECT_LT(stats.max_abs, 1e-4f);
}

}  // namespace mini_trt_llm
