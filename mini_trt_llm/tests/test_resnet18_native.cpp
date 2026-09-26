#include "cv_test_support.hpp"
#include "diff_stats.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/resnet18_model_builder.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::ComputeDiffStats;
using test_support::DiffStats;
using test_support::FindFile;
using test_support::kCvClasses;
using test_support::MakeRampInput;
using test_support::ReadF32File;

constexpr int32_t kBatch = 8;
const size_t kInputElements = static_cast<size_t>(kBatch) * 3 * 224 * 224;
const size_t kLogitsElements = static_cast<size_t>(kBatch) * kCvClasses;

const char* const kNativeFp32Engine = "/tmp/mini_trt_llm_resnet18_native_fp32.engine";

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

EngineBuilder::Config Fp32Config() {
    EngineBuilder::Config config;
    config.precision = Precision::FP32;  // 与其它对拍一致：显式写出目标精度（#21）
    return config;
}

// 在原生引擎上跑一次前向（与 test_resnet18_onnx.cpp 的 RunCnnEngine 同一形态：
// 按**引擎声明**的 I/O 绑定，不假定名字与精度）。
std::vector<float> RunNative(Engine* engine, const std::vector<float>& input,
                             int32_t batch = kBatch) {
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        return {};
    }
    const size_t input_bytes = input.size() * sizeof(float);
    const size_t output_count = static_cast<size_t>(batch) * kCvClasses;
    DeviceBuffer d_input(input_bytes);
    DeviceBuffer d_output(output_count * sizeof(float));
    if (!d_input.Allocate(input_bytes) || !d_output.Allocate(output_count * sizeof(float))) {
        return {};
    }
    CUDA_CHECK(cudaMemcpy(d_input.data(), input.data(), input_bytes, cudaMemcpyHostToDevice));
    if (!engine->SetOptimizationProfile(0, nullptr) ||
        !engine->SetInputShape("input", nvinfer1::Dims4{batch, 3, 224, 224}) ||
        !engine->SetTensorAddress("input", d_input.data()) ||
        !engine->SetTensorAddress("output", d_output.data())) {
        return {};
    }
    if (!engine->Enqueue(nullptr)) {
        MINI_TRT_LOG_ERROR("native ResNet18 enqueue failed");
        return {};
    }
    engine->Synchronize(nullptr);
    return test_support::ReadFloats(d_output.data(), output_count);
}

}  // namespace

// ---------------------------------------------------------------------------
// L1：建网（真机；只建网络不建引擎，秒级）
// ---------------------------------------------------------------------------

TEST(ResNet18NetworkBuildTest, BuildsWithExpectedIo) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    if (dir.empty() || !std::filesystem::exists(dir + "/model.safetensors")) {
        GTEST_SKIP() << "需要 models/resnet18（先跑 tools/convert/onnx_to_mini_trt_llm.py）";
    }
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    ASSERT_NE(builder, nullptr) << "CUDA 可用却建不出 builder（见 TROUBLESHOOTING #19 的教训）";
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    ASSERT_NE(network, nullptr);

    const ModelConfig model_config = ModelConfig::Load(dir);
    WeightLoader weights;
    ASSERT_TRUE(weights.Load(dir));
    weights.SetWeightMap(model_config.weight_map);

    ResNet18ModelBuilder resnet;
    BuildOptions options;
    options.stage = BuildStage::kSingle;
    options.weight_dtype = nvinfer1::DataType::kFLOAT;
    ASSERT_TRUE(resnet.Build(network.get(), weights, model_config, options));

    // I/O 契约必须与 ONNX 路径同名同义（两条路共用 OnnxIoContractFor）
    ASSERT_EQ(network->getNbInputs(), 1);
    EXPECT_STREQ(network->getInput(0)->getName(), "input");
    ASSERT_EQ(network->getNbOutputs(), 1);
    EXPECT_STREQ(network->getOutput(0)->getName(), "output");
    const nvinfer1::Dims out_dims = network->getOutput(0)->getDimensions();
    ASSERT_EQ(out_dims.nbDims, 2);
    EXPECT_EQ(out_dims.d[0], -1) << "batch 维必须保持动态";
    EXPECT_EQ(out_dims.d[1], kCvClasses);

    // 逐层 getDimensions() 冒烟：TRT 的 shape 推导错误是**延迟报告**的
    // （addShuffle / addReduce 等在调用时不返回 nullptr），真机踩过一次。
    for (int32_t i = 0; i < network->getNbLayers(); ++i) {
        nvinfer1::ILayer* layer = network->getLayer(i);
        ASSERT_NE(layer, nullptr);
        for (int32_t out = 0; out < layer->getNbOutputs(); ++out) {
            const nvinfer1::Dims dims = layer->getOutput(out)->getDimensions();
            EXPECT_GT(dims.nbDims, 0)
                << "层 " << i << " (" << layer->getName() << ") 输出 " << out
                << " 维数无效，说明建图阶段 shape 推导出了问题";
        }
    }
    std::cout << "[ResNet18] 原生网络层数 = " << network->getNbLayers() << "\n";
}

// stage 语义：CV 没有 prefill/decode 之分，传别的 stage 必须**显式失败**
// （静默忽略会建出一个语义不明的引擎）。
TEST(ResNet18NetworkBuildTest, RejectsNonSingleStage) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    if (dir.empty() || !std::filesystem::exists(dir + "/model.safetensors")) {
        GTEST_SKIP() << "需要 models/resnet18";
    }
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    ASSERT_NE(builder, nullptr);
    const ModelConfig model_config = ModelConfig::Load(dir);
    WeightLoader weights;
    ASSERT_TRUE(weights.Load(dir));
    weights.SetWeightMap(model_config.weight_map);

    ResNet18ModelBuilder resnet;
    for (BuildStage stage : {BuildStage::kPrefill, BuildStage::kDecode}) {
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
        ASSERT_NE(network, nullptr);
        BuildOptions options;
        options.stage = stage;
        options.weight_dtype = nvinfer1::DataType::kFLOAT;
        EXPECT_FALSE(resnet.Build(network.get(), weights, model_config, options))
            << "CV 模型不该接受 prefill/decode 切面";
    }
}

// 缺权重必须建网失败：临时目录里放**改坏的 config**（某个 weight_map 指向不存在的键），
// safetensors 用软链接指过去——不复制那 46 MB。
TEST(ResNet18NetworkBuildTest, MissingWeightFailsTheBuild) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    if (dir.empty() || !std::filesystem::exists(dir + "/model.safetensors")) {
        GTEST_SKIP() << "需要 models/resnet18";
    }
    const std::string tmp = "/tmp/mini_trt_llm_resnet18_missing_weight";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    // config：把 conv1.weight 指到一个不存在的 source key
    {
        std::ifstream in(dir + "/config.json");
        std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        const std::string needle = R"("conv1.weight": "conv1.weight")";
        const size_t pos = text.find(needle);
        ASSERT_NE(pos, std::string::npos) << "config.json 里找不到预期的 weight_map 项";
        text.replace(pos, needle.size(), R"("conv1.weight": "does_not_exist")");
        std::ofstream out(tmp + "/config.json");
        out << text;
    }
    std::filesystem::create_symlink(
        std::filesystem::absolute(dir + "/model.safetensors"), tmp + "/model.safetensors");

    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    ASSERT_NE(builder, nullptr);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    ASSERT_NE(network, nullptr);
    const ModelConfig model_config = ModelConfig::Load(tmp);
    WeightLoader weights;
    ASSERT_TRUE(weights.Load(tmp));
    weights.SetWeightMap(model_config.weight_map);

    ResNet18ModelBuilder resnet;
    BuildOptions options;
    options.stage = BuildStage::kSingle;
    options.weight_dtype = nvinfer1::DataType::kFLOAT;
    EXPECT_FALSE(resnet.Build(network.get(), weights, model_config, options))
        << "weight_map 指向不存在的键时必须建网失败，而不是建出一个缺权重的网络";
    std::filesystem::remove_all(tmp);
}

// ---------------------------------------------------------------------------
// L2：数值（真机，引擎级）
// ---------------------------------------------------------------------------

TEST(ResNet18NativeAccuracyTest, MatchesOnnxPath) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    if (dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18 与 0_resnet18_onnx/resnet18.onnx";
    }
    Logger logger;
    EngineBuilder builder(logger, Fp32Config());
    ASSERT_TRUE(builder.BuildFromConfig(dir, kNativeFp32Engine, BuildStage::kSingle))
        << "原生建图/建引擎失败";
    const std::string onnx_engine = "/tmp/mini_trt_llm_resnet18_onnx_fp32.engine";
    EngineBuilder onnx_builder(logger, Fp32Config());
    ASSERT_TRUE(onnx_builder.BuildFromOnnx(dir, onnx, onnx_engine, {}));

    Engine native(kNativeFp32Engine, logger);
    Engine onnx_ctx(onnx_engine, logger);
    const std::vector<float> input = MakeRampInput(kBatch);  // 未归一化的 ramp（与基线同式）
    const std::vector<float> native_logits = RunNative(&native, input);
    const std::vector<float> onnx_logits = RunNative(&onnx_ctx, input);
    ASSERT_EQ(native_logits.size(), kLogitsElements);
    ASSERT_EQ(onnx_logits.size(), kLogitsElements);

    const DiffStats stats = ComputeDiffStats(onnx_logits, native_logits);
    int32_t mismatches = 0;
    for (int32_t b = 0; b < kBatch; ++b) {
        if (test_support::ArgmaxOfRow(onnx_logits, b) != test_support::ArgmaxOfRow(native_logits, b)) {
            ++mismatches;
        }
    }
    std::cout << "[ResNet18] 原生 vs ONNX（同一份权重）: max_abs = " << stats.max_abs
              << "  max_rel = " << stats.max_rel << "  argmax 不一致 = " << mismatches << "/"
              << kBatch << "\n";
    EXPECT_EQ(mismatches, 0);
    // 阈值 **1e-5**，出处：两条路消费**逐位相同的权重**、同一台 GPU、同为 FP32，
    // 差异只剩 TRT 为两张图选的 kernel/tactic → 实测 max_abs = 1.07e-6（2026-09-26），
    // 阈值取实测的约 10 倍留构建间余量。**刻意比 R2.4（vs torchvision，1e-4）紧一个数量级**：
    // 那边还要承受 BN 折叠带来的 ~1.9e-5 差异，这边没有那项，用同一个尺子等于白白放宽。
    EXPECT_LT(stats.max_abs, 1e-5f);
}

TEST(ResNet18NativeAccuracyTest, MatchesBaseline) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string baseline_dir = test_support::FindBaselineDir();
    if (dir.empty() || baseline_dir.empty()) {
        GTEST_SKIP() << "需要 models/resnet18（含 P4-1 基线）";
    }
    Logger logger;
    EngineBuilder builder(logger, Fp32Config());
    ASSERT_TRUE(builder.BuildFromConfig(dir, kNativeFp32Engine, BuildStage::kSingle));
    Engine engine(kNativeFp32Engine, logger);

    const std::vector<float> input =
        ReadF32File(baseline_dir + "/inputs/ref_ramp_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_ramp_b8.bin", kLogitsElements);
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(reference.empty());

    const std::vector<float> actual = RunNative(&engine, input);
    ASSERT_EQ(actual.size(), kLogitsElements);
    const DiffStats stats = ComputeDiffStats(reference, actual);
    int32_t mismatches = 0;
    for (int32_t b = 0; b < kBatch; ++b) {
        if (test_support::ArgmaxOfRow(reference, b) != test_support::ArgmaxOfRow(actual, b)) {
            ++mismatches;
        }
    }
    std::cout << "[ResNet18] 原生 vs torchvision 基线: max_abs = " << stats.max_abs
              << "  max_rel = " << stats.max_rel << "  argmax 不一致 = " << mismatches << "/"
              << kBatch << "\n";
    EXPECT_EQ(mismatches, 0);
    // 与 R2.1 同口径（1e-4 = 最大无关差异的 5 倍，推导见 test_resnet18_onnx.cpp 的注释）
    EXPECT_LT(stats.max_abs, 1e-4f);
}

// profile 验收必须在**原生引擎**上再做一遍。
//
// 为什么不能只靠 R1.5（那条跑的是 ONNX 引擎）：profile 的范围取自 `EngineBuilder::Config`，
// 但它挂在**网络输入的维度**上；原生网络的输入维度是建图时手写的（`{-1, C, H, W}`），
// 与 ONNX 图无关。手写错了（比如把 batch 维写成常量）在 batch=1 时可能看不出来，
// 到 batch=16 才炸。这条把原生侧的形状声明也钉住。
TEST(ResNet18NativeProfileTest, AcceptsBatchRangeAndRejectsOutOfRange) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    if (dir.empty() || !std::filesystem::exists(dir + "/model.safetensors")) {
        GTEST_SKIP() << "需要 models/resnet18（先跑 tools/convert/onnx_to_mini_trt_llm.py）";
    }
    Logger logger;
    EngineBuilder builder(logger, Fp32Config());
    ASSERT_TRUE(builder.BuildFromConfig(dir, kNativeFp32Engine, BuildStage::kSingle));
    Engine engine(kNativeFp32Engine, logger);

    const int32_t max_batch = 16;
    for (int32_t batch : {1, kBatch, max_batch}) {
        const std::vector<float> logits = RunNative(&engine, MakeRampInput(batch), batch);
        EXPECT_EQ(logits.size(), static_cast<size_t>(batch) * kCvClasses)
            << "batch=" << batch << " 在 profile 范围内，原生引擎必须能真跑";
    }
    EXPECT_FALSE(engine.SetInputShape("input", nvinfer1::Dims4{max_batch + 1, 3, 224, 224}))
        << "batch=17 超出 profile 上界，必须显式失败";
}

}  // namespace mini_trt_llm
