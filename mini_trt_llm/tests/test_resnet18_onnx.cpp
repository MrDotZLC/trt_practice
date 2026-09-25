#include "diff_stats.hpp"
#include "cv_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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
using test_support::FindBaselineDir;
using test_support::FindFile;
using test_support::ArgmaxOfRow;
using test_support::MakeRampInput;
using test_support::ReadF32File;

constexpr int32_t kBatch = 8;
constexpr int32_t kClasses = test_support::kCvClasses;

std::string FindResNet18ModelDir() {
    return FindFile({"models/resnet18/config.json", "../models/resnet18/config.json",
                     "../../models/resnet18/config.json",
                     "../../../models/resnet18/config.json"});
}

// 这条路径专用于 **FP32** 引擎。
//
// 踩过的坑（务必看清）：`EngineBuilder::Config` 的默认精度是 **FP16**，而弱类型网络下
// TRT 会把 I/O 声明成 FP32——于是"日志里 input/output 都是 FP32"完全掩盖了内部是 FP16 计算。
// 第一版这里传了 `Config{}`，拿 FP16 引擎去比 torchvision FP32，量到 max_abs=0.033（rel 3.6e-3，
// 正是 FP16 的量级），差点被当成"TRT 实现差异大"而放宽阈值。
// 教训与测量过程见 docs/TROUBLESHOOTING.md #21。
const char* const kFp32EnginePath = "/tmp/mini_trt_llm_resnet18_onnx_fp32.engine";

EngineBuilder::Config Fp32BuilderConfig() {
    EngineBuilder::Config config;
    config.precision = Precision::FP32;  // 显式写出来，不依赖默认值
    return config;
}

// FP32 对拍阈值：`max_abs < 1e-4`。
//
// **出处（2026-09-25 真机实测，不是拍的）**——先量"与正确性无关的差异"，再取合理倍数：
//   · ONNX 图已把 BatchNorm 折叠进 Conv，而 torchvision 基线里 BN 是独立算子：
//     实测 max_abs = 1.9e-5（同一输入、同一 torch 进程内对拍）
//   · torch FP32 CPU vs GPU（同一模型同一权重，只换设备）：max_abs = 7.6e-6
//   · TRT FP32 引擎 vs torchvision CPU 基线：max_abs = 9.5e-6（观测值）
// 三者同量级（~1e-5）。阈值取 **1e-4 ≈ 最大无关差异的 5 倍**：给 TRT 的 tactic 选择 /
// 构建间差异留余量，同时仍是"精度选错"的强判据——实测把引擎错建成 FP16 时 max_abs 是
// **3.3e-2**（大 330 倍），仍会被这条拦住。
constexpr float kRampFp32MaxAbs = 1e-4f;

std::string FindOnnxPath() {
    return FindFile({"0_resnet18_onnx/resnet18.onnx", "../0_resnet18_onnx/resnet18.onnx",
                     "../../0_resnet18_onnx/resnet18.onnx",
                     "../../../0_resnet18_onnx/resnet18.onnx"});
}



// 在已建好的 ONNX 引擎上跑一次 batch=kBatch 的前向，返回 [batch, 1000] logits。
//
// 绑定按**引擎声明的** I/O 来（不假定名字与精度）——Phase 3 的教训：ONNX 图的
// 输入是 INT64 还是 INT32、输出是不是 FP32，都要查，不能猜。
std::vector<float> RunCnnEngine(Engine* engine, const std::vector<float>& input,
                                int32_t batch = kBatch) {
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        return {};
    }
    const size_t input_bytes = input.size() * sizeof(float);
    DeviceBuffer d_input(input_bytes);
    DeviceBuffer d_output(static_cast<size_t>(batch) * kClasses * sizeof(float));
    if (!d_input.Allocate(input_bytes) ||
        !d_output.Allocate(static_cast<size_t>(batch) * kClasses * sizeof(float))) {
        return {};
    }
    CUDA_CHECK(cudaMemcpy(d_input.data(), input.data(), input_bytes, cudaMemcpyHostToDevice));

    // 这里用 EXPECT_* 而不是 ASSERT_*：ASSERT 展开成 `return;`，在本函数（返回 vector）里编译不过。
    if (!engine->SetOptimizationProfile(0, nullptr) ||
        !engine->SetInputShape("input", nvinfer1::Dims4{batch, 3, 224, 224}) ||
        !engine->SetTensorAddress("input", d_input.data()) ||
        !engine->SetTensorAddress("output", d_output.data())) {
        return {};
    }
    if (!engine->Enqueue(nullptr)) {
        return {};
    }
    engine->Synchronize(nullptr);
    return test_support::ReadFloats(d_output.data(),
                                    static_cast<size_t>(batch) * kClasses);
}


// 建（或用缓存）一个 FP32 的 ResNet18 ONNX 引擎。返回空串表示前置产物缺失。
std::string EnsureFp32Engine(EngineBuilder* builder, bool* built_now) {
    const std::string config = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (config.empty() || onnx.empty()) {
        return {};
    }
    const std::string dir = std::filesystem::path(config).parent_path().string();
    if (!std::filesystem::exists(kFp32EnginePath)) {
        *built_now = true;
        if (!builder->BuildFromOnnx(dir, onnx, kFp32EnginePath, {})) {
            return {};
        }
    }
    return kFp32EnginePath;
}

}  // namespace

// ---------------------------------------------------------------------------
// L0：ONNX I/O 契约映射（纯 host，沙箱可跑）
//
// 为什么这组用例必须能进 CI：`BuildFromOnnx` 里"名字对不上就失败"这条护栏原先只在
// 解析完 ONNX、且 `createInferBuilder` 成功之后才执行——那两步都要 CUDA，等于护栏
// 只在真机上才被验证。把映射抽成 OnnxIoContractFor 之后，它就能在沙箱里被测到。
// ---------------------------------------------------------------------------

TEST(OnnxIoContractTest, CnnUsesInputAndOutput) {
    const OnnxIoContract contract = OnnxIoContractFor("cnn");
    EXPECT_STREQ(contract.input_name, "input");
    EXPECT_STREQ(contract.output_name, "output");
}

TEST(OnnxIoContractTest, LlmKeepsInputIdsAndLogits) {
    // 默认分支必须保持 Phase 3 的既有契约，否则 GPT-2 的 ONNX 用例会静默改变含义。
    for (const std::string& architecture : {"decoder_only", "encoder_decoder", "unknown"}) {
        const OnnxIoContract contract = OnnxIoContractFor(architecture);
        EXPECT_STREQ(contract.input_name, "input_ids") << architecture;
        EXPECT_STREQ(contract.output_name, "logits") << architecture;
    }
}

// 子图名校验发生在 createInferBuilder **之前**（先 Load config → 探 ONNX 文件 →
// 校验子图名 → 才建 builder），所以这条能在沙箱里跑——这正是"把能在沙箱跑的都摘出来"。
//
// 注意必须用**不在白名单里**的名字（白名单：attention / layernorm / position_embedding）。
// 第一版写成 {"attention"} 是错的：它是已知名字，校验会放过，测试随后撞上
// createInferBuilder 的 CUDA 依赖而失败（把"环境不具备"误报成"护栏失效"）。
TEST(ResNet18OnnxBuildTest, RejectsUnknownSubgraphName) {
    const std::string config = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (config.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/config.json 与 0_resnet18_onnx/resnet18.onnx";
    }
    const std::string dir = std::filesystem::path(config).parent_path().string();

    Logger logger;
    EngineBuilder builder(logger, EngineBuilder::Config{});
    EXPECT_FALSE(builder.BuildFromOnnx(dir, onnx, "/tmp/mini_trt_llm_resnet18_reject.engine",
                                       {"conv_stem"}))
        << "未登记的子图名应当被拒绝（写错即失败，避免假安全感）";
}

// ---------------------------------------------------------------------------
// L1/L2：真机——用 ONNX 路径建 ResNet18 引擎，并与 P4-1 的 torchvision 基线对拍
// ---------------------------------------------------------------------------

TEST(ResNet18OnnxBuildTest, BuildsFromCnnConfig) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string config = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (config.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/config.json 与 0_resnet18_onnx/resnet18.onnx";
    }
    const std::string dir = std::filesystem::path(config).parent_path().string();
    const std::string engine_path = kFp32EnginePath;

    Logger logger;
    EngineBuilder builder(logger, Fp32BuilderConfig());
    if (!std::filesystem::exists(engine_path)) {
        ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, engine_path, {}))
            << "CNN 的 I/O 契约（input/output）应当被接受";
    }
    Engine engine(engine_path, logger);
    nvinfer1::ICudaEngine* cuda = engine.GetCudaEngine();
    ASSERT_NE(cuda, nullptr);

    // I/O 契约按引擎声明核对（不假定）
    ASSERT_EQ(cuda->getNbIOTensors(), 2);
    EXPECT_TRUE(cuda->getTensorIOMode("input") == nvinfer1::TensorIOMode::kINPUT);
    EXPECT_TRUE(cuda->getTensorIOMode("output") == nvinfer1::TensorIOMode::kOUTPUT);
    const nvinfer1::Dims out_dims = cuda->getTensorShape("output");
    ASSERT_EQ(out_dims.nbDims, 2);
    EXPECT_EQ(out_dims.d[1], kClasses);
    std::cout << "[ResNet18] ONNX 引擎 I/O: input="
              << (cuda->getTensorDataType("input") == nvinfer1::DataType::kFLOAT ? "FP32"
                                                                                : "其它")
              << ", output="
              << (cuda->getTensorDataType("output") == nvinfer1::DataType::kFLOAT ? "FP32"
                                                                                  : "其它")
              << "\n";
}

TEST(ResNet18OnnxAccuracyTest, MatchesBaselineOnRampInput) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string baseline_dir = FindBaselineDir();
    const std::string config = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (baseline_dir.empty() || config.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18 的 P4-1 基线产物与 resnet18.onnx";
    }
    const std::string dir = std::filesystem::path(config).parent_path().string();
    const std::string engine_path = kFp32EnginePath;

    Logger logger;
    EngineBuilder builder(logger, Fp32BuilderConfig());
    if (!std::filesystem::exists(engine_path)) {
        ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, engine_path, {}));
    }
    Engine engine(engine_path, logger);

    const size_t elements = static_cast<size_t>(kBatch) * 3 * 224 * 224;
    // ramp 输入**不归一化**（与历史工程 src/main.cpp 同式）；喂错张量会让误差看起来像"引擎错"。
    const std::vector<float> input = ReadF32File(
        baseline_dir + "/inputs/ref_ramp_b8.contract_input.f32.bin", elements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_ramp_b8.bin",
                    static_cast<size_t>(kBatch) * kClasses);
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(reference.empty());

    const std::vector<float> actual = RunCnnEngine(&engine, input);
    ASSERT_EQ(actual.size(), reference.size());
    const DiffStats stats = ComputeDiffStats(reference, actual);

    int32_t argmax_mismatches = 0;
    for (int32_t b = 0; b < kBatch; ++b) {
        if (ArgmaxOfRow(reference, b) != ArgmaxOfRow(actual, b)) {
            ++argmax_mismatches;
        }
    }
    std::cout << "[ResNet18 对拍] ONNX 引擎 vs torchvision 基线（ramp, batch=" << kBatch
              << "）\n"
              << "  max_abs = " << stats.max_abs << "  max_rel = " << stats.max_rel
              << "  argmax 不一致 = " << argmax_mismatches << "/" << kBatch << "\n";

    EXPECT_EQ(argmax_mismatches, 0);
    EXPECT_LT(stats.max_abs, kRampFp32MaxAbs)
        << "max_abs 见上面的实测行；阈值出处见 kRampFp32MaxAbs 的注释。"
           "若显著超出，先查（例如引擎精度、I/O 绑定、输入是否喂错），不要放宽阈值";
}

// 第二套输入：真实分布（calib_data 反归一化后重新归一化的像素质）。
//
// 与 ramp 的区别不止在数值范围：ramp **不归一化**，输入分布是模型没见过的那种，
// 激活幅度大 → 实现差异会被放大；这里喂的是**归一化后**的张量，才更接近真实用法。
TEST(ResNet18OnnxAccuracyTest, MatchesBaselineOnPixels) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string baseline_dir = FindBaselineDir();
    const std::string config = FindResNet18ModelDir();
    const std::string onnx = FindOnnxPath();
    if (baseline_dir.empty() || config.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/resnet18 的 P4-1 基线产物与 resnet18.onnx";
    }
    const std::string dir = std::filesystem::path(config).parent_path().string();

    Logger logger;
    EngineBuilder builder(logger, Fp32BuilderConfig());
    if (!std::filesystem::exists(kFp32EnginePath)) {
        ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, kFp32EnginePath, {}));
    }
    Engine engine(kFp32EnginePath, logger);

    const size_t elements = static_cast<size_t>(kBatch) * 3 * 224 * 224;
    // **喂归一化后的张量**（CVRunner 内部做的就是这一步；这里直接喂其结果，用于引擎级对拍）
    const std::vector<float> input = ReadF32File(
        baseline_dir + "/inputs/ref_pixels_b8.normalized.f32.bin", elements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_pixels_b8.bin",
                    static_cast<size_t>(kBatch) * kClasses);
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(reference.empty());

    const std::vector<float> actual = RunCnnEngine(&engine, input);
    ASSERT_EQ(actual.size(), reference.size());
    const DiffStats stats = ComputeDiffStats(reference, actual);
    int32_t argmax_mismatches = 0;
    for (int32_t b = 0; b < kBatch; ++b) {
        if (ArgmaxOfRow(reference, b) != ArgmaxOfRow(actual, b)) {
            ++argmax_mismatches;
        }
    }
    std::cout << "[ResNet18 对拍] ONNX 引擎 vs torchvision 基线（pixels 归一化输入, batch="
              << kBatch << "）\n"
              << "  max_abs = " << stats.max_abs << "  max_rel = " << stats.max_rel
              << "  argmax 不一致 = " << argmax_mismatches << "/" << kBatch << "\n";
    // argmax 在这一路上判别力有限（8 张里 5 张是 class 1，见 phase4_test_plan §4.1），
    // 所以它只是兜底；主判据是数值。
    EXPECT_LT(stats.max_abs, kRampFp32MaxAbs)
        << "同一阈值口径（1e-4 = 最大无关差异的 5 倍，见 kRampFp32MaxAbs 注释）";
}

// CV profile 的验收：min/opt/max = 1/8/16（历史工程口径）。
//
// 两件事都要验，缺一不可：
//   1. 范围内的 batch **真的能跑**（只 SetInputShape 就宣布"可用"是假证据）；
//   2. 范围外的必须**显式失败**——"能跑但用错形状"是最坏结果（Phase 1.5 的 E3.3 定下的纪律）。
TEST(ResNet18OnnxProfileTest, AcceptsBatchRangeAndRejectsOutOfRange) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    Logger logger;
    EngineBuilder builder(logger, Fp32BuilderConfig());
    bool built_now = false;
    const std::string path = EnsureFp32Engine(&builder, &built_now);
    if (path.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/config.json 与 0_resnet18_onnx/resnet18.onnx";
    }
    Engine engine(path, logger);

    // 这里只验"形状可用性"，不比对数值——数值由 R2.1/R2.2 负责。
    const int32_t max_batch = 16;
    for (int32_t batch : {1, kBatch, max_batch}) {
        const std::vector<float> input = MakeRampInput(batch);  // 每次按目标 batch 生成，避免越界切片
        const std::vector<float> logits = RunCnnEngine(&engine, input, batch);
        EXPECT_EQ(logits.size(), static_cast<size_t>(batch) * kClasses)
            << "batch=" << batch << " 在 profile 范围内，必须能真跑";
    }

    // 超出 max_batch：必须被拒绝，而不是静默跑出结果
    EXPECT_FALSE(engine.SetInputShape("input", nvinfer1::Dims4{max_batch + 1, 3, 224, 224}))
        << "batch=" << (max_batch + 1) << " 超出 profile 上界，必须显式失败";
}

// R1.4：拿 **LLM 的 architecture 声明**去配 CV 图，必须被 I/O 契约校验拦下。
//
// 与 P4-2 的 R0.7（未登记子图名，host 可跑）不同：这条走到 I/O 名比对，需要先解析 ONNX
// （→ CUDA），所以只能在真机跑。它守的是"契约映射真的按 architecture 生效"——
// 若退化成"任何 architecture 都接受 input/output"，两条路的同名同义就名存实亡。
TEST(ResNet18OnnxBuildTest, RejectsLlmConfigForCnnGraph) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string onnx = FindOnnxPath();
    if (onnx.empty()) {
        GTEST_SKIP() << "需要 0_resnet18_onnx/resnet18.onnx";
    }
    // 临时目录：声明成 decoder_only（LLM 契约），但图是 ResNet18（input/output）
    const std::string tmp = "/tmp/mini_trt_llm_resnet18_llm_config";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    {
        std::ofstream out(tmp + "/config.json");
        out << R"({"model_type": "gpt2", "architecture": "decoder_only",
                   "hyper_params": {}, "weight_map": {}})";
    }
    Logger logger;
    EngineBuilder builder(logger, EngineBuilder::Config{});
    EXPECT_FALSE(builder.BuildFromOnnx(tmp, onnx, tmp + "/should_not_exist.engine", {}))
        << "decoder_only 的契约要求 input_ids/logits，ResNet18 图必须被拒";
    EXPECT_FALSE(std::filesystem::exists(tmp + "/should_not_exist.engine"));
    std::filesystem::remove_all(tmp);
}

}  // namespace mini_trt_llm
