#include "cv_test_support.hpp"
#include "diff_stats.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "mini_trt_llm/utils/json.hpp"
#include "test_gpu_guard.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <fstream>
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
using test_support::kCvChannels;
using test_support::kCvClasses;
using test_support::kCvSize;
using test_support::ReadF32File;

constexpr int32_t kBatch = 8;
const size_t kInputElements = static_cast<size_t>(kBatch) * kCvChannels * kCvSize * kCvSize;
const size_t kLogitsElements = static_cast<size_t>(kBatch) * kCvClasses;

// Q/DQ 图与 INT8 引擎的路径可用环境变量覆盖——用于**隔离实验**（例如只把权重量化粒度从
// per-channel 换成 per-tensor，其它一律不动，看是哪一个变量导致精度崩）。默认走正式产物。
// 覆盖时必须**同时**换引擎路径，否则会复用到上一次的引擎缓存（缓存只按路径区分，见 #19）。
std::string QdqOnnxPath() {
    const char* override = std::getenv("MINI_TRT_QDQ_ONNX");
    return override != nullptr ? std::string(override) : std::string("models/resnet18/resnet18_qdq.onnx");
}

std::string Int8EnginePath() {
    const char* override = std::getenv("MINI_TRT_INT8_ENGINE");
    return override != nullptr ? std::string(override)
                               : std::string("/tmp/mini_trt_llm_resnet18_qdq_int8.engine");
}

const char* const kFp32Engine = "/tmp/mini_trt_llm_resnet18_onnx_fp32.engine";
const char* const kNativeFp32Engine = "/tmp/mini_trt_llm_resnet18_native_fp32.engine";

// R2.6 的判据（见 docs/phase4_int8_plan.md §4）：**分输入集**，因为 INT8 的固有误差比 FP16 大得多。
//
// 出处（2026-09-26 实测，先测后定）：
//   · ramp 输入（合成、确定性、与 legacy 同年口径）：argmax 必须**全一致**；
//   · 真实图（calib_data 的归一化张量）：按 **top-1 一致率**判——脚本侧 fake-quant 预检在
//     64 张上是 6/64 不一致（90.6% 一致），所以"全一致"对 INT8 是不现实的要求。
// **判据（2026-09-26 实测后定稿）**：
//
// 1) **FP32 有余量的子集**（`top1 − top2 ≥ kConfidentMargin`）是唯一有判别力的指标：
//    实测 per-tensor 方案下 12/12 = 100% 一致（256 张里满足该条件的样本）——阈值取 **≥ 90%**
//    （留 1 个样本余量）。依据：这批图 FP32 自身摇摆（margin 中位数 1.45、55% < 2），
//    "全体一致率"主要在测测试集噪声，见 TROUBLESHOOTING §29.4。
// 2) **全体系数只做"没崩坏"的下界**：实测 per-tensor 37.9%、per-channel 25.0% →
//    取 **≥ 30%** 作为下界，**不作为质量指标**（换一批图这个数会变）。
// 3) **ramp 不做一致性断言**（判据已作废，见本文件里那条用例的注释）。
constexpr float kConfidentMargin = 5.0f;
constexpr double kConfidentAgreementFloor = 0.90;
constexpr double kOverallAgreementFloor = 0.30;

// 分层区间与桶数：从用例体内提到文件作用域，是为了让"dump + 交叉校验"那条用例与
// 精度用例共用同一份口径——两处各写一份边界值正是本项目反复吃过的漂移来源。
// 边界取实测 margin 分布的经验值（见 TROUBLESHOOTING §29.4）。
constexpr float kBucketEdges[] = {1.0f, 2.0f, 5.0f, 10.0f, 1e30f};
constexpr int32_t kNumBuckets = 5;

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

std::string FindQdqOnnx() {
    const std::string path = QdqOnnxPath();
    return FindFile({path, "../" + path, "../../" + path, "../../../" + path});
}

EngineBuilder::Config Int8Config() {
    EngineBuilder::Config config;
    config.precision = Precision::INT8;
    // 必须在**建引擎时**就打开——引擎建完后再设没有用，而且默认 verbosity 下逐层精度读不出来，
    // 会让人误判成"没跑 INT8"（TROUBLESHOOTING #27 与 phase4_int8_plan §1.3 的 S3）。
    config.detailed_profiling = true;
    return config;
}

EngineBuilder::Config Fp32Config() {
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    return config;
}

// 建（或用缓存）Q/DQ 的 INT8 引擎。
// 这里**不自己判断"文件在不在"**：缓存的复用/失效由 `EngineBuilder` 的构建指纹统一决定
// （见 core/engine_cache.hpp 与 docs/TROUBLESHOOTING.md #34）——测试各自判断存在性正是
// "缓存不随代码失效"那个老坑的来源。
void EnsureInt8Engine(EngineBuilder* builder, const std::string& model_dir,
                      const std::string& qdq_onnx) {
    ASSERT_TRUE(builder->BuildFromOnnx(model_dir, qdq_onnx, Int8EnginePath(), {}))
        << "对称 Q/DQ 图应当能被解析并建成引擎";
}

void EnsureFp32Engine(EngineBuilder* builder, const std::string& model_dir,
                      const std::string& onnx) {
    ASSERT_TRUE(builder->BuildFromOnnx(model_dir, onnx, kFp32Engine, {}));
}

// 跑一批（batch = 输入张量的批大小）并返回 logits。
std::vector<float> RunBatch(Engine* engine, const std::vector<float>& input, int32_t batch) {
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
        !engine->SetInputShape("input", nvinfer1::Dims4{batch, kCvChannels, kCvSize, kCvSize}) ||
        !engine->SetTensorAddress("input", d_input.data()) ||
        !engine->SetTensorAddress("output", d_output.data())) {
        return {};
    }
    if (!engine->Enqueue(nullptr)) {
        MINI_TRT_LOG_ERROR("ResNet18 INT8 enqueue failed");
        return {};
    }
    engine->Synchronize(nullptr);
    return test_support::ReadFloats(d_output.data(), output_count);
}

// 统计引擎逐层信息里的 INT8 证据。
struct LayerPrecisionStats {
    int32_t total_layers = 0;
    int32_t int8_tensors = 0;   // 出现 "Format/Datatype: Int8" 的层数
    int32_t int8_tactics = 0;   // tactic 名带 "i8i8"（INT8 隐式 GEMM）的层数
};

LayerPrecisionStats InspectEngine(Engine* engine) {
    LayerPrecisionStats stats;
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        return stats;
    }
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(cuda->createEngineInspector());
    if (inspector == nullptr) {
        return stats;
    }
    stats.total_layers = cuda->getNbLayers();
    for (int32_t i = 0; i < stats.total_layers; ++i) {
        const char* line =
            inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kONELINE);
        if (line == nullptr) {
            continue;
        }
        const std::string text(line);
        // 注意：**没有 `[I8]` 这种标签**——按标签判会把"确实跑了 INT8"误判成"没跑"。
        if (text.find("Format/Datatype: Int8") != std::string::npos) {
            ++stats.int8_tensors;
        }
        if (text.find("i8i8") != std::string::npos) {
            ++stats.int8_tactics;
        }
    }
    return stats;
}

}  // namespace

// ---------------------------------------------------------------------------
// P4-7-2 余项：**用真实 ResNet18 的 Q/DQ 图建引擎，并证明它真的在跑 INT8**
//
// 为什么必须验：Q/DQ 是显式类型约束，**不受我们传的 precision 影响**——同一张 QDQ 图在
// precision=FP32 与 INT8 两种配置下都跑 INT8 卷积（phase4_int8_plan §1.3 的实测）。
// 所以"引擎是不是 INT8"只能看层信息，不能看我们传了什么。对照组用**非 QDQ 的 FP32 引擎**。
// ---------------------------------------------------------------------------

TEST(ResNet18Int8EngineTest, IsActuallyInt8) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    const std::string qdq = FindQdqOnnx();
    if (model_dir.empty() || onnx.empty() || qdq.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/resnet18_qdq.onnx（先跑 quantize_resnet18.py）";
    }
    Logger logger;
    EngineBuilder builder(logger, Int8Config());
    EnsureInt8Engine(&builder, model_dir, qdq);
    Engine int8_engine(Int8EnginePath(), logger);
    const LayerPrecisionStats int8_stats = InspectEngine(&int8_engine);
    std::cout << "[ResNet18 INT8] QDQ 引擎: 层数=" << int8_stats.total_layers
              << "  含 Int8 张量的层=" << int8_stats.int8_tensors
              << "  i8i8 tactic 的层=" << int8_stats.int8_tactics << "\n";

    // 20 个卷积都该落在 Int8 上（Q/DQ 与 Conv 融合后一层一个卷积）
    EXPECT_GE(int8_stats.int8_tensors, 20) << "层信息里 Int8 张量太少，说明没真的量化执行";
    EXPECT_GE(int8_stats.int8_tactics, 1) << "没有任何 INT8 隐式 GEMM 的 tactic";

    // 对照组：非 QDQ 的 FP32 引擎不该出现 Int8 张量
    EngineBuilder fp32_builder(logger, Fp32Config());
    EnsureFp32Engine(&fp32_builder, model_dir, onnx);
    Engine fp32_engine(kFp32Engine, logger);
    const LayerPrecisionStats fp32_stats = InspectEngine(&fp32_engine);
    std::cout << "[ResNet18 INT8] 对照 FP32 引擎: 含 Int8 张量的层="
              << fp32_stats.int8_tensors << "\n";
    EXPECT_EQ(fp32_stats.int8_tensors, 0) << "FP32 引擎里出现 Int8 张量，对照组失效";
}

// ---------------------------------------------------------------------------
// R2.6a：ramp 输入（合成、确定性）—— 与 legacy 同年口径，argmax 必须全一致
// ---------------------------------------------------------------------------

// ramp 输入：**不做 argmax 一致性断言**（判据本身已作废）。
//
// 为什么作废：ramp 是**未归一化**的合成输入（值域 0..1），与标定分布（归一化真实图）完全不同，
// 而 INT8 的 scale 是**按标定集定死的** → 分布外输入会退化，退化的**程度取决于方案**：
// 实测 per-channel 权重下 argmax 8/8 不一致、per-tensor 下 0/8。所以它**不能当判据**
// （既不能要求"必须一致"，也不能假定"必然不一致"），只能记录实测值。
// FP16/FP32 没有"标定范围"约束，所以它们能用 ramp 判；INT8 不行。
// 详见 docs/TROUBLESHOOTING.md §29.3。
//
// 这条用例保留下来做两件事：① 对照组——FP32 引擎在 ramp 上必须仍与基线一致；
// ② 记录 INT8 在分布外输入上的退化量（供将来决定"要不要对输入做分布检查"）。
TEST(ResNet18Int8AccuracyTest, RampInputIsOutOfDistribution) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    const std::string qdq = FindQdqOnnx();
    const std::string baseline_dir = FindBaselineDir();
    if (model_dir.empty() || onnx.empty() || qdq.empty() || baseline_dir.empty()) {
        GTEST_SKIP() << "需要 models/resnet18（QDQ 图 + P4-1 基线）与 resnet18.onnx";
    }
    Logger logger;
    EngineBuilder int8_builder(logger, Int8Config());
    EnsureInt8Engine(&int8_builder, model_dir, qdq);
    EngineBuilder fp32_builder(logger, Fp32Config());
    EnsureFp32Engine(&fp32_builder, model_dir, onnx);
    Engine int8_engine(Int8EnginePath(), logger);
    Engine fp32_engine(kFp32Engine, logger);

    const std::vector<float> input = ReadF32File(
        baseline_dir + "/inputs/ref_ramp_b8.contract_input.f32.bin", kInputElements);
    const std::vector<float> reference =
        ReadF32File(baseline_dir + "/ref_ramp_b8.bin", kLogitsElements);
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(reference.empty());

    const std::vector<float> fp32_logits = RunBatch(&fp32_engine, input, kBatch);
    const std::vector<float> int8_logits = RunBatch(&int8_engine, input, kBatch);
    ASSERT_EQ(fp32_logits.size(), kLogitsElements);
    ASSERT_EQ(int8_logits.size(), kLogitsElements);

    // 先把三条对拍都打出来，便于"阈值凭什么定"有据可查
    const DiffStats int8_vs_fp32 = ComputeDiffStats(fp32_logits, int8_logits);
    const DiffStats fp32_vs_base = ComputeDiffStats(reference, fp32_logits);
    int32_t int8_mismatches = 0;
    int32_t fp32_mismatches = 0;
    for (int32_t b = 0; b < kBatch; ++b) {
        int8_mismatches += (ArgmaxOfRow(fp32_logits, b) != ArgmaxOfRow(int8_logits, b)) ? 1 : 0;
        fp32_mismatches += (ArgmaxOfRow(reference, b) != ArgmaxOfRow(fp32_logits, b)) ? 1 : 0;
    }
    std::cout << "[ResNet18 INT8] ramp（分布外）: INT8 vs FP32引擎 max_abs = "
              << int8_vs_fp32.max_abs << "，argmax 不一致 " << int8_mismatches << "/" << kBatch
              // 实测两种权重方案给出不同结果（per-channel 8/8 不一致、per-tensor 0/8），
              // 所以这里**只记录不断言**——既不能要求"必须一致"（分布外输入），
              // 也不能假定"必然不一致"（那只是某一种方案的表现）。
              << "（**只记录、不断言**）；\n[ResNet18 INT8]   对照 FP32 vs torchvision 基线 max_abs = "
              << fp32_vs_base.max_abs << "，argmax 不一致 " << fp32_mismatches << "/" << kBatch
              << "\n";
    // ① 对照：FP32 引擎在 ramp 上必须仍与基线一致（说明 ramp 本身没坏）
    EXPECT_EQ(fp32_mismatches, 0) << "FP32 引擎在 ramp 上对不上基线，问题不在 INT8";
    // ② INT8 在分布外输入上：只要求"能跑完并给出有限值"，不要求分类正确
    for (float v : int8_logits) {
        ASSERT_TRUE(std::isfinite(v)) << "分布外输入下出现了 NaN/Inf，这是真缺陷";
    }
}

// ---------------------------------------------------------------------------
// R2.6b：真实图（calib_data 的归一化张量）—— 按 top-1 一致率判
// ---------------------------------------------------------------------------

TEST(ResNet18Int8AccuracyTest, Top1AgreementOnRealImages) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    const std::string qdq = FindQdqOnnx();
    if (model_dir.empty() || onnx.empty() || qdq.empty()) {
        GTEST_SKIP() << "需要 models/resnet18/resnet18_qdq.onnx";
    }
    // calib_data 只在本地存在（.gitignore 忽略），缺了就以 77 跳过
    const std::string calib_root =
        FindFile({"0_resnet18_onnx/calib_data", "../0_resnet18_onnx/calib_data",
                  "../../0_resnet18_onnx/calib_data", "../../../0_resnet18_onnx/calib_data"});
    if (calib_root.empty()) {
        GTEST_SKIP() << "需要 0_resnet18_onnx/calib_data";
    }
    Logger logger;
    EngineBuilder int8_builder(logger, Int8Config());
    EnsureInt8Engine(&int8_builder, model_dir, qdq);
    EngineBuilder fp32_builder(logger, Fp32Config());
    EnsureFp32Engine(&fp32_builder, model_dir, onnx);
    Engine int8_engine(Int8EnginePath(), logger);
    Engine fp32_engine(kFp32Engine, logger);

    // 取前 64 张，按 batch=8 分块跑——profile 的 max_batch 是 16，不能一次喂 64。
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(calib_root)) {
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    // 样本量：整体一致率需要足够多的样本才有意义，而"余量子集"更稀缺（64 张时只有 11 张）。
    const int32_t total = std::min<int32_t>(256, static_cast<int32_t>(files.size()));
    ASSERT_GE(total, kBatch);

    int32_t agreed = 0;
    int32_t confident_total = 0;   // FP32 有余量的样本
    int32_t confident_agreed = 0;
    float max_abs = 0.0f;
    // 按"FP32 的判别余量"分层统计一致性：用来回答"整体一致率为什么不是 100%"。
    // 若一致率只在小余量区间塌陷、在余量大的区间接近 100%，就说明是**指标被测试集限制**，
    // 而不是 INT8 破坏了模型。边界与桶数见文件作用域的 kBucketEdges / kNumBuckets。
    int32_t bucket_total[kNumBuckets] = {0, 0, 0, 0, 0};
    int32_t bucket_agreed[kNumBuckets] = {0, 0, 0, 0, 0};
    for (int32_t start = 0; start + kBatch <= total; start += kBatch) {
        std::vector<float> batch_input(kInputElements);
        for (int32_t i = 0; i < kBatch; ++i) {
            const std::vector<float> image =
                ReadF32File(files[static_cast<size_t>(start + i)],
                            static_cast<size_t>(kCvChannels) * kCvSize * kCvSize);
            ASSERT_EQ(image.size(), static_cast<size_t>(kCvChannels) * kCvSize * kCvSize);
            std::copy(image.begin(), image.end(),
                      batch_input.begin() + static_cast<ptrdiff_t>(i) *
                                                static_cast<ptrdiff_t>(kCvChannels * kCvSize * kCvSize));
        }
        const std::vector<float> fp32_logits = RunBatch(&fp32_engine, batch_input, kBatch);
        const std::vector<float> int8_logits = RunBatch(&int8_engine, batch_input, kBatch);
        ASSERT_EQ(int8_logits.size(), kLogitsElements);
        const DiffStats stats = ComputeDiffStats(fp32_logits, int8_logits);
        max_abs = std::max(max_abs, stats.max_abs);
        for (int32_t i = 0; i < kBatch; ++i) {
            agreed += (ArgmaxOfRow(fp32_logits, i) == ArgmaxOfRow(int8_logits, i)) ? 1 : 0;
            // **分层**：这批图是 tiny-imagenet 的 64×64 放大件，FP32 自己就摇摆——
            // 实测 top1−top2 的中位数只有 1.45、55% 的样本 margin < 2，而 INT8 对 logits 的
            // 扰动是 O(1~23)。所以"全体一致率"主要在测测试集的噪声。只在 FP32 **有余量**
            // 的样本上比，才反映 INT8 自身的质量（依据见 TROUBLESHOOTING §29.4）。
            const float* row = fp32_logits.data() + static_cast<size_t>(i) * kCvClasses;
            float best = row[0];
            float second = -1e30f;
            for (int32_t c = 1; c < kCvClasses; ++c) {
                if (row[c] > best) {
                    second = best;
                    best = row[c];
                } else if (row[c] > second) {
                    second = row[c];
                }
            }
            if (best - second >= kConfidentMargin) {
                ++confident_total;
                confident_agreed +=
                    (ArgmaxOfRow(fp32_logits, i) == ArgmaxOfRow(int8_logits, i)) ? 1 : 0;
            }
            const float margin = best - second;
            const bool same = ArgmaxOfRow(fp32_logits, i) == ArgmaxOfRow(int8_logits, i);
            for (int32_t b = 0; b < kNumBuckets; ++b) {
                if (margin < kBucketEdges[b]) {
                    ++bucket_total[b];
                    bucket_agreed[b] += same ? 1 : 0;
                    break;
                }
            }
        }
    }
    const double rate = static_cast<double>(agreed) / total;
    const double confident_rate =
        confident_total > 0 ? static_cast<double>(confident_agreed) / confident_total : 0.0;
    std::cout << "[ResNet18 INT8] 真实图 " << total << " 张：top-1 一致 " << agreed << "/"
              << total << " = " << rate * 100.0 << "%，max_abs(vs FP32 引擎) = " << max_abs
              << "\n[ResNet18 INT8]   └ FP32 有余量(margin≥" << kConfidentMargin << ") 的子集："
              << confident_agreed << "/" << confident_total << " = " << confident_rate * 100.0
              << "%\n";
    // **这条交叉统计就是"整体一致率为什么不是 100%"的答案**
    for (int32_t b = 0; b < kNumBuckets; ++b) {
        if (bucket_total[b] == 0) {
            continue;
        }
        const double r = 100.0 * bucket_agreed[b] / bucket_total[b];
        std::cout << "[ResNet18 INT8]   按 FP32 余量分层：margin<" << kBucketEdges[b]
                  << " → 一致 " << bucket_agreed[b] << "/" << bucket_total[b] << " = " << r
                  << "%\n";
    }
    EXPECT_GE(rate, kOverallAgreementFloor)
        << "整体一致率只是'没崩坏'的下界，不是质量指标";
    // **主判据**：FP32 有余量的子集上必须高度一致（实测 100%）
    ASSERT_GT(confident_total, 0) << "没有任何有余量的样本——这批图不适合作精度判据";
    EXPECT_GE(confident_rate, kConfidentAgreementFloor)
        << "FP32 有余量的样本上仍然不一致，说明 INT8 真的在破坏分类";
}

// ---------------------------------------------------------------------------
// A2-5：把两个引擎的 logits 落盘 + 写出 C++ 侧的统计，供 Python 侧交叉校验。
//
// 为什么要这条：`tools/validate/int8_eval.py` 是"判据规格"的第二种实现，
// 而**两条独立实现对同一批数据必须给出同一组数字**（余量子集的 n 与分子）。
// 数值不一致就说明口径漂移——那时要查口径，不是改阈值。
//
// 为什么不做断言：本用例只负责**产出物**（logits / manifest / meta / cpp_report），
// 正确性判据仍由上面那条精度用例负责；交叉校验由 `ctest -R int8_crosscheck` 完成。
// 因此这里必须断言"文件确实写出来了"——否则一次静默失败会伪装成"跑过了"。
//
// **注意这批数据本身就与标定集同源**（`calib_data` 既是标定集又是当前的测试图），
// 所以跑交叉校验时必须用 `--legacy-mode`：脚本会在报告里写明"不满足 §1.6 规格、
// 一致率会被高估，仅用于历史口径交叉校验"。默认（严格）路径仍然会拒绝这种输入。
// ---------------------------------------------------------------------------
TEST(ResNet18Int8AccuracyTest, DumpsLogitsAndCppReportForCrossCheck) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindModelDir();
    const std::string onnx = FindOnnxPath();
    const std::string qdq = FindQdqOnnx();
    if (model_dir.empty() || onnx.empty() || qdq.empty()) {
        GTEST_SKIP() << "需要 models/resnet18 的 onnx / qdq 产物";
    }
    const std::string calib_root =
        FindFile({"0_resnet18_onnx/calib_data", "../0_resnet18_onnx/calib_data",
                  "../../0_resnet18_onnx/calib_data", "../../../0_resnet18_onnx/calib_data"});
    if (calib_root.empty()) {
        GTEST_SKIP() << "需要 0_resnet18_onnx/calib_data";
    }

    Logger logger;
    EngineBuilder int8_builder(logger, Int8Config());
    EnsureInt8Engine(&int8_builder, model_dir, qdq);
    EngineBuilder fp32_builder(logger, Fp32Config());
    EnsureFp32Engine(&fp32_builder, model_dir, onnx);
    Engine int8_engine(Int8EnginePath(), logger);
    Engine fp32_engine(kFp32Engine, logger);

    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(calib_root)) {
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    const int32_t total = std::min<int32_t>(256, static_cast<int32_t>(files.size()));
    ASSERT_GE(total, kBatch);
    const int32_t usable = (total / kBatch) * kBatch;  // 只跑整 batch，与精度用例同口径

    std::vector<float> fp32_all;
    std::vector<float> int8_all;
    fp32_all.reserve(static_cast<size_t>(usable) * kCvClasses);
    int8_all.reserve(static_cast<size_t>(usable) * kCvClasses);

    int32_t agreed = 0;
    int32_t confident_total = 0;
    int32_t confident_agreed = 0;
    int32_t bucket_total[kNumBuckets] = {0, 0, 0, 0, 0};
    int32_t bucket_agreed[kNumBuckets] = {0, 0, 0, 0, 0};
    float max_abs = 0.0f;

    for (int32_t start = 0; start + kBatch <= usable; start += kBatch) {
        std::vector<float> batch_input(kInputElements);
        for (int32_t i = 0; i < kBatch; ++i) {
            const std::vector<float> image =
                ReadF32File(files[static_cast<size_t>(start + i)],
                            static_cast<size_t>(kCvChannels) * kCvSize * kCvSize);
            ASSERT_EQ(image.size(), static_cast<size_t>(kCvChannels) * kCvSize * kCvSize);
            std::copy(image.begin(), image.end(),
                      batch_input.begin() + static_cast<ptrdiff_t>(i) *
                                                static_cast<ptrdiff_t>(kCvChannels * kCvSize * kCvSize));
        }
        const std::vector<float> fp32_logits = RunBatch(&fp32_engine, batch_input, kBatch);
        const std::vector<float> int8_logits = RunBatch(&int8_engine, batch_input, kBatch);
        ASSERT_EQ(fp32_logits.size(), kLogitsElements);
        ASSERT_EQ(int8_logits.size(), kLogitsElements);
        max_abs = std::max(max_abs, ComputeDiffStats(fp32_logits, int8_logits).max_abs);

        fp32_all.insert(fp32_all.end(), fp32_logits.begin(), fp32_logits.end());
        int8_all.insert(int8_all.end(), int8_logits.begin(), int8_logits.end());

        for (int32_t i = 0; i < kBatch; ++i) {
            const bool same = ArgmaxOfRow(fp32_logits, i) == ArgmaxOfRow(int8_logits, i);
            agreed += same ? 1 : 0;
            const float* row = fp32_logits.data() + static_cast<size_t>(i) * kCvClasses;
            float best = row[0];
            float second = -1e30f;
            for (int32_t c = 1; c < kCvClasses; ++c) {
                if (row[c] > best) {
                    second = best;
                    best = row[c];
                } else if (row[c] > second) {
                    second = row[c];
                }
            }
            const float margin = best - second;
            if (margin >= kConfidentMargin) {
                ++confident_total;
                confident_agreed += same ? 1 : 0;
            }
            for (int32_t b = 0; b < kNumBuckets; ++b) {
                if (margin < kBucketEdges[b]) {
                    ++bucket_total[b];
                    bucket_agreed[b] += same ? 1 : 0;
                    break;
                }
            }
        }
    }
    ASSERT_EQ(static_cast<int32_t>(fp32_all.size()), usable * kCvClasses);

    const std::filesystem::path out_dir =
        std::filesystem::temp_directory_path() / "mini_trt_llm_int8_crosscheck";
    std::filesystem::create_directories(out_dir);
    const std::string fp32_path = (out_dir / "fp32.f32.bin").string();
    const std::string int8_path = (out_dir / "int8.f32.bin").string();
    const std::string manifest_path = (out_dir / "val_manifest.json").string();
    const std::string meta_path = (out_dir / "meta.json").string();
    const std::string report_path = (out_dir / "cpp_report.json").string();

    const auto write_floats = [](const std::string& path, const std::vector<float>& values) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    };
    write_floats(fp32_path, fp32_all);
    write_floats(int8_path, int8_all);

    // manifest 只给文件路径：sha256 由脚本在 legacy 模式下自己算（C++ 侧没有哈希实现，
    // 而"要求提供哈希却不核对"比"明确说由消费方计算"更糟）。
    JsonValue::Array manifest;
    for (int32_t i = 0; i < usable; ++i) {
        JsonValue::Object entry;
        entry["file"] = JsonValue(std::filesystem::absolute(files[static_cast<size_t>(i)]).string());
        manifest.emplace_back(entry);
    }
    JsonValue::Object meta;
    {
        JsonValue::Object validation;
        validation["name"] = JsonValue("resnet18-calib-images-legacy");
        validation["version"] = JsonValue("2026-09-26");
        validation["num_samples"] = JsonValue(usable);
        validation["manifest_path"] = JsonValue(manifest_path);
        validation["source"] = JsonValue(JsonValue::Object{
            {"url", JsonValue("local: 0_resnet18_onnx/calib_data")},
            {"retrieved_utc", JsonValue("n/a")},
            {"license", JsonValue("n/a")}});
        validation["preprocessing"] = JsonValue(JsonValue::Object{
            {"resize", JsonValue(JsonValue::Array{JsonValue(224), JsonValue(224)})},
            {"layout", JsonValue("NCHW")},
            {"dtype", JsonValue("float32")},
            {"mean", JsonValue(JsonValue::Array{JsonValue(0.485), JsonValue(0.456), JsonValue(0.406)})},
            {"std", JsonValue(JsonValue::Array{JsonValue(0.229), JsonValue(0.224), JsonValue(0.225)})}});
        meta["validation_set"] = JsonValue(validation);

        JsonValue::Object calibration;
        calibration["dir"] = JsonValue(calib_root);
        calibration["num_samples"] = JsonValue(static_cast<int>(files.size()));
        meta["calibration_set"] = JsonValue(calibration);
        meta["labels"] = JsonValue(JsonValue::Object{
            {"num_classes", JsonValue(kCvClasses)},
            {"source", JsonValue("n/a（legacy 模式：无真值标签）")}});
    }
    JsonValue::Object report;
    {
        JsonValue::Object thresholds;
        thresholds["confident_margin"] = JsonValue(static_cast<double>(kConfidentMargin));
        thresholds["bucket_edges"] = JsonValue(JsonValue::Array{
            JsonValue(1.0), JsonValue(2.0), JsonValue(5.0), JsonValue(10.0)});
        thresholds["provenance"] = JsonValue("tests/test_resnet18_int8.cpp + docs/phase4_int8_plan.md §4");
        report["thresholds"] = JsonValue(thresholds);

        JsonValue::Object overall;
        overall["n"] = JsonValue(usable);
        overall["agree"] = JsonValue(agreed);
        overall["agree_rate"] = JsonValue(static_cast<double>(agreed) / usable);
        report["overall"] = JsonValue(overall);

        JsonValue::Object confident;
        confident["n"] = JsonValue(confident_total);
        confident["agree"] = JsonValue(confident_agreed);
        confident["agree_rate"] = confident_total > 0
                                      ? JsonValue(static_cast<double>(confident_agreed) / confident_total)
                                      : JsonValue();
        confident["margin_min"] = JsonValue(static_cast<double>(kConfidentMargin));
        report["confident"] = JsonValue(confident);

        JsonValue::Array strata;
        float lower = -1e30f;
        for (int32_t b = 0; b < kNumBuckets; ++b) {
            JsonValue::Object stratum;
            stratum["bucket"] = JsonValue(std::string("bucket") + std::to_string(b));
            stratum["lo"] = (b == 0) ? JsonValue() : JsonValue(static_cast<double>(lower));
            stratum["hi"] = (kBucketEdges[b] > 1e29f) ? JsonValue() : JsonValue(static_cast<double>(kBucketEdges[b]));
            stratum["n"] = JsonValue(bucket_total[b]);
            stratum["agree"] = JsonValue(bucket_agreed[b]);
            stratum["agree_rate"] = bucket_total[b] > 0
                                        ? JsonValue(static_cast<double>(bucket_agreed[b]) / bucket_total[b])
                                        : JsonValue();
            strata.emplace_back(stratum);
            lower = kBucketEdges[b];
        }
        report["strata"] = JsonValue(strata);
        report["max_abs"] = JsonValue(static_cast<double>(max_abs));
    }

    const auto write_text = [](const std::string& path, const std::string& text) {
        std::ofstream out(path);
        out << text;
    };
    write_text(manifest_path, JsonValue(manifest).Dump(1));
    write_text(meta_path, JsonValue(meta).Dump(1));
    write_text(report_path, JsonValue(report).Dump(2));

    // 断言"确实写出来了"（大小 > 0）——静默失败不能伪装成跑过了。
    for (const std::string& path : {fp32_path, int8_path, manifest_path, meta_path, report_path}) {
        ASSERT_TRUE(std::filesystem::exists(path)) << "没写出 " << path;
        ASSERT_GT(std::filesystem::file_size(path), 0u) << "写出的文件是空的：" << path;
    }

    std::cout << "[ResNet18 INT8 交叉校验] 产物目录：" << out_dir.string() << "\n"
              << "        C++ 侧：整体 " << agreed << "/" << usable
              << "、余量子集 " << confident_agreed << "/" << confident_total << "、max_abs=" << max_abs
              << "\n        下一步（在真机执行，顺序不能反）：\n"
              << "          python3 mini_trt_llm/tools/validate/int8_eval.py \\\n"
              << "            --fp32-logits " << fp32_path << " \\\n"
              << "            --int8-logits " << int8_path << " \\\n"
              << "            --meta " << meta_path << " --calib-dir " << calib_root
              << " --legacy-mode --json-out " << (out_dir / "py_report.json").string() << "\n"
              << "          ctest --test-dir build -R int8_crosscheck --output-on-failure\n";
}

}  // namespace mini_trt_llm
