#include "gpt2_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_runtime.h>
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

constexpr int32_t kRealVocab = 50257;
constexpr int32_t kRealLayers = 12;
constexpr int32_t kRealHeads = 12;
constexpr int32_t kRealHeadSize = 64;
constexpr int32_t kRealBlockSize = 16;
constexpr int32_t kRealPositions = 1024;

// `1_gpt2_onnx/ref_output.bin` 是 HF FP32 对该 prompt 的 logits，形状 [1,4,50257]。
// 已核对过：它与 `models/gpt2/model.safetensors` 的权重**逐比特可复现**（§0.1）。
const std::vector<int64_t> kPrompt = {464, 2068, 7586, 21831};  // "The quick brown fox"
constexpr int32_t kSeq = 4;

// 在若干候选位置里找仓库根下的参考文件。
std::string FindRefOutput() {
    const char* candidates[] = {"1_gpt2_onnx/ref_output.bin",
                                "../1_gpt2_onnx/ref_output.bin",
                                "../../1_gpt2_onnx/ref_output.bin",
                                "../../../1_gpt2_onnx/ref_output.bin"};
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

std::string FindRealModelDir() {
    const char* candidates[] = {"models/gpt2", "../models/gpt2", "../../models/gpt2",
                                "../../../models/gpt2"};
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(std::string(candidate) + "/config.json")) {
            return candidate;
        }
    }
    return {};
}

// 余弦相似度：与量级无关，是"分布形状是否一致"的判据，
// 比逐元素相对误差更不容易被个别小分量带偏（D6 选它的理由）。
double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

}  // namespace

// P2-3：真实 GPT-2 的 prefill logits 与 HF FP32 参考对拍。
//
// 为什么还需要它（P2-8 的"8 个贪心 token 全中"已经间接支持了正确性）：
// token 只反映 argmax，argmax 一致并不排除 logits 整体有系统性偏移——
// 那会影响后续采样（Top-K/Top-P）的行为，也会掩盖"某个算子在弱信号上算错"。
TEST(Gpt2PrefillAccuracyTest, RealGpt2LogitsMatchReference) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindRealModelDir();
    const std::string ref_path = FindRefOutput();
    if (dir.empty() || ref_path.empty()) {
        GTEST_SKIP() << "需要 models/gpt2 与 1_gpt2_onnx/ref_output.bin";
    }

    // 参考数据先读进来并校验大小：形状不对就没必要浪费一次引擎构建（分钟级）。
    const std::vector<char> ref_bytes = ReadFile(ref_path);
    const size_t expected_bytes =
        static_cast<size_t>(kSeq) * kRealVocab * sizeof(float);
    ASSERT_EQ(ref_bytes.size(), expected_bytes)
        << "ref_output.bin 大小与 [1,4,50257] FP32 不符";
    const auto* ref = reinterpret_cast<const float*>(ref_bytes.data());

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    config.min_prefill_batch = 1;
    config.opt_prefill_batch = 1;
    config.max_prefill_batch = 1;
    config.min_prefill_seq_len = kSeq;
    config.opt_prefill_seq_len = kSeq;
    config.max_prefill_seq_len = kSeq;
    EngineBuilder builder(logger, config);
    // kSingle：只要能出 logits，不需要导出 K/V，绑定最简单
    const std::string engine_path = "/tmp/mini_trt_llm_gpt2_accuracy.engine";
    ASSERT_TRUE(builder.BuildFromConfig(dir, engine_path, BuildStage::kSingle))
        << "真实 GPT-2 engine 构建失败";
    Engine engine(engine_path, logger);

    std::vector<int32_t> tokens(kPrompt.begin(), kPrompt.end());
    std::vector<int32_t> positions(kSeq);
    for (int32_t i = 0; i < kSeq; ++i) {
        positions[static_cast<size_t>(i)] = i;
    }
    const size_t logits_count = static_cast<size_t>(kSeq) * kRealVocab;
    DeviceBuffer d_tokens(kSeq * sizeof(int32_t));
    DeviceBuffer d_positions(kSeq * sizeof(int32_t));
    DeviceBuffer d_logits(logits_count * sizeof(float));
    ASSERT_TRUE(d_tokens.Allocate(kSeq * sizeof(int32_t)));
    ASSERT_TRUE(d_positions.Allocate(kSeq * sizeof(int32_t)));
    ASSERT_TRUE(d_logits.Allocate(logits_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokens.data(), tokens.data(), d_tokens.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions.data(), positions.data(), d_positions.size(),
                          cudaMemcpyHostToDevice));

    ASSERT_TRUE(engine.SetOptimizationProfile(0, nullptr));
    ASSERT_TRUE(engine.SetInputShape("input_ids", nvinfer1::Dims{2, {1, kSeq}}));
    ASSERT_TRUE(engine.SetInputShape("position_ids", nvinfer1::Dims{2, {1, kSeq}}));
    ASSERT_TRUE(engine.SetTensorAddress("input_ids", d_tokens.data()));
    ASSERT_TRUE(engine.SetTensorAddress("position_ids", d_positions.data()));
    ASSERT_TRUE(engine.SetTensorAddress("logits", d_logits.data()));
    ASSERT_TRUE(engine.Enqueue(nullptr));
    engine.Synchronize(nullptr);

    std::vector<float> actual(logits_count);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_logits.data(), logits_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const std::vector<float> reference(ref, ref + logits_count);

    float max_abs_ref = 0.0f;
    for (float value : reference) {
        max_abs_ref = std::max(max_abs_ref, std::fabs(value));
    }
    const DiffStats diff = ComputeDiffStats(reference, actual);
    const double cosine = CosineSimilarity(reference, actual);
    const float relative = diff.max_abs / (max_abs_ref > 0.0f ? max_abs_ref : 1.0f);

    // 把实测值打出来：D6 的阈值是首次运行前定的，这里是"实测收敛"的输入数据。
    std::cout << "[诊断] 真实 GPT-2 prefill vs HF FP32 参考\n"
              << "        max_abs = " << diff.max_abs << "  max_rel = " << diff.max_rel
              << "  (max_abs / max|ref| = " << relative << ")\n"
              << "        cosine  = " << cosine << "  max|ref| = " << max_abs_ref << "\n";

    // 阈值口径源自 D6（FP32：cosine ≥ 0.9999 + 相对界 < 1e-3），并按**首次实测收紧**：
    // 2026-09-25 真机实测 max_abs = 9.92e-05、max_abs/max|ref| = 9.19e-07、cosine = 1.0。
    // 收紧到"实测值的约 10 倍余量"：既能吸收不同 kernel/编译器的差异，
    // 又比 D6 原文严 100 倍。**放宽必须走 AGENTS.md §7 的流程**（先量无关差异），
    // 不允许为了让它变绿而调这两个数字。
    EXPECT_GT(cosine, 0.999999);
    EXPECT_LT(relative, 1e-5f);

    // 逐位置的贪心 token 也必须与参考一致（logits 级对拍里最有语义的一条）
    for (int32_t row = 0; row < kSeq; ++row) {
        const float* ref_row = reference.data() + static_cast<size_t>(row) * kRealVocab;
        const float* got_row = actual.data() + static_cast<size_t>(row) * kRealVocab;
        const int32_t ref_argmax = static_cast<int32_t>(
            std::distance(ref_row, std::max_element(ref_row, ref_row + kRealVocab)));
        const int32_t got_argmax = static_cast<int32_t>(
            std::distance(got_row, std::max_element(got_row, got_row + kRealVocab)));
        EXPECT_EQ(ref_argmax, got_argmax) << "位置 " << row << " 的 argmax 不同";
    }
}

}  // namespace mini_trt_llm
