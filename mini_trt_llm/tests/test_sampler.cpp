#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/sampler/nucleus_cutoff.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "mini_trt_llm/utils/timer.hpp"
#include "sampler_test_support.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {
namespace {

// 固定 seed：Q12 要求参考结果可复现，采样用例必须由 seed 完全决定。
constexpr uint64_t kSeed = 42;

float DeterministicValue(int64_t index) {
    return std::sin(0.29f * static_cast<float>(index)) * 1.2f;
}

// 构造 [batch, vocab] 的 logits，并保证每行的最大值下标唯一（避免 argmax 并列歧义）。
std::vector<float> MakeLogits(int32_t batch_size, int32_t vocab_size,
                              std::vector<int32_t>* argmax_out) {
    std::vector<float> logits(static_cast<size_t>(batch_size) * vocab_size);
    argmax_out->assign(batch_size, 0);
    for (int32_t b = 0; b < batch_size; ++b) {
        int32_t best_index = 0;
        float best_value = -1e30f;
        for (int32_t v = 0; v < vocab_size; ++v) {
            // 行间用不同偏移，避免所有行选出同一个 token
            const float value =
                DeterministicValue(static_cast<int64_t>(b) * vocab_size + v) -
                0.001f * static_cast<float>(b);
            logits[static_cast<size_t>(b) * vocab_size + v] = value;
            if (value > best_value) {
                best_value = value;
                best_index = v;
            }
        }
        (*argmax_out)[b] = best_index;
    }
    return logits;
}

struct DeviceLogits {
    DeviceBuffer buffer;
    size_t bytes = 0;
};

DeviceLogits UploadLogits(const std::vector<float>& logits) {
    DeviceLogits uploaded;
    uploaded.bytes = logits.size() * sizeof(float);
    if (!uploaded.buffer.Allocate(uploaded.bytes)) {
        throw std::runtime_error("Sampler test: failed to allocate logits buffer");
    }
    CUDA_CHECK(cudaMemcpy(uploaded.buffer.data(), logits.data(), uploaded.bytes,
                          cudaMemcpyHostToDevice));
    return uploaded;
}

std::vector<int32_t> DownloadTokens(const DeviceBuffer& tokens, int32_t batch_size) {
    std::vector<int32_t> result(batch_size);
    CUDA_CHECK(cudaMemcpy(result.data(), tokens.data(), batch_size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    return result;
}

std::vector<int32_t> RunGreedy(const std::vector<float>& logits, int32_t batch_size,
                               int32_t vocab_size) {
    const DeviceLogits device_logits = UploadLogits(logits);
    DeviceBuffer tokens(batch_size * sizeof(int32_t));
    if (!tokens.Allocate(batch_size * sizeof(int32_t))) {
        throw std::runtime_error("Sampler test: failed to allocate token buffer");
    }

    SamplerArgs args;
    args.logits = device_logits.buffer.data();
    args.token_ids = static_cast<int32_t*>(tokens.data());
    args.batch_size = batch_size;
    args.vocab_size = vocab_size;
    args.is_half = false;
    CUDA_CHECK(LaunchGreedySampler(args, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadTokens(tokens, batch_size);
}

std::vector<int32_t> RunTopK(const std::vector<float>& logits, int32_t batch_size,
                             int32_t vocab_size, int32_t k, uint64_t seed,
                             uint64_t offset = 0) {
    const DeviceLogits device_logits = UploadLogits(logits);
    DeviceBuffer tokens(batch_size * sizeof(int32_t));
    const std::vector<int32_t> host_k(batch_size, k);
    DeviceBuffer device_k(batch_size * sizeof(int32_t));
    const size_t workspace_bytes = TopKSamplerWorkspaceBytes(batch_size, vocab_size);
    DeviceBuffer workspace(workspace_bytes);
    if (!tokens.Allocate(batch_size * sizeof(int32_t)) ||
        !device_k.Allocate(batch_size * sizeof(int32_t)) || !workspace.Allocate(workspace_bytes)) {
        throw std::runtime_error("Sampler test: failed to allocate buffers");
    }
    CUDA_CHECK(cudaMemcpy(device_k.data(), host_k.data(), batch_size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    TopKSamplerArgs args;
    args.logits = device_logits.buffer.data();
    args.token_ids = static_cast<int32_t*>(tokens.data());
    args.top_k = static_cast<const int32_t*>(device_k.data());
    args.batch_size = batch_size;
    args.vocab_size = vocab_size;
    args.is_half = false;
    args.seed = seed;
    args.offset = offset;
    CUDA_CHECK(LaunchTopKSampler(args, nullptr, workspace.data(), workspace_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadTokens(tokens, batch_size);
}

// 快速路径（LaunchTopKSamplerFast）：与 RunTopK 同输入，用于与旧路径逐 token 对拍。
std::vector<int32_t> RunTopKFast(const std::vector<float>& logits, int32_t batch_size,
                                 int32_t vocab_size, int32_t k, uint64_t seed,
                                 uint64_t offset = 0) {
    const DeviceLogits device_logits = UploadLogits(logits);
    DeviceBuffer tokens(batch_size * sizeof(int32_t));
    DeviceBuffer device_k(batch_size * sizeof(int32_t));
    const std::vector<int32_t> host_k(batch_size, k);
    if (!tokens.Allocate(batch_size * sizeof(int32_t)) ||
        !device_k.Allocate(batch_size * sizeof(int32_t))) {
        throw std::runtime_error("Sampler test: failed to allocate buffers");
    }
    CUDA_CHECK(cudaMemcpy(device_k.data(), host_k.data(), batch_size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    TopKSamplerArgs args;
    args.logits = device_logits.buffer.data();
    args.token_ids = static_cast<int32_t*>(tokens.data());
    args.top_k = static_cast<const int32_t*>(device_k.data());
    args.batch_size = batch_size;
    args.vocab_size = vocab_size;
    args.is_half = false;
    args.seed = seed;
    args.offset = offset;
    CUDA_CHECK(LaunchTopKSamplerFast(args, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadTokens(tokens, batch_size);
}

// Top-P：默认走生产路径（行内并行，P9_2-5）；`legacy = true` 时走旧的逐行串行实现，
// 仅用于对照（见测试计划 §9.4 的"允许差异"）。
std::vector<int32_t> RunTopP(const std::vector<float>& logits, int32_t batch_size,
                             int32_t vocab_size, float p, uint64_t seed,
                             uint64_t offset = 0, bool legacy = false) {
    const DeviceLogits device_logits = UploadLogits(logits);
    DeviceBuffer tokens(batch_size * sizeof(int32_t));
    const std::vector<float> host_p(batch_size, p);
    DeviceBuffer device_p(batch_size * sizeof(float));
    const size_t workspace_bytes = TopPSamplerWorkspaceBytes(batch_size, vocab_size);
    DeviceBuffer workspace(workspace_bytes);
    if (!tokens.Allocate(batch_size * sizeof(int32_t)) ||
        !device_p.Allocate(batch_size * sizeof(float)) || !workspace.Allocate(workspace_bytes)) {
        throw std::runtime_error("Sampler test: failed to allocate buffers");
    }
    CUDA_CHECK(cudaMemcpy(device_p.data(), host_p.data(), batch_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    TopPSamplerArgs args;
    args.logits = device_logits.buffer.data();
    args.token_ids = static_cast<int32_t*>(tokens.data());
    args.top_p = static_cast<const float*>(device_p.data());
    args.batch_size = batch_size;
    args.vocab_size = vocab_size;
    args.is_half = false;
    args.seed = seed;
    args.offset = offset;
    if (legacy) {
        CUDA_CHECK(LaunchTopPSamplerLegacy(args, nullptr, workspace.data(), workspace_bytes));
    } else {
        CUDA_CHECK(LaunchTopPSampler(args, nullptr, workspace.data(), workspace_bytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadTokens(tokens, batch_size);
}

// 统计多次采样的词频。fixed seed + 递增 offset 等价于"同一随机流的连续抽样"，
// 因此结果完全可复现，又足以做分布检验（D3 要求验证采样分布而非逐 token 一致）。
std::vector<int32_t> CollectTokenCounts(const std::vector<float>& logits,
                                        int32_t vocab_size, int32_t k, int32_t draws,
                                        bool use_top_p, float p, bool legacy = false) {
    std::vector<int32_t> counts(vocab_size, 0);
    for (int32_t draw = 0; draw < draws; ++draw) {
        const uint64_t offset = static_cast<uint64_t>(draw);
        const std::vector<int32_t> tokens =
            use_top_p ? RunTopP(logits, 1, vocab_size, p, kSeed, offset, legacy)
                      : RunTopK(logits, 1, vocab_size, k, kSeed, offset);
        ++counts[tokens[0]];
    }
    return counts;
}

std::vector<double> SoftmaxProbabilities(const std::vector<float>& row_logits) {
    double max_value = row_logits[0];
    for (float v : row_logits) {
        max_value = std::max(max_value, static_cast<double>(v));
    }
    std::vector<double> probabilities(row_logits.size());
    double total = 0.0;
    for (size_t i = 0; i < row_logits.size(); ++i) {
        probabilities[i] = std::exp(row_logits[i] - max_value);
        total += probabilities[i];
    }
    for (double& probability : probabilities) {
        probability /= total;
    }
    return probabilities;
}

// 按内核的同一分块口径算分块和（块内按下标升序串行累加，最后一块可能短）。
std::vector<float> ChunkSums(const std::vector<float>& values, int32_t chunk_size,
                             int32_t chunk_count) {
    std::vector<float> sums(static_cast<size_t>(chunk_count), 0.0f);
    const int32_t size = static_cast<int32_t>(values.size());
    for (int32_t c = 0; c < chunk_count; ++c) {
        const int32_t begin = c * chunk_size;
        const int32_t end = std::min(begin + chunk_size, size);
        float sum = 0.0f;
        for (int32_t i = begin; i < end; ++i) {
            sum += values[static_cast<size_t>(i)];
        }
        sums[static_cast<size_t>(c)] = sum;
    }
    return sums;
}

}  // namespace

TEST(SamplerTest, WorkspaceSizingIsPositiveAndRejectsInvalidDims) {
    EXPECT_GT(TopKSamplerWorkspaceBytes(4, 1024), 0u);
    EXPECT_GT(TopPSamplerWorkspaceBytes(4, 1024), 0u);
    EXPECT_EQ(TopKSamplerWorkspaceBytes(0, 1024), 0u);
    EXPECT_EQ(TopPSamplerWorkspaceBytes(4, 0), 0u);
}

TEST(SamplerTest, RejectsNullPointersAndWorkspace) {
    SamplerArgs greedy_args;
    EXPECT_NE(LaunchGreedySampler(greedy_args, nullptr), cudaSuccess);

    TopKSamplerArgs top_k_args;
    EXPECT_NE(LaunchTopKSampler(top_k_args, nullptr, nullptr, 0), cudaSuccess);

    TopPSamplerArgs top_p_args;
    EXPECT_NE(LaunchTopPSampler(top_p_args, nullptr, nullptr, 0), cudaSuccess);

    EXPECT_NE(LaunchTopPSamplerLegacy(top_p_args, nullptr, nullptr, 0), cudaSuccess);
}

// ---------------------------------------------------------------------------
// NucleusCutoffTest：`FindFirstPrefixCrossing` 的 host 裁决（沙箱可跑，S-22）。
//
// 为什么要有这一组：Top-P 的 kernel 在沙箱里跑不了（无 GPU），而"第一个累计 >= 阈值"的前缀
// 定位是整段代码里最容易出 off-by-one 的地方（多截/少截一个元素会让边界 token 永远采不到）。
// 被测对象与产品实现是**同一份代码**（`sampler/nucleus_cutoff.hpp`，__host__ __device__），
// 因此这组用例没有"参考实现漂移"的风险；但它也**不能**替代真机（内核装配仍需 GPU 验证）。
// ---------------------------------------------------------------------------

// 判据是 `>=`（包含等号）：阈值恰好等于某个前缀和时必须截到该元素。
// 写成 `>` 会让 nucleus 的最后一个 token 永远采不到（正是 S-9 锁的那类缺陷）。
TEST(NucleusCutoffTest, UsesInclusiveComparisonAtExactBoundary) {
    constexpr int32_t kChunkSize = 2;
    constexpr int32_t kChunkCount = 4;
    const std::vector<float> values(8, 1.0f);
    const std::vector<float> chunk_sums = ChunkSums(values, kChunkSize, kChunkCount);
    const auto exp_at = [&values](int32_t index) { return values[static_cast<size_t>(index)]; };

    float prefix_sum = 0.0f;
    // 前缀和序列是 1,2,3,...：threshold = 4 应给出 cutoff = 4（含第 4 个元素）
    const int32_t cutoff = FindFirstPrefixCrossing(chunk_sums.data(), kChunkCount, 8, kChunkSize,
                                                   4.0f, exp_at, &prefix_sum);
    EXPECT_EQ(cutoff, 4);
    EXPECT_FLOAT_EQ(prefix_sum, 4.0f);
}

// 阈值 = 全行和（p = 1 + 浮点边界）时不得提前截断：cutoff 必须到行尾。
TEST(NucleusCutoffTest, ReturnsWholeRowWhenThresholdCoversTotal) {
    constexpr int32_t kSize = 13;
    constexpr int32_t kChunkSize = 5;
    const int32_t chunk_count = (kSize + kChunkSize - 1) / kChunkSize;
    std::vector<float> values(kSize, 0.25f);
    const std::vector<float> chunk_sums = ChunkSums(values, kChunkSize, chunk_count);
    const auto exp_at = [&values](int32_t index) { return values[static_cast<size_t>(index)]; };

    float total = 0.0f;
    for (float value : values) {
        total += value;
    }

    float prefix_sum = 0.0f;
    const int32_t cutoff = FindFirstPrefixCrossing(chunk_sums.data(), chunk_count, kSize,
                                                   kChunkSize, total, exp_at, &prefix_sum);
    EXPECT_EQ(cutoff, kSize) << "阈值等于全行和时不得截断（p=1 的语义）";
}

// 阈值 <= 0：第一个元素就满足，cutoff = 1（对应 target = u * kept_total 在 u=0 时的语义）。
TEST(NucleusCutoffTest, ReturnsFirstElementWhenThresholdIsNonPositive) {
    constexpr int32_t kSize = 6;
    constexpr int32_t kChunkSize = 2;
    const int32_t chunk_count = (kSize + kChunkSize - 1) / kChunkSize;
    std::vector<float> values{0.5f, 0.25f, 0.25f, 0.1f, 0.05f, 0.05f};
    const std::vector<float> chunk_sums = ChunkSums(values, kChunkSize, chunk_count);
    const auto exp_at = [&values](int32_t index) { return values[static_cast<size_t>(index)]; };

    for (float threshold : {0.0f, -1.0f}) {
        float prefix_sum = -1.0f;
        const int32_t cutoff = FindFirstPrefixCrossing(chunk_sums.data(), chunk_count, kSize,
                                                       kChunkSize, threshold, exp_at, &prefix_sum);
        EXPECT_EQ(cutoff, 1) << "threshold=" << threshold;
        EXPECT_FLOAT_EQ(prefix_sum, values[0]);
    }
}

// 阈值超过全行和：报告"整行都不够"（返回 size），并把整行的串行累计和写回给调用方。
TEST(NucleusCutoffTest, ReportsNoCrossingAboveTotal) {
    constexpr int32_t kSize = 9;
    constexpr int32_t kChunkSize = 4;
    const int32_t chunk_count = (kSize + kChunkSize - 1) / kChunkSize;
    std::vector<float> values{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.25f, 0.25f, 0.25f};
    const std::vector<float> chunk_sums = ChunkSums(values, kChunkSize, chunk_count);
    const auto exp_at = [&values](int32_t index) { return values[static_cast<size_t>(index)]; };

    float total = 0.0f;
    for (float value : values) {
        total += value;
    }

    float prefix_sum = 0.0f;
    const int32_t cutoff = FindFirstPrefixCrossing(chunk_sums.data(), chunk_count, kSize,
                                                   kChunkSize, total + 1.0f, exp_at, &prefix_sum);
    EXPECT_EQ(cutoff, kSize);
    EXPECT_FLOAT_EQ(prefix_sum, total);
}

// ---------------------------------------------------------------------------
// SamplerReferenceTest：**参考实现自身的 host 自证**（沙箱可跑，S-23）。
//
// 为什么必须有：`PROGRESS.md` §2.13 的纪律——参考实现是裁决对错的标尺，标尺错了会给出错误
// 裁决（Phase 1.5 的 `ReferenceRoPE` 漏 batch 维就是这么把对的判成错的）。这些断言与 GPU 无关，
// 所以它们必须在沙箱里跑得到；同时它们只能证明参考"是良定义的分布"，**不能**替真机证明 kernel。
// ---------------------------------------------------------------------------

TEST(SamplerReferenceTest, TopKSoftmaxProbabilitiesMatchesTruncationSemantics) {
    const std::vector<float> logits{2.0f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -1.2f, -2.0f};
    const std::vector<int32_t> order = test_support::SortedDescendingIndices(logits);
    ASSERT_EQ(order.size(), logits.size());

    for (int32_t k : {1, 3, 6, 8, 100}) {
        const std::vector<double> probabilities =
            test_support::TopKSoftmaxProbabilities(logits, k);
        const size_t kept = std::min<size_t>(static_cast<size_t>(k), logits.size());

        double sum = 0.0;
        std::vector<int32_t> support;
        for (size_t v = 0; v < probabilities.size(); ++v) {
            sum += probabilities[v];
            if (probabilities[v] > 0.0) {
                support.push_back(static_cast<int32_t>(v));
            }
        }
        EXPECT_NEAR(sum, 1.0, 1e-12) << "k=" << k;
        // 支持集必须**正好**是解析 top-k：多一个/少一个都会让"落在集合内"这条判据失去意义
        const std::vector<int32_t> expected_support(order.begin(),
                                                    order.begin() + static_cast<ptrdiff_t>(kept));
        EXPECT_EQ(support, expected_support) << "k=" << k;
    }

    // 参考必须"能判别截断"：截断后 top-1 的概率**必然严格变大**（被丢掉的质量重新分配给它），
    // 所以若参考把 k 忽略掉（退化成全词表 softmax），这条会红。
    const std::vector<double> full = test_support::TopKSoftmaxProbabilities(logits, 8);
    const std::vector<double> top3 = test_support::TopKSoftmaxProbabilities(logits, 3);
    EXPECT_GT(top3[order[0]], full[order[0]] + 1e-3);
    EXPECT_GT(top3[order[0]], top3[order[1]]);
    EXPECT_EQ(top3[order[3]], 0.0) << "k=3 时第 4 名必须拿到 0 概率";
}

TEST(SamplerReferenceTest, TruncatedSoftmaxProbabilitiesAgreesWithNucleus) {
    const std::vector<float> logits{2.0f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -1.2f, -2.0f};
    for (float p : {1e-6f, 0.6f, 0.9f, 1.0f}) {
        const std::vector<double> probabilities =
            test_support::TruncatedSoftmaxProbabilities(logits, p);
        const std::vector<int32_t> nucleus = test_support::AnalyticNucleus(logits, p);

        double sum = 0.0;
        std::vector<int32_t> support;
        for (size_t v = 0; v < probabilities.size(); ++v) {
            sum += probabilities[v];
            if (probabilities[v] > 0.0) {
                support.push_back(static_cast<int32_t>(v));
            }
        }
        EXPECT_NEAR(sum, 1.0, 1e-12) << "p=" << p;
        // 两条参考必须自洽：分布的支持集 ≤ nucleus，且差额不超过登记的那 1 个边界余量
        for (int32_t token : support) {
            EXPECT_NE(std::find(nucleus.begin(), nucleus.end(), token), nucleus.end())
                << "p=" << p << " token " << token << " 不在 nucleus 内";
        }
        EXPECT_LE(nucleus.size(), support.size() + 1) << "p=" << p;
        // p 极小 → 退化为 argmax（与 S-7 的语义一致）
        if (p < 1e-3f) {
            EXPECT_EQ(support.size(), 1u);
        }
    }
}

// 事故回归（`TROUBLESHOOTING.md` + TS-036）：真机上 S-14 的第一版在这里是红的——
// 128000 词表（`MakeLogits` 的 sin 造数据）第 64 名有 4 个 token 精确并列，
// 采样到的 45721 是其中一员，而旧的 `partial_sort` 参考会把它排掉。这条用例把**事故本身**
// 固化：同样的数据、同样的 k，新的参考必须包含 45721，且必须比"前 k 个"严格更宽（含并列组）。
// 跑在 host 上 → 沙箱即可裁决，不必等真机。
TEST(SamplerReferenceTest, IncidentRow64TieIsNotASetMembershipFailure) {
    constexpr int32_t kVocab = 128000;
    constexpr int32_t kIncidentToken = 45721;
    std::vector<float> row(kVocab);
    for (int32_t v = 0; v < kVocab; ++v) {
        row[static_cast<size_t>(v)] =
            std::sin(0.29f * static_cast<float>(v)) * 1.2f;  // 与 MakeLogits(batch=0) 逐字一致
    }

    const std::vector<int32_t> allowed = test_support::TopKSetByValue(row, 64);
    EXPECT_NE(std::find(allowed.begin(), allowed.end(), kIncidentToken), allowed.end())
        << "45721 的值等于第 64 大值，必须被并列安全的参考接受";
    EXPECT_GT(allowed.size(), 64u) << "并列组跨过第 64 名 → 合法集合必然比 64 个更宽";
    // 事故现场的另一半：45721 与第 63 / 65 / 66 名的值精确相等（4 个 token 并列）
    int32_t equal_count = 0;
    for (float value : row) {
        if (value == row[kIncidentToken]) {
            ++equal_count;
        }
    }
    EXPECT_EQ(equal_count, 4) << "这条用例的前提就是并列组有 4 个成员；数据变了要重新取证";
}

// ---------------------------------------------------------------------------
// P9_2-5b：三级定位（块和 → 子块和 → 元素）的 host 裁决（S-24，沙箱可跑）。
//
// 为什么能在沙箱裁：内核那一段用的就是 `FindCrossingByLevels` 这一份实现（`__host__ __device__`），
// 与 S-22 锁 `FindFirstPrefixCrossing` 是同一个函数。**它挡不住的是内核装配**（共享内存布局、
// 子块和怎么算出来的），那部分只能真机验证。
// ---------------------------------------------------------------------------

// 通用数据上，三级定位必须与"整行逐元素串行扫描"给出同一个 cutoff（阈值取得远离前缀和的整数
// 边界，只比较判定结果，不比较末位）。
TEST(NucleusCutoffTest, ThreeLevelLookupMatchesFlatScanOnGenericData) {
    constexpr int32_t kSize = 1000;
    constexpr int32_t kChunkSize = 100;                  // 10 个块
    constexpr int32_t kSubChunks = 4;                    // 每块 4 个子块
    constexpr int32_t kSubSize = kChunkSize / kSubChunks;  // 25
    const int32_t chunk_count = (kSize + kChunkSize - 1) / kChunkSize;

    std::vector<float> values(kSize);
    for (int32_t i = 0; i < kSize; ++i) {
        values[static_cast<size_t>(i)] = 1.0f + 0.001f * static_cast<float>(i);
    }
    const std::vector<float> chunk_sums = ChunkSums(values, kChunkSize, chunk_count);
    std::vector<float> sub_sums(static_cast<size_t>(chunk_count) * kSubChunks, 0.0f);
    for (int32_t c = 0; c < chunk_count; ++c) {
        for (int32_t s = 0; s < kSubChunks; ++s) {
            float sum = 0.0f;
            const int32_t begin = c * kChunkSize + s * kSubSize;
            const int32_t end = std::min(begin + kSubSize, kSize);
            for (int32_t i = begin; i < end; ++i) {
                sum += values[static_cast<size_t>(i)];
            }
            sub_sums[static_cast<size_t>(c) * kSubChunks + s] = sum;
        }
    }
    const auto exp_at = [&values](int32_t index) { return values[static_cast<size_t>(index)]; };
    const auto flat_cutoff = [&values](float threshold) {
        float running = 0.0f;
        for (int32_t i = 0; i < static_cast<int32_t>(values.size()); ++i) {
            running += values[static_cast<size_t>(i)];
            if (running >= threshold) {
                return i + 1;
            }
        }
        return static_cast<int32_t>(values.size());
    };

    for (float threshold : {0.5f, 137.3f, 500.75f, 900.1f, 1000.5f}) {
        float prefix_sum = -1.0f;
        const int32_t cutoff =
            FindCrossingByLevels(chunk_sums.data(), chunk_count, sub_sums.data(), kSubChunks,
                                 kChunkSize, kSubSize, kSize, threshold, exp_at, &prefix_sum);
        EXPECT_EQ(cutoff, flat_cutoff(threshold)) << "threshold=" << threshold;
        float flat_prefix = 0.0f;
        for (int32_t i = 0; i < cutoff; ++i) {
            flat_prefix += values[static_cast<size_t>(i)];
        }
        EXPECT_FLOAT_EQ(prefix_sum, flat_prefix) << "threshold=" << threshold;
    }
}

// 子块级"没跨过"时（两级加法结合序不同，1 ulp 边界才会出现）必须退化为**整块重扫**，
// 且**不能把整块的和算两遍**——那会让阈值提前命中、cutoff 偏小。
// 这里用故意不自洽的 sub_sums 把这条路径逼出来（与 S-22 的 `FallsBackToChunkEndWhenChunkSumDisagrees` 同法）。
TEST(NucleusCutoffTest, ThreeLevelDegradesWithoutDoubleCountingTheChunk) {
    const std::vector<float> values{1.0f};         // 真正的元素和 = 1.0
    const std::vector<float> chunk_sums{2.0f};     // 块和声称 2.0（模拟舍入差到跨过阈值）
    const std::vector<float> sub_sums{0.5f, 0.5f}; // 子块和加起来只有 1.0，故意跨不过阈值
    const auto exp_at = [&values](int32_t index) { return values[static_cast<size_t>(index)]; };

    float prefix_sum = 0.0f;
    const int32_t cutoff =
        FindCrossingByLevels(chunk_sums.data(), 1, sub_sums.data(), 2, /*chunk_size=*/1,
                             /*sub_size=*/1, /*size=*/1, 2.0f, exp_at, &prefix_sum);
    EXPECT_EQ(cutoff, 1) << "必须退化到块末尾，且不能越过 size";
    EXPECT_FLOAT_EQ(prefix_sum, 1.0f) << "前缀和只能是逐元素口径的 1.0；2.0 说明把块和重复计入了";
}

// 第二次调用（采样点）要把搜索限制在 `size = cutoff` 之内：子块若越过该边界，重扫必须被截断。
TEST(NucleusCutoffTest, ThreeLevelRespectsTheSizeLimit) {
    const std::vector<float> values(8, 1.0f);
    const std::vector<float> chunk_sums{4.0f, 4.0f};  // chunk_size = 4
    const std::vector<float> sub_sums{2.0f, 2.0f, 2.0f, 2.0f};  // sub_size = 2
    const auto exp_at = [&values](int32_t index) { return values[static_cast<size_t>(index)]; };

    float prefix_sum = 0.0f;
    // threshold = 2.5：第一级命中第 0 块，第二级命中该块的第 1 个子块（元素 2~3），
    // 但 size = 3 只允许扫到元素 2 → 结果必须是 3 而不是 4。
    const int32_t cutoff = FindCrossingByLevels(chunk_sums.data(), 2, sub_sums.data(), 2,
                                                /*chunk_size=*/4, /*sub_size=*/2, /*size=*/3,
                                                2.5f, exp_at, &prefix_sum);
    EXPECT_EQ(cutoff, 3);
    EXPECT_FLOAT_EQ(prefix_sum, 3.0f);
}

TEST(SamplerKernelTest, GreedyMatchesArgmax) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kBatch = 4;
    constexpr int32_t kVocab = 1000;

    std::vector<int32_t> expected_argmax;
    const std::vector<float> logits = MakeLogits(kBatch, kVocab, &expected_argmax);
    EXPECT_EQ(RunGreedy(logits, kBatch, kVocab), expected_argmax);
}

TEST(SamplerKernelTest, GreedyPrefersLowestIndexOnTie) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // 与 torch.argmax 一致：并列时取最小下标
    constexpr int32_t kBatch = 1;
    constexpr int32_t kVocab = 8;
    std::vector<float> logits(kVocab, 0.0f);
    logits[3] = 1.0f;
    logits[5] = 1.0f;
    EXPECT_EQ(RunGreedy(logits, kBatch, kVocab)[0], 3);
}

TEST(SamplerKernelTest, TopKWithKEqualsOneBehavesLikeGreedy) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // k=1 时候选集只有 argmax，采样结果必须与 greedy 完全一致
    constexpr int32_t kBatch = 4;
    constexpr int32_t kVocab = 512;

    std::vector<int32_t> expected_argmax;
    const std::vector<float> logits = MakeLogits(kBatch, kVocab, &expected_argmax);
    EXPECT_EQ(RunTopK(logits, kBatch, kVocab, /*k=*/1, kSeed), expected_argmax);
}

// 快速路径与旧路径**逐 token 相同**——"语义等价"最硬的形态（不是分布、不是集合）。
// 两者排出的 top-k 集合、顺序、并列规则与随机数消费完全一致，因此同一 seed 必须同结果；
// 不一致就是缺陷（不是"允许的差异"，见测试计划 §9.4）。
TEST(SamplerKernelTest, TopKFastMatchesLegacyTokens) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    struct Case {
        int32_t vocab;
        int32_t batch;
        int32_t k;
    };
    // 覆盖：小词表（k == vocab）/ 中等 / GPT-2 真实词表 / §9.2 提到的 128K 目标规模。
    const Case kCases[] = {{4, 1, 4}, {1024, 2, 8}, {50257, 1, 64}, {128000, 1, 64}};
    for (const Case& item : kCases) {
        std::vector<int32_t> argmax;
        const std::vector<float> logits = MakeLogits(item.batch, item.vocab, &argmax);
        const std::vector<int32_t> legacy =
            RunTopK(logits, item.batch, item.vocab, item.k, kSeed);
        const std::vector<int32_t> fast =
            RunTopKFast(logits, item.batch, item.vocab, item.k, kSeed);
        EXPECT_EQ(fast, legacy) << "vocab=" << item.vocab << " batch=" << item.batch
                                << " k=" << item.k;
    }
}

// 契约护栏：k > kTopKFastMaxK 时快速路径必须写哨兵 -1，不许静默给错答案。
// 生产路径不会踩到（`LLMRunner` 在 host 侧判断），但契约本身要能被验证。
TEST(SamplerKernelTest, TopKFastPoisonsRowsAboveContractLimit) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kVocab = 1024;
    constexpr int32_t kTooLarge = kTopKFastMaxK + 1;
    std::vector<int32_t> argmax;
    const std::vector<float> logits = MakeLogits(1, kVocab, &argmax);
    const std::vector<int32_t> tokens = RunTopKFast(logits, 1, kVocab, kTooLarge, kSeed);
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0], -1) << "越过契约上限时必须写哨兵 -1，而不是给一个错答案";
}

TEST(SamplerKernelTest, TopKResultAlwaysWithinTopKSet) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kBatch = 3;
    constexpr int32_t kVocab = 256;
    constexpr int32_t kK = 5;

    std::vector<int32_t> argmax;
    const std::vector<float> logits = MakeLogits(kBatch, kVocab, &argmax);
    const std::vector<int32_t> tokens = RunTopK(logits, kBatch, kVocab, kK, kSeed);

    for (int32_t b = 0; b < kBatch; ++b) {
        std::vector<float> row(logits.begin() + static_cast<size_t>(b) * kVocab,
                               logits.begin() + static_cast<size_t>(b + 1) * kVocab);
        const std::vector<int32_t> allowed = test_support::TopKSetByValue(row, kK);
        EXPECT_NE(std::find(allowed.begin(), allowed.end(), tokens[b]), allowed.end())
            << "batch " << b << " sampled token " << tokens[b] << " outside top-" << kK;
    }
}

TEST(SamplerKernelTest, TopKSamplingIsDeterministicForFixedSeed) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // Q12：固定 seed 必须得到完全相同的结果，否则参考对比无法回归
    constexpr int32_t kBatch = 4;
    constexpr int32_t kVocab = 512;

    std::vector<int32_t> argmax;
    const std::vector<float> logits = MakeLogits(kBatch, kVocab, &argmax);
    const std::vector<int32_t> first = RunTopK(logits, kBatch, kVocab, /*k=*/8, kSeed);
    const std::vector<int32_t> second = RunTopK(logits, kBatch, kVocab, /*k=*/8, kSeed);
    EXPECT_EQ(first, second);

    // 不同 seed 应产生不同结果（统计上极大概率，用于确认 seed 真的参与了随机源）
    const std::vector<int32_t> other = RunTopK(logits, kBatch, kVocab, /*k=*/8, kSeed + 1);
    EXPECT_NE(first, other);
}

// S-14：大词表下采样 token 必须落在解析 top-K 集合内（k ∈ {1, 8, 64}）。
// 判据是集合成员关系——**无阈值，因此无"阈值出处"问题**。
//
// 判据必须是**并列安全**的（`TopKSetByValue`：值 ≥ 第 k 大值）。**第一次真机运行就是红的**
// （2026-09-27）：128000 词表上第 64 名有 4 个 token 精确并列，采样到的 45721 属于这个并列组，
// 而参考当时用 `std::partial_sort` 取"前 65 个"——不稳定排序可能把 45721 排掉，于是**在实现
// 完全正确时报红**。完整推导见 `TROUBLESHOOTING.md` + TS-036；事故本身由
// `SamplerReferenceTest.IncidentRow64TieIsNotASetMembershipFailure` 固化。
//
// 走的是生产入口 `LaunchTopKSampler`（CUB 分段排序）：`LaunchTopKSamplerFast` 与它的
// **逐 token 等价性**由 S-19 单独锁住，不在这里重复。
TEST(SamplerKernelTest, TopKOnLargeVocabStaysWithinTopKSet) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    struct Shape {
        int32_t vocab;
        int32_t batch;
    };
    // 50257 = GPT-2 真实词表；128000 = §9.2 的目标规模（合成 logits）
    const Shape kShapes[] = {{50257, 1}, {50257, 2}, {128000, 1}};
    const int32_t kKs[] = {1, 8, 64};
    constexpr int32_t kDraws = 3;

    for (const Shape& shape : kShapes) {
        std::vector<int32_t> argmax;
        const std::vector<float> logits = MakeLogits(shape.batch, shape.vocab, &argmax);
        for (int32_t k : kKs) {
            for (int32_t draw = 0; draw < kDraws; ++draw) {
                const std::vector<int32_t> tokens = RunTopK(
                    logits, shape.batch, shape.vocab, k, kSeed, static_cast<uint64_t>(draw));
                for (int32_t b = 0; b < shape.batch; ++b) {
                    const std::vector<float> row(
                        logits.begin() + static_cast<size_t>(b) * shape.vocab,
                        logits.begin() + static_cast<size_t>(b + 1) * shape.vocab);
                    const std::vector<int32_t> allowed = test_support::TopKSetByValue(row, k);
                    EXPECT_NE(std::find(allowed.begin(), allowed.end(), tokens[b]), allowed.end())
                        << "vocab=" << shape.vocab << " batch=" << b << " k=" << k
                        << " draw=" << draw << " token=" << tokens[b] << " 落在 top-" << k
                        << " 集合之外";
                    if (k == 1) {
                        // k=1 的额外含义：采样值必须等于全行最大值（并列时**不能**要求下标相等——
                        // 128000 词表的最大值有 8 个 token 精确并列，取谁由排序的 tie-break 决定）
                        EXPECT_FLOAT_EQ(row[static_cast<size_t>(tokens[b])], row[argmax[b]])
                            << "vocab=" << shape.vocab << " batch=" << b
                            << " k=1 的采样值必须等于全行最大值";
                    }
                }
            }
        }
    }
}

TEST(SamplerKernelTest, TopPWithTinyPicksArgmax) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // p 极小 → 截断到最前面的 token，退化为 greedy
    constexpr int32_t kBatch = 2;
    constexpr int32_t kVocab = 128;

    std::vector<int32_t> expected_argmax;
    const std::vector<float> logits = MakeLogits(kBatch, kVocab, &expected_argmax);
    EXPECT_EQ(RunTopP(logits, kBatch, kVocab, /*p=*/1e-6f, kSeed), expected_argmax);
}

TEST(SamplerKernelTest, TopPIsDeterministicForFixedSeed) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kBatch = 3;
    constexpr int32_t kVocab = 512;

    std::vector<int32_t> argmax;
    const std::vector<float> logits = MakeLogits(kBatch, kVocab, &argmax);
    const std::vector<int32_t> first = RunTopP(logits, kBatch, kVocab, 0.9f, kSeed);
    const std::vector<int32_t> second = RunTopP(logits, kBatch, kVocab, 0.9f, kSeed);
    EXPECT_EQ(first, second);
}

TEST(SamplerKernelTest, TopKDistributionMatchesSoftmaxProbabilities) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // k = vocab 时等价于全词表 softmax 采样，词频应收敛到解析概率。
    // 这是 D3「Generation 层对比分布」在 Phase 1 可落地的最小形式：
    // 逐 token 一致做不到，但分布必须对得上。
    constexpr int32_t kVocab = 4;
    constexpr int32_t kDraws = 8000;
    const std::vector<float> logits{0.5f, -0.5f, 0.0f, 1.5f};

    const std::vector<double> expected = SoftmaxProbabilities(logits);
    const std::vector<int32_t> counts =
        CollectTokenCounts(logits, kVocab, /*k=*/kVocab, kDraws, /*use_top_p=*/false, 0.0f);

    for (int32_t v = 0; v < kVocab; ++v) {
        const double observed = static_cast<double>(counts[v]) / kDraws;
        // 二项分布标准差 sqrt(p(1-p)/N)，取 3 sigma 作为容差
        const double sigma = std::sqrt(expected[v] * (1.0 - expected[v]) / kDraws);
        EXPECT_NEAR(observed, expected[v], 3.0 * sigma + 1e-3)
            << "vocab " << v << " expected " << expected[v] << " observed " << observed;
    }
}

TEST(SamplerKernelTest, TopPWithFullProbabilityDoesNotOverTruncate) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // p = 1.0 时 nucleus 应覆盖整个词表；若截断逻辑有 off-by-one 或提前收敛，
    // 概率最小的 token 会永远采不到。
    constexpr int32_t kVocab = 4;
    constexpr int32_t kDraws = 4000;
    const std::vector<float> logits{0.5f, -0.5f, 0.0f, 1.5f};

    const std::vector<int32_t> counts =
        CollectTokenCounts(logits, kVocab, /*k=*/0, kDraws, /*use_top_p=*/true, 1.0f);
    for (int32_t v = 0; v < kVocab; ++v) {
        EXPECT_GT(counts[v], 0) << "token " << v << " never sampled with p=1.0";
    }
}

// S-15：大词表下采样 token 必须落在解析 nucleus 内。
// 判据是集合成员关系（不含阈值 → 没有"阈值出处"问题）；与 legacy 的一致率只**打印**，
// 不作判据——两者累加顺序不同，边界处允许差一格（测试计划 §9.4）。
TEST(SamplerKernelTest, TopPOnLargeVocabStaysWithinNucleus) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    struct Shape {
        int32_t vocab;
        int32_t batch;
    };
    // 50257 = GPT-2 真实词表；128000 = §9.2 的目标规模（合成 logits）
    const Shape kShapes[] = {{50257, 1}, {50257, 8}, {128000, 1}};
    const float kPs[] = {0.9f, 0.99f};
    constexpr int32_t kDraws = 3;

    for (const Shape& shape : kShapes) {
        std::vector<int32_t> argmax;
        const std::vector<float> logits = MakeLogits(shape.batch, shape.vocab, &argmax);
        for (float p : kPs) {
            int32_t agreed = 0;
            int32_t compared = 0;
            for (int32_t draw = 0; draw < kDraws; ++draw) {
                const uint64_t offset = static_cast<uint64_t>(draw);
                const std::vector<int32_t> parallel =
                    RunTopP(logits, shape.batch, shape.vocab, p, kSeed, offset);
                const std::vector<int32_t> legacy =
                    RunTopP(logits, shape.batch, shape.vocab, p, kSeed, offset, /*legacy=*/true);
                for (int32_t b = 0; b < shape.batch; ++b) {
                    const std::vector<float> row(
                        logits.begin() + static_cast<size_t>(b) * shape.vocab,
                        logits.begin() + static_cast<size_t>(b + 1) * shape.vocab);
                    const std::vector<int32_t> allowed = test_support::AnalyticNucleus(row, p);
                    EXPECT_NE(std::find(allowed.begin(), allowed.end(), parallel[b]), allowed.end())
                        << "vocab=" << shape.vocab << " batch=" << b << " p=" << p
                        << " draw=" << draw << " token=" << parallel[b] << " 落在 nucleus 之外";
                    EXPECT_NE(std::find(allowed.begin(), allowed.end(), legacy[b]), allowed.end())
                        << "legacy: vocab=" << shape.vocab << " batch=" << b << " p=" << p
                        << " draw=" << draw << " token=" << legacy[b] << " 落在 nucleus 之外";
                    ++compared;
                    if (parallel[b] == legacy[b]) {
                        ++agreed;
                    }
                }
            }
            std::cout << "[TopP] vocab=" << shape.vocab << " batch=" << shape.batch << " p=" << p
                      << " 新实现与 legacy 逐 token 一致：" << agreed << "/" << compared
                      << "（观测，非判据）\n";
        }
    }
}

// S-16（改写）：p = 1.0 + 均匀 logits（nucleus = 整行）时不得过截断。
//
// 为什么用"被选下标的最小/最大值"而不是"每个 token 都被采到"：词表 5 万、抽样次数有限，
// 逐 token 计数几乎全是 0，判不出问题；而一旦实现把 cutoff 截到前 m 个元素，观测范围会立刻
// 塌缩到前 m 个。若 m <= 0.9V，200 次抽样全部落在前 10% 的概率是 0.1^200 ≈ 1e-200，
// 因此"观测最大值 > 0.9V"是一条几乎不可能误报的判据。
TEST(SamplerKernelTest, TopPWithFullProbabilityCoversHugeNucleus) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kVocab = 50257;
    constexpr int32_t kDraws = 200;
    const std::vector<float> logits(kVocab, 0.0f);  // 均匀 → 每个 token 概率相同

    int32_t min_token = kVocab;
    int32_t max_token = -1;
    for (int32_t draw = 0; draw < kDraws; ++draw) {
        const std::vector<int32_t> tokens =
            RunTopP(logits, 1, kVocab, 1.0f, kSeed, static_cast<uint64_t>(draw));
        ASSERT_GE(tokens[0], 0);
        ASSERT_LT(tokens[0], kVocab);
        min_token = std::min(min_token, tokens[0]);
        max_token = std::max(max_token, tokens[0]);
    }
    EXPECT_LT(min_token, kVocab / 10) << "被选下标的最小值落在前 10%：疑似 cutoff 提前收敛";
    EXPECT_GT(max_token, kVocab * 9 / 10) << "被选下标的最大值没进最后 10%：疑似截断过激";
}

// S-21：分布级主判据——"截断 + 前缀内重新归一化"之后的词频必须收敛到解析分布（3σ + 1e-3，
// 口径沿用 `TopKDistributionMatchesSoftmaxProbabilities` 的注释）。这条能抓到集合成员关系
// 抓不到的错：cutoff 多截/少截一个元素会直接改变分布形状。两条实现都跑，互为对照。
TEST(SamplerKernelTest, TopPDistributionMatchesTruncatedSoftmaxProbabilities) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kVocab = 8;
    constexpr int32_t kDraws = 8000;
    // 三个 p 对应三个不同的 cutoff（2 个 / 4 个 / 8 个 token），覆盖"截断在行内"与"整行保留"
    const std::vector<float> logits{2.0f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -1.2f, -2.0f};
    const float kPs[] = {0.6f, 0.9f, 1.0f};

    for (float p : kPs) {
        const std::vector<double> expected =
            test_support::TruncatedSoftmaxProbabilities(logits, p);
        for (bool legacy : {false, true}) {
            const std::vector<int32_t> counts =
                CollectTokenCounts(logits, kVocab, /*k=*/0, kDraws, /*use_top_p=*/true, p, legacy);
            for (int32_t v = 0; v < kVocab; ++v) {
                const double observed = static_cast<double>(counts[v]) / kDraws;
                const double sigma = std::sqrt(expected[v] * (1.0 - expected[v]) / kDraws);
                EXPECT_NEAR(observed, expected[v], 3.0 * sigma + 1e-3)
                    << (legacy ? "legacy" : "parallel") << " p=" << p << " token " << v
                    << " expected " << expected[v] << " observed " << observed;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// P9_2-0：采样器性能基线（计划见 future_iterations_development_plan.md §10，用例编号 S-18）
//
// 为什么先有它：§9.2 的触发条件是"profile 确认 sampler 占比显著"，而这个数据至今不存在
// （`PROGRESS.md` 里查不到这个数字；同类端到端数字只有 `CVRunner` 的 benchmark，
// 见 `docs/dev/REQ-007-resnet18/phase4_development_plan.md` 的 P4-3 行）。本用例量的是**采样器自身**的耗时，
// 用来做"改实现前后各跑一次"的对照。
//
// 协议按 G6（docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md §5）：先 warmup，再多次采样，**报中位数与极差**，
// 不报单次点值——单次点值已经在 Phase 3 的性能结论上吃过一次亏。
//
// 两点口径说明：
// 1) `CudaTimer::Stop` 会同步，所以每次迭代含一次同步（微秒级，相对这里的毫秒级可忽略）；
//    报的是"设备端执行 + 一次同步"的时间。前后对比在同一台机器上有效。
// 2) 本用例**不对耗时做断言**（仪器不是判据），只断言"确实跑起来了"（中位数 > 0）
//    以及采样结果落在合法下标范围内。
// ---------------------------------------------------------------------------
TEST(SamplerPerf, ThroughputByShape) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    struct Shape {
        int32_t vocab;
        int32_t batch;
    };
    // 50257 = GPT-2 真实词表；128000 = §9.2 提到的"可达 128K"的目标规模（合成 logits）。
    const Shape kShapes[] = {{50257, 1}, {50257, 8}, {128000, 1}, {128000, 8}};
    constexpr int32_t kTopK = 64;
    constexpr float kTopP = 0.9f;
    constexpr int32_t kWarmup = 3;
    // 轮数。每轮现在做"正向 + 反向"两遍、且每个被测变体要跑 1 次与 4 次两个窗口，所以单轮比
    // 之前贵一倍多；但斜率估计量本身比"单发中位数"稳得多（固定开销被减掉），15 轮足够。
    constexpr int32_t kIterations = 15;

    const auto median_of = [](std::vector<float> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    // **配对测量**：每一轮把 4 个变体挨着各测一次，再逐轮算比值/差值。
    // 为什么不能分段测（先把 greedy 测 31 次、再测 top-k 31 次…）：块与块之间的时钟/温度漂移
    // 各自独立，而 P9_2-5b 要看的信号（`top-p − top-k` ~70 µs）只占总耗时的 5%~13%——
    // 分段测出来的差值会出现负值（实测那一次就是），比噪声还小。配对后漂移对四者同向，
    // 逐轮比值/差值再取中位数，估计量才对得上"同 session 比"的初衷。
    std::cout << "[SamplerPerf] 口径：设备端执行 + 每次一次同步（CudaTimer）；warmup=" << kWarmup
              << "，每轮把 4 个变体各测一次（配对），共 " << kIterations << " 轮，报中位数与极差\n";

    for (const Shape& shape : kShapes) {
        std::vector<float> host_logits(static_cast<size_t>(shape.batch) * shape.vocab);
        for (size_t i = 0; i < host_logits.size(); ++i) {
            host_logits[i] = DeterministicValue(static_cast<int64_t>(i));
        }
        const DeviceLogits logits = UploadLogits(host_logits);
        DeviceBuffer tokens(static_cast<size_t>(shape.batch) * sizeof(int32_t));
        DeviceBuffer device_k(static_cast<size_t>(shape.batch) * sizeof(int32_t));
        DeviceBuffer device_p(static_cast<size_t>(shape.batch) * sizeof(float));
        const size_t ws_topk_bytes = TopKSamplerWorkspaceBytes(shape.batch, shape.vocab);
        const size_t ws_topp_bytes = TopPSamplerWorkspaceBytes(shape.batch, shape.vocab);
        DeviceBuffer ws_topk(ws_topk_bytes);
        DeviceBuffer ws_topp(ws_topp_bytes);
        ASSERT_TRUE(tokens.Allocate(static_cast<size_t>(shape.batch) * sizeof(int32_t)));
        ASSERT_TRUE(device_k.Allocate(static_cast<size_t>(shape.batch) * sizeof(int32_t)));
        ASSERT_TRUE(device_p.Allocate(static_cast<size_t>(shape.batch) * sizeof(float)));
        ASSERT_TRUE(ws_topk.Allocate(ws_topk_bytes));
        ASSERT_TRUE(ws_topp.Allocate(ws_topp_bytes));

        const std::vector<int32_t> host_k(shape.batch, kTopK);
        const std::vector<float> host_p(shape.batch, kTopP);
        CUDA_CHECK(cudaMemcpy(device_k.data(), host_k.data(), host_k.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_p.data(), host_p.data(), host_p.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        SamplerArgs greedy_args;
        greedy_args.logits = logits.buffer.data();
        greedy_args.token_ids = static_cast<int32_t*>(tokens.data());
        greedy_args.batch_size = shape.batch;
        greedy_args.vocab_size = shape.vocab;
        greedy_args.is_half = false;

        TopKSamplerArgs topk_args;
        static_cast<SamplerArgs&>(topk_args) = greedy_args;
        topk_args.top_k = static_cast<const int32_t*>(device_k.data());
        topk_args.seed = kSeed;

        TopPSamplerArgs topp_args;
        static_cast<SamplerArgs&>(topp_args) = greedy_args;
        topp_args.top_p = static_cast<const float*>(device_p.data());
        topp_args.seed = kSeed;

        // 一次测量 = Start → launch × N → Stop（Stop 内部同步，所以每个窗口互相隔离）。
        CudaTimer timer;
        const auto time_n = [&timer](const std::function<void()>& launch, int32_t repeats) {
            timer.Start(nullptr);
            for (int32_t r = 0; r < repeats; ++r) {
                launch();
            }
            return timer.Stop(nullptr);
        };
        // **斜率**：同一窗口里发射 1 次与 4 次的差 / 3 = 每次发射的**净成本**。
        // 为什么需要它：这个环境里"一个窗口"的固定开销（事件 + 同步 + 首次发射，WSL2 尤其明显）
        // 与我们要看的信号同量级——实测一个平凡的 greedy 内核单发就要 64~154 µs。单发比较
        // 里这份固定开销会跟着差值一起进来，于是 P9_2-5b 那 16 倍的重扫缩短根本量不出来
        // （`TROUBLESHOOTING.md` + TS-038）。斜率把固定项减掉，只留"多做一次要花多少"。
        const auto slope_of = [&time_n](const std::function<void()>& launch) {
            return (time_n(launch, 4) - time_n(launch, 1)) / 3.0f;
        };
        const auto report = [&shape, &median_of](const char* name, const std::vector<float>& times) {
            const float median = median_of(times);
            const auto range = std::minmax_element(times.begin(), times.end());
            std::cout << "[SamplerPerf] vocab=" << shape.vocab << " batch=" << shape.batch << " "
                      << name << "：median=" << median << " ms, min=" << *range.first
                      << ", max=" << *range.second << " (n=" << times.size() << ")\n";
            return median;
        };
        const auto report_net = [&shape, &median_of](const char* name, const std::vector<float>& slopes) {
            const float median = median_of(slopes);
            const auto range = std::minmax_element(slopes.begin(), slopes.end());
            std::cout << "[SamplerPerf]   （净）" << name << "：median=" << median
                      << " ms/次, min=" << *range.first << ", max=" << *range.second
                      << " (n=" << slopes.size() << ")\n";
            return median;
        };

        // 五个变体各自的 launch。greedy 作"一趟扫描"的参照（无排序 = 该形状的下界参考）；
        // top-k fast（P9_2-4）在 k = 64 = kTopKFastMaxK 这个契约边界上与 legacy 对比。
        const std::function<void()> launch_greedy = [&]() {
            CUDA_CHECK(LaunchGreedySampler(greedy_args, nullptr));
        };
        const std::function<void()> launch_topk = [&]() {
            CUDA_CHECK(LaunchTopKSampler(topk_args, nullptr, ws_topk.data(), ws_topk_bytes));
        };
        const std::function<void()> launch_topk_fast = [&]() {
            CUDA_CHECK(LaunchTopKSamplerFast(topk_args, nullptr));
        };
        const std::function<void()> launch_topp = [&]() {
            CUDA_CHECK(LaunchTopPSampler(topp_args, nullptr, ws_topp.data(), ws_topp_bytes));
        };
        // P9_2-5b 的 A/B 对照：同一排序、同一第一趟，只把"子块级定位"换成两级。两版同轮交替测，
        // 差值就是这改动的真实幅度（跨 session/跨协议的问题全部消失）。
        const std::function<void()> launch_topp_two_level = [&]() {
            CUDA_CHECK(
                LaunchTopPSamplerTwoLevel(topp_args, nullptr, ws_topp.data(), ws_topp_bytes));
        };
        // legacy 对照：P9_2-5 的验收要求"同 harness、同形状、同 session"给出加速比，
        // 所以旧实现必须在同一次运行里再测一遍（§10.5 那张基线表是改动前的数字）。
        // 它本身 ≫ 固定开销（6~24 ms），所以只测单发，不再求斜率。
        const std::function<void()> launch_topp_legacy = [&]() {
            CUDA_CHECK(
                LaunchTopPSamplerLegacy(topp_args, nullptr, ws_topp.data(), ws_topp_bytes));
        };

        for (int32_t i = 0; i < kWarmup; ++i) {
            launch_greedy();
            launch_topk();
            launch_topk_fast();
            launch_topp();
            launch_topp_two_level();
            launch_topp_legacy();
        }

        std::vector<float> greedy_raw, topk_raw, topk_fast_raw, topp_raw, topp_legacy_raw;
        std::vector<float> greedy_net, topk_net, topp_net;
        std::vector<float> paired_speedup_net;      // legacy 单发（扣掉固定项）/ top-p 净成本
        std::vector<float> paired_delta_net_us;     // top-p 净成本 − top-k 净成本
        std::vector<float> topp_two_level_net;
        std::vector<float> paired_ab_delta_us;      // top-p（子块版）净成本 − 两级版净成本
        for (int32_t i = 0; i < kIterations; ++i) {
            // 一轮里**正向 + 反向**各测一遍（ABBA）：固定顺序会把"谁先测谁后测"的系统偏差
            // 直接算进 (top-p − top-k)，反向再来一遍后取平均即可抵消线性漂移。
            const float greedy_f = slope_of(launch_greedy);
            const float topk_f = slope_of(launch_topk);
            const float topp_f = slope_of(launch_topp);
            const float two_level_f = slope_of(launch_topp_two_level);
            const float topk_fast_f = time_n(launch_topk_fast, 1);
            const float legacy_f = time_n(launch_topp_legacy, 1);
            const float legacy_b = time_n(launch_topp_legacy, 1);
            const float topk_fast_b = time_n(launch_topk_fast, 1);
            const float two_level_b = slope_of(launch_topp_two_level);
            const float topp_b = slope_of(launch_topp);
            const float topk_b = slope_of(launch_topk);
            const float greedy_b = slope_of(launch_greedy);

            greedy_raw.push_back(time_n(launch_greedy, 1));
            topk_raw.push_back(time_n(launch_topk, 1));
            topk_fast_raw.push_back(0.5f * (topk_fast_f + topk_fast_b));
            topp_raw.push_back(time_n(launch_topp, 1));
            topp_legacy_raw.push_back(0.5f * (legacy_f + legacy_b));

            const float greedy_s = 0.5f * (greedy_f + greedy_b);
            const float topk_s = 0.5f * (topk_f + topk_b);
            const float topp_s = 0.5f * (topp_f + topp_b);
            const float two_level_s = 0.5f * (two_level_f + two_level_b);
            greedy_net.push_back(greedy_s);
            topk_net.push_back(topk_s);
            topp_net.push_back(topp_s);
            topp_two_level_net.push_back(two_level_s);
            // A/B：同一轮内两版各测一次（正反各一遍取平均），差值就是"子块级定位"的净收益。
            // 负值 = 生产版（子块）更快。
            paired_ab_delta_us.push_back((topp_s - two_level_s) * 1000.0f);
            // 固定项 = 单发 − 净成本（用最便宜的内核量，最保守）：legacy 的净成本 = 单发 − 固定项。
            const float fixed_overhead = std::max(0.0f, greedy_raw.back() - greedy_s);
            const float legacy_net = std::max(0.0f, topp_legacy_raw.back() - fixed_overhead);
            paired_speedup_net.push_back(legacy_net / topp_s);
            paired_delta_net_us.push_back((topp_s - topk_s) * 1000.0f);
        }

        // 单发口径（与历史基线表同口径，供对齐；判据不看它）
        const float greedy_ms = report("greedy", greedy_raw);
        const float topk_ms = report("top-k(k=64)", topk_raw);
        const float topk_fast_ms = report("top-k fast(k=64)", topk_fast_raw);
        const float topp_ms = report("top-p(p=0.9)", topp_raw);
        const float topp_legacy_ms = report("top-p legacy(p=0.9)", topp_legacy_raw);
        // 净成本口径（= 这次真正要看的数）
        const float greedy_net_ms = report_net("greedy", greedy_net);
        const float topk_net_ms = report_net("top-k(k=64)", topk_net);
        const float topp_net_ms = report_net("top-p(p=0.9)", topp_net);
        const float topp_two_level_net_ms = report_net("top-p 两级对照(P9_2-5b 前)", topp_two_level_net);

        // 仪器也要自证"确实跑了"：耗时为 0 说明没执行，采样结果越界说明写坏了。
        EXPECT_GT(greedy_ms, 0.0f);
        EXPECT_GT(topk_ms, 0.0f);
        EXPECT_GT(topk_fast_ms, 0.0f);
        EXPECT_GT(topp_ms, 0.0f);
        EXPECT_GT(topp_legacy_ms, 0.0f);
        // greedy 的净成本只有十几 µs，和斜率估计量的抖动同量级 → 只要求非负（它是"读一遍行"的
        // 下界参照，不是判据）；top-k / top-p 的净成本 ≫ 抖动，才要求严格为正。
        EXPECT_GE(greedy_net_ms, 0.0f);
        EXPECT_GT(topk_net_ms, 0.0f);
        EXPECT_GT(topp_net_ms, 0.0f);
        EXPECT_GT(topp_two_level_net_ms, 0.0f);
        for (const int32_t token : DownloadTokens(tokens, shape.batch)) {
            EXPECT_GE(token, 0);
            EXPECT_LT(token, shape.vocab);
        }
        std::cout << "[SamplerPerf] 单发对照：top-k/greedy=" << (topk_ms / greedy_ms)
                  << "×，top-k-fast/greedy=" << (topk_fast_ms / greedy_ms)
                  << "×，top-p/greedy=" << (topp_ms / greedy_ms) << "×，"
                  << "top-p legacy/parallel=" << (topp_legacy_ms / topp_ms) << "×，"
                  << "top-k legacy/fast=" << (topk_ms / topk_fast_ms) << "×\n";
        // 配对（净）估计量：P9_2-5b 的判定看这两个——它扣掉了每窗口固定开销，且每轮 ABBA。
        const auto paired_range = std::minmax_element(paired_speedup_net.begin(),
                                                     paired_speedup_net.end());
        const auto delta_range =
            std::minmax_element(paired_delta_net_us.begin(), paired_delta_net_us.end());
        std::vector<float> delta_sorted = paired_delta_net_us;
        std::sort(delta_sorted.begin(), delta_sorted.end());
        std::vector<float> ab_sorted = paired_ab_delta_us;
        std::sort(ab_sorted.begin(), ab_sorted.end());
        const auto ab_range = std::minmax_element(paired_ab_delta_us.begin(),
                                                 paired_ab_delta_us.end());
        std::cout << "[SamplerPerf] A/B（净，同一轮配对）：子块版 − 两级版 median="
                  << median_of(paired_ab_delta_us) << " µs, p25=" << ab_sorted[ab_sorted.size() / 4]
                  << ", p75=" << ab_sorted[ab_sorted.size() * 3 / 4] << ", min=" << *ab_range.first
                  << ", max=" << *ab_range.second
                  << "（负值 = P9_2-5b 更快）\n";
        std::cout << "[SamplerPerf] 配对（净）：legacy/parallel 加速比 median="
                  << median_of(paired_speedup_net) << "×, min=" << *paired_range.first
                  << ", max=" << *paired_range.second
                  << "；(top-p − top-k) median=" << median_of(paired_delta_net_us)
                  << " µs, p25=" << delta_sorted[delta_sorted.size() / 4]
                  << ", p75=" << delta_sorted[delta_sorted.size() * 3 / 4]
                  << ", min=" << *delta_range.first << ", max=" << *delta_range.second << "\n";
    }
}

}  // namespace mini_trt_llm
