#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

std::vector<int32_t> RunTopP(const std::vector<float>& logits, int32_t batch_size,
                             int32_t vocab_size, float p, uint64_t seed,
                             uint64_t offset = 0) {
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
    CUDA_CHECK(LaunchTopPSampler(args, nullptr, workspace.data(), workspace_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    return DownloadTokens(tokens, batch_size);
}

std::vector<int32_t> TopKIndices(const std::vector<float>& row_logits, int32_t k) {
    std::vector<int32_t> indices(row_logits.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int32_t>(i);
    }
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&row_logits](int32_t a, int32_t b) {
                          return row_logits[a] > row_logits[b];
                      });
    indices.resize(k);
    return indices;
}

// 统计多次采样的词频。fixed seed + 递增 offset 等价于"同一随机流的连续抽样"，
// 因此结果完全可复现，又足以做分布检验（D3 要求验证采样分布而非逐 token 一致）。
std::vector<int32_t> CollectTokenCounts(const std::vector<float>& logits,
                                        int32_t vocab_size, int32_t k, int32_t draws,
                                        bool use_top_p, float p) {
    std::vector<int32_t> counts(vocab_size, 0);
    for (int32_t draw = 0; draw < draws; ++draw) {
        const uint64_t offset = static_cast<uint64_t>(draw);
        const std::vector<int32_t> tokens =
            use_top_p ? RunTopP(logits, 1, vocab_size, p, kSeed, offset)
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
        const std::vector<int32_t> allowed = TopKIndices(row, kK);
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

}  // namespace mini_trt_llm
