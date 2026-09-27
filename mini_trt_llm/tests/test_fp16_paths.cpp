#include "mini_trt_llm/plugins/paged_attention_kernel.hpp"
#include "mini_trt_llm/plugins/paged_attention_split.hpp"
#include "mini_trt_llm/plugins/rope_kernel.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "paged_attention_test_support.hpp"
#include "sampler_test_support.hpp"
#include "test_gpu_guard.hpp"
#include "test_reference.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::WithinTolerance;

// FP16 覆盖缺口：Phase 1 的 L1 用例原先只有 RMSNorm 覆盖了 FP16，
// RoPE / PagedAttention / Sampler 的 is_half 分支同样在生产路径上，必须一并验证。

std::vector<__half> ToHalf(const std::vector<float>& values) {
    std::vector<__half> halves(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        halves[i] = __float2half_rn(values[i]);
    }
    return halves;
}

std::vector<float> ToFloat(const std::vector<__half>& halves) {
    std::vector<float> values(halves.size());
    for (size_t i = 0; i < halves.size(); ++i) {
        values[i] = __half2float(halves[i]);
    }
    return values;
}

void RequireAllocate(DeviceBuffer& buffer, size_t bytes) {
    if (!buffer.Allocate(bytes)) {
        throw std::runtime_error("fp16 test: failed to allocate device buffer");
    }
}

float DeterministicValue(int64_t index, float phase) {
    return std::sin(0.41f * static_cast<float>(index) + phase) * 0.6f;
}

// 采样用例的固定 seed：Q12 要求参考结果可复现（与 test_sampler.cpp 的 kSeed 同值）。
constexpr uint64_t kSamplerSeed = 42;

// 在同一份 logits 上按指定精度连续抽 draws 次（offset 递增 = 同一随机流的连续抽样），
// 返回词表计数。比 test_sampler.cpp 的 CollectTokenCounts 多一个 is_half 维度，
// 其余口径（每次一 draw、固定 seed）保持一致。
std::vector<int32_t> CollectTopPTokenCounts(const void* device_logits, bool is_half, int32_t vocab,
                                            float p, int32_t draws) {
    const size_t workspace_bytes = TopPSamplerWorkspaceBytes(1, vocab);
    DeviceBuffer d_p(sizeof(float));
    DeviceBuffer d_tokens(sizeof(int32_t));
    DeviceBuffer workspace(workspace_bytes);
    RequireAllocate(d_p, sizeof(float));
    RequireAllocate(d_tokens, sizeof(int32_t));
    RequireAllocate(workspace, workspace_bytes);
    CUDA_CHECK(cudaMemcpy(d_p.data(), &p, sizeof(float), cudaMemcpyHostToDevice));

    TopPSamplerArgs args;
    args.logits = device_logits;
    args.token_ids = static_cast<int32_t*>(d_tokens.data());
    args.top_p = static_cast<const float*>(d_p.data());
    args.batch_size = 1;
    args.vocab_size = vocab;
    args.is_half = is_half;
    args.seed = kSamplerSeed;

    std::vector<int32_t> counts(static_cast<size_t>(vocab), 0);
    for (int32_t draw = 0; draw < draws; ++draw) {
        args.offset = static_cast<uint64_t>(draw);
        CUDA_CHECK(LaunchTopPSampler(args, nullptr, workspace.data(), workspace_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        int32_t token = -1;
        CUDA_CHECK(cudaMemcpy(&token, d_tokens.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("fp16 top-p: sampled token out of range");
        }
        ++counts[static_cast<size_t>(token)];
    }
    return counts;
}

// Top-K 版（同上，多一个 per-batch 的 k）。走生产入口 `LaunchTopKSampler`（CUB 分段排序
// + 逐行采样），即 `LLMRunner` 实际使用的那条路。
std::vector<int32_t> CollectTopKTokenCounts(const void* device_logits, bool is_half, int32_t vocab,
                                            int32_t k, int32_t draws) {
    const size_t workspace_bytes = TopKSamplerWorkspaceBytes(1, vocab);
    DeviceBuffer d_k(sizeof(int32_t));
    DeviceBuffer d_tokens(sizeof(int32_t));
    DeviceBuffer workspace(workspace_bytes);
    RequireAllocate(d_k, sizeof(int32_t));
    RequireAllocate(d_tokens, sizeof(int32_t));
    RequireAllocate(workspace, workspace_bytes);
    CUDA_CHECK(cudaMemcpy(d_k.data(), &k, sizeof(int32_t), cudaMemcpyHostToDevice));

    TopKSamplerArgs args;
    args.logits = device_logits;
    args.token_ids = static_cast<int32_t*>(d_tokens.data());
    args.top_k = static_cast<const int32_t*>(d_k.data());
    args.batch_size = 1;
    args.vocab_size = vocab;
    args.is_half = is_half;
    args.seed = kSamplerSeed;

    std::vector<int32_t> counts(static_cast<size_t>(vocab), 0);
    for (int32_t draw = 0; draw < draws; ++draw) {
        args.offset = static_cast<uint64_t>(draw);
        CUDA_CHECK(LaunchTopKSampler(args, nullptr, workspace.data(), workspace_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        int32_t token = -1;
        CUDA_CHECK(cudaMemcpy(&token, d_tokens.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("fp16 top-k: sampled token out of range");
        }
        ++counts[static_cast<size_t>(token)];
    }
    return counts;
}

}  // namespace

// 一条用例同时补三个缺口：FP16 计算、GQA（heads != kv_heads）、batch > 1。
TEST(Fp16PathTest, RoPEHandlesFp16WithGqaAndMultipleBatches) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kBatch = 2;
    constexpr int32_t kHeads = 4;
    constexpr int32_t kKvHeads = 2;  // GQA：每 2 个 query head 共享 1 个 kv head
    constexpr int32_t kSeq = 2;
    constexpr int32_t kHeadSize = 8;
    constexpr int32_t kRotaryDim = 4;  // 部分旋转
    constexpr float kBase = 10000.0f;

    const size_t q_elements = static_cast<size_t>(kBatch) * kHeads * kSeq * kHeadSize;
    const size_t k_elements = static_cast<size_t>(kBatch) * kKvHeads * kSeq * kHeadSize;
    const size_t pos_elements = static_cast<size_t>(kBatch) * kSeq;

    std::vector<float> query(q_elements), key(k_elements);
    std::vector<int32_t> positions{2, 5, 7, 11};  // 非连续
    for (size_t i = 0; i < q_elements; ++i) {
        query[i] = DeterministicValue(static_cast<int64_t>(i), 0.0f);
    }
    for (size_t i = 0; i < k_elements; ++i) {
        key[i] = DeterministicValue(static_cast<int64_t>(i), 1.7f);
    }

    const std::vector<__half> query_h = ToHalf(query);
    const std::vector<__half> key_h = ToHalf(key);

    // 参考实现用 FP16 舍入后的输入，避免把输入量化误差算成 kernel 误差
    std::vector<double> query_d(q_elements), key_d(k_elements);
    for (size_t i = 0; i < q_elements; ++i) {
        query_d[i] = __half2float(query_h[i]);
    }
    for (size_t i = 0; i < k_elements; ++i) {
        key_d[i] = __half2float(key_h[i]);
    }
    std::vector<double> expected_q, expected_k;
    test_support::ReferenceRoPE(query_d, positions, kBatch, kHeads, kSeq, kHeadSize,
                                kRotaryDim, kBase, &expected_q);
    test_support::ReferenceRoPE(key_d, positions, kBatch, kKvHeads, kSeq, kHeadSize,
                                kRotaryDim, kBase, &expected_k);

    DeviceBuffer d_query(q_elements * sizeof(__half));
    DeviceBuffer d_key(k_elements * sizeof(__half));
    DeviceBuffer d_pos(pos_elements * sizeof(int32_t));
    DeviceBuffer d_q_out(q_elements * sizeof(__half));
    DeviceBuffer d_k_out(k_elements * sizeof(__half));
    RequireAllocate(d_query, q_elements * sizeof(__half));
    RequireAllocate(d_key, k_elements * sizeof(__half));
    RequireAllocate(d_pos, pos_elements * sizeof(int32_t));
    RequireAllocate(d_q_out, q_elements * sizeof(__half));
    RequireAllocate(d_k_out, k_elements * sizeof(__half));
    CUDA_CHECK(cudaMemcpy(d_query.data(), query_h.data(), q_elements * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key.data(), key_h.data(), k_elements * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.data(), positions.data(), pos_elements * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    RoPEKernelArgs args;
    args.query = d_query.data();
    args.key = d_key.data();
    args.position_ids = static_cast<const int32_t*>(d_pos.data());
    args.query_out = d_q_out.data();
    args.key_out = d_k_out.data();
    args.batch_size = kBatch;
    args.seq_len = kSeq;
    args.num_heads = kHeads;
    args.num_kv_heads = kKvHeads;
    args.head_size = kHeadSize;
    args.rotary_dim = kRotaryDim;
    args.base = kBase;
    args.is_half = true;
    ASSERT_EQ(LaunchRoPE(args, nullptr), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> q_out(q_elements), k_out(k_elements);
    CUDA_CHECK(cudaMemcpy(q_out.data(), d_q_out.data(), q_elements * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_out.data(), d_k_out.data(), k_elements * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    const std::vector<float> q_actual = ToFloat(q_out);
    const std::vector<float> k_actual = ToFloat(k_out);

    for (size_t i = 0; i < q_elements; ++i) {
        EXPECT_TRUE(WithinTolerance(static_cast<float>(expected_q[i]), q_actual[i], 1e-3f, 1e-3f))
            << "query index " << i;
    }
    for (size_t i = 0; i < k_elements; ++i) {
        EXPECT_TRUE(WithinTolerance(static_cast<float>(expected_k[i]), k_actual[i], 1e-3f, 1e-3f))
            << "key index " << i;
    }
}

TEST(Fp16PathTest, PagedAttentionMatchesFp16Reference) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kHeads = 2;
    constexpr int32_t kHeadSize = 8;
    constexpr int32_t kBlockSize = 8;
    constexpr int32_t kNumBlocks = 4;
    constexpr int32_t kMaxBlocks = 2;
    constexpr int32_t kContextLen = 11;  // 跨两个物理块

    const size_t cache_elements =
        static_cast<size_t>(kNumBlocks) * kBlockSize * kHeads * kHeadSize;
    std::vector<float> query(static_cast<size_t>(kHeads) * kHeadSize);
    std::vector<float> key_cache(cache_elements), value_cache(cache_elements);
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = DeterministicValue(static_cast<int64_t>(i), 0.0f);
    }
    for (size_t i = 0; i < cache_elements; ++i) {
        key_cache[i] = DeterministicValue(static_cast<int64_t>(i), 2.1f);
        value_cache[i] = DeterministicValue(static_cast<int64_t>(i), 4.3f);
    }
    const std::vector<int32_t> block_table{3, 1};  // 故意不是顺序块
    const int32_t context_len = kContextLen;

    const std::vector<__half> query_h = ToHalf(query);
    const std::vector<__half> key_h = ToHalf(key_cache);
    const std::vector<__half> value_h = ToHalf(value_cache);

    std::vector<double> query_d(query.size()), key_d(cache_elements), value_d(cache_elements);
    for (size_t i = 0; i < query.size(); ++i) {
        query_d[i] = __half2float(query_h[i]);
    }
    for (size_t i = 0; i < cache_elements; ++i) {
        key_d[i] = __half2float(key_h[i]);
        value_d[i] = __half2float(value_h[i]);
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
    std::vector<double> expected;
    test_support::ReferencePagedAttentionDecode(query_d, key_d, value_d, block_table,
                                                context_len, kHeads, kHeads, kHeadSize,
                                                kBlockSize, scale, &expected);

    DeviceBuffer d_query(query.size() * sizeof(__half));
    DeviceBuffer d_key(cache_elements * sizeof(__half));
    DeviceBuffer d_value(cache_elements * sizeof(__half));
    DeviceBuffer d_table(kMaxBlocks * sizeof(int32_t));
    DeviceBuffer d_context(sizeof(int32_t));
    DeviceBuffer d_out(query.size() * sizeof(__half));
    RequireAllocate(d_query, query.size() * sizeof(__half));
    RequireAllocate(d_key, cache_elements * sizeof(__half));
    RequireAllocate(d_value, cache_elements * sizeof(__half));
    RequireAllocate(d_table, kMaxBlocks * sizeof(int32_t));
    RequireAllocate(d_context, sizeof(int32_t));
    RequireAllocate(d_out, query.size() * sizeof(__half));
    CUDA_CHECK(cudaMemcpy(d_query.data(), query_h.data(), query.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key.data(), key_h.data(), cache_elements * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value.data(), value_h.data(), cache_elements * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table.data(), block_table.data(), kMaxBlocks * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_context.data(), &context_len, sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    PagedAttentionKernelArgs args;
    args.query = d_query.data();
    args.key_cache = d_key.data();
    args.value_cache = d_value.data();
    args.block_tables = static_cast<const int32_t*>(d_table.data());
    args.context_lens = static_cast<const int32_t*>(d_context.data());
    args.output = d_out.data();
    args.batch_size = 1;
    args.num_heads = kHeads;
    args.num_kv_heads = kHeads;
    args.head_size = kHeadSize;
    args.block_size = kBlockSize;
    args.max_blocks_per_seq = kMaxBlocks;
    args.scale = static_cast<float>(scale);
    args.is_half = true;
    ASSERT_EQ(LaunchPagedAttention(args, nullptr), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> out_h(query.size());
    CUDA_CHECK(cudaMemcpy(out_h.data(), d_out.data(), query.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    const std::vector<float> actual = ToFloat(out_h);
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(WithinTolerance(static_cast<float>(expected[i]), actual[i], 1e-3f, 1e-3f))
            << "index " << i << " expected=" << expected[i] << " actual=" << actual[i];
    }
}

// FP16 的 split-K 分支（PG-3）。
//
// partial（m / l / acc）在 kernel 里**一律按 float 存**，FP16 只用于读写 K/V 与输出——
// 这条用例锁住"FP16 下不会退化成 FP16 累加"：退化的表现是误差随片数放大，而分成 2 片 /
// 8 片两种切法都用**同一份** double 参考裁决，所以切法本身不改变判据。
// 阈值沿用本文件既有的 FP16 算子口径（1e-3 / 1e-3）；该阈值的出处需作者确认后补注
// （开发计划 §12.1 的"反向查"）。
TEST(Fp16PathTest, PagedAttentionSplitMatchesFp16Reference) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kHeads = 2;
    constexpr int32_t kHeadSize = 8;
    constexpr int32_t kBlockSize = 8;
    constexpr int32_t kNumBlocks = 20;
    constexpr int32_t kMaxBlocks = 18;
    constexpr int32_t kContextLen = 137;  // 大于 kTargetChunk(128) → 自适应下切成 2 片

    const size_t cache_elements =
        static_cast<size_t>(kNumBlocks) * kBlockSize * kHeads * kHeadSize;
    std::vector<float> query(static_cast<size_t>(kHeads) * kHeadSize);
    std::vector<float> key_cache(cache_elements), value_cache(cache_elements);
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = DeterministicValue(static_cast<int64_t>(i), 0.0f);
    }
    for (size_t i = 0; i < cache_elements; ++i) {
        key_cache[i] = DeterministicValue(static_cast<int64_t>(i), 2.1f);
        value_cache[i] = DeterministicValue(static_cast<int64_t>(i), 4.3f);
    }
    // 物理块顺序与逻辑顺序不同：步长 7 与 20 互质 → 前 18 个物理块互不相同，
    // 确保 kernel 真的走块表寻址（同 FP32 用例的做法）。
    std::vector<int32_t> block_table(kMaxBlocks);
    for (int32_t i = 0; i < kMaxBlocks; ++i) {
        block_table[i] = (i * 7 + 3) % kNumBlocks;
    }

    const std::vector<__half> query_h = ToHalf(query);
    const std::vector<__half> key_h = ToHalf(key_cache);
    const std::vector<__half> value_h = ToHalf(value_cache);
    std::vector<double> query_d(query.size()), key_d(cache_elements), value_d(cache_elements);
    for (size_t i = 0; i < query.size(); ++i) {
        query_d[i] = __half2float(query_h[i]);
    }
    for (size_t i = 0; i < cache_elements; ++i) {
        key_d[i] = __half2float(key_h[i]);
        value_d[i] = __half2float(value_h[i]);
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
    std::vector<double> expected;
    test_support::ReferencePagedAttentionDecode(query_d, key_d, value_d, block_table, kContextLen,
                                                kHeads, kHeads, kHeadSize, kBlockSize, scale,
                                                &expected);

    const size_t workspace_bytes = PagedAttentionWorkspaceBytes(
        1, kHeads, kHeadSize, kPagedAttentionMaxSplits);
    ASSERT_GT(workspace_bytes, 0u);
    const size_t guard_floats = 32;
    const size_t workspace_alloc = workspace_bytes + guard_floats * sizeof(float);

    // 同一份输入、同一份参考，跑两遍不同切法：自适应（2 片）与强制 8 片（含空片）。
    auto run_once = [&]() {
        DeviceBuffer d_query(query.size() * sizeof(__half));
        DeviceBuffer d_key(cache_elements * sizeof(__half));
        DeviceBuffer d_value(cache_elements * sizeof(__half));
        DeviceBuffer d_table(kMaxBlocks * sizeof(int32_t));
        DeviceBuffer d_context(sizeof(int32_t));
        DeviceBuffer d_out(query.size() * sizeof(__half));
        DeviceBuffer d_workspace(workspace_alloc);
        RequireAllocate(d_query, query.size() * sizeof(__half));
        RequireAllocate(d_key, cache_elements * sizeof(__half));
        RequireAllocate(d_value, cache_elements * sizeof(__half));
        RequireAllocate(d_table, kMaxBlocks * sizeof(int32_t));
        RequireAllocate(d_context, sizeof(int32_t));
        RequireAllocate(d_out, query.size() * sizeof(__half));
        RequireAllocate(d_workspace, workspace_alloc);

        std::vector<float> workspace_host(workspace_alloc / sizeof(float),
                                          test_support::kWorkspaceGuardSentinel);
        CUDA_CHECK(cudaMemcpy(d_workspace.data(), workspace_host.data(), workspace_alloc,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_query.data(), query_h.data(), query.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_key.data(), key_h.data(), cache_elements * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_value.data(), value_h.data(), cache_elements * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table.data(), block_table.data(), kMaxBlocks * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_context.data(), &kContextLen, sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        PagedAttentionKernelArgs args;
        args.query = d_query.data();
        args.key_cache = d_key.data();
        args.value_cache = d_value.data();
        args.block_tables = static_cast<const int32_t*>(d_table.data());
        args.context_lens = static_cast<const int32_t*>(d_context.data());
        args.output = d_out.data();
        args.batch_size = 1;
        args.num_heads = kHeads;
        args.num_kv_heads = kHeads;
        args.head_size = kHeadSize;
        args.block_size = kBlockSize;
        args.max_blocks_per_seq = kMaxBlocks;
        args.scale = static_cast<float>(scale);
        args.is_half = true;
        ASSERT_EQ(LaunchPagedAttentionSplit(args, d_workspace.data(), workspace_bytes, nullptr),
                  cudaSuccess);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__half> out_h(query.size());
        CUDA_CHECK(cudaMemcpy(out_h.data(), d_out.data(), query.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        const std::vector<float> actual = ToFloat(out_h);
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_TRUE(
                WithinTolerance(static_cast<float>(expected[i]), actual[i], 1e-3f, 1e-3f))
                << "index " << i << " expected=" << expected[i] << " actual=" << actual[i];
        }

        // 护栏区必须原封不动：分片数上限就是 workspace 契约的上界
        std::vector<float> tail(guard_floats);
        CUDA_CHECK(cudaMemcpy(tail.data(),
                              static_cast<const char*>(d_workspace.data()) + workspace_bytes,
                              tail.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < tail.size(); ++i) {
            EXPECT_FLOAT_EQ(tail[i], test_support::kWorkspaceGuardSentinel);
        }
    };

    run_once();  // 自适应切法
    {
        // 强制 8 片对 137 个位置 → 每片约 17 个位置；再对 1 个位置强制 8 片制造空片
        test_support::ScopedSplitsOverride forced(kPagedAttentionMaxSplits);
        run_once();
    }
}

TEST(Fp16PathTest, GreedySamplerMatchesArgmaxOnFp16Logits) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kBatch = 3;
    constexpr int32_t kVocab = 512;

    std::vector<float> logits(static_cast<size_t>(kBatch) * kVocab);
    std::vector<int32_t> expected(kBatch, 0);
    for (int32_t b = 0; b < kBatch; ++b) {
        int32_t best = 0;
        for (int32_t v = 0; v < kVocab; ++v) {
            const float value = DeterministicValue(static_cast<int64_t>(b) * kVocab + v, 0.9f);
            logits[static_cast<size_t>(b) * kVocab + v] = value;
            if (v == 0 ||
                __half2float(__float2half_rn(value)) >
                    __half2float(__float2half_rn(logits[static_cast<size_t>(b) * kVocab + best]))) {
                best = v;
            }
        }
        expected[b] = best;
    }

    const std::vector<__half> logits_h = ToHalf(logits);
    DeviceBuffer d_logits(logits.size() * sizeof(__half));
    DeviceBuffer d_tokens(kBatch * sizeof(int32_t));
    RequireAllocate(d_logits, logits.size() * sizeof(__half));
    RequireAllocate(d_tokens, kBatch * sizeof(int32_t));
    CUDA_CHECK(cudaMemcpy(d_logits.data(), logits_h.data(), logits.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));

    SamplerArgs args;
    args.logits = d_logits.data();
    args.token_ids = static_cast<int32_t*>(d_tokens.data());
    args.batch_size = kBatch;
    args.vocab_size = kVocab;
    args.is_half = true;
    ASSERT_EQ(LaunchGreedySampler(args, nullptr), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> actual(kBatch);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_tokens.data(), kBatch * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, expected);
}

// S-12：Top-K 的 FP16 支路（补 future_iterations.md §11 的 P1.5-a 的另一半）。
//
// 判据沿用 S-8 的分布口径（3σ + 1e-3）：k = 3 / 6（vocab = 8）下**真的发生截断**，
// 参考是"限制在 top-k 内后重新归一化"的解析分布（`TopKSoftmaxProbabilities`）。
// 两条精度各自对参考，且互相在 3√2·σ 内（两个独立样本之差）。
// 为什么 k 不能取 vocab：那样退化成全词表 softmax，**截断路径根本没被走到**。
TEST(Fp16PathTest, TopKSamplingDistributionMatchesAnalyticProbabilitiesInFp16) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kVocab = 8;
    constexpr int32_t kDraws = 8000;

    const std::vector<float> logits{2.0f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -1.2f, -2.0f};
    const std::vector<__half> logits_h = ToHalf(logits);
    const std::vector<float> logits_as_half = ToFloat(logits_h);

    DeviceBuffer d_logits_fp32(logits.size() * sizeof(float));
    DeviceBuffer d_logits_fp16(logits_h.size() * sizeof(__half));
    RequireAllocate(d_logits_fp32, logits.size() * sizeof(float));
    RequireAllocate(d_logits_fp16, logits_h.size() * sizeof(__half));
    CUDA_CHECK(cudaMemcpy(d_logits_fp32.data(), logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_logits_fp16.data(), logits_h.data(), logits_h.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));

    for (int32_t k : {3, 6}) {
        const std::vector<double> expected_fp32 =
            test_support::TopKSoftmaxProbabilities(logits, k);
        const std::vector<double> expected_fp16 =
            test_support::TopKSoftmaxProbabilities(logits_as_half, k);
        const std::vector<int32_t> counts_fp32 =
            CollectTopKTokenCounts(d_logits_fp32.data(), /*is_half=*/false, kVocab, k, kDraws);
        const std::vector<int32_t> counts_fp16 =
            CollectTopKTokenCounts(d_logits_fp16.data(), /*is_half=*/true, kVocab, k, kDraws);

        for (int32_t v = 0; v < kVocab; ++v) {
            const double observed_fp32 = static_cast<double>(counts_fp32[v]) / kDraws;
            const double observed_fp16 = static_cast<double>(counts_fp16[v]) / kDraws;
            const double sigma_fp32 =
                std::sqrt(expected_fp32[v] * (1.0 - expected_fp32[v]) / kDraws);
            const double sigma_fp16 =
                std::sqrt(expected_fp16[v] * (1.0 - expected_fp16[v]) / kDraws);
            EXPECT_NEAR(observed_fp32, expected_fp32[v], 3.0 * sigma_fp32 + 1e-3)
                << "k=" << k << " fp32 token " << v;
            EXPECT_NEAR(observed_fp16, expected_fp16[v], 3.0 * sigma_fp16 + 1e-3)
                << "k=" << k << " fp16 token " << v;
            EXPECT_NEAR(observed_fp16, observed_fp32, 3.0 * std::sqrt(2.0) * sigma_fp16 + 1e-3)
                << "k=" << k << " fp16 vs fp32 token " << v;
        }
    }
}

// S-13：Top-P 的 FP16 支路（补 future_iterations.md §11 的 P1.5-a）。
//
// 判据沿用 S-8 的分布口径（3σ + 1e-3，出处见 test_sampler.cpp 的
// `TopKDistributionMatchesSoftmaxProbabilities` 注释）：FP16 分支的采样词频必须收敛到
// "截断 + 前缀内重新归一化"的解析分布，并且与 FP32 分支的词频也要互相落在 3σ 内。
// 参考实现用**各自精度下**的 logits（FP16 那份取舍入后的值），避免把输入量化误差算成
// kernel 误差——与本文件其它 FP16 用例同一纪律。
TEST(Fp16PathTest, TopPSamplingDistributionMatchesAnalyticProbabilitiesInFp16) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kVocab = 8;
    constexpr int32_t kDraws = 8000;
    constexpr float kP = 0.9f;

    const std::vector<float> logits{2.0f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -1.2f, -2.0f};
    const std::vector<__half> logits_h = ToHalf(logits);
    const std::vector<float> logits_as_half = ToFloat(logits_h);

    const std::vector<double> expected_fp32 =
        test_support::TruncatedSoftmaxProbabilities(logits, kP);
    const std::vector<double> expected_fp16 =
        test_support::TruncatedSoftmaxProbabilities(logits_as_half, kP);

    DeviceBuffer d_logits_fp32(logits.size() * sizeof(float));
    DeviceBuffer d_logits_fp16(logits_h.size() * sizeof(__half));
    RequireAllocate(d_logits_fp32, logits.size() * sizeof(float));
    RequireAllocate(d_logits_fp16, logits_h.size() * sizeof(__half));
    CUDA_CHECK(cudaMemcpy(d_logits_fp32.data(), logits.data(), logits.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_logits_fp16.data(), logits_h.data(), logits_h.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));

    const std::vector<int32_t> counts_fp32 =
        CollectTopPTokenCounts(d_logits_fp32.data(), /*is_half=*/false, kVocab, kP, kDraws);
    const std::vector<int32_t> counts_fp16 =
        CollectTopPTokenCounts(d_logits_fp16.data(), /*is_half=*/true, kVocab, kP, kDraws);

    for (int32_t v = 0; v < kVocab; ++v) {
        const double observed_fp32 = static_cast<double>(counts_fp32[v]) / kDraws;
        const double observed_fp16 = static_cast<double>(counts_fp16[v]) / kDraws;
        // 二项分布标准差 sqrt(p(1-p)/N)，取 3 sigma 作为容差（与 S-8 同一把尺子）
        const double sigma_fp32 =
            std::sqrt(expected_fp32[v] * (1.0 - expected_fp32[v]) / kDraws);
        const double sigma_fp16 =
            std::sqrt(expected_fp16[v] * (1.0 - expected_fp16[v]) / kDraws);
        EXPECT_NEAR(observed_fp32, expected_fp32[v], 3.0 * sigma_fp32 + 1e-3)
            << "fp32 token " << v;
        EXPECT_NEAR(observed_fp16, expected_fp16[v], 3.0 * sigma_fp16 + 1e-3)
            << "fp16 token " << v;
        // 两个独立样本之差的方差是各自方差之和；两种精度的解析概率本身只差 ~1e-5，
        // 由 1e-3 的常数项吸收。
        EXPECT_NEAR(observed_fp16, observed_fp32,
                    3.0 * std::sqrt(2.0) * sigma_fp16 + 1e-3)
            << "fp16 vs fp32 token " << v;
    }
}

}  // namespace mini_trt_llm
