#include "mini_trt_llm/plugins/paged_attention_kernel.hpp"
#include "mini_trt_llm/plugins/rope_kernel.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
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

}  // namespace mini_trt_llm
