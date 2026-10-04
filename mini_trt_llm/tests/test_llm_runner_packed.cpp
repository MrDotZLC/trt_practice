#include "gpt2_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/llm_runner.hpp"
#include "mini_trt_llm/kv_cache/paged_kv_cache.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::kBlockSize;
using test_support::kBlocksPerSeq;
using test_support::kHeadSize;
using test_support::kHeads;
using test_support::kLayers;
using test_support::kPositions;
using test_support::kVocab;

// 用例的 profile 必须允许批 > 1（`SmallGpt2BuilderConfig` 的默认是 1/1/1）。
constexpr int32_t kMaxBatch = 4;

PagedKVCache::Config PackedCacheConfig() {
    PagedKVCache::Config config;
    config.num_blocks = 16;
    config.block_size = kBlockSize;
    config.num_layers = kLayers;
    config.num_kv_heads = kHeads;
    config.head_size = kHeadSize;
    config.is_half = false;
    config.max_blocks_per_seq = kBlocksPerSeq;
    config.max_batch = kMaxBatch;
    return config;
}

std::vector<float> MakeKVRow(int32_t tokens, float base) {
    const size_t count = static_cast<size_t>(tokens) * kHeads * kHeadSize;
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = base + static_cast<float>(i);
    }
    return data;
}

std::vector<float> ReadBack(const void* device_ptr, size_t floats) {
    std::vector<float> host(floats);
    CUDA_CHECK(cudaMemcpy(host.data(), device_ptr, floats * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return host;
}

// 逻辑位置 (t, h, d) 在**某条序列自己的**块表里的偏移（独立算一遍，不复用产品代码）。
size_t ExpectedOffsetForSeq(const std::vector<int32_t>& table, int32_t t, int32_t h, int32_t d) {
    const int32_t physical = table[static_cast<size_t>(t) / kBlockSize];
    const int32_t slot = t % kBlockSize;
    return ((static_cast<size_t>(physical) * kBlockSize + slot) * kHeads + h) * kHeadSize + d;
}

std::string SequenceToString(const std::vector<int64_t>& tokens) {
    std::string out = "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(tokens[i]);
    }
    return out + "]";
}

// ---------------------------------------------------------------------------
// cache 层：packed 源 + 每行不等长的写回
// ---------------------------------------------------------------------------

// `PackedWriteBackMapsCorrectly`：packed 源（行长不等）+ 行映射的写回，必须落到**各序列自己的块**，
// 且每行的长度由 `row_lengths`/`cu_seqlens_ctx` 决定 —— 通用 kernel 的"统一 stride × 行号"
// 在 packed 下不成立，所以走的是"一个 block 一行"的专用 kernel。
TEST(LlmRunnerPackedTest, PackedWriteBackMapsCorrectly) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PagedKVCache cache(PackedCacheConfig());
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/7, /*max_tokens=*/10));  // 第 0 行
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/9, /*max_tokens=*/10));  // 第 1 行

    // packed 源：第 0 条 2 个 token、第 1 条 3 个 token，首尾相接（共 5 个 token）
    constexpr int32_t kLen0 = 2;
    constexpr int32_t kLen1 = 3;
    const std::vector<float> row0 = MakeKVRow(kLen0, 100.0f);
    const std::vector<float> row1 = MakeKVRow(kLen1, 900.0f);
    std::vector<float> packed;
    packed.insert(packed.end(), row0.begin(), row0.end());
    packed.insert(packed.end(), row1.begin(), row1.end());

    DeviceBuffer d_packed(packed.size() * sizeof(float));
    ASSERT_TRUE(d_packed.Allocate(packed.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_packed.data(), packed.data(), d_packed.size(),
                          cudaMemcpyHostToDevice));
    // 段内前缀和 [0, 2, 5]：与 row_lengths 同源（写回 kernel 按它取每行的源区间）
    const std::vector<int32_t> host_cu_seqlens = {0, kLen0, kLen0 + kLen1};
    DeviceBuffer d_cu_seqlens(host_cu_seqlens.size() * sizeof(int32_t));
    ASSERT_TRUE(d_cu_seqlens.Allocate(d_cu_seqlens.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_cu_seqlens.data(), host_cu_seqlens.data(), d_cu_seqlens.size(),
                          cudaMemcpyHostToDevice));

    const std::vector<int32_t> rows = {0, 1};
    const std::vector<int32_t> lengths = {kLen0, kLen1};
    ASSERT_EQ(cache.WritePrefillKV(/*layer=*/0, d_packed.data(), d_packed.data(),
                                   /*tokens=*/kLen1 /* 形状占位：packed 下按 row_lengths 校验 */,
                                   rows.data(), 2, lengths.data(), nullptr,
                                   static_cast<const int32_t*>(d_cu_seqlens.data()), 2),
              cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> layer =
        ReadBack(cache.key_cache(0), cache.bytes_per_layer() / sizeof(float));
    std::vector<int32_t> table0;
    std::vector<int32_t> table1;
    ASSERT_TRUE(cache.GetBlockTable(7, &table0));
    ASSERT_TRUE(cache.GetBlockTable(9, &table1));
    for (int32_t t = 0; t < kLen0; ++t) {
        for (int32_t h = 0; h < kHeads; ++h) {
            for (int32_t d = 0; d < kHeadSize; ++d) {
                const size_t src = (static_cast<size_t>(h) * kLen0 + t) * kHeadSize + d;
                EXPECT_FLOAT_EQ(layer[ExpectedOffsetForSeq(table0, t, h, d)], row0[src])
                    << "seq 7 t=" << t << " h=" << h << " d=" << d;
            }
        }
    }
    for (int32_t t = 0; t < kLen1; ++t) {
        for (int32_t h = 0; h < kHeads; ++h) {
            for (int32_t d = 0; d < kHeadSize; ++d) {
                const size_t src = (static_cast<size_t>(h) * kLen1 + t) * kHeadSize + d;
                EXPECT_FLOAT_EQ(layer[ExpectedOffsetForSeq(table1, t, h, d)], row1[src])
                    << "seq 9 t=" << t << " h=" << h << " d=" << d;
            }
        }
    }
    // 每行长度各自不同 —— 语境长度必须逐行对（统一 stride 的写法会在这里露馅）
    EXPECT_EQ(cache.SequenceLength(7), kLen0);
    EXPECT_EQ(cache.SequenceLength(9), kLen1);
}

// ---------------------------------------------------------------------------
// runner 层：packed 混合批
// ---------------------------------------------------------------------------

struct PackedFixture {
    Logger logger;
    test_support::ModelDirectory directory = test_support::ModelDirectory::Create("llm_packed");
    std::shared_ptr<Engine> engine;
    std::unique_ptr<LLMRunner> runner;
    bool ok = false;
};

// packed 模式只建**一个**引擎（那张 packed 图），并按约定把它同时放进 prefill/decode 两个槽位
// （runner 的构造期校验要求两个 shared_ptr 非空；packed 路径只用 prefill 那个）。
PackedFixture MakePackedFixture(const std::string& engine_prefix, int32_t max_batch,
                                int32_t eos_token_id = -1, int32_t num_blocks = 16) {
    PackedFixture fixture;
    if (!fixture.directory.valid()) {
        return fixture;
    }
    if (!fixture.directory.WriteConfig(test_support::SmallGpt2ConfigJson()) ||
        !fixture.directory.WriteWeights(test_support::SmallGpt2Weights())) {
        return fixture;
    }
    EngineBuilder::Config builder_config = test_support::SmallGpt2BuilderConfig(kMaxBatch);
    builder_config.packed_mixed_prefill = true;
    EngineBuilder builder(fixture.logger, builder_config);
    const std::string engine_path = fixture.directory.EnginePath(engine_prefix + "_packed.engine");
    // packed 图只对 prefill 阶段有意义（模型构建器会拒 decode）
    if (!builder.BuildFromConfig(fixture.directory.path(), engine_path, BuildStage::kPrefill)) {
        return fixture;
    }
    fixture.engine = std::make_shared<Engine>(engine_path, fixture.logger);

    LLMRunner::Config runner_config;
    runner_config.num_layers = kLayers;
    runner_config.num_kv_heads = kHeads;
    runner_config.head_size = kHeadSize;
    runner_config.block_size = kBlockSize;
    runner_config.max_blocks_per_seq = kBlocksPerSeq;
    runner_config.num_blocks = num_blocks;
    runner_config.is_half = false;
    runner_config.vocab_size = kVocab;
    runner_config.max_batch = max_batch;
    runner_config.eos_token_id = eos_token_id;
    // S5：packed 模式下 runner 必须知道位置表长度（引擎侧查不到，见 Config::max_positions）；
    // 本夹具的模型就是 kPositions 个位置。
    runner_config.max_positions = kPositions;
    // S5：同理必填 —— 单步每序列的 token 上界（= 建图时的 max_prefill_seq_len，本夹具也是 kPositions）。
    runner_config.max_prefill_seq_len = kPositions;
    runner_config.prefill_mode = LLMRunner::Config::PrefillMode::kPackedMixed;
    fixture.runner = std::make_unique<LLMRunner>(runner_config, fixture.engine, fixture.engine,
                                                 nullptr);
    fixture.ok = fixture.runner->ok();
    return fixture;
}

LLMRunner::GenerateOptions GreedyOptions(int32_t max_new_tokens) {
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.top_k = 1;
    return options;
}

LLMRunner::GenerateOptions TopPOptions(int32_t max_new_tokens, float top_p, uint64_t seed) {
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.top_p = top_p;  // < 1 → Top-P 分支
    options.top_k = 1;
    options.seed = seed;
    return options;
}

LLMRunner::SchedulerRequest MakeRequest(const std::vector<int64_t>& prompt,
                                        const LLMRunner::GenerateOptions& options,
                                        int32_t arrival_step) {
    LLMRunner::SchedulerRequest request;
    request.request.input_ids = prompt;
    request.request.options = options;
    request.arrival_step = arrival_step;
    return request;
}

// packed 路径的"逐条单跑"参考实现 = **单请求的 RunScheduler**
// （packed 模式下 GenerateBatch/Generate 会明确拒绝：它们是 padding 路径的入口）。
std::vector<int64_t> PackedReference(LLMRunner* runner, const std::vector<int64_t>& prompt,
                                     const LLMRunner::GenerateOptions& options) {
    const std::vector<LLMRunner::GenerateResult> single =
        runner->RunScheduler({MakeRequest(prompt, options, 0)});
    if (single.size() != 1) {
        return {};
    }
    return single.front().tokens;
}

void ExpectMatchesPackedReference(LLMRunner* runner,
                                  const std::vector<LLMRunner::SchedulerRequest>& requests,
                                  const std::vector<LLMRunner::GenerateResult>& scheduled) {
    ASSERT_EQ(scheduled.size(), requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const std::vector<int64_t> alone =
            PackedReference(runner, requests[i].request.input_ids, requests[i].request.options);
        std::cout << "[诊断] 第 " << i << " 条 packed批="
                  << SequenceToString(scheduled[i].tokens)
                  << " 单跑=" << SequenceToString(alone) << "\n";
        ASSERT_EQ(scheduled[i].tokens.size(), alone.size()) << "第 " << i << " 条长度不同";
        for (size_t k = 0; k < alone.size(); ++k) {
            EXPECT_EQ(scheduled[i].tokens[k], alone[k]) << "第 " << i << " 条第 " << k << " 个";
        }
    }
}

// AC1 在 packed 路径**内部**成立：packed 批跑 == 单请求跑（逐位）。
TEST(LlmRunnerPackedTest, PackedEqualsSequential) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PackedFixture fixture = MakePackedFixture("packed_ac1", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, TopPOptions(5, 0.9f, 101), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10}, TopPOptions(5, 0.9f, 202), 0));

    const std::vector<LLMRunner::GenerateResult> scheduled =
        fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    ExpectMatchesPackedReference(fixture.runner.get(), requests, scheduled);
}

// **S4 的核心场景**：同一步里既有新入批的 context 行、又有在跑的 generation 行。
// 做法：第二条请求晚到 → 它入批的那一步，第一条已经在 generation 上。
TEST(LlmRunnerPackedTest, MixedStepContextAndGeneration) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PackedFixture fixture = MakePackedFixture("packed_mixed", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(5), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10, 11}, GreedyOptions(5), 2));  // 晚到 → 混合步

    const std::vector<LLMRunner::GenerateResult> scheduled =
        fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    const LLMRunner::SchedulerStats stats = fixture.runner->scheduler_stats();
    EXPECT_EQ(stats.context_rows, 2) << "两条各 prefill 一次";
    EXPECT_GT(stats.generation_rows, 0) << "应当出现在跑的行";
    ExpectMatchesPackedReference(fixture.runner.get(), requests, scheduled);
}

// 打包顺序：**context token 必须排在 generation token 之前**。
//
// **判据是"由构造保证 + 由结果兜住"**：packed 顺序由 runner 自己生成（布局是它的产物），
// 所以外部没有"传错顺序"的入口 —— 顺序写反会直接表现为 generation 行读到别人的缓存
// （位置/长度/块表全错位），token 立刻与单跑不一致。这里用一个对它**敏感**的场景
// （两条序列的 prompt 内容与长度都不同）把这条锁住，而不是假装有一个入口校验。
TEST(LlmRunnerPackedTest, ContextTokensPrecedeGeneration) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PackedFixture fixture = MakePackedFixture("packed_order", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(6), 0));
    requests.push_back(MakeRequest({12, 13, 14, 15, 16, 17}, GreedyOptions(6), 1));
    const std::vector<LLMRunner::GenerateResult> scheduled =
        fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    ExpectMatchesPackedReference(fixture.runner.get(), requests, scheduled);
}

// `cu_seqlens_ctx` 的两个极端：`context_seq_count == 0`（纯 generation 步）与
// `== B_total`（纯 context 步）。两条请求同一步到齐 → 首步是纯 context；
// 之后没有新到达 → 其余步是纯 generation。用聚合统计把"两种步都发生过"钉住。
TEST(LlmRunnerPackedTest, CuSeqlensBoundaryCases) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PackedFixture fixture = MakePackedFixture("packed_boundary", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    constexpr int32_t kMaxNew = 4;
    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(kMaxNew), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10}, GreedyOptions(kMaxNew), 0));

    const std::vector<LLMRunner::GenerateResult> scheduled =
        fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    const LLMRunner::SchedulerStats stats = fixture.runner->scheduler_stats();
    EXPECT_EQ(stats.context_rows, 2) << "首步是纯 context（两条一起入批）";
    // 两条各 kMaxNew 个 token：首步在 context 段采第 0 个，其余 kMaxNew-1 步在 generation 段。
    EXPECT_EQ(stats.generation_rows, 2 * (kMaxNew - 1))
        << "首步必须没有 generation 行（否则说明分段把两相混进了同一步）";
    ExpectMatchesPackedReference(fixture.runner.get(), requests, scheduled);
}

// 元数据（`block_tables` / `context_lens`）必须**按 packed 行序**重建。
//
// **同样只能"由结果兜住"**：runner 不暴露内部缓冲。这条用一个对行序敏感的场景
// （两条序列的块表与语境长度都不同）锁住 —— 照 S3 那样直传缓存镜像会让行号错位，
// generation 行于是读到别人的块，token 立刻与单跑不一致。
TEST(LlmRunnerPackedTest, PackedMetadataFollowsPackedOrder) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PackedFixture fixture = MakePackedFixture("packed_meta", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6, 7, 8}, GreedyOptions(5), 0));  // 长 prompt
    requests.push_back(MakeRequest({9, 10}, GreedyOptions(5), 2));             // 短 prompt、晚到
    const std::vector<LLMRunner::GenerateResult> scheduled =
        fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    ExpectMatchesPackedReference(fixture.runner.get(), requests, scheduled);
}

// AC7 的**可观测部分**：同批长度差很大时，短序列的结果不受最长序列影响。
// （"算力不随最长序列增长"是 packed 的结构性属性——它只算 T 个 token；
//   **代价**那条要 P4/P7 实测，不在本用例的判据里。）
TEST(LlmRunnerPackedTest, PackedShortSequenceNotPenalized) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PackedFixture fixture = MakePackedFixture("packed_short", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4}, GreedyOptions(4), 0));                       // 很短
    requests.push_back(MakeRequest({5, 6, 7, 8, 9, 10, 11, 12}, GreedyOptions(4), 0));  // 长
    const std::vector<LLMRunner::GenerateResult> scheduled =
        fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    ExpectMatchesPackedReference(fixture.runner.get(), requests, scheduled);
}

// AC8：两条路径**各自**成立、可回退（**不要求跨路径逐位相同** —— kernel 不同、
// 浮点累加顺序不同，见 design.md D13）。packed 侧由本文件其它用例覆盖，这里验回退路径。
TEST(LlmRunnerPackedTest, FallbackSwitchKeepsResults) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    Logger logger;
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("llm_packed_fallback");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(test_support::SmallGpt2ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(test_support::SmallGpt2Weights()));
    EngineBuilder builder(logger, test_support::SmallGpt2BuilderConfig(/*max_batch=*/2));
    const std::string prefill_path = directory.EnginePath("fallback_prefill.engine");
    const std::string decode_path = directory.EnginePath("fallback_decode.engine");
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), prefill_path, BuildStage::kPrefill));
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), decode_path, BuildStage::kDecode));
    auto prefill = std::make_shared<Engine>(prefill_path, logger);
    auto decode = std::make_shared<Engine>(decode_path, logger);

    LLMRunner::Config runner_config;
    runner_config.num_layers = kLayers;
    runner_config.num_kv_heads = kHeads;
    runner_config.head_size = kHeadSize;
    runner_config.block_size = kBlockSize;
    runner_config.max_blocks_per_seq = kBlocksPerSeq;
    runner_config.num_blocks = 16;
    runner_config.is_half = false;
    runner_config.vocab_size = kVocab;
    runner_config.max_batch = 2;
    runner_config.prefill_mode = LLMRunner::Config::PrefillMode::kPaddedTwoPhase;  // 回退路径
    LLMRunner runner(runner_config, prefill, decode, nullptr);
    ASSERT_TRUE(runner.ok());

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(4), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10}, GreedyOptions(4), 0));
    const std::vector<LLMRunner::GenerateResult> scheduled = runner.RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    // 回退路径的逐条参考 = 它自己的单请求 RunScheduler（padding 路径内部对拍）
    for (size_t i = 0; i < requests.size(); ++i) {
        const std::vector<LLMRunner::GenerateResult> single = runner.RunScheduler(
            {MakeRequest(requests[i].request.input_ids, requests[i].request.options, 0)});
        ASSERT_EQ(single.size(), 1u);
        ASSERT_EQ(scheduled[i].tokens.size(), single[0].tokens.size());
        for (size_t k = 0; k < single[0].tokens.size(); ++k) {
            EXPECT_EQ(scheduled[i].tokens[k], single[0].tokens[k])
                << "第 " << i << " 条第 " << k << " 个 token";
        }
    }
}

}  // namespace
}  // namespace mini_trt_llm
