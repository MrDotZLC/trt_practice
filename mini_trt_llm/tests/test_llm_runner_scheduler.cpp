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
using test_support::kVocab;

// 调度用例的 profile 必须允许批 > 1：`SmallGpt2BuilderConfig()` 的默认是 1/1/1（单序列用例的口径），
// 那种 profile 下批 2 的 prefill/decode 直接落在形状范围之外。这里显式传批上限
// （公共 fixture 已支持该参数，S1 的批量用例同样受益）。
constexpr int32_t kMaxBatch = 4;

EngineBuilder::Config SchedulerBuilderConfig() {
    return test_support::SmallGpt2BuilderConfig(kMaxBatch);
}

PagedKVCache::Config SchedulerCacheConfig() {
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

// 构造可辨识的 K/V：同一份基址 + 逐元素偏移，便于区分"写错行"与"写错位置"。
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

// 逻辑位置 (t, h, d) 在**某条序列自己的**块表里的偏移：按块表 + 块内槽位独立算一遍，
// 不复用产品代码的算法（否则是自我印证）。
size_t ExpectedOffsetForSeq(const std::vector<int32_t>& table, int32_t t, int32_t h, int32_t d) {
    const int32_t physical = table[static_cast<size_t>(t) / kBlockSize];
    const int32_t slot = t % kBlockSize;
    return ((static_cast<size_t>(physical) * kBlockSize + slot) * kHeads + h) * kHeadSize + d;
}

// ---------------------------------------------------------------------------
// 缓存层面的两条：守门用例与行映射
// ---------------------------------------------------------------------------

// **S3 路线修正的守门用例**（p5_s3_interface_spec §8）。
//
// 锁的性质：本步的 context 段只装**新入批**的序列，其它序列（正在 generation 的那些）的
// K/V 必须逐位不变 —— 整批跑 context 会把它们算出无意义的 K/V 再写回它们自己的块，
// 把真实 prompt K/V 覆盖掉（D12 的"静默算错"）。这里在 PagedKVCache 层直接验：
// 只映射到第 1 行的写回，第 0 行的每个字节都不许动。
//
// 说明：runner 层面没有"读回 cache"的观测口（cache 是 runner 私有的），
// 所以这条锁的是**写回机制**；"调度器确实只装新入批的行"由
// `BatchEqualsSequentialUnderScheduling` 从结果侧兜住（真按整批跑，正在跑的序列
// 后续 token 就会与逐条单跑不一致）。
TEST(LlmRunnerSchedulerTest, ContextPassDoesNotTouchInactiveSequences) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PagedKVCache cache(SchedulerCacheConfig());
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/0, /*max_tokens=*/10));  // 已在跑（第 0 行）
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/1, /*max_tokens=*/10));  // 本步新入批（第 1 行）

    constexpr int32_t kTokens = 4;
    const std::vector<float> first = MakeKVRow(kTokens, 100.0f);
    const std::vector<float> second = MakeKVRow(kTokens, 900.0f);
    DeviceBuffer d_first(first.size() * sizeof(float));
    DeviceBuffer d_second(second.size() * sizeof(float));
    ASSERT_TRUE(d_first.Allocate(first.size() * sizeof(float)));
    ASSERT_TRUE(d_second.Allocate(second.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_first.data(), first.data(), d_first.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_second.data(), second.data(), d_second.size(),
                          cudaMemcpyHostToDevice));

    const std::vector<int32_t> lengths = {kTokens};
    const std::vector<int32_t> rows_first = {0};
    ASSERT_EQ(cache.WritePrefillKV(/*layer=*/0, d_first.data(), d_first.data(), kTokens,
                                   rows_first.data(), 1, lengths.data(), nullptr),
              cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
    const size_t layer_floats = cache.bytes_per_layer() / sizeof(float);
    const std::vector<float> before = ReadBack(cache.key_cache(0), layer_floats);

    // 本步的 context 段：引擎第 0 行 → 缓存第 1 行
    const std::vector<int32_t> rows_second = {1};
    ASSERT_EQ(cache.WritePrefillKV(/*layer=*/0, d_second.data(), d_second.data(), kTokens,
                                   rows_second.data(), 1, lengths.data(), nullptr),
              cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> after = ReadBack(cache.key_cache(0), layer_floats);

    // 已登记序列自己的块里，一个字节都不许变
    for (size_t i = 0; i < before.size(); ++i) {
        ASSERT_EQ(before[i], after[i]) << "第 0 行的第 " << i << " 个元素被本次写回改了";
    }
    // 反向自证：这次写回确实写了东西（否则"没变"是空断言）
    bool wrote_second = false;
    for (float v : after) {
        if (v == second[0]) {
            wrote_second = true;
        }
    }
    EXPECT_TRUE(wrote_second) << "第 1 行没有被写进去 —— 这条用例失去判别力";
}

// 行映射真的起作用：`rows[i]` 必须与 `RowOf()` 一致；故意错位时 K/V 会落到**别人**的块里。
TEST(LlmRunnerSchedulerTest, WriteBackRowsMapCorrectly) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PagedKVCache cache(SchedulerCacheConfig());
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/7, /*max_tokens=*/10));
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/9, /*max_tokens=*/10));
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/11, /*max_tokens=*/10));

    // 行号来源必须是显式查询，不是"追加一定在尾部"的隐式约定
    EXPECT_EQ(cache.RowOf(7), 0);
    EXPECT_EQ(cache.RowOf(9), 1);
    EXPECT_EQ(cache.RowOf(11), 2);
    EXPECT_EQ(cache.RowOf(4242), -1) << "未登记的序列必须回 -1，而不是撞到某一行";

    constexpr int32_t kTokens = 2;
    const std::vector<float> source = MakeKVRow(kTokens, 500.0f);
    DeviceBuffer d_source(source.size() * sizeof(float));
    ASSERT_TRUE(d_source.Allocate(source.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_source.data(), source.data(), d_source.size(),
                          cudaMemcpyHostToDevice));

    // 引擎第 0 行 → 缓存第 2 行（= seq 11）。错位就是缺陷：这条路径正是"拿残留行覆盖别人"的入口。
    const std::vector<int32_t> rows = {2};
    const std::vector<int32_t> lengths = {kTokens};
    ASSERT_EQ(cache.WritePrefillKV(/*layer=*/0, d_source.data(), d_source.data(), kTokens,
                                   rows.data(), 1, lengths.data(), nullptr),
              cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> table;
    ASSERT_TRUE(cache.GetBlockTable(11, &table));
    const std::vector<float> layer =
        ReadBack(cache.key_cache(0), cache.bytes_per_layer() / sizeof(float));
    // 逐元素核对落点：t/h/d → 块表第 2 行（row_slot = 2）
    for (int32_t t = 0; t < kTokens; ++t) {
        for (int32_t h = 0; h < kHeads; ++h) {
            for (int32_t d = 0; d < kHeadSize; ++d) {
                const size_t src =
                    ((static_cast<size_t>(0) * kHeads + h) * kTokens + t) * kHeadSize + d;
                const size_t offset = ExpectedOffsetForSeq(table, t, h, d);
                EXPECT_FLOAT_EQ(layer[offset], source[src])
                    << "t=" << t << " h=" << h << " d=" << d;
            }
        }
    }
    EXPECT_EQ(cache.SequenceLength(11), kTokens) << "被映射行的长度必须跟着写回走";
    EXPECT_EQ(cache.SequenceLength(7), 0) << "没参与本次写回的行长度必须不变";
    EXPECT_EQ(cache.SequenceLength(9), 0);
}

// ---------------------------------------------------------------------------
// runner 层面的六条：调度循环本体
// ---------------------------------------------------------------------------

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

// 小模型 + prefill/decode 引擎 + 一个 runner。成员声明顺序即析构顺序的逆序。
struct SchedulerFixture {
    Logger logger;
    test_support::ModelDirectory directory = test_support::ModelDirectory::Create("llm_sched");
    std::shared_ptr<Engine> prefill;
    std::shared_ptr<Engine> decode;
    std::unique_ptr<LLMRunner> runner;
    bool ok = false;
};

SchedulerFixture MakeSchedulerFixture(const std::string& engine_prefix, int32_t max_batch,
                                      int32_t eos_token_id = -1, int32_t num_blocks = 8) {
    SchedulerFixture fixture;
    if (!fixture.directory.valid()) {
        return fixture;
    }
    if (!fixture.directory.WriteConfig(test_support::SmallGpt2ConfigJson()) ||
        !fixture.directory.WriteWeights(test_support::SmallGpt2Weights())) {
        return fixture;
    }
    EngineBuilder builder(fixture.logger, SchedulerBuilderConfig());
    const std::string prefill_path = fixture.directory.EnginePath(engine_prefix + "_prefill.engine");
    const std::string decode_path = fixture.directory.EnginePath(engine_prefix + "_decode.engine");
    if (!builder.BuildFromConfig(fixture.directory.path(), prefill_path, BuildStage::kPrefill) ||
        !builder.BuildFromConfig(fixture.directory.path(), decode_path, BuildStage::kDecode)) {
        return fixture;
    }
    fixture.prefill = std::make_shared<Engine>(prefill_path, fixture.logger);
    fixture.decode = std::make_shared<Engine>(decode_path, fixture.logger);

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
    fixture.runner = std::make_unique<LLMRunner>(runner_config, fixture.prefill, fixture.decode,
                                                 nullptr);
    fixture.ok = fixture.runner->ok();
    return fixture;
}

LLMRunner::GenerateOptions GreedyOptions(int32_t max_new_tokens) {
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.top_k = 1;  // greedy：与随机流无关
    return options;
}

LLMRunner::GenerateOptions TopPOptions(int32_t max_new_tokens, float top_p, uint64_t seed) {
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.top_p = top_p;   // < 1 → 走 Top-P 分支（与 S1 的用例同一约定）
    options.top_k = 1;       // 入口要求 top_k ≥ 1；Top-P 分支里它不参与
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

// 逐条单独跑：AC1 的参考实现（不变量：批量路径必须与它逐位相同）。
std::vector<int64_t> SequentialReference(LLMRunner* runner,
                                         const std::vector<int64_t>& prompt,
                                         const LLMRunner::GenerateOptions& options) {
    return runner->Generate(prompt, options);
}

// 每条序列的 token 必须与"逐条单跑"逐位相同。
void ExpectMatchesSequential(LLMRunner* runner,
                             const std::vector<LLMRunner::SchedulerRequest>& requests,
                             const std::vector<LLMRunner::GenerateResult>& scheduled) {
    ASSERT_EQ(scheduled.size(), requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const std::vector<int64_t> alone = SequentialReference(
            runner, requests[i].request.input_ids, requests[i].request.options);
        std::cout << "[诊断] 第 " << i << " 条 调度=" << SequenceToString(scheduled[i].tokens)
                  << "（ok=" << (scheduled[i].ok ? "true" : "false") << "）单跑="
                  << SequenceToString(alone) << "\n";
        ASSERT_EQ(scheduled[i].tokens.size(), alone.size()) << "第 " << i << " 条长度不同";
        for (size_t k = 0; k < alone.size(); ++k) {
            EXPECT_EQ(scheduled[i].tokens[k], alone[k]) << "第 " << i << " 条第 " << k << " 个 token";
        }
    }
}

// 一条退出后其余序列的行号必须前移（压实），且不影响它们的数值结果。
// 行号压实本身没有对外观测口（cache 是 runner 私有的），这里用**结果**兜住：
// 压实/行映射只要错一格，幸存序列后续的 decode 就会读到别人的 K/V，token 立刻与单跑不一致。
TEST(LlmRunnerSchedulerTest, SequenceRetiresAndRowCompacts) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    SchedulerFixture fixture = MakeSchedulerFixture("sched_retire", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    // 三条请求：max_batch = 2，所以必须"退出一条才能进下一条"（正是要验的路径）。
    // 第一条只生成 2 个 token，比另外两条短 —— 它先退出，后两条的行号随之压实。
    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(/*max_new_tokens=*/2), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10}, GreedyOptions(/*max_new_tokens=*/5), 0));
    requests.push_back(MakeRequest({11, 12, 13, 14}, GreedyOptions(/*max_new_tokens=*/5), 0));

    const int32_t free_before = fixture.runner->NumFreeKvBlocks();
    const std::vector<LLMRunner::GenerateResult> scheduled = fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());

    ExpectMatchesSequential(fixture.runner.get(), requests, scheduled);
    EXPECT_EQ(fixture.runner->NumFreeKvBlocks(), free_before) << "跑完后块必须全部归还";
}

// 守门用例的 **runner 层同伴**（对应 p5_s3 §8 的 `ContextPassDoesNotTouchInactiveSequences`）。
//
// cache 层那条锁的是"写回不会碰别的行"（逐字节）；这条从**调度怎么走**的角度锁同一件事：
// 第二条请求晚到，它入批那一步的 context 段只能装**它自己**，不能把正在 generation 的第一条
// 也算一遍 —— 真按整批跑的话 `context_rows` 会变成 3（1 + 2）而不是 2。
// 这正是 D12 推翻"固定槽位 + 全批定长"的理由：整批跑会把正在跑的行的 prompt K/V 覆盖掉。
TEST(LlmRunnerSchedulerTest, ContextSegmentOnlyCoversNewRows) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    SchedulerFixture fixture = MakeSchedulerFixture("sched_ctx_rows", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(/*max_new_tokens=*/4), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10}, GreedyOptions(/*max_new_tokens=*/4), 2));

    const std::vector<LLMRunner::GenerateResult> scheduled = fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    const LLMRunner::SchedulerStats stats = fixture.runner->scheduler_stats();
    std::cout << "[诊断] context_rows=" << stats.context_rows
              << " prefill_calls=" << stats.prefill_calls << " decode_calls=" << stats.decode_calls
              << " max_active=" << stats.max_active << "\n";
    EXPECT_EQ(stats.context_rows, 2) << "只该写回两条各自的 prompt（整批跑会变成 3）";
    EXPECT_EQ(stats.prefill_calls, 2) << "两次准入各一段 context";
    EXPECT_GE(stats.decode_calls, 1) << "第一条在第二条入批前应当已经在 generation 上跑过";
    EXPECT_EQ(stats.max_active, 2);
    ExpectMatchesSequential(fixture.runner.get(), requests, scheduled);
}

// AC2：批内 prompt 长度不齐（右填充 + padding mask）时，每条序列的位置与语境长度都要对。
TEST(LlmRunnerSchedulerTest, UnequalPromptLengthsInFlight) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    SchedulerFixture fixture = MakeSchedulerFixture("sched_unequal", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(/*max_new_tokens=*/4), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10, 11, 12}, GreedyOptions(/*max_new_tokens=*/4), 0));

    const std::vector<LLMRunner::GenerateResult> scheduled = fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    // 短的那条必须只按自己的长度算：采样的末位是第 3 个位置，不是 stride-1（填充位置）
    ExpectMatchesSequential(fixture.runner.get(), requests, scheduled);
}

// 采到 EOS 的序列按 EOS 收口：结果与"逐条单跑"的截断口径一致，且不影响同批其它序列。
//
// "提前退出"的**时刻**用观测口判：`max_batch = 1` 让第二条必须等第一条腾出位置。
//   * 提前退出：A 在第 1 步采到 EOS、第 2 步退出 → B 第 2 步入批 → 总步数 ≈ 7（≤ 8）；
//   * 若"跑满 max_new 再截断"：A 要占满 5 步，B 第 6 步才入批 → 总步数 ≈ 11。
// 结果是等价的（EOS 之后的 token 都被截掉），所以**只有步数能区分这两种实现**。
TEST(LlmRunnerSchedulerTest, EosRetiresImmediately) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // 先跑一次贪心，拿到该 prompt 的首个 token，再把它设成 EOS
    SchedulerFixture probe = MakeSchedulerFixture("sched_eos_probe", /*max_batch=*/1);
    ASSERT_TRUE(probe.ok);
    const std::vector<int64_t> prompt = {3, 4, 5, 6};
    const LLMRunner::GenerateOptions options = GreedyOptions(/*max_new_tokens=*/5);
    const std::vector<int64_t> greedy = probe.runner->Generate(prompt, options);
    ASSERT_FALSE(greedy.empty());
    const int32_t eos_token = static_cast<int32_t>(greedy.front());

    SchedulerFixture fixture = MakeSchedulerFixture("sched_eos", /*max_batch=*/1, eos_token);
    ASSERT_TRUE(fixture.ok);
    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest(prompt, options, 0));                    // 第一个 token 就是 EOS
    requests.push_back(MakeRequest({7, 8, 9, 10}, options, 0));             // 同批的正常序列

    const std::vector<LLMRunner::GenerateResult> scheduled = fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    const LLMRunner::SchedulerStats stats = fixture.runner->scheduler_stats();
    std::cout << "[诊断] EOS 用例步数=" << stats.steps << "（提前退出应 ≤ 8；不退出则 ≈ 11）\n";
    EXPECT_LE(stats.steps, 8) << "EOS 没有让序列提前退出（位子没腾出来，B 只能等 A 跑满）";
    EXPECT_EQ(stats.max_active, 1) << "max_batch = 1：同时只能有一条";
    EXPECT_EQ(stats.context_rows, 2) << "两条各 prefill 一次";
    // EOS 截断口径与 S1 一致：末尾那个 EOS 不进结果（所以这条序列的结果是空的、ok = false）
    EXPECT_TRUE(scheduled[0].tokens.empty()) << "采到 EOS 的序列不该把 EOS 交出去";
    EXPECT_FALSE(scheduled[0].ok);
    // 同批的另一条不受影响
    const std::vector<int64_t> alone =
        SequentialReference(fixture.runner.get(), requests[1].request.input_ids,
                            requests[1].request.options);
    ASSERT_EQ(scheduled[1].tokens.size(), alone.size());
    for (size_t k = 0; k < alone.size(); ++k) {
        EXPECT_EQ(scheduled[1].tokens[k], alone[k]) << "第 " << k << " 个 token";
    }
}

// 调度必须可复现：同一组请求换一组 arrival_step，逐条结果逐位相同。
// 这条同时锁住"随机步号逐行"——标量 offset 下，到得晚的请求会拿到不同的随机流。
TEST(LlmRunnerSchedulerTest, DeterminismWithArrivalSteps) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    SchedulerFixture fixture = MakeSchedulerFixture("sched_determinism", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    const std::vector<std::vector<int64_t>> prompts = {
        {3, 4, 5, 6}, {7, 8, 9, 10}, {11, 12, 13, 14}};
    const std::vector<LLMRunner::GenerateOptions> options_list = {
        TopPOptions(/*max_new_tokens=*/4, 0.9f, /*seed=*/11),
        TopPOptions(/*max_new_tokens=*/4, 0.9f, /*seed=*/22),
        TopPOptions(/*max_new_tokens=*/4, 0.9f, /*seed=*/33)};

    std::vector<LLMRunner::SchedulerRequest> early;
    std::vector<LLMRunner::SchedulerRequest> late;
    for (size_t i = 0; i < prompts.size(); ++i) {
        early.push_back(MakeRequest(prompts[i], options_list[i], /*arrival_step=*/0));
        late.push_back(MakeRequest(prompts[i], options_list[i], static_cast<int32_t>(i)));
    }

    const std::vector<LLMRunner::GenerateResult> first = fixture.runner->RunScheduler(early);
    const std::vector<LLMRunner::GenerateResult> second = fixture.runner->RunScheduler(late);
    ASSERT_EQ(first.size(), early.size());
    ASSERT_EQ(second.size(), late.size());
    for (size_t i = 0; i < first.size(); ++i) {
        std::cout << "[诊断] 第 " << i << " 条 早到=" << SequenceToString(first[i].tokens)
                  << " 晚到=" << SequenceToString(second[i].tokens) << "\n";
        ASSERT_EQ(first[i].tokens.size(), second[i].tokens.size()) << "第 " << i << " 条长度不同";
        for (size_t k = 0; k < first[i].tokens.size(); ++k) {
            EXPECT_EQ(first[i].tokens[k], second[i].tokens[k])
                << "第 " << i << " 条第 " << k << " 个 token 随 arrival_step 变了";
        }
    }
}

// AC3：连续跑完（含整批被拒的失败路径）后空闲块必须回到初始水位。
TEST(LlmRunnerSchedulerTest, BlocksReturnAtEnd) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    SchedulerFixture fixture = MakeSchedulerFixture("sched_blocks", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);
    const int32_t free_before = fixture.runner->NumFreeKvBlocks();
    ASSERT_GT(free_before, 0);

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, GreedyOptions(/*max_new_tokens=*/4), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10}, GreedyOptions(/*max_new_tokens=*/3), 1));
    requests.push_back(MakeRequest({11, 12, 13, 14}, GreedyOptions(/*max_new_tokens=*/3), 2));
    ASSERT_EQ(fixture.runner->RunScheduler(requests).size(), requests.size());
    EXPECT_EQ(fixture.runner->NumFreeKvBlocks(), free_before) << "正常路径必须全部归还";

    // 失败路径（重复 seq_id → 整批拒绝）也不能漏块
    std::vector<LLMRunner::SchedulerRequest> bad = requests;
    bad[0].request.seq_id = 5;
    bad[1].request.seq_id = 5;
    EXPECT_TRUE(fixture.runner->RunScheduler(bad).empty());
    EXPECT_EQ(fixture.runner->NumFreeKvBlocks(), free_before) << "失败路径也必须全部归还";
}

// **总闸**：AC1 在动态批下仍成立 —— 同一请求同 seed，无论 arrival_step 怎么排、
// 同批伙伴是谁、批内长度齐不齐，token 都与"逐条单独运行"逐位相同。
TEST(LlmRunnerSchedulerTest, BatchEqualsSequentialUnderScheduling) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    SchedulerFixture fixture = MakeSchedulerFixture("sched_ac1", /*max_batch=*/2,
                                                    /*eos_token_id=*/-1, /*num_blocks=*/16);
    ASSERT_TRUE(fixture.ok);

    // 4 条请求 > max_batch(2)：必须靠"退出 → 准入"才跑得完；长度不齐；采样走 Top-P（含随机流）
    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest({3, 4, 5, 6}, TopPOptions(5, 0.9f, 101), 0));
    requests.push_back(MakeRequest({7, 8, 9, 10, 11}, TopPOptions(5, 0.9f, 202), 0));
    requests.push_back(MakeRequest({12, 13, 14}, TopPOptions(5, 0.9f, 303), 3));
    requests.push_back(MakeRequest({15, 16, 17, 18}, TopPOptions(5, 0.9f, 404), 3));

    const std::vector<LLMRunner::GenerateResult> scheduled = fixture.runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), requests.size());
    const LLMRunner::SchedulerStats stats = fixture.runner->scheduler_stats();
    // 总闸顺带锁调度形状：每条请求恰好 prefill 一次；两批准入（arrival 0,0 / 3,3）；并发不超上限
    EXPECT_EQ(stats.context_rows, static_cast<int32_t>(requests.size()))
        << "每条请求只该被 context 段装一次";
    EXPECT_EQ(stats.prefill_calls, 2) << "两批准入 = 两段 context";
    EXPECT_LE(stats.max_active, 2) << "并发不得超过 max_batch";
    ExpectMatchesSequential(fixture.runner.get(), requests, scheduled);
}

}  // namespace
}  // namespace mini_trt_llm
