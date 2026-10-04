#include "gpt2_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/llm_runner.hpp"
#include "test_gpu_guard.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

// S5-3 用例（chunked prefill：长 prompt 跨多步分批）。
//
// **为什么单独一个文件**（而不是并进 `test_llm_runner_packed.cpp`）：
//   ① 这里的夹具是"一个引擎文件 + **每个切法一个 runner**"——切法 = `Config::max_prefill_seq_len`
//      （**构造期**字段，之后只读），AC9 的三种切法没法在同一个 runner 上跑；packed 文件的夹具只有单 runner。
//   ② `test_plan.md` 已把这组登记成 `LlmRunnerChunkedTest.*`，文件与用例名一一对应最省心。
//   ③ `tests/CMakeLists.txt` 用 `file(GLOB *.cpp)` 收源文件，新增文件不需要改构建脚本。
//
// **判据的共同形状**：runner 不外露 KV cache / position_ids / 内部缓冲，所以"分块算对了"只能从
// 两侧夹——结果侧与"同一条 prompt 不分块"逐位相同；形状侧用 `SchedulerStats::context_rows`
// （= Σ 本步拿到 chunk 的行数）钉住"确实分了这么多步"。少了形状侧那半，"分块根本没发生"也能让
// 结果侧全绿（AC9 的用例就失去判别力）。
//
// **判据里的一个假设**：不同切法每步的 token 数 T 不同（T=1 vs T=prompt_len），逐位相同等于假定
// TRT 对不同 T 选到同一套 per-token 算法。真机若出现个别 token 不同，按 AGENTS §7 先诊断
// （对比 logits、找第一个分叉位置），**不要**把判据降级成"前缀相同"或加容差。
constexpr int32_t kMaxBatch = 4;

// 单序列能容纳的 token 数（每序列块数 × 块大小）。用例拿它当"池容量"上界：
// prompt_len + max_new 超过它的请求永远装不下，入口就该拒。
constexpr int32_t kTokensPerSeq = kBlocksPerSeq * kBlockSize;  // 16

struct ChunkedFixture {
    Logger logger;
    test_support::ModelDirectory directory = test_support::ModelDirectory::Create("llm_chunked");
    // 引擎**文件**路径。每个 runner 各自反序列化一份 `Engine`：既不必重复 `BuildFromConfig`
    // （分钟级开销），也不必论证"多个 runner 顺序复用同一个 execution context 安全"。
    std::string engine_path;
    bool ok = false;
};

// `builder_max_prefill_seq_len` = **建图侧**的 per-row 上界（默认与本夹具模型的位置数相同）。
// 只有"交叉校验 ③"那条用例会把它改小 —— 那是 S5 的真实形态（分块真的会启用）。
ChunkedFixture MakeChunkedFixture(const std::string& name, int32_t max_batch,
                                  int32_t builder_max_prefill_seq_len = kPositions) {
    ChunkedFixture fixture;
    if (!fixture.directory.valid()) {
        return fixture;
    }
    if (!fixture.directory.WriteConfig(test_support::SmallGpt2ConfigJson()) ||
        !fixture.directory.WriteWeights(test_support::SmallGpt2Weights())) {
        return fixture;
    }
    EngineBuilder::Config builder_config = test_support::SmallGpt2BuilderConfig(max_batch);
    // 建图侧的 per-row 上界可覆盖：`max_prefill_seq_len < n_positions` 就是 S5 的**真实形态**
    // （允许一条 prompt 跨多步推进），也是交叉校验 ③ 唯一能独立触发的配置（见下面那条用例）。
    builder_config.max_prefill_seq_len = builder_max_prefill_seq_len;
    builder_config.packed_mixed_prefill = true;  // S5 是 packed 路径内部的能力，没有独立开关
    EngineBuilder builder(fixture.logger, builder_config);
    fixture.engine_path = fixture.directory.EnginePath(name + "_packed.engine");
    // packed 图只对 prefill 阶段有意义（模型构建器会拒 decode）
    if (!builder.BuildFromConfig(fixture.directory.path(), fixture.engine_path,
                                 BuildStage::kPrefill)) {
        return fixture;
    }
    fixture.ok = true;
    return fixture;
}

LLMRunner::Config ChunkedRunnerConfig(int32_t max_batch, int32_t max_prefill_seq_len) {
    LLMRunner::Config config;
    config.num_layers = kLayers;
    config.num_kv_heads = kHeads;
    config.head_size = kHeadSize;
    config.block_size = kBlockSize;
    config.max_blocks_per_seq = kBlocksPerSeq;
    config.num_blocks = 16;
    config.is_half = false;
    config.vocab_size = kVocab;
    config.max_batch = max_batch;
    // packed 模式下这里必给的是**单步 per-row 上界**（= 切法）。位置表长度**不用给**：
    // 它由建图期写进引擎侧车、构造期读回（B1+A1，见 `p5_s5_interface_spec.md` §2）。
    config.max_prefill_seq_len = max_prefill_seq_len;
    config.prefill_mode = LLMRunner::Config::PrefillMode::kPackedMixed;
    return config;
}

// 在**指定切法**下造一个 runner。切法 = `Config::max_prefill_seq_len`（构造期字段，之后只读；
// 2026-10-05 起取代了原先的 `SetChunkLimitOverride` 进程级钩子，见 `p5_s5_interface_spec.md` §2）。
std::unique_ptr<LLMRunner> MakeRunner(const ChunkedFixture& fixture, int32_t max_batch,
                                      int32_t max_prefill_seq_len) {
    auto engine = std::make_shared<Engine>(fixture.engine_path, fixture.logger);
    return std::make_unique<LLMRunner>(ChunkedRunnerConfig(max_batch, max_prefill_seq_len), engine,
                                       engine, nullptr);
}

LLMRunner::GenerateOptions GreedyOptions(int32_t max_new_tokens) {
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.top_k = 1;  // greedy：与随机流无关，逐位对拍才干净
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

// 造一条长度为 length 的 prompt：token 值在 [0, vocab) 内且**逐位置不同**——
// 位置写错 / chunk 内容取错都要能从 token 结果上看出来（全同值的 prompt 会把这类缺陷盖住）。
std::vector<int64_t> MakePrompt(int32_t length, int32_t base = 3) {
    std::vector<int64_t> prompt;
    prompt.reserve(static_cast<size_t>(length));
    for (int32_t i = 0; i < length; ++i) {
        prompt.push_back((base + i) % kVocab);
    }
    return prompt;
}

int32_t ChunkCount(int32_t prompt_len, int32_t chunk_limit) {
    return (prompt_len + chunk_limit - 1) / chunk_limit;
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

// packed 路径的"逐条单跑"参考 = **同一个 runner 的单请求 RunScheduler**
// （packed 模式下 `Generate` / `GenerateBatch` 会明确拒绝：它们是 padding 路径的入口）。
// 参考也要用同一个 runner，因为 `chunk_limit` 是这条 runner 的属性 —— "结果与切法无关"
// 正是 AC9 要验的东西，所以参考侧不能换切法。
std::vector<int64_t> SingleRun(LLMRunner* runner, const std::vector<int64_t>& prompt,
                               const LLMRunner::GenerateOptions& options) {
    const std::vector<LLMRunner::GenerateResult> single =
        runner->RunScheduler({MakeRequest(prompt, options, 0)});
    if (single.size() != 1) {
        return {};
    }
    return single.front().tokens;
}

void ExpectMatchesSingleRun(LLMRunner* runner,
                            const std::vector<LLMRunner::SchedulerRequest>& requests,
                            const std::vector<LLMRunner::GenerateResult>& scheduled) {
    ASSERT_EQ(scheduled.size(), requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const std::vector<int64_t> alone =
            SingleRun(runner, requests[i].request.input_ids, requests[i].request.options);
        std::cout << "[诊断] 第 " << i << " 条 批内=" << SequenceToString(scheduled[i].tokens)
                  << " 单跑=" << SequenceToString(alone) << "\n";
        ASSERT_EQ(scheduled[i].tokens.size(), alone.size()) << "第 " << i << " 条长度不同";
        for (size_t k = 0; k < alone.size(); ++k) {
            EXPECT_EQ(scheduled[i].tokens[k], alone[k]) << "第 " << i << " 条第 " << k << " 个";
        }
    }
}

// ---------------------------------------------------------------------------
// ① AC9 的主体：三种切法逐位相同
// ---------------------------------------------------------------------------

// 同一条 prompt 在 `chunk_limit` = 1 / 中间值 / ≥ prompt_len 三种切法下，token **逐位相同**；
// 三种切法同时覆盖两种形状：limit=1 → 12 步全对齐（每一步都刚好填满 chunk_limit）、
// limit=5 → 5+5+2（含末块）、limit=kPositions → 一步送完（= S4 的形态）。
//
// **这条同时锁住 `step_limit`**：老上界 = max_new×1 + 1 + 4 = 9 步，而 limit=1 要走 12 个分块步
// —— 不把分块步数算进防呆上界，这里会以 "scheduler step limit exceeded" 收场（S5 剩余 ②）。
TEST(LlmRunnerChunkedTest, ChunkedEqualsWholePrompt) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_ac9", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    const int32_t kPromptLen = 12;
    const int32_t kMaxNew = 4;  // 12 + 4 = 16 = 单序列容量，刚好装满还不越界
    const std::vector<int64_t> prompt = MakePrompt(kPromptLen);
    const std::vector<int32_t> limits = {1, 5, kPositions};

    std::vector<int64_t> baseline;
    for (size_t index = 0; index < limits.size(); ++index) {
        const int32_t limit = limits[index];
        std::unique_ptr<LLMRunner> runner = MakeRunner(fixture, /*max_batch=*/1, limit);
        ASSERT_TRUE(runner->ok()) << "chunk_limit = " << limit;
        const std::vector<LLMRunner::GenerateResult> results =
            runner->RunScheduler({MakeRequest(prompt, GreedyOptions(kMaxNew), 0)});
        ASSERT_EQ(results.size(), 1u) << "chunk_limit = " << limit;
        const std::vector<int64_t>& tokens = results[0].tokens;
        EXPECT_EQ(tokens.size(), static_cast<size_t>(kMaxNew)) << "chunk_limit = " << limit;
        // 形状侧：每步拿到 chunk 的行数之和 = ceil(prompt_len / chunk_limit) —— 证明真的分了块。
        EXPECT_EQ(runner->scheduler_stats().context_rows, ChunkCount(kPromptLen, limit))
            << "chunk_limit = " << limit;
        // 生成段行数 = max_new - 1（首 token 来自"prefill 完成"的那一步，不占生成段）。
        // 这条同时是 `TS-051` 第 2 条（空活跃表的旧行号被重复累加）的回归守卫。
        EXPECT_EQ(runner->scheduler_stats().generation_rows, kMaxNew - 1)
            << "chunk_limit = " << limit;
        std::cout << "[诊断] chunk_limit=" << limit << " tokens=" << SequenceToString(tokens)
                  << "\n";
        if (index == 0) {
            baseline = tokens;
        } else {
            ASSERT_EQ(tokens.size(), baseline.size()) << "chunk_limit = " << limit;
            for (size_t k = 0; k < baseline.size(); ++k) {
                EXPECT_EQ(tokens[k], baseline[k])
                    << "chunk_limit=" << limit << " 第 " << k << " 个 token 与不分块不同";
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ② 绝对位置
// ---------------------------------------------------------------------------

// 第 2 块起的 `position_ids` 必须是 `prompt_done + i`（段内 `i` 会查错位置表，而且**不报错**）。
//
// 为什么判据只能从结果侧拿：runner 不外露 position_ids。所以这条用 **limit=1（逐 token 一块）**
// 把"第 2..10 块都从 prompt_done > 0 起"这个形状钉死（`context_rows == prompt_len`），
// 再用与不分块（limit = kPositions ≥ prompt_len）逐位相同把位置正确性兜住 —— 位置错了 →
// K/V 与 wpe 查表全错 → token 逐位不同。
TEST(LlmRunnerChunkedTest, ChunkedPositionsAreAbsolute) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_positions", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    const int32_t kPromptLen = 10;
    const int32_t kMaxNew = 3;
    const std::vector<int64_t> prompt = MakePrompt(kPromptLen);

    std::unique_ptr<LLMRunner> chunked = MakeRunner(fixture, /*max_batch=*/1, /*max_prefill_seq_len=*/1);
    std::unique_ptr<LLMRunner> whole =
        MakeRunner(fixture, /*max_batch=*/1, /*max_prefill_seq_len=*/kPositions);
    ASSERT_TRUE(chunked->ok());
    ASSERT_TRUE(whole->ok());

    const std::vector<LLMRunner::GenerateResult> chunked_results =
        chunked->RunScheduler({MakeRequest(prompt, GreedyOptions(kMaxNew), 0)});
    const std::vector<LLMRunner::GenerateResult> whole_results =
        whole->RunScheduler({MakeRequest(prompt, GreedyOptions(kMaxNew), 0)});
    ASSERT_EQ(chunked_results.size(), 1u);
    ASSERT_EQ(whole_results.size(), 1u);

    EXPECT_EQ(chunked->scheduler_stats().context_rows, kPromptLen)
        << "limit=1 下每一步只送一个 token：第 2 块起必须真的从 prompt_done > 0 续写";
    EXPECT_EQ(whole->scheduler_stats().context_rows, 1) << "不分块：一步送完";
    std::cout << "[诊断] chunked=" << SequenceToString(chunked_results[0].tokens)
              << " whole=" << SequenceToString(whole_results[0].tokens) << "\n";
    ASSERT_EQ(chunked_results[0].tokens.size(), whole_results[0].tokens.size());
    for (size_t k = 0; k < whole_results[0].tokens.size(); ++k) {
        EXPECT_EQ(chunked_results[0].tokens[k], whole_results[0].tokens[k])
            << "第 " << k << " 个 token：分块与不分块必须逐位相同";
    }
}

// ---------------------------------------------------------------------------
// ③ 分块不影响同批其它序列
// ---------------------------------------------------------------------------

// 一条序列分块跨多步推进时，同批**正在 generation 的**行必须照常出 token。
// 场景：short（prompt 2，一步完成）先入批，long（prompt 8，4 个 chunk）晚一步入批 ——
// 于是 long 的每一步 chunk 都与 short 的 generation 步同批。判据：两条都与各自单跑逐位相同
// （分块要是把同批的行序 / 语境长度弄脏，short 的后续 token 立刻不一致）。
TEST(LlmRunnerChunkedTest, ChunkBoundaryDoesNotDisturbOthers) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_boundary", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    std::unique_ptr<LLMRunner> runner = MakeRunner(fixture, /*max_batch=*/2, /*max_prefill_seq_len=*/2);
    ASSERT_TRUE(runner->ok());

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest(MakePrompt(2), GreedyOptions(4), /*arrival_step=*/0));
    requests.push_back(MakeRequest(MakePrompt(8, /*base=*/11), GreedyOptions(4),
                                   /*arrival_step=*/1));  // 晚到 → 与在跑的行混批
    const std::vector<LLMRunner::GenerateResult> scheduled = runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), 2u);
    EXPECT_EQ(runner->scheduler_stats().context_rows, 1 + 4)  // short 1 块 + long 4 块
        << "分块步数不符合预期，场景没摆成（用例失去判别力）";
    ExpectMatchesSingleRun(runner.get(), requests, scheduled);
}

// ---------------------------------------------------------------------------
// ④ chunk_limit ≥ prompt_len 时保持 S4 的形态
// ---------------------------------------------------------------------------

// `chunk_limit ≥ prompt_len` ⇒ 每条序列一步 prefill 完（`context_rows` 恰好等于请求数，
// 与 S4 完全同形），同时结果必须与把同一条 prompt 切碎（limit=2）时逐位相同 —— 前半句锁
// "短 prompt 没有被平白拆步"，后半句锁 AC9 在短 prompt 上也成立。
TEST(LlmRunnerChunkedTest, ChunkedShortPromptsUnchanged) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_short", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    const int32_t kPromptLen = 4;
    const int32_t kMaxNew = 3;
    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest(MakePrompt(kPromptLen), GreedyOptions(kMaxNew), 0));
    requests.push_back(MakeRequest(MakePrompt(kPromptLen, /*base=*/15), GreedyOptions(kMaxNew), 0));

    std::unique_ptr<LLMRunner> whole = MakeRunner(fixture, /*max_batch=*/2, /*max_prefill_seq_len=*/8);
    std::unique_ptr<LLMRunner> chunked = MakeRunner(fixture, /*max_batch=*/2, /*max_prefill_seq_len=*/2);
    ASSERT_TRUE(whole->ok());
    ASSERT_TRUE(chunked->ok());

    const std::vector<LLMRunner::GenerateResult> whole_results = whole->RunScheduler(requests);
    const std::vector<LLMRunner::GenerateResult> chunked_results = chunked->RunScheduler(requests);
    ASSERT_EQ(whole_results.size(), 2u);
    ASSERT_EQ(chunked_results.size(), 2u);
    EXPECT_EQ(whole->scheduler_stats().context_rows, 2) << "不分块：两条各一步（S4 的形态）";
    EXPECT_EQ(chunked->scheduler_stats().context_rows, 2 * ChunkCount(kPromptLen, 2));
    // 两条各 kMaxNew 个 token：首 token 在完成 prefill 的那步，剩下 kMaxNew-1 个在生成段
    // （=`TS-051` 第 2 条的回归守卫：旧实现的空活跃表轮会把上一步的行数再加一遍）。
    EXPECT_EQ(whole->scheduler_stats().generation_rows, 2 * (kMaxNew - 1));
    EXPECT_EQ(chunked->scheduler_stats().generation_rows, 2 * (kMaxNew - 1));

    for (size_t i = 0; i < requests.size(); ++i) {
        std::cout << "[诊断] 第 " << i << " 条 不分块=" << SequenceToString(whole_results[i].tokens)
                  << " 分块=" << SequenceToString(chunked_results[i].tokens) << "\n";
        ASSERT_EQ(whole_results[i].tokens.size(), static_cast<size_t>(kMaxNew));
        EXPECT_EQ(chunked_results[i].tokens.size(), whole_results[i].tokens.size());
        for (size_t k = 0; k < whole_results[i].tokens.size(); ++k) {
            EXPECT_EQ(chunked_results[i].tokens[k], whole_results[i].tokens[k])
                << "第 " << i << " 条第 " << k << " 个 token";
        }
    }
    ExpectMatchesSingleRun(whole.get(), requests, whole_results);
}

// ---------------------------------------------------------------------------
// ⑤ 进度状态：chunk 期间不出 token，完成后才计 max_new
// ---------------------------------------------------------------------------

// prompt 8 切成 4 块：前 3 块只推进进度、**一个 token 都不出**；第 4 块完成 prefill 才采第 0 个
// token，`max_new = 3` 从那时起计。判据：
//   * token 数恰好 = max_new（早出或晚出都会变）；
//   * 与不分块（limit=8）逐位相同（早出的 token 会占掉位置、使整串错位）；
//   * `context_rows == 4`（真的分了 4 步，而不是"一步送完碰巧结果一样"）。
TEST(LlmRunnerChunkedTest, ChunkProgressStateIsCorrect) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_progress", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    const int32_t kPromptLen = 8;
    const int32_t kMaxNew = 3;
    const int32_t kChunkLimit = 2;
    const std::vector<int64_t> prompt = MakePrompt(kPromptLen);

    std::unique_ptr<LLMRunner> runner = MakeRunner(fixture, /*max_batch=*/1, kChunkLimit);
    std::unique_ptr<LLMRunner> whole = MakeRunner(fixture, /*max_batch=*/1, kPromptLen);
    ASSERT_TRUE(runner->ok());
    ASSERT_TRUE(whole->ok());

    const std::vector<LLMRunner::GenerateResult> results =
        runner->RunScheduler({MakeRequest(prompt, GreedyOptions(kMaxNew), 0)});
    const std::vector<LLMRunner::GenerateResult> whole_results =
        whole->RunScheduler({MakeRequest(prompt, GreedyOptions(kMaxNew), 0)});
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(whole_results.size(), 1u);

    EXPECT_EQ(runner->scheduler_stats().context_rows, ChunkCount(kPromptLen, kChunkLimit));
    EXPECT_EQ(results[0].tokens.size(), static_cast<size_t>(kMaxNew))
        << "chunk 期间不得出 token，max_new 必须从 prefill 完成起计";
    std::cout << "[诊断] chunked=" << SequenceToString(results[0].tokens)
              << " whole=" << SequenceToString(whole_results[0].tokens) << "\n";
    ASSERT_EQ(results[0].tokens.size(), whole_results[0].tokens.size());
    for (size_t k = 0; k < whole_results[0].tokens.size(); ++k) {
        EXPECT_EQ(results[0].tokens[k], whole_results[0].tokens[k])
            << "第 " << k << " 个 token 与不分块不同";
    }
}

// ---------------------------------------------------------------------------
// ⑥ 采样行集的紧凑暂存
// ---------------------------------------------------------------------------

// "分块中的长 prompt"排在"本步完成的短 prompt"**之前**时，本步只有后者出 token —— 采样行集
// 是 {1} 这种**带洞**的集合（S4 的"行集 = 前缀"在这里不成立）。判据：两条结果都与单跑逐位
// 相同。行集若按前缀处理，完成的那行会拿错行的末位 logits / 随机参数（少出或多出一个 token
// 也会在这里露出来）。
TEST(LlmRunnerChunkedTest, ChunkedSamplingRowSetIsCompacted) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_sample_rows", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    std::unique_ptr<LLMRunner> runner = MakeRunner(fixture, /*max_batch=*/2, /*max_prefill_seq_len=*/2);
    ASSERT_TRUE(runner->ok());

    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest(MakePrompt(8), GreedyOptions(3), 0));  // 第 0 行：4 块
    requests.push_back(
        MakeRequest(MakePrompt(2, /*base=*/20), GreedyOptions(3), 0));  // 第 1 行：1 块（本步完成）
    const std::vector<LLMRunner::GenerateResult> scheduled = runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), 2u);
    EXPECT_EQ(scheduled[0].tokens.size(), 3u);
    EXPECT_EQ(scheduled[1].tokens.size(), 3u)
        << "本步完成 prefill 的那行必须照样出 token（行集带洞不是漏采样的理由）";
    EXPECT_EQ(runner->scheduler_stats().context_rows, 4 + 1);
    ExpectMatchesSingleRun(runner.get(), requests, scheduled);
}

// ---------------------------------------------------------------------------
// ⑦ 分块跨步时的块记账与退出归还
// ---------------------------------------------------------------------------

// AC3 在分块下的形态：① 正常路径（两条都分块）跑完后空闲块回到调用前水位；② 失败路径
// （某条请求永远装不下池）整批拒绝后水位同样不变 —— 失败发生在分配之前，不该有泄漏。
TEST(LlmRunnerChunkedTest, ChunkedRetireAndBlocks) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_blocks", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    std::unique_ptr<LLMRunner> runner = MakeRunner(fixture, /*max_batch=*/2, /*max_prefill_seq_len=*/2);
    ASSERT_TRUE(runner->ok());
    const int32_t free_before = runner->NumFreeKvBlocks();
    ASSERT_GT(free_before, 0);

    // ① 正常路径：prompt 8 + max_new 3 = 11 token → 每条 3 块，两条 6 块（池 16 块，够）。
    std::vector<LLMRunner::SchedulerRequest> requests;
    requests.push_back(MakeRequest(MakePrompt(8), GreedyOptions(3), 0));
    requests.push_back(MakeRequest(MakePrompt(8, /*base=*/17), GreedyOptions(3), 0));
    const std::vector<LLMRunner::GenerateResult> scheduled = runner->RunScheduler(requests);
    ASSERT_EQ(scheduled.size(), 2u);
    ExpectMatchesSingleRun(runner.get(), requests, scheduled);
    EXPECT_EQ(runner->NumFreeKvBlocks(), free_before)
        << "分块跨步不该改变块归还的结论（正常路径必须全归还）";

    // ② 失败路径：prompt 12 + max_new 8 = 20 > 16 = 单序列容量 —— 永远服务不了，入口整批拒绝。
    const std::vector<LLMRunner::GenerateResult> rejected =
        runner->RunScheduler({MakeRequest(MakePrompt(12), GreedyOptions(8), 0)});
    EXPECT_TRUE(rejected.empty()) << "装不下的请求必须在入口被拒";
    EXPECT_EQ(runner->NumFreeKvBlocks(), free_before) << "被拒路径也要保持水位不变";
}

// ---------------------------------------------------------------------------
// ⑧ 配置 / 形状类不可用必须显式拒绝
// ---------------------------------------------------------------------------

// 两组判据（**分组用 (A)/(B)**；`①②③④⑤` 在本文件里只留给 spec §2 的交叉校验编号）：
//   (A) 非法 Config 在构造期（任何引擎 / 显存动作之前）就被拒 —— 这一段**不需要 GPU**，沙箱里也真的
//       跑过（放在 skip 之前，且用 ASSERT：失败会直接以"红"收场，不会被后面的 skip 吞掉）；
//   (B) 真机段 · 构造期拒绝：`max_prefill_seq_len`（切法）**未声明** / 越界 —— 声明值 = 上界
//       必须**接受**（自证"拒绝的是越界，不是字段本身"）。
// 判据是"**没有静默换路**"，错误信息带实际值与上界（日志不进判据）。
//
// **原先的 (C)（入口位置上界）已并入池容量检查**（2026-10-05 B1+A1 修订）：`n_positions` 现在来自
// 引擎侧车且 B1 要求整除 ⇒ "池容量 == `n_positions`" ⇒ 位置越界与池装不下是**同一个条件**，
// 单列会制造"两条独立判据"的错觉。入口拒绝 + 水位不变的断言仍在（见上面那条装不下的用例）。
//
// **本夹具的覆盖说明（诚实登记）**：**spec §2 的交叉校验 ③**（`L × rows_max ≤ T_max`）在本夹具里
// **无法独立触发** —— rows_max = `max_batch` = 4、T_max = 4 × 16 = 64 ⇒ 交叉校验 ③ 等价于 `L ≤ 16`，
// 而交叉校验 ② 的上界（`n_positions` = 16）先把它拦下（**② 先报**）。要独立触发它需要
// `n_positions ≠ max_prefill_seq_len` 的配置 —— 见下面第 ⑨ 条用例
// （`ChunkLimitCrossCheckRejectsOverStepBudget`）；这里只断言"越界被拒 + 边界值被接受"。
TEST(LlmRunnerChunkedTest, ChunkLimitRejectedConfigs) {
    // (A) 非法 Config（全 0）：构造期第一道校验就拒绝，不碰引擎、不碰显存。
    LLMRunner::Config bogus;
    LLMRunner invalid_runner(bogus, nullptr, nullptr, nullptr);
    ASSERT_FALSE(invalid_runner.ok()) << "非法 Config 必须在构造期被拒";

    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_reject", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    // (B-a) 切法**未声明**（`0` 只是"未声明"的哨兵）→ 构造期拒绝。
    std::unique_ptr<LLMRunner> undeclared =
        MakeRunner(fixture, kMaxBatch, /*max_prefill_seq_len=*/0);
    EXPECT_FALSE(undeclared->ok()) << "packed 模式下未声明 max_prefill_seq_len 必须构造期拒绝";

    // (B-b) 声明值 = 位置表真值（本夹具 = 侧车里的 16）：必须**接受**（自证拒绝的是越界）。
    std::unique_ptr<LLMRunner> on_bound =
        MakeRunner(fixture, kMaxBatch, /*max_prefill_seq_len=*/kPositions);
    EXPECT_TRUE(on_bound->ok()) << "max_prefill_seq_len = " << kPositions << " 被误拒";

    // (B-c) 声明值 = 上界 + 1：构造期必须拒绝（本夹具里 **spec §2 交叉校验 ②**（≤ n_positions）先报）。
    std::unique_ptr<LLMRunner> over_bound =
        MakeRunner(fixture, kMaxBatch, /*max_prefill_seq_len=*/kPositions + 1);
    EXPECT_FALSE(over_bound->ok())
        << "max_prefill_seq_len 越界（" << (kPositions + 1) << " > " << kPositions
        << "）必须构造期拒绝";
    // B1 的不变量（本夹具成立）：`n_positions == 单序列池容量`（`kBlocksPerSeq × kBlockSize`）——
    // 这正是"位置越界与池装不下是同一个条件"的前提，(B-c) 同时守住两者。
    static_assert(kPositions == kTokensPerSeq,
                  "夹具里 n_positions 必须等于单序列池容量（B1 的前提）");
}

// ⑨ **spec §2 交叉校验 ③ 的独立触发**（`p5_s5_interface_spec.md` §2 的第 ③ 条）：
// `L × block_tables.dim0.max ≤ input_ids.dim1.max` —— 保证"每步 Σ 每行 chunk 不越出引擎 profile
// 的 T 上界"，即"多行同批不越出形状"的充分条件。
//
// **为什么需要单独一条**：本文件其它用例里 `max_prefill_seq_len` 与 `n_positions` 相等（都是 16），
// 于是**交叉校验 ③** 等价于 `L ≤ 16`，会被**交叉校验 ②**（≤ 侧车真值 `n_positions`）先拦下 —— 拿不到
// "**③ 自己拦人**"的证据。这条把**建图侧的 per-row 上界**降到 8（模型 `n_positions` 仍是 16，
// 即 S5 的真实形态），于是 `rows_max = 4`、`T_max = 4 × 8 = 32` ⇒ 交叉校验 ③ 的合法上界 =
// `32 ÷ 4 = 8`，而交叉校验 ② 的合法上界仍是 `n_positions = 16` —— 两者不再重合。
//
// 判据（只到"构造期拒绝 + 边界值接受"；错误信息的文案不进判据）：
//   * `L = 8`（= `T_max ÷ rows_max`，恰在交叉校验 ③ 的边界上）→ **接受**（正向对照）；
//   * `L = 9`（> 交叉校验 ③ 的上界，但 ≤ `n_positions`）→ **拒绝**：交叉校验 ② 放它过去，
//     只有交叉校验 ③ 能拦。
TEST(LlmRunnerChunkedTest, ChunkLimitCrossCheckRejectsOverStepBudget) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kBuilderPerRowCap = 8;  // 建图侧的 max_prefill_seq_len
    ChunkedFixture fixture = MakeChunkedFixture("chunked_cross_check", kMaxBatch,
                                                /*builder_max_prefill_seq_len=*/kBuilderPerRowCap);
    ASSERT_TRUE(fixture.ok);

    // `L = 8`：`8 × 4 = 32 = T_max` —— 恰好压在交叉校验 ③ 的边界上，必须接受。
    std::unique_ptr<LLMRunner> on_cross_bound =
        MakeRunner(fixture, kMaxBatch, kBuilderPerRowCap);
    EXPECT_TRUE(on_cross_bound->ok())
        << "L = T_max ÷ rows_max（" << kBuilderPerRowCap << "）应当接受";

    // `L = 9`：交叉校验 ②（≤ n_positions 16）不拦，只有交叉校验 ③（9 × 4 = 36 > 32）能拦 →
    // 这就是"③ 自己拦人"的独立证据。
    std::unique_ptr<LLMRunner> over_step_budget =
        MakeRunner(fixture, kMaxBatch, kBuilderPerRowCap + 1);
    EXPECT_FALSE(over_step_budget->ok())
        << "L = " << (kBuilderPerRowCap + 1) << " 超过 T_max ÷ rows_max（" << kBuilderPerRowCap
        << "）：交叉校验 ② 不拦（≤ n_positions），必须由 spec §2 交叉校验 ③ 拒绝";
}

// ---------------------------------------------------------------------------
// ⑩ n_positions 的真值来源 = 引擎侧车（B1+A1）：缺 / 不可信必须在构造期拒绝
// ---------------------------------------------------------------------------

// `n_positions` 只从 `<engine>.fingerprint` 读（引擎侧没有别的**精确**载体，只剩
// `ceil(n_positions / block_size)` 这个上界）—— 所以侧车缺了或被改过，就必须**拒绝启动**，
// 不能猜默认值、也不能"取 min"。两条都落在构造期（`Engine` 反序列化照常成功，拒绝发生在
// `LLMRunner` 的构造期校验里）。
TEST(LlmRunnerChunkedTest, MissingFingerprintSidecarRejected) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_sidecar_missing", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    // 只拷 `.engine`（不带 `.fingerprint`）—— 正是"单独拷贝引擎"的那种用法，行为已变更。
    const std::string bare_engine = fixture.directory.EnginePath("bare.engine");
    std::error_code copy_error;
    std::filesystem::copy_file(fixture.engine_path, bare_engine, copy_error);
    ASSERT_FALSE(copy_error) << "拷贝引擎失败：" << copy_error.message();
    ASSERT_FALSE(std::filesystem::exists(bare_engine + ".fingerprint"))
        << "这个副本本来就不该带侧车（带上了用例就不成立）";

    auto engine = std::make_shared<Engine>(bare_engine, fixture.logger);
    LLMRunner runner(ChunkedRunnerConfig(kMaxBatch, kPositions), engine, engine, nullptr);
    EXPECT_FALSE(runner.ok()) << "缺 .fingerprint 侧车时必须构造期拒绝启动（不猜默认值）";
}

TEST(LlmRunnerChunkedTest, SidecarPositionsMismatchRejected) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    ChunkedFixture fixture = MakeChunkedFixture("chunked_sidecar_mismatch", kMaxBatch);
    ASSERT_TRUE(fixture.ok);

    // 把 `.engine` 与它的侧车**一起**拷走：先做**正对照**（只拷贝、不改内容必须照常接受），
    // 再改侧车里的那一行。正对照把"拷贝"与"侧车不可信"两个原因分开，免得失败时归因错。
    const std::string copy_engine = fixture.directory.EnginePath("mismatch.engine");
    std::error_code copy_error;
    std::filesystem::copy_file(fixture.engine_path, copy_engine, copy_error);
    ASSERT_FALSE(copy_error) << "拷贝引擎失败：" << copy_error.message();
    std::filesystem::copy_file(fixture.engine_path + ".fingerprint",
                               copy_engine + ".fingerprint", copy_error);
    ASSERT_FALSE(copy_error) << "拷贝侧车失败：" << copy_error.message();
    {
        auto engine = std::make_shared<Engine>(copy_engine, fixture.logger);
        LLMRunner runner(ChunkedRunnerConfig(kMaxBatch, kPositions), engine, engine, nullptr);
        EXPECT_TRUE(runner.ok()) << "正对照：引擎与侧车一起拷贝（内容未改）必须照常接受";
    }

    // 只改侧车里的真值：16 → 12。`12 % 4 == 0`（仍然整除，走不到 B1 那条建图期约束），但
    // `ceil(12 / 4) = 3` ≠ `block_tables` 的 dim1 = 4 → 构造期自检必须拒绝（不取 min、不静默）。
    const std::string fingerprint_path = copy_engine + ".fingerprint";
    const std::string from = "num.model.n_positions=16";
    const std::string to = "num.model.n_positions=12";
    std::string text;
    {
        std::ifstream in(fingerprint_path);
        ASSERT_TRUE(in.good()) << "读不到侧车：" << fingerprint_path;
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const size_t at = text.find(from);
    ASSERT_NE(at, std::string::npos) << "侧车里没有 '" << from << "'（写侧格式变了？请同步本用例）";
    text.replace(at, from.size(), to);
    {
        std::ofstream out(fingerprint_path, std::ios::trunc);
        ASSERT_TRUE(out.good()) << "写不回侧车：" << fingerprint_path;
        out << text;
    }

    auto engine = std::make_shared<Engine>(copy_engine, fixture.logger);
    LLMRunner runner(ChunkedRunnerConfig(kMaxBatch, kPositions), engine, engine, nullptr);
    EXPECT_FALSE(runner.ok())
        << "侧车 n_positions=12 与引擎 block_tables（4 块/序列）不自洽 —— 必须构造期拒绝启动";
}

// **建图期**的硬失败（B1）：`n_positions % block_size != 0` 时拒绝建 packed 图。
// 只改 `block_size`（4 → 3），`n_positions` 仍是 16 ⇒ `ceil(16 / 3) = 6`、池容量 18 ≠ 16：
// 窗口 `16 < L <= 18` 会被静默放行，正是 B1 要挡的形态。
TEST(LlmRunnerChunkedTest, NonDivisibleBlockSizeRejected) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("chunked_non_divisible");
    ASSERT_TRUE(directory.valid());
    std::string config_json = test_support::SmallGpt2ConfigJson();
    const std::string from = "\"block_size\": 4";
    const std::string to = "\"block_size\": 3";
    const size_t at = config_json.find(from);
    ASSERT_NE(at, std::string::npos) << "夹具 JSON 里没有 '" << from << "'（写法变了？请同步本用例）";
    config_json.replace(at, from.size(), to);
    ASSERT_TRUE(directory.WriteConfig(config_json));
    ASSERT_TRUE(directory.WriteWeights(test_support::SmallGpt2Weights()));

    Logger logger;
    EngineBuilder::Config packed_config = test_support::SmallGpt2BuilderConfig(kMaxBatch);
    packed_config.max_prefill_seq_len = 8;
    packed_config.packed_mixed_prefill = true;
    EngineBuilder packed_builder(logger, packed_config);
    EXPECT_FALSE(packed_builder.BuildFromConfig(
        directory.path(), directory.EnginePath("non_divisible_packed.engine"),
        BuildStage::kPrefill))
        << "n_positions = 16、block_size = 3（不整除）必须让 packed 建图硬失败";

    // 反向对照：整除约束**只加在 packed 建图**上 —— padding 路径每步把 prompt 一次装下，
    // 越界由 `SetInputShape` 响亮拦住，所以同一份配置建 padding 引擎必须成功。
    EngineBuilder::Config padded_config = test_support::SmallGpt2BuilderConfig(kMaxBatch);
    padded_config.max_prefill_seq_len = 8;
    EngineBuilder padded_builder(logger, padded_config);
    EXPECT_TRUE(padded_builder.BuildFromConfig(
        directory.path(), directory.EnginePath("non_divisible_padded.engine"),
        BuildStage::kPrefill))
        << "整除约束只限 packed 建图；padding 路径必须照常建图";
}

}  // namespace
}  // namespace mini_trt_llm
