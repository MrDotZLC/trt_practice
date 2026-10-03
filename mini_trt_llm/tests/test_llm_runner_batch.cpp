#include "gpt2_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/llm_runner.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
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
using test_support::SmallGpt2BuilderConfig;
using test_support::SmallGpt2ConfigJson;
using test_support::SmallGpt2Weights;

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

// 小模型 + prefill/decode 引擎 + 一个 runner。
// **成员声明顺序即析构顺序的逆序**：runner 最先析构，directory / logger 活得最久，
// 否则引擎文件可能在 runner 之前被清掉。
struct RunnerFixture {
    Logger logger;
    test_support::ModelDirectory directory = test_support::ModelDirectory::Create("llm_batch");
    std::shared_ptr<Engine> prefill;
    std::shared_ptr<Engine> decode;
    std::unique_ptr<LLMRunner> runner;
    bool ok = false;
};

RunnerFixture MakeFixture(const std::string& engine_prefix, int32_t max_batch) {
    RunnerFixture fixture;
    if (!fixture.directory.valid()) {
        return fixture;
    }
    if (!fixture.directory.WriteConfig(SmallGpt2ConfigJson()) ||
        !fixture.directory.WriteWeights(SmallGpt2Weights())) {
        return fixture;
    }
    EngineBuilder::Config builder_config = SmallGpt2BuilderConfig();
    EngineBuilder builder(fixture.logger, builder_config);
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
    runner_config.num_blocks = 8;  // 每条 prompt(4) + 生成(6) 只占 1 块，8 块放得下两条
    runner_config.is_half = false;
    runner_config.vocab_size = kVocab;
    runner_config.max_batch = max_batch;
    fixture.runner = std::make_unique<LLMRunner>(runner_config, fixture.prefill,
                                                 fixture.decode, nullptr);
    fixture.ok = fixture.runner->ok();
    return fixture;
}

LLMRunner::GenerateOptions GreedyOptions(int32_t max_new_tokens) {
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.top_k = 1;  // greedy：与随机流无关，批量与单跑的对比才是确定性的
    return options;
}

constexpr int32_t kNewTokens = 6;

// AC1：批量运行 == 逐条单独运行（逐 token 逐位相同）。
TEST(LlmRunnerBatchTest, BatchEqualsSequential) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_eq", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::GenerateRequest> requests(2);
    requests[0].input_ids = {3, 4, 5, 6};
    requests[0].options = GreedyOptions(kNewTokens);
    requests[1].input_ids = {7, 8, 9, 10};
    requests[1].options = GreedyOptions(kNewTokens);

    const std::vector<LLMRunner::GenerateResult> batch = fixture.runner->GenerateBatch(requests);
    ASSERT_EQ(batch.size(), requests.size());
    ASSERT_TRUE(batch[0].ok);
    ASSERT_TRUE(batch[1].ok);

    for (size_t b = 0; b < requests.size(); ++b) {
        const std::vector<int64_t> alone =
            fixture.runner->Generate(requests[b].input_ids, requests[b].options);
        ASSERT_EQ(alone.size(), static_cast<size_t>(kNewTokens));
        std::cout << "[诊断] 第 " << b << " 条 批量=" << SequenceToString(batch[b].tokens)
                  << " 单跑=" << SequenceToString(alone) << "\n";
        ASSERT_EQ(batch[b].tokens.size(), alone.size());
        for (size_t i = 0; i < alone.size(); ++i) {
            EXPECT_EQ(batch[b].tokens[i], alone[i])
                << "第 " << b << " 条第 " << i << " 个 token 不一致";
        }
    }
}

// AC5：批量为 1 时，批量入口与单序列入口逐 token 相同。
TEST(LlmRunnerBatchTest, BatchSingleRowMatchesGenerate) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_one", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    const std::vector<int64_t> prompt = {3, 4, 5, 6};
    const LLMRunner::GenerateOptions options = GreedyOptions(kNewTokens);

    LLMRunner::GenerateRequest request;
    request.input_ids = prompt;
    request.options = options;
    const std::vector<LLMRunner::GenerateResult> batch =
        fixture.runner->GenerateBatch({request});
    ASSERT_EQ(batch.size(), 1u);
    ASSERT_TRUE(batch[0].ok);

    const std::vector<int64_t> alone = fixture.runner->Generate(prompt, options);
    ASSERT_EQ(alone.size(), batch[0].tokens.size());
    for (size_t i = 0; i < alone.size(); ++i) {
        EXPECT_EQ(batch[0].tokens[i], alone[i]) << "第 " << i << " 个 token 不一致";
    }
    EXPECT_EQ(batch[0].seq_id, 0);
}

// D2=A：S1 只支持等长批；不等长必须整批拒绝，而不是静默填充。
TEST(LlmRunnerBatchTest, RejectsUnequalPromptLengths) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_unequal", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::GenerateRequest> requests(2);
    requests[0].input_ids = {3, 4, 5, 6};
    requests[0].options = GreedyOptions(kNewTokens);
    requests[1].input_ids = {7, 8, 9, 10, 11};  // 长度 5 ≠ 4
    requests[1].options = GreedyOptions(kNewTokens);

    EXPECT_TRUE(fixture.runner->GenerateBatch(requests).empty());
}

// 入口自己拦批上限，不靠引擎报错兜底。
TEST(LlmRunnerBatchTest, RejectsBatchOverMaxBatch) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_over", /*max_batch=*/1);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::GenerateRequest> requests(2);
    for (LLMRunner::GenerateRequest& request : requests) {
        request.input_ids = {3, 4, 5, 6};
        request.options = GreedyOptions(kNewTokens);
    }
    EXPECT_TRUE(fixture.runner->GenerateBatch(requests).empty());
}

// S1 收窄：批内必须同一种采样策略（三种 kernel 各自是整批一个分支）。
TEST(LlmRunnerBatchTest, RejectsMixedSamplingStrategy) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_strategy", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::GenerateRequest> requests(2);
    requests[0].input_ids = {3, 4, 5, 6};
    requests[0].options = GreedyOptions(kNewTokens);
    requests[1].input_ids = {7, 8, 9, 10};
    requests[1].options = GreedyOptions(kNewTokens);
    requests[1].options.top_p = 0.9f;  // 第 0 条 greedy、第 1 条 top-p

    EXPECT_TRUE(fixture.runner->GenerateBatch(requests).empty());
}

// seed 是 per-row 的：随机流只由 (请求 seed, 步数) 决定，**与批内位置无关**。
// 所以"同一请求 + 同一 seed，批量与单跑逐 token 相同"对**随机采样**也成立——这正是 AC1
// 对所有策略的要求；"换个批邻居就换输出"的旧行为在这里被锁死。
TEST(LlmRunnerBatchTest, BatchEqualsSequentialWithTopP) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_topp", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::GenerateRequest> requests(2);
    requests[0].input_ids = {3, 4, 5, 6};
    requests[0].options = GreedyOptions(kNewTokens);
    requests[0].options.top_p = 0.9f;  // 转到 Top-P（随机路径）
    requests[0].options.seed = 11;
    requests[1].input_ids = {7, 8, 9, 10};
    requests[1].options = GreedyOptions(kNewTokens);
    requests[1].options.top_p = 0.9f;
    requests[1].options.seed = 22;     // 与第 0 条不同 seed —— 现在允许（per-row seed）

    const std::vector<LLMRunner::GenerateResult> batch = fixture.runner->GenerateBatch(requests);
    ASSERT_EQ(batch.size(), requests.size());
    ASSERT_TRUE(batch[0].ok);
    ASSERT_TRUE(batch[1].ok);

    for (size_t b = 0; b < requests.size(); ++b) {
        const std::vector<int64_t> alone =
            fixture.runner->Generate(requests[b].input_ids, requests[b].options);
        ASSERT_EQ(alone.size(), batch[b].tokens.size());
        for (size_t i = 0; i < alone.size(); ++i) {
            EXPECT_EQ(batch[b].tokens[i], alone[i])
                << "第 " << b << " 条第 " << i << " 个 token 不一致（seed 不得与批位置相关）";
        }
    }
}

// 采样器契约：Top-K 路径的 top_k 不得超过 kTopKFastMaxK，否则该行会被写哨兵 -1。
// 入口必须拦住，不能让它表现为"生成了 token = -1"。
TEST(LlmRunnerBatchTest, RejectsTopKOverFastMax) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_topk", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::GenerateRequest> requests(2);
    for (LLMRunner::GenerateRequest& request : requests) {
        request.input_ids = {3, 4, 5, 6};
        request.options = GreedyOptions(kNewTokens);
        request.options.top_k = kTopKFastMaxK + 1;
    }
    EXPECT_TRUE(fixture.runner->GenerateBatch(requests).empty());
}

// seq_id 由调用方指定时必须在批内唯一。
TEST(LlmRunnerBatchTest, RejectsDuplicateSeqId) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    RunnerFixture fixture = MakeFixture("batch_seqid", /*max_batch=*/2);
    ASSERT_TRUE(fixture.ok);

    std::vector<LLMRunner::GenerateRequest> requests(2);
    for (LLMRunner::GenerateRequest& request : requests) {
        request.input_ids = {3, 4, 5, 6};
        request.options = GreedyOptions(kNewTokens);
        request.seq_id = 7;  // 两条用同一个 id
    }
    EXPECT_TRUE(fixture.runner->GenerateBatch(requests).empty());
}

}  // namespace
}  // namespace mini_trt_llm