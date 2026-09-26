#include "gpt2_test_support.hpp"
#include "tokenizer_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/llm_runner.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/tokenizer/bpe_tokenizer.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::kBlockSize;
using test_support::kBlocksPerSeq;
using test_support::kHeadSize;
using test_support::kHeads;
using test_support::kHidden;
using test_support::kLayers;
using test_support::kPositions;
using test_support::kVocab;
using test_support::SmallGpt2BuilderConfig;
using test_support::SmallGpt2ConfigJson;
using test_support::SmallGpt2Weights;

// 真机 GPT-2 的规模（与 models/gpt2/config.json 一致）
constexpr int32_t kRealLayers = 12;
constexpr int32_t kRealHeads = 12;
constexpr int32_t kRealHeadSize = 64;
constexpr int32_t kRealVocab = 50257;
constexpr int32_t kRealBlockSize = 16;
constexpr int32_t kRealPositions = 1024;
constexpr int32_t kRealEos = 50256;

// 已核对过的外部基线（§0.3）：HF 与"全序列重算"两条路径给出同一串 token。
// kPromptText 与 kExpectedPrompt 必须成对看：前者是文本，后者是它的分词结果
// （A1-10 用 BpeTokenizer 重新算一遍，把"文本入口"这一段桥接起来）。
constexpr const char* kPromptText = "The quick brown fox";
const std::vector<int64_t> kExpectedPrompt = {464, 2068, 7586, 21831};  // kPromptText 的分词
const std::vector<int64_t> kExpectedTokens = {274, 389, 257, 1049, 835, 284, 651, 257};

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

// 不用 KV Cache 的参考生成：每一步都把"prompt + 已生成"整段重新喂给 prefill 引擎。
//
// **为什么它能当参考**：它与 runner 的差别只有"有没有缓存"这一件事——数学式完全相同，
// 实现路径完全独立（每次都是全新的全序列前向，不涉及分页布局、context_lens、K/V 追加）。
// 因此两者生成的 token 序列一致，是对循环本身最直接的验证，且**不需要外部基线**。
// kv_scratch 非空时，额外绑定每层的 K/V 输出——kPrefill 引擎必须绑全所有输出才能 enqueue，
///因此复用同一个引擎跑这条参考路径时要把它们指到临时缓冲上。
std::vector<int64_t> GenerateWithoutCache(Engine* prefill, const std::vector<int64_t>& prompt,
                                          int32_t max_new_tokens, int32_t vocab_size,
                                          bool is_half,
                                          std::vector<std::unique_ptr<DeviceBuffer>>*
                                              kv_scratch = nullptr,
                                          int32_t num_layers = 0) {
    // vocab 与 dtype 必须由调用方给出：这里曾把 vocab 写死成小模型的 32，
    // 用到真实模型（50257）时就**越界写了显存**（缓冲按 32 分配、引擎按 50257 输出），
    // 比数值错更危险。测试辅助函数同样不能有隐藏假设。
    std::vector<int64_t> produced;
    std::vector<int64_t> sequence = prompt;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const int32_t length = static_cast<int32_t>(sequence.size());
        std::vector<int32_t> tokens(sequence.begin(), sequence.end());
        std::vector<int32_t> positions(static_cast<size_t>(length));
        for (int32_t i = 0; i < length; ++i) {
            positions[static_cast<size_t>(i)] = i;
        }

        DeviceBuffer d_tokens(tokens.size() * sizeof(int32_t));
        DeviceBuffer d_positions(positions.size() * sizeof(int32_t));
        const size_t elem = is_half ? 2u : 4u;
        DeviceBuffer d_logits(static_cast<size_t>(length) * vocab_size * elem);
        DeviceBuffer d_next(sizeof(int32_t));
        if (!d_tokens.Allocate(tokens.size() * sizeof(int32_t)) ||
            !d_positions.Allocate(positions.size() * sizeof(int32_t)) ||
            !d_logits.Allocate(static_cast<size_t>(length) * vocab_size * elem) ||
            !d_next.Allocate(sizeof(int32_t))) {
            return {};
        }
        CUDA_CHECK(cudaMemcpy(d_tokens.data(), tokens.data(), d_tokens.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_positions.data(), positions.data(), d_positions.size(),
                              cudaMemcpyHostToDevice));

        if (!prefill->SetOptimizationProfile(0, nullptr) ||
            !prefill->SetInputShape("input_ids", nvinfer1::Dims{2, {1, length}}) ||
            !prefill->SetInputShape("position_ids", nvinfer1::Dims{2, {1, length}}) ||
            !prefill->SetTensorAddress("input_ids", d_tokens.data()) ||
            !prefill->SetTensorAddress("position_ids", d_positions.data()) ||
            !prefill->SetTensorAddress("logits", d_logits.data())) {
            return {};
        }
        if (kv_scratch != nullptr) {
            const size_t kv_elems = static_cast<size_t>(kHeads) * length * kHeadSize;
            kv_scratch->clear();
            for (int32_t layer = 0; layer < num_layers; ++layer) {
                for (const char* tag : {"k_layer", "v_layer"}) {
                    auto buffer = std::make_unique<DeviceBuffer>();
                    if (!buffer->Allocate(kv_elems * sizeof(float)) ||
                        !prefill->SetTensorAddress(
                            (std::string(tag) + std::to_string(layer)).c_str(),
                            buffer->data())) {
                        return {};
                    }
                    kv_scratch->push_back(std::move(buffer));
                }
            }
        }
        if (!prefill->Enqueue(nullptr)) {
            return {};
        }
        prefill->Synchronize(nullptr);

        SamplerArgs args;
        args.logits = static_cast<const char*>(d_logits.data()) +
                      static_cast<size_t>(length - 1) * vocab_size * elem;
        args.token_ids = static_cast<int32_t*>(d_next.data());
        args.batch_size = 1;
        args.vocab_size = vocab_size;
        args.is_half = is_half;
        args.seed = 42;
        args.offset = static_cast<uint64_t>(step);
        CUDA_CHECK(LaunchGreedySampler(args, nullptr));

        int32_t next = -1;
        CUDA_CHECK(cudaMemcpy(&next, d_next.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
        produced.push_back(next);
        sequence.push_back(next);
    }
    return produced;
}

}  // namespace

// 自洽判据：runner（带 KV Cache 的 Prefill→Decode 循环）与"每步全序列重算"必须同结果。
//
// 这条能抓住循环本身的所有错误：cache 写错位置、context_lens 差一、position_ids 递推错、
// 当前 token 被重复或漏掉、K/V 追加错层……而且不依赖任何外部参考数据。
TEST(Gpt2GenerateTest, RunnerMatchesFullRecomputeWithoutCache) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    Logger logger;
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_generate_small");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(SmallGpt2ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(SmallGpt2Weights()));

    // 参考路径只需要 logits，所以用 kSingle（不导出 K/V，省一层绑定）
    EngineBuilder::Config builder_config = SmallGpt2BuilderConfig();
    EngineBuilder builder(logger, builder_config);
    const std::string single_path = directory.EnginePath("single.engine");
    const std::string prefill_path = directory.EnginePath("prefill.engine");
    const std::string decode_path = directory.EnginePath("decode.engine");
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), single_path, BuildStage::kSingle));
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), prefill_path,
                                        BuildStage::kPrefill));
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), decode_path,
                                        BuildStage::kDecode));
    Engine single(single_path, logger);
    auto prefill = std::make_shared<Engine>(prefill_path, logger);
    auto decode = std::make_shared<Engine>(decode_path, logger);

    LLMRunner::Config runner_config;
    runner_config.num_layers = kLayers;
    runner_config.num_kv_heads = kHeads;
    runner_config.head_size = kHeadSize;
    runner_config.block_size = kBlockSize;
    runner_config.max_blocks_per_seq = kBlocksPerSeq;
    runner_config.num_blocks = 8;  // 足够放下 prompt(4) + 生成(6)
    runner_config.is_half = false;
    runner_config.vocab_size = kVocab;
    // token-id 级别的 runner 不需要 tokenizer（GPT-2 是 BPE，与 SentencePiece 不对齐）
    LLMRunner runner(runner_config, prefill, decode, nullptr);
    ASSERT_TRUE(runner.ok());

    const std::vector<int64_t> prompt = {3, 4, 5, 6};
    constexpr int32_t kNewTokens = 6;
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = kNewTokens;
    options.top_k = 1;  // greedy

    const std::vector<int64_t> with_cache = runner.Generate(prompt, options);
    ASSERT_EQ(with_cache.size(), static_cast<size_t>(kNewTokens));
    const std::vector<int64_t> without_cache =
        GenerateWithoutCache(&single, prompt, kNewTokens, kVocab, /*is_half=*/false);
    ASSERT_EQ(without_cache.size(), static_cast<size_t>(kNewTokens));

    // 打印两条序列 + 首个分叉点：小模型只花十秒，是迭代这个缺陷的主战场
    int32_t first_diff = -1;
    for (int32_t i = 0; i < kNewTokens; ++i) {
        if (with_cache[static_cast<size_t>(i)] != without_cache[static_cast<size_t>(i)]) {
            first_diff = first_diff < 0 ? i : first_diff;
        }
    }
    std::cout << "[诊断] 小模型贪心 " << kNewTokens << " token\n"
              << "        带 cache  : " << SequenceToString(with_cache) << "\n"
              << "        无 cache  : " << SequenceToString(without_cache) << "\n"
              << "        首个分叉  : " << (first_diff < 0 ? -1 : first_diff) << "\n";

    for (int32_t i = 0; i < kNewTokens; ++i) {
        EXPECT_EQ(with_cache[static_cast<size_t>(i)], without_cache[static_cast<size_t>(i)])
            << "第 " << i << " 个生成 token 不一致：带 cache="
            << with_cache[static_cast<size_t>(i)]
            << " 不带 cache=" << without_cache[static_cast<size_t>(i)];
    }
}

// 接口契约：temperature != 1.0 必须显式失败（D5），不能静默忽略。
TEST(Gpt2GenerateTest, RejectsUnsupportedTemperature) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    Logger logger;
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_generate_temp");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(SmallGpt2ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(SmallGpt2Weights()));

    EngineBuilder builder(logger, SmallGpt2BuilderConfig());
    const std::string prefill_path = directory.EnginePath("prefill.engine");
    const std::string decode_path = directory.EnginePath("decode.engine");
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), prefill_path,
                                        BuildStage::kPrefill));
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), decode_path,
                                        BuildStage::kDecode));

    LLMRunner::Config runner_config;
    runner_config.num_layers = kLayers;
    runner_config.num_kv_heads = kHeads;
    runner_config.head_size = kHeadSize;
    runner_config.block_size = kBlockSize;
    runner_config.max_blocks_per_seq = kBlocksPerSeq;
    runner_config.num_blocks = 8;
    runner_config.is_half = false;
    runner_config.vocab_size = kVocab;
    LLMRunner runner(runner_config, std::make_shared<Engine>(prefill_path, logger),
                     std::make_shared<Engine>(decode_path, logger), nullptr);
    ASSERT_TRUE(runner.ok());

    LLMRunner::GenerateOptions options;
    options.temperature = 0.7f;
    // 返回空 vector 即失败（成功时至少返回 1 个 token）
    EXPECT_TRUE(runner.Generate({3, 4, 5, 6}, options).empty());
}

// 外部判据：真实 GPT-2 上的贪心生成必须与已核对的基线逐 token 一致（P2-8）。
//
// 这条最贵（要建两个 12 层引擎），因此放在最后、且模型目录不存在时跳过。
TEST(Gpt2GenerateTest, RealGpt2GreedyMatchesReferenceTokens) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindRealModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/gpt2 不存在（先跑 hf_to_mini_trt_llm.py 转换）";
    }

    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;
    builder_config.min_prefill_batch = 1;
    builder_config.opt_prefill_batch = 1;
    builder_config.max_prefill_batch = 1;
    builder_config.min_prefill_seq_len = 1;
    builder_config.opt_prefill_seq_len = static_cast<int32_t>(kExpectedPrompt.size());
    // 上限必须覆盖"prompt + 生成长度"：下面的无 cache 参考路径每步都要把整段重新喂进去。
    builder_config.max_prefill_seq_len =
        static_cast<int32_t>(kExpectedPrompt.size() + kExpectedTokens.size());
    builder_config.min_decode_batch = 1;
    builder_config.opt_decode_batch = 1;
    builder_config.max_decode_batch = 1;

    EngineBuilder builder(logger, builder_config);
    const std::string prefill_path = "/tmp/mini_trt_llm_gpt2_real_prefill.engine";
    const std::string decode_path = "/tmp/mini_trt_llm_gpt2_real_decode.engine";
    ASSERT_TRUE(builder.BuildFromConfig(dir, prefill_path, BuildStage::kPrefill))
        << "真实 GPT-2 prefill 引擎构建失败";
    ASSERT_TRUE(builder.BuildFromConfig(dir, decode_path, BuildStage::kDecode))
        << "真实 GPT-2 decode 引擎构建失败";

    LLMRunner::Config runner_config;
    runner_config.num_layers = kRealLayers;
    runner_config.num_kv_heads = kRealHeads;
    runner_config.head_size = kRealHeadSize;
    runner_config.block_size = kRealBlockSize;
    runner_config.max_blocks_per_seq = kRealPositions / kRealBlockSize;  // = 64
    runner_config.num_blocks = 64;
    runner_config.is_half = false;
    runner_config.vocab_size = kRealVocab;
    runner_config.eos_token_id = kRealEos;

    LLMRunner runner(runner_config, std::make_shared<Engine>(prefill_path, logger),
                     std::make_shared<Engine>(decode_path, logger), nullptr);
    ASSERT_TRUE(runner.ok());

    LLMRunner::GenerateOptions options;
    options.max_new_tokens = static_cast<int>(kExpectedTokens.size());
    options.top_k = 1;
    const std::vector<int64_t> generated = runner.Generate(kExpectedPrompt, options);
    ASSERT_EQ(generated.size(), kExpectedTokens.size());

    // 同一条基线上再跑一遍**不用 cache** 的参考生成（复用 prefill 引擎，只多绑 K/V 输出）。
    // 目的是把"prefill 图谱本身与 HF 的偏差"和"decode 路径的偏差"分开：
    //   * 无 cache 路径也不一致 → 偏差来自 prefill 图谱（与 KV Cache / 插件无关）；
    //   * 无 cache 路径一致、只有 runner 不一致 → 偏差在 decode 路径（插件 / 追加 / 位置递推）。
    std::vector<std::unique_ptr<DeviceBuffer>> kv_scratch;
    auto prefill_engine = std::make_shared<Engine>(prefill_path, logger);
    const std::vector<int64_t> without_cache = GenerateWithoutCache(
        prefill_engine.get(), kExpectedPrompt, static_cast<int32_t>(kExpectedTokens.size()),
        kRealVocab, /*is_half=*/false, &kv_scratch, kRealLayers);
    ASSERT_EQ(without_cache.size(), kExpectedTokens.size());

    std::cout << "[诊断] 真实 GPT-2 贪心前 8 token\n"
              << "        HF 基线   : " << SequenceToString(kExpectedTokens) << "\n"
              << "        runner    : " << SequenceToString(generated) << "\n"
              << "        无 cache  : " << SequenceToString(without_cache) << "\n";

    for (size_t i = 0; i < kExpectedTokens.size(); ++i) {
        EXPECT_EQ(without_cache[i], kExpectedTokens[i])
            << "无 cache 的 prefill 路径与 HF 基线不一致（第 " << i << " 个），"
               "说明偏差在 prefill 图谱而不在 decode 路径";
        EXPECT_EQ(generated[i], kExpectedTokens[i]) << "第 " << i << " 个 token 不符";
    }
}


// ---------------------------------------------------------------------------
// G2-1：FP16 端到端（Phase 2 测试计划 §5 的缺口）
// ---------------------------------------------------------------------------
//
// **为什么要它**：`EngineBuilder::Config` 的**默认精度就是 FP16**，而 Phase 2 的端到端
// 生成只在 FP32 验过——「默认路径必须被端到端覆盖」这条空档是真实的：FP16 下有额外的
// 舍入（LayerNorm 的 var+eps、10 位尾数的累加），且有额外的 Cast 节点，两者都可能改变
// 贪心 token。
//
// 判据用的是**语义判据**（8 个 token 必须全中）——与 FP32 那条同源、同一份 HF 基线；
// 数值阈值按 D6 的 FP16 档，此处不额外断言 logits，避免在没有实测依据的前提下拍阈值
// （AGENTS.md §7）。生成序列打印出来，供后续按"实测收敛"处理。
//
// **一个必须同时改的地方**：`LLMRunner::Config::is_half = true`。
// cache 精度必须与引擎激活精度一致，否则 PagedAttention 会按错误宽度读 cache——
// 那是"能跑但数值全错"的类型，不会报错。
TEST(Gpt2GenerateTest, RealGpt2Fp16GreedyMatchesReferenceTokens) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindRealModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/gpt2 不存在（先跑 hf_to_mini_trt_llm.py 转换）";
    }

    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP16;  // 默认值，显式写出以表明这是被测配置
    builder_config.min_prefill_batch = 1;
    builder_config.opt_prefill_batch = 1;
    builder_config.max_prefill_batch = 1;
    builder_config.min_prefill_seq_len = 1;
    builder_config.opt_prefill_seq_len = static_cast<int32_t>(kExpectedPrompt.size());
    builder_config.max_prefill_seq_len =
        static_cast<int32_t>(kExpectedPrompt.size() + kExpectedTokens.size());
    builder_config.min_decode_batch = 1;
    builder_config.opt_decode_batch = 1;
    builder_config.max_decode_batch = 1;
    EngineBuilder builder(logger, builder_config);

    const std::string prefill_path = "/tmp/mini_trt_llm_gpt2_real_prefill_fp16.engine";
    const std::string decode_path = "/tmp/mini_trt_llm_gpt2_real_decode_fp16.engine";
    if (!std::filesystem::exists(prefill_path)) {
        ASSERT_TRUE(builder.BuildFromConfig(dir, prefill_path, BuildStage::kPrefill))
            << "FP16 prefill 引擎构建失败";
    }
    if (!std::filesystem::exists(decode_path)) {
        ASSERT_TRUE(builder.BuildFromConfig(dir, decode_path, BuildStage::kDecode))
            << "FP16 decode 引擎构建失败";
    }

    LLMRunner::Config runner_config;
    runner_config.num_layers = kRealLayers;
    runner_config.num_kv_heads = kRealHeads;
    runner_config.head_size = kRealHeadSize;
    runner_config.block_size = kRealBlockSize;
    runner_config.max_blocks_per_seq = kRealPositions / kRealBlockSize;
    runner_config.num_blocks = 64;
    runner_config.is_half = true;  // 必须与引擎激活精度一致
    runner_config.vocab_size = kRealVocab;
    runner_config.eos_token_id = kRealEos;

    LLMRunner runner(runner_config, std::make_shared<Engine>(prefill_path, logger),
                     std::make_shared<Engine>(decode_path, logger), nullptr);
    ASSERT_TRUE(runner.ok());

    LLMRunner::GenerateOptions options;
    options.max_new_tokens = static_cast<int>(kExpectedTokens.size());
    options.top_k = 1;
    const std::vector<int64_t> generated = runner.Generate(kExpectedPrompt, options);
    ASSERT_EQ(generated.size(), kExpectedTokens.size());

    std::cout << "[诊断] G2-1 FP16 贪心 8 token\n"
              << "        HF 基线（FP32）: " << SequenceToString(kExpectedTokens) << "\n"
              << "        runner（FP16） : " << SequenceToString(generated) << "\n";
    for (size_t i = 0; i < kExpectedTokens.size(); ++i) {
        EXPECT_EQ(generated[i], kExpectedTokens[i]) << "第 " << i << " 个 token 不符";
    }
}


// ---------------------------------------------------------------------------
// 诊断用例：FP16 prefill 引擎的**每个输出**是 NaN 还是正常值
// ---------------------------------------------------------------------------
//
// 为什么放在测试里而不是产品代码里：这是"定位某一层内部"的一次性手段，
// 穿进 `LLMRunner` 只会让生产路径变复杂；而测试里可以照 Phase 3 的做法
// **从引擎查询 I/O**（名字 + 声明精度），通用地绑定全部输出。
//
// 它取代了"猜哪个算子产生 NaN"：逐输出给出 max|v| 与 NaN 标记，
// 先把范围钉到**具体的张量**，再决定改什么（见 TROUBLESHOOTING #18）。
TEST(Gpt2GenerateTest, Fp16PrefillOutputsDiagnostic) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindRealModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/gpt2 不存在";
    }

    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP16;
    // 这是**唯一**该打开诊断输出的用例：它逐输出读回中途张量，正是靠这些输出才不用猜
    // 哪个算子产生 NaN。默认关闭是刻意的——诊断输出会改 I/O 契约（见 TROUBLESHOOTING #19）。
    builder_config.export_diagnostics = true;
    builder_config.min_prefill_batch = 1;
    builder_config.opt_prefill_batch = 1;
    builder_config.max_prefill_batch = 1;
    builder_config.min_prefill_seq_len = 1;
    builder_config.opt_prefill_seq_len = static_cast<int32_t>(kExpectedPrompt.size());
    builder_config.max_prefill_seq_len = static_cast<int32_t>(kExpectedPrompt.size());
    EngineBuilder builder(logger, builder_config);

    // 引擎路径必须与 `RealGpt2Fp16Greedy...` 用的那个分开：引擎缓存只按路径名区分、
    // 不随代码或开关失效，共用一条路径会让"要诊断输出"与"不要诊断输出"互相踩成假结果。
    const std::string prefill_path = "/tmp/mini_trt_llm_gpt2_real_prefill_fp16_diag.engine";
    if (!std::filesystem::exists(prefill_path)) {
        ASSERT_TRUE(builder.BuildFromConfig(dir, prefill_path, BuildStage::kPrefill));
    }
    Engine engine(prefill_path, logger);
    nvinfer1::ICudaEngine* cuda = engine.GetCudaEngine();
    ASSERT_NE(cuda, nullptr);

    const int32_t seq = static_cast<int32_t>(kExpectedPrompt.size());
    std::vector<int32_t> tokens(kExpectedPrompt.begin(), kExpectedPrompt.end());

    struct Binding {
        std::string name;
        std::unique_ptr<DeviceBuffer> buffer;
        bool is_input = false;
        bool is_half = false;
        size_t count = 0;
    };
    std::vector<Binding> bindings;
    for (int32_t i = 0; i < cuda->getNbIOTensors(); ++i) {
        const std::string name = cuda->getIOTensorName(i);
        const bool is_input = cuda->getTensorIOMode(name.c_str()) ==
                              nvinfer1::TensorIOMode::kINPUT;
        const nvinfer1::DataType dtype = cuda->getTensorDataType(name.c_str());
        const bool is_half = dtype == nvinfer1::DataType::kHALF;
        const bool is_index = dtype == nvinfer1::DataType::kINT32;
        // 按名字推断规模（避免猜维数）：
        //   索引输入        → [1, seq]
        //   logits 输出     → [1, seq, vocab]
        //   K/V 输出        → [1, heads, seq, head_size]
        //   残差类诊断输出  → [1, seq, hidden]
        // **logits 也必须绑定**：漏绑会在 enqueue 时报
        // "Neither address or allocator is set for output tensor logits"。
        size_t count = 0;
        if (is_index) {
            count = static_cast<size_t>(seq);
        } else if (name == "logits") {
            count = static_cast<size_t>(seq) * kRealVocab;
        } else if (name.find("k_layer") != std::string::npos ||
                   name.find("v_layer") != std::string::npos) {
            count = static_cast<size_t>(kRealHeads) * seq * kRealHeadSize;
        } else if (name.find("mlp_fc") != std::string::npos ||
                   name.find("mlp_gelu") != std::string::npos) {
            count = static_cast<size_t>(seq) * 4 * 768;  // c_fc 把 hidden 放大 4 倍
        } else {
            count = static_cast<size_t>(seq) * 768;  // hidden = n_embd
        }
        Binding binding;
        binding.name = name;
        binding.is_input = is_input;
        binding.is_half = is_half;
        binding.count = count;
        binding.buffer = std::make_unique<DeviceBuffer>();
        if (!binding.buffer->Allocate(count * (is_half ? 2u : 4u))) {
            break;
        }
        if (is_input) {
            std::vector<int32_t> data(static_cast<size_t>(seq));
            for (int32_t j = 0; j < seq; ++j) {
                data[static_cast<size_t>(j)] = j;  // 供 position_ids 用；input_ids 下面覆盖
            }
            if (name == "input_ids") {
                data.assign(tokens.begin(), tokens.end());
            }
            CUDA_CHECK(cudaMemcpy(binding.buffer->data(), data.data(),
                                  data.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
            ASSERT_TRUE(engine.SetInputShape(name, nvinfer1::Dims{2, {1, seq}}));
        }
        ASSERT_TRUE(engine.SetTensorAddress(name, binding.buffer->data()));
        bindings.push_back(std::move(binding));
    }
    ASSERT_TRUE(engine.SetOptimizationProfile(0, nullptr));
    ASSERT_TRUE(engine.Enqueue(nullptr));
    engine.Synchronize(nullptr);

    std::cout << "[诊断] FP16 prefill 各输出（max|v| / 是否 NaN）\n";
    for (const Binding& binding : bindings) {
        if (binding.is_input) {
            continue;
        }
        std::vector<char> raw(binding.count * (binding.is_half ? 2u : 4u));
        CUDA_CHECK(cudaMemcpy(raw.data(), binding.buffer->data(), raw.size(),
                              cudaMemcpyDeviceToHost));
        float max_abs = 0.0f;
        bool has_nan = false;
        for (size_t k = 0; k < binding.count; ++k) {
            const float v = binding.is_half
                                ? __half2float(reinterpret_cast<const __half*>(raw.data())[k])
                                : reinterpret_cast<const float*>(raw.data())[k];
            if (std::isnan(v) || std::isinf(v)) {
                has_nan = true;
                break;
            }
            max_abs = std::max(max_abs, std::fabs(v));
        }
        std::cout << "        " << binding.name << "  " << (binding.is_half ? "FP16" : "FP32")
                  << "  max|v|=" << max_abs << "  NaN=" << (has_nan ? "是" : "否") << "\n";
    }
}

// A1-10：把"文本 → prompt token"这一段桥接起来验证（future_iterations A1 的收口用例）。
//
// 为什么是 **host** 用例而不是真机：这里要证明的只有"分词结果 == 既有基线常量"这一个变量；
// runner 的数值路径已由上面的 RealGpt2GreedyMatchesReferenceTokens（真机）覆盖。
// 再在真机上搭一遍双引擎，只会把有限的真机预算烧在没有新信息的路径上。
TEST(BpeTokenizerWithRunnerTest, TextPromptMatchesReferenceTokens) {
    const std::string dir = test_support::FindTokenizerDir();
    if (dir.empty()) GTEST_SKIP() << test_support::DescribeTokenizerProbe();

    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(dir)) << "资产存在但加载失败：" << dir;

    const std::vector<int64_t> from_text = tokenizer.Encode(kPromptText);
    EXPECT_EQ(from_text, kExpectedPrompt)
        << "文本 \"" << kPromptText << "\" 的分词与真机基线不一致——"
           "说明【文本入口】与【token 入口】是两条不同的路，先查 tokenizer 再看 runner";
}

// 清单 B：**文本进 / 文本出**的真机端到端。
//
// 与 A1-10 的分工：A1-10（host）只证明"分词 == 既有基线常量"；本用例把三段接起来——
//   文本 → Encode → LLMRunner → Decode → 文本，
// 从而把"文本入口"与 Phase 2 已真机验证过的 runner 数值路径合成一条链。
//
// 第 3 段的期望文本出处：本机 HF `GPT2TokenizerFast.decode([464, 2068, 7586, 21831, 274, 389,
// 257, 1049, 835, 284, 651, 257])` 的实测输出（transformers 4.44.0，离线 local_files_only），
// 命令与结果见 docs/future_iterations_test_plan.md §2.1 的 B 项说明。
constexpr const char* kExpectedFullText = "The quick brown foxes are a great way to get a";

TEST(Gpt2GenerateTest, RealGpt2TextPromptEndToEnd) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindRealModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/gpt2 不存在（先跑 hf_to_mini_trt_llm.py 转换）";
    }
    const std::string tokenizer_dir = test_support::FindTokenizerDir();
    if (tokenizer_dir.empty()) {
        GTEST_SKIP() << test_support::DescribeTokenizerProbe();
    }

    // 第 1 段：文本 -> token。分词不过关就没必要启动引擎（省一次真机往返的等待）。
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(tokenizer_dir)) << "tokenizer 加载失败：" << tokenizer_dir;
    const std::vector<int64_t> prompt = tokenizer.Encode(kPromptText);
    ASSERT_EQ(prompt, kExpectedPrompt) << "分词与基线不一致，先修 tokenizer";

    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;
    builder_config.min_prefill_batch = 1;
    builder_config.opt_prefill_batch = 1;
    builder_config.max_prefill_batch = 1;
    builder_config.min_prefill_seq_len = 1;
    builder_config.opt_prefill_seq_len = static_cast<int32_t>(kExpectedPrompt.size());
    builder_config.max_prefill_seq_len =
        static_cast<int32_t>(kExpectedPrompt.size() + kExpectedTokens.size());
    builder_config.min_decode_batch = 1;
    builder_config.opt_decode_batch = 1;
    builder_config.max_decode_batch = 1;

    // 引擎路径与 RealGpt2GreedyMatchesReferenceTokens 共用：同一份图，构建一次即可复用，
    // 不为了避免"共路径"而多花一次分钟级构建（缓存只按路径名区分的坑在这里正好是有利的，
    // 因为两条用例的建图配置逐字段相同）。
    EngineBuilder builder(logger, builder_config);
    const std::string prefill_path = "/tmp/mini_trt_llm_gpt2_real_prefill.engine";
    const std::string decode_path = "/tmp/mini_trt_llm_gpt2_real_decode.engine";
    ASSERT_TRUE(builder.BuildFromConfig(dir, prefill_path, BuildStage::kPrefill));
    ASSERT_TRUE(builder.BuildFromConfig(dir, decode_path, BuildStage::kDecode));

    LLMRunner::Config runner_config;
    runner_config.num_layers = kRealLayers;
    runner_config.num_kv_heads = kRealHeads;
    runner_config.head_size = kRealHeadSize;
    runner_config.block_size = kRealBlockSize;
    runner_config.max_blocks_per_seq = kRealPositions / kRealBlockSize;
    runner_config.num_blocks = 64;
    runner_config.is_half = false;
    runner_config.vocab_size = kRealVocab;
    runner_config.eos_token_id = kRealEos;

    LLMRunner runner(runner_config, std::make_shared<Engine>(prefill_path, logger),
                     std::make_shared<Engine>(decode_path, logger), nullptr);
    ASSERT_TRUE(runner.ok());

    // 第 2 段：token -> 新 token。
    LLMRunner::GenerateOptions options;
    options.max_new_tokens = static_cast<int>(kExpectedTokens.size());
    options.top_k = 1;
    const std::vector<int64_t> generated = runner.Generate(prompt, options);
    ASSERT_EQ(generated, kExpectedTokens)
        << "文本 prompt 的生成结果与 HF 基线不一致（这是 runner 侧的问题，不是分词）";

    // 第 3 段：token -> 文本。整段（prompt + 生成）一起解码，覆盖"续写会粘在前一个词上"这类
    // 边界——实测基线里 274 就是 "es"，与 "fox" 拼成 "foxes"。
    std::vector<int64_t> full = prompt;
    full.insert(full.end(), generated.begin(), generated.end());
    EXPECT_EQ(tokenizer.Decode(full), kExpectedFullText)
        << "解码结果与 HF 不一致（查 Decode 的 byte 回退路径）";
}

}  // namespace mini_trt_llm
