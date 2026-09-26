// decode 端到端性能画像（`future_iterations.md` §6.3 与 §11 的 G6）。
//
// 开发计划：`docs/future_iterations_development_plan.md` §11（P6_3-3）
// 测试计划：`docs/future_iterations_test_plan.md` §10（PF-3 / PF-6）
//
// **本文件只测量、不判定正确性**：P 层用例把观测值打印出来（中位数 / 分位 / 极差 /
// 斜率），不 assert 任何数值阈值。为什么：这是"回答一问"而不是"达标判据"——
// 给它设阈值会退化成 `AGENTS.md` §7 禁止的"用阈值换绿"（判据出处：
// `TROUBLESHOOTING.md` #37 / #38，判别下限约 ±400~600 µs）。
//
// **唯一的 assert 是"这一轮跑起来了"**（runner 构造成功、`Generate` 没返回空 vector）。
// 那是失败信号，不是性能阈值；测量质量的观测量（同 session 两次测量的漂移）只打印，
// 因为它没有可推导的阈值——真机第一次跑就是在这里把 profile target 判失败的
// （`TROUBLESHOOTING.md` #39）。

#include "perf_stats.hpp"
#include "gpt2_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/llm_runner.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "mini_trt_llm/utils/timer.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::Median;
using test_support::MinMax;
using test_support::Percentile;
using test_support::SlopePerExtraLaunch;

// ---------------------------------------------------------------------------
// H 层：统计量自身的语义锁（沙箱 / CI 可跑）
//
// 为什么要有它们：报告里出现的每个"中位数 / 分位 / 斜率"都要能回答"它怎么算的"。
// 内核在沙箱跑不了，但**统计学这一层能**——把它锁住，真机上就只剩"数字对不对"，
// 不会再有"口径不明"（`PROGRESS.md` §2.13）。
// ---------------------------------------------------------------------------

TEST(PerfStatsTest, MedianHandlesOddAndEvenCounts) {
    EXPECT_DOUBLE_EQ(Median({3.0, 1.0, 2.0}), 2.0);       // 奇数：取中间
    EXPECT_DOUBLE_EQ(Median({4.0, 1.0, 3.0, 2.0}), 2.5);  // 偶数：取平均
    EXPECT_DOUBLE_EQ(Median({5.0}), 5.0);                 // 单元素
}

TEST(PerfStatsTest, MedianOfEmptyIsDocumentedSentinel) {
    // 空输入的 0.0 是**哨兵**，不是"中位数就是 0"——调用方必须先保证非空。
    EXPECT_DOUBLE_EQ(Median({}), 0.0);
}

TEST(PerfStatsTest, PercentileUsesNearestRankWithoutInterpolation) {
    const std::vector<double> v = {10.0, 20.0, 30.0, 40.0};  // 已排序
    // 最近秩：p25 → ceil(0.25·4)=1 → 第 1 个（10）；p75 → ceil(3)=3 → 第 3 个（30）。
    // 若用线性插值，p25 会给出 17.5 / p75 给出 32.5 —— 那两个数从未被实测到。
    EXPECT_DOUBLE_EQ(Percentile(v, 25.0), 10.0);
    EXPECT_DOUBLE_EQ(Percentile(v, 75.0), 30.0);
    EXPECT_DOUBLE_EQ(Percentile(v, 0.0), 10.0);    // 下界
    EXPECT_DOUBLE_EQ(Percentile(v, 100.0), 40.0);  // 上界
    EXPECT_DOUBLE_EQ(Percentile({}, 50.0), 0.0);   // 哨兵
}

TEST(PerfStatsTest, MinMaxReportsBothEnds) {
    const auto mm = MinMax({2.0, -1.0, 7.5, 3.0});
    EXPECT_DOUBLE_EQ(mm.first, -1.0);
    EXPECT_DOUBLE_EQ(mm.second, 7.5);
}

TEST(PerfStatsTest, SlopeCancelsPerWindowFixedCost) {
    // 构造 T(k) = fixed + k · net，其中 fixed = 100（每窗口固定开销）、net = 2。
    const std::vector<double> windows = {102.0, 104.0, 106.0, 108.0};
    EXPECT_DOUBLE_EQ(SlopePerExtraLaunch(windows), 2.0);
    // 少于 2 个窗口无法相减 → 哨兵 0.0。
    EXPECT_DOUBLE_EQ(SlopePerExtraLaunch({5.0}), 0.0);
    EXPECT_DOUBLE_EQ(SlopePerExtraLaunch({}), 0.0);
}

// ---------------------------------------------------------------------------
// G / P 层：真实 GPT-2 的 decode 计时
// ---------------------------------------------------------------------------

// 与 `models/gpt2/config.json` 一致的真实规模（与 `test_gpt2_generate.cpp` 的常量同源）。
constexpr int32_t kRealLayers = 12;
constexpr int32_t kRealHeads = 12;
constexpr int32_t kRealHeadSize = 64;
constexpr int32_t kRealVocab = 50257;
constexpr int32_t kRealBlockSize = 16;
constexpr int32_t kRealPositions = 1024;

// 固定 prompt（"The quick brown fox" 的 HF 分词）。与精度用例同源，否则
// "两次测量的输入不同"会让数字不可比。
const std::vector<int64_t> kPromptTokens = {464, 2068, 7586, 21831};

constexpr int32_t kMaxNewTokens = 32;  // decode 步数（决定 KV 增长与每步工作量）
constexpr int32_t kRounds = 15;        // G6：≥15 轮（同一 session、同一二进制）
constexpr int32_t kWarmup = 3;

// **采样器类**比较的判别下限（出处：`TROUBLESHOOTING.md` #37 / #38 的实测）。
// **只用来提醒"别把采样器那把尺子套到整步 decode 上"**——量级不同、阈值不可跨场景复用。
constexpr double kDiscriminationFloorMs = 0.6;

std::string FindRealModelDir() {
    if (const char* env = std::getenv("MINI_TRT_LLM_GPT2_DIR")) {
        if (std::filesystem::exists(std::string(env) + "/config.json")) {
            return env;
        }
    }
    const char* candidates[] = {"models/gpt2", "../models/gpt2", "../../models/gpt2",
                                "../../../models/gpt2"};
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(std::string(candidate) + "/config.json")) {
            return candidate;
        }
    }
    return {};
}

double NowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// 造一个固定长度的 prompt。token **取值**对计时没有影响，但必须是合法 id
// （vocab = 50257）：用 `1 + i%1000` 避开 0 与特殊 id 的边界讨论。
std::vector<int64_t> MakePrompt(int32_t length) {
    std::vector<int64_t> tokens(static_cast<size_t>(length));
    for (int32_t i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] = 1 + (i % 1000);
    }
    return tokens;
}

// 记录温度 / 功耗（协议 §11.3 A）。拿不到不是失败——WSL2 上 nvidia-smi 可能受限，
// 而"采集不到"必须**显式写出来**，不能让读者以为它被量过（`AGENTS.md` §7 的"观测缺口"）。
void PrintGpuState(const char* tag) {
    const char* cmd =
        "nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader";
    FILE* pipe = popen(cmd, "r");
    if (pipe == nullptr) {
        std::cout << "[Gpt2DecodePerf] " << tag << " GPU 状态：<nvidia-smi 不可用>\n";
        return;
    }
    std::array<char, 256> buf{};
    const size_t n = fread(buf.data(), 1, buf.size() - 1, pipe);
    pclose(pipe);
    std::string text(buf.data(), n);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    std::cout << "[Gpt2DecodePerf] " << tag << " GPU 状态 (temp C, sm MHz, power W)="
              << (text.empty() ? "<无输出>" : text) << "\n";
}

void PrintStats(const char* label, const std::vector<double>& samples) {
    const double median = Median(samples);
    const auto mm = MinMax(samples);
    std::cout << "[Gpt2DecodePerf]   " << label << " median=" << median
              << " p25=" << Percentile(samples, 25.0) << " p75=" << Percentile(samples, 75.0)
              << " min=" << mm.first << " max=" << mm.second
              << " (n=" << samples.size() << ")\n";
}

// PF-3：Prefill 一步 vs Decode 每步的耗时。
//
// **怎么把两段分开而不改 runner**：用斜率口径——测 T(1) 与 T(32)：
//   T(1)  ≈ prefill + 1 个 decode 步
//   T(32) ≈ prefill + 32 个 decode 步
// 于是 每个 decode 步 = (T(32) − T(1)) / 31，prefill ≈ T(1) − 每个 decode 步。
// 这样既不需要在 runner 里插桩（本轮"零产品代码改动"），也顺带扣掉了每次调用的固定开销。
TEST(Gpt2DecodePerf, StepLatencyByPhase) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindRealModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/gpt2 不存在（先跑 hf_to_mini_trt_llm.py 转换）";
    }

    Logger logger;
    EngineBuilder::Config builder_config;
    // FP16 端到端产 NaN（PROGRESS §5.11）→ 尺子架在可用路径（FP32）上。
    builder_config.precision = Precision::FP32;
    builder_config.min_prefill_batch = 1;
    builder_config.opt_prefill_batch = 1;
    builder_config.max_prefill_batch = 1;
    builder_config.min_prefill_seq_len = 1;
    builder_config.opt_prefill_seq_len = static_cast<int32_t>(kPromptTokens.size());
    // 与 `Gpt2GenerateTest.RealGpt2Greedy...` 完全相同的形状参数——**共用同一条引擎路径**
    // 才可能命中缓存；参数一旦不同，指纹就变了，会触发分钟级重建。
    builder_config.max_prefill_seq_len = static_cast<int32_t>(kPromptTokens.size() + 8);
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
    runner_config.max_blocks_per_seq = kRealPositions / kRealBlockSize;
    runner_config.num_blocks = 64;
    runner_config.is_half = false;
    runner_config.vocab_size = kRealVocab;
    // 关掉 EOS 截断：一旦命中，返回的 token 变少 → 测到的是"更少的步数"，
    // 计时的分母就错了（这是"诊断口径"错误，不是性能现象）。
    runner_config.eos_token_id = -1;

    LLMRunner runner(runner_config, std::make_shared<Engine>(prefill_path, logger),
                     std::make_shared<Engine>(decode_path, logger), nullptr);
    ASSERT_TRUE(runner.ok());

    LLMRunner::GenerateOptions options;
    options.top_k = 1;  // greedy：消除采样策略分支的差异
    options.top_p = 1.0f;

    bool failed = false;
    auto time_once = [&](int max_new_tokens) -> double {
        LLMRunner::GenerateOptions local = options;
        local.max_new_tokens = max_new_tokens;
        const double t0 = NowMs();
        const std::vector<int64_t> out = runner.Generate(kPromptTokens, local);
        const double t1 = NowMs();
        if (out.empty()) {
            failed = true;
        }
        return t1 - t0;
    };

    PrintGpuState("(before)");
    for (int32_t i = 0; i < kWarmup; ++i) {
        time_once(1);
        time_once(kMaxNewTokens);
    }
    ASSERT_FALSE(failed) << "warmup 期间 Generate 返回空 vector（失败），先查引擎/cache";

    std::vector<double> one_token_ms;
    std::vector<double> full_ms;
    for (int32_t round = 0; round < kRounds; ++round) {
        // ABBA：正反交替，抵消"谁先测谁后测"的系统偏差（§11.3 B）。
        if (round % 2 == 0) {
            one_token_ms.push_back(time_once(1));
            full_ms.push_back(time_once(kMaxNewTokens));
        } else {
            full_ms.push_back(time_once(kMaxNewTokens));
            one_token_ms.push_back(time_once(1));
        }
    }
    ASSERT_FALSE(failed) << "测量期间 Generate 返回空 vector（失败）";

    const double median_one = Median(one_token_ms);
    const double median_full = Median(full_ms);
    const double per_decode_step = (median_full - median_one) / (kMaxNewTokens - 1);
    const double prefill_estimate = median_one - per_decode_step;

    std::cout << "[Gpt2DecodePerf] 形状：batch=1 prompt=" << kPromptTokens.size()
              << " new_tokens=" << kMaxNewTokens << " vocab=" << kRealVocab
              << " precision=FP32 top_k=1\n";
    std::cout << "[Gpt2DecodePerf] 构建态：上面若有 `Engine cache hit` 即复用；"
                 "`stale` 表示本轮重建（跨构建结论必须报极差，见 §11.3 B）\n";
    std::cout << "[Gpt2DecodePerf] warmup=" << kWarmup << " rounds=" << kRounds
              << "（每轮 ABBA）\n";
    PrintStats("T(1)  ms", one_token_ms);
    PrintStats("T(32) ms", full_ms);
    std::cout << "[Gpt2DecodePerf] 派生：decode 每步=" << per_decode_step
              << " ms（斜率口径 (T32−T1)/31）；prefill≈" << prefill_estimate << " ms\n";
    std::cout << "[Gpt2DecodePerf] 采样器类的判别下限=" << kDiscriminationFloorMs
              << " ms（#37/#38）；**它不套用到整步 decode 上**——量级不同，见下\n";
    PrintGpuState("(after)");

    // 同一 session 内把整组测量再跑一遍，报**绝对与相对**两次漂移。
    //
    // **为什么只打印、不 assert**：这是"测量质量"的观测量，不是被测量的正确性属性。
    // 曾经在这里写死 `EXPECT_LT(drift, 0.6 ms)`——那个 0.6 ms 是**采样器类**比较的
    // 判别下限（#37/#38），套到量级大一到两个数量级的整步 decode 上属于"阈值跨场景
    // 复用"，正是 `AGENTS.md` §7 禁止的"阈值来路不明"。真机第一次跑就因此把整个
    // profile target 判失败（`TROUBLESHOOTING.md` #39）。
    // 现在的口径：打印绝对 / 相对漂移，由读者判断"漂移是否远小于待判信号"。
    std::vector<double> repeat_one;
    std::vector<double> repeat_full;
    for (int32_t round = 0; round < kRounds; ++round) {
        repeat_one.push_back(time_once(1));
        repeat_full.push_back(time_once(kMaxNewTokens));
    }
    const double drift_one = std::fabs(Median(repeat_one) - median_one);
    const double drift_full = std::fabs(Median(repeat_full) - median_full);
    const double rel_one = (median_one > 0.0) ? 100.0 * drift_one / median_one : 0.0;
    const double rel_full = (median_full > 0.0) ? 100.0 * drift_full / median_full : 0.0;
    std::cout << "[Gpt2DecodePerf] 复现性（同 session 两组测量）："
              << "|Δmedian T(1)|=" << drift_one << " ms (" << rel_one << "%)，"
              << "|Δmedian T(32)|=" << drift_full << " ms (" << rel_full << "%)\n";
    std::cout << "[Gpt2DecodePerf] 读法：漂移若与待判差异同量级，本轮结论无效；"
                 "本次测量无阈值判据（测试计划 §10.3）\n";

    // ------------------------------------------------------------------
    // 同 session 的 sampler 净成本 → **占 decode 一步的比例**（`TROUBLESHOOTING.md` #41 绕法 1）
    //
    // 为什么放在这里而不是另起一个用例：本机两条 CLI profiling 路径都拿不到 kernel
    // 时间线（nsys 无 kernel 数据、ncu 报 `Unknown Error on device 0`），
    // 而"sampler 占整步 decode 多少"只要两个数**同处一个 session** 就能回答。
    // 放在同一个用例里 → 顺序、温度、时钟全部一致，不存在跨 session 可比性问题（#38）。
    //
    // 量法沿用 §9.2 的**斜率口径**：同一份 logits 上发射 1 次与 4 次，取 `(T4−T1)/3`
    // ——扣掉每窗口固定开销（事件 + 同步 + 首次发射），剩下的才是 kernel 净成本。
    //
    // **边界（必须一起读）**：
    //   * 这是**比值**，不是逐 kernel 分解；attention / MLP 仍然包在"decode 一步"里；
    //   * logits 是静态缓冲（不是引擎刚写出来的那份），cache 状态与真实循环不同；
    //   * 因此结论只到"占比量级"，要精确分解仍得靠 profiler（见 #41）。
    // ------------------------------------------------------------------
    {
        DeviceBuffer d_logits(static_cast<size_t>(kRealVocab) * sizeof(float));
        DeviceBuffer d_tokens(sizeof(int32_t));
        DeviceBuffer d_k(sizeof(int32_t));
        DeviceBuffer d_p(sizeof(float));
        const size_t ws_topk = TopKSamplerWorkspaceBytes(/*batch=*/1, kRealVocab);
        const size_t ws_topp = TopPSamplerWorkspaceBytes(/*batch=*/1, kRealVocab);
        DeviceBuffer d_ws_topk(ws_topk);
        DeviceBuffer d_ws_topp(ws_topp);
        if (!d_logits.Allocate(static_cast<size_t>(kRealVocab) * sizeof(float)) ||
            !d_tokens.Allocate(sizeof(int32_t)) || !d_k.Allocate(sizeof(int32_t)) ||
            !d_p.Allocate(sizeof(float)) || !d_ws_topk.Allocate(ws_topk) ||
            !d_ws_topp.Allocate(ws_topp)) {
            ADD_FAILURE() << "sampler 占比测量：显存分配失败";
        } else {
            const int32_t kTopK = 64;
            const float kTopP = 0.9f;
            // logits 用与 `SamplerPerf` **同一套确定性模式**（`sin(0.29·i)·1.2`），两个原因：
            //   ① 全等输入是退化情形——真实的 logits 行不会全等，排序路径的负载也就不能代表它；
            //   ② 只有同一套输入，本段的数与 `SamplerPerf` 的数才**直接可比**：
            //      首轮实测两者对同一 (k=64, p=0.9, vocab=50257) 差了约 2.5 倍，
            //      而 greedy（不排序）却几乎一致（见 TROUBLESHOOTING #42）。
            std::vector<float> host_logits(static_cast<size_t>(kRealVocab));
            for (int32_t i = 0; i < kRealVocab; ++i) {
                host_logits[static_cast<size_t>(i)] =
                    std::sin(0.29f * static_cast<float>(i)) * 1.2f;
            }
            CUDA_CHECK(cudaMemcpy(d_logits.data(), host_logits.data(), d_logits.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_k.data(), &kTopK, sizeof(int32_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_p.data(), &kTopP, sizeof(float),
                                  cudaMemcpyHostToDevice));

            SamplerArgs greedy_args;
            greedy_args.logits = d_logits.data();
            greedy_args.token_ids = static_cast<int32_t*>(d_tokens.data());
            greedy_args.batch_size = 1;
            greedy_args.vocab_size = kRealVocab;
            greedy_args.is_half = false;
            greedy_args.seed = 42;

            TopKSamplerArgs topk_args;
            static_cast<SamplerArgs&>(topk_args) = greedy_args;
            topk_args.top_k = static_cast<const int32_t*>(d_k.data());

            TopPSamplerArgs topp_args;
            static_cast<SamplerArgs&>(topp_args) = greedy_args;
            topp_args.top_p = static_cast<const float*>(d_p.data());

            // 三次变体都是**runner 可能走到的**那几条路（LLMRunner::SampleInto 的分派）。
            const std::function<void()> launch_greedy = [&]() {
                CUDA_CHECK(LaunchGreedySampler(greedy_args, nullptr));
            };
            const std::function<void()> launch_topk = [&]() {
                CUDA_CHECK(LaunchTopKSampler(topk_args, nullptr, d_ws_topk.data(), ws_topk));
            };
            const std::function<void()> launch_topp = [&]() {
                CUDA_CHECK(LaunchTopPSampler(topp_args, nullptr, d_ws_topp.data(), ws_topp));
            };

            CudaTimer sampler_timer;
            const auto time_n = [&sampler_timer](const std::function<void()>& launch,
                                                 int32_t repeats) {
                sampler_timer.Start(nullptr);
                for (int32_t r = 0; r < repeats; ++r) {
                    launch();
                }
                return sampler_timer.Stop(nullptr);
            };
            constexpr int32_t kSamplerRounds = 9;
            const auto slope_samples = [&](const std::function<void()>& launch) {
                std::vector<double> samples;
                for (int32_t i = 0; i < kSamplerRounds; ++i) {
                    // 正反交替，抵消先后顺序带来的系统偏差（与主测量同一纪律）。
                    const double forward = (time_n(launch, 4) - time_n(launch, 1)) / 3.0;
                    const double backward = (time_n(launch, 1) - time_n(launch, 4)) / -3.0;
                    samples.push_back(0.5 * (forward + backward));
                }
                return samples;
            };

            std::cout << "[Gpt2DecodePerf] ---- 同 session 的 sampler 占比（斜率口径，n="
                      << kSamplerRounds << "）----\n";
            const auto report_share = [&](const char* name,
                                          const std::vector<double>& samples) {
                const double net = Median(samples);
                const double share = (per_decode_step > 0.0) ? 100.0 * net / per_decode_step : 0.0;
                std::cout << "[Gpt2DecodePerf]   " << name << " 净=" << net << " ms/次 → 占 decode 每步("
                          << per_decode_step << " ms) 的 " << share << "%  [p25="
                          << Percentile(samples, 25.0) << " p75=" << Percentile(samples, 75.0)
                          << "]\n";
            };
            report_share("greedy（本用例 decode 实际走的路）", slope_samples(launch_greedy));
            report_share("top-k(k=64)", slope_samples(launch_topk));
            report_share("top-p(p=0.9)", slope_samples(launch_topp));
            std::cout << "[Gpt2DecodePerf] 边界：比值而非逐 kernel 分解；attention/MLP 仍包在"
                         "decode 一步里；logits 是静态缓冲（见测试计划 §10.4、#41）\n";
        }
    }
}

// ---------------------------------------------------------------------------
// PF-9：attention 随上下文长度的增长（profiler 拿不到 kernel 时间线时的替代）
//
// 依据：开发计划 §11.4.1、`TROUBLESHOOTING.md` #41。
//
// **为什么这样做能替代 profiler**：attention 的开销随**已缓存位置数**增长，而每步的
// matmul / LayerNorm / GELU / KV 写入 / 位置填充 / 采样都与上下文无关。所以
// "长 prompt 的每步耗时 − 短 prompt 的每步耗时"就是 attention 的**边际成本**。
// `LLMRunner::Generate` 每次调用开头都 `FreeSequence` + `AllocateSequence`
// （`src/core/llm_runner.cpp`），所以 prompt 长度直接决定上下文区间，不需要改 runner。
//
// **边界**：① 这是"随上下文增长的部分"，不是 attention 的绝对时间；② 三个 prompt 共用
// 同一个 prefill 引擎（opt 形状对 kernel 选择有影响），横向自洽但绝对值不可与别处直接比；
// ③ 只打印、不设阈值（P 层纪律，测试计划 §10.3）。
// ---------------------------------------------------------------------------
TEST(Gpt2DecodePerf, ContextLengthSweep) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindRealModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/gpt2 不存在（先跑 hf_to_mini_trt_llm.py 转换）";
    }

    constexpr int32_t kPromptLengths[] = {4, 256, 960};
    constexpr int32_t kSweepRounds = 7;
    constexpr int32_t kSweepWarmup = 1;
    // prompt(960) + max_new_tokens(32) = 992，留一点余量。
    constexpr int32_t kPrefillMaxSeq = 992;

    Logger logger;
    EngineBuilder::Config builder_config;
    builder_config.precision = Precision::FP32;  // FP16 端到端产 NaN（PROGRESS §5.11）
    builder_config.min_prefill_batch = 1;
    builder_config.opt_prefill_batch = 1;
    builder_config.max_prefill_batch = 1;
    builder_config.min_prefill_seq_len = 1;
    // opt 取 512：让 TRT 按"长 prompt 场景"挑 kernel；三档 prompt 共用这一个引擎。
    builder_config.opt_prefill_seq_len = 512;
    builder_config.max_prefill_seq_len = kPrefillMaxSeq;
    builder_config.min_decode_batch = 1;
    builder_config.opt_decode_batch = 1;
    builder_config.max_decode_batch = 1;
    EngineBuilder builder(logger, builder_config);

    // **两条都必须用自己的路径**：指纹覆盖的是**整个** `EngineBuilder::Config`，
    // 这个用例的 `max_prefill_seq_len` 与主用例不同 → **两个引擎的指纹都变了**。
    // 曾经这里只给 prefill 换了路径、decode 复用主用例那条，结果真机上直接看到
    // `Engine cache stale: gpt2_real_decode.engine → 重建`：两个用例会**交替**把
    // 对方的 decode 引擎判为过期，每次来回都要重建一次（分钟级）。
    // ——这正是 `PROGRESS.md` §2.15 "缓存路径不得在不同配置间共用"那条规矩。
    const std::string prefill_path = "/tmp/mini_trt_llm_gpt2_ctxsweep_prefill.engine";
    const std::string decode_path = "/tmp/mini_trt_llm_gpt2_ctxsweep_decode.engine";
    ASSERT_TRUE(builder.BuildFromConfig(dir, prefill_path, BuildStage::kPrefill))
        << "上下文扫描的 prefill 引擎构建失败（首次会构建，分钟级）";
    ASSERT_TRUE(builder.BuildFromConfig(dir, decode_path, BuildStage::kDecode))
        << "decode 引擎构建失败";

    LLMRunner::Config runner_config;
    runner_config.num_layers = kRealLayers;
    runner_config.num_kv_heads = kRealHeads;
    runner_config.head_size = kRealHeadSize;
    runner_config.block_size = kRealBlockSize;
    runner_config.max_blocks_per_seq = kRealPositions / kRealBlockSize;  // 64 块 = 1024 位置
    runner_config.num_blocks = 64;
    runner_config.is_half = false;
    runner_config.vocab_size = kRealVocab;
    runner_config.eos_token_id = -1;  // 不截断：否则分母（步数）就不是 32 了

    LLMRunner runner(runner_config, std::make_shared<Engine>(prefill_path, logger),
                     std::make_shared<Engine>(decode_path, logger), nullptr);
    ASSERT_TRUE(runner.ok());

    LLMRunner::GenerateOptions options;
    options.top_k = 1;  // greedy：与主用例一致，排除采样策略差异
    options.top_p = 1.0f;

    PrintGpuState("(before)");
    std::cout << "[Gpt2DecodePerf] ---- 上下文长度扫描（PF-9）----\n";
    std::cout << "[Gpt2DecodePerf]  口径：prompt∈{4,256,960}，每档 warmup=" << kSweepWarmup
              << " rounds=" << kSweepRounds << "，每步=(T32−T1)/31；只打印不设阈值\n";

    bool failed = false;
    std::vector<double> avg_contexts;
    std::vector<double> avg_steps;
    for (int32_t prompt_len : kPromptLengths) {
        const std::vector<int64_t> prompt = MakePrompt(prompt_len);
        auto time_once = [&](int max_new_tokens) -> double {
            LLMRunner::GenerateOptions local = options;
            local.max_new_tokens = max_new_tokens;
            const double t0 = NowMs();
            const std::vector<int64_t> out = runner.Generate(prompt, local);
            const double t1 = NowMs();
            if (out.empty()) {
                failed = true;
            }
            return t1 - t0;
        };

        for (int32_t i = 0; i < kSweepWarmup; ++i) {
            time_once(1);
            time_once(kMaxNewTokens);
        }
        std::vector<double> one_token_ms;
        std::vector<double> full_ms;
        for (int32_t round = 0; round < kSweepRounds; ++round) {
            if (round % 2 == 0) {
                one_token_ms.push_back(time_once(1));
                full_ms.push_back(time_once(kMaxNewTokens));
            } else {
                full_ms.push_back(time_once(kMaxNewTokens));
                one_token_ms.push_back(time_once(1));
            }
        }
        const double median_one = Median(one_token_ms);
        const double median_full = Median(full_ms);
        const double per_step = (median_full - median_one) / (kMaxNewTokens - 1);
        // T(32) 覆盖的是"上下文从 p+1 涨到 p+32"这一段，取其均值作为横坐标。
        const double avg_context = prompt_len + 0.5 * (kMaxNewTokens + 1);
        avg_contexts.push_back(avg_context);
        avg_steps.push_back(per_step);
        std::cout << "[Gpt2DecodePerf]  prompt=" << prompt_len << " → 每步 " << per_step
                  << " ms（平均上下文≈" << avg_context << "；T(1)=" << median_one
                  << " T(32)=" << median_full << "）\n";
    }
    ASSERT_FALSE(failed) << "扫描期间 Generate 返回空 vector（失败）";

    const double delta_context = avg_contexts.back() - avg_contexts.front();
    const double delta_step = avg_steps.back() - avg_steps.front();
    const double per_1k = (delta_context > 0.0) ? 1000.0 * delta_step / delta_context : 0.0;
    std::cout << "[Gpt2DecodePerf]  斜率：上下文 +" << delta_context << " 个位置 → 每步 +"
              << delta_step << " ms（≈ 每 1000 位置 " << per_1k << " ms）\n";
    std::cout << "[Gpt2DecodePerf]  线性外推到 1024 位置：≈ 每步 +"
              << per_1k * (1024.0 - avg_contexts.front()) / 1000.0
              << " ms（相对最短上下文；**线性假设未验证**）\n";
    std::cout << "[Gpt2DecodePerf]  读法：这个增量就是 attention 的边际成本——"
                 "它占短上下文每步的比例，决定 §2.2 的取舍\n";
    PrintGpuState("(after)");
}

}  // namespace
}  // namespace mini_trt_llm
