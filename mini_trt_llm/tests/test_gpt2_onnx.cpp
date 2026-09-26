#include "gpt2_test_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cuda_fp16.h>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::ComputeDiffStats;
using test_support::DiffStats;
using test_support::ArgmaxAgreement;
using test_support::CompareArgmaxByDecidability;

constexpr int32_t kVocab = 50257;
constexpr int32_t kSeq = 4;
const std::vector<int64_t> kPrompt = {464, 2068, 7586, 21831};

// 每个形状**允许不可判的行号**（钉死版，替代早先的"计数 ≤ 2"上界）。
//
// **这张表怎么来的**：2026-09-26 真机实测（`TROUBLESHOOTING.md` §34.9 末尾的表）——
// `(1,512)` 上只有第 118 行不可判（余量 1.53e-05 vs 两侧差 1.14e-04，两个引擎各自都只"看到"
// 约 1.5e-05 的间距且方向相反），其余四个形状 0 行。
//
// **为什么要钉到行号而不是只数个数**：并列是**图对的性质**，不是随机量。钉住之后，
// 任何**新增**的不可判行都会让这条用例报红——那意味着要么图变了、要么某处真坏了，
// 两种都需要人看一眼。只数个数会让"换一行并列"悄悄通过。
//
// **要改这张表，必须给出实测依据**（真机跑一遍，把新数字与原因写进 `TROUBLESHOOTING.md` #34.9）
// ——不许为了让用例变绿而加行（`AGENTS.md` §7）。
struct ShapeUndecidableRows {
    int32_t batch;
    int32_t seq;
    std::vector<int32_t> rows;
};

const std::vector<ShapeUndecidableRows>& ExpectedUndecidableRows() {
    static const std::vector<ShapeUndecidableRows> kTable = {
        {1, 1, {}},
        {1, 64, {}},
        {1, 512, {118}},
        {2, 4, {}},
        {2, 64, {}},
    };
    return kTable;
}

const ShapeUndecidableRows* FindExpectedUndecidableRows(int32_t batch, int32_t seq) {
    for (const auto& entry : ExpectedUndecidableRows()) {
        if (entry.batch == batch && entry.seq == seq) return &entry;
    }
    return nullptr;
}

std::string FindFile(const std::vector<std::string>& candidates) {
    for (const std::string& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

std::string FindModelDir() {
    return FindFile({"models/gpt2", "../models/gpt2", "../../models/gpt2",
                     "../../../models/gpt2"});
}

std::string FindOnnx() {
    return FindFile({"1_gpt2_onnx/gpt2.onnx", "../1_gpt2_onnx/gpt2.onnx",
                     "../../1_gpt2_onnx/gpt2.onnx", "../../../1_gpt2_onnx/gpt2.onnx"});
}

std::string FindRefOutput() {
    return FindFile({"1_gpt2_onnx/ref_output.bin", "../1_gpt2_onnx/ref_output.bin",
                     "../../1_gpt2_onnx/ref_output.bin",
                     "../../../1_gpt2_onnx/ref_output.bin"});
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

bool ArgmaxAll(const std::vector<float>& logits, std::vector<int32_t>* out) {
    out->clear();
    for (int32_t row = 0; row < kSeq; ++row) {
        const float* begin = logits.data() + static_cast<size_t>(row) * kVocab;
        out->push_back(static_cast<int32_t>(
            std::distance(begin, std::max_element(begin, begin + kVocab))));
    }
    return true;
}

// 跑一个引擎，返回 [1, kSeq, vocab] 的 logits。
//
// **必须从引擎查询 I/O，而不是假定**：两条图的契约并不相同——
//   * 原生图：`input_ids` + `position_ids`，两者都是 INT32；
//   * ONNX 图：只有 `input_ids`（位置编码是图内的常量），且是 **INT64**
//     （PyTorch 导出索引张量的惯例，TRT 会警告 "Make sure input input_ids has Int64 binding"）。
// 辅助函数里写死任一方的契约，都会在另一方上以"引擎报无效张量名 / 数值错"的形式失败。
// batch 感知：tokens 是 [batch, seq] 的行主序展平，`batch` 默认 1（此时 seq = tokens.size()）。
// **为什么必须参数化**：ONNX 路径的 profile 允许 batch 1..4，而只测 batch=1 等于没覆盖
// 那个维度（缺口 G4b）——分页/掩码之外，batch 维主要影响 profile 选择与绑定形状。
std::vector<float> RunEngine(Engine* engine, const std::vector<int64_t>& prompt,
                             int32_t batch = 1) {
    const int32_t kSeqRun = static_cast<int32_t>(prompt.size()) / batch;
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        return {};
    }
    std::vector<std::string> inputs;
    for (int32_t i = 0; i < cuda->getNbIOTensors(); ++i) {
        const char* name = cuda->getIOTensorName(i);
        if (cuda->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputs.emplace_back(name);
        }
    }

    const size_t elements = prompt.size();
    std::vector<int32_t> tokens32(elements);
    std::vector<int64_t> tokens64(elements);
    std::vector<int32_t> positions(elements);
    for (int32_t b = 0; b < batch; ++b) {
        for (int32_t i = 0; i < kSeqRun; ++i) {
            const size_t idx = static_cast<size_t>(b) * kSeqRun + static_cast<size_t>(i);
            tokens32[idx] = static_cast<int32_t>(prompt[idx]);
            tokens64[idx] = prompt[idx];
            positions[idx] = i;  // 每个 batch 行都从位置 0 开始
        }
    }

    struct Binding {
        std::string name;
        std::unique_ptr<DeviceBuffer> buffer;
    };
    std::vector<Binding> bindings;
    for (const std::string& name : inputs) {
        auto buffer = std::make_unique<DeviceBuffer>();
        size_t bytes = 0;
        const nvinfer1::DataType dtype = cuda->getTensorDataType(name.c_str());
        if (name == "position_ids") {
            bytes = elements * sizeof(int32_t);
        } else if (dtype == nvinfer1::DataType::kINT64) {
            bytes = elements * sizeof(int64_t);
        } else {
            bytes = elements * sizeof(int32_t);
        }
        if (!buffer->Allocate(bytes)) {
            return {};
        }
        if (name == "position_ids") {
            CUDA_CHECK(cudaMemcpy(buffer->data(), positions.data(), bytes,
                                  cudaMemcpyHostToDevice));
        } else if (dtype == nvinfer1::DataType::kINT64) {
            CUDA_CHECK(cudaMemcpy(buffer->data(), tokens64.data(), bytes,
                                  cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(buffer->data(), tokens32.data(), bytes,
                                  cudaMemcpyHostToDevice));
        }
        if (!engine->SetInputShape(name, nvinfer1::Dims{2, {batch, kSeqRun}}) ||
            !engine->SetTensorAddress(name, buffer->data())) {
            return {};
        }
        bindings.push_back(Binding{name, std::move(buffer)});
    }

    const size_t count = elements * static_cast<size_t>(kVocab);
    // 输出精度也要**按引擎声明的**读：FP16 引擎的 `logits` 可能是 half，
    // 按 float 读会得到 32 位错位的垃圾（同 #17 的 INT64 教训）。
    const nvinfer1::DataType logits_type = cuda->getTensorDataType("logits");
    const size_t logits_elem = logits_type == nvinfer1::DataType::kHALF ? 2u : 4u;
    DeviceBuffer d_logits(count * logits_elem);
    if (!d_logits.Allocate(count * logits_elem) ||
        !engine->SetTensorAddress("logits", d_logits.data()) ||
        !engine->SetOptimizationProfile(0, nullptr) ||
        !engine->Enqueue(nullptr)) {
        return {};
    }
    engine->Synchronize(nullptr);
    std::vector<float> logits(count);
    if (logits_type == nvinfer1::DataType::kHALF) {
        std::vector<__half> raw(count);
        CUDA_CHECK(cudaMemcpy(raw.data(), d_logits.data(), count * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            logits[i] = __half2float(raw[i]);
        }
    } else {
        CUDA_CHECK(cudaMemcpy(logits.data(), d_logits.data(), count * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    return logits;
}

}  // namespace

// P3-2：ONNX 路径（方案 B）与原生构建（方案 A）在**同一输入**下的 logits 对拍。
//
// 为什么这个判据干净：两条路用的是**同一份数值**——`gpt2.onnx` 的 initializer 与
// `models/gpt2/model.safetensors` 已核对逐比特一致（见 phase3 计划 §0.1）。
// 所以任何差异都只可能来自"图的分解方式"或"我们的实现"，不涉及外部基线是否可靠。
// 同时与 `ref_output.bin`（HF FP32）做三方对照。
TEST(Gpt2OnnxTest, MatchesNativeBuildOnSamePrompt) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string onnx = FindOnnx();
    const std::string ref_path = FindRefOutput();
    if (dir.empty() || onnx.empty() || ref_path.empty()) {
        GTEST_SKIP() << "需要 models/gpt2、1_gpt2_onnx/gpt2.onnx 与 ref_output.bin";
    }

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP32;  // 与方案 A 的对拍用同一精度
    config.min_prefill_seq_len = kSeq;
    config.opt_prefill_seq_len = kSeq;
    config.max_prefill_seq_len = kSeq;
    EngineBuilder builder(logger, config);

    const std::string onnx_engine = "/tmp/mini_trt_llm_gpt2_onnx.engine";
    // P3-4：构建耗时是"要不要做子图替换（D1=B）"的输入数据之一——若 ONNX 解析+构建
    // 本身就占了大头，优化图上的算子分布并不划算。
    double onnx_build_s = 0.0;
    {
        const auto begin = std::chrono::steady_clock::now();
        // D1=C：只声明要核对的子图，不做替换（识别由 tools/inspect_onnx.py --check 落地）
        // 缓存命中时这次调用是"读缓存"而不是构建，计时因此接近 0——与改动前的行为一致
        // （改动前是"文件不存在才构建并计时"）。
        ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, onnx_engine,
                                          {"attention", "layernorm", "position_embedding"}))
            << "ONNX 引擎构建失败";
        onnx_build_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - begin).count();
    }
    const std::string native_engine = "/tmp/mini_trt_llm_gpt2_accuracy.engine";
    // 原生侧也要计时：只量一边的构建耗时无法比较，也回答不了
    // "ONNX 的解析+建图开销占多少"（P3-4 的原始目的）。
    double native_build_s = 0.0;
    {
        const auto begin = std::chrono::steady_clock::now();
        ASSERT_TRUE(builder.BuildFromConfig(dir, native_engine, BuildStage::kSingle))
            << "原生引擎构建失败";
        native_build_s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - begin).count();
    }
    Engine onnx_engine_ctx(onnx_engine, logger);
    Engine native_engine_ctx(native_engine, logger);

    // 把两条路的 I/O 契约差异**显式打出来**：ONNX 图（torch 导出）只有 input_ids 且是 INT64，
    // 原生图还有 position_ids 且是 INT32。这正是"ONNX 路径要成为一等公民"目前差的一步——
    // 若要让两条路可互换，需要在 ONNX 路径把输入统一（加 Cast / 改名），属图改造，需另行确认。
    for (const auto& item : {std::make_pair("ONNX", &onnx_engine_ctx),
                             std::make_pair("原生", &native_engine_ctx)}) {
        nvinfer1::ICudaEngine* cuda = item.second->GetCudaEngine();
        std::string summary;
        for (int32_t i = 0; i < cuda->getNbIOTensors(); ++i) {
            const char* name = cuda->getIOTensorName(i);
            if (cuda->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) {
                continue;
            }
            const nvinfer1::DataType dtype = cuda->getTensorDataType(name);
            summary += std::string(name) + "(" +
                       (dtype == nvinfer1::DataType::kINT64 ? "INT64" : "INT32") + ") ";
        }
        std::cout << "[诊断] " << item.first << " 引擎输入: " << summary << "\n";
    }

    const std::vector<float> onnx_logits = RunEngine(&onnx_engine_ctx, kPrompt);
    const std::vector<float> native_logits = RunEngine(&native_engine_ctx, kPrompt);

    // prefill 延迟：各跑 5 次取平均（含一次暖机）。只为量级参考，不做统计严谨性主张。
    const std::vector<int64_t> prompt_for_measure = kPrompt;
    const auto measure = [&prompt_for_measure](Engine* engine) {
        RunEngine(engine, prompt_for_measure);  // 暖机
        const auto begin = std::chrono::steady_clock::now();
        for (int32_t i = 0; i < 5; ++i) {
            RunEngine(engine, prompt_for_measure);
        }
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - begin).count() / 5.0;
    };
    const double onnx_ms = measure(&onnx_engine_ctx);
    const double native_ms = measure(&native_engine_ctx);
    ASSERT_EQ(onnx_logits.size(), native_logits.size());
    ASSERT_FALSE(onnx_logits.empty());

    const std::vector<char> ref_bytes = ReadFile(ref_path);
    ASSERT_EQ(ref_bytes.size(), static_cast<size_t>(kSeq) * kVocab * sizeof(float));
    const auto* ref = reinterpret_cast<const float*>(ref_bytes.data());
    const std::vector<float> reference(ref, ref + kSeq * kVocab);

    const DiffStats onnx_vs_native = ComputeDiffStats(native_logits, onnx_logits);
    const DiffStats onnx_vs_ref = ComputeDiffStats(reference, onnx_logits);
    const DiffStats native_vs_ref = ComputeDiffStats(reference, native_logits);
    float max_ref = 0.0f;
    for (float value : reference) {
        max_ref = std::max(max_ref, std::fabs(value));
    }
    std::vector<int32_t> onnx_argmax;
    std::vector<int32_t> native_argmax;
    std::vector<int32_t> ref_argmax;
    ArgmaxAll(onnx_logits, &onnx_argmax);
    ArgmaxAll(native_logits, &native_argmax);
    ArgmaxAll(reference, &ref_argmax);

    std::cout << "[诊断] P3-4 性能：构建 ONNX " << onnx_build_s << " s vs 原生 "
              << native_build_s << " s（0 表示复用了已有引擎，未测到）；"
              << "prefill(4 token) ONNX " << onnx_ms << " ms vs 原生 " << native_ms
              << " ms\n";
    std::cout << "[诊断] Phase 3 ONNX vs 原生 vs HF 参考（max_abs / max_abs÷max|ref| / cosine）\n"
              << "        ONNX  vs 原生 : " << onnx_vs_native.max_abs << " / "
              << onnx_vs_native.max_abs / max_ref << " / " << Cosine(native_logits, onnx_logits)
              << "\n"
              << "        ONNX  vs HF   : " << onnx_vs_ref.max_abs << " / "
              << onnx_vs_ref.max_abs / max_ref << " / " << Cosine(reference, onnx_logits)
              << "\n"
              << "        原生  vs HF   : " << native_vs_ref.max_abs << " / "
              << native_vs_ref.max_abs / max_ref << " / "
              << Cosine(reference, native_logits) << "\n";

    // D2/D1 冻结口径：与**原生构建**逐项对齐；这条比"与 HF 对齐"更严——
    // 两条路用的是同一份数值，没有理由容忍更大的差异。
    EXPECT_GT(Cosine(native_logits, onnx_logits), 0.999999);
    EXPECT_LT(onnx_vs_native.max_abs / max_ref, 1e-5f);
    EXPECT_EQ(native_argmax, onnx_argmax) << "两条路的 argmax 不一致";
    // 与 HF 参考的三方对照（口径同 Phase 2 的 P2-3）
    EXPECT_GT(Cosine(reference, onnx_logits), 0.999999);
    EXPECT_LT(onnx_vs_ref.max_abs / max_ref, 1e-5f);
    EXPECT_EQ(ref_argmax, onnx_argmax);
}


// ---------------------------------------------------------------------------
// G3：FP16 覆盖
// ---------------------------------------------------------------------------
//
// 方案 A 的 FP16 路径已在 Phase 1.5 覆盖，方案 B（ONNX）此前**只测过 FP32**。
// 阈值口径用 D6 的 FP16 档（`cosine ≥ 0.999`、相对界 `< 5e-3`）——这是首次运行前的
// 冻结口径；用例把实测值打印出来，供后续按"实测收敛"收紧（收紧无需额外证据，放宽要按 §7）。
//
// 注意 ONNX 与原生在 FP16 下**各自的 Cast/精度处理不同**（见 TROUBLESHOOTING #17），
// 因此这里比的是"两条路在 FP16 下是否一致"，而不是"与 HF 是否一致"（HF 参考是 FP32）。
TEST(Gpt2OnnxTest, Fp16PathsAgree) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string onnx = FindOnnx();
    if (dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/gpt2 与 1_gpt2_onnx/gpt2.onnx";
    }

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP16;
    config.min_prefill_seq_len = kSeq;
    config.opt_prefill_seq_len = kSeq;
    config.max_prefill_seq_len = kSeq;
    EngineBuilder builder(logger, config);

    const std::string onnx_engine = "/tmp/mini_trt_llm_gpt2_onnx_fp16.engine";
    const std::string native_engine = "/tmp/mini_trt_llm_gpt2_accuracy_fp16.engine";
    ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, onnx_engine, {}))
        << "FP16 ONNX 引擎构建失败";
    ASSERT_TRUE(builder.BuildFromConfig(dir, native_engine, BuildStage::kSingle))
        << "FP16 原生引擎构建失败";
    Engine onnx_ctx(onnx_engine, logger);
    Engine native_ctx(native_engine, logger);

    const std::vector<float> onnx_logits = RunEngine(&onnx_ctx, kPrompt);
    const std::vector<float> native_logits = RunEngine(&native_ctx, kPrompt);
    ASSERT_EQ(onnx_logits.size(), native_logits.size());
    ASSERT_FALSE(onnx_logits.empty());

    const DiffStats diff = ComputeDiffStats(native_logits, onnx_logits);
    float max_ref = 0.0f;
    for (float value : native_logits) {
        max_ref = std::max(max_ref, std::fabs(value));
    }
    std::vector<int32_t> onnx_argmax;
    std::vector<int32_t> native_argmax;
    ArgmaxAll(onnx_logits, &onnx_argmax);
    ArgmaxAll(native_logits, &native_argmax);

    std::cout << "[诊断] G3 FP16：ONNX vs 原生 max_abs " << diff.max_abs << " / 相对 "
              << diff.max_abs / max_ref << " / cosine " << Cosine(native_logits, onnx_logits)
              << "\n";

    EXPECT_GT(Cosine(native_logits, onnx_logits), 0.999);
    EXPECT_LT(diff.max_abs / max_ref, 5e-3f);
    // 语义判据：两条路的逐位置 argmax 必须一致（数值差可以容忍，token 不能变）
    EXPECT_EQ(native_argmax, onnx_argmax) << "FP16 下两条路的 argmax 不一致";
}

// ---------------------------------------------------------------------------
// G4：shape 覆盖（profile 的 opt / max）
// ---------------------------------------------------------------------------
//
// 此前只用 `[1,4]` 验过，而 profile 范围是 batch 1..4、seq 1..512（opt 为 64）。
// 只测 4 个 token 等于没覆盖"TRT 真正优化的那个形状"，也没测边界。
// 本用例建一组宽 profile 引擎（min 1 / opt 64 / max 512），对三个形状两两对照。
// 此形状下没有 HF 参考（`ref_output.bin` 只有 seq=4），因此比的是两条路彼此——
// 它们用的是同一份数值（TROUBLESHOOTING #17 已核对），任何差异都是实现差异。
TEST(Gpt2OnnxTest, MatchesAcrossProfileShapes) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string dir = FindModelDir();
    const std::string onnx = FindOnnx();
    if (dir.empty() || onnx.empty()) {
        GTEST_SKIP() << "需要 models/gpt2 与 1_gpt2_onnx/gpt2.onnx";
    }

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    config.min_prefill_seq_len = 1;
    config.opt_prefill_seq_len = 64;
    config.max_prefill_seq_len = 512;
    EngineBuilder builder(logger, config);

    const std::string onnx_engine = "/tmp/mini_trt_llm_gpt2_onnx_wide.engine";
    const std::string native_engine = "/tmp/mini_trt_llm_gpt2_accuracy_wide.engine";
    ASSERT_TRUE(builder.BuildFromOnnx(dir, onnx, onnx_engine, {}));
    ASSERT_TRUE(builder.BuildFromConfig(dir, native_engine, BuildStage::kSingle));
    Engine onnx_ctx(onnx_engine, logger);
    Engine native_ctx(native_engine, logger);

    // (batch, seq) 组合：覆盖 profile 的 opt(64)、max(512) 与 batch 维（G4b）。
    // batch=2 只配小 seq：`[2,512,50257]` 的 logits 是 206MB/引擎，加上主机端副本
    // 会接近 WSL2 的可控范围，收益不抵风险。
    const std::pair<int32_t, int32_t> shapes[] = {{1, 1}, {1, 64}, {1, 512}, {2, 4}, {2, 64}};
    for (const auto& [batch, seq] : shapes) {
        // 造一个确定性的"prompt"：不追求语义，只要能区分位置（token id 必须在 [0, vocab)）
        std::vector<int64_t> tokens(static_cast<size_t>(batch) * static_cast<size_t>(seq));
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = (static_cast<int64_t>(i) * 7 + 3) % kVocab;
        }
        const std::vector<float> onnx_logits = RunEngine(&onnx_ctx, tokens, batch);
        const std::vector<float> native_logits = RunEngine(&native_ctx, tokens, batch);
        const size_t expected = tokens.size() * static_cast<size_t>(kVocab);
        ASSERT_EQ(onnx_logits.size(), expected) << "batch=" << batch << " seq=" << seq;
        ASSERT_EQ(onnx_logits.size(), native_logits.size()) << "batch=" << batch
                                                           << " seq=" << seq;

        const DiffStats diff = ComputeDiffStats(native_logits, onnx_logits);
        float max_ref = 0.0f;
        for (float value : native_logits) {
            max_ref = std::max(max_ref, std::fabs(value));
        }
        std::cout << "[诊断] G4 batch=" << batch << " seq=" << seq << "：ONNX vs 原生 max_abs "
                  << diff.max_abs << " / 相对 " << diff.max_abs / max_ref << " / cosine "
                  << Cosine(native_logits, onnx_logits) << "\n";
        EXPECT_GT(Cosine(native_logits, onnx_logits), 0.999999)
            << "batch=" << batch << " seq=" << seq;
        EXPECT_LT(diff.max_abs / max_ref, 1e-5f) << "batch=" << batch << " seq=" << seq;

        // 逐行 argmax：按"可判性"判（方案 B，判据与推导见 gpt2_test_support.hpp 的
        // CompareArgmaxByDecidability 注释与 TROUBLESHOOTING.md #34.9）。
        // 采样只看最后一行，但非最后一行错也说明中间层有问题——所以**可判行仍然逐行严格比对**。
        const ArgmaxAgreement agreement = CompareArgmaxByDecidability(
            onnx_logits.data(), native_logits.data(), batch * seq, kVocab);

        std::cout << "[诊断] G4 batch=" << batch << " seq=" << seq
                  << "：argmax 可判行 " << (agreement.rows - agreement.undecidable_rows)
                  << " / " << agreement.rows
                  << "；不可判行（余量 ≤ 2×两侧差异）= " << agreement.undecidable_rows;
        if (!agreement.undecidable_row_indices.empty()) {
            std::cout << " → 行 ";
            for (size_t k = 0; k < agreement.undecidable_row_indices.size(); ++k) {
                if (k != 0) std::cout << ",";
                std::cout << agreement.undecidable_row_indices[k];
            }
        }
        std::cout << "\n";

        // 不可判行里的**具体数值**要打出来：下个会话要能判"这一行是不是同一个并列"。
        // 精度必须够——|logit|≈87 处的并列间距只有 1e-5 量级，默认 6 位有效数字会把
        // 两个不同的数打印成同一个值（见 TROUBLESHOOTING.md #34.7）。
        for (const int32_t row : agreement.undecidable_row_indices) {
            const float* a = onnx_logits.data() + static_cast<size_t>(row) * kVocab;
            const float* b = native_logits.data() + static_cast<size_t>(row) * kVocab;
            int32_t ia = 0;
            int32_t ib = 0;
            float row_max_abs_diff = 0.0f;
            for (int32_t c = 0; c < kVocab; ++c) {
                if (a[c] > a[ia]) ia = c;
                if (b[c] > b[ib]) ib = c;
                row_max_abs_diff = std::max(row_max_abs_diff, std::fabs(a[c] - b[c]));
            }
            float second = -1e30f;
            for (int32_t c = 0; c < kVocab; ++c) {
                if (c == ib) continue;
                second = std::max(second, b[c]);
            }
            std::cout << std::setprecision(9) << "        不可判行 " << row
                      << "：onnx_argmax=" << ia << " native_argmax=" << ib
                      << " | native 余量=" << (b[ib] - second)
                      << " | 该行两侧最大差=" << row_max_abs_diff
                      << " | onnx[ia]=" << a[ia] << " onnx[ib]=" << a[ib]
                      << " | native[ib]=" << b[ib] << " native[ia]=" << b[ia] << "\n";
        }

        // 判据 1（严格）：**可判行**必须 argmax 相等。不等就是缺陷信号。
        EXPECT_EQ(agreement.violations, 0)
            << "batch=" << batch << " seq=" << seq
            << "：可判行上 argmax 不等（native 余量 > 2×两侧差异，仍指到不同类别）→ 真问题，继续查，"
               "不许把上界调大或直接放宽判据";
        // 判据 2（护栏）：不可判行**必须是实测登记过的那几行**（钉到行号）。
        // 新增行 = 红：要么图变了、要么真坏了，两种都要人看一眼；改表必须有实测依据。
        const ShapeUndecidableRows* expected_rows = FindExpectedUndecidableRows(batch, seq);
        ASSERT_NE(expected_rows, nullptr)
            << "形状 (" << batch << "," << seq << ") 没有登记预期不可判行——"
               "新增形状时必须先在真机测出它的并列行号并登记（见 ExpectedUndecidableRows 注释）";
        EXPECT_EQ(agreement.undecidable_row_indices, expected_rows->rows)
            << "batch=" << batch << " seq=" << seq
            << "：不可判行与实测登记不符（新增/缺失都算）→ 先查原因；"
               "若要改登记表，必须给出真机实测依据并写入 TROUBLESHOOTING.md #34.9，"
               "不许为了让用例变绿而加行";
    }
}

}  // namespace mini_trt_llm
