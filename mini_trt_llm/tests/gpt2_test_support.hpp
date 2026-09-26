#pragma once

// GPT-2 用例的共享测试件：
//   1. 小型 GPT-2 夹具（结构与真实模型同构，维度缩到可随手造）；
//   2. 数值差异诊断——**实现已迁到 `diff_stats.hpp`**（Phase 4 的 CV 用例也要用，
//      放这里会让 ResNet18 的用例去包含一个 GPT-2 头文件）。本文件继续转发它，
//      既有 GPT-2 用例无需改动，且口径仍只有一份来源。
//
// **为什么要提成共享头**：权重配方（`sin(0.31*i + 1.0) * scale`）与超参数必须让所有
// 用例完全一致——一旦有两份副本，它们会各自演化，而"权重不一样"造成的失败会伪装成
// 实现 bug（见 docs/PROGRESS.md §2.13：参考实现必须唯一）。差异诊断同理：
// 各用例各写一份，阈值口径就会漂移（见 AGENTS.md §7）。本文件是它们的唯一来源。

#include "e2e_fixture.hpp"
#include "diff_stats.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/gpt2_model_builder.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// ---------------------------------------------------------------------------
// "跨实现比较 argmax" 的可判性判据（对应 TROUBLESHOOTING.md #34，方案 B）
//
// 背景：两条**独立实现**（例如 ONNX 图 vs 原生图）的 logits 只做到"有界接近"，
// 而 argmax 判等是**精确**判据。在"并列间距小于两侧差异"的行上，argmax 归谁
// 只由那点误差的**符号**决定——结果不携带"实现是否正确"的信息，且会随构建态翻绿翻红。
//
// 判据（**可证**的形式，不是拍脑袋的容差）：
//   记 m = native 该行 top1 − top2，d = 该行两侧逐元素最大绝对差。
//   若存在满足 |δ_c| ≤ d 的扰动把 argmax 翻过去，则必有 m < 2d
//   （因为 a[ia] − a[ib] = −m + (δ_ia − δ_ib)，而 |δ_ia − δ_ib| ≤ 2d）。
//   所以 **m > 2d 是"该行 argmax 可判"的充分条件**：
//   · 可判行（m > 2d）→ argmax **必须相等**；
//   · 不可判行（m ≤ 2d）→ 允许不同，但**必须计数与打印**，并受上界约束
//     （上界用来把"真回归"变成红：真出错会让不可判行数变大，或让可判行出现不等）。
//
// 为什么阈值用 2d 而不是 d：2d 是能给出**证明**的那一档；用 d 只是经验值。
// 判据强度在有信息的地方没有下降——所有可判行仍要求逐行全等。
// ---------------------------------------------------------------------------
struct ArgmaxAgreement {
    int32_t rows = 0;              // 参与比较的行数
    int32_t undecidable_rows = 0;  // m ≤ 2d：并列低于两侧差异，不可判
    int32_t violations = 0;        // 可判行却 argmax 不等 —— 这才是缺陷信号
    std::vector<int32_t> undecidable_row_indices;
    std::vector<int32_t> violating_row_indices;
};

// 逐行比较两条实现的 argmax，按上面的可判性判据分类。
// 两个 logits 都是行主序 [rows, vocab]，vocab 为每行元素数。
inline ArgmaxAgreement CompareArgmaxByDecidability(const float* onnx_logits,
                                                   const float* native_logits,
                                                   int32_t rows, int32_t vocab) {
    ArgmaxAgreement result;
    result.rows = rows;
    for (int32_t row = 0; row < rows; ++row) {
        const float* a = onnx_logits + static_cast<size_t>(row) * vocab;
        const float* b = native_logits + static_cast<size_t>(row) * vocab;

        int32_t onnx_argmax = 0;
        int32_t native_argmax = 0;
        float row_max_abs_diff = 0.0f;
        for (int32_t c = 0; c < vocab; ++c) {
            if (a[c] > a[onnx_argmax]) onnx_argmax = c;
            if (b[c] > b[native_argmax]) native_argmax = c;
            row_max_abs_diff = std::max(row_max_abs_diff, std::fabs(a[c] - b[c]));
        }

        // native 该行的并列间距：top1 与 top2 之差（top2 是除 top1 之外的最大值）。
        float second = -1e30f;
        for (int32_t c = 0; c < vocab; ++c) {
            if (c == native_argmax) continue;
            second = std::max(second, b[c]);
        }
        const float margin = b[native_argmax] - second;

        if (margin > 2.0f * row_max_abs_diff) {
            if (onnx_argmax != native_argmax) {
                ++result.violations;
                result.violating_row_indices.push_back(row);
            }
        } else {
            ++result.undecidable_rows;
            result.undecidable_row_indices.push_back(row);
        }
    }
    return result;
}

// 用小型同构模型做"两条路径互相对拍"：快、可控，且不依赖外部基线。
// 真实 GPT-2 上的同样对比放到 P2-3 / P2-8 的精度用例里。
inline constexpr int32_t kLayers = 2;
inline constexpr int32_t kHeads = 2;
inline constexpr int32_t kHidden = 8;
inline constexpr int32_t kPositions = 16;
inline constexpr int32_t kVocab = 32;
inline constexpr int32_t kBlockSize = 4;
inline constexpr int32_t kSmallGpt2PromptTokens = 4;
inline constexpr int32_t kHeadSize = kHidden / kHeads;
inline constexpr int32_t kBlocksPerSeq = kPositions / kBlockSize;  // = 4

// 容差**由实测的模型敏感性决定，而不是由"能不能过"决定**。
//
// 实测（numpy 复刻同一份合成权重）：
//   * 一次性 softmax 与 online softmax 在 float32 下算同一个式子，
//     logits 最大绝对差 6.0e-08；
//   * 权重相对扰动 1e-7 → logits 最大绝对变化 1.13e-07，
//     即这个模型对扰动的放大倍数约等于 1（不是病态模型）。
//
// 因此两条路径在 FP32 下的差异应当落在 1e-7 量级；这里给到 1e-5，
// 已经留了约 100 倍余量来吸收"不同 kernel / 不同形状"带来的实现差异，
// 同时比真机上观测到的 1e-3 严 100 倍——**观测到 1e-3 说明有真实缺陷，
// 不能靠放宽阈值把它盖过去**。
constexpr float kSingleKeyTol = 1e-5f;  // 只有一个 key：softmax 权重恒为 1
constexpr float kMultiKeyTol = 1e-5f;   // 多个 key：见上面的实测依据

inline std::string SmallGpt2ConfigJson() {
    return R"({
        "model_type": "gpt2",
        "architecture": "decoder_only",
        "hyper_params": {
            "n_layer": 2, "n_head": 2, "n_embd": 8, "n_positions": 16,
            "vocab_size": 32, "layer_norm_epsilon": 1e-05,
            "tie_word_embeddings": true, "block_size": 4
        },
        "weight_map": {}
    })";
}

// 权重按固定配方生成：数值本身不重要，但必须能区分不同位置/不同 head，
// 否则"位置写错""head 写错"这类缺陷会看不出来。
inline std::map<std::string, TensorSpec> SmallGpt2Weights() {
    std::map<std::string, TensorSpec> tensors;
    const auto add = [&tensors](const std::string& name, std::vector<size_t> shape,
                                float scale) {
        size_t count = 1;
        for (size_t dim : shape) {
            count *= dim;
        }
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            // 用确定性的伪随机填充，避免全同值导致"位置写错也看不出来"
            values[i] = scale * std::sin(0.31f * static_cast<float>(i) + 1.0f);
        }
        tensors[name] = TensorSpec{
            std::move(shape), TensorSpec::Dtype::kF32, std::move(values)};
    };

    const size_t hidden = kHidden;
    add("wte.weight", {static_cast<size_t>(kVocab), hidden}, 0.2f);
    add("wpe.weight", {static_cast<size_t>(kPositions), hidden}, 0.2f);
    for (int32_t i = 0; i < kLayers; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        // LayerNorm 的 scale 必须非零，否则整层被归一化成 0，测试会失去判别力
        add(p + "ln_1.weight", {hidden}, 1.0f);
        add(p + "ln_1.bias", {hidden}, 0.05f);
        add(p + "attn.c_attn.weight", {hidden, 3 * hidden}, 0.2f);
        add(p + "attn.c_attn.bias", {3 * hidden}, 0.05f);
        add(p + "attn.c_proj.weight", {hidden, hidden}, 0.2f);
        add(p + "attn.c_proj.bias", {hidden}, 0.05f);
        add(p + "ln_2.weight", {hidden}, 1.0f);
        add(p + "ln_2.bias", {hidden}, 0.05f);
        add(p + "mlp.c_fc.weight", {hidden, 4 * hidden}, 0.2f);
        add(p + "mlp.c_fc.bias", {4 * hidden}, 0.05f);
        add(p + "mlp.c_proj.weight", {4 * hidden, hidden}, 0.2f);
        add(p + "mlp.c_proj.bias", {hidden}, 0.05f);
    }
    add("ln_f.weight", {hidden}, 1.0f);
    add("ln_f.bias", {hidden}, 0.05f);
    return tensors;
}

inline EngineBuilder::Config SmallGpt2BuilderConfig() {
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    config.min_prefill_batch = 1;
    config.opt_prefill_batch = 1;
    config.max_prefill_batch = 1;
    config.min_prefill_seq_len = 1;
    config.opt_prefill_seq_len = kSmallGpt2PromptTokens;
    // 上限必须覆盖"prompt + 生成长度"：无 cache 的参考路径每步都要把整段重新喂进去。
    // 用 n_positions 当上界，正好也是引擎 profile 允许的最大值。
    config.max_prefill_seq_len = kPositions;
    config.min_decode_batch = 1;
    config.opt_decode_batch = 1;
    config.max_decode_batch = 1;
    return config;
}

}  // namespace test_support
}  // namespace mini_trt_llm
