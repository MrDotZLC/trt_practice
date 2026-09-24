#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// 确定性伪随机输入：用 sin 生成，避免 rand() 带来的跨环境差异，
// 否则 CPU 参考实现与 GPU kernel 之间无法做可复现的对比。
inline float DeterministicValue(int64_t index) {
    return std::sin(0.37f * static_cast<float>(index)) * 1.5f;
}

// 前置声明：CpuRmsNorm 只是浮点入参的薄封装，唯一实现在下面。
// 之所以这样组织而不是各写一份：同一算子的两份参考实现必然漂移，
// 这一点在 RoPE 上已经踩过一次（见 docs/TROUBLESHOOTING.md #9）。
inline void ReferenceRmsNorm(const std::vector<double>& input,
                             const std::vector<double>& weight, int32_t rows,
                             int32_t hidden, double eps, std::vector<double>* output);

// 浮点入参的 RMSNorm 参考封装，委托给 ReferenceRmsNorm。
inline void CpuRmsNorm(const std::vector<float>& input, const std::vector<float>& weight,
                       int64_t rows, int32_t hidden, float eps,
                       std::vector<float>* output) {
    const std::vector<double> input_d(input.begin(), input.end());
    const std::vector<double> weight_d(weight.begin(), weight.end());
    std::vector<double> output_d;
    ReferenceRmsNorm(input_d, weight_d, static_cast<int32_t>(rows), hidden,
                     static_cast<double>(eps), &output_d);
    output->assign(output_d.begin(), output_d.end());
}

// 精度判定沿用 Phase 1 已确认标准：相对误差阈值 + 小值时的绝对误差 Guardrail。
// 参考值接近 0 时相对误差会被放大，必须改判绝对误差，否则会误报。
inline bool WithinTolerance(float reference, float actual, float rel_tol, float abs_tol) {
    const float diff = std::fabs(reference - actual);
    if (std::fabs(reference) < 1e-4f) {
        return diff < abs_tol;
    }
    return diff / std::fabs(reference) < rel_tol;
}

// -----------------------------------------------------------------------------
// 算子级参考实现（双精度）。
//
// 这里的函数是"参考实现的唯一来源"，因此每个入口都带**前置断言**：
// 参数形状不满足约定时直接抛异常，而不是静默按错误的方式读数据。
//
// 为什么强调这一点：曾经有一版 ReferenceRoPE 没有 batch 维度，调用方传入
// batch*seq_len 个位置却只用了前 seq_len 个——编译不报错、运行不报错，
// 直到真机数值对不上才暴露，白白浪费一次真机往返（见 docs/TROUBLESHOOTING.md #9）。
// -----------------------------------------------------------------------------

inline void ReferenceRmsNorm(const std::vector<double>& input,
                             const std::vector<double>& weight, int32_t rows,
                             int32_t hidden, double eps, std::vector<double>* output) {
    if (input.size() != static_cast<size_t>(rows) * hidden ||
        weight.size() != static_cast<size_t>(hidden)) {
        throw std::invalid_argument("ReferenceRmsNorm: shape mismatch");
    }
    output->assign(input.size(), 0.0);
    for (int32_t r = 0; r < rows; ++r) {
        const size_t base = static_cast<size_t>(r) * hidden;
        double sum_sq = 0.0;
        for (int32_t i = 0; i < hidden; ++i) {
            sum_sq += input[base + i] * input[base + i];
        }
        const double inv_rms = 1.0 / std::sqrt(sum_sq / hidden + eps);
        for (int32_t i = 0; i < hidden; ++i) {
            (*output)[base + i] = input[base + i] * inv_rms * weight[i];
        }
    }
}

// RoPE（half-split，与 HuggingFace 一致，已由 scripts/ref_rope.py 交叉验证）。
// input: [batch, heads, seq_len, head_size]；positions: [batch, seq_len]
inline void ReferenceRoPE(const std::vector<double>& input,
                          const std::vector<int32_t>& positions, int32_t batch_size,
                          int32_t heads, int32_t seq_len, int32_t head_size, int32_t rotary_dim,
                          double base, std::vector<double>* output) {
    if (positions.size() != static_cast<size_t>(batch_size) * seq_len) {
        throw std::invalid_argument("ReferenceRoPE: positions must be [batch, seq_len]");
    }
    if (input.size() !=
        static_cast<size_t>(batch_size) * heads * seq_len * head_size) {
        throw std::invalid_argument("ReferenceRoPE: input must be [batch, heads, seq, head]");
    }
    if (rotary_dim <= 0 || rotary_dim > head_size || (rotary_dim % 2) != 0) {
        throw std::invalid_argument("ReferenceRoPE: rotary_dim must be even and <= head_size");
    }

    *output = input;
    const int32_t half = rotary_dim / 2;
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t h = 0; h < heads; ++h) {
            for (int32_t s = 0; s < seq_len; ++s) {
                const size_t row =
                    ((static_cast<size_t>(b) * heads + h) * seq_len + s) * head_size;
                const int32_t position = positions[static_cast<size_t>(b) * seq_len + s];
                for (int32_t j = 0; j < half; ++j) {
                    const double inv_freq =
                        std::pow(base, -2.0 * j / static_cast<double>(rotary_dim));
                    const double angle = position * inv_freq;
                    const double cos_v = std::cos(angle);
                    const double sin_v = std::sin(angle);
                    const double lo = input[row + j];
                    const double hi = input[row + j + half];
                    (*output)[row + j] = lo * cos_v - hi * sin_v;
                    (*output)[row + j + half] = hi * cos_v + lo * sin_v;
                }
            }
        }
    }
}

// Decoding 阶段的分页注意力，q: [heads, head_size]（query 序列长度为 1）
inline void ReferencePagedAttentionDecode(
    const std::vector<double>& query, const std::vector<double>& key_cache,
    const std::vector<double>& value_cache, const std::vector<int32_t>& block_table,
    int32_t context_len, int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    int32_t block_size, double scale, std::vector<double>* output) {
    if (query.size() != static_cast<size_t>(num_heads) * head_size) {
        throw std::invalid_argument("ReferencePagedAttentionDecode: query shape mismatch");
    }
    if (key_cache.size() != value_cache.size()) {
        throw std::invalid_argument("ReferencePagedAttentionDecode: K/V cache size mismatch");
    }
    if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
        throw std::invalid_argument("ReferencePagedAttentionDecode: invalid head counts");
    }

    output->assign(static_cast<size_t>(num_heads) * head_size, 0.0);
    const int32_t group = num_heads / num_kv_heads;

    for (int32_t h = 0; h < num_heads; ++h) {
        const int32_t kv_head = h / group;
        std::vector<double> scores(context_len, 0.0);
        double max_score = -1e300;
        for (int32_t t = 0; t < context_len; ++t) {
            const int32_t physical = block_table[static_cast<size_t>(t / block_size)];
            const int32_t slot = t % block_size;
            const size_t offset =
                ((static_cast<size_t>(physical) * block_size + slot) * num_kv_heads + kv_head) *
                head_size;
            double dot = 0.0;
            for (int32_t d = 0; d < head_size; ++d) {
                dot += query[static_cast<size_t>(h) * head_size + d] * key_cache[offset + d];
            }
            scores[t] = dot * scale;
            max_score = std::max(max_score, scores[t]);
        }

        double sum = 0.0;
        for (int32_t t = 0; t < context_len; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum += scores[t];
        }
        if (sum <= 0.0) {
            continue;
        }
        for (int32_t d = 0; d < head_size; ++d) {
            double acc = 0.0;
            for (int32_t t = 0; t < context_len; ++t) {
                const int32_t physical = block_table[static_cast<size_t>(t / block_size)];
                const int32_t slot = t % block_size;
                const size_t offset =
                    ((static_cast<size_t>(physical) * block_size + slot) * num_kv_heads +
                     kv_head) *
                    head_size;
                acc += scores[t] * value_cache[offset + d];
            }
            (*output)[static_cast<size_t>(h) * head_size + d] = acc / sum;
        }
    }
}

}  // namespace test_support
}  // namespace mini_trt_llm
