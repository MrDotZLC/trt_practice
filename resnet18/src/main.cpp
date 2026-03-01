#include "builder.hpp"
#include "infer.hpp"
#include <NvInferPlugin.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// ── 精度对比工具函数 ──────────────────────────────────────────────────────────

/**
 * 余弦相似度（Cosine Similarity）
 *
 * 公式：cos(θ) = (A · B) / (||A|| * ||B||)
 *
 * 含义：
 *   衡量两个向量的方向差异，与向量模长无关。
 *   范围 [-1, 1]，越接近 1 表示输出分布越一致。
 *   用于精度对比比 MSE 更能反映分类结果的一致性：
 *   即使绝对值有偏差，只要各类别的相对大小顺序一致，cos 仍接近 1。
 */
static float cosineSim(const std::vector<float>& a,
                       const std::vector<float>& b) {
    double dot = 0, norm_A = 0, norm_B = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot    += static_cast<double>(a[i]) * b[i];
        norm_A += static_cast<double>(a[i]) * a[i];
        norm_B += static_cast<double>(b[i]) * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(norm_A) * std::sqrt(norm_B) + 1e-12));
}

/**
 * 最大绝对误差（Max Absolute Difference）
 *
 * 公式：max( |a[i] - b[i]| )
 *
 * 含义：
 *   找出所有输出元素中偏差最大的一个。
 *   反映量化引入的最坏情况误差。
 */
static float maxAbsDiff(const std::vector<float>& a,
                        const std::vector<float>& b) {
    float max_d = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_d = std::max(max_d, std::abs(a[i] - b[i]));
    }
    return max_d;
}

/**
 * 均方误差（Mean Squared Error）
 *
 * 公式：MSE = (1/N) * Σ (a[i] - b[i])²
 *
 * 含义：
 *   衡量整体误差的平均水平，对大误差敏感（平方放大）。
 */
static float mse(const std::vector<float>& a,
                 const std::vector<float>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(sum / a.size());
}

// ── 最大值索引 ────────────────────────────────────────────────────────────────
static int argmax(const std::vector<float>& logits, int offset, int numCls) {
    // 找 [offset, offset+numCls) 范围内最大值的下标
    int mx_idx = 0;
    float mx = logits[offset];
    for (int i = 1; i < numCls; ++i) {
        if (logits[offset + i] > mx) {
            mx = logits[offset + i];
            mx_idx = i;
        }
    }
    return mx_idx;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    Logger logger;

    // 初始化 TRT 内置插件库，必须在任何 Runtime 创建前调用
    initLibNvInferPlugins(&logger, "");

    const std::string onnxPath = std::string(PROJECT_SOURCE_DIR) + "/resnet18.onnx";
    const int BATCH = 8;
    const int C = 3, H = 224, W = 224;
    const int NUM_CLS = 1000;

    // ── Step 1: 构建三种精度 Engine ──────────────────────────────────────────
    struct Task
    {
        std::string engine;
        Precision   prec;
    };

    std::vector<Task> tasks = {
        {"resnet18_fp32.engine", Precision::FP32},
        {"resnet18_fp16.engine", Precision::FP16},
        {"resnet18_int8.engine", Precision::INT8},
    };

    for (auto& t : tasks)
    {
        std::cout << "\n========== Build: " << t.engine << " ==========\n";
        buildEngine(onnxPath, t.engine, t.prec, logger);
    }

    // ── Step 2: 准备统一输入数据 ─────────────────────────────────────────────
    // 使用相同输入保证精度对比公平，值域 [0,1] 模拟归一化后的图片数据
    std::vector<float> input(BATCH * C * H * W);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 255) / 255.f;
    }
    
    // ── Step 3: 推理 + 精度对比 + Benchmark ─────────────────────────────────
    std::vector<float> fp32_out;

    for (auto &t : tasks) {
        std::cout << "\n========== Infer: " << t.engine << " ==========\n";
        InferSession sess(t.engine, logger);

        // 精度验证推理
        auto out = sess.infer(input, BATCH);

        if (t.prec == Precision::FP32) {
            fp32_out = out;
            std::cout << "[Accuracy] FP32 baseline, argmax class (batch[0]): "
                      << argmax(fp32_out, 0, NUM_CLS) << "\n";
        } else {
            // 如果 fp32Out 为空，说明 FP32 推理结果没有正确保存
            if (fp32_out.empty())
            {
                std::cerr << "[ERROR] fp32Out is empty, FP32 must run first.\n";
                continue;
            }
            // 与 FP32 基准对比
            float cs   = cosineSim(fp32_out, out);
            float diff = maxAbsDiff(fp32_out, out);
            float err  = mse(fp32_out, out);

            std::cout << "[Accuracy vs FP32]"
                      << "  cosine_sim="   << cs
                      << "  max_abs_diff=" << diff
                      << "  mse="          << err << "\n";

            // 最大值索引 是否一致（对分类任务最直接的精度指标）
            bool allMatch = true;
            for (int b = 0; b < BATCH; ++b) {
                int fp32_argmax = argmax(fp32_out, b * NUM_CLS, NUM_CLS);
                int cur_argmax  = argmax(out,     b * NUM_CLS, NUM_CLS);
                if (fp32_argmax != cur_argmax) {
                    allMatch = false;
                    std::cout << "  [!] batch[" << b << "] argmax mismatch:"
                              << " FP32=" << fp32_argmax
                              << " " << precisionStr(t.prec) << "=" << cur_argmax << "\n";
                }
            }
            if (allMatch) {
                std::cout << "  Top-1 all match FP32 across "
                          << BATCH << " samples.\n";
            }
        }
        // 性能 Benchmark
        sess.benchmark(BATCH, 50, 200);
    }

    std::cout << "\n========== Done ==========\n";
    return 0;
}