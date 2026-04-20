#include "builder.hpp"
#include "infer.hpp"
#include <NvInferPlugin.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>
#include <array>
#include <sstream>
#include <algorithm>

// ── 精度对比工具 ──────────────────────────────────────────────────────────
static float cosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    return (float)(dot / (std::sqrt(na * nb) + 1e-12));
}

static float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

/**
 * 调用 GPT-2 tokenizer 将文本转为 token ids
 * 通过 popen 执行 Python 命令，解析输出
 *
 * @param text  输入文本
 * @return      token ids（int64_t vector）
 */
static std::vector<int64_t> tokenize(const std::string& text) {
    // 构造 Python 单行命令
    std::string cmd =
        "python3 -c \""
        "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; "
        "from transformers import GPT2Tokenizer; "
        "t = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True); "
        "ids = t('" + text + "')['input_ids']; "
        "print(' '.join(map(str, ids)))"
        "\" 2>/dev/null";

    // popen：执行命令并读取 stdout
    std::array<char, 512> buf;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("[tokenize] popen failed");
    while (fgets(buf.data(), buf.size(), pipe))
        result += buf.data();
    pclose(pipe);

    // 去掉末尾换行
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();

    if (result.empty())
        throw std::runtime_error("[tokenize] No output from tokenizer");

    // 解析空格分隔的 id 列表
    std::vector<int64_t> ids;
    std::istringstream ss(result);
    std::string token;
    while (ss >> token)
        ids.push_back(std::stoll(token));

    return ids;
}

static std::string decodeNextToken(const std::vector<float>& logits,
                                    int seqLen, int vocabSize) {
    // 只取最后一个位置的 logits，预测下一个 token
    const float* last_pos = logits.data() + (seqLen - 1) * vocabSize;
    int next_id = static_cast<int>(
        std::max_element(last_pos, last_pos + vocabSize) - last_pos);

    std::string cmd =
        "python3 -c \""
        "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; "
        "from transformers import GPT2Tokenizer; "
        "t = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True); "
        "print(t.decode([" + std::to_string(next_id) + "]))"
        "\" 2>/dev/null";

    std::array<char, 512> buf;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "[decode failed]";
    while (fgets(buf.data(), buf.size(), pipe))
        result += buf.data();
    pclose(pipe);
    while (!result.empty() &&
           (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

int main() {
    Logger logger;
    initLibNvInferPlugins(&logger, "");

    const std::string onnxPath = std::string(PROJECT_SOURCE_DIR) + "/gpt2.onnx";

    // ── Step 1: 构建 FP32 / FP16 Engine ──────────────────────────────────
    struct Task { std::string engine; Precision prec; };
    std::vector<Task> tasks = {
        {"gpt2_fp32.engine", Precision::FP32},
        {"gpt2_fp16.engine", Precision::FP16},
    };

    for (auto& t : tasks) {
        std::cout << "\n========== Build: " << t.engine << " ==========\n";
        buildEngine(onnxPath, t.engine, t.prec, logger);
    }

    // ── Step 2: 加载参考输出（PyTorch FP32 基准）─────────────────────────
    // // ref_output.bin：shape [1, 4, 50257]，对应输入 "The quick brown fox"
    const int REF_SEQ = 4;
    const int VOCAB   = 50257;
    std::vector<float> ref_out(1 * REF_SEQ * VOCAB);
    {
        std::ifstream fin(std::string(PROJECT_SOURCE_DIR) + "/ref_output.bin",
                          std::ios::binary);
        if (!fin) {
            std::cerr << "[ERROR] ref_output.bin not found\n";
            return 1;
        }
        fin.read(reinterpret_cast<char*>(ref_out.data()),
                 ref_out.size() * sizeof(float));
    }
    std::cout << "\n[Reference] Loaded PyTorch FP32 output, "
              << "seq_len=" << REF_SEQ << " hidden=" << VOCAB << "\n";

    // ── Step 3: 定义输入文本和 token ids ─────────────────────────────────────
    //
    const std::string text = "The quick brown fox";
    auto input_ids = tokenize(text);
    const int BATCH = 1;
    const int SEQ   = static_cast<int>(input_ids.size());

    std::cout << "[Input] text=\""  << text << "\"\n"
            << "[Input] seq_len=" << SEQ  << "  token_ids: ";
    for (auto id : input_ids) std::cout << id << " ";
    std::cout << "\n";

    std::vector<float> fp32_out;

    for (auto& t : tasks) {
        std::cout << "\n========== Infer: " << t.engine << " ==========\n";
        InferSession sess(t.engine, logger);

        auto out = sess.infer(input_ids, BATCH, SEQ);

        // 解码输出文本
        std::string next_token = decodeNextToken(out, SEQ, 50257);
        std::cout << "[Output] input=\""      << text        << "\"\n"
                << "[Output] next_token=\"" << next_token  << "\"\n";

        if (t.prec == Precision::FP32) {
            fp32_out = out;
            float cs = cosineSim(ref_out, out);
            float d  = maxAbsDiff(ref_out, out);
            std::cout << "[Accuracy vs PyTorch FP32]"
                      << "  cosine_sim="   << cs
                      << "  max_abs_diff=" << d << "\n";
        } else {
            float cs   = cosineSim(fp32_out, out);
            float d    = maxAbsDiff(fp32_out, out);
            std::cout << "[Accuracy vs TRT FP32]"
                      << "  cosine_sim="   << cs
                      << "  max_abs_diff=" << d << "\n";
        }

        // ── Step 4: Multi seq_len Benchmark ──────────────────────────────
        std::vector<int> seq_lens = {16, 64, 128, 256, 512};

        std::cout << "\n"
                  << std::left
                  << std::setw(10) << "Precision"
                  << std::setw(10) << "SeqLen"
                  << std::setw(12) << "mean(ms)"
                  << std::setw(12) << "p50(ms)"
                  << std::setw(12) << "p99(ms)"
                  << std::setw(16) << "throughput(tok/s)"
                  << "\n"
                  << std::string(72, '-') << "\n";

        for (int sl : seq_lens) {
            auto r = sess.benchmark(BATCH, sl, 20, 100);
            std::cout << std::left
                      << std::setw(10) << precisionStr(t.prec)
                      << std::setw(10) << sl
                      << std::setw(12) << r.mean_ms
                      << std::setw(12) << r.p50_ms
                      << std::setw(12) << r.p99_ms
                      << std::setw(16) << r.throughput
                      << "\n";
        }
    }

    std::cout << "\n========== Done ==========\n";
    return 0;
}