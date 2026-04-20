#include "infer.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <nvtx3/nvToolsExt.h>

static std::vector<char> readFile(const std::string& path) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("[Infer] Cannot open: " + path);
    size_t sz = fin.tellg();
    fin.seekg(0);
    std::vector<char> buf(sz);
    fin.read(buf.data(), sz);
    return buf;
}

// ── 构造 ──────────────────────────────────────────────────────────────────
InferSession::InferSession(const std::string& enginePath,
                           Logger& logger,
                           int maxBatch,
                           int maxSeqLen)
    : m_logger(logger), m_max_batch(maxBatch), m_max_seq_len(maxSeqLen)
{
    auto buf = readFile(enginePath);

    m_runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!m_runtime) throw std::runtime_error("[Infer] createInferRuntime failed");

    m_engine.reset(m_runtime->deserializeCudaEngine(buf.data(), buf.size()));
    if (!m_engine) throw std::runtime_error("[Infer] deserializeCudaEngine failed");

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) throw std::runtime_error("[Infer] createExecutionContext failed");

    cudaStreamCreate(&m_stream);
    allocBuffers();

    m_logger.log(nvinfer1::ILogger::Severity::kVERBOSE,
                 ("[Infer] Session ready: " + enginePath).c_str());
}

// ── 析构 ──────────────────────────────────────────────────────────────────
InferSession::~InferSession() {
    if (m_device_input)  cudaFree(m_device_input);
    if (m_device_output) cudaFree(m_device_output);
    if (m_pinned_input)  cudaFreeHost(m_pinned_input);
    if (m_pinned_output) cudaFreeHost(m_pinned_output);
    if (m_stream)        cudaStreamDestroy(m_stream);
}

// ── 分配缓冲区 ────────────────────────────────────────────────────────────
void InferSession::allocBuffers() {
    // 输入：INT64，[maxBatch, maxSeqLen]
    size_t input_bytes  = m_max_batch * m_max_seq_len * sizeof(int64_t);
    // 输出：FP32，[maxBatch, maxSeqLen, k_VOCAB]
    // size_t output_bytes = m_max_batch * m_max_seq_len * k_HIDDEN * sizeof(float);
    size_t output_bytes = m_max_batch * m_max_seq_len * k_VOCAB * sizeof(float);

    cudaMalloc(&m_device_input,  input_bytes);
    cudaMalloc(&m_device_output, output_bytes);
    cudaMallocHost(&m_pinned_input,  input_bytes);
    cudaMallocHost(&m_pinned_output, output_bytes);

    m_logger.log(nvinfer1::ILogger::Severity::kVERBOSE,
                 "[Infer] GPU buffers allocated");
}

// ── 单次推理 ──────────────────────────────────────────────────────────────
std::vector<float> InferSession::infer(const std::vector<int64_t>& inputIds,
                                       int batchSize, int seqLen) {
    size_t input_bytes  = batchSize * seqLen * sizeof(int64_t);
    size_t output_bytes = batchSize * seqLen * k_VOCAB * sizeof(float);

    // 1. 设置动态 shape：两个动态轴 batch 和 seq_len
    m_context->setInputShape("input_ids",
                             nvinfer1::Dims2{batchSize, seqLen});
    m_context->setTensorAddress("input_ids",          m_device_input);
    // m_context->setTensorAddress("last_hidden_state",  m_device_output); // 隐藏状态向量，不能转成文本
    m_context->setTensorAddress("logits", m_device_output);

    // 2. H2D
    memcpy(m_pinned_input, inputIds.data(), input_bytes);

    nvtxRangePushA("H2D");
    cudaMemcpyAsync(m_device_input, m_pinned_input, input_bytes,
                    cudaMemcpyHostToDevice, m_stream);
    nvtxRangePop();

    // 3. 推理
    nvtxRangePushA("Infer");
    if (!m_context->enqueueV3(m_stream))
        throw std::runtime_error("[Infer] enqueueV3 failed");
    nvtxRangePop();

    // 4. D2H
    nvtxRangePushA("D2H");
    cudaMemcpyAsync(m_pinned_output, m_device_output, output_bytes,
                    cudaMemcpyDeviceToHost, m_stream);
    nvtxRangePop();

    cudaStreamSynchronize(m_stream);

    return std::vector<float>(m_pinned_output,
                           m_pinned_output + batchSize * seqLen * k_VOCAB);
}

// ── Benchmark ─────────────────────────────────────────────────────────────
InferSession::BenchResult InferSession::benchmark(int batchSize, int seqLen,
                                                   int nWarmup, int nRun) {
    // 固定输入：全 1 的 token id（GPT-2 vocab size 50257，1 是合法 id）
    std::vector<int64_t> input(batchSize * seqLen, 1);

    m_context->setInputShape("input_ids", nvinfer1::Dims2{batchSize, seqLen});
    m_context->setTensorAddress("input_ids",         m_device_input);
    // m_context->setTensorAddress("last_hidden_state", m_device_output);
    m_context->setTensorAddress("logits", m_device_output);

    size_t input_bytes = batchSize * seqLen * sizeof(int64_t);
    memcpy(m_pinned_input, input.data(), input_bytes);
    cudaMemcpyAsync(m_device_input, m_pinned_input, input_bytes,
                    cudaMemcpyHostToDevice, m_stream);
    cudaStreamSynchronize(m_stream);

    // Warmup
    for (int i = 0; i < nWarmup; ++i) {
        m_context->enqueueV3(m_stream);
        cudaStreamSynchronize(m_stream);
    }

    // 计时
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::vector<float> latencies(nRun);
    for (int i = 0; i < nRun; ++i) {
        nvtxRangePushA("benchmark_iter");
        cudaEventRecord(ev_start, m_stream);
        m_context->enqueueV3(m_stream);
        cudaEventRecord(ev_stop, m_stream);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&latencies[i], ev_start, ev_stop);
        nvtxRangePop();
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    float mean = std::accumulate(latencies.begin(), latencies.end(), 0.f) / nRun;
    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    float p50 = sorted[nRun * 50 / 100];
    float p99 = sorted[nRun * 99 / 100];

    // throughput：每秒处理的 token 数
    float throughput = (batchSize * seqLen) / (mean / 1000.f);

    return BenchResult{mean, p50, p99, throughput};
}