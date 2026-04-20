#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include "logger.hpp"

/**
 * GPT-2 TensorRT 推理会话
 *
 * 与 ResNet18 版本的关键差异：
 *   1. 输入 dtype：INT64（token id），不是 FP32
 *   2. 动态维度：batch 和 seq_len 均动态
 *   3. 输出 shape：[batch, seq_len, 50257]，随 seq_len 变化
 *   4. Pinned Memory：输入 int64，输出 float32
 *
 * 推理流程：
 *   setInputShape → setTensorAddress → H2D → enqueueV3 → D2H → Synchronize
 */
class InferSession {
public:
    struct BenchResult {
        float mean_ms;
        float p50_ms;
        float p99_ms;
        float throughput;   // tokens/s = batch * seq_len / mean_ms * 1000
    };

    /**
     * @param enginePath  .engine 文件路径
     * @param logger      TRT Logger
     * @param maxBatch    最大 batch size，须 ≤ 构建时 kMAX
     * @param maxSeqLen   最大序列长度，须 ≤ 构建时 kMAX
     */
    explicit InferSession(const std::string& enginePath,
                          Logger& logger,
                          int maxBatch  = 4,
                          int maxSeqLen = 512);
    ~InferSession();

    InferSession(const InferSession&)            = delete;
    InferSession& operator=(const InferSession&) = delete;

    /**
     * 单次推理
     *
     * @param inputIds   token id 序列，shape [batchSize * seqLen]，INT64
     * @param batchSize  本次 batch 大小
     * @param seqLen     本次序列长度
     * @return           last_hidden_state，shape [batchSize * seqLen * 50257]，FP32
     */
    std::vector<float> infer(const std::vector<int64_t>& inputIds,
                             int batchSize,
                             int seqLen);

    /**
     * Benchmark
     * throughput 单位：tokens/s
     */
    BenchResult benchmark(int batchSize, int seqLen,
                          int nWarmup = 20, int nRun = 100);

private:
    void allocBuffers();

    Logger& m_logger;
    int     m_max_batch;
    int     m_max_seq_len;

    std::unique_ptr<nvinfer1::IRuntime>          m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>        m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext>  m_context;

    // GPU 缓冲区
    void* m_device_input  = nullptr;   // INT64
    void* m_device_output = nullptr;   // FP32

    // Pinned Memory
    int64_t* m_pinned_input  = nullptr;
    float*   m_pinned_output = nullptr;

    cudaStream_t m_stream = nullptr;

    // static constexpr int k_HIDDEN = 768;  // GPT-2 hidden size
    static constexpr int k_VOCAB  = 50257;  // ← 替换 k_HIDDEN = 768
};