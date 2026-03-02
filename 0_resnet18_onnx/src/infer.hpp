#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include "logger.hpp"

/**
 * TensorRT 推理会话（Inference Session）
 *
 * 对象生命周期与内部结构：
 *
 *   IRuntime
 *     └── ICudaEngine          反序列化 engine，包含编译好的 CUDA kernel
 *           └── IExecutionContext  执行上下文，持有推理状态
 *                                  单线程复用，多线程需每线程独立创建
 *
 * 显存布局：
 *   mDeviceInput  → GPU 输入缓冲区  [maxBatch * 3 * 224 * 224 * sizeof(float)]
 *   mDeviceOutput → GPU 输出缓冲区  [maxBatch * 1000 * sizeof(float)]
 *
 * 推理流程（异步）：
 *   1. setInputShape()       告知 context 本次实际 batch 大小
 *   2. setTensorAddress()    绑定输入/输出 GPU 指针
 *   3. cudaMemcpyAsync()     H2D（Host to Device）异步拷贝输入数据
 *   4. enqueueV3()           将推理任务提交到 CUDA Stream（异步，立即返回）
 *   5. cudaMemcpyAsync()     D2H（Device to Host）异步拷贝输出数据
 *   6. cudaStreamSynchronize() 等待 stream 中所有任务完成
 *
 * CUDA Stream（流）：
 *   GPU 任务队列，同一 stream 内任务顺序执行，不同 stream 可并行。
 *   使用异步接口 + stream 可以将 H2D / 推理 / D2H 流水线化，
 *   提高 GPU 利用率（本实现单 stream，流水线优化留作扩展）。
 */
class InferSession {
public:
    /**
     * @param enginePath  .engine 文件路径
     * @param logger      TRT Logger，需与构建时同一实例或同类型实例
     * @param maxBatch    最大 batch 大小，须 ≤ 构建时的 kMAX
     */
    explicit InferSession(const std::string &enginePath, Logger &logger,
                          int maxBatch = 16);
    ~InferSession();

    // 禁止拷贝（持有 GPU 资源，拷贝语义不安全）
    InferSession(const InferSession &) = delete;
    InferSession &operator=(const InferSession &) = delete;

    /**
     * 执行一次推理
     *
     * @param inputHost  CPU 端输入数据，NCHW 格式，FP32
     *                   大小必须 = batchSize * 3 * 224 * 224
     * @param batchSize  本次实际 batch 大小，须在 [kMIN, kMAX] 范围内
     * @return           CPU 端输出 logits，大小 = batchSize * 1000，1000 维向量
     *                      （ResNet 系列最初是为 ImageNet (ILSVRC)
     * 大规模视觉识别挑战赛设计的， ImageNet 数据集共有 1000
     * 个类别，故这里默认设为 1000 维）
     */
    std::vector<float> infer(const std::vector<float> &inputHost,
                             int batchSize);

    /**
     * 性能基准测试（Benchmark）
     *
     * 使用 CUDA Event 计时（比 std::chrono 精确，避免 CPU 调度 jitter）。
     * 统计指标：mean / p50 / p99 延迟，吞吐量（img/s）。
     *
     * @param batchSize  测试用 batch 大小
     * @param nWarmup    预热次数（排除 GPU 首次启动 JIT 开销）
     * @param nRun       正式计时次数
     */
    void benchmark(int batchSize, int nWarmup = 50, int nRun = 200,
                   const std::string &label = "");

private:
    // 根据 maxBatch 分配 GPU 输入/输出缓冲区
    void allocBuffers();

    Logger &m_logger;
    int m_max_batch;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;  // 负责反序列化 engine
    std::unique_ptr<nvinfer1::ICudaEngine>
        m_engine;  // 包含编译好的 CUDA kernel，线程安全，可多线程共享
    std::unique_ptr<nvinfer1::IExecutionContext>
        m_context;  // 执行上下文，持有推理状态，单线程复用

    // GPU 缓冲区
    void *m_device_input = nullptr;
    void *m_device_output = nullptr;

    // 模型固定参数（ResNet18 ImageNet）
    static constexpr int k_C = 3;
    static constexpr int k_H = 224;
    static constexpr int k_W = 224;
    static constexpr int k_CLS = 1000;  // 输出类别

    // CUDA Stream
    cudaStream_t m_stream = nullptr;

    float *m_pinned_input = nullptr;  // Pinned Memory 输出缓冲区
    // 替换原来的临时 std::vector<float> outputHost
    float *m_pinned_output = nullptr;  // Pinned Memory 输出缓冲区
};