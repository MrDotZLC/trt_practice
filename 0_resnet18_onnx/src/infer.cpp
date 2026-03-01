#include "infer.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <nvtx3/nvToolsExt.h>   // ← 加这行：NVTX 标注

// ── 辅助：读取二进制文件到 buffer ────────────────────────────────────────────
static std::vector<char> readFile(const std::string& path)
{
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        throw std::runtime_error("[Infer] Cannot open: " + path);
    }

    // ate：打开时定位到文件末尾，tellg() 直接得到文件大小
    size_t sz = fin.tellg();
    fin.seekg(0);

    std::vector<char> buf(sz);
    fin.read(buf.data(), sz);
    return buf;
}

// ── 构造函数 ──────────────────────────────────────────────────────────────────
InferSession::InferSession(const std::string& enginePath,
                           Logger& logger,
                           int maxBatch)
    : m_logger(logger), m_max_batch(maxBatch) {
    // 1. 读取 engine 文件
    auto buf = readFile(enginePath);

    // 2. 创建 Runtime
    //    IRuntime 是推理侧入口，负责反序列化 engine。
    //    与构建侧 IBuilder 完全独立，生产部署时只需 Runtime。
    m_runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!m_runtime) {
        std::cerr << "[Infer] createInferRuntime returned nullptr\n";
        std::cerr << "[Infer] TRT version: " << NV_TENSORRT_VERSION << "\n";
        throw std::runtime_error("[Infer] createInferRuntime failed");
    }
    
    // 3. 反序列化 Engine
    //    deserializeCudaEngine 将二进制数据还原为 ICudaEngine 对象，
    //    并将编译好的 CUDA kernel 加载到 GPU。
    //    此步骤比构建快得多（秒级），适合每次启动时执行。
    m_engine.reset(m_runtime->deserializeCudaEngine(buf.data(), buf.size()));
    if (!m_engine)
        throw std::runtime_error("[Infer] deserializeCudaEngine failed");

    // 4. 创建执行上下文
    //    IExecutionContext 持有推理状态（动态 shape 绑定、中间激活值显存等）。
    //    同一 engine 可创建多个 context 用于多线程并发推理，
    //    本实现单线程，只创建一个。
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        throw std::runtime_error("[Infer] createExecutionContext failed");
    }

    // 5. 创建 CUDA Stream
    //    Stream 是 GPU 任务队列，异步操作都提交到 stream 中顺序执行。
    cudaStreamCreate(&m_stream);

    // 6. 分配 GPU 缓冲区
    allocBuffers();

    std::cout << "[Infer] Session ready: " << enginePath << "\n";
}

// ── 析构函数 ──────────────────────────────────────────────────────────────────
InferSession::~InferSession()
{
    // 释放 GPU 资源
    if (m_device_input)  cudaFree(m_device_input);
    if (m_device_output) cudaFree(m_device_output);
    if (m_stream)       cudaStreamDestroy(m_stream);
    // mContext / mEngine / mRuntime 由 unique_ptr 自动析构
}

// ── 分配 GPU 缓冲区 ───────────────────────────────────────────────────────────
void InferSession::allocBuffers() {
    // 按最大 batch 分配，推理时实际使用的 batch ≤ mMaxBatch
    size_t input_bytes = m_max_batch * k_C * k_H * k_W * sizeof(float);
    size_t output_bytes = m_max_batch * k_CLS * sizeof(float);

    cudaMalloc(&m_device_input, input_bytes);
    cudaMalloc(&m_device_output, output_bytes);

    std::cout << "[Infer] GPU buffers allocated:"
              << " input="  << input_bytes  / 1024 << " KB"
              << " output=" << output_bytes / 1024 << " KB\n";
}

// ── 单次推理 ──────────────────────────────────────────────────────────────────
std::vector<float> InferSession::infer(const std::vector<float>& inputHost,
                                       int batchSize) {
    // 1. 设置本次推理的动态输入 shape
    //    动态 shape 必须在每次推理前设置，告知 context 实际 batch 大小。
    //    TRT 据此确定输出 shape 和中间缓冲区大小。
    m_context->setInputShape("input", nvinfer1::Dims4{batchSize, k_C, k_H, k_W});

    // 2. 绑定 tensor 地址（TRT 10.x API）
    //    context 不缓存地址，每次推理前必须重新设置。
    m_context->setTensorAddress("input", m_device_input);
    m_context->setTensorAddress("output", m_device_output);

    size_t input_bytes = batchSize * k_C * k_H * k_W * sizeof(float);
    size_t output_bytes = batchSize * k_CLS * sizeof(float);

    // 3. H2D 异步拷贝（Host to Device）
    //    将 CPU 数据异步传输到 GPU，不阻塞 CPU。
    //    数据在 stream 中排队，保证在推理之前完成。
    nvtxRangePushA("H2D");      // NVTX 标注
    cudaMemcpyAsync(m_device_input, inputHost.data(), input_bytes, cudaMemcpyHostToDevice, m_stream);
    nvtxRangePop();             // NVTX 标注

    // 4. 异步推理
    //    enqueueV3 将推理任务提交到 stream，立即返回，不等待 GPU 完成。
    //    GPU 在后台执行，CPU 可继续做其他工作（本实现直接等待）。
    nvtxRangePushA("Infer");    // NVTX 标注
    if (!m_context->enqueueV3(m_stream)) {
        throw std::runtime_error("[Infer] enqueueV3 failed");
    }
    nvtxRangePop();             // NVTX 标注

    // 5. D2H 异步拷贝（Device to Host）
    //    将 GPU 输出结果异步拷贝回 CPU。
    //    因为在同一 stream 中，保证在推理完成后才执行。
    nvtxRangePushA("D2H");      // NVTX 标注
    std::vector<float> outputHost(batchSize * k_CLS);
    cudaMemcpyAsync(outputHost.data(), m_device_output, output_bytes, cudaMemcpyDeviceToHost, m_stream);
    nvtxRangePop();             // NVTX 标注

    // 6. 同步：等待 stream 中所有任务完成
    cudaStreamSynchronize(m_stream);

    return outputHost;
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
void InferSession::benchmark(int batchSize, int nWarmup, int nRun) {
    // 构造固定输入数据
    std::vector<float> input(batchSize * k_C * k_H * k_W, 0.5f);

    // 设置 shape 和地址（benchmark 期间固定不变）
    m_context->setInputShape("input", nvinfer1::Dims4{batchSize, k_C, k_H, k_W});
    m_context->setTensorAddress("input", m_device_input);
    m_context->setTensorAddress("output", m_device_output);

    size_t input_bytes = batchSize * k_C * k_H * k_W * sizeof(float);
    cudaMemcpyAsync(m_device_input, input.data(), input_bytes, cudaMemcpyHostToDevice, m_stream);
    cudaStreamSynchronize(m_stream);

    // ── Warmup（预热）──────────────────────────────────────────────────────
    // GPU kernel 首次启动存在初始化开销（JIT 编译缓存、显存页锁定等），
    // warmup 排除此干扰，使计时结果稳定。
    for (int i = 0; i < nWarmup; ++i) {
        m_context->enqueueV3(m_stream);
        cudaStreamSynchronize(m_stream);
    }

    // ── CUDA Event 计时 ────────────────────────────────────────────────────
    // CUDA Event 在 GPU 时间线上打点，比 std::chrono 更精确：
    //   - chrono 包含 CPU 调度 jitter（线程被抢占导致的误差）
    //   - CUDA Event 直接测量 GPU 执行时间，精度约 0.5 μs
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::vector<float> latencies(nRun);
    for (int i = 0; i < nRun; ++i) {
        nvtxRangePushA("benchmark_iter");           // NVTX 标注

        // EventRecord：在 stream 当前位置插入时间戳
        cudaEventRecord(ev_start, m_stream);
        m_context->enqueueV3(m_stream);
        cudaEventRecord(ev_stop, m_stream);

        // EventSynchronize：等待 evStop 完成（只等这一个 event，不等整个 stream）
        cudaEventSynchronize(ev_stop);

        // ElapsedTime：计算两个 event 之间的 GPU 时间，单位 ms
        cudaEventElapsedTime(&latencies[i], ev_start, ev_stop);

        nvtxRangePop();                             // NVTX 标注
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // ── 统计 ───────────────────────────────────────────────────────────────
    float mean = std::accumulate(latencies.begin(), latencies.end(), 0.f) / nRun;

    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    float p50 = sorted[nRun * 50 / 100];
    float p99 = sorted[nRun * 99 / 100];
    
    // 吞吐量：每秒处理的图片数
    float throughput = batchSize / (mean / 1000.f); // 张/ms -> 张/s

    std::cout << "[Benchmark]"
              << "  batch="      << batchSize
              << "  mean="       << mean      << " ms"
              << "  p50="        << p50       << " ms"
              << "  p99="        << p99       << " ms"
              << "  throughput=" << throughput << " img/s\n";
}