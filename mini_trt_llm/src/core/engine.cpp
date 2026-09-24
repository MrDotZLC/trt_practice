#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/timer.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mini_trt_llm {

Engine::Engine(const std::string& engine_path, Logger& logger) : logger_(logger) {
    auto buffer = ReadFile(engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(logger));
    NVINFER_CHECK(runtime_);

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    NVINFER_CHECK(engine_);

    context_.reset(engine_->createExecutionContext());
    NVINFER_CHECK(context_);
}

Engine::~Engine() = default;

bool Engine::SetInputShape(const std::string& name, nvinfer1::Dims dims) {
    if (!context_->setInputShape(name.c_str(), dims)) {
        MINI_TRT_LOG_ERROR("setInputShape failed for tensor: " << name);
        return false;
    }
    return true;
}

bool Engine::SetOptimizationProfile(int32_t index, cudaStream_t stream) {
    if (!context_ || index < 0) {
        return false;
    }
    if (!context_->setOptimizationProfileAsync(index, stream)) {
        MINI_TRT_LOG_ERROR("setOptimizationProfileAsync failed for profile: " << index);
        return false;
    }
    return true;
}

bool Engine::SetTensorAddress(const std::string& name, void* ptr) {
    if (!context_->setTensorAddress(name.c_str(), ptr)) {
        MINI_TRT_LOG_ERROR("setTensorAddress failed for tensor: " << name);
        return false;
    }
    return true;
}

bool Engine::Enqueue(cudaStream_t stream) {
    return context_->enqueueV3(stream);
}

void Engine::Synchronize(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

Engine::BenchResult Engine::Benchmark(int batch_size, int seq_len,
                                      int n_warmup, int n_run) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < n_warmup; ++i) {
        Enqueue(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaTimer timer;
    std::vector<float> latencies;
    latencies.reserve(n_run);

    // 记录每次推理的 GPU 耗时，Stop 内部已同步。
    for (int i = 0; i < n_run; ++i) {
        timer.Start(stream);
        Enqueue(stream);
        latencies.push_back(timer.Stop(stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // 计算平均 latency；mean / 1000.0f 将毫秒转换为秒，便于 throughput 单位统一。
    float mean = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / n_run;
    std::vector<float> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    // 使用简单索引分位数：对少量样本足够，非线性插值。
    float p50 = sorted[n_run * 50 / 100];
    float p99 = sorted[n_run * 99 / 100];
    // throughput 单位：tokens / second，假设每次处理 batch_size * seq_len 个 token。
    float throughput = static_cast<float>(batch_size * seq_len) / (mean / 1000.0f);

    return BenchResult{mean, p50, p99, throughput};
}

}  // namespace mini_trt_llm
