#pragma once

#include "logger.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace mini_trt_llm {

// 封装 TensorRT runtime / engine / execution context。
// 支持动态 shape 与 tensor address 绑定。
class Engine {
 public:
    struct BenchResult {
        float mean_ms;
        float p50_ms;
        float p99_ms;
        float throughput;
    };

    explicit Engine(const std::string& engine_path, Logger& logger);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool SetInputShape(const std::string& name, nvinfer1::Dims dims);
    bool SetTensorAddress(const std::string& name, void* ptr);

    // 提交推理到指定 stream（异步）
    bool Enqueue(cudaStream_t stream);

    // 同步 stream
    void Synchronize(cudaStream_t stream);

    // 对固定 shape 执行 n_warmup + n_run 次 benchmark
    BenchResult Benchmark(int batch_size, int seq_len,
                          int n_warmup, int n_run);

    nvinfer1::ICudaEngine* GetCudaEngine() const { return engine_.get(); }
    nvinfer1::IExecutionContext* GetContext() const { return context_.get(); }

 private:
    Logger& logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}  // namespace mini_trt_llm
