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

    // 多 optimization profile 场景下选择使用哪一组形状。
    // 单 profile 或全静态网络的 engine 无需调用。必须在 enqueue 之前设置；
    // 切换 profile 后需要重新绑定 tensor address（TRT 会按新形状校验地址）。
    bool SetOptimizationProfile(int32_t index, cudaStream_t stream);

    // 查询某个张量在 optimization profile 里的形状（`kMIN` / `kOPT` / `kMAX`）。
    //
    // **为什么要有这个接口**：运行时需要知道"每步最多能送多少 token"之类的**上界**，而这个上界
    // 是建图时定下的、运行时的 Config 里没有（例如 `max_prefill_seq_len`）。按 `Config` 假定会
    // 与真实建的图漂移 —— 项目既有纪律是"**按对方查询、不按配置假定**"（workspace 版见
    // `paged_attention_split.hpp` 的注释与 `PROGRESS.md` §2.15）。
    //
    // 只查 0 号 profile：运行时要求每个引擎**恰好一个** profile（不变量 2 / D8）。
    // 查询失败（张量名不存在、不是动态输入、引擎为空）返回 `false`，`*dims` 不写。
    bool GetProfileDims(const std::string& name, nvinfer1::OptProfileSelector selector,
                        nvinfer1::Dims* dims) const;

    // 上述查询的某一维（越界 / 查询失败返回 -1）。`dim` 从 0 起。
    int32_t GetProfileDim(const std::string& name, nvinfer1::OptProfileSelector selector,
                          int32_t dim) const;

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
