#pragma once

#include <cuda_runtime.h>

namespace mini_trt_llm {

// 基于 CUDA Event 的 GPU 计时器。
// Start/Stop 记录同 stream 上的两个事件，通过 cudaEventElapsedTime 返回毫秒级耗时。
// 非线程安全，且 Stop 会同步等待 stop 事件完成。
class CudaTimer {
 public:
    CudaTimer();
    ~CudaTimer();

    // 禁用拷贝：cudaEvent_t 不可安全复制。
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    // 在指定 stream 上记录开始事件。
    void Start(cudaStream_t stream = nullptr);
    // 在指定 stream 上记录结束事件，同步等待后返回耗时（单位：毫秒）。
    // 若 Start 未调用，返回 0.0f。
    float Stop(cudaStream_t stream = nullptr);

 private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    bool started_;
};

}  // namespace mini_trt_llm
