#include "mini_trt_llm/utils/timer.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"

namespace mini_trt_llm {

CudaTimer::CudaTimer() : started_(false) {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void CudaTimer::Start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
    started_ = true;
}

float CudaTimer::Stop(cudaStream_t stream) {
    if (!started_) {
        return 0.0f;
    }
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    started_ = false;
    return ms;
}

}  // namespace mini_trt_llm
