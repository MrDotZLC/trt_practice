#include "mini_trt_llm/utils/timer.hpp"
#include "test_gpu_guard.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace mini_trt_llm {

TEST(CudaTimerTest, CreateDestroy) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    EXPECT_NO_THROW(CudaTimer timer);
}

TEST(CudaTimerTest, StartStop) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    CudaTimer timer;
    cudaStream_t stream = nullptr;
    EXPECT_NO_THROW(timer.Start(stream));
    EXPECT_NO_THROW(timer.Stop(stream));
}

}  // namespace mini_trt_llm
