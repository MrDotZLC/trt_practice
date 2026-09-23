#include "mini_trt_llm/utils/timer.hpp"
#include "test_gpu_guard.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace mini_trt_llm {

TEST(CudaTimerTest, CreateDestroy) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    EXPECT_NO_THROW(CudaTimer timer);
}

TEST(CudaTimerTest, StartStop) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    CudaTimer timer;
    cudaStream_t stream = nullptr;
    EXPECT_NO_THROW(timer.Start(stream));
    EXPECT_NO_THROW(timer.Stop(stream));
}

}  // namespace mini_trt_llm
