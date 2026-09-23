#include "mini_trt_llm/utils/timer.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace mini_trt_llm {

TEST(CudaTimerTest, CreateDestroy) {
    EXPECT_NO_THROW(CudaTimer timer);
}

TEST(CudaTimerTest, StartStop) {
    CudaTimer timer;
    cudaStream_t stream = nullptr;
    EXPECT_NO_THROW(timer.Start(stream));
    EXPECT_NO_THROW(timer.Stop(stream));
}

}  // namespace mini_trt_llm
