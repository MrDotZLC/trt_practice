#include "mini_trt_llm/utils/cuda_check.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>

namespace mini_trt_llm {

TEST(CudaCheckTest, SuccessDoesNotThrow) {
    int device = 0;
    EXPECT_NO_THROW(CUDA_CHECK(cudaGetDevice(&device)));
}

TEST(CudaCheckTest, InvalidDeviceThrows) {
    EXPECT_THROW(CUDA_CHECK(cudaSetDevice(999)), std::runtime_error);
}

}  // namespace mini_trt_llm
