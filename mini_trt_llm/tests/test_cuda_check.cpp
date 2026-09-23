#include "mini_trt_llm/utils/cuda_check.hpp"
#include "test_gpu_guard.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>

namespace mini_trt_llm {

TEST(CudaCheckTest, SuccessDoesNotThrow) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    int device = 0;
    EXPECT_NO_THROW(CUDA_CHECK(cudaGetDevice(&device)));
}

// 无 GPU 时 cudaSetDevice 返回的是驱动类错误而非 kInvalidDevice，但同样应当抛异常，
// 因此这一条不需要门控，两种环境下都在验证 CUDA_CHECK 的失败路径。
TEST(CudaCheckTest, InvalidDeviceThrows) {
    EXPECT_THROW(CUDA_CHECK(cudaSetDevice(999)), std::runtime_error);
}

}  // namespace mini_trt_llm
