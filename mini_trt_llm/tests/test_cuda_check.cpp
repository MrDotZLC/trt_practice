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
    // **必须把这条错误读走**：CUDA 的 last-error 是粘性的，读一次才会清空。
    // 否则它会一直挂在线程上，被后面第一个 cudaGetLastError() 的调用者（例如某个
    // plugin 的 launch 检查）误当成本次调用失败——表现为"某个不相干的用例莫名其妙
    // 报 invalid device ordinal"，而失败点离真正的污染源很远。
    (void)cudaGetLastError();
    // 有真机时才断言"这次消费真的生效"：读完一次后，错误槽应当回到 cudaSuccess。
    // 无 GPU 环境下 CUDA 每次调用都会重新报初始化失败（每次读都不是 success），
    // 那种情况下这条断言没有意义。
    if (test_support::HasCudaDevice()) {
        EXPECT_EQ(cudaGetLastError(), cudaSuccess)
            << "上一条错误必须被消费掉，否则会污染后续用例";
    }
}

}  // namespace mini_trt_llm
