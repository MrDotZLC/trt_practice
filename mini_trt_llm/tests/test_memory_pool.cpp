#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace mini_trt_llm {

TEST(DeviceBufferTest, AllocateAndFree) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    DeviceBuffer buf;
    EXPECT_TRUE(buf.Allocate(1024));
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 1024u);
    buf.Free();
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(DeviceBufferTest, Resize) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    DeviceBuffer buf(1024);
    EXPECT_TRUE(buf.Resize(2048));
    EXPECT_GE(buf.size(), 2048u);
}

TEST(PinnedBufferTest, AllocateAndFree) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    PinnedBuffer buf;
    EXPECT_TRUE(buf.Allocate(1024));
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 1024u);
    buf.Free();
    EXPECT_EQ(buf.data(), nullptr);
}

}  // namespace mini_trt_llm
