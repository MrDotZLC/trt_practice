#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace mini_trt_llm {

TEST(DeviceBufferTest, AllocateAndFree) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    DeviceBuffer buf;
    EXPECT_TRUE(buf.Allocate(1024));
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 1024u);
    buf.Free();
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(DeviceBufferTest, Resize) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    DeviceBuffer buf(1024);
    EXPECT_TRUE(buf.Resize(2048));
    EXPECT_GE(buf.size(), 2048u);
}

TEST(PinnedBufferTest, AllocateAndFree) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PinnedBuffer buf;
    EXPECT_TRUE(buf.Allocate(1024));
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 1024u);
    buf.Free();
    EXPECT_EQ(buf.data(), nullptr);
}

}  // namespace mini_trt_llm
