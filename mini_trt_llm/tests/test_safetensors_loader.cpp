#include "mini_trt_llm/utils/safetensors_loader.hpp"
#include <gtest/gtest.h>
#include <string>

namespace mini_trt_llm {

TEST(SafetensorsLoaderTest, LoadNonExistentReturnsFalse) {
    SafetensorsLoader loader;
    EXPECT_FALSE(loader.LoadFromFile("/tmp/non_existent.safetensors"));
}

}  // namespace mini_trt_llm
