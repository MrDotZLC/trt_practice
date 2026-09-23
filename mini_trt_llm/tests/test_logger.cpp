#include "mini_trt_llm/utils/logger.hpp"
#include <gtest/gtest.h>

namespace mini_trt_llm {

TEST(LoggerTest, SetLogLevel) {
    SetLogLevel(LogLevel::kWARN);
    EXPECT_EQ(GlobalLogLevel(), LogLevel::kWARN);
    SetLogLevel(LogLevel::kINFO);
    EXPECT_EQ(GlobalLogLevel(), LogLevel::kINFO);
}

TEST(LoggerTest, MacrosDoNotThrow) {
    SetLogLevel(LogLevel::kVERBOSE);
    EXPECT_NO_THROW(MINI_TRT_LOG_INFO("test info message"));
    EXPECT_NO_THROW(MINI_TRT_LOG_WARN("test warn message"));
    EXPECT_NO_THROW(MINI_TRT_LOG_ERROR("test error message"));
}

}  // namespace mini_trt_llm
