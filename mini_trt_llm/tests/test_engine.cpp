#include "mini_trt_llm/core/engine.hpp"
#include "logger.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

namespace mini_trt_llm {

TEST(EngineTest, MissingEngineFileThrows) {
    Logger logger;
    EXPECT_THROW(Engine("/tmp/non_existent.engine", logger),
                 std::runtime_error);
}

}  // namespace mini_trt_llm
