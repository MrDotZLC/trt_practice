# Unit tests for mini_trt_llm

cmake_minimum_required(VERSION 3.18)

enable_testing()

file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(mini_trt_llm_tests ${TEST_SRCS})

target_link_libraries(mini_trt_llm_tests
    mini_trt_llm
    gtest
    gtest_main
    ${CUDA_RT_LIB}
)

target_compile_options(mini_trt_llm_tests PRIVATE -Wno-deprecated-declarations)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(mini_trt_llm_tests)
