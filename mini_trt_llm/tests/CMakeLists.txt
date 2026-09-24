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

# ---------------------------------------------------------------------------
# ONNX 图结构探针（Phase 3 G7）
#
# 为什么进 ctest：它是"图变了要立刻知道"的护栏——`gpt2.onnx` 一旦重新导出、
# 换 opset 或换模型，算子分布与子图形态都会变，而方案 B 的识别结论与对齐判据
# 都建立在这张分布上。靠人记得跑等于没有。
#
# 环境语义：缺 Python / 缺 onnx 包 / 缺 1_gpt2_onnx/gpt2.onnx 时**跳过**（返回 77），
# 不是失败——缺环境不等于图有问题，混在一起会让回归信号失真。
# 未找到 Python 解释器时干脆不注册这条 test（同理）。
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME onnx_graph_probe
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/inspect_onnx.py
                ${CMAKE_CURRENT_SOURCE_DIR}/../../1_gpt2_onnx/gpt2.onnx
                --check --skip-if-missing)
    set_tests_properties(onnx_graph_probe PROPERTIES SKIP_RETURN_CODE 77)
endif()

