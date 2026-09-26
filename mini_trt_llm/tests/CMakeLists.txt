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

# ---------------------------------------------------------------------------
# ResNet18 转换脚本的"护栏自检"（Phase 4 P4-4）
#
# 为什么进 ctest：转换脚本里有 5 道自检（拒绝未消费的 initializer / 拒绝不支持的带权算子 /
# 拒绝逻辑名冲突 / 拒绝形状不自洽 / 拒绝缺 fc）。**护栏没有用例证明"它会拦人"就等于没有护栏**——
# 这个项目已经吃过"声明了却没人核对"的亏（见 docs/phase3_test_plan.md §5 的 G1c）。
# 自检把真模型按 5 种方式改坏，逐个确认每道护栏都会拦。
#
# 环境语义：缺 Python / 缺 onnx 包 / 缺 resnet18.onnx 时**跳过**（返回 77），不是失败
# ——与 onnx_graph_probe 同口径。无需 GPU，沙箱即可跑。
# ---------------------------------------------------------------------------
if(Python3_Interpreter_FOUND)
    add_test(NAME resnet18_convert_selftest
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/convert/onnx_to_mini_trt_llm.py
                --onnx ${CMAKE_CURRENT_SOURCE_DIR}/../../0_resnet18_onnx/resnet18.onnx
                --output_dir ${CMAKE_CURRENT_BINARY_DIR}/resnet18_convert_selftest
                --self-test --skip-if-missing)
    set_tests_properties(resnet18_convert_selftest PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---------------------------------------------------------------------------
# INT8 判据脚本的自检（future_iterations §1.6 的离线子项）
#
# 为什么进 ctest：`int8_eval.py` 里有 7 道护栏（缺 provenance / 标定集重叠 /
# 类型错 / 清单 sha256 不符 / logits 大小不符 / 只报率不报 n / 率与分子分母不自洽），
# 外加分层数学的自证。**护栏没有用例证明"它会拦人"就等于没有护栏**——同
# resnet18_convert_selftest 的理由（docs/phase3_test_plan.md §5 的 G1c）。
#
# 环境语义：不需要 GPU，也不需要任何外部数据（自造合成 fixture）→ 沙箱即可跑，
# 因此**不用** SKIP_RETURN_CODE：它没跑起来就是真问题，不是"缺环境"。
# 唯一的外部条件是 Python 解释器（找不到就干脆不注册，与上面两条一致）。
# ---------------------------------------------------------------------------
if(Python3_Interpreter_FOUND)
    add_test(NAME int8_eval_selftest
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/validate/int8_eval.py --self-test)
endif()

# ---------------------------------------------------------------------------
# GPT-2 BPE 参考数据的可复现性校验（future_iterations A1）
#
# 为什么进 ctest：`BpeTokenizer` 的判据是"与 HF 逐 token 全等"，那份 golden 就是标尺。
# 标尺必须能自证"它确实来自声称的 tokenizer 文件与版本"——否则参考被改错、或本机
# tokenizer 换了版本，测试会安静地跟着一起错。
#
# 环境语义：缺 Python / 缺 transformers / 缺本地 vocab.json+merges.txt → 返回 77（Skipped）。
# **脚本自身不联网**（local_files_only=True 且先查文件是否存在），所以这个跳过不等于"躲开故障"。
# ---------------------------------------------------------------------------
if(Python3_Interpreter_FOUND)
    add_test(NAME tokenizer_golden_check
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/make_tokenizer_golden.py
                --check ${CMAKE_CURRENT_SOURCE_DIR}/data/gpt2_tokenizer_golden.json
                --skip-if-missing)
    set_tests_properties(tokenizer_golden_check PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---------------------------------------------------------------------------
# INT8 口径交叉校验（future_iterations A2-5）
#
# 为什么进 ctest：`int8_eval.py` 是判据规格的第二种实现，C++ 精度用例是第一种。
# 两条独立实现必须给出同一组数字——不一致说明口径漂移，这时要查口径而不是改阈值。
#
# 环境语义：两份报告都在真机上才会产生（C++ 侧要 GPU 跑引擎，Python 侧要读那批 logits），
# **缺报告 → 77 跳过**。跳过不是"通过"：真机执行清单里要求先跑 dump 用例、再跑脚本、
# 最后跑这一条（顺序见 docs/future_iterations_development_plan.md §9）。
# ---------------------------------------------------------------------------
if(Python3_Interpreter_FOUND)
    add_test(NAME int8_crosscheck
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/validate/crosscheck_reports.py
                --cpp-report /tmp/mini_trt_llm_int8_crosscheck/cpp_report.json
                --py-report /tmp/mini_trt_llm_int8_crosscheck/py_report.json
                --skip-if-missing)
    set_tests_properties(int8_crosscheck PROPERTIES SKIP_RETURN_CODE 77)

    # 比对器自己的护栏自证（一致放行 / 分母漂移 / 阈值漂移 / 逐桶漂移 / 缺报告 → 77）。
    # 不需要任何外部产物，所以**不用** SKIP_RETURN_CODE：没跑起来就是真问题。
    add_test(NAME int8_crosscheck_selftest
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/validate/crosscheck_reports.py --self-test)
endif()
