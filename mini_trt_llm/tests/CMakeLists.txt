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
# 这个项目已经吃过"声明了却没人核对"的亏（见 docs/phase3_test_plan.md + PH3-GAPS 的 G1c）。
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
# resnet18_convert_selftest 的理由（docs/phase3_test_plan.md + PH3-GAPS 的 G1c）。
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
# P4-INT8-a（`future_iterations.md` §1.5）的两件仪器自检
#
# 为什么进 ctest：这一条排查的结论全建立在两件仪器上——
#   ① `qdq_reference.py` 是**数值标尺**（ONNX 官方参考实现），它必须先被校准
#      （`TROUBLESHOOTING.md` #29.1/#30.4 的教训：标尺错了会给出错误裁决）；
#   ② `add_probe_outputs.py` 是**采样器**，它必须保证"探针图 = 产物图 + 探针"
#      （节点 / initializer 逐字节不变），否则测的不是原来那张图。
# 两件都自造 fixture、不依赖 GPU 与外部数据 → 沙箱即可跑，因此**不用** SKIP_RETURN_CODE。
# ---------------------------------------------------------------------------
if(Python3_Interpreter_FOUND)
    add_test(NAME qdq_reference_selftest
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/validate/qdq_reference.py --self-test)
    add_test(NAME add_probe_outputs_selftest
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/convert/add_probe_outputs.py --self-test)
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
# 最后跑这一条（顺序见 docs/future_iterations_development_plan.md + OI-RUNBOOK-RESULTS）。
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

# ---------------------------------------------------------------------------
# 性能画像 target（`future_iterations.md` §6.3 与 §11 的 G6）
#
# 为什么不进默认构建：profile 要独占 GPU、产出体积大的报告，且只有真机才有意义。
# 所以做成一堆**显式** target（`cmake --build build --target profile_gpt2`）：
# 不挂在 all 上，也不进 ctest。
#
# 报告默认落 /tmp（**不入库**，作者 2026-09-27 决定）：体积大、且与机器态强相关。
# 完整口径（同轮 ABBA、跨构建报极差、判别下限 ±400~600 µs）见开发计划 §11.3。
# nsys 跑完会自动导出 kernel / API 摘要并交给 `summarize_nsys.py` 分桶——那张三层表
# 就是"sampler 在整步 decode 里占多少"的答案（P6_3-4 / PF-4）。
# ---------------------------------------------------------------------------
set(MINI_TRT_PROFILE_OUTPUT_DIR "/tmp/mini_trt_llm_profiles" CACHE PATH
    "profile_* target 的报告输出目录（默认 /tmp，不入库）")

# 分桶脚本的护栏自检：合成 CSV → 已知分桶；名单写错 / 坏输入必须报错。
# 不需要 GPU，也不依赖任何外部产物 → 沙箱即可跑，因此不用 SKIP_RETURN_CODE。
if(Python3_Interpreter_FOUND)
    add_test(NAME profile_summary_selftest
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/profile/summarize_nsys.py --self-test)
endif()

if(UNIX)
    set(MINI_TRT_PROFILE_RUNNER
        "${CMAKE_CURRENT_SOURCE_DIR}/../tools/profile/run_profile.sh")

    # GPT-2 decode：端到端计时 + kernel 分桶 + 分配开销（开发计划 §11.5 的 P6_3-3~5）。
    add_custom_target(profile_gpt2
        COMMAND bash "${MINI_TRT_PROFILE_RUNNER}" nsys gpt2_decode "Gpt2DecodePerf.*"
                $<TARGET_FILE:mini_trt_llm_tests> "${MINI_TRT_PROFILE_OUTPUT_DIR}"
        DEPENDS mini_trt_llm_tests
        COMMENT "nsys: GPT-2 decode 画像（报告落 ${MINI_TRT_PROFILE_OUTPUT_DIR}，不入库）"
        VERBATIM)

    # ResNet18 / CV：复用既有的 CVRunner benchmark 用例。
    add_custom_target(profile_resnet18
        COMMAND bash "${MINI_TRT_PROFILE_RUNNER}" nsys resnet18 "CvRunnerTest.BenchmarkReportsFiniteStats"
                $<TARGET_FILE:mini_trt_llm_tests> "${MINI_TRT_PROFILE_OUTPUT_DIR}"
        DEPENDS mini_trt_llm_tests
        COMMENT "nsys: ResNet18 画像（报告落 ${MINI_TRT_PROFILE_OUTPUT_DIR}，不入库）"
        VERBATIM)

    # ncu 变体：默认只采 20 个 kernel（整网采集过久）；用 MINI_TRT_NCU_KERNEL_FILTER
    # 可以只盯我们自己的 kernel（如 "regex:PagedAttention|WriteKV"）。
    # WSL2 上若报 performance counter 权限错，按 AGENTS.md §1 无头导出后拷 Windows 看。
    add_custom_target(profile_gpt2_ncu
        COMMAND bash "${MINI_TRT_PROFILE_RUNNER}" ncu gpt2_decode_ncu "Gpt2DecodePerf.*"
                $<TARGET_FILE:mini_trt_llm_tests> "${MINI_TRT_PROFILE_OUTPUT_DIR}"
        DEPENDS mini_trt_llm_tests
        COMMENT "ncu: GPT-2 decode（报告落 ${MINI_TRT_PROFILE_OUTPUT_DIR}，不入库）"
        VERBATIM)

    add_custom_target(profile_resnet18_ncu
        COMMAND bash "${MINI_TRT_PROFILE_RUNNER}" ncu resnet18_ncu "CvRunnerTest.BenchmarkReportsFiniteStats"
                $<TARGET_FILE:mini_trt_llm_tests> "${MINI_TRT_PROFILE_OUTPUT_DIR}"
        DEPENDS mini_trt_llm_tests
        COMMENT "ncu: ResNet18（报告落 ${MINI_TRT_PROFILE_OUTPUT_DIR}，不入库）"
        VERBATIM)
endif()
