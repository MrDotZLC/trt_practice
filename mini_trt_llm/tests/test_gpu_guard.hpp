#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace mini_trt_llm {
namespace test_support {

// CUDA 可用性探测结果。
//
// 为什么连 driver / runtime 版本一起带回：排障时要区分"这台机器没有 GPU"、
// "驱动版本不匹配"与"被沙箱屏蔽"，三者的下一步动作完全不同，而只回一个 bool 就都糊在一起了。
struct CudaProbe {
    bool available = false;
    int device_count = -1;
    cudaError_t error = cudaErrorUnknown;
    int driver_version = 0;
    int runtime_version = 0;
};

// 显式探测 CUDA 设备。
//
// 为什么主判定用 cudaGetDeviceCount 而不是 cudaGetDevice：后者只回答"当前设备选没选中"，
// 在"没有设备 / 驱动不匹配 / 被沙箱屏蔽"三种情况下都是同一个失败，给不出"设备数"这个事实。
// 实测（2026-09-25）：沙箱内 err=35(cudaErrorInsufficientDriver)、count=-1、driver=0；
// 真机 err=0、count=1、driver=12060 —— 这条差异就是本框架区分两种环境的依据。
inline CudaProbe ProbeCudaDevice() {
    CudaProbe probe;
    probe.error = cudaGetDeviceCount(&probe.device_count);
    probe.available = probe.error == cudaSuccess && probe.device_count > 0;
    (void)cudaDriverGetVersion(&probe.driver_version);
    (void)cudaRuntimeGetVersion(&probe.runtime_version);
    // 探测失败也会留下粘性错误码，这里读走它：否则第一个调用 cudaGetLastError() 的
    // 用例会把别人的残留当成本次失败（TROUBLESHOOTING #13 的教训）。
    (void)cudaGetLastError();
    return probe;
}

// 探测结果的一行摘要，供 skip 信息与 GpuEnvProbe 用例共用。
inline std::string CudaProbeToString(const CudaProbe& probe) {
    return "cudaGetDeviceCount -> err=" + std::to_string(static_cast<int>(probe.error)) + " (" +
           cudaGetErrorString(probe.error) + "), count=" + std::to_string(probe.device_count) +
           ", driver=" + std::to_string(probe.driver_version) +
           ", runtime=" + std::to_string(probe.runtime_version);
}

// 判断当前环境是否真的能用 CUDA。
//
// 存在的理由：沙箱 / CI 里 GPU 被屏蔽，驱动不匹配时 CUDA 上下文无法初始化，
// 直接调用 CUDA API 会抛 cudaErrorInsufficientDriver 让用例"失败"。对这类用例来说
// "环境不具备" 与 "代码有 bug" 是两回事，必须区分，否则 ctest 长期不绿、真实回归信号被淹没。
//
// 该 header 只有声明（.hpp），不会被打进 tests 的源文件 glob。
inline bool HasCudaDevice() { return ProbeCudaDevice().available; }

// 环境变量 MINI_TRT_REQUIRE_GPU=1 表示"本环境必须能跑 GPU 用例"。
inline bool RequireGpu() {
    const char* flag = std::getenv("MINI_TRT_REQUIRE_GPU");
    return flag != nullptr && *flag != '\0' && std::string(flag) != "0";
}

// skip / 失败信息的一行文本。
inline std::string NoCudaMessage(const CudaProbe& probe,
                                 const char* why = "No CUDA device available") {
    return std::string(why) + " —— " + CudaProbeToString(probe);
}

}  // namespace test_support
}  // namespace mini_trt_llm

// 没有可用 CUDA 设备时跳过本用例；但设了 MINI_TRT_REQUIRE_GPU=1 时**判失败**。
//
// 为什么需要这道闸门：无 GPU 的沙箱里跳过是设计要求（否则 CI 永远不绿，真回归信号被淹没，
// 见 PROGRESS §5.7）；但"静默"跳过会制造假绿——本项目 625939c 的真缺陷正是靠两条被
// 静默跳过的断言一路滑到提交（TROUBLESHOOTING #19）。所以这里默认跳过但**必须把探测结果
// 打出来**，而在真机 / 带 GPU 的 CI 上设 MINI_TRT_REQUIRE_GPU=1，任何一次跳过都变成失败。
//
// **必须是宏，不能封装成函数**：`GTEST_SKIP()` 与 `GTEST_FAIL()` 展开后都是 `return` 语句，
// 放进函数时只退出那个函数，调用方的测试体会**继续往下跑**，于是跳过变成"第一次 CUDA 调用处
// 的一串真失败"。本轮就踩过：封装成 `SkipIfNoCuda()` 后，沙箱里 63 条用例从 Skipped 变成 Failed。
// 宏展开在用例函数体内，`return` 退的才是用例本身。
#define MINI_TRT_SKIP_IF_NO_CUDA(...)                                                  \
    do {                                                                               \
        const ::mini_trt_llm::test_support::CudaProbe probe_ =                         \
            ::mini_trt_llm::test_support::ProbeCudaDevice();                           \
        if (probe_.available) {                                                        \
            break;                                                                     \
        }                                                                              \
        const ::std::string why_ =                                                     \
            ::mini_trt_llm::test_support::NoCudaMessage(probe_, ##__VA_ARGS__);        \
        if (::mini_trt_llm::test_support::RequireGpu()) {                              \
            GTEST_FAIL() << why_                                                       \
                         << "；已设置 MINI_TRT_REQUIRE_GPU，本环境必须能跑 GPU 用例";   \
        }                                                                              \
        GTEST_SKIP() << why_;                                                          \
    } while (false)
