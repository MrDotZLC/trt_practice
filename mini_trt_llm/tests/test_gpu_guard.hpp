#pragma once

#include <cuda_runtime.h>

namespace mini_trt_llm {
namespace test_support {

// 判断当前环境是否真的能用 CUDA。
//
// 存在的理由：沙箱 / CI 里 GPU 被屏蔽，驱动不匹配时 CUDA 上下文无法初始化，
// 直接调用 CUDA API 会抛 cudaErrorInsufficientDriver 让用例"失败"。对这类用例来说
// "环境不具备" 与 "代码有 bug" 是两回事，必须区分，否则 ctest 长期不绿、真实回归信号被淹没。
//
// 该 header 只有声明（.hpp），不会被打进 tests 的源文件 glob。
inline bool HasCudaDevice() {
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess;
}

}  // namespace test_support
}  // namespace mini_trt_llm
