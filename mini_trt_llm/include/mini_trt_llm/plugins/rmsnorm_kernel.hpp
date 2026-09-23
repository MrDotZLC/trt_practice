#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mini_trt_llm {

// RMSNorm CUDA kernel 的启动参数。
//
// 归一化语义：对最后一维做归一化，其余维度展平为 row。
//   rms = sqrt(mean(x^2) + eps)
//   out = x / rms * weight
struct RmsNormKernelArgs {
    const void* input = nullptr;   // 行优先 [rows, hidden_size]
    const void* weight = nullptr;  // [hidden_size]，dtype 必须与 input 一致
    void* output = nullptr;        // 形状与 input 相同
    int64_t rows = 0;
    int32_t hidden_size = 0;
    float eps = 1e-6f;
    bool is_half = false;  // false → FP32，true → FP16
};

// 启动 RMSNorm kernel。
//
// 之所以把 kernel 入口单独暴露给测试，而不是只留在 Plugin 内部：L1 层单测需要绕过
// TensorRT engine 构建直接验证数值正确性（无 GPU 环境下会跳过），Plugin::enqueue 与
// 测试共用同一入口可避免两处实现漂移。
//
// 返回 cudaGetLastError() 的结果而非抛异常：enqueue 处于 noexcept 契约中，插件侧需要
// 把失败转换成 TensorRT 的错误码。
cudaError_t LaunchRmsNorm(const RmsNormKernelArgs& args, cudaStream_t stream);

}  // namespace mini_trt_llm
