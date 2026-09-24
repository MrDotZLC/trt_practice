#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace mini_trt_llm {

// 采样器公共参数（设备侧 API）。
//
// logits 与 token_ids 都是设备指针：Decode 自回归循环必须全程驻留显存，
// 采样结果直接写回 device，不允许经过 host（AGENTS.md §3.A.3）。
struct SamplerArgs {
    const void* logits = nullptr;  // [batch_size, vocab_size]，FP16 / FP32
    int32_t* token_ids = nullptr;  // [batch_size]，输出
    int32_t batch_size = 0;
    int32_t vocab_size = 0;
    bool is_half = false;
    // 随机源：host 传 seed + offset，device 侧用 Philox 确定性生成（Q8）。
    // 这样测试可用固定 seed 复现，同时避免 host-device 频繁同步。
    uint64_t seed = 0;
    uint64_t offset = 0;
};

// Top-K / Top-P 的 k 与 p 都是 per-batch tensor（Q7），以支持连续批处理中
// 每个请求独立配置采样参数。
struct TopKSamplerArgs : SamplerArgs {
    const int32_t* top_k = nullptr;  // [batch_size]
};

struct TopPSamplerArgs : SamplerArgs {
    const float* top_p = nullptr;  // [batch_size]
};

// Greedy：取 logits 最大值的下标，并列时取最小下标（与 torch.argmax 一致）。
cudaError_t LaunchGreedySampler(const SamplerArgs& args, cudaStream_t stream);

// Top-K / Top-P 内部按行降序排序，需要调用方提供 workspace（避免在 Decode 循环里
// 反复 cudaMalloc）。用 *WorkspaceBytes() 计算所需大小。
size_t TopKSamplerWorkspaceBytes(int32_t batch_size, int32_t vocab_size);
cudaError_t LaunchTopKSampler(const TopKSamplerArgs& args, cudaStream_t stream,
                              void* workspace, size_t workspace_bytes);

size_t TopPSamplerWorkspaceBytes(int32_t batch_size, int32_t vocab_size);
cudaError_t LaunchTopPSampler(const TopPSamplerArgs& args, cudaStream_t stream,
                              void* workspace, size_t workspace_bytes);

}  // namespace mini_trt_llm
