#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mini_trt_llm {

// RoPE kernel 的启动参数。query / key 均为 [batch, heads, seq_len, head_size] 行优先布局，
// position_ids 为 [batch, seq_len]。
//
// 旋转约定采用 HuggingFace LLaMA 的"前后半对"形式（half-split）：
//   对 j ∈ [0, rotary_dim/2)：out[j] = x[j]*cos - x[j+h]*sin
//                           out[j+h] = x[j+h]*cos + x[j]*sin，其中 h = rotary_dim/2
// 之所以不采用相邻两维配对（GPT-NeoX 风格）：Phase 1 的参考实现是
// transformers 的 apply_rotary_pos_emb，两者必须一致才能做逐元素对比。
struct RoPEKernelArgs {
    const void* query = nullptr;       // [batch, num_heads, seq_len, head_size]
    const void* key = nullptr;         // [batch, num_kv_heads, seq_len, head_size]
    const int32_t* position_ids = nullptr;  // [batch, seq_len]；允许非连续（KV Cache 场景）
    void* query_out = nullptr;
    void* key_out = nullptr;
    int32_t batch_size = 0;
    int32_t seq_len = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_size = 0;
    int32_t rotary_dim = 0;  // 必须为偶数且 <= head_size
    float base = 10000.0f;
    bool is_half = false;
};

// 启动 RoPE kernel（query 与 key 各一次，共用一个实现）。
// 返回 cudaGetLastError() 的结果，便于插件侧转成错误码而不抛异常。
cudaError_t LaunchRoPE(const RoPEKernelArgs& args, cudaStream_t stream);

}  // namespace mini_trt_llm
