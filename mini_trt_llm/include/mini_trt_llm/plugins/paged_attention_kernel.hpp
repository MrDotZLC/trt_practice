#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mini_trt_llm {

// PagedAttention（仅 Decoding 阶段）的启动参数。
//
// 语义：query 的序列长度为 1（自回归每步只生成一个 token），注意力作用于
// Paged KV Cache 中该 batch 已写入的 context_len 个历史位置。
//
// 布局约定：
//   query        [batch, num_heads, 1, head_size]
//   key_cache    [num_blocks, block_size, num_kv_heads, head_size]
//   value_cache  同 key_cache
//   block_tables [batch, max_blocks_per_seq]，物理块号
//   context_lens [batch]，每个序列当前的有效长度
struct PagedAttentionKernelArgs {
    const void* query = nullptr;
    const void* key_cache = nullptr;
    const void* value_cache = nullptr;
    const int32_t* block_tables = nullptr;
    const int32_t* context_lens = nullptr;
    void* output = nullptr;
    int32_t batch_size = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;  // MHA 时等于 num_heads；MQA 时为 1
    int32_t head_size = 0;
    int32_t block_size = 0;
    int32_t max_blocks_per_seq = 0;
    float scale = 0.0f;  // 通常为 1/sqrt(head_size)
    bool is_half = false;
};

// 启动 Decoding 阶段的 PagedAttention kernel。
// 返回 cudaGetLastError() 的结果，便于插件侧转成错误码而不抛异常。
cudaError_t LaunchPagedAttention(const PagedAttentionKernelArgs& args, cudaStream_t stream);

}  // namespace mini_trt_llm
