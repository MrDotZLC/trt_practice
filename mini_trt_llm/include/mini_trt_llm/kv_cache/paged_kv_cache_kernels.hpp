#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mini_trt_llm {

// KV Cache 的分页布局（与 PagedAttentionPlugin 的输入契约一致）：
//   key_cache / value_cache  [num_blocks, block_size, num_kv_heads, head_size]
//
// 源张量来自引擎输出，布局为 [batch, num_kv_heads, tokens, head_size]。
// 两侧的 "token 轴" 位置不同，所以必须显式做一次重排，不能直接 memcpy。
struct PagedKVWriteArgs {
    const void* key = nullptr;
    const void* value = nullptr;
    void* key_cache = nullptr;
    void* value_cache = nullptr;
    const int32_t* block_tables = nullptr;  // [batch, max_blocks_per_seq]
    int32_t* context_lens = nullptr;        // [batch]
    int32_t batch_size = 0;
    int32_t tokens = 0;  // 本次写入的 token 数：prefill 为序列长度，decode 为 1
    int32_t num_kv_heads = 0;
    int32_t head_size = 0;
    int32_t block_size = 0;
    int32_t max_blocks_per_seq = 0;
    bool is_half = false;
    // false：从每个序列的第 0 个位置开始覆盖写（prefill）。调用方负责在写完后
    //        把 host 侧的 context_lens 设为 tokens 并 UploadMetadata。
    // true ：从 context_lens[b] 开始追加（decode），并在写完后于**设备端**
    //        把 context_lens[b] 增加 tokens——每步都回 host 改一次会让
    //        自回归循环里出现 H2D 拷贝（AGENTS.md §3.A.3）。
    bool append = false;
};

// 按分页布局写入 K/V。
cudaError_t LaunchWriteKV(const PagedKVWriteArgs& args, cudaStream_t stream);

// 在设备端把 context_lens 整体增加 tokens（每个序列一个线程）。
// 单独一个 kernel 是为了让"读位置"与"推进长度"严格分成两个阶段，
// 避免同一 batch 内出现"有些线程还在按旧长度写、有些已经推进"的竞态。
cudaError_t LaunchAdvanceContextLens(int32_t* context_lens, int32_t batch_size,
                                     int32_t tokens, cudaStream_t stream);

}  // namespace mini_trt_llm
