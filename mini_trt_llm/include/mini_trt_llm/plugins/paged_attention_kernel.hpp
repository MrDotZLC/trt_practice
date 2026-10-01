#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
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
//   key_new      [batch, num_kv_heads, 1, head_size]（可选，见下）
//   value_new    同 key_new
//
// 关于 key_new / value_new：decode 第 t 步的注意力必须包含**当前 token 自己**的 K/V
// （数学上是 K_{0..t}），而这份 K/V 由本次前向才算出来，不可能预先写进 cache。
// 因此把它们作为并行输入传入：注意力在扫完 cache 里的 context_len 个历史位置后，
// 再单独把这一份当成第 context_len 个位置参与 online softmax。
// 若不传（has_current_token = false），行为与之前完全一致——
// 这也是"给定 cache 算注意力"这一原有契约的保留形式。
struct PagedAttentionKernelArgs {
    const void* query = nullptr;
    const void* key_cache = nullptr;
    const void* value_cache = nullptr;
    const int32_t* block_tables = nullptr;
    const int32_t* context_lens = nullptr;
    // 当前 token 的 K/V。仅当 has_current_token 为 true 时读取。
    const void* key_new = nullptr;
    const void* value_new = nullptr;
    void* output = nullptr;
    int32_t batch_size = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;  // MHA 时等于 num_heads；MQA 时为 1
    int32_t head_size = 0;
    int32_t block_size = 0;
    int32_t max_blocks_per_seq = 0;
    float scale = 0.0f;  // 通常为 1/sqrt(head_size)
    bool is_half = false;
    bool has_current_token = false;
};

// 启动 Decoding 阶段的 PagedAttention kernel。
// 返回 cudaGetLastError() 的结果，便于插件侧转成错误码而不抛异常。
cudaError_t LaunchPagedAttention(const PagedAttentionKernelArgs& args, cudaStream_t stream);

// 启动 split-K 版本的 Decoding PagedAttention（上下文维切开 + 两阶段归约）。
//
// `workspace` 由调用方（插件）按 `getWorkspaceSize()` 预留，尺寸契约 =
// `PagedAttentionWorkspaceBytes(batch_size, num_heads, head_size, kPagedAttentionMaxSplits)`
// ——布局与入参校验都走 `paged_attention_split.hpp` 里那一份实现，这里不重复算。
//
// 与单趟版本的**语义完全相同**（同一份 online softmax，只是累加顺序不同）；
// 保留单趟版本一是为了 `num_splits` 目标下的 A/B（同二进制、同 session 对比），
// 二是当 workspace 不可用时的兜底路径（见 `PagedAttentionPlugin::enqueue`）。
cudaError_t LaunchPagedAttentionSplit(const PagedAttentionKernelArgs& args, void* workspace,
                                      size_t workspace_bytes, cudaStream_t stream);

// 测试用：分片数覆盖。**不是产品开关**——它给三类用法：
//   ① `> 0`：强制该片数（钳到上限）——用于同轮 A/B 的"多片"档，以及制造空分片的边界用例；
//   ② `0`：按上下文自适应（生产默认）；
//   ③ `< 0`：**强制走旧单趟 kernel**——插件的 `enqueue` 会绕过 split 路径、直接调
//      `LaunchPagedAttention`。这是"同二进制、同 session 对比两版实现"的开关
//      （`TROUBLESHOOTING` + TS-037/TS-038：跨 session 的差值不可直接比）。
// 进程级、非线程安全，仅测试使用；用完必须复位为 0（`tests/paged_attention_test_support.hpp`
// 的 RAII 守卫就是干这个的）。
void SetPagedAttentionNumSplitsOverride(int32_t splits) noexcept;
int32_t PagedAttentionNumSplitsOverride() noexcept;

}  // namespace mini_trt_llm
