#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mini_trt_llm {

// KV Cache 的分页布局（与 PagedAttentionPlugin 的输入契约一致）：
//   key_cache / value_cache  [num_blocks, block_size, num_kv_heads, head_size]
//
// 源张量来自引擎输出，布局为 [batch, num_kv_heads, tokens, head_size]。
// 这里的 batch 是**参与本次写入的行数**（= row_count），不等于缓存批已登记的序列数：
// 两者由 rows 的行映射联系起来（见下）。
// 两侧的 "token 轴" 位置不同，所以必须显式做一次重排，不能直接 memcpy。
struct PagedKVWriteArgs {
    const void* key = nullptr;
    const void* value = nullptr;
    void* key_cache = nullptr;
    void* value_cache = nullptr;
    const int32_t* block_tables = nullptr;  // [batch, max_blocks_per_seq]
    int32_t* context_lens = nullptr;        // [batch]
    // 行映射 [row_count]：引擎第 i 行 → 缓存批内第 rows[i] 行。**必须是设备可读的地址**
    // （kernel 在设备侧按它寻址），每个元素的值必须落在块表的行数以内。
    //
    // 为什么写回必须带映射：源张量的行序与块表的行序**不总是同一个集合**——S3 的活跃批下
    // 上下文段只装本步新入批的行（B_new），而块表里还有正在 generation 的行。少了映射就会
    // 按批内已登记序列数逐行写，拿源缓冲里上一轮的残留行去覆盖**别的序列自己的** prompt K/V
    // （静默算错，见 p5_s3_interface_spec §3）。
    // decode 追加的行序就等于批内顺序，调用方传恒等表（`PagedKVCache` 构造期备好一份，
    // 免得每步都拷一次 —— 解码循环内不得有 H2D，AGENTS.md §3.A.3）。
    const int32_t* rows = nullptr;  // [row_count]
    // 本次参与写入的行数（= 源张量第 0 维）。**元素总数按它算**，不是批内已登记序列数。
    int32_t row_count = 0;
    int32_t tokens = 0;  // 本次写入的 token 数：prefill 为序列长度，decode 为 1
    int32_t num_kv_heads = 0;
    int32_t head_size = 0;
    int32_t block_size = 0;
    int32_t max_blocks_per_seq = 0;
    // 目标 cache 的元素类型（由 PagedAttention 的 cache 输入决定）
    bool is_half = false;
    // 源 K/V（引擎输出）的元素类型。**与 cache 可能不同**：弱类型网络下导出的 K/V
    // 由 TRT 决定类型（FP16 引擎里实测是 FP32），而 cache 是 `weight_dtype` 精度。
    // 内核因此按"源→目标"做一次转换，而不是假定两者一致（见 TROUBLESHOOTING + TS-018）。
    bool source_is_half = false;
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
