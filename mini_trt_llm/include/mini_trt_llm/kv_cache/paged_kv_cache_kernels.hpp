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
    // **S4 的 packed 源寻址（2026-10-04 加；默认 null/0 = 既有行为）**。
    //
    // 为什么需要它：S4 的源张量是**一个 packed 张量**（context 段的全部 token 在前、
    // generation 段的 1 token/行在后），所以"引擎第 b 行的 token 从源缓冲的哪里开始"不再等于
    // `b * tokens`：
    //   * prefill（`append == false`）：源行基址 = `cu_seqlens_ctx[b]`（**段内下标**，从 0 起）；
    //   * decode （`append == true`）：源行基址 = `cu_seqlens_ctx[context_seq_count] + b`
    //     （generation 段每行恰好 1 个 token，所以是连续的）。
    //
    // 两个量都是**设备值**（`cu_seqlens_ctx` 是设备数组），所以只能由 kernel 自己读 ——
    // 调用方拿不到、也不该拿（那会引入 D2H 同步）。nullptr 时维持 `b * tokens` 的既有语义
    // （S1/S2/S3 与既有用例都这么用）。
    const int32_t* cu_seqlens_ctx = nullptr;  // [context_seq_count + 1]
    int32_t context_seq_count = 0;            // S4 的段边界（context 段的序列数）
    // 目标缓存行的映射仍由 `rows`（+`row_count`）给出，与上面两个量正交。
    // S5：每行的**写回起点**（分块 prefill 的第 2 块起不能从 0 覆盖写）。
    // 同样必须是**设备**数组（kernel 直接读），nullptr = 每行从 0 覆盖写（S3/S4 的行为）。
    const int32_t* row_starts = nullptr;  // [row_count]
};

// 按分页布局写入 K/V。
cudaError_t LaunchWriteKV(const PagedKVWriteArgs& args, cudaStream_t stream);

// 在设备端把 context_lens 整体增加 tokens（每个序列一个线程）。
// 单独一个 kernel 是为了让"读位置"与"推进长度"严格分成两个阶段，
// 避免同一 batch 内出现"有些线程还在按旧长度写、有些已经推进"的竞态。
cudaError_t LaunchAdvanceContextLens(int32_t* context_lens, int32_t batch_size,
                                     int32_t tokens, cudaStream_t stream,
                                     const int32_t* rows = nullptr);

}  // namespace mini_trt_llm
