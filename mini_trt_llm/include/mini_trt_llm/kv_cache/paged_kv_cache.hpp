#pragma once

#include "mini_trt_llm/kv_cache/block_allocator.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <map>
#include <vector>

namespace mini_trt_llm {

// Paged KV Cache。
//
// 物理布局：[num_layers, num_blocks, block_size, num_kv_heads, head_size]，
// 每一层单独切出一段交给 PagedAttentionPlugin（它要的是 4-D 的
// [num_blocks, block_size, num_kv_heads, head_size]）。
//
// 两套"长度"要分清：
//   - host 侧 sequences_[id].length：本实现记账用，供调用方查询；
//   - 设备侧 context_lens[b]      ：引擎与 kernel 实际读的那个值。
// decode 每步都在设备端推进后者（避免自回归循环里出现 H2D 拷贝），
// 因此 host 侧的记账只在"每个请求开始时"与设备对齐一次。
class PagedKVCache {
 public:
    struct Config {
        // 物理块总数（显存池大小），与序列数无关。
        int32_t num_blocks = 0;
        // 每块容纳的 token 数。必须与建引擎时的 PagedAttention 属性一致。
        int32_t block_size = 0;
        int32_t num_layers = 0;
        int32_t num_kv_heads = 0;
        int32_t head_size = 0;
        // cache 元素精度（由 PagedAttention 的 cache 输入精度决定）。
        bool is_half = false;
        // 引擎导出的 K/V **源**精度。弱类型网络下 TRT 决定它（FP16 引擎里实测是 FP32），
        // 与 cache 精度不一定相同——写入内核按"源→目标"转换（TROUBLESHOOTING + TS-018）。
        bool source_is_half = false;
        // block table 的宽度（每个序列最多多少块）。
        // **必须等于引擎侧 block_tables 输入的第二维**，也就是
        // ceil(n_positions / block_size)——那条推导在 GPT2ModelBuilder 里；
        // 这里要求显式传入而不是自己猜，避免两处推导悄悄不一致。
        // **为什么物理池（num_blocks）可以比这个宽度大**（design.md D6 / §不变量 1）：
        // PagedAttention 插件用**运行期算出的偏移**在传进来的连续缓冲里寻址，而 TRT 不校验输入
        // 缓冲的实际容量——所以只要"池 ≥ 宽度"，插件的寻址就不会越过我们真正分配的那段。
        // 这条依据以前只写在设计文档里，落到这里是因为**约束的是本类的构造期校验**
        // （见构造函数里的 max_blocks_per_seq > num_blocks 检查）。
        int32_t max_blocks_per_seq = 0;
        // 同时存在的序列数上限（= `LLMRunner::Config::max_batch`）。
        // **元数据缓冲按它预分配**，所以必须与 runner 侧一致；超出时直接报错而不是扩容
        // ——扩容会让 `block_tables` 指针变化，破坏 design.md §不变量 5。
        // 默认 4：取项目 MVP 引擎 profile 的批上限（`core/builder.hpp` 的 max_prefill_batch /
        // max_decode_batch 默认就是 4）。**不能默认成 1**——那会让"登记第二条序列"直接失败，
        // 而多序列是这套分页 cache 的常态用法；多备几行的代价可忽略。
        int32_t max_batch = 4;
    };

    explicit PagedKVCache(const Config& config);
    ~PagedKVCache();

    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    bool valid() const { return valid_; }
    const Config& config() const { return config_; }

    // 为一个序列预留 ceil(max_tokens / block_size) 个物理块，并加入批内顺序尾部。
    // 为一个序列预留 `ceil(max_tokens / block_size)` 个物理块，并**追加到批内顺序尾部**。
    // 只填自己那一行：元数据缓冲已在构造期按 `max_batch` 备好，这里不重建、不重分配
    // （指针恒定，design.md §不变量 5）。批内顺序 = 引擎的行号，调用方必须按请求顺序登记。
    bool AllocateSequence(int32_t seq_id, int32_t max_tokens);
    // 释放该序列的块，并把批内顺序里它之后的行**整体前移、重建 host 镜像**。
    // 为什么内部就把镜像改对：只从 `order_` 移除的话，后面的行会错位而镜像不动，引擎会读到
    // 别的序列的块表——**静默算错**。这类错在 S1 被"下次登记会整体重建"掩盖着，批内退出（S3）
    // 一出现就会显形。设备侧由调用方随后 `UploadMetadata` 同步。
    void FreeSequence(int32_t seq_id);

    // 批内登记顺序：第 i 个序列对应引擎输入的第 i 行。
    const std::vector<int32_t>& sequence_order() const { return order_; }
    int32_t batch_size() const { return static_cast<int32_t>(order_.size()); }
    // 该序列当前在批内第几行（= 引擎输入的行号）；未登记时返回 -1。
    // 写回映射的来源。**为什么是显式查询**：靠"AllocateSequence 一定追加在尾部"推出
    // 行号是一种隐式约定，D6 的教训就是不把这种约定写进契约（p5_s3_interface_spec §3）。
    int32_t RowOf(int32_t seq_id) const;
    bool GetBlockTable(int32_t seq_id, std::vector<int32_t>* blocks) const;
    int32_t SequenceLength(int32_t seq_id) const;

    // 块池里还有多少块空闲。AC3 的断言依据；D9 的预算日志也用它。
    int32_t NumFreeBlocks() const;

    // 交给引擎的输入指针（设备地址）。按层返回，因为插件只看每层那 4-D 的一段。
    void* key_cache(int32_t layer);
    void* value_cache(int32_t layer);
    const int32_t* block_tables() const {
        return static_cast<const int32_t*>(block_tables_device_.data());
    }
    const int32_t* context_lens() const {
        return static_cast<const int32_t*>(context_lens_device_.data());
    }

    // 把 host 侧的 block table / context_lens 同步到设备。
    // 每个请求开始时调一次即可；decode 每步的推进在设备端完成。
    cudaError_t UploadMetadata(cudaStream_t stream);

    // 把 prefill 引擎输出的 K/V（[row_count, kv_heads, tokens, head_size]）写进 cache，
    // 并把**被映射行**的语境长度设为各自的 row_lengths[i]——host 与设备两侧一起更新。
    //
    // rows[i] = 引擎第 i 行 → 缓存批内第 rows[i] 行；row_count 是**本次参与 prefill 的行数**
    // （源缓冲里真正有效的行数，不是 cache 已登记的序列数）。
    // **契约**：rows / row_lengths 必须非空、row_count > 0，rows[i] 落在 [0, batch_size())，
    // row_lengths[i] 落在 (0, tokens]。
    // rows 是 host 数组；本函数把它拷进构造期备好的常驻设备缓冲，再交给 kernel
    // （本函数只在 prefill 段被调用，不违反"解码循环内不得有 H2D"）。
    //
    // **为什么必须带映射**：S3 的活跃批下，本步要 prefill 的序列只是缓存已登记序列的一个子集
    // （B_new < 活跃序列数），按"已登记序列数"逐行写就会拿源缓冲里上一轮的残留行去覆盖
    // **别的序列自己的** prompt K/V（静默算错，p5_s3_interface_spec §3）。
    // 静态批（S1/S2）批内顺序 == 请求顺序，调用方传恒等映射 rows[i] = i、row_count = batch，
    // 行为与改动前逐位相同（AC5）。
    //
    // **为什么 `tokens` 之外还要 row_lengths**（p5_s3_interface_spec §3）：`tokens` 是源张量的
    // token 轴长度，也就是每行写入的**位置数**（padding 路径下 = 本步最大 prompt 长度 S_step）；
    // 而语境长度必须停在每行**真实**长度上，否则 decode 会从填充位置起算、并把填充位置纳入注意力
    // （AC2 不成立）。S1/S2 批内等长 → row_lengths[i] == tokens，与原行为逐位相同。
    // 注意填充位置的 K/V 仍会被写进 cache（写入按 stride 走）：它们不在语境长度内，因此从不参与
    // 注意力、且会被后续 decode 覆盖；代价是块预留必须按 `tokens` 算而不是按真实长度。
    //
    // 设备侧那一步不能省：decode 追加的位置就是设备端 context_lens[rows[i]]；
    // 若只更新 host 镜像，后续追加会写回位置 0，**静默覆盖 prefill 的第一个 token**。
    // 因此由本函数负责把长度推上去，而不是要求调用方记得补一次 UploadMetadata。
    // （每个请求只发生一次，不违反"解码循环内不得有 H2D"。）
    cudaError_t WritePrefillKV(int32_t layer, const void* key, const void* value,
                              int32_t tokens, const int32_t* rows, int32_t row_count,
                              const int32_t* row_lengths, cudaStream_t stream);

    // 追加 decode 当前 token 的**某一层** K/V（[batch, kv_heads, 1, head_size]）。
    // 只负责写数据，**不推进语境长度**——长度是"每个 token 一个"的量，
    // 按层推进会把它算成 n_layer 倍（真机上表现为第 3 个 token 就发散，
    // 因为第 1、2 个 token 只用到 prefill 写好的长度）。
    // `row_count`：本次只写前几行（引擎行 = 批内行）；S3 的生成段只有活跃表前缀有本步的 K/V。
    cudaError_t AppendDecodeKV(int32_t layer, const void* key, const void* value,
                               int32_t row_count, cudaStream_t stream);

    // 追加一步 decode 的**所有层**，只追加**前 row_count 行**，写完后只推进这一批的语境长度。
    // runner 应当用这个入口：把"必须恰好推进一次"这件事收进 API，
    // 而不是留给调用方记得。
    // **为什么需要 row_count**：S3 的生成段只装本步在跑的序列（活跃表前缀），本步刚入批的
    // context 行还没算出 decode 的 K/V —— 给它们也追加会写进**它们自己的块**、并把它们的语境
    // 长度多推一格（静默算错，p5_s3_interface_spec §3）。静态批传 batch_size() 即原行为。
    cudaError_t AppendDecodeStep(const std::vector<const void*>& keys,
                                 const std::vector<const void*>& values, int32_t row_count,
                                 cudaStream_t stream);

    // 整块缓冲（含所有层）与单层的字节数。单层尺寸才是使用方需要的：
    // 引擎的 K/V 输入是"每层一段"的 4-D 张量。
    size_t key_cache_bytes() const { return key_cache_bytes_; }
    size_t value_cache_bytes() const { return value_cache_bytes_; }
    size_t bytes_per_layer() const { return layer_stride_bytes_; }

 private:
    struct Sequence {
        std::vector<int32_t> blocks;
        int32_t length = 0;
        int32_t reserved_tokens = 0;
    };

    // 每层起点：同一份大缓冲按层切片。
    void* LayerBase(DeviceBuffer* buffer, int32_t layer) const;

    Config config_;
    bool valid_ = false;

    BlockAllocator allocator_;
    std::map<int32_t, Sequence> sequences_;
    std::vector<int32_t> order_;

    // host 侧镜像，UploadMetadata 时整块拷到设备。
    std::vector<int32_t> block_tables_host_;
    std::vector<int32_t> context_lens_host_;
    DeviceBuffer block_tables_device_;
    DeviceBuffer context_lens_device_;
    // 写回行映射的设备暂存，按 max_batch 构造期备好（指针恒定，§不变量 5）。
    // rows_device_：调用方给的映射在本次调用内拷进来（prefill 段，不在解码循环里）；
    // identity_rows_device_：恒等表 [0, max_batch)，构造期推一次 —— decode 追加的行序就等于
    // 批内顺序，用它就不必在解码循环里传/拷映射（AGENTS.md §3.A.3）。
    DeviceBuffer rows_device_;
    DeviceBuffer identity_rows_device_;
    DeviceBuffer key_cache_device_;
    DeviceBuffer value_cache_device_;

    size_t key_cache_bytes_ = 0;
    size_t value_cache_bytes_ = 0;
    size_t layer_stride_bytes_ = 0;
};

}  // namespace mini_trt_llm
