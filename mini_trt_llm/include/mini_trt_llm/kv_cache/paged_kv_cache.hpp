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
        // 与 cache 精度不一定相同——写入内核按"源→目标"转换（TROUBLESHOOTING #18）。
        bool source_is_half = false;
        // block table 的宽度（每个序列最多多少块）。
        // **必须等于引擎侧 block_tables 输入的第二维**，也就是
        // ceil(n_positions / block_size)——那条推导在 GPT2ModelBuilder 里；
        // 这里要求显式传入而不是自己猜，避免两处推导悄悄不一致。
        int32_t max_blocks_per_seq = 0;
    };

    explicit PagedKVCache(const Config& config);
    ~PagedKVCache();

    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    bool valid() const { return valid_; }
    const Config& config() const { return config_; }

    // 为一个序列预留 ceil(max_tokens / block_size) 个物理块，并加入批内顺序尾部。
    bool AllocateSequence(int32_t seq_id, int32_t max_tokens);
    void FreeSequence(int32_t seq_id);

    // 批内登记顺序：第 i 个序列对应引擎输入的第 i 行。
    const std::vector<int32_t>& sequence_order() const { return order_; }
    int32_t batch_size() const { return static_cast<int32_t>(order_.size()); }
    bool GetBlockTable(int32_t seq_id, std::vector<int32_t>* blocks) const;
    int32_t SequenceLength(int32_t seq_id) const;

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

    // 把 prefill 引擎输出的 K/V（[batch, kv_heads, tokens, head_size]）写进 cache，
    // 并把语境长度设为 tokens——host 与设备两侧一起更新。
    //
    // 设备侧那一步不能省：decode 追加的位置就是设备端 context_lens[b]；
    // 若只更新 host 镜像，后续追加会写回位置 0，**静默覆盖 prefill 的第一个 token**。
    // 因此由本函数负责把长度推上去，而不是要求调用方记得补一次 UploadMetadata。
    // （每个请求只发生一次，不违反"解码循环内不得有 H2D"。）
    cudaError_t WritePrefillKV(int32_t layer, const void* key, const void* value,
                              int32_t tokens, cudaStream_t stream);

    // 追加 decode 当前 token 的**某一层** K/V（[batch, kv_heads, 1, head_size]）。
    // 只负责写数据，**不推进语境长度**——长度是"每个 token 一个"的量，
    // 按层推进会把它算成 n_layer 倍（真机上表现为第 3 个 token 就发散，
    // 因为第 1、2 个 token 只用到 prefill 写好的长度）。
    cudaError_t AppendDecodeKV(int32_t layer, const void* key, const void* value,
                              cudaStream_t stream);

    // 追加一步 decode 的**所有层**，写完后只推进一次语境长度。
    // runner 应当用这个入口：把"必须恰好推进一次"这件事收进 API，
    // 而不是留给调用方记得。
    cudaError_t AppendDecodeStep(const std::vector<const void*>& keys,
                                 const std::vector<const void*>& values,
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
    DeviceBuffer key_cache_device_;
    DeviceBuffer value_cache_device_;

    size_t key_cache_bytes_ = 0;
    size_t value_cache_bytes_ = 0;
    size_t layer_stride_bytes_ = 0;
};

}  // namespace mini_trt_llm
