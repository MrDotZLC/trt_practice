#include "mini_trt_llm/kv_cache/paged_kv_cache.hpp"

#include "mini_trt_llm/kv_cache/paged_kv_cache_kernels.hpp"
#include "mini_trt_llm/utils/logger.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace mini_trt_llm {
namespace {

size_t ElementSize(bool is_half) { return is_half ? 2u : 4u; }

// 需要多少块才能装下 tokens 个 token。
int32_t BlocksForTokens(int32_t tokens, int32_t block_size) {
    return block_size > 0 ? (tokens + block_size - 1) / block_size : 0;
}

}  // namespace

PagedKVCache::PagedKVCache(const Config& config)
    : config_(config), allocator_(static_cast<size_t>(config.num_blocks > 0
                                                         ? config.num_blocks
                                                         : 0)) {
    if (config_.num_blocks <= 0 || config_.block_size <= 0 || config_.num_layers <= 0 ||
        config_.num_kv_heads <= 0 || config_.head_size <= 0 ||
        config_.max_blocks_per_seq <= 0 || config_.max_batch <= 0) {
        MINI_TRT_LOG_ERROR("PagedKVCache: invalid config");
        return;
    }
    if (config_.max_blocks_per_seq > config_.num_blocks) {
        // 单个序列能占满整个池子还算合理，但宽度超过池子说明两个参数来自不同的假设
        MINI_TRT_LOG_ERROR("PagedKVCache: max_blocks_per_seq ("
                           << config_.max_blocks_per_seq << ") exceeds num_blocks ("
                           << config_.num_blocks << ")");
        return;
    }

    const size_t element_size = ElementSize(config_.is_half);
    layer_stride_bytes_ = static_cast<size_t>(config_.num_blocks) *
                          config_.block_size * config_.num_kv_heads * config_.head_size *
                          element_size;
    key_cache_bytes_ = layer_stride_bytes_ * static_cast<size_t>(config_.num_layers);
    value_cache_bytes_ = key_cache_bytes_;

    // 元数据缓冲**在构造期按 max_batch 一次备好**：此后跨请求、跨批大小都不再重分配，
    // 指针恒定（design.md §不变量 5）。
    // 注意 `DeviceBuffer::Allocate(0)` 的语义是**释放**，所以这里必须是正尺寸——
    // max_batch ≥ 1 已在上面的校验里保证。
    block_tables_host_.assign(static_cast<size_t>(config_.max_batch) *
                                  static_cast<size_t>(config_.max_blocks_per_seq), 0);
    context_lens_host_.assign(static_cast<size_t>(config_.max_batch), 0);

    if (!block_tables_device_.Allocate(block_tables_host_.size() * sizeof(int32_t)) ||
        !context_lens_device_.Allocate(context_lens_host_.size() * sizeof(int32_t))) {
        MINI_TRT_LOG_ERROR("PagedKVCache: failed to pre-allocate metadata buffers");
        return;
    }
    // 写回行映射的暂存：同样按 max_batch 预分配（§不变量 5）。
    // 恒等表在这里就推到设备——decode 每步都要用它，放到解码循环里拷会引入 H2D（§不变量 3）。
    const size_t rows_bytes = static_cast<size_t>(config_.max_batch) * sizeof(int32_t);
    if (!rows_device_.Allocate(rows_bytes) || !identity_rows_device_.Allocate(rows_bytes) ||
        !row_starts_device_.Allocate(rows_bytes)) {
        MINI_TRT_LOG_ERROR("PagedKVCache: failed to pre-allocate row-mapping buffers");
        return;
    }
    std::vector<int32_t> identity_rows(static_cast<size_t>(config_.max_batch));
    for (int32_t i = 0; i < config_.max_batch; ++i) {
        identity_rows[static_cast<size_t>(i)] = i;
    }
    if (cudaMemcpy(identity_rows_device_.data(), identity_rows.data(), rows_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        MINI_TRT_LOG_ERROR("PagedKVCache: failed to upload the identity row mapping");
        return;
    }
    if (!key_cache_device_.Allocate(key_cache_bytes_) ||
        !value_cache_device_.Allocate(value_cache_bytes_)) {
        MINI_TRT_LOG_ERROR("PagedKVCache: failed to allocate cache buffers ("
                           << key_cache_bytes_ / (1024 * 1024) << " MB per side)");
        return;
    }
    valid_ = true;
}

PagedKVCache::~PagedKVCache() = default;

void* PagedKVCache::LayerBase(DeviceBuffer* buffer, int32_t layer) const {
    if (layer < 0 || layer >= config_.num_layers) {
        return nullptr;
    }
    return static_cast<char*>(buffer->data()) +
           static_cast<size_t>(layer) * layer_stride_bytes_;
}

void* PagedKVCache::key_cache(int32_t layer) { return LayerBase(&key_cache_device_, layer); }

void* PagedKVCache::value_cache(int32_t layer) {
    return LayerBase(&value_cache_device_, layer);
}

bool PagedKVCache::AllocateSequence(int32_t seq_id, int32_t max_tokens) {
    if (!valid_ || max_tokens <= 0 || sequences_.count(seq_id) != 0) {
        return false;
    }
    // 元数据缓冲按 max_batch 预分配，没有第 max_batch 行可以放 —— 直接拒绝，不扩容
    // （扩容会让 block_tables 指针变化，破坏 §不变量 5）。
    if (static_cast<int32_t>(order_.size()) >= config_.max_batch) {
        MINI_TRT_LOG_ERROR("PagedKVCache: batch limit reached (" << config_.max_batch
                           << " sequences already registered)");
        return false;
    }
    const int32_t blocks_needed = BlocksForTokens(max_tokens, config_.block_size);
    if (blocks_needed > config_.max_blocks_per_seq ||
        allocator_.NumFree() < static_cast<size_t>(blocks_needed)) {
        MINI_TRT_LOG_ERROR("PagedKVCache: cannot reserve " << max_tokens
                                                          << " tokens for seq " << seq_id
                                                          << " (需要 " << blocks_needed
                                                          << " 块，空闲 "
                                                          << allocator_.NumFree() << " 块)");
        return false;
    }
    Sequence sequence;
    sequence.blocks.reserve(static_cast<size_t>(blocks_needed));
    for (int32_t i = 0; i < blocks_needed; ++i) {
        sequence.blocks.push_back(allocator_.Allocate());
    }
    sequence.length = 0;
    sequence.reserved_tokens = max_tokens;
    sequences_.emplace(seq_id, std::move(sequence));
    order_.push_back(seq_id);

    // 只填自己那一行：缓冲在构造期已按 max_batch 备好，这里既不重建也不重分配。
    const size_t row = order_.size() - 1;
    const size_t width = static_cast<size_t>(config_.max_blocks_per_seq);
    const Sequence& stored = sequences_.at(seq_id);
    // 整行都写：该行可能被上一轮的序列用过，尾部残留必须清掉（块表宽度是引擎契约的一部分）。
    for (size_t i = 0; i < width; ++i) {
        block_tables_host_[row * width + i] =
            i < stored.blocks.size() ? stored.blocks[i] : 0;
    }
    context_lens_host_[row] = stored.length;
    return true;
}

void PagedKVCache::FreeSequence(int32_t seq_id) {
    const auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) {
        return;
    }
    for (int32_t block : it->second.blocks) {
        allocator_.Free(block);
    }
    sequences_.erase(it);
    order_.erase(std::remove(order_.begin(), order_.end(), seq_id), order_.end());

    // 把幸存的行整体前移、重建 host 镜像。只从 order_ 移除是不够的：order_ 是引擎的行号，
    // 后面的行会错位而镜像不动 —— 引擎会读到**别的序列**的块表，静默算错（§不变量 4/5）。
    // 这类错在 S1 被"下次登记会整体重建"掩盖着，批内退出（S3）一出现就会显形。
    // 设备侧由调用方随后 UploadMetadata 同步。
    const size_t width = static_cast<size_t>(config_.max_blocks_per_seq);
    for (size_t b = 0; b < order_.size(); ++b) {
        const Sequence& seq = sequences_.at(order_[b]);
        for (size_t i = 0; i < width; ++i) {
            block_tables_host_[b * width + i] = i < seq.blocks.size() ? seq.blocks[i] : 0;
        }
        context_lens_host_[b] = seq.length;
    }
}

int32_t PagedKVCache::RowOf(int32_t seq_id) const {
    for (size_t b = 0; b < order_.size(); ++b) {
        if (order_[b] == seq_id) {
            return static_cast<int32_t>(b);
        }
    }
    return -1;
}

bool PagedKVCache::GetBlockTable(int32_t seq_id, std::vector<int32_t>* blocks) const {
    const auto it = sequences_.find(seq_id);
    if (it == sequences_.end() || blocks == nullptr) {
        return false;
    }
    *blocks = it->second.blocks;
    return true;
}

int32_t PagedKVCache::SequenceLength(int32_t seq_id) const {
    const auto it = sequences_.find(seq_id);
    return it == sequences_.end() ? -1 : it->second.length;
}

int32_t PagedKVCache::NumFreeBlocks() const {
    return static_cast<int32_t>(allocator_.NumFree());
}

cudaError_t PagedKVCache::UploadMetadata(cudaStream_t stream) {
    if (!valid_) {
        return cudaErrorInvalidValue;
    }
    if (!block_tables_host_.empty()) {
        const cudaError_t err = cudaMemcpyAsync(
            block_tables_device_.data(), block_tables_host_.data(),
            block_tables_host_.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (!context_lens_host_.empty()) {
        const cudaError_t err = cudaMemcpyAsync(
            context_lens_device_.data(), context_lens_host_.data(),
            context_lens_host_.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return err;
        }
    }
    return cudaSuccess;
}

cudaError_t PagedKVCache::WritePrefillKV(int32_t layer, const void* key, const void* value,
                                         int32_t tokens, const int32_t* rows,
                                         int32_t row_count, const int32_t* row_lengths,
                                         cudaStream_t stream,
                                         const int32_t* cu_seqlens_ctx,
                                         int32_t context_seq_count,
                                         const int32_t* row_starts) {
    if (!valid_ || key == nullptr || value == nullptr || tokens <= 0 || rows == nullptr ||
        row_count <= 0 || row_lengths == nullptr) {
        return cudaErrorInvalidValue;
    }
    void* cache_key = key_cache(layer);
    void* cache_value = value_cache(layer);
    if (cache_key == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int32_t batch = batch_size();
    // row_count 是**本次参与写入的行数**，它只能小于等于批内已登记的行数：
    // 反过来意味着调用方给的映射指向了不存在的行，写进去就是踩别的序列。
    if (batch <= 0 || row_count > batch) {
        MINI_TRT_LOG_ERROR("PagedKVCache: row_count " << row_count
                           << " does not fit the registered batch " << batch);
        return cudaErrorInvalidValue;
    }
    // S4：源是 packed 张量时，每行的长度各不相同（由 `row_lengths` 给出、`cu_seqlens_ctx` 同源），
    // 所以"统一 stride"的块数检查不适用；下面的逐行检查改用 `row_lengths[i]`。
    // `tokens` 在这条路径上只是一个形状占位（调用方传**最大行长**），不参与校验。
    const bool packed_source = (cu_seqlens_ctx != nullptr);
    if (!packed_source) {
        const int32_t blocks_needed = BlocksForTokens(tokens, config_.block_size);
        if (blocks_needed > config_.max_blocks_per_seq) {
            MINI_TRT_LOG_ERROR("PagedKVCache: prefill length " << tokens
                                                              << " exceeds reserved blocks");
            return cudaErrorInvalidValue;
        }
    }
    // 逐行核对：① 行号必须落在已登记范围内（越界会寻址到墙外/别的行），
    // ② 真实长度必须落在 (0, tokens]（越界会把语境长度写成未来位置/0），
    // ③ 被映射行的预留量必须够——块表里没被预留的位置在 host 镜像里是 0，
    // 越界写入会**静默写进物理块 0**（通常是别的序列的数据），必须在入口拦住。
    // 预留量按 `tokens`（写入的 stride）比，不是按真实长度：填充位置也会被写进 cache。
    // 只检查被映射的行：未参与本次写入的行一个字节都不该被碰。
    for (int32_t i = 0; i < row_count; ++i) {
        const int32_t row = rows[i];
        if (row < 0 || row >= batch) {
            MINI_TRT_LOG_ERROR("PagedKVCache: row mapping out of range at " << i << " (row "
                               << row << ", registered batch " << batch << ")");
            return cudaErrorInvalidValue;
        }
        if (row_lengths[i] <= 0 || row_lengths[i] > tokens) {
            MINI_TRT_LOG_ERROR("PagedKVCache: row length " << row_lengths[i] << " at " << i
                               << " is out of (0, " << tokens << "]");
            return cudaErrorInvalidValue;
        }
        const Sequence& sequence = sequences_.at(order_[static_cast<size_t>(row)]);
        const int32_t length = packed_source ? row_lengths[i] : tokens;
        // S5：写了起点之后，真正要保证的是"这一段写完不越过预留"（累计口径）。
        const int32_t end = (row_starts != nullptr) ? row_starts[i] + length : length;
        if (end > sequence.reserved_tokens) {
            MINI_TRT_LOG_ERROR("PagedKVCache: prefill length "
                               << end << " exceeds reserved "
                               << sequence.reserved_tokens << " tokens of seq "
                               << order_[static_cast<size_t>(row)]);
            return cudaErrorInvalidValue;
        }
    }
    // 映射必须落到设备侧：kernel 按 rows[b] 寻址。缓冲在构造期备好，这里只拷 row_count 个 int32。
    if (cudaMemcpyAsync(rows_device_.data(), rows,
                        static_cast<size_t>(row_count) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return cudaErrorInvalidValue;
    }

    PagedKVWriteArgs args;
    args.key = key;
    args.value = value;
    args.key_cache = cache_key;
    args.value_cache = cache_value;
    args.block_tables = block_tables();
    args.context_lens = const_cast<int32_t*>(context_lens());
    args.rows = static_cast<const int32_t*>(rows_device_.data());
    args.row_count = row_count;
    args.tokens = tokens;
    args.num_kv_heads = config_.num_kv_heads;
    args.head_size = config_.head_size;
    args.block_size = config_.block_size;
    args.max_blocks_per_seq = config_.max_blocks_per_seq;
    args.is_half = config_.is_half;
    args.source_is_half = config_.source_is_half;
    args.append = false;
    // S4：源是 packed 张量时，源行基址由 kernel 从设备端读（见 PagedKVWriteArgs 的说明）
    args.cu_seqlens_ctx = cu_seqlens_ctx;
    args.context_seq_count = context_seq_count;
    // S5：写回起点（host → device）。nullptr = 从 0 覆盖写（S3/S4 的行为）。
    if (row_starts != nullptr) {
        if (cudaMemcpyAsync(row_starts_device_.data(), row_starts,
                            static_cast<size_t>(row_count) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return cudaErrorInvalidValue;
        }
        args.row_starts = static_cast<const int32_t*>(row_starts_device_.data());
    }

    const cudaError_t err = LaunchWriteKV(args, stream);
    if (err != cudaSuccess) {
        return err;
    }
    // host 侧记账与设备侧保持一致：写完之后**被映射行**的有效长度是它自己的真实长度。
    // 为什么不是 `tokens`：`tokens` 是写入的 stride，padding 路径下它会大于短行的真实长度，
    // 把语境长度写成 stride 会让 decode 从填充位置起算、并把填充位置纳入注意力。
    // 只记这几行：未参与本次写入的行长度不变，否则会被写上不属于它的长度，
    // 后续 UploadMetadata 把错长度推回设备 —— 与"写回覆盖"是同一类静默错。
    for (int32_t i = 0; i < row_count; ++i) {
        const size_t row = static_cast<size_t>(rows[i]);
        // S5：给了写回起点就是**累加**（分块 prefill 每块只写自己那一段）；没给仍是赋值。
        const int32_t written =
            (row_starts != nullptr) ? row_starts[i] + row_lengths[i] : row_lengths[i];
        context_lens_host_[row] = written;
        sequences_[order_[row]].length = written;
    }
    // 立刻把长度推到设备：decode 追加的位置取自设备端 context_lens，
    // 漏掉这一步的话追加会写回位置 0（静默覆盖第一个 token）。
    // 每个请求只发生一次，因此不违反"解码循环内不得有 H2D 拷贝"。
    return cudaMemcpyAsync(context_lens_device_.data(), context_lens_host_.data(),
                           context_lens_host_.size() * sizeof(int32_t),
                           cudaMemcpyHostToDevice, stream);
}

cudaError_t PagedKVCache::AppendDecodeKV(int32_t layer, const void* key, const void* value,
                                         int32_t row_count, cudaStream_t stream,
                                         const int32_t* cu_seqlens_ctx,
                                         int32_t context_seq_count, const int32_t* rows) {
    if (!valid_ || key == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    void* cache_key = key_cache(layer);
    void* cache_value = value_cache(layer);
    if (cache_key == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int32_t batch = batch_size();
    // 只追加前 row_count 行：S3 的生成段只有活跃表前缀有本步的 K/V；越界直接拒绝，
    // 不静默截断（静默截断会让"少写了谁"变成无人知晓的错）。
    if (batch <= 0 || row_count <= 0 || row_count > batch) {
        return cudaErrorInvalidValue;
    }
    // S5：显式行列表（nullptr = 恒等，S1/S2/S3 的行为）。行号必须落在已登记范围内 ——
    // 越界寻址会写到别的序列的块里（静默错），必须在入口拦住。
    if (rows != nullptr) {
        for (int32_t i = 0; i < row_count; ++i) {
            if (rows[i] < 0 || rows[i] >= batch) {
                MINI_TRT_LOG_ERROR("PagedKVCache: append row mapping out of range at " << i
                                   << " (row " << rows[i] << ", registered batch " << batch << ")");
                return cudaErrorInvalidValue;
            }
        }
    }

    PagedKVWriteArgs args;
    args.key = key;
    args.value = value;
    args.key_cache = cache_key;
    args.value_cache = cache_value;
    args.block_tables = block_tables();
    args.context_lens = const_cast<int32_t*>(context_lens());
    // decode 的行序就等于批内顺序（= order_），所以用构造期备好的恒等表：
    // 这里每步都会被调用，另拷一份映射会让解码循环里出现 H2D（AGENTS.md §3.A.3）。
    // 行映射：给了显式列表就用它（S5 分块后"本步追加的行"不再是活跃前缀）；否则走构造期备好的恒等表。
    // **`rows` 是 host 指针，而 kernel 按设备端寻址** —— 给了显式列表就必须先拷进
    // 构造期备好的 `rows_device_`（容量按 max_batch，与 prefill 路径同一个缓冲；
    // 两步之间同 stream 串行，所以复用它不会与 prefill 的那次拷贝打架）。
    if (rows != nullptr) {
        if (cudaMemcpyAsync(rows_device_.data(), rows,
                            static_cast<size_t>(row_count) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return cudaErrorInvalidValue;
        }
        args.rows = static_cast<const int32_t*>(rows_device_.data());
    } else {
        args.rows = static_cast<const int32_t*>(identity_rows_device_.data());
    }
    args.row_count = row_count;
    args.tokens = 1;
    args.num_kv_heads = config_.num_kv_heads;
    args.head_size = config_.head_size;
    args.block_size = config_.block_size;
    args.max_blocks_per_seq = config_.max_blocks_per_seq;
    args.is_half = config_.is_half;
    args.source_is_half = config_.source_is_half;
    args.append = true;
    // S4：generation 段的源在 packed 张量里从 `cu_seqlens_ctx[B_ctx]` 起（每行 1 个 token、连续）
    args.cu_seqlens_ctx = cu_seqlens_ctx;
    args.context_seq_count = context_seq_count;

    return LaunchWriteKV(args, stream);
}

cudaError_t PagedKVCache::AppendDecodeStep(const std::vector<const void*>& keys,
                                           const std::vector<const void*>& values,
                                           int32_t row_count,
                                           cudaStream_t stream,
                                           const int32_t* cu_seqlens_ctx,
                                           int32_t context_seq_count, const int32_t* rows) {
    if (keys.size() != values.size() ||
        keys.size() != static_cast<size_t>(config_.num_layers)) {
        MINI_TRT_LOG_ERROR("PagedKVCache: AppendDecodeStep expects one K/V pair per layer");
        return cudaErrorInvalidValue;
    }
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const cudaError_t err = AppendDecodeKV(layer, keys[static_cast<size_t>(layer)],
                                              values[static_cast<size_t>(layer)], row_count,
                                              stream, cu_seqlens_ctx, context_seq_count, rows);
        if (err != cudaSuccess) {
            return err;
        }
    }
    // 推进必须发生在所有写入之后，所以独立成一个 kernel（同一 stream 上串行）；
    // 且**只推进这一批（前 row_count 行）一次**。
    // **`rows` 是 host 数组**（契约见 hpp），而 kernel 要按**设备地址**读同一份映射 ——
    // `AppendDecodeKV` 刚把它拷进 `rows_device_`（同 stream、在本 kernel 之前），这里直接复用；
    // 传 host 指针会让内核对 host 地址做设备解引用（`TS-052` 发现 1：非法访存，或按垃圾行号推进长度）。
    const int32_t* rows_device =
        (rows != nullptr) ? static_cast<const int32_t*>(rows_device_.data()) : nullptr;
    const cudaError_t err = LaunchAdvanceContextLens(
        const_cast<int32_t*>(context_lens()), row_count, /*tokens=*/1, stream, rows_device);
    if (err != cudaSuccess) {
        return err;
    }
    for (int32_t b = 0; b < row_count; ++b) {
        // 行映射必须与设备侧一致：给了显式列表就按它推进（S5），否则恒等（S3/S4）。
        const int32_t row = (rows != nullptr) ? rows[b] : b;
        context_lens_host_[static_cast<size_t>(row)] += 1;
        sequences_[order_[static_cast<size_t>(row)]].length += 1;
    }
    return cudaSuccess;
}

}  // namespace mini_trt_llm
