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
        config_.max_blocks_per_seq <= 0) {
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

    block_tables_host_.clear();
    context_lens_host_.clear();

    if (!block_tables_device_.Allocate(0) || !context_lens_device_.Allocate(0)) {
        MINI_TRT_LOG_ERROR("PagedKVCache: failed to allocate metadata buffers");
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
    const int32_t blocks_needed = BlocksForTokens(max_tokens, config_.block_size);
    if (blocks_needed > config_.max_blocks_per_seq ||
        allocator_.NumFree() < static_cast<size_t>(blocks_needed)) {
        MINI_TRT_LOG_ERROR("PagedKVCache: cannot reserve " << max_tokens
                                                          << " tokens for seq " << seq_id);
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

    // 元数据缓冲按当前批大小重建（序列数很少变化，重分配比"取最大宽度"更直观）。
    const size_t batch = order_.size();
    block_tables_host_.assign(batch * static_cast<size_t>(config_.max_blocks_per_seq), 0);
    context_lens_host_.assign(batch, 0);
    if (!block_tables_device_.Allocate(block_tables_host_.size() * sizeof(int32_t)) ||
        !context_lens_device_.Allocate(context_lens_host_.size() * sizeof(int32_t))) {
        MINI_TRT_LOG_ERROR("PagedKVCache: failed to grow metadata buffers");
        valid_ = false;
        return false;
    }

    // 把每个已登记序列的块表填进 host 镜像
    for (size_t b = 0; b < order_.size(); ++b) {
        const auto it = sequences_.find(order_[b]);
        if (it == sequences_.end()) {
            continue;
        }
        for (size_t i = 0; i < it->second.blocks.size(); ++i) {
            block_tables_host_[b * static_cast<size_t>(config_.max_blocks_per_seq) + i] =
                it->second.blocks[i];
        }
        context_lens_host_[b] = it->second.length;
    }
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
                                         int32_t tokens, cudaStream_t stream) {
    if (!valid_ || key == nullptr || value == nullptr || tokens <= 0) {
        return cudaErrorInvalidValue;
    }
    void* cache_key = key_cache(layer);
    void* cache_value = value_cache(layer);
    if (cache_key == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int32_t batch = batch_size();
    if (batch <= 0) {
        return cudaErrorInvalidValue;
    }
    const int32_t blocks_needed = BlocksForTokens(tokens, config_.block_size);
    if (blocks_needed > config_.max_blocks_per_seq) {
        MINI_TRT_LOG_ERROR("PagedKVCache: prefill length " << tokens
                                                          << " exceeds reserved blocks");
        return cudaErrorInvalidValue;
    }
    // 逐序列核对预留量：块表里没被预留的位置在 host 镜像里是 0，
    // 越界写入会**静默写进物理块 0**（通常是别的序列的数据），必须在入口拦住。
    for (int32_t seq_id : order_) {
        const Sequence& sequence = sequences_.at(seq_id);
        if (tokens > sequence.reserved_tokens) {
            MINI_TRT_LOG_ERROR("PagedKVCache: prefill length "
                               << tokens << " exceeds reserved "
                               << sequence.reserved_tokens << " tokens of seq " << seq_id);
            return cudaErrorInvalidValue;
        }
    }

    PagedKVWriteArgs args;
    args.key = key;
    args.value = value;
    args.key_cache = cache_key;
    args.value_cache = cache_value;
    args.block_tables = block_tables();
    args.context_lens = const_cast<int32_t*>(context_lens());
    args.batch_size = batch;
    args.tokens = tokens;
    args.num_kv_heads = config_.num_kv_heads;
    args.head_size = config_.head_size;
    args.block_size = config_.block_size;
    args.max_blocks_per_seq = config_.max_blocks_per_seq;
    args.is_half = config_.is_half;
    args.source_is_half = config_.source_is_half;
    args.append = false;

    const cudaError_t err = LaunchWriteKV(args, stream);
    if (err != cudaSuccess) {
        return err;
    }
    // host 侧记账与设备侧保持一致：写完之后每个序列的有效长度就是 tokens。
    for (size_t b = 0; b < order_.size(); ++b) {
        context_lens_host_[b] = tokens;
        sequences_[order_[b]].length = tokens;
    }
    // 立刻把长度推到设备：decode 追加的位置取自设备端 context_lens，
    // 漏掉这一步的话追加会写回位置 0（静默覆盖第一个 token）。
    // 每个请求只发生一次，因此不违反"解码循环内不得有 H2D 拷贝"。
    return cudaMemcpyAsync(context_lens_device_.data(), context_lens_host_.data(),
                           context_lens_host_.size() * sizeof(int32_t),
                           cudaMemcpyHostToDevice, stream);
}

cudaError_t PagedKVCache::AppendDecodeKV(int32_t layer, const void* key, const void* value,
                                         cudaStream_t stream) {
    if (!valid_ || key == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    void* cache_key = key_cache(layer);
    void* cache_value = value_cache(layer);
    if (cache_key == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int32_t batch = batch_size();
    if (batch <= 0) {
        return cudaErrorInvalidValue;
    }

    PagedKVWriteArgs args;
    args.key = key;
    args.value = value;
    args.key_cache = cache_key;
    args.value_cache = cache_value;
    args.block_tables = block_tables();
    args.context_lens = const_cast<int32_t*>(context_lens());
    args.batch_size = batch;
    args.tokens = 1;
    args.num_kv_heads = config_.num_kv_heads;
    args.head_size = config_.head_size;
    args.block_size = config_.block_size;
    args.max_blocks_per_seq = config_.max_blocks_per_seq;
    args.is_half = config_.is_half;
    args.source_is_half = config_.source_is_half;
    args.append = true;

    return LaunchWriteKV(args, stream);
}

cudaError_t PagedKVCache::AppendDecodeStep(const std::vector<const void*>& keys,
                                           const std::vector<const void*>& values,
                                           cudaStream_t stream) {
    if (keys.size() != values.size() ||
        keys.size() != static_cast<size_t>(config_.num_layers)) {
        MINI_TRT_LOG_ERROR("PagedKVCache: AppendDecodeStep expects one K/V pair per layer");
        return cudaErrorInvalidValue;
    }
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const cudaError_t err = AppendDecodeKV(layer, keys[static_cast<size_t>(layer)],
                                              values[static_cast<size_t>(layer)], stream);
        if (err != cudaSuccess) {
            return err;
        }
    }
    // 推进必须发生在所有写入之后，所以独立成一个 kernel（同一 stream 上串行）；
    // 且**整步只推进一次**。
    const cudaError_t err = LaunchAdvanceContextLens(
        const_cast<int32_t*>(context_lens()), batch_size(), /*tokens=*/1, stream);
    if (err != cudaSuccess) {
        return err;
    }
    for (size_t b = 0; b < order_.size(); ++b) {
        context_lens_host_[b] += 1;
        sequences_[order_[b]].length += 1;
    }
    return cudaSuccess;
}

}  // namespace mini_trt_llm
