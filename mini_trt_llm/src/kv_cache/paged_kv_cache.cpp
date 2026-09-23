#include "mini_trt_llm/kv_cache/paged_kv_cache.hpp"
#include <stdexcept>

namespace mini_trt_llm {

PagedKVCache::PagedKVCache(int num_layers, int num_heads, int head_size,
                           int block_size, int max_blocks)
    : num_layers_(num_layers),
      num_heads_(num_heads),
      head_size_(head_size),
      block_size_(block_size),
      allocator_(max_blocks) {}

bool PagedKVCache::AllocateSequence(int seq_id, int init_len) {
    // Phase 2 实现
    (void)seq_id;
    (void)init_len;
    throw std::runtime_error(
        "PagedKVCache::AllocateSequence not implemented in Phase 0");
}

bool PagedKVCache::AppendToken(int seq_id, int token_id) {
    // Phase 2 实现
    (void)seq_id;
    (void)token_id;
    throw std::runtime_error(
        "PagedKVCache::AppendToken not implemented in Phase 0");
}

std::vector<int> PagedKVCache::GetBlockTable(int seq_id) const {
    // Phase 2 实现
    (void)seq_id;
    throw std::runtime_error(
        "PagedKVCache::GetBlockTable not implemented in Phase 0");
}

}  // namespace mini_trt_llm
