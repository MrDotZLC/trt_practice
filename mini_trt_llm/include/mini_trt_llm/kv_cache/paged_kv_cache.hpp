#pragma once

#include "mini_trt_llm/kv_cache/block_allocator.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mini_trt_llm {

// Paged KV Cache 管理器。
// Phase 0 仅提供接口声明，Phase 1/2 实现。
class PagedKVCache {
 public:
    PagedKVCache(int num_layers, int num_heads, int head_size,
                 int block_size, int max_blocks);

    bool AllocateSequence(int seq_id, int init_len);
    bool AppendToken(int seq_id, int token_id);
    std::vector<int> GetBlockTable(int seq_id) const;

 private:
    int num_layers_;
    int num_heads_;
    int head_size_;
    int block_size_;
    BlockAllocator allocator_;
};

}  // namespace mini_trt_llm
