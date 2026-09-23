#include "mini_trt_llm/kv_cache/block_allocator.hpp"
#include <stdexcept>

namespace mini_trt_llm {

BlockAllocator::BlockAllocator(size_t num_blocks) : allocated_(num_blocks, false) {
    for (int i = 0; i < static_cast<int>(num_blocks); ++i) {
        free_list_.push_back(i);
    }
}

int BlockAllocator::Allocate() {
    if (free_list_.empty()) {
        throw std::runtime_error("BlockAllocator out of memory");
    }
    int id = free_list_.front();
    free_list_.pop_front();
    allocated_[id] = true;
    return id;
}

void BlockAllocator::Free(int block_id) {
    if (block_id < 0 || block_id >= static_cast<int>(allocated_.size())) {
        throw std::runtime_error("BlockAllocator::Free invalid block id");
    }
    if (!allocated_[block_id]) {
        throw std::runtime_error("BlockAllocator::Free double free");
    }
    allocated_[block_id] = false;
    free_list_.push_back(block_id);
}

size_t BlockAllocator::NumFree() const { return free_list_.size(); }

}  // namespace mini_trt_llm
