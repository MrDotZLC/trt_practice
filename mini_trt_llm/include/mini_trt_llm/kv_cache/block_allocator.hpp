#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <vector>

namespace mini_trt_llm {

// KV Cache 块分配器。
// Phase 0 仅提供接口声明，Phase 1/2 实现。
class BlockAllocator {
 public:
    explicit BlockAllocator(size_t num_blocks);

    int Allocate();
    void Free(int block_id);
    size_t NumFree() const;

 private:
    std::vector<bool> allocated_;
    std::list<int> free_list_;
};

}  // namespace mini_trt_llm
