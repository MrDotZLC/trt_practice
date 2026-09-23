#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace mini_trt_llm {

// Phase 0：简单的 RAII 显存封装，后续迭代替换为池化实现。
class DeviceBuffer {
 public:
    DeviceBuffer();
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    bool Allocate(size_t bytes);
    void Free();
    void* data();
    const void* data() const;
    size_t size() const;

    // 若当前容量不足则重新分配
    bool Resize(size_t bytes);

 private:
    void* ptr_;
    size_t size_;
};

class PinnedBuffer {
 public:
    PinnedBuffer();
    explicit PinnedBuffer(size_t bytes);
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    bool Allocate(size_t bytes);
    void Free();
    void* data();
    const void* data() const;
    size_t size() const;

    bool Resize(size_t bytes);

 private:
    void* ptr_;
    size_t size_;
};

}  // namespace mini_trt_llm
