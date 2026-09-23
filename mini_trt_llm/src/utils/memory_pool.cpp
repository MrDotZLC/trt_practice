#include "mini_trt_llm/utils/memory_pool.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include <cstring>

namespace mini_trt_llm {

// ---------------------------------------------------------------------------
// DeviceBuffer 实现
// ---------------------------------------------------------------------------
DeviceBuffer::DeviceBuffer() : ptr_(nullptr), size_(0) {}

DeviceBuffer::DeviceBuffer(size_t bytes) : ptr_(nullptr), size_(0) {
    Allocate(bytes);
}

DeviceBuffer::~DeviceBuffer() { Free(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        Free();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool DeviceBuffer::Allocate(size_t bytes) {
    // 申请 0 字节时释放已有显存，避免悬空资源。
    if (bytes == 0) {
        Free();
        return true;
    }
    // 当前容量已满足需求时直接复用，减少 cudaMalloc 开销。
    if (ptr_ != nullptr && size_ >= bytes) {
        return true;
    }
    Free();
    cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
        ptr_ = nullptr;
        size_ = 0;
        return false;
    }
    size_ = bytes;
    return true;
}

void DeviceBuffer::Free() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
    size_ = 0;
}

void* DeviceBuffer::data() { return ptr_; }
const void* DeviceBuffer::data() const { return ptr_; }
size_t DeviceBuffer::size() const { return size_; }

bool DeviceBuffer::Resize(size_t bytes) {
    return Allocate(bytes);
}

// ---------------------------------------------------------------------------
// PinnedBuffer 实现
// ---------------------------------------------------------------------------
PinnedBuffer::PinnedBuffer() : ptr_(nullptr), size_(0) {}

PinnedBuffer::PinnedBuffer(size_t bytes) : ptr_(nullptr), size_(0) {
    Allocate(bytes);
}

PinnedBuffer::~PinnedBuffer() { Free(); }

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        Free();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool PinnedBuffer::Allocate(size_t bytes) {
    if (bytes == 0) {
        Free();
        return true;
    }
    if (ptr_ != nullptr && size_ >= bytes) {
        return true;
    }
    Free();
    cudaError_t err = cudaMallocHost(&ptr_, bytes);
    if (err != cudaSuccess) {
        ptr_ = nullptr;
        size_ = 0;
        return false;
    }
    size_ = bytes;
    return true;
}

void PinnedBuffer::Free() {
    if (ptr_) {
        cudaFreeHost(ptr_);
        ptr_ = nullptr;
    }
    size_ = 0;
}

void* PinnedBuffer::data() { return ptr_; }
const void* PinnedBuffer::data() const { return ptr_; }
size_t PinnedBuffer::size() const { return size_; }

bool PinnedBuffer::Resize(size_t bytes) {
    return Allocate(bytes);
}

}  // namespace mini_trt_llm
