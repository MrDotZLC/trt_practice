#include "mini_trt_llm/utils/safetensors_loader.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <algorithm>
#include <cstring>

namespace mini_trt_llm {

size_t SafetensorsDtypeSize(safetensors::dtype dtype) {
    // 按 Safetensors dtype 返回单个元素字节数；未命中类型返回 0 由调用方处理。
    switch (dtype) {
        case safetensors::kBOOL:
            return 1;
        case safetensors::kUINT8:
        case safetensors::kINT8:
            return 1;
        case safetensors::kINT16:
        case safetensors::kUINT16:
        case safetensors::kFLOAT16:
        case safetensors::kBFLOAT16:
            return 2;
        case safetensors::kINT32:
        case safetensors::kUINT32:
        case safetensors::kFLOAT32:
            return 4;
        case safetensors::kFLOAT64:
        case safetensors::kINT64:
        case safetensors::kUINT64:
            return 8;
    }
    return 0;
}

nvinfer1::DataType SafetensorsToTrtDtype(safetensors::dtype dtype) {
    switch (dtype) {
        case safetensors::kFLOAT32:
            return nvinfer1::DataType::kFLOAT;
        case safetensors::kFLOAT16:
            return nvinfer1::DataType::kHALF;
        case safetensors::kINT8:
            return nvinfer1::DataType::kINT8;
        case safetensors::kINT32:
            return nvinfer1::DataType::kINT32;
        case safetensors::kBOOL:
            return nvinfer1::DataType::kBOOL;
        case safetensors::kUINT8:
            return nvinfer1::DataType::kUINT8;
        default:
            MINI_TRT_LOG_WARN("Unsupported safetensors dtype " << dtype
                           << ", fallback to kFLOAT");
            return nvinfer1::DataType::kFLOAT;
    }
}

SafetensorsLoader::SafetensorsLoader() = default;

SafetensorsLoader::~SafetensorsLoader() = default;

bool SafetensorsLoader::LoadFromFile(const std::string& path) {
    std::string warn;
    std::string err;
    if (!safetensors::load_from_file(path, &st_, &warn, &err)) {
        MINI_TRT_LOG_ERROR("Failed to load safetensors: " << path
                         << ", error: " << err);
        return false;
    }
    if (!warn.empty()) {
        MINI_TRT_LOG_WARN("Safetensors load warning: " << warn);
    }
    return true;
}

bool SafetensorsLoader::HasTensor(const std::string& name) const {
    return st_.tensors.count(name);
}

bool SafetensorsLoader::GetTensorInfo(const std::string& name,
                                      safetensors::dtype* dtype,
                                      std::vector<size_t>* shape) const {
    safetensors::tensor_t tensor;
    if (!st_.tensors.at(name, &tensor)) {
        return false;
    }
    if (dtype) *dtype = tensor.dtype;
    if (shape) *shape = tensor.shape;
    return true;
}

const void* SafetensorsLoader::GetRawData(const std::string& name,
                                          size_t* bytes) const {
    safetensors::tensor_t tensor;
    if (!st_.tensors.at(name, &tensor)) {
        return nullptr;
    }
    size_t size = SafetensorsDtypeSize(tensor.dtype);
    for (auto dim : tensor.shape) {
        size *= dim;
    }
    if (bytes) *bytes = size;

    // 根据是否 mmap 选择基地址：mmap 时使用文件映射地址，否则使用内部 storage。
    const uint8_t* base = st_.mmaped ? st_.databuffer_addr : st_.storage.data();
    return base + tensor.data_offsets[0];
}

const void* SafetensorsLoader::GetConvertedData(const std::string& name,
                                                nvinfer1::DataType target_type,
                                                size_t* bytes) {
    safetensors::tensor_t tensor;
    if (!st_.tensors.at(name, &tensor)) {
        return nullptr;
    }

    size_t num_elements = 1;
    for (auto dim : tensor.shape) {
        num_elements *= dim;
    }
    size_t src_dtype_size = SafetensorsDtypeSize(tensor.dtype);
    size_t src_bytes = num_elements * src_dtype_size;

    nvinfer1::DataType src_trt_type = SafetensorsToTrtDtype(tensor.dtype);
    if (src_trt_type == target_type) {
        if (bytes) *bytes = src_bytes;
        return GetRawData(name, nullptr);
    }

    // 仅支持 BF16 -> FP32/FP16 转换
    if (tensor.dtype != safetensors::kBFLOAT16) {
        MINI_TRT_LOG_ERROR("Cannot convert dtype " << tensor.dtype
                         << " to target TRT dtype");
        return nullptr;
    }

    const void* raw = GetRawData(name, nullptr);
    if (!raw) return nullptr;

    size_t dst_dtype_size = 0;
    if (target_type == nvinfer1::DataType::kFLOAT) {
        dst_dtype_size = 4;
    } else if (target_type == nvinfer1::DataType::kHALF) {
        dst_dtype_size = 2;
    } else {
        MINI_TRT_LOG_ERROR("BF16 conversion target must be FP32 or FP16");
        return nullptr;
    }

    size_t dst_bytes = num_elements * dst_dtype_size;
    conversion_buffer_.Resize(dst_bytes);
    void* dst = conversion_buffer_.data();

    const uint16_t* src_bf16 = static_cast<const uint16_t*>(raw);
    if (target_type == nvinfer1::DataType::kFLOAT) {
        float* dst_f32 = static_cast<float*>(dst);
        for (size_t i = 0; i < num_elements; ++i) {
            // BF16 用 16 位表示高 16 位尾数，低 16 位补 0 即等价于 FP32 的 bit 模式。
            // 因此将 uint16_t 左移 16 位后按 float reinterpret，完成 BF16 -> FP32。
            uint32_t val = static_cast<uint32_t>(src_bf16[i]) << 16;
            std::memcpy(&dst_f32[i], &val, sizeof(float));
        }
    } else {
        // BF16 -> FP16：直接截断尾数（简单实现，后续可优化为 round-nearest）
        uint16_t* dst_f16 = static_cast<uint16_t*>(dst);
        for (size_t i = 0; i < num_elements; ++i) {
            // BF16 与 FP16 的指数位相同，直接保留高 16 位即截断 FP16 没有的尾数部分。
            dst_f16[i] = src_bf16[i];
        }
    }

    if (bytes) *bytes = dst_bytes;
    return dst;
}

std::vector<std::string> SafetensorsLoader::GetTensorNames() const {
    std::vector<std::string> names;
    // ordered_dict 无直接遍历接口，此处通过索引访问
    for (size_t i = 0;; ++i) {
        safetensors::tensor_t tensor;
        if (!st_.tensors.at(i, &tensor)) break;
        // 无法直接从 tensor 获取 name，ordered_dict 的 at(idx) 只返回值。
        // 需要遍历 key。由于 ordered_dict 不暴露 keys，我们暂时返回空列表。
        // 后续迭代可替换为支持遍历的 safetensors 库或自研解析。
        (void)tensor;
    }
    return names;
}

}  // namespace mini_trt_llm
