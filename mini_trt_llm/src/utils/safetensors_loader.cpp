#include "mini_trt_llm/utils/safetensors_loader.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <algorithm>
#include <cuda_fp16.h>
#include <cstring>

namespace mini_trt_llm {
namespace {

// BF16 的位布局是 1 符号 + 8 指数 + 7 尾数，FP16 是 1 + 5 + 10，两者指数宽度不同，
// 所以不能靠位截断互相转换，必须先还原成 FP32 再降到目标精度。
inline float Bf16BitsToFloat(uint16_t bits) {
    const uint32_t widened = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &widened, sizeof(float));
    return value;
}

inline float Fp16BitsToFloat(uint16_t bits) {
    __half half = __ushort_as_half(bits);
    return __half2float(half);
}

// 源类型与目标类型本就一致时可直接零拷贝。刻意不用 SafetensorsToTrtDtype 做这个判断：
// 该函数对 BF16 / FLOAT64 会回退成 kFLOAT，会让「源是 BF16、目标是 kFLOAT」被误判为同类型，
// 从而把 BF16 原始数据当成 FP32 返回。
bool IsDirectCopy(safetensors::dtype src, nvinfer1::DataType target) {
    return (src == safetensors::kFLOAT32 && target == nvinfer1::DataType::kFLOAT) ||
           (src == safetensors::kFLOAT16 && target == nvinfer1::DataType::kHALF);
}

// 把受支持的源 dtype 转成 FP32 / FP16。累加统一走 FP32，只在写回时降到目标精度。
bool ConvertTensorData(safetensors::dtype src_dtype, const void* src, size_t count,
                       nvinfer1::DataType target, void* dst) {
    const bool to_half = (target == nvinfer1::DataType::kHALF);
    char* out = static_cast<char*>(dst);
    for (size_t i = 0; i < count; ++i) {
        float value = 0.0f;
        switch (src_dtype) {
            case safetensors::kFLOAT32:
                value = static_cast<const float*>(src)[i];
                break;
            case safetensors::kFLOAT16:
                value = Fp16BitsToFloat(static_cast<const uint16_t*>(src)[i]);
                break;
            case safetensors::kBFLOAT16:
                value = Bf16BitsToFloat(static_cast<const uint16_t*>(src)[i]);
                break;
            case safetensors::kFLOAT64:
                value = static_cast<float>(static_cast<const double*>(src)[i]);
                break;
            default:
                return false;
        }
        if (to_half) {
            const __half half = __float2half_rn(value);
            std::memcpy(out + i * 2, &half, 2);
        } else {
            std::memcpy(out + i * 4, &value, 4);
        }
    }
    return true;
}

}  // namespace

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
                                                size_t* bytes) const {
    safetensors::tensor_t tensor;
    if (!st_.tensors.at(name, &tensor)) {
        return nullptr;
    }
    if (bytes != nullptr) {
        *bytes = 0;
    }

    size_t num_elements = 1;
    for (auto dim : tensor.shape) {
        num_elements *= dim;
    }

    if (IsDirectCopy(tensor.dtype, target_type)) {
        return GetRawData(name, bytes);
    }

    if (target_type != nvinfer1::DataType::kFLOAT &&
        target_type != nvinfer1::DataType::kHALF) {
        MINI_TRT_LOG_ERROR("Conversion target must be FP32 or FP16");
        return nullptr;
    }

    const size_t dtype_size = (target_type == nvinfer1::DataType::kFLOAT) ? 4u : 2u;
    const size_t dst_bytes = num_elements * dtype_size;

    // 命中缓存直接返回：其一是省去重复转换，其二是保证同一张量多次请求拿到同一个稳定指针。
    auto cached = conversion_cache_.find(name);
    if (cached != conversion_cache_.end() && cached->second.size() == dst_bytes) {
        if (bytes != nullptr) {
            *bytes = dst_bytes;
        }
        return cached->second.data();
    }

    const void* raw = GetRawData(name, nullptr);
    if (raw == nullptr) {
        return nullptr;
    }

    std::vector<char> converted(dst_bytes);
    if (!ConvertTensorData(tensor.dtype, raw, num_elements, target_type,
                           converted.data())) {
        MINI_TRT_LOG_ERROR("Unsupported dtype conversion for tensor "
                           << name << " (source dtype " << tensor.dtype << ")");
        return nullptr;
    }

    auto& slot = conversion_cache_[name];
    slot = std::move(converted);
    if (bytes != nullptr) {
        *bytes = dst_bytes;
    }
    return slot.data();
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
