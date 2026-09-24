#include "e2e_safetensors_writer.hpp"

#include <cuda_fp16.h>
#include <safetensors.hh>

#include <cstdint>
#include <cstring>
#include <vector>

namespace mini_trt_llm {
namespace test_support {
namespace {

// BF16 的位布局是 1 符号 + 8 指数 + 7 尾数，与 FP16（5 位指数）不同，
// 只能从 FP32 截断得到。这里用 round-to-nearest-even，避免直接截断带来的系统性偏小。
uint16_t FloatToBf16Bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

void AppendTensor(safetensors::safetensors_t* file, const std::string& name,
                  const TensorSpec& spec) {
    size_t element_size = 4;
    safetensors::dtype dtype = safetensors::kFLOAT32;
    if (spec.dtype == TensorSpec::Dtype::kF16) {
        element_size = 2;
        dtype = safetensors::kFLOAT16;
    } else if (spec.dtype == TensorSpec::Dtype::kBF16) {
        element_size = 2;
        dtype = safetensors::kBFLOAT16;
    }

    const size_t count = spec.values.size();
    const size_t offset = file->storage.size();
    file->storage.resize(offset + count * element_size);
    uint8_t* dst = file->storage.data() + offset;

    for (size_t i = 0; i < count; ++i) {
        const float value = spec.values[i];
        if (dtype == safetensors::kFLOAT32) {
            std::memcpy(dst + i * 4, &value, 4);
        } else if (dtype == safetensors::kFLOAT16) {
            const __half half = __float2half_rn(value);
            std::memcpy(dst + i * 2, &half, 2);
        } else {
            const uint16_t bf16 = FloatToBf16Bits(value);
            std::memcpy(dst + i * 2, &bf16, 2);
        }
    }

    safetensors::tensor_t tensor;
    tensor.dtype = dtype;
    tensor.data_offsets[0] = offset;
    tensor.data_offsets[1] = offset + count * element_size;
    tensor.shape = spec.shape;
    file->tensors.insert(name, tensor);
}

}  // namespace

bool WriteSafetensorsFile(const std::string& path,
                          const std::map<std::string, TensorSpec>& tensors) {
    safetensors::safetensors_t file;
    for (const auto& [name, spec] : tensors) {
        AppendTensor(&file, name, spec);
    }

    std::string warn;
    std::string err;
    return safetensors::save_to_file(file, path, &warn, &err);
}

}  // namespace test_support
}  // namespace mini_trt_llm
