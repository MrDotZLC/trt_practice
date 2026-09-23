#pragma once

#include "mini_trt_llm/utils/memory_pool.hpp"
#include <NvInfer.h>
#include <safetensors.hh>
#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

// Safetensors dtype 到字节数的映射
size_t SafetensorsDtypeSize(safetensors::dtype dtype);

// Safetensors dtype 到 TRT DataType 的映射（未支持的返回 kFLOAT 并记录）
nvinfer1::DataType SafetensorsToTrtDtype(safetensors::dtype dtype);

// 简单封装 safetensors-cpp，提供张量查询与 BF16/F32/F16 数据访问。
class SafetensorsLoader {
 public:
    SafetensorsLoader();
    ~SafetensorsLoader();

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;

    // 加载 .safetensors 文件
    bool LoadFromFile(const std::string& path);

    // 是否存在指定 tensor
    bool HasTensor(const std::string& name) const;

    // 获取 tensor 信息
    bool GetTensorInfo(const std::string& name,
                       safetensors::dtype* dtype,
                       std::vector<size_t>* shape) const;

    // 获取 tensor 原始数据指针（指向 loader 内部存储）
    // 注意：指针生命周期与 SafetensorsLoader 对象相同。
    const void* GetRawData(const std::string& name, size_t* bytes) const;

    // 获取转换为 target_type 后的数据指针。
    // 若原始类型与目标类型一致，返回原始指针；
    // 若不一致（如 BF16 -> FP16/FP32），在内部临时缓冲区转换并返回其指针。
    // 注意：转换结果会被下一次 GetConvertedData 调用覆盖，调用方应及时拷贝。
    const void* GetConvertedData(const std::string& name,
                                 nvinfer1::DataType target_type,
                                 size_t* bytes);

    std::vector<std::string> GetTensorNames() const;

 private:
    safetensors::safetensors_t st_;
    DeviceBuffer conversion_buffer_;
};

}  // namespace mini_trt_llm
