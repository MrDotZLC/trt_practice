#pragma once

#include <NvInfer.h>
#include <safetensors.hh>
#include <cstdint>
#include <map>
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

    // 获取转换为 target_type 后的数据指针（仅支持目标 FP32 / FP16）。
    //
    // 若源类型与目标类型本就一致（F32→F32、F16→F16），直接返回原始数据指针，零拷贝；
    // 否则在内部按张量名缓存一份转换结果并返回其指针。
    //
    // 指针在 SafetensorsLoader 生命周期内稳定，且**不同张量的结果互不覆盖**——
    // 这一点是必须的：nvinfer1::Weights 只持有裸指针，要等到 buildSerializedNetwork
    // 才真正读数据，因此调用方连续取多个权重时，先取的不能被后取的冲掉。
    // 声明为 const：转换缓存属于实现细节，不改变 loader 的可观测状态。
    // 这样 WeightLoader 才能把读取接口整体做成 const，供 IModelBuilder::Build 的
    // const 引用参数调用。
    const void* GetConvertedData(const std::string& name,
                                 nvinfer1::DataType target_type,
                                 size_t* bytes) const;

    std::vector<std::string> GetTensorNames() const;

 private:
    safetensors::safetensors_t st_;
    mutable std::map<std::string, std::vector<char>> conversion_cache_;
};

}  // namespace mini_trt_llm
