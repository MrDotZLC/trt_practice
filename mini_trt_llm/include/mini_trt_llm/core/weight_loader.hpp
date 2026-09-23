#pragma once

#include "mini_trt_llm/utils/json.hpp"
#include "mini_trt_llm/utils/safetensors_loader.hpp"
#include <NvInfer.h>
#include <cstddef>
#include <map>
#include <string>

namespace mini_trt_llm {

// 权重加载器。
// 从 Safetensors 读取权重，并支持通过 JSON weight_map 做 key 映射。
// BF16 权重会在查询时自动转换为 FP32/FP16。
class WeightLoader {
 public:
    WeightLoader();
    ~WeightLoader();

    // 加载目录下的 model.safetensors
    bool Load(const std::string& model_dir);

    // 设置 weight_map：source_key -> trt_name
    void SetWeightMap(const JsonValue& map);

    // 直接通过 source key 获取权重
    const void* GetWeightBySourceKey(const std::string& source_key,
                                     nvinfer1::DataType target_type,
                                     size_t* bytes);

    // 通过 TRT 名称获取权重：先在 weight_map 中查找 source_key，再读取
    const void* GetWeight(const std::string& trt_name,
                          nvinfer1::DataType target_type,
                          size_t* bytes);

    bool HasWeight(const std::string& trt_name) const;

    // 设置全局目标精度，影响默认转换行为
    void SetDefaultPrecision(nvinfer1::DataType dtype) { default_dtype_ = dtype; }

 private:
    std::string model_dir_;
    SafetensorsLoader loader_;
    std::map<std::string, std::string> weight_map_;
    nvinfer1::DataType default_dtype_ = nvinfer1::DataType::kFLOAT;
};

}  // namespace mini_trt_llm
