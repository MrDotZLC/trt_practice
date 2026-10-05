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

    // 设置 weight_map。方向是 TRT 层权重名 -> safetensors 中的 source key，
    // 与 ModelConfig::weight_map 的语义一致，不要写反。
    void SetWeightMap(const JsonValue& map);

    // 直接通过 source key 获取权重
    const void* GetWeightBySourceKey(const std::string& source_key,
                                     nvinfer1::DataType target_type,
                                     size_t* bytes) const;

    // 通过 TRT 名称获取权重：先在 weight_map 中查找 source_key，再读取
    // const 是为了配合 IModelBuilder::Build 的 const WeightLoader& 参数
    const void* GetWeight(const std::string& trt_name,
                          nvinfer1::DataType target_type,
                          size_t* bytes) const;

    bool HasWeight(const std::string& trt_name) const;

    // 设置全局目标精度，影响默认转换行为
    void SetDefaultPrecision(nvinfer1::DataType dtype) { default_dtype_ = dtype; }

    // ---- REQ-017 路线 C：离线量化产物的读取（int8 权重） ----
    //
    // 单独一个入口而不是复用一个"多文件"抽象：量化产物只在建图期被 int8 常量消费，
    // 与"默认精度的权重取数"是两条语义。合并会把 int8 的零拷贝条件扩散到所有调用点。
    //
    // 载入失败（缺文件 / 格式错）返回 false：**必须响亮失败**——静默退回 FP32 权重
    // 就是"声明了量化、其实没量化"，正是本 feature 要防的事。
    bool LoadQuantized(const std::string& path);
    bool has_quantized() const { return quantized_loaded_; }

    // 按 TRT 名取 int8 权重（同样先经 weight_map 解析到 source key）。
    // 只在目标类型为 kINT8 时有意义；返回 nullptr 表示该张量不在量化产物里。
    const void* GetQuantizedWeight(const std::string& trt_name, size_t* bytes) const;

    // 把 TRT 名解析成 safetensors 里的 source key（找不到时回退成同名）。
    // 暴露它是为了让建图侧能核对"清单声明的 source_key"与"weight_map 解析出的 source_key"
    // 是否一致——不一致就是"尺子量 A、裁剪 B"，必须当场失败（见 design.md 的 Data Structure）。
    std::string ResolveSourceKey(const std::string& trt_name) const;

 private:
    std::string model_dir_;
    SafetensorsLoader loader_;
    // 量化产物用**独立的 loader**：两张表的生命周期与来源都不同，
    // 共用一个 loader 会让"int8 权重来自哪个文件"变成不可回答的问题。
    SafetensorsLoader quantized_;
    bool quantized_loaded_ = false;
    std::map<std::string, std::string> weight_map_;
    nvinfer1::DataType default_dtype_ = nvinfer1::DataType::kFLOAT;
};

}  // namespace mini_trt_llm
