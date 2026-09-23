#pragma once

#include "mini_trt_llm/plugins/iplugin_v3_base.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

// Plugin 名带 MiniTrtLlm 前缀，规避与 TensorRT 原生 layer / plugin 重名（方案 B 的关键约束）。
inline constexpr char kRmsNormPluginName[] = "MiniTrtLlmRmsNorm";
inline constexpr char kRmsNormPluginVersion[] = "1";

// 对最后一维做 RMSNorm：
//   out = x / sqrt(mean(x^2) + eps) * weight
//
// 输入：
//   input  [d0, ..., d(n-1)]，FP16 / FP32，最后一维为 hidden_size
//   weight [hidden_size]，dtype 与 input 一致
// 输出：
//   与 input 同形状、同 dtype
//
// 支持 MHA 之外的任意前缀维度（batch / seq_len 可动态），因为归一化只沿最后一维进行。
class RmsNormPlugin : public IPluginV3Base {
 public:
    // 默认 eps 取 1e-6：与 LLaMA 系实现一致，且在 FP16 下不会被舍入吃掉。
    static constexpr float kDefaultEps = 1e-6f;

    RmsNormPlugin() = default;
    RmsNormPlugin(float eps, int32_t hidden_size);

    // 从序列化字段恢复属性；fc 为 nullptr（无属性）时使用默认值。
    explicit RmsNormPlugin(const nvinfer1::PluginFieldCollection* fc);

    // --- IPluginV3Base 接口 ---
    IPluginV3Base* clone() noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
                               const nvinfer1::DataType* inputTypes,
                               int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
                            const nvinfer1::DimsExprs* shapeInputs, int32_t nbShapeInputs,
                            nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos,
                                   const nvinfer1::DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
                            const nvinfer1::DynamicPluginTensorDesc* out,
                            int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const nvinfer1::DynamicPluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;
    int32_t onShapeChange(const nvinfer1::PluginTensorDesc* in, int32_t nbInputs,
                          const nvinfer1::PluginTensorDesc* out,
                          int32_t nbOutputs) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    // 供 Creator 在创建后写入 namespace（基类的 setter 是 protected，Creator 非派生类）。
    void SetNamespace(const char* ns) noexcept { SetPluginNamespaceInternal(ns); }

    float eps() const noexcept { return eps_; }
    int32_t hidden_size() const noexcept { return hidden_size_; }

 private:
    float eps_ = kDefaultEps;
    int32_t hidden_size_ = 0;

    // 序列化字段必须指向生命周期稳定的内存，因此用独立成员缓存而非临时量。
    float serialized_eps_ = kDefaultEps;
    int32_t serialized_hidden_size_ = 0;
};

// IPluginCreatorV3One 实现，负责构建期创建与运行期反序列化。
class RmsNormPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
    RmsNormPluginCreator();
    ~RmsNormPluginCreator() noexcept override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV3* createPlugin(const char* name,
                                      const nvinfer1::PluginFieldCollection* fc,
                                      nvinfer1::TensorRTPhase phase) noexcept override;

 private:
    std::string namespace_;
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection field_collection_{};
};

// 返回进程内固定的 creator 实例，供 PluginRegistry 登记。
//
// REGISTER_TENSORRT_PLUGIN 已经让 TensorRT 全局 registry 能按名找到 creator；
// 这个访问器是给 mini_trt_llm 自己的 PluginRegistry 用的，避免各处各自构造实例后
// 指针失效。Creator 无状态，因此存在两个实例是安全的。
RmsNormPluginCreator& GetRmsNormPluginCreator() noexcept;

}  // namespace mini_trt_llm
