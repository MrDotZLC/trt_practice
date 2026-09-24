#pragma once

#include "mini_trt_llm/plugins/iplugin_v3_base.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

inline constexpr char kRoPEPluginName[] = "MiniTrtLlmRoPE";
inline constexpr char kRoPEPluginVersion[] = "1";

// Rotary Position Embedding。对 query / key 就地等价地施加旋转（输出为独立 tensor）。
//
// 输入：
//   query        [batch, num_heads,    seq_len, head_size]
//   key          [batch, num_kv_heads, seq_len, head_size]
//   position_ids [batch, seq_len]，INT32
// 输出：
//   rotated_query / rotated_key，形状与对应输入一致
//
// position_ids 作为输入而非内部生成：Decode 阶段必须由调用方给出真实位置（KV Cache 续写），
// 且要支持非连续位置。
class RoPEPlugin : public IPluginV3Base {
 public:
    // 默认 base 取 10000.0：LLaMA / GPT-NeoX 系通用取值，改 base 需要与训练时保持一致。
    static constexpr float kDefaultBase = 10000.0f;

    RoPEPlugin() = default;
    RoPEPlugin(int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
               int32_t rotary_dim, float base);
    explicit RoPEPlugin(const nvinfer1::PluginFieldCollection* fc);

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

    void SetNamespace(const char* ns) noexcept { SetPluginNamespaceInternal(ns); }

    int32_t num_heads() const noexcept { return num_heads_; }
    int32_t num_kv_heads() const noexcept { return num_kv_heads_; }
    int32_t head_size() const noexcept { return head_size_; }
    int32_t rotary_dim() const noexcept { return rotary_dim_; }
    float base() const noexcept { return base_; }

 private:
    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_size_ = 0;
    int32_t rotary_dim_ = 0;
    float base_ = kDefaultBase;

    // 序列化字段需指向生命周期稳定的内存，因此用独立成员缓存。
    int32_t serialized_rotary_dim_ = 0;
    float serialized_base_ = kDefaultBase;
};

class RoPEPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
    RoPEPluginCreator();
    ~RoPEPluginCreator() noexcept override = default;

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
RoPEPluginCreator& GetRoPEPluginCreator() noexcept;

}  // namespace mini_trt_llm
