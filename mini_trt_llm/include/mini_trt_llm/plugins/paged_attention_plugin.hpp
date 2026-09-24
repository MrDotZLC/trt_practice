#pragma once

#include "mini_trt_llm/plugins/iplugin_v3_base.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

inline constexpr char kPagedAttentionPluginName[] = "MiniTrtLlmPagedAttention";
inline constexpr char kPagedAttentionPluginVersion[] = "1";

// PagedAttention，Phase 1 只实现 Decoding 阶段（query 序列长度为 1）。
//
// 输入：
//   query        [batch, num_heads, 1, head_size]
//   key_cache    [num_blocks, block_size, num_kv_heads, head_size]
//   value_cache  同 key_cache
//   block_tables [batch, max_blocks_per_seq]，INT32
//   context_lens [batch]，INT32
//   key_new      [batch, num_kv_heads, 1, head_size]（可选，第 6 个输入）
//   value_new    同 key_new（可选，第 7 个输入）
// 输出：
//   [batch, num_heads, 1, head_size]
//
// 为什么要 key_new / value_new：decode 第 t 步的注意力必须包含当前 token 自己的 K/V，
// 而这份 K/V 由本次前向算出、不可能预先写进 cache。传了这两个输入，注意力就会在
// 扫完 cache 后再把当前 token 当作第 context_lens[b] 个位置参与 softmax；
// 不传则保持"只按 cache 内容算注意力"的原有行为。
// **arity 由 nbInputs 决定，不做成序列化属性**：它是网络连线的直接结果，
// 再存一份状态只会多出一处可能与连线失配的来源（同 §2.12 对 RoPE head 配置的处理）。
//
// Prefill 阶段（query 序列长度 > 1）留到后续迭代：它需要因果 mask 与按位置分块，
// 与解码路径的 kernel 结构差异较大，混在一起会同时拖慢两条路径。
class PagedAttentionPlugin : public IPluginV3Base {
 public:
    PagedAttentionPlugin() = default;
    PagedAttentionPlugin(int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                         int32_t block_size, float scale);
    explicit PagedAttentionPlugin(const nvinfer1::PluginFieldCollection* fc);

    // 按 head_size 推导默认 scale，与主流实现（1/sqrt(d)）一致。
    static float DefaultScale(int32_t head_size) {
        return head_size > 0 ? 1.0f / sqrtf(static_cast<float>(head_size)) : 0.0f;
    }

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
    int32_t block_size() const noexcept { return block_size_; }
    float scale() const noexcept { return scale_; }

 private:
    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_size_ = 0;
    int32_t block_size_ = 0;  // Q5：强制显式配置，不提供默认值
    float scale_ = 0.0f;      // <=0 时按 1/sqrt(head_size) 推导
    // 是否连接了当前 token 的 K/V（第 6/7 个输入）。由 configurePlugin / onShapeChange
    // 依 nbInputs 刷新，因此反序列化后的实例同样能拿到正确值。
    bool has_current_token_ = false;

    int32_t serialized_block_size_ = 0;
    float serialized_scale_ = 0.0f;
};

class PagedAttentionPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
    PagedAttentionPluginCreator();
    ~PagedAttentionPluginCreator() noexcept override = default;

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
PagedAttentionPluginCreator& GetPagedAttentionPluginCreator() noexcept;

}  // namespace mini_trt_llm
