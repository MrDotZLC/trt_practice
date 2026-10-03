#pragma once

#include "mini_trt_llm/plugins/iplugin_v3_base.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

inline constexpr char kPackedAttentionPluginName[] = "MiniTrtLlmPackedAttention";
inline constexpr char kPackedAttentionPluginVersion[] = "1";

// **S4 的混合批注意力**（design.md D14 / p5_s4_interface_spec.md §7 的 A1：单插件内部分派）。
//
// 一个 packed 张量里装两相，attention 按段分派两条 kernel：
//   * context 段（前 B_ctx 条序列的**全部** token）→ varlen 因果自注意力（K/V 来自 packed 张量本身）；
//   * generation 段（后 B_gen 行的 1 个 token）→ 分页注意力（K/V 来自分页缓存 + 当前 token 自包含）。
//
// 输入（顺序即契约；全部 LINEAR）：
//   0 query          [T, num_heads, head_size]        FP16 / FP32
//   1 key            [T, num_kv_heads, head_size]     （context 段的 K）
//   2 value          [T, num_kv_heads, head_size]
//   3 key_cache      [num_blocks, block_size, num_kv_heads, head_size]
//   4 value_cache    同 key_cache
//   5 block_tables   [B_total, max_blocks_per_seq]    INT32（**按 packed 行序**排列）
//   6 context_lens   [B_total]                        INT32（同上；generation 段取"推进前"的值）
//   7 cu_seqlens_ctx [B_ctx + 1]                      INT32（**段内**下标，从 0 起）
//   8 context_seq_count [1]                           INT32（段边界 B_ctx；kernel 读它取 T_ctx）
// 输出：
//   0 attention_out  [T, num_heads, head_size]        与 query 同 dtype/format
//
// **两条 kernel 的分派发生在 enqueue**：grid 所需的是"多少条 context 序列"，
// 它可以从 `cu_seqlens_ctx` 的**形状**推出（B_ctx = dim[0] - 1），
// 而段内 token 总数 T_ctx = cu_seqlens_ctx[B_ctx] 是**设备值** —— 由 kernel 自己读，
// 因此分派**不需要**任何 D2H 同步（见 p5_s4_interface_spec.md §3 的下标纪律）。
//
// **共享内存**：context kernel 按编译期上限 `kPackedAttentionMaxContextSeqLen` 分配 score 数组；
// `max_seq_len` 属性必须 ≤ 该上限，否则 `configurePlugin` 直接失败（宁可在建图时拦住）。
inline constexpr int32_t kPackedAttentionMaxContextSeqLen = 1024;

class PackedAttentionPlugin : public IPluginV3Base {
 public:
    PackedAttentionPlugin() = default;
    PackedAttentionPlugin(int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                          int32_t block_size, int32_t max_seq_len, float scale);
    explicit PackedAttentionPlugin(const nvinfer1::PluginFieldCollection* fc);

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
    int32_t max_seq_len() const noexcept { return max_seq_len_; }
    float scale() const noexcept { return scale_; }

    // 输入个数（含段边界标量）。写成常量是为了让 enqueue 与建图两侧都盯着同一个数。
    static constexpr int32_t kInputCount = 9;

 private:
    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_size_ = 0;
    int32_t block_size_ = 0;    // 强制显式配置，不提供默认值
    int32_t max_seq_len_ = 0;   // context kernel 的 grid.z 上限（context 段单序列的最大长度）
    float scale_ = 0.0f;        // <=0 时按 1/sqrt(head_size) 推导
};

class PackedAttentionPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
    PackedAttentionPluginCreator();
    ~PackedAttentionPluginCreator() noexcept override = default;

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
PackedAttentionPluginCreator& GetPackedAttentionPluginCreator() noexcept;

}  // namespace mini_trt_llm
