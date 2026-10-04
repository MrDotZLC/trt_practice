#include "mini_trt_llm/plugins/packed_attention_plugin.hpp"

#include "mini_trt_llm/utils/cuda_dtype.cuh"
#include "mini_trt_llm/utils/cuda_reduce.cuh"
#include "mini_trt_llm/utils/logger.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <new>
#include <string>

namespace mini_trt_llm {
namespace {

// blockDim 取 head_size 向上取整到 32，因此最多 32 个 warp 参与归约（与 paged 插件同一约定）。
constexpr int kMaxWarps = 32;
constexpr int32_t kMaxHeadSize = kMaxWarps * 32;

using cuda::BlockReduceMax;
using cuda::BlockReduceSum;
using cuda::FromFloat;
using cuda::ToFloat;

// ---------------------------------------------------------------------------
// context 段：varlen 因果自注意力（K/V 来自 packed 张量本身，不读缓存）
// ---------------------------------------------------------------------------
//
// 网格 (num_heads, B_ctx, max_seq_len)：z 维是**段内位置**，宿主侧只知道上限
// （设备端的单序列长度早退由 kernel 自己判），所以超出 `len` 的块立即返回 ——
// 这与 paged split-K 那条"片数由设备端推导、越界块立即返回"是同一套纪律。
//
// 一个块负责 (head, sequence, query position)，线程两用：
//   * pass 1/3 里沿 **key 维** 切分（每个线程认领若干个 key）；
//   * pass 4 里沿 **head_size 维** 切分（每个线程认领若干个 d）。
// score 存进 shared（上限 1024 → 4 KB），因此不需要把整行 logits 物化到显存，
// 也不需要 workspace（getWorkspaceSize 恒为 0）。
//
// **正确性优先**：这是 v1，复杂度 O(T_ctx × L × D)；tiling / 在线 softmax / split-K
// 之类的优化等 P4/P7 有数据之后再谈（与采样器 fast/legacy 的处理方式一致）。
template <typename T>
__global__ void PackedContextAttentionKernel(
    const T* __restrict__ query, const T* __restrict__ key, const T* __restrict__ value,
    T* __restrict__ output, const int32_t* __restrict__ cu_seqlens_ctx, int32_t num_heads,
    int32_t num_kv_heads, int32_t head_size, float scale) {
    __shared__ float s_scores[kPackedAttentionMaxContextSeqLen];
    __shared__ float s_reduce[kMaxWarps];

    const int32_t head = blockIdx.x;
    const int32_t seq = blockIdx.y;
    const int32_t pos = blockIdx.z;  // 段内相对位置
    const int32_t tid = threadIdx.x;
    const int32_t nt = blockDim.x;

    const int32_t begin = cu_seqlens_ctx[seq];
    const int32_t len = cu_seqlens_ctx[seq + 1] - begin;
    if (pos >= len) {
        return;  // grid.z 是上限：越界块立即返回，不读不写
    }
    if (len > kPackedAttentionMaxContextSeqLen) {
        return;  // 兜底：configurePlugin 已按 max_seq_len 拦过
    }

    // GQA/MQA：一组 query head 共享同一个 kv head（与 paged 插件同一约定）
    const int32_t group = num_heads / num_kv_heads;
    const int32_t kv_head = (group > 0) ? (head / group) : 0;

    const T* query_row =
        query + (static_cast<size_t>(begin + pos) * num_heads + head) * head_size;
    T* output_row =
        output + (static_cast<size_t>(begin + pos) * num_heads + head) * head_size;
    const int32_t key_count = pos + 1;  // 因果：只看段内 0..pos

    // pass 1：scores（每线程认领若干 key，各自写自己的槽位）
    for (int32_t t = tid; t < key_count; t += nt) {
        const T* key_row =
            key + (static_cast<size_t>(begin + t) * num_kv_heads + kv_head) * head_size;
        float dot = 0.0f;
        for (int32_t d = 0; d < head_size; ++d) {
            dot += ToFloat(query_row[d]) * ToFloat(key_row[d]);
        }
        s_scores[t] = dot * scale;
    }

    // pass 2：段内最大值（只需各自读回自己写的槽位，再 block 归约）
    float local_max = -CUDART_INF_F;
    for (int32_t t = tid; t < key_count; t += nt) {
        local_max = fmaxf(local_max, s_scores[t]);
    }
    const float row_max = BlockReduceMax(local_max, s_reduce);

    // pass 3：exp 与求和，顺手把 exp 结果写回 shared 供 pass 4 复用
    float local_sum = 0.0f;
    for (int32_t t = tid; t < key_count; t += nt) {
        const float e = __expf(s_scores[t] - row_max);
        s_scores[t] = e;
        local_sum += e;
    }
    const float row_sum = BlockReduceSum(local_sum, s_reduce);
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

    // pass 4：加权求和（这次沿 head_size 维切分，读全部 key 的权重）
    for (int32_t d = tid; d < head_size; d += nt) {
        float acc = 0.0f;
        for (int32_t t = 0; t < key_count; ++t) {
            const T* value_row =
                value + (static_cast<size_t>(begin + t) * num_kv_heads + kv_head) * head_size;
            acc += s_scores[t] * ToFloat(value_row[d]);
        }
        output_row[d] = FromFloat<T>(acc * inv_sum);
    }
}

// ---------------------------------------------------------------------------
// generation 段：分页注意力（K/V 来自缓存 + 当前 token 自包含）
// ---------------------------------------------------------------------------
//
// 与 `PagedAttentionDecodeKernel` 是同一套算法（每块负责一个 (head, 行)，线程沿 head_size 切分、
// 在线 softmax），区别只在**下标基准**：packed 的行序里 generation 段排在 context 段之后，
// 所以这一段的
//   * query / key_new / value_new / output 的行号 = `T_ctx + j`
//   * block_tables / context_lens 的行号        = `B_ctx + j`
// 其中 `T_ctx = cu_seqlens_ctx[B_ctx]` 是**设备值**，只能由 kernel 自己读 ——
// 宿主侧因此不需要任何 D2H（见 p5_s4_interface_spec.md §3 的下标纪律）。
//
// **已知回退（必须修，2026-10-04 作者指出）**：这份是**单趟**实现，而 `PagedAttentionPlugin` 的
// **生产路径早就是 split-K**（REQ-014 交付；单趟只是 A/B 参考与 workspace 缺失时的兜底）。
// S4 是默认路径，照这份落地等于在默认路径上丢掉 REQ-014 的收益 —— **这不是"v1 取舍"**。
//
// 修法（见 STATE.md 的 Current Blockers，不动 split-K 算法本身）：给**已交付的**
// `PagedAttentionKernelArgs` 与两个 kernel 加"行/token 基址的设备端读取"参数
// （`cu_seqlens_ctx` + `context_seq_count`，默认 null/0 即现状），本插件的 generation 段改为
// **直接调 `LaunchPagedAttentionSplit`**（`paged_attention_kernel.hpp` 是公开 API），
// `getWorkspaceSize` 相应改报 split-K 的构建期上界 —— 这样两条路径的 generation 段共用同一份实现。
template <typename T>
__global__ void PackedGenerationAttentionKernel(
    const T* __restrict__ query, const T* __restrict__ key_new, const T* __restrict__ value_new,
    const T* __restrict__ key_cache, const T* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables, const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ cu_seqlens_ctx, T* __restrict__ output, int32_t context_seq_count,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size, int32_t block_size,
    int32_t max_blocks_per_seq, float scale) {
    __shared__ float reduce_scratch[kMaxWarps];

    const int32_t head = blockIdx.x;
    const int32_t j = blockIdx.y;  // generation 段的段内行号
    const int32_t d = threadIdx.x;
    const bool active = d < head_size;

    const int32_t t_ctx = cu_seqlens_ctx[context_seq_count];  // 设备值：context 段的 token 总数
    const int32_t row = context_seq_count + j;                // packed 行号（按行输入的基准）

    const int32_t group = num_heads / num_kv_heads;
    const int32_t kv_head = (group > 0) ? (head / group) : 0;

    const T* query_row =
        query + (static_cast<size_t>(t_ctx + j) * num_heads + head) * head_size;
    T* output_row = output + (static_cast<size_t>(t_ctx + j) * num_heads + head) * head_size;
    const float query_value = active ? ToFloat(query_row[d]) : 0.0f;

    const int32_t context_len = context_lens[row];
    const int32_t* block_table =
        block_tables + static_cast<size_t>(row) * max_blocks_per_seq;

    float accumulator = 0.0f;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;

    // 参与 softmax 的位置数 = context_len + 1：前 context_len 个来自分页缓存，
    // 最后一个来自 packed 张量里它自己那个 token（key_new / value_new）。
    const int32_t total_len = context_len + 1;
    for (int32_t t = 0; t < total_len; ++t) {
        const T* key_row;
        const T* value_row;
        if (t < context_len) {
            const int32_t physical_block = block_table[t / block_size];
            const int32_t slot = t % block_size;
            const size_t kv_offset =
                ((static_cast<size_t>(physical_block) * block_size + slot) * num_kv_heads +
                 kv_head) *
                head_size;
            key_row = key_cache + kv_offset;
            value_row = value_cache + kv_offset;
        } else {
            const size_t kv_offset =
                (static_cast<size_t>(t_ctx + j) * num_kv_heads + kv_head) * head_size;
            key_row = key_new + kv_offset;
            value_row = value_new + kv_offset;
        }

        float partial = 0.0f;
        if (active) {
            partial = query_value * ToFloat(key_row[d]);
        }
        const float score = BlockReduceSum(partial, reduce_scratch) * scale;

        const float new_max = fmaxf(running_max, score);
        const float alpha = __expf(running_max - new_max);
        const float probability = __expf(score - new_max);
        running_sum = running_sum * alpha + probability;
        if (active) {
            accumulator = accumulator * alpha + probability * ToFloat(value_row[d]);
        }
        running_max = new_max;
    }

    if (active) {
        output_row[d] =
            FromFloat<T>(running_sum > 0.0f ? accumulator / running_sum : 0.0f);
    }
}

// blockDim 取 head_size 向上取整到 32（与 paged 插件同一约定）。
int32_t ThreadsForHeadSize(int32_t head_size) {
    return ((head_size + 31) / 32) * 32;
}

}  // namespace

// =============================================================================
// PackedAttentionPlugin
// =============================================================================

PackedAttentionPlugin::PackedAttentionPlugin(int32_t num_heads, int32_t num_kv_heads,
                                             int32_t head_size, int32_t block_size,
                                             int32_t max_seq_len, float scale)
    : num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      block_size_(block_size),
      max_seq_len_(max_seq_len),
      scale_(scale) {}

PackedAttentionPlugin::PackedAttentionPlugin(const nvinfer1::PluginFieldCollection* fc) {
    if (fc != nullptr && fc->fields != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const nvinfer1::PluginField& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr) {
                continue;
            }
            const std::string name(field.name);
            if (name == "block_size") {
                block_size_ = *static_cast<const int32_t*>(field.data);
            } else if (name == "max_seq_len") {
                max_seq_len_ = *static_cast<const int32_t*>(field.data);
            } else if (name == "scale") {
                scale_ = *static_cast<const float*>(field.data);
            }
        }
    }
}

IPluginV3Base* PackedAttentionPlugin::clone() noexcept {
    auto* copy = new (std::nothrow) PackedAttentionPlugin(
        num_heads_, num_kv_heads_, head_size_, block_size_, max_seq_len_, scale_);
    if (copy != nullptr) {
        copy->SetNamespace(GetPluginNamespaceInternal());
    }
    return copy;
}

const char* PackedAttentionPlugin::getPluginType() const noexcept {
    return kPackedAttentionPluginName;
}

const char* PackedAttentionPlugin::getPluginVersion() const noexcept {
    return kPackedAttentionPluginVersion;
}

int32_t PackedAttentionPlugin::getNbOutputs() const noexcept { return 1; }

int32_t PackedAttentionPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes,
                                                  int32_t nbOutputs,
                                                  const nvinfer1::DataType* inputTypes,
                                                  int32_t nbInputs) const noexcept {
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs < 1 || nbInputs < 1) {
        return 1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t PackedAttentionPlugin::getOutputShapes(const nvinfer1::DimsExprs* inputs,
                                               int32_t nbInputs,
                                               const nvinfer1::DimsExprs* shapeInputs,
                                               int32_t nbShapeInputs,
                                               nvinfer1::DimsExprs* outputs,
                                               int32_t nbOutputs,
                                               nvinfer1::IExprBuilder& exprBuilder) noexcept {
    (void)shapeInputs;
    (void)nbShapeInputs;
    (void)exprBuilder;
    if (inputs == nullptr || outputs == nullptr || nbInputs < 1 || nbOutputs < 1) {
        return 1;
    }
    outputs[0] = inputs[0];  // [T, num_heads, head_size]，与 query 同形
    return 0;
}

bool PackedAttentionPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut == nullptr || pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    const nvinfer1::PluginTensorDesc& desc = inOut[pos].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    // pos 5..8 是 block_tables / context_lens / cu_seqlens_ctx / context_seq_count，
    // 四个都是 INT32（不是"与 query 同类型"的参与者）
    if (pos == 5 || pos == 6 || pos == 7 || pos == 8) {
        return desc.type == nvinfer1::DataType::kINT32;
    }
    if (desc.type != nvinfer1::DataType::kFLOAT && desc.type != nvinfer1::DataType::kHALF) {
        return false;
    }
    if (pos == 0) {
        return true;
    }
    // 只允许读 inOut[0..pos]：TRT 保证该范围内有效，之后是未初始化内存
    return inOut[pos].desc.format == inOut[0].desc.format &&
           inOut[pos].desc.type == inOut[0].desc.type;
}

int32_t PackedAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                               int32_t nbInputs,
                                               const nvinfer1::DynamicPluginTensorDesc* out,
                                               int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < kInputCount) {
        MINI_TRT_LOG_ERROR("PackedAttention: expects " << kInputCount << " inputs, got "
                                                       << nbInputs);
        return 1;
    }
    // query / key / value 的 token 维必须一致（同一个 packed 张量）
    const nvinfer1::Dims& query_dims = in[0].desc.dims;
    const nvinfer1::Dims& key_dims = in[1].desc.dims;
    if (query_dims.nbDims != 3 || key_dims.nbDims != 3) {
        MINI_TRT_LOG_ERROR("PackedAttention: query/key must be 3-D [T, heads, head_size]");
        return 1;
    }
    if (query_dims.d[0] != key_dims.d[0]) {
        MINI_TRT_LOG_ERROR("PackedAttention: query and key must share the token dimension");
        return 1;
    }
    const nvinfer1::Dims& cache_dims = in[3].desc.dims;
    if (cache_dims.nbDims != 4) {
        MINI_TRT_LOG_ERROR("PackedAttention: key_cache must be 4-D");
        return 1;
    }
    if (in[8].desc.dims.nbDims != 1 || in[8].desc.dims.d[0] != 1) {
        MINI_TRT_LOG_ERROR("PackedAttention: context_seq_count must be [1]");
        return 1;
    }
    // head 数从张量形状取（与 paged 插件同口径：形状是连线的直接结果，不另存状态）
    num_heads_ = query_dims.d[1];
    num_kv_heads_ = key_dims.d[1];
    head_size_ = query_dims.d[2];
    if (num_heads_ <= 0 || num_kv_heads_ <= 0 || head_size_ <= 0 ||
        num_heads_ % num_kv_heads_ != 0) {
        MINI_TRT_LOG_ERROR("PackedAttention: invalid head geometry (num_heads="
                           << num_heads_ << ", num_kv_heads=" << num_kv_heads_ << ")");
        return 1;
    }
    if (head_size_ > kMaxHeadSize) {
        MINI_TRT_LOG_ERROR("PackedAttention: head_size " << head_size_ << " exceeds "
                                                          << kMaxHeadSize);
        return 1;
    }
    if (max_seq_len_ <= 0 || max_seq_len_ > kPackedAttentionMaxContextSeqLen) {
        MINI_TRT_LOG_ERROR("PackedAttention: max_seq_len must be in (0, "
                           << kPackedAttentionMaxContextSeqLen << "], got " << max_seq_len_);
        return 1;
    }
    if (scale_ <= 0.0f) {
        scale_ = DefaultScale(head_size_);
    }
    return 0;
}

size_t PackedAttentionPlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t nbOutputs) const noexcept {
    (void)inputs;
    (void)nbInputs;
    (void)outputs;
    (void)nbOutputs;
    return 0;  // 两条 kernel 只用静态 shared，不需要 workspace（也不允许 enqueue 内分配）
}

int32_t PackedAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                       const nvinfer1::PluginTensorDesc* outputDesc,
                                       const void* const* inputs, void* const* outputs,
                                       void* workspace, cudaStream_t stream) noexcept {
    (void)outputDesc;
    (void)workspace;
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr) {
        return 1;
    }
    // **分派用的量全部来自形状**（不需要读设备值，也就不需要 D2H）：
    //   B_ctx   = cu_seqlens_ctx 的形状 - 1
    //   B_total = block_tables 的行数
    //   T       = query 的 token 维
    const int32_t b_ctx = inputDesc[7].dims.d[0] - 1;
    const int32_t b_total = inputDesc[5].dims.d[0];
    const int32_t t_total = inputDesc[0].dims.d[0];
    if (b_ctx < 0 || b_total < b_ctx || t_total <= 0) {
        MINI_TRT_LOG_ERROR("PackedAttention: inconsistent shapes (B_ctx="
                           << b_ctx << ", B_total=" << b_total << ", T=" << t_total << ")");
        return 1;
    }
    const int32_t b_gen = b_total - b_ctx;

    const bool is_half = (inputDesc[0].type == nvinfer1::DataType::kHALF);
    const int32_t max_blocks_per_seq = inputDesc[5].dims.d[1];
    const int32_t threads = ThreadsForHeadSize(head_size_);
    if (threads <= 0 || threads > 1024) {
        MINI_TRT_LOG_ERROR("PackedAttention: bad thread count " << threads);
        return 1;
    }

    // 两条 kernel 在同一 stream 上串行；任一段为空就跳过对应 launch。
    if (b_ctx > 0) {
        const dim3 grid(static_cast<unsigned int>(num_heads_),
                        static_cast<unsigned int>(b_ctx),
                        static_cast<unsigned int>(max_seq_len_));
        if (is_half) {
            PackedContextAttentionKernel<__half><<<grid, threads, 0, stream>>>(
                static_cast<const __half*>(inputs[0]), static_cast<const __half*>(inputs[1]),
                static_cast<const __half*>(inputs[2]), static_cast<__half*>(outputs[0]),
                static_cast<const int32_t*>(inputs[7]), num_heads_, num_kv_heads_, head_size_,
                scale_);
        } else {
            PackedContextAttentionKernel<float><<<grid, threads, 0, stream>>>(
                static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
                static_cast<const float*>(inputs[2]), static_cast<float*>(outputs[0]),
                static_cast<const int32_t*>(inputs[7]), num_heads_, num_kv_heads_, head_size_,
                scale_);
        }
    }
    if (b_gen > 0) {
        const dim3 grid(static_cast<unsigned int>(num_heads_),
                        static_cast<unsigned int>(b_gen));
        if (is_half) {
            PackedGenerationAttentionKernel<__half><<<grid, threads, 0, stream>>>(
                static_cast<const __half*>(inputs[0]), static_cast<const __half*>(inputs[1]),
                static_cast<const __half*>(inputs[2]), static_cast<const __half*>(inputs[3]),
                static_cast<const __half*>(inputs[4]),
                static_cast<const int32_t*>(inputs[5]), static_cast<const int32_t*>(inputs[6]),
                static_cast<const int32_t*>(inputs[7]), static_cast<__half*>(outputs[0]),
                b_ctx, num_heads_, num_kv_heads_, head_size_, block_size_, max_blocks_per_seq,
                scale_);
        } else {
            PackedGenerationAttentionKernel<float><<<grid, threads, 0, stream>>>(
                static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
                static_cast<const float*>(inputs[2]), static_cast<const float*>(inputs[3]),
                static_cast<const float*>(inputs[4]),
                static_cast<const int32_t*>(inputs[5]), static_cast<const int32_t*>(inputs[6]),
                static_cast<const int32_t*>(inputs[7]), static_cast<float*>(outputs[0]),
                b_ctx, num_heads_, num_kv_heads_, head_size_, block_size_, max_blocks_per_seq,
                scale_);
        }
    }
    return static_cast<int32_t>(cudaGetLastError() == cudaSuccess ? 0 : 1);
}

int32_t PackedAttentionPlugin::onShapeChange(const nvinfer1::PluginTensorDesc* in,
                                             int32_t nbInputs,
                                             const nvinfer1::PluginTensorDesc* out,
                                             int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < kInputCount) {
        return 1;
    }
    // 形状每步都在变（T 与 B_ctx 都是动态的），所以这里只做"别把 head 几何读错"的刷新：
    // head 数与 head_size 由张量形状决定，若与 configurePlugin 记录的不一致就是连线错了。
    const nvinfer1::Dims& query_dims = in[0].dims;
    const nvinfer1::Dims& key_dims = in[1].dims;
    if (query_dims.nbDims != 3 || key_dims.nbDims != 3) {
        MINI_TRT_LOG_ERROR("PackedAttention: query/key must be 3-D on shape change");
        return 1;
    }
    if (query_dims.d[1] != num_heads_ || key_dims.d[1] != num_kv_heads_ ||
        query_dims.d[2] != head_size_) {
        MINI_TRT_LOG_ERROR("PackedAttention: head geometry changed at runtime ("
                           << query_dims.d[1] << "/" << key_dims.d[1] << "/" << query_dims.d[2]
                           << " vs " << num_heads_ << "/" << num_kv_heads_ << "/" << head_size_
                           << ")");
        return 1;
    }
    return 0;
}

nvinfer1::PluginFieldCollection const* PackedAttentionPlugin::getFieldsToSerialize() noexcept {
    ResetFields();
    AddField("block_size", &block_size_, nvinfer1::PluginFieldType::kINT32, 1);
    AddField("max_seq_len", &max_seq_len_, nvinfer1::PluginFieldType::kINT32, 1);
    AddField("scale", &scale_, nvinfer1::PluginFieldType::kFLOAT, 1);
    return GetFields();
}

// =============================================================================
// PackedAttentionPluginCreator
// =============================================================================

PackedAttentionPluginCreator::PackedAttentionPluginCreator() {
    fields_.resize(3);
    fields_[0].name = "block_size";
    fields_[0].type = nvinfer1::PluginFieldType::kINT32;
    fields_[0].length = 1;
    fields_[1].name = "max_seq_len";
    fields_[1].type = nvinfer1::PluginFieldType::kINT32;
    fields_[1].length = 1;
    fields_[2].name = "scale";
    fields_[2].type = nvinfer1::PluginFieldType::kFLOAT;
    fields_[2].length = 1;
    field_collection_.nbFields = static_cast<int32_t>(fields_.size());
    field_collection_.fields = fields_.data();
}

const char* PackedAttentionPluginCreator::getPluginName() const noexcept {
    return kPackedAttentionPluginName;
}

const char* PackedAttentionPluginCreator::getPluginVersion() const noexcept {
    return kPackedAttentionPluginVersion;
}

const char* PackedAttentionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::PluginFieldCollection const* PackedAttentionPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}

nvinfer1::IPluginV3* PackedAttentionPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    (void)name;
    (void)phase;
    auto* plugin = new (std::nothrow) PackedAttentionPlugin(fc);
    return plugin;
}

PackedAttentionPluginCreator& GetPackedAttentionPluginCreator() noexcept {
    static PackedAttentionPluginCreator creator;
    return creator;
}

}  // namespace mini_trt_llm
