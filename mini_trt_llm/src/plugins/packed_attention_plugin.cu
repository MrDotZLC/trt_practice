#include "mini_trt_llm/plugins/packed_attention_plugin.hpp"

#include "mini_trt_llm/plugins/paged_attention_kernel.hpp"
#include "mini_trt_llm/plugins/paged_attention_split.hpp"
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
// score 存进 shared（上限 1024 → 4 KB），因此不需要把整行 logits 物化到显存；
// **这一段自己不占 workspace** —— 插件的 workspace 需求来自 generation 段的 split-K（见下）。
//
// **正确性优先**：context 段这是 v1，复杂度 O(T_ctx × L × D)；tiling / 在线 softmax 之类的优化
// 等 P4/P7 有数据之后再谈（与采样器 fast/legacy 的处理方式一致）。
// **generation 段不走这条**：它复用 paged 插件的 split-K（REQ-014 的生产路径），见下方注释。
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

// generation 段：**直接复用 paged 插件的 split-K**（`LaunchPagedAttentionSplit`，见 enqueue），
// 不再自写 kernel —— 那等于在默认路径上丢掉 REQ-014 的收益（REQ-014 交付的正是 split-K，
// 单趟只是它的 A/B 参考与 workspace 缺失时的兜底）。
// 段内行/token 基址由那三条 kernel 从 `cu_seqlens_ctx` **设备端**读取（默认 null/0 = 加参数前的行为）。

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
    (void)outputs;
    (void)nbOutputs;
    if (inputs == nullptr || nbInputs < 6) {
        return 0;
    }
    // **generation 段复用 split-K，所以这里不再返回 0**。与 paged 插件同一纪律：
    // 必须用 `.max`（动态轴在 `desc.dims` 里是 -1，用它算出来的 workspace 会偏小 → 越界写，
    // 见 TROUBLESHOOTING + TS-018 同类的"边界尺寸必须向对方查询"）。
    // 上界取行数（block_tables 的 max 行数）——它是 B_gen 的上界，够用且不需要宿主知道设备值。
    const nvinfer1::Dims& query_max = inputs[0].max;
    const nvinfer1::Dims& rows_max = inputs[5].max;
    if (query_max.nbDims != 3 || rows_max.nbDims != 2) {
        return 0;
    }
    return PagedAttentionWorkspaceBytes(rows_max.d[0], query_max.d[1], query_max.d[2],
                                        kPagedAttentionMaxSplits);
}

int32_t PackedAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                       const nvinfer1::PluginTensorDesc* outputDesc,
                                       const void* const* inputs, void* const* outputs,
                                       void* workspace, cudaStream_t stream) noexcept {
    (void)outputDesc;
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
        // **复用 paged 插件的 split-K**（REQ-014 的生产路径）：把这一段的行/token 基址交给那三条
        // kernel 自己在**设备端**读（`cu_seqlens_ctx` + B_ctx），workspace 槽位仍按段内 batch
        // （0..B_gen-1），所以归并 kernel 的 batch_size 口径不用变。
        // 与 paged 插件同一条兜底：workspace 拿不到时才退单趟（结果仍正确，只是没有加速）。
        PagedAttentionKernelArgs paged_args;
        paged_args.query = inputs[0];
        paged_args.key_cache = inputs[3];
        paged_args.value_cache = inputs[4];
        paged_args.block_tables = static_cast<const int32_t*>(inputs[5]);
        paged_args.context_lens = static_cast<const int32_t*>(inputs[6]);
        paged_args.key_new = inputs[1];
        paged_args.value_new = inputs[2];
        paged_args.output = outputs[0];
        paged_args.batch_size = b_gen;
        paged_args.num_heads = num_heads_;
        paged_args.num_kv_heads = num_kv_heads_;
        paged_args.head_size = head_size_;
        paged_args.block_size = block_size_;
        paged_args.max_blocks_per_seq = max_blocks_per_seq;
        paged_args.scale = scale_;
        paged_args.is_half = is_half;
        paged_args.has_current_token = true;  // packed 里那一行的 token 自带 K/V
        paged_args.cu_seqlens_ctx = static_cast<const int32_t*>(inputs[7]);
        paged_args.context_seq_count = b_ctx;
        const size_t workspace_needed =
            PagedAttentionWorkspaceBytes(b_gen, num_heads_, head_size_,
                                        kPagedAttentionMaxSplits);
        const cudaError_t err =
            (workspace != nullptr && workspace_needed > 0)
                ? LaunchPagedAttentionSplit(paged_args, workspace, workspace_needed, stream)
                : LaunchPagedAttention(paged_args, stream);
        if (err != cudaSuccess) {
            MINI_TRT_LOG_ERROR("PackedAttention: generation segment failed: "
                               << cudaGetErrorString(err));
            return 1;
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
