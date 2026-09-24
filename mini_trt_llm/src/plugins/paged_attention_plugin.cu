#include "mini_trt_llm/plugins/paged_attention_kernel.hpp"
#include "mini_trt_llm/plugins/paged_attention_plugin.hpp"
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

// blockDim 取 head_size 向上取整到 32，因此最多 32 个 warp 参与归约。
constexpr int kMaxWarps = 32;
// head_size 上限直接由 CUDA 的 blockDim 上限（1024）决定。
constexpr int32_t kMaxHeadSize = kMaxWarps * 32;

using cuda::BlockReduceSum;
using cuda::FromFloat;
using cuda::ToFloat;

// 每个 block 负责一个 (batch, head)，线程沿 head_size 维度切分。
//
// 采用 online softmax（单趟扫描）而不是"先求最大值再求归一化"的两趟方案：
// 单趟不需要按 max_context_len 预分配缓存，也不需要把整行 logits 物化到显存。
//
// 代价是每处理一个历史位置就要做一次 block 归约。这是 Phase 1 为换取正确性接受的
// 取舍；warp 级优化留到后续迭代（见 docs/phase1_development_plan.md §9）。
template <typename T>
__global__ void PagedAttentionDecodeKernel(
    const T* __restrict__ query, const T* __restrict__ key_cache,
    const T* __restrict__ value_cache, const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, const T* __restrict__ key_new,
    const T* __restrict__ value_new, T* __restrict__ output, int32_t num_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t block_size,
    int32_t max_blocks_per_seq, float scale, bool has_current_token) {
    __shared__ float reduce_scratch[kMaxWarps];

    const int32_t head = blockIdx.x;
    const int32_t batch = blockIdx.y;
    // GQA/MQA：一组 query head 共享同一个 kv head
    const int32_t kv_head = head / (num_heads / num_kv_heads);

    const int32_t d = threadIdx.x;
    const bool active = d < head_size;

    const T* query_row =
        query + (static_cast<size_t>(batch) * num_heads + head) * head_size;
    T* output_row = output + (static_cast<size_t>(batch) * num_heads + head) * head_size;

    const float query_value = active ? ToFloat(query_row[d]) : 0.0f;

    const int32_t context_len = context_lens[batch];
    const int32_t* block_table =
        block_tables + static_cast<size_t>(batch) * max_blocks_per_seq;

    float accumulator = 0.0f;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;

    // 有当前 token 时，参与 softmax 的位置数是 context_len + 1：
    // 前 context_len 个来自分页 cache，最后一个来自 key_new / value_new。
    const int32_t total_len = context_len + (has_current_token ? 1 : 0);
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
            // 当前 token 只有一份，不经过 block table
            const size_t kv_offset =
                (static_cast<size_t>(batch) * num_kv_heads + kv_head) * head_size;
            key_row = key_new + kv_offset;
            value_row = value_new + kv_offset;
        }

        float partial = 0.0f;
        if (active) {
            partial = query_value * ToFloat(key_row[d]);
        }
        const float score = BlockReduceSum(partial, reduce_scratch) * scale;

        const float new_max = fmaxf(running_max, score);
        // 首轮 running_max 为负无穷，alpha 自然为 0，无需特判
        const float alpha = __expf(running_max - new_max);
        const float probability = __expf(score - new_max);
        running_sum = running_sum * alpha + probability;
        if (active) {
            accumulator =
                accumulator * alpha + probability * ToFloat(value_row[d]);
        }
        running_max = new_max;
    }

    if (active) {
        // context_len 为 0 时 running_sum 仍为 0，必须避免除零
        output_row[d] =
            FromFloat<T>(running_sum > 0.0f ? accumulator / running_sum : 0.0f);
    }
}

}  // namespace

cudaError_t LaunchPagedAttention(const PagedAttentionKernelArgs& args, cudaStream_t stream) {
    if (args.query == nullptr || args.key_cache == nullptr || args.value_cache == nullptr ||
        args.block_tables == nullptr || args.context_lens == nullptr ||
        args.output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (args.batch_size <= 0 || args.num_heads <= 0 || args.num_kv_heads <= 0 ||
        args.head_size <= 0 || args.head_size > kMaxHeadSize || args.block_size <= 0 ||
        args.max_blocks_per_seq <= 0) {
        return cudaErrorInvalidValue;
    }
    if (args.num_heads % args.num_kv_heads != 0) {
        return cudaErrorInvalidValue;
    }
    // 声明了当前 token 就必须真的给出 K/V：漏给会让注意力少一项，
    // 结果是"能跑但数值错"，所以在这里直接拒绝。
    if (args.has_current_token && (args.key_new == nullptr || args.value_new == nullptr)) {
        return cudaErrorInvalidValue;
    }

    // CUDA 的 last-error 是粘性的：先清掉入口处可能残留的旧错误（例如别处故意触发的失败），
    // 后面 cudaGetLastError() 的结果才只反映本次 launch。
    (void)cudaGetLastError();

    // blockDim 必须是 32 的整数倍，BlockReduceSum 的 warp 内 shuffle 才安全
    const int32_t threads = ((args.head_size + 31) / 32) * 32;
    const dim3 grid(static_cast<unsigned int>(args.num_heads),
                    static_cast<unsigned int>(args.batch_size));
    const dim3 block(static_cast<unsigned int>(threads));

    if (args.is_half) {
        PagedAttentionDecodeKernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(args.query),
            static_cast<const __half*>(args.key_cache),
            static_cast<const __half*>(args.value_cache), args.block_tables,
            args.context_lens, static_cast<const __half*>(args.key_new),
            static_cast<const __half*>(args.value_new),
            static_cast<__half*>(args.output), args.num_heads, args.num_kv_heads,
            args.head_size, args.block_size, args.max_blocks_per_seq, args.scale,
            args.has_current_token);
    } else {
        PagedAttentionDecodeKernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(args.query),
            static_cast<const float*>(args.key_cache),
            static_cast<const float*>(args.value_cache), args.block_tables,
            args.context_lens, static_cast<const float*>(args.key_new),
            static_cast<const float*>(args.value_new),
            static_cast<float*>(args.output), args.num_heads, args.num_kv_heads,
            args.head_size, args.block_size, args.max_blocks_per_seq, args.scale,
            args.has_current_token);
    }
    return cudaGetLastError();
}

// =============================================================================
// PagedAttentionPlugin
// =============================================================================

PagedAttentionPlugin::PagedAttentionPlugin(int32_t num_heads, int32_t num_kv_heads,
                                           int32_t head_size, int32_t block_size,
                                           float scale)
    : num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      block_size_(block_size),
      scale_(scale),
      serialized_block_size_(block_size),
      serialized_scale_(scale) {}

PagedAttentionPlugin::PagedAttentionPlugin(const nvinfer1::PluginFieldCollection* fc) {
    if (fc != nullptr && fc->fields != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const nvinfer1::PluginField& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr) {
                continue;
            }
            const std::string name(field.name);
            if (name == "block_size") {
                block_size_ = *static_cast<const int32_t*>(field.data);
            } else if (name == "scale") {
                scale_ = *static_cast<const float*>(field.data);
            }
        }
    }
    serialized_block_size_ = block_size_;
    serialized_scale_ = scale_;
}

IPluginV3Base* PagedAttentionPlugin::clone() noexcept {
    auto* copy = new (std::nothrow) PagedAttentionPlugin(num_heads_, num_kv_heads_,
                                                         head_size_, block_size_, scale_);
    if (copy != nullptr) {
        copy->SetNamespace(GetPluginNamespaceInternal());
    }
    return copy;
}

const char* PagedAttentionPlugin::getPluginType() const noexcept {
    return kPagedAttentionPluginName;
}

const char* PagedAttentionPlugin::getPluginVersion() const noexcept {
    return kPagedAttentionPluginVersion;
}

int32_t PagedAttentionPlugin::getNbOutputs() const noexcept { return 1; }

int32_t PagedAttentionPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes,
                                                 int32_t nbOutputs,
                                                 const nvinfer1::DataType* inputTypes,
                                                 int32_t nbInputs) const noexcept {
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs < 1 || nbInputs < 1) {
        return 1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t PagedAttentionPlugin::getOutputShapes(const nvinfer1::DimsExprs* inputs,
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
    outputs[0] = inputs[0];
    return 0;
}

bool PagedAttentionPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut == nullptr || pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    const nvinfer1::PluginTensorDesc& desc = inOut[pos].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }

    // pos 3/4 分别是 block_tables 与 context_lens，固定为 INT32
    if (pos == 3 || pos == 4) {
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

int32_t PagedAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                              int32_t nbInputs,
                                              const nvinfer1::DynamicPluginTensorDesc* out,
                                              int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < 5) {
        return 1;
    }

    const nvinfer1::Dims& query_dims = in[0].desc.dims;
    const nvinfer1::Dims& cache_dims = in[1].desc.dims;
    if (query_dims.nbDims != 4 || cache_dims.nbDims != 4) {
        MINI_TRT_LOG_ERROR("PagedAttention: query/cache must be 4-D");
        return 1;
    }
    // 第 6/7 个输入是当前 token 的 K/V（可选）。传了就必须成对、
    // 且形状为 [batch, num_kv_heads, 1, head_size]——半连接状态只会静默丢掉自注意力项。
    has_current_token_ = nbInputs >= 7;
    if (nbInputs == 6) {
        MINI_TRT_LOG_ERROR("PagedAttention: key_new and value_new must be connected together");
        return 1;
    }
    if (has_current_token_) {
        const nvinfer1::Dims& key_new_dims = in[5].desc.dims;
        const nvinfer1::Dims& value_new_dims = in[6].desc.dims;
        if (key_new_dims.nbDims != 4 || value_new_dims.nbDims != 4) {
            MINI_TRT_LOG_ERROR("PagedAttention: key_new/value_new must be 4-D");
            return 1;
        }
        if (key_new_dims.d[2] > 0 && key_new_dims.d[2] != 1) {
            MINI_TRT_LOG_ERROR("PagedAttention: key_new seq dim must be 1");
            return 1;
        }
        if (value_new_dims.d[2] > 0 && value_new_dims.d[2] != 1) {
            MINI_TRT_LOG_ERROR("PagedAttention: value_new seq dim must be 1");
            return 1;
        }
    }

    // Phase 1 只支持 Decoding：query 的 seq_len 必须为 1
    if (query_dims.d[2] > 0 && query_dims.d[2] != 1) {
        MINI_TRT_LOG_ERROR("PagedAttention: prefill (seq_len > 1) is not supported in Phase 1");
        return 1;
    }

    // Q5：block_size 强制显式配置。不同 block_size 对应不同的缓存布局，静默补默认值
    // 会让 engine 与 KV Cache 分配策略悄悄不一致。
    if (block_size_ <= 0) {
        MINI_TRT_LOG_ERROR("PagedAttention: block_size must be configured explicitly");
        return 1;
    }
    if (cache_dims.d[1] > 0 && cache_dims.d[1] != block_size_) {
        MINI_TRT_LOG_ERROR("PagedAttention: cache block dim "
                           << cache_dims.d[1] << " does not match block_size " << block_size_);
        return 1;
    }

    if (query_dims.d[1] > 0) {
        num_heads_ = query_dims.d[1];
    }
    if (cache_dims.d[2] > 0) {
        num_kv_heads_ = cache_dims.d[2];
    }
    if (query_dims.d[3] > 0) {
        head_size_ = query_dims.d[3];
    }
    if (num_heads_ <= 0 || num_kv_heads_ <= 0 || head_size_ <= 0) {
        MINI_TRT_LOG_ERROR("PagedAttention: head configuration is unknown");
        return 1;
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        MINI_TRT_LOG_ERROR("PagedAttention: num_heads "
                           << num_heads_ << " is not divisible by num_kv_heads "
                           << num_kv_heads_);
        return 1;
    }
    if (cache_dims.d[3] > 0 && cache_dims.d[3] != head_size_) {
        MINI_TRT_LOG_ERROR("PagedAttention: cache head dim "
                           << cache_dims.d[3] << " does not match head_size " << head_size_);
        return 1;
    }

    // Q6：scale 作为属性，未显式配置时按 1/sqrt(head_size) 推导
    if (scale_ <= 0.0f) {
        scale_ = DefaultScale(head_size_);
    }
    return 0;
}

size_t PagedAttentionPlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t nbOutputs) const noexcept {
    (void)inputs;
    (void)nbInputs;
    (void)outputs;
    (void)nbOutputs;
    // online softmax 只用到 static shared memory
    return 0;
}

int32_t PagedAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                      const nvinfer1::PluginTensorDesc* outputDesc,
                                      const void* const* inputs, void* const* outputs,
                                      void* workspace, cudaStream_t stream) noexcept {
    (void)outputDesc;
    (void)workspace;
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr) {
        return 1;
    }

    const nvinfer1::Dims& query_dims = inputDesc[0].dims;
    const nvinfer1::Dims& block_table_dims = inputDesc[3].dims;
    if (query_dims.nbDims != 4 || block_table_dims.nbDims != 2) {
        return 1;
    }

    PagedAttentionKernelArgs args;
    args.query = inputs[0];
    args.key_cache = inputs[1];
    args.value_cache = inputs[2];
    args.block_tables = static_cast<const int32_t*>(inputs[3]);
    args.context_lens = static_cast<const int32_t*>(inputs[4]);
    // has_current_token_ 由 configurePlugin / onShapeChange 依 nbInputs 刷新，
    // 两者都先于 enqueue（反序列化路径同样会走 onShapeChange，见 TROUBLESHOOTING #8）。
    args.has_current_token = has_current_token_;
    if (has_current_token_) {
        args.key_new = inputs[5];
        args.value_new = inputs[6];
    }
    args.output = outputs[0];
    args.batch_size = query_dims.d[0];
    args.num_heads = query_dims.d[1];
    args.head_size = query_dims.d[3];
    // 兜底：若未经 onShapeChange 直接进入 enqueue，num_kv_heads_ 可能仍是默认值，此时取 cache 形状
    const nvinfer1::Dims& cache_dims = inputDesc[1].dims;
    args.num_kv_heads = num_kv_heads_ > 0
                            ? num_kv_heads_
                            : (cache_dims.nbDims == 4 ? cache_dims.d[2] : 0);
    args.block_size = block_size_;
    args.max_blocks_per_seq = block_table_dims.d[1];
    args.scale = scale_ > 0.0f ? scale_ : DefaultScale(args.head_size);
    args.is_half = (inputDesc[0].type == nvinfer1::DataType::kHALF);

    const cudaError_t err = LaunchPagedAttention(args, stream);
    if (err != cudaSuccess) {
        MINI_TRT_LOG_ERROR("PagedAttention enqueue failed: " << cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int32_t PagedAttentionPlugin::onShapeChange(const nvinfer1::PluginTensorDesc* in,
                                            int32_t nbInputs,
                                            const nvinfer1::PluginTensorDesc* out,
                                            int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < 5) {
        return 1;
    }
    const nvinfer1::Dims& query_dims = in[0].dims;
    const nvinfer1::Dims& cache_dims = in[1].dims;
    if (query_dims.nbDims != 4 || cache_dims.nbDims != 4) {
        return 1;
    }
    // arity 是网络连线的直接结果，序列化往返后由 TRT 原样恢复，因此每次形状变化
    // 都按 nbInputs 刷新，不额外做序列化属性（同 §2.12 对 RoPE head 配置的处理）。
    has_current_token_ = nbInputs >= 7;
    if (nbInputs == 6) {
        MINI_TRT_LOG_ERROR("PagedAttention: key_new and value_new must be connected together");
        return 1;
    }
    if (query_dims.d[2] != 1) {
        MINI_TRT_LOG_ERROR("PagedAttention: shape change requests prefill (seq_len > 1)");
        return 1;
    }
    // 同 RoPE：反序列化后的实例没经过 configurePlugin，head 配置成员仍是默认值 0，
    // 因此以运行期形状为准刷新。block_size / scale 是序列化属性，可以放心校验。
    if (query_dims.d[1] > 0) {
        num_heads_ = query_dims.d[1];
    }
    if (cache_dims.d[2] > 0) {
        num_kv_heads_ = cache_dims.d[2];
    }
    if (query_dims.d[3] > 0) {
        head_size_ = query_dims.d[3];
    }
    if (block_size_ <= 0) {
        MINI_TRT_LOG_ERROR("PagedAttention: block_size missing on deserialized plugin");
        return 1;
    }
    if (cache_dims.d[1] > 0 && cache_dims.d[1] != block_size_) {
        MINI_TRT_LOG_ERROR("PagedAttention: cache block dim "
                           << cache_dims.d[1] << " does not match block_size " << block_size_);
        return 1;
    }
    if (num_heads_ <= 0 || num_kv_heads_ <= 0 || head_size_ <= 0) {
        MINI_TRT_LOG_ERROR("PagedAttention: head configuration is unknown");
        return 1;
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        MINI_TRT_LOG_ERROR("PagedAttention: num_heads "
                           << num_heads_ << " is not divisible by num_kv_heads "
                           << num_kv_heads_);
        return 1;
    }
    if (query_dims.d[3] > 0 && cache_dims.d[3] > 0 && query_dims.d[3] != cache_dims.d[3]) {
        MINI_TRT_LOG_ERROR("PagedAttention: cache head dim "
                           << cache_dims.d[3] << " does not match query head_size "
                           << query_dims.d[3]);
        return 1;
    }
    if (scale_ <= 0.0f) {
        scale_ = DefaultScale(head_size_);
    }
    return 0;
}

nvinfer1::PluginFieldCollection const* PagedAttentionPlugin::getFieldsToSerialize() noexcept {
    serialized_block_size_ = block_size_;
    serialized_scale_ = scale_;

    ResetFields();
    AddField("block_size", &serialized_block_size_,
             static_cast<int32_t>(nvinfer1::PluginFieldType::kINT32), 1);
    AddField("scale", &serialized_scale_,
             static_cast<int32_t>(nvinfer1::PluginFieldType::kFLOAT32), 1);
    return GetFields();
}

// =============================================================================
// PagedAttentionPluginCreator
// =============================================================================

PagedAttentionPluginCreator::PagedAttentionPluginCreator() {
    fields_.push_back(nvinfer1::PluginField{"block_size", nullptr,
                                            nvinfer1::PluginFieldType::kINT32, 1});
    fields_.push_back(
        nvinfer1::PluginField{"scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    field_collection_.nbFields = static_cast<int32_t>(fields_.size());
    field_collection_.fields = fields_.data();
}

const char* PagedAttentionPluginCreator::getPluginName() const noexcept {
    return kPagedAttentionPluginName;
}

const char* PagedAttentionPluginCreator::getPluginVersion() const noexcept {
    return kPagedAttentionPluginVersion;
}

const char* PagedAttentionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::PluginFieldCollection const* PagedAttentionPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}

nvinfer1::IPluginV3* PagedAttentionPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    (void)name;
    (void)phase;
    auto* plugin = new (std::nothrow) PagedAttentionPlugin(fc);
    if (plugin == nullptr) {
        return nullptr;
    }
    plugin->SetNamespace(namespace_.c_str());
    return plugin;
}

PagedAttentionPluginCreator& GetPagedAttentionPluginCreator() noexcept {
    static PagedAttentionPluginCreator creator;
    return creator;
}

REGISTER_TENSORRT_PLUGIN(PagedAttentionPluginCreator);

}  // namespace mini_trt_llm
