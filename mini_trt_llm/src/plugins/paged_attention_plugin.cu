#include "mini_trt_llm/plugins/paged_attention_kernel.hpp"
#include "mini_trt_llm/plugins/paged_attention_plugin.hpp"
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

// blockDim 取 head_size 向上取整到 32，因此最多 32 个 warp 参与归约。
constexpr int kMaxWarps = 32;
// head_size 上限直接由 CUDA 的 blockDim 上限（1024）决定。
constexpr int32_t kMaxHeadSize = kMaxWarps * 32;

using cuda::BlockReduceSum;
using cuda::FromFloat;
using cuda::ToFloat;

// 测试用的分片数覆盖（0 = 自适应）。放在 TU 级：`LaunchPagedAttentionSplit` 读它，
// 测试通过 `SetPagedAttentionNumSplitsOverride` 写它。生产路径从不设置 → 恒为 0。
int32_t g_num_splits_override = 0;

// 每个 block 负责一个 (batch, head)，线程沿 head_size 维度切分。
//
// 采用 online softmax（单趟扫描）而不是"先求最大值再求归一化"的两趟方案：
// 单趟不需要按 max_context_len 预分配缓存，也不需要把整行 logits 物化到显存。
//
// 代价是每处理一个历史位置就要做一次 block 归约。这是 Phase 1 为换取正确性接受的
// 取舍；warp 级优化留到后续迭代（见 docs/phase1_development_plan.md + PH1-RISKS）。
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

// split-K 第一阶段：每个 (head, batch, split) 只扫**自己那一段**逻辑位置，产出本片的
// 局部三元组 (m, l, acc[0..head_size))，写进 workspace。
//
// 为什么把上下文维切开：改动前每层只有 `num_heads × batch` 个 block（GPT-2 是 12 个），
// 而本机 24 个 SM —— 一半闲置，且每块要串行扫完 976 个位置。切分后并行度 = 12 × 8 = 96 块。
// 参考实现（单趟）保留在 `PagedAttentionDecodeKernel`，供 A/B 与兜底使用。
//
// **网格恒为 `kMaxSplits` 层**：宿主侧拿不到"当前上下文有多长"（`block_tables` 形状固定、
// `context_lens` 在设备上，见开发计划 §12.3 D2 的落地修正），所以片数只能由设备端按
// 自己那本 `context_lens[b]` 推导；超出有效片数的 block **立即返回、不读不写**。
template <typename T>
__global__ void PagedAttentionSplitKernel(
    const T* __restrict__ query, const T* __restrict__ key_cache,
    const T* __restrict__ value_cache, const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, const T* __restrict__ key_new,
    const T* __restrict__ value_new, float* __restrict__ workspace, int32_t num_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t block_size,
    int32_t max_blocks_per_seq, float scale, bool has_current_token,
    int32_t override_splits) {
    __shared__ float reduce_scratch[kMaxWarps];

    const int32_t head = blockIdx.x;
    const int32_t batch = blockIdx.y;
    const int32_t split = blockIdx.z;
    // grid = (num_heads, batch, kMaxSplits) → gridDim.y 就是 batch_size，
    // 与 `PagedAttentionWorkspaceSlotOffset` 需要的 batch_size 同一来源，避免两处各传一份。
    const int32_t batch_size = gridDim.y;

    const int32_t context_len = context_lens[batch];
    const int32_t total_len = context_len + (has_current_token ? 1 : 0);
    const int32_t effective = PagedAttentionResolveSplits(total_len, override_splits);
    // 本 batch 不需要这么多片（或压根没有位置）→ 不读不写。stage-2 也只读 [0, effective)。
    if (split >= effective) {
        return;
    }

    const int32_t d = threadIdx.x;
    const bool active = d < head_size;
    const int32_t kv_head = head / (num_heads / num_kv_heads);

    int32_t begin = 0;
    int32_t end = 0;
    PagedAttentionSplitRange(total_len, split, effective, &begin, &end);

    float* slot = workspace + PagedAttentionWorkspaceSlotOffset(split, batch, head,
                                                                batch_size, num_heads,
                                                                head_size);
    const T* query_row =
        query + (static_cast<size_t>(batch) * num_heads + head) * head_size;
    const float query_value = active ? ToFloat(query_row[d]) : 0.0f;
    const int32_t* block_table =
        block_tables + static_cast<size_t>(batch) * max_blocks_per_seq;

    float accumulator = 0.0f;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;

    for (int32_t t = begin; t < end; ++t) {
        const T* key_row;
        const T* value_row;
        if (t < context_len) {
            const int32_t physical_block = block_table[t / block_size];
            const int32_t slot_in_block = t % block_size;
            const size_t kv_offset =
                ((static_cast<size_t>(physical_block) * block_size + slot_in_block) *
                     num_kv_heads +
                 kv_head) *
                head_size;
            key_row = key_cache + kv_offset;
            value_row = value_cache + kv_offset;
        } else {
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
        const float alpha = __expf(running_max - new_max);
        const float probability = __expf(score - new_max);
        running_sum = running_sum * alpha + probability;
        if (active) {
            accumulator = accumulator * alpha + probability * ToFloat(value_row[d]);
        }
        running_max = new_max;
    }

    // **空片也要显式写哨兵**（m = -inf / l = 0）：stage-2 用 `l <= 0` 判定"这片没内容"。
    // 不写就会读到 workspace 上一轮的残留值 —— 输出与历史调用有关（PROGRESS.md §2.12 /
    // TROUBLESHOOTING #4 是同一类坑）。
    if (threadIdx.x == 0) {
        slot[0] = running_max;
        slot[1] = running_sum;
    }
    if (active) {
        slot[2 + d] = accumulator;
    }
}

// split-K 第二阶段：把各片的 (m, l, acc[]) 用 max-trick 归并成最终输出。
//
//   global_max = max_i m_i
//   l          = Σ_i l_i · exp(m_i − global_max)
//   acc[d]     = Σ_i acc_i[d] · exp(m_i − global_max)
//   out[d]     = acc[d] / l
//
// **不用原子累加**：原子加无法保序 → 结果随调度变化、不可复现（开发计划 §12.3 D2）。
// 归并严格按 split 下标递增顺序进行，所以同样的输入必然得到同样的输出。
template <typename T>
__global__ void PagedAttentionMergeKernel(const float* __restrict__ workspace,
                                          const int32_t* __restrict__ context_lens,
                                          T* __restrict__ output, int32_t num_heads,
                                          int32_t head_size, bool has_current_token,
                                          int32_t override_splits) {
    const int32_t head = blockIdx.x;
    const int32_t batch = blockIdx.y;
    const int32_t batch_size = gridDim.y;
    const int32_t d = threadIdx.x;
    const bool active = d < head_size;

    T* output_row = output + (static_cast<size_t>(batch) * num_heads + head) * head_size;
    const int32_t total_len = context_lens[batch] + (has_current_token ? 1 : 0);
    const int32_t effective = PagedAttentionResolveSplits(total_len, override_splits);
    if (effective <= 0) {
        // 与单趟 kernel 的兜底一致：没有任何位置可看时输出 0（那里是 `running_sum > 0` 判据）
        if (active) {
            output_row[d] = FromFloat<T>(0.0f);
        }
        return;
    }

    const float* first = workspace + PagedAttentionWorkspaceSlotOffset(
                                         0, batch, head, batch_size, num_heads, head_size);
    const size_t per_split = static_cast<size_t>(batch_size) *
                             static_cast<size_t>(num_heads) *
                             static_cast<size_t>(PagedAttentionWorkspaceStride(head_size));

    float global_max = -CUDART_INF_F;
    for (int32_t s = 0; s < effective; ++s) {
        global_max = fmaxf(global_max, first[static_cast<size_t>(s) * per_split]);
    }

    float running_sum = 0.0f;
    float accumulator = 0.0f;
    for (int32_t s = 0; s < effective; ++s) {
        const float* slot = first + static_cast<size_t>(s) * per_split;
        const float local_max = slot[0];
        const float local_sum = slot[1];
        // 空片（l == 0）先跳过再算 exp —— 这样即使 global_max 还是 -inf，
        // 也不会出现 exp(-inf - -inf) = NaN。非空片的 l >= 1（至少有一个位置取到 max）。
        if (local_sum <= 0.0f) {
            continue;
        }
        const float weight = __expf(local_max - global_max);
        running_sum += local_sum * weight;
        if (active) {
            accumulator += slot[2 + d] * weight;
        }
    }

    if (active) {
        output_row[d] =
            FromFloat<T>(running_sum > 0.0f ? accumulator / running_sum : 0.0f);
    }
}

}  // namespace

namespace {

// 两条启动路径（单趟 / split-K）共用的入参校验。
//
// **为什么抽出来**：两份校验迟早会漂移，而"两条路径的拒绝条件不同"意味着同一个输入在
// 一条路上是错误、在另一条路上"能跑但数值错"——那是最难查的一类偏差（`PROGRESS.md` §2.13
// 对"参考实现唯一"的同一要求）。
cudaError_t ValidatePagedAttentionArgs(const PagedAttentionKernelArgs& args) {
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
    return cudaSuccess;
}

}  // namespace

cudaError_t LaunchPagedAttention(const PagedAttentionKernelArgs& args, cudaStream_t stream) {
    const cudaError_t invalid = ValidatePagedAttentionArgs(args);
    if (invalid != cudaSuccess) {
        return invalid;
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

cudaError_t LaunchPagedAttentionSplit(const PagedAttentionKernelArgs& args, void* workspace,
                                      size_t workspace_bytes, cudaStream_t stream) {
    const cudaError_t invalid = ValidatePagedAttentionArgs(args);
    if (invalid != cudaSuccess) {
        return invalid;
    }
    // 契约：调用方按 `PagedAttentionWorkspaceBytes(..., kPagedAttentionMaxSplits)` 预留。
    // 这里按**运行期**形状再算一遍并比对——布局函数只有一份（paged_attention_split.hpp），
    // 所以不一致只可能来自"调用方自己另算了一遍"。
    const size_t needed =
        PagedAttentionWorkspaceBytes(args.batch_size, args.num_heads, args.head_size,
                                     kPagedAttentionMaxSplits);
    if (workspace == nullptr || needed == 0 || workspace_bytes < needed) {
        return cudaErrorInvalidValue;
    }

    (void)cudaGetLastError();

    // blockDim 与单趟版本一致（每线程负责 head_size 维里的一个 d），这样两条路径的
    // 归约语义、精度、边界处理都不需要各自再证一遍。
    const int32_t threads = ((args.head_size + 31) / 32) * 32;
    const dim3 block(static_cast<unsigned int>(threads));
    // z 维恒为上限：片数由设备端按各自 `context_lens[b]` 推导（宿主侧拿不到，见内核注释）
    const dim3 split_grid(static_cast<unsigned int>(args.num_heads),
                          static_cast<unsigned int>(args.batch_size),
                          static_cast<unsigned int>(kPagedAttentionMaxSplits));
    const dim3 merge_grid(static_cast<unsigned int>(args.num_heads),
                          static_cast<unsigned int>(args.batch_size));
    const int32_t override_splits = g_num_splits_override;

    if (args.is_half) {
        PagedAttentionSplitKernel<__half><<<split_grid, block, 0, stream>>>(
            static_cast<const __half*>(args.query),
            static_cast<const __half*>(args.key_cache),
            static_cast<const __half*>(args.value_cache), args.block_tables,
            args.context_lens, static_cast<const __half*>(args.key_new),
            static_cast<const __half*>(args.value_new),
            static_cast<float*>(workspace), args.num_heads, args.num_kv_heads,
            args.head_size, args.block_size, args.max_blocks_per_seq, args.scale,
            args.has_current_token, override_splits);
        PagedAttentionMergeKernel<__half><<<merge_grid, block, 0, stream>>>(
            static_cast<const float*>(workspace), args.context_lens,
            static_cast<__half*>(args.output), args.num_heads, args.head_size,
            args.has_current_token, override_splits);
    } else {
        PagedAttentionSplitKernel<float><<<split_grid, block, 0, stream>>>(
            static_cast<const float*>(args.query),
            static_cast<const float*>(args.key_cache),
            static_cast<const float*>(args.value_cache), args.block_tables,
            args.context_lens, static_cast<const float*>(args.key_new),
            static_cast<const float*>(args.value_new),
            static_cast<float*>(workspace), args.num_heads, args.num_kv_heads,
            args.head_size, args.block_size, args.max_blocks_per_seq, args.scale,
            args.has_current_token, override_splits);
        PagedAttentionMergeKernel<float><<<merge_grid, block, 0, stream>>>(
            static_cast<const float*>(workspace), args.context_lens,
            static_cast<float*>(args.output), args.num_heads, args.head_size,
            args.has_current_token, override_splits);
    }
    return cudaGetLastError();
}

void SetPagedAttentionNumSplitsOverride(int32_t splits) noexcept {
    // 负数**不钳到 0**：`< 0` 是"强制旧单趟路径"的语义（A/B 用），钳掉它 A/B 就失效了。
    g_num_splits_override = splits;
}

int32_t PagedAttentionNumSplitsOverride() noexcept { return g_num_splits_override; }

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
    (void)outputs;
    (void)nbOutputs;
    if (inputs == nullptr || nbInputs < 1) {
        return 0;
    }
    // **必须用 `.max`，不能用 `desc.dims`**：动态轴在 `desc.dims` 里是 -1
    // （TRT 头文件原话："desc.dims has -1 in place of any runtime dimension"），
    // 拿它算出来的 workspace 会偏小，而 kernel 照样按运行期形状往里写 → 越界写。
    // 这与 `TROUBLESHOOTING` #18（按假定精度分配缓冲）是同一类错误的两种形态：
    // **边界尺寸必须向对方查询，不能自己假定**。
    const nvinfer1::Dims& query_max = inputs[0].max;
    if (query_max.nbDims != 4) {
        return 0;
    }
    return PagedAttentionWorkspaceBytes(query_max.d[0], query_max.d[1], query_max.d[3],
                                        kPagedAttentionMaxSplits);
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

    // 生产路径 = split-K（上下文维切开）。`getWorkspaceSize` 用的是**构建期上界**（`.max`），
    // 而这里算的是**运行期**需求；上界 ≥ 运行期，所以 TRT 给的缓冲一定够。
    // 唯一会走到兜底的情况是"上界拿不到"（`getWorkspaceSize` 当时返回 0、TRT 便不分配）——
    // 那时退回单趟 kernel：**结果仍然正确，只是没有加速**，比直接失败更合适。
    const size_t workspace_needed =
        PagedAttentionWorkspaceBytes(args.batch_size, args.num_heads, args.head_size,
                                     kPagedAttentionMaxSplits);
    // `< 0` = 测试开关"强制旧单趟路径"（同二进制 A/B 用）；生产路径从不设置它。
    const bool force_single_pass = PagedAttentionNumSplitsOverride() < 0;
    cudaError_t err = cudaErrorInvalidValue;
    if (!force_single_pass && workspace != nullptr && workspace_needed > 0) {
        err = LaunchPagedAttentionSplit(args, workspace, workspace_needed, stream);
    } else {
        static bool warned_once = false;
        // 只在"本该有 workspace 却没有"时告警；A/B 主动选单趟不是异常。
        if (!force_single_pass && !warned_once) {
            warned_once = true;
            MINI_TRT_LOG_WARN("PagedAttention: workspace unavailable at runtime "
                              << "(batch=" << args.batch_size << " heads=" << args.num_heads
                              << " head_size=" << args.head_size
                              << ", computed need=" << workspace_needed
                              << " B); falling back to the single-pass kernel "
                                 "(correct but not accelerated)");
        }
        err = LaunchPagedAttention(args, stream);
    }
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
