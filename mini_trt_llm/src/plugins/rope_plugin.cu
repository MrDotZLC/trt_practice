#include "mini_trt_llm/plugins/rope_kernel.hpp"
#include "mini_trt_llm/plugins/rope_plugin.hpp"
#include "mini_trt_llm/utils/cuda_dtype.cuh"
#include "mini_trt_llm/utils/logger.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <new>
#include <string>

namespace mini_trt_llm {
namespace {

constexpr int kThreadsPerBlock = 256;

using cuda::FromFloat;
using cuda::ToFloat;

// 每个线程负责一个输出元素（batch, head, position, dim）。
//
// 角度按公式实时计算而非预计算 sin/cos 表：position_ids 允许非连续（KV Cache 续写），
// 建表需要按 max(position) 定尺寸，反而引入额外 workspace 与边界判断。
//
// 之所以按"输出元素"而不是"旋转对"来分配线程：query_out / key_out 是与输入分离的
// buffer，当 rotary_dim < head_size 时 [rotary_dim, head_size) 这一段没有任何线程会写，
// 不显式拷贝就会在输出里留下未初始化数据（rotary_dim == head_size 时整行都被覆盖，
// 因此这个缺陷只在部分旋转场景暴露）。
template <typename T>
__global__ void RoPEApplyKernel(const T* __restrict__ input,
                                const int32_t* __restrict__ position_ids,
                                T* __restrict__ output, int32_t num_heads, int32_t seq_len,
                                int32_t head_size, int32_t rotary_dim, int32_t half_rotary,
                                float base, int64_t total_elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_elements) {
        return;
    }

    const int32_t dim = static_cast<int32_t>(index % head_size);
    const int64_t row = index / head_size;  // row = (batch * num_heads + head) * seq_len + s

    const T* row_in = input + row * head_size;
    T* row_out = output + row * head_size;

    // 尾部不参与旋转，但仍必须写入输出
    if (dim >= rotary_dim) {
        row_out[dim] = row_in[dim];
        return;
    }

    const int position = static_cast<int>(row % seq_len);
    const int64_t batch_head = row / seq_len;
    const int batch = static_cast<int>(batch_head / num_heads);

    const int32_t pos = position_ids[static_cast<int64_t>(batch) * seq_len + position];

    // 与 HuggingFace 的 inv_freq = 1 / base^(arange(0, dim, 2)/dim) 等价
    const int32_t pair = (dim < half_rotary) ? dim : dim - half_rotary;
    const float inv_freq =
        powf(base, -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_v = cosf(angle);
    const float sin_v = sinf(angle);

    const float lo = ToFloat(row_in[pair]);
    const float hi = ToFloat(row_in[pair + half_rotary]);
    row_out[dim] = FromFloat<T>((dim < half_rotary) ? (lo * cos_v - hi * sin_v)
                                                    : (hi * cos_v + lo * sin_v));
}

template <typename T>
cudaError_t LaunchOne(const T* input, const int32_t* position_ids, T* output,
                      int32_t batch_size, int32_t num_heads, int32_t seq_len,
                      int32_t head_size, int32_t rotary_dim, float base,
                      cudaStream_t stream) {
    const int32_t half_rotary = rotary_dim / 2;
    const int64_t total_elements =
        static_cast<int64_t>(batch_size) * num_heads * seq_len * head_size;
    if (total_elements <= 0) {
        return cudaErrorInvalidValue;
    }
    const int64_t blocks = (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > 0x7fffffffLL) {
        return cudaErrorInvalidValue;
    }

    // CUDA 的 last-error 是**粘性**的：任何一次失败的 runtime 调用都会把它设上，
    // 直到有人读走。若不清掉入口处的残留，launch 之后的 cudaGetLastError() 读到的
    // 可能是完全无关的旧错误（例如别处故意触发的失败），把一次成功的 launch 判成失败。
    // 先清一次，后面的结果才只反映本次 launch。
    (void)cudaGetLastError();
    RoPEApplyKernel<T><<<static_cast<unsigned int>(blocks), kThreadsPerBlock, 0, stream>>>(
        input, position_ids, output, num_heads, seq_len, head_size, rotary_dim, half_rotary,
        base, total_elements);
    return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchRoPE(const RoPEKernelArgs& args, cudaStream_t stream) {
    if (args.query == nullptr || args.key == nullptr || args.position_ids == nullptr ||
        args.query_out == nullptr || args.key_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (args.batch_size <= 0 || args.seq_len <= 0 || args.num_heads <= 0 ||
        args.num_kv_heads <= 0 || args.head_size <= 0) {
        return cudaErrorInvalidValue;
    }
    // rotary_dim 必须为偶数才能成对旋转；上限是 head_size
    if (args.rotary_dim <= 0 || args.rotary_dim > args.head_size ||
        (args.rotary_dim % 2) != 0) {
        return cudaErrorInvalidValue;
    }
    // GQA/MQA 要求 num_heads 能被 num_kv_heads 整除
    if (args.num_heads % args.num_kv_heads != 0) {
        return cudaErrorInvalidValue;
    }

    if (args.is_half) {
        const auto* q = static_cast<const __half*>(args.query);
        const auto* k = static_cast<const __half*>(args.key);
        auto* q_out = static_cast<__half*>(args.query_out);
        auto* k_out = static_cast<__half*>(args.key_out);
        cudaError_t err = LaunchOne(q, args.position_ids, q_out, args.batch_size,
                                    args.num_heads, args.seq_len, args.head_size,
                                    args.rotary_dim, args.base, stream);
        if (err != cudaSuccess) {
            return err;
        }
        return LaunchOne(k, args.position_ids, k_out, args.batch_size, args.num_kv_heads,
                         args.seq_len, args.head_size, args.rotary_dim, args.base, stream);
    }

    const auto* q = static_cast<const float*>(args.query);
    const auto* k = static_cast<const float*>(args.key);
    auto* q_out = static_cast<float*>(args.query_out);
    auto* k_out = static_cast<float*>(args.key_out);
    cudaError_t err =
        LaunchOne(q, args.position_ids, q_out, args.batch_size, args.num_heads, args.seq_len,
                  args.head_size, args.rotary_dim, args.base, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return LaunchOne(k, args.position_ids, k_out, args.batch_size, args.num_kv_heads,
                     args.seq_len, args.head_size, args.rotary_dim, args.base, stream);
}

// =============================================================================
// RoPEPlugin
// =============================================================================

RoPEPlugin::RoPEPlugin(int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                       int32_t rotary_dim, float base)
    : num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      rotary_dim_(rotary_dim),
      base_(base),
      serialized_rotary_dim_(rotary_dim),
      serialized_base_(base) {}

RoPEPlugin::RoPEPlugin(const nvinfer1::PluginFieldCollection* fc) {
    if (fc != nullptr && fc->fields != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const nvinfer1::PluginField& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr) {
                continue;
            }
            const std::string name(field.name);
            if (name == "rotary_dim") {
                rotary_dim_ = *static_cast<const int32_t*>(field.data);
            } else if (name == "base") {
                base_ = *static_cast<const float*>(field.data);
            }
        }
    }
    serialized_rotary_dim_ = rotary_dim_;
    serialized_base_ = base_;
}

IPluginV3Base* RoPEPlugin::clone() noexcept {
    auto* copy = new (std::nothrow)
        RoPEPlugin(num_heads_, num_kv_heads_, head_size_, rotary_dim_, base_);
    if (copy != nullptr) {
        copy->SetNamespace(GetPluginNamespaceInternal());
    }
    return copy;
}

const char* RoPEPlugin::getPluginType() const noexcept { return kRoPEPluginName; }

const char* RoPEPlugin::getPluginVersion() const noexcept { return kRoPEPluginVersion; }

int32_t RoPEPlugin::getNbOutputs() const noexcept { return 2; }

int32_t RoPEPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
                                       const nvinfer1::DataType* inputTypes,
                                       int32_t nbInputs) const noexcept {
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs < 2 || nbInputs < 1) {
        return 1;
    }
    // 两个输出均为浮点，与 query 同类型；position_ids 只影响角度，不参与类型推导
    outputTypes[0] = inputTypes[0];
    outputTypes[1] = inputTypes[0];
    return 0;
}

int32_t RoPEPlugin::getOutputShapes(const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
                                    const nvinfer1::DimsExprs* shapeInputs,
                                    int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
                                    int32_t nbOutputs,
                                    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    (void)shapeInputs;
    (void)nbShapeInputs;
    (void)exprBuilder;
    if (inputs == nullptr || outputs == nullptr || nbInputs < 2 || nbOutputs < 2) {
        return 1;
    }
    // 逐维透传，动态 batch / seq_len 自动保持动态
    outputs[0] = inputs[0];
    outputs[1] = inputs[1];
    return 0;
}

bool RoPEPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut == nullptr || pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    const nvinfer1::PluginTensorDesc& desc = inOut[pos].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }

    // pos 2 是 position_ids，必须是 INT32，与浮点输入类型不同，不能参与"同类型"比对
    if (pos == 2) {
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

int32_t RoPEPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                    int32_t nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc* out,
                                    int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < 3) {
        return 1;
    }

    const nvinfer1::Dims& query_dims = in[0].desc.dims;
    const nvinfer1::Dims& key_dims = in[1].desc.dims;
    if (query_dims.nbDims != 4 || key_dims.nbDims != 4) {
        MINI_TRT_LOG_ERROR("RoPE: query/key must be 4-D [batch, heads, seq_len, head_size]");
        return 1;
    }

    // 形状是权威来源；属性仅在形状为动态轴时作为兜底
    const int32_t query_heads = query_dims.d[1];
    const int32_t kv_heads = key_dims.d[1];
    const int32_t head_size = query_dims.d[3];

    if (query_heads > 0) {
        num_heads_ = query_heads;
    }
    if (kv_heads > 0) {
        num_kv_heads_ = kv_heads;
    }
    if (head_size > 0) {
        head_size_ = head_size;
    }

    if (num_heads_ <= 0 || num_kv_heads_ <= 0 || head_size_ <= 0) {
        MINI_TRT_LOG_ERROR("RoPE: head configuration is unknown");
        return 1;
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        MINI_TRT_LOG_ERROR("RoPE: num_heads " << num_heads_ << " is not divisible by num_kv_heads "
                                              << num_kv_heads_);
        return 1;
    }

    // 默认 rotary_dim = head_size（Q3）；属性可配置为更小的偶数以支持部分旋转
    if (rotary_dim_ <= 0) {
        rotary_dim_ = head_size_;
    }
    if (rotary_dim_ > head_size_ || (rotary_dim_ % 2) != 0) {
        MINI_TRT_LOG_ERROR("RoPE: rotary_dim " << rotary_dim_ << " must be even and <= head_size "
                                               << head_size_);
        return 1;
    }
    return 0;
}

size_t RoPEPlugin::getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                    int32_t nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc* outputs,
                                    int32_t nbOutputs) const noexcept {
    (void)inputs;
    (void)nbInputs;
    (void)outputs;
    (void)nbOutputs;
    // 角度实时计算，无需预计算 sin/cos 表
    return 0;
}

int32_t RoPEPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                            const nvinfer1::PluginTensorDesc* outputDesc,
                            const void* const* inputs, void* const* outputs,
                            void* workspace, cudaStream_t stream) noexcept {
    (void)outputDesc;
    (void)workspace;
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr) {
        return 1;
    }

    const nvinfer1::Dims& query_dims = inputDesc[0].dims;
    const nvinfer1::Dims& key_dims = inputDesc[1].dims;
    if (query_dims.nbDims != 4 || key_dims.nbDims != 4) {
        return 1;
    }
    const nvinfer1::Dims& position_dims = inputDesc[2].dims;
    if (position_dims.nbDims != 2) {
        return 1;
    }

    RoPEKernelArgs args;
    args.query = inputs[0];
    args.key = inputs[1];
    args.position_ids = static_cast<const int32_t*>(inputs[2]);
    args.query_out = outputs[0];
    args.key_out = outputs[1];
    args.batch_size = query_dims.d[0];
    args.num_heads = query_dims.d[1];
    args.seq_len = query_dims.d[2];
    args.head_size = query_dims.d[3];
    args.num_kv_heads = key_dims.d[1];
    args.rotary_dim = rotary_dim_ > 0 ? rotary_dim_ : args.head_size;
    args.base = base_;
    args.is_half = (inputDesc[0].type == nvinfer1::DataType::kHALF);

    const cudaError_t err = LaunchRoPE(args, stream);
    if (err != cudaSuccess) {
        MINI_TRT_LOG_ERROR("RoPE enqueue failed: " << cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int32_t RoPEPlugin::onShapeChange(const nvinfer1::PluginTensorDesc* in, int32_t nbInputs,
                                  const nvinfer1::PluginTensorDesc* out,
                                  int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < 3) {
        return 1;
    }
    const nvinfer1::Dims& query_dims = in[0].dims;
    const nvinfer1::Dims& key_dims = in[1].dims;
    if (query_dims.nbDims != 4 || key_dims.nbDims != 4) {
        return 1;
    }
    // 关键：TensorRT 会在**反序列化出来的实例**上直接调用 onShapeChange，而该实例从未执行过
    // configurePlugin——head 配置是构建期从形状推导的、不作为序列化属性，因此此时成员仍是默认值 0。
    // 所以这里必须以运行期形状为准刷新缓存，而不是拿形状去校验成员；只在真正矛盾时失败。
    if (query_dims.d[1] > 0) {
        num_heads_ = query_dims.d[1];
    }
    if (key_dims.d[1] > 0) {
        num_kv_heads_ = key_dims.d[1];
    }
    if (query_dims.d[3] > 0) {
        head_size_ = query_dims.d[3];
    }
    if (query_dims.d[3] > 0 && key_dims.d[3] > 0 && query_dims.d[3] != key_dims.d[3]) {
        MINI_TRT_LOG_ERROR("RoPE: query/key head_size mismatch");
        return 1;
    }
    if (num_heads_ <= 0 || num_kv_heads_ <= 0 || head_size_ <= 0) {
        MINI_TRT_LOG_ERROR("RoPE: head configuration is unknown");
        return 1;
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        MINI_TRT_LOG_ERROR("RoPE: num_heads " << num_heads_
                                              << " is not divisible by num_kv_heads "
                                              << num_kv_heads_);
        return 1;
    }
    if (rotary_dim_ <= 0) {
        rotary_dim_ = head_size_;
    }
    if (rotary_dim_ > head_size_ || (rotary_dim_ % 2) != 0) {
        MINI_TRT_LOG_ERROR("RoPE: rotary_dim " << rotary_dim_
                                               << " must be even and <= head_size "
                                               << head_size_);
        return 1;
    }
    return 0;
}

nvinfer1::PluginFieldCollection const* RoPEPlugin::getFieldsToSerialize() noexcept {
    serialized_rotary_dim_ = rotary_dim_;
    serialized_base_ = base_;

    ResetFields();
    AddField("rotary_dim", &serialized_rotary_dim_,
             static_cast<int32_t>(nvinfer1::PluginFieldType::kINT32), 1);
    AddField("base", &serialized_base_,
             static_cast<int32_t>(nvinfer1::PluginFieldType::kFLOAT32), 1);
    return GetFields();
}

// =============================================================================
// RoPEPluginCreator
// =============================================================================

RoPEPluginCreator::RoPEPluginCreator() {
    fields_.push_back(nvinfer1::PluginField{"rotary_dim", nullptr,
                                            nvinfer1::PluginFieldType::kINT32, 1});
    fields_.push_back(nvinfer1::PluginField{"base", nullptr,
                                            nvinfer1::PluginFieldType::kFLOAT32, 1});
    field_collection_.nbFields = static_cast<int32_t>(fields_.size());
    field_collection_.fields = fields_.data();
}

const char* RoPEPluginCreator::getPluginName() const noexcept { return kRoPEPluginName; }

const char* RoPEPluginCreator::getPluginVersion() const noexcept {
    return kRoPEPluginVersion;
}

const char* RoPEPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::PluginFieldCollection const* RoPEPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}

nvinfer1::IPluginV3* RoPEPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    (void)name;
    (void)phase;
    auto* plugin = new (std::nothrow) RoPEPlugin(fc);
    if (plugin == nullptr) {
        return nullptr;
    }
    plugin->SetNamespace(namespace_.c_str());
    return plugin;
}

RoPEPluginCreator& GetRoPEPluginCreator() noexcept {
    static RoPEPluginCreator creator;
    return creator;
}

REGISTER_TENSORRT_PLUGIN(RoPEPluginCreator);

}  // namespace mini_trt_llm
