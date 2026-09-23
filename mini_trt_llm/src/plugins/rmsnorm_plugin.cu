#include "mini_trt_llm/plugins/rmsnorm_kernel.hpp"
#include "mini_trt_llm/plugins/rmsnorm_plugin.hpp"
#include "mini_trt_llm/utils/logger.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <new>
#include <string>

namespace mini_trt_llm {
namespace {

// 256 线程/行：足以在一个 block 内覆盖常见 hidden_size（768~8192），
// 且避开 512 线程带来的寄存器压力。
constexpr int kThreadsPerBlock = 256;

// 向量化宽度选择 16 字节：float4 与 8×half 都是 16B，是 sm_75 上单线程访存的最优粒度。
constexpr int kVecFloat = 4;
constexpr int kVecHalf = 8;

// 归约过程中第一级 warp 内归约用的掩码：blockDim 固定为 kThreadsPerBlock 的整数倍，
// 因此 block 内每个 warp 都是满编的。
constexpr unsigned int kFullWarpMask = 0xffffffffu;

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T FromFloat(float v);
template <>
__device__ __forceinline__ float FromFloat<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __half FromFloat<__half>(float v) {
    return __float2half_rn(v);
}

// kVec 个元素打包成一个 16B 单元，配合 alignas 让 reinterpret_cast 合法。
// 调用方负责保证 hidden_size 能被 kVec 整除，否则走标量 kernel。
template <typename T, int kVec>
struct alignas(16) VecPack {
    T v[kVec];
};

// Block 内求和归约，结果通过 shared memory 广播给所有线程。
//
// 用 __shfl_down_sync 而非共享内存树形归约：warp 内 shuffle 不需要同步且没有 bank 冲突。
__device__ __forceinline__ float BlockReduceSum(float val, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullWarpMask, val, offset);
    }
    if (lane == 0) {
        scratch[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float total = (lane < num_warps) ? scratch[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(kFullWarpMask, total, offset);
        }
        if (lane == 0) {
            scratch[0] = total;
        }
    }
    __syncthreads();
    return scratch[0];
}

// 向量化路径：一次读写 16B，只在设备端做两趟扫描（统计平方和 → 归一化写回）。
template <typename T, int kVec>
__global__ void RmsNormVectorKernel(const T* __restrict__ input,
                                    const T* __restrict__ weight,
                                    T* __restrict__ output, int hidden_size, float eps) {
    __shared__ float reduce_scratch[kThreadsPerBlock / 32];

    const size_t row = static_cast<size_t>(blockIdx.x);
    const T* row_in = input + row * hidden_size;
    T* row_out = output + row * hidden_size;

    using Pack = VecPack<T, kVec>;
    const int pack_count = hidden_size / kVec;
    const Pack* in_pack = reinterpret_cast<const Pack*>(row_in);
    const Pack* weight_pack = reinterpret_cast<const Pack*>(weight);

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < pack_count; i += blockDim.x) {
        const Pack p = in_pack[i];
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
            const float v = ToFloat(p.v[k]);
            sum_sq = fmaf(v, v, sum_sq);
        }
    }

    const float mean_sq =
        BlockReduceSum(sum_sq, reduce_scratch) / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_sq + eps);

    Pack* out_pack = reinterpret_cast<Pack*>(row_out);
    for (int i = threadIdx.x; i < pack_count; i += blockDim.x) {
        const Pack p = in_pack[i];
        const Pack w = weight_pack[i];
        Pack result;
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
            result.v[k] = FromFloat<T>(ToFloat(p.v[k]) * inv_rms * ToFloat(w.v[k]));
        }
        out_pack[i] = result;
    }
}

// 标量回退路径：hidden_size 不能被向量宽度整除时使用（如 768 之外的奇数值）。
template <typename T>
__global__ void RmsNormScalarKernel(const T* __restrict__ input,
                                    const T* __restrict__ weight,
                                    T* __restrict__ output, int hidden_size, float eps) {
    __shared__ float reduce_scratch[kThreadsPerBlock / 32];

    const size_t row = static_cast<size_t>(blockIdx.x);
    const T* row_in = input + row * hidden_size;
    T* row_out = output + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = ToFloat(row_in[i]);
        sum_sq = fmaf(v, v, sum_sq);
    }

    const float mean_sq =
        BlockReduceSum(sum_sq, reduce_scratch) / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_sq + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = FromFloat<T>(ToFloat(row_in[i]) * inv_rms * ToFloat(weight[i]));
    }
}

}  // namespace

cudaError_t LaunchRmsNorm(const RmsNormKernelArgs& args, cudaStream_t stream) {
    if (args.input == nullptr || args.weight == nullptr || args.output == nullptr ||
        args.rows <= 0 || args.hidden_size <= 0) {
        return cudaErrorInvalidValue;
    }

    // 一行一个 block：行之间完全独立，无需跨 block 同步，也省掉 workspace。
    const dim3 grid(static_cast<unsigned int>(args.rows));
    const dim3 block(kThreadsPerBlock);

    if (args.is_half) {
        const auto* in = static_cast<const __half*>(args.input);
        const auto* w = static_cast<const __half*>(args.weight);
        auto* out = static_cast<__half*>(args.output);
        if (args.hidden_size % kVecHalf == 0) {
            RmsNormVectorKernel<__half, kVecHalf><<<grid, block, 0, stream>>>(
                in, w, out, args.hidden_size, args.eps);
        } else {
            RmsNormScalarKernel<__half><<<grid, block, 0, stream>>>(in, w, out,
                                                                    args.hidden_size, args.eps);
        }
    } else {
        const auto* in = static_cast<const float*>(args.input);
        const auto* w = static_cast<const float*>(args.weight);
        auto* out = static_cast<float*>(args.output);
        if (args.hidden_size % kVecFloat == 0) {
            RmsNormVectorKernel<float, kVecFloat><<<grid, block, 0, stream>>>(
                in, w, out, args.hidden_size, args.eps);
        } else {
            RmsNormScalarKernel<float><<<grid, block, 0, stream>>>(in, w, out,
                                                                   args.hidden_size, args.eps);
        }
    }

    // 捕获非法 launch 配置（如 grid 超出上限），让调用方在 enqueue 中转换成错误码。
    return cudaGetLastError();
}

// =============================================================================
// RmsNormPlugin
// =============================================================================

RmsNormPlugin::RmsNormPlugin(float eps, int32_t hidden_size)
    : eps_(eps),
      hidden_size_(hidden_size),
      serialized_eps_(eps),
      serialized_hidden_size_(hidden_size) {}

RmsNormPlugin::RmsNormPlugin(const nvinfer1::PluginFieldCollection* fc) {
    if (fc != nullptr && fc->fields != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const nvinfer1::PluginField& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr) {
                continue;
            }
            const std::string name(field.name);
            if (name == "eps") {
                eps_ = *static_cast<const float*>(field.data);
            } else if (name == "hidden_size") {
                hidden_size_ = *static_cast<const int32_t*>(field.data);
            }
        }
    }
    serialized_eps_ = eps_;
    serialized_hidden_size_ = hidden_size_;
}

IPluginV3Base* RmsNormPlugin::clone() noexcept {
    auto* copy = new (std::nothrow) RmsNormPlugin(eps_, hidden_size_);
    if (copy != nullptr) {
        copy->SetNamespace(GetPluginNamespaceInternal());
    }
    return copy;
}

const char* RmsNormPlugin::getPluginType() const noexcept { return kRmsNormPluginName; }

const char* RmsNormPlugin::getPluginVersion() const noexcept {
    return kRmsNormPluginVersion;
}

int32_t RmsNormPlugin::getNbOutputs() const noexcept { return 1; }

int32_t RmsNormPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
                                          const nvinfer1::DataType* inputTypes,
                                          int32_t nbInputs) const noexcept {
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs < 1 || nbInputs < 1) {
        return 1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t RmsNormPlugin::getOutputShapes(const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
                                       const nvinfer1::DimsExprs* shapeInputs,
                                       int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
                                       int32_t nbOutputs,
                                       nvinfer1::IExprBuilder& exprBuilder) noexcept {
    (void)shapeInputs;
    (void)nbShapeInputs;
    (void)exprBuilder;
    if (inputs == nullptr || outputs == nullptr || nbInputs < 1 || nbOutputs < 1) {
        return 1;
    }
    // 逐维透传输入形状表达式，动态批量/序列维因此自动保持动态。
    outputs[0] = inputs[0];
    return 0;
}

bool RmsNormPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut == nullptr || pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    const nvinfer1::PluginTensorDesc& desc = inOut[pos].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    if (desc.type != nvinfer1::DataType::kFLOAT && desc.type != nvinfer1::DataType::kHALF) {
        return false;
    }
    if (pos == 0) {
        return true;
    }

    // 激活、weight、输出强制同 dtype/format：kernel 按单一类型实例化，混精度应由 builder 插 Cast。
    //
    // 只允许与 inOut[0] 比对。TensorRT 按 pos 递增调用，保证 inOut[0..pos] 有值，而
    // inOut[pos+1..] 是未初始化内存；早期实现扫描了整个数组，导致所有格式组合都被判为
    // 不支持，engine 构建直接报 "could not find any supported formats consistent with
    // input/output data types"（详见 docs/PROGRESS.md §5.9）。
    return inOut[pos].desc.format == inOut[0].desc.format &&
           inOut[pos].desc.type == inOut[0].desc.type;
}

int32_t RmsNormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                       int32_t nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc* out,
                                       int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < 2) {
        return 1;
    }

    const nvinfer1::Dims& activation_dims = in[0].desc.dims;
    const nvinfer1::Dims& weight_dims = in[1].desc.dims;
    if (activation_dims.nbDims < 1 || weight_dims.nbDims != 1) {
        return 1;
    }

    int32_t hidden = activation_dims.d[activation_dims.nbDims - 1];
    if (hidden <= 0) {
        // 最后一维被声明为动态轴时退回属性值，weight 的静态形状仍会参与校验。
        hidden = hidden_size_;
    }
    if (hidden <= 0) {
        MINI_TRT_LOG_ERROR("RmsNorm: hidden size is unknown (dynamic last dim and no attribute)");
        return 1;
    }
    if (weight_dims.d[0] > 0 && weight_dims.d[0] != hidden) {
        MINI_TRT_LOG_ERROR("RmsNorm: weight length "
                           << weight_dims.d[0] << " does not match hidden size " << hidden);
        return 1;
    }

    hidden_size_ = hidden;
    return 0;
}

size_t RmsNormPlugin::getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                       int32_t nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                                       int32_t nbOutputs) const noexcept {
    (void)inputs;
    (void)nbInputs;
    (void)outputs;
    (void)nbOutputs;
    // 归约只用 static shared memory，不需要 workspace；也避免 enqueue 内做任何分配。
    return 0;
}

int32_t RmsNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                               const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs,
                               void* workspace, cudaStream_t stream) noexcept {
    (void)outputDesc;
    (void)workspace;
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr) {
        return 1;
    }

    const nvinfer1::Dims& dims = inputDesc[0].dims;
    if (dims.nbDims < 1) {
        return 1;
    }
    const int32_t hidden_size = dims.d[dims.nbDims - 1];
    if (hidden_size <= 0) {
        return 1;
    }

    int64_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return 1;
        }
        volume *= dims.d[i];
    }

    RmsNormKernelArgs args;
    args.input = inputs[0];
    args.weight = inputs[1];
    args.output = outputs[0];
    args.rows = volume / hidden_size;
    args.hidden_size = hidden_size;
    args.eps = eps_;
    args.is_half = (inputDesc[0].type == nvinfer1::DataType::kHALF);

    const cudaError_t err = LaunchRmsNorm(args, stream);
    if (err != cudaSuccess) {
        MINI_TRT_LOG_ERROR("RmsNorm enqueue failed: " << cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int32_t RmsNormPlugin::onShapeChange(const nvinfer1::PluginTensorDesc* in, int32_t nbInputs,
                                     const nvinfer1::PluginTensorDesc* out,
                                     int32_t nbOutputs) noexcept {
    (void)out;
    (void)nbOutputs;
    if (in == nullptr || nbInputs < 1) {
        return 1;
    }
    const nvinfer1::Dims& dims = in[0].dims;
    if (dims.nbDims < 1) {
        return 1;
    }
    if (dims.d[dims.nbDims - 1] != hidden_size_) {
        MINI_TRT_LOG_ERROR("RmsNorm: shape change alters last dim from "
                           << hidden_size_ << " to " << dims.d[dims.nbDims - 1]);
        return 1;
    }
    return 0;
}

nvinfer1::PluginFieldCollection const* RmsNormPlugin::getFieldsToSerialize() noexcept {
    serialized_eps_ = eps_;
    serialized_hidden_size_ = hidden_size_;

    ResetFields();
    AddField("eps", &serialized_eps_,
             static_cast<int32_t>(nvinfer1::PluginFieldType::kFLOAT32), 1);
    AddField("hidden_size", &serialized_hidden_size_,
             static_cast<int32_t>(nvinfer1::PluginFieldType::kINT32), 1);
    return GetFields();
}

// =============================================================================
// RmsNormPluginCreator
// =============================================================================

RmsNormPluginCreator::RmsNormPluginCreator() {
    // getFieldNames() 只声明字段契约，data 在构建期由 TRT 填充，因此这里传 nullptr。
    fields_.push_back(nvinfer1::PluginField{"eps", nullptr,
                                            nvinfer1::PluginFieldType::kFLOAT32, 1});
    fields_.push_back(nvinfer1::PluginField{"hidden_size", nullptr,
                                            nvinfer1::PluginFieldType::kINT32, 1});
    field_collection_.nbFields = static_cast<int32_t>(fields_.size());
    field_collection_.fields = fields_.data();
}

const char* RmsNormPluginCreator::getPluginName() const noexcept {
    return kRmsNormPluginName;
}

const char* RmsNormPluginCreator::getPluginVersion() const noexcept {
    return kRmsNormPluginVersion;
}

const char* RmsNormPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::PluginFieldCollection const* RmsNormPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}

nvinfer1::IPluginV3* RmsNormPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    (void)name;
    (void)phase;
    auto* plugin = new (std::nothrow) RmsNormPlugin(fc);
    if (plugin == nullptr) {
        return nullptr;
    }
    plugin->SetNamespace(namespace_.c_str());
    return plugin;
}

RmsNormPluginCreator& GetRmsNormPluginCreator() noexcept {
    static RmsNormPluginCreator creator;
    return creator;
}

// 静态注册到 TensorRT 全局 registry，使 engine 反序列化时能按名找到 creator。
// 必须放在 namespace 内，否则宏里的非限定类型名无法解析到 mini_trt_llm::RmsNormPluginCreator。
REGISTER_TENSORRT_PLUGIN(RmsNormPluginCreator);

}  // namespace mini_trt_llm
