#include "mini_trt_llm/plugins/rmsnorm_kernel.hpp"
#include "mini_trt_llm/plugins/rmsnorm_plugin.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include "test_reference.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::HasCudaDevice;
using test_support::CpuRmsNorm;
using test_support::DeterministicValue;
using test_support::WithinTolerance;

// DeviceBuffer::Allocate 用返回值而非异常报错，而汇编这些 helper 的函数有返回值，
// 无法使用 ASSERT_* 宏，因此统一转成异常由 gtest 捕获并报告为用例失败。
void RequireAllocate(DeviceBuffer& buffer, size_t bytes) {
    if (!buffer.Allocate(bytes)) {
        throw std::runtime_error("Failed to allocate device buffer (" +
                                 std::to_string(bytes) + " bytes)");
    }
}

// 构造 FP32 输入并跑 kernel，返回设备端结果（供 FP32/FP16 两个用例复用）。
std::vector<float> RunKernelFloat(int64_t rows, int32_t hidden_size, float eps) {
    const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(hidden_size);
    const size_t bytes = element_count * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_weight(static_cast<size_t>(hidden_size));
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] = DeterministicValue(static_cast<int64_t>(i));
    }
    for (int32_t i = 0; i < hidden_size; ++i) {
        // weight 偏离 1.0 才能暴露"忘记乘 weight"这类错误
        h_weight[i] = 0.5f + 0.01f * static_cast<float>(i % 17);
    }

    DeviceBuffer d_input(bytes);
    DeviceBuffer d_weight(static_cast<size_t>(hidden_size) * sizeof(float));
    DeviceBuffer d_output(bytes);
    RequireAllocate(d_input, bytes);
    RequireAllocate(d_weight, static_cast<size_t>(hidden_size) * sizeof(float));
    RequireAllocate(d_output, bytes);

    CUDA_CHECK(cudaMemcpy(d_input.data(), h_input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight.data(), h_weight.data(),
                          static_cast<size_t>(hidden_size) * sizeof(float),
                          cudaMemcpyHostToDevice));

    RmsNormKernelArgs args;
    args.input = d_input.data();
    args.weight = d_weight.data();
    args.output = d_output.data();
    args.rows = rows;
    args.hidden_size = hidden_size;
    args.eps = eps;
    args.is_half = false;
    CUDA_CHECK(LaunchRmsNorm(args, /*stream=*/nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output(element_count);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output.data(), bytes, cudaMemcpyDeviceToHost));
    return h_output;
}

std::vector<float> RunKernelHalf(int64_t rows, int32_t hidden_size, float eps) {
    const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(hidden_size);
    const size_t half_bytes = element_count * sizeof(__half);

    std::vector<__half> h_input(element_count);
    std::vector<__half> h_weight(static_cast<size_t>(hidden_size));
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] = __float2half_rn(DeterministicValue(static_cast<int64_t>(i)));
    }
    for (int32_t i = 0; i < hidden_size; ++i) {
        h_weight[i] = __float2half_rn(0.5f + 0.01f * static_cast<float>(i % 17));
    }

    DeviceBuffer d_input(half_bytes);
    DeviceBuffer d_weight(static_cast<size_t>(hidden_size) * sizeof(__half));
    DeviceBuffer d_output(half_bytes);
    RequireAllocate(d_input, half_bytes);
    RequireAllocate(d_weight, static_cast<size_t>(hidden_size) * sizeof(__half));
    RequireAllocate(d_output, half_bytes);

    CUDA_CHECK(cudaMemcpy(d_input.data(), h_input.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight.data(), h_weight.data(),
                          static_cast<size_t>(hidden_size) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    RmsNormKernelArgs args;
    args.input = d_input.data();
    args.weight = d_weight.data();
    args.output = d_output.data();
    args.rows = rows;
    args.hidden_size = hidden_size;
    args.eps = eps;
    args.is_half = true;
    CUDA_CHECK(LaunchRmsNorm(args, /*stream=*/nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_output(element_count);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output.data(), half_bytes,
                          cudaMemcpyDeviceToHost));

    std::vector<float> result(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        result[i] = __half2float(h_output[i]);
    }
    return result;
}

}  // namespace

// -----------------------------------------------------------------------------
// Host 侧用例：不依赖 GPU，验证 Plugin 契约与序列化
// -----------------------------------------------------------------------------

TEST(RmsNormPluginTest, TypeAndVersionMatchCreatorContract) {
    RmsNormPlugin plugin(1e-5f, 64);
    EXPECT_STREQ(plugin.getPluginType(), kRmsNormPluginName);
    EXPECT_STREQ(plugin.getPluginVersion(), kRmsNormPluginVersion);
    EXPECT_EQ(plugin.getNbOutputs(), 1);
}

TEST(RmsNormPluginTest, DefaultEpsIsLlamaConvention) {
    RmsNormPlugin plugin;
    EXPECT_FLOAT_EQ(plugin.eps(), RmsNormPlugin::kDefaultEps);
    EXPECT_EQ(plugin.hidden_size(), 0);
}

TEST(RmsNormPluginTest, CloneCopiesAttributes) {
    RmsNormPlugin plugin(1e-5f, 128);
    std::unique_ptr<IPluginV3Base> copy(plugin.clone());
    ASSERT_NE(copy, nullptr);

    const auto* restored = static_cast<const RmsNormPlugin*>(copy.get());
    EXPECT_FLOAT_EQ(restored->eps(), 1e-5f);
    EXPECT_EQ(restored->hidden_size(), 128);
    // 克隆后类型必须仍然自洽，否则 TRT 反序列化会找不到对应 creator
    EXPECT_STREQ(restored->getPluginType(), kRmsNormPluginName);
}

TEST(RmsNormPluginTest, SerializedFieldsRoundTrip) {
    RmsNormPlugin plugin(1e-5f, 768);
    const nvinfer1::PluginFieldCollection* fields = plugin.getFieldsToSerialize();
    ASSERT_NE(fields, nullptr);
    ASSERT_EQ(fields->nbFields, 2);

    // 用序列化出的字段重新构造，模拟 TRT 反序列化路径
    RmsNormPlugin restored(fields);
    EXPECT_FLOAT_EQ(restored.eps(), 1e-5f);
    EXPECT_EQ(restored.hidden_size(), 768);
}

TEST(RmsNormPluginCreatorTest, FieldNamesDeclareTwoAttributes) {
    RmsNormPluginCreator creator;
    EXPECT_STREQ(creator.getPluginName(), kRmsNormPluginName);
    EXPECT_STREQ(creator.getPluginVersion(), kRmsNormPluginVersion);
    EXPECT_STREQ(creator.getPluginNamespace(), "");

    const nvinfer1::PluginFieldCollection* names = creator.getFieldNames();
    ASSERT_NE(names, nullptr);
    ASSERT_EQ(names->nbFields, 2);
    EXPECT_STREQ(names->fields[0].name, "eps");
    EXPECT_STREQ(names->fields[1].name, "hidden_size");
}

TEST(RmsNormPluginCreatorTest, CreatePluginFromFields) {
    float eps = 1e-5f;
    int32_t hidden_size = 128;
    nvinfer1::PluginField fields[2] = {
        nvinfer1::PluginField("eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1),
        nvinfer1::PluginField("hidden_size", &hidden_size,
                              nvinfer1::PluginFieldType::kINT32, 1)};
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = 2;
    fc.fields = fields;

    RmsNormPluginCreator creator;
    std::unique_ptr<nvinfer1::IPluginV3> plugin(
        creator.createPlugin(kRmsNormPluginName, &fc, nvinfer1::TensorRTPhase::kBUILD));
    ASSERT_NE(plugin, nullptr);

    const auto* rmsnorm = static_cast<const RmsNormPlugin*>(plugin.get());
    EXPECT_FLOAT_EQ(rmsnorm->eps(), eps);
    EXPECT_EQ(rmsnorm->hidden_size(), hidden_size);
}

TEST(RmsNormPluginTest, SupportsLinearFp32AndFp16Only) {
    RmsNormPlugin plugin(1e-5f, 8);

    nvinfer1::DynamicPluginTensorDesc in_out[3];
    for (int i = 0; i < 3; ++i) {
        in_out[i].desc.dims = nvinfer1::Dims{3, {1, 4, 8}};
        in_out[i].desc.type = nvinfer1::DataType::kFLOAT;
        in_out[i].desc.format = nvinfer1::TensorFormat::kLINEAR;
    }
    EXPECT_TRUE(plugin.supportsFormatCombination(0, in_out, 2, 1));
    EXPECT_TRUE(plugin.supportsFormatCombination(1, in_out, 2, 1));
    EXPECT_TRUE(plugin.supportsFormatCombination(2, in_out, 2, 1));

    // pos>0 时 dtype 必须与 inOut[0] 一致，混精度由 builder 插 Cast 负责
    in_out[1].desc.type = nvinfer1::DataType::kHALF;
    EXPECT_FALSE(plugin.supportsFormatCombination(1, in_out, 2, 1));

    in_out[1].desc.type = nvinfer1::DataType::kFLOAT;
    in_out[0].desc.format = nvinfer1::TensorFormat::kCHW32;
    EXPECT_FALSE(plugin.supportsFormatCombination(0, in_out, 2, 1));

    in_out[0].desc.format = nvinfer1::TensorFormat::kLINEAR;
    in_out[0].desc.type = nvinfer1::DataType::kINT32;
    EXPECT_FALSE(plugin.supportsFormatCombination(0, in_out, 2, 1));

    // 越界与空指针由基类契约要求安全返回 false，不能越界读
    EXPECT_FALSE(plugin.supportsFormatCombination(3, in_out, 2, 1));
    EXPECT_FALSE(plugin.supportsFormatCombination(0, nullptr, 2, 1));
}

// 回归用例：TensorRT 只保证 inOut[0..pos] 有效，pos 之后是未初始化内存。
// 早期实现把整个数组都拿去比对，导致所有组合都被判为不支持，engine 构建报
// "could not find any supported formats consistent with input/output data types"。
// 这里刻意在"不可读"的位置塞入与插件不兼容的值，确认它们被忽略。
TEST(RmsNormPluginTest, IgnoresInvalidDescriptorsAfterPos) {
    RmsNormPlugin plugin(1e-5f, 8);

    nvinfer1::DynamicPluginTensorDesc in_out[3];
    for (int i = 0; i < 3; ++i) {
        in_out[i].desc.dims = nvinfer1::Dims{3, {1, 4, 8}};
        in_out[i].desc.type = nvinfer1::DataType::kFLOAT;
        in_out[i].desc.format = nvinfer1::TensorFormat::kLINEAR;
    }
    in_out[1].desc.type = nvinfer1::DataType::kINT32;
    in_out[2].desc.format = nvinfer1::TensorFormat::kCHW32;

    EXPECT_TRUE(plugin.supportsFormatCombination(0, in_out, 2, 1));
}

TEST(RmsNormPluginTest, ConfigureRejectsWeightLengthMismatch) {
    RmsNormPlugin plugin(1e-5f, 8);

    nvinfer1::DynamicPluginTensorDesc inputs[2];
    inputs[0].desc.dims = nvinfer1::Dims{3, {1, 4, 8}};
    inputs[1].desc.dims = nvinfer1::Dims{1, {16}};  // 与 hidden=8 不符

    EXPECT_EQ(plugin.configurePlugin(inputs, 2, nullptr, 1), 1);

    inputs[1].desc.dims = nvinfer1::Dims{1, {8}};
    EXPECT_EQ(plugin.configurePlugin(inputs, 2, nullptr, 1), 0);
    EXPECT_EQ(plugin.hidden_size(), 8);
}

TEST(RmsNormPluginTest, WorkspaceIsZeroBecauseReductionUsesSharedMemory) {
    RmsNormPlugin plugin(1e-5f, 64);
    EXPECT_EQ(plugin.getWorkspaceSize(nullptr, 0, nullptr, 0), 0u);
}

// -----------------------------------------------------------------------------
// GPU 用例：需要真机，沙箱内自动跳过
// -----------------------------------------------------------------------------

TEST(RmsNormKernelTest, FloatMatchesCpuReference) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int64_t kRows = 8;
    constexpr int32_t kHidden = 64;
    constexpr float kEps = 1e-6f;

    const size_t element_count = static_cast<size_t>(kRows) * kHidden;
    std::vector<float> h_input(element_count);
    std::vector<float> h_weight(kHidden);
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] = DeterministicValue(static_cast<int64_t>(i));
    }
    for (int32_t i = 0; i < kHidden; ++i) {
        h_weight[i] = 0.5f + 0.01f * static_cast<float>(i % 17);
    }

    std::vector<float> reference;
    CpuRmsNorm(h_input, h_weight, kRows, kHidden, kEps, &reference);
    const std::vector<float> actual = RunKernelFloat(kRows, kHidden, kEps);

    ASSERT_EQ(reference.size(), actual.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_TRUE(WithinTolerance(reference[i], actual[i], 1e-5f, 1e-6f))
            << "index " << i << " reference=" << reference[i] << " actual=" << actual[i];
    }
}

TEST(RmsNormKernelTest, HalfMatchesCpuReferenceWithinFp16Tolerance) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int64_t kRows = 4;
    constexpr int32_t kHidden = 128;
    constexpr float kEps = 1e-6f;

    const size_t element_count = static_cast<size_t>(kRows) * kHidden;
    std::vector<float> h_input(element_count);
    std::vector<float> h_weight(kHidden);
    for (size_t i = 0; i < element_count; ++i) {
        // 量化到 FP16 后再算参考值，避免把 FP16 输入的舍入误差算成 kernel 误差
        h_input[i] = __half2float(__float2half_rn(DeterministicValue(static_cast<int64_t>(i))));
    }
    for (int32_t i = 0; i < kHidden; ++i) {
        h_weight[i] = __half2float(__float2half_rn(0.5f + 0.01f * static_cast<float>(i % 17)));
    }

    std::vector<float> reference;
    CpuRmsNorm(h_input, h_weight, kRows, kHidden, kEps, &reference);
    const std::vector<float> actual = RunKernelHalf(kRows, kHidden, kEps);

    ASSERT_EQ(reference.size(), actual.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        // 已确认标准：FP16 相对误差 < 1e-3，小值走绝对误差 Guardrail
        EXPECT_TRUE(WithinTolerance(reference[i], actual[i], 1e-3f, 1e-4f))
            << "index " << i << " reference=" << reference[i] << " actual=" << actual[i];
    }
}

TEST(RmsNormKernelTest, OddHiddenSizeFallsBackToScalarPath) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // 33 不能被 4 整除，强制走标量 kernel，覆盖向量化路径之外的边界
    constexpr int64_t kRows = 3;
    constexpr int32_t kHidden = 33;
    constexpr float kEps = 1e-6f;

    const size_t element_count = static_cast<size_t>(kRows) * kHidden;
    std::vector<float> h_input(element_count);
    std::vector<float> h_weight(kHidden);
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] = DeterministicValue(static_cast<int64_t>(i));
    }
    for (int32_t i = 0; i < kHidden; ++i) {
        h_weight[i] = 0.5f + 0.01f * static_cast<float>(i % 17);
    }

    std::vector<float> reference;
    CpuRmsNorm(h_input, h_weight, kRows, kHidden, kEps, &reference);
    const std::vector<float> actual = RunKernelFloat(kRows, kHidden, kEps);

    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_TRUE(WithinTolerance(reference[i], actual[i], 1e-5f, 1e-6f))
            << "index " << i << " reference=" << reference[i] << " actual=" << actual[i];
    }
}

TEST(RmsNormKernelTest, SingleRowDynamicDecodeShape) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Decode 阶段的典型形状：[batch=1, seq=1, hidden]，rows 退化为 1
    constexpr int64_t kRows = 1;
    constexpr int32_t kHidden = 768;

    const size_t element_count = static_cast<size_t>(kRows) * kHidden;
    std::vector<float> h_input(element_count);
    std::vector<float> h_weight(kHidden);
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] = DeterministicValue(static_cast<int64_t>(i));
    }
    for (int32_t i = 0; i < kHidden; ++i) {
        h_weight[i] = 0.5f + 0.01f * static_cast<float>(i % 17);
    }

    std::vector<float> reference;
    CpuRmsNorm(h_input, h_weight, kRows, kHidden, 1e-6f, &reference);
    const std::vector<float> actual = RunKernelFloat(kRows, kHidden, 1e-6f);

    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_TRUE(WithinTolerance(reference[i], actual[i], 1e-5f, 1e-6f))
            << "index " << i;
    }
}

TEST(RmsNormKernelTest, RejectsNullAndEmptyArguments) {
    RmsNormKernelArgs args;
    // 参数非法时返回错误码而不是抛异常，保证 enqueue 的 noexcept 契约成立
    EXPECT_NE(LaunchRmsNorm(args, nullptr), cudaSuccess);

    args.rows = 1;
    args.hidden_size = 8;
    EXPECT_NE(LaunchRmsNorm(args, nullptr), cudaSuccess);
}

}  // namespace mini_trt_llm
