#include "mini_trt_llm/plugins/plugin_registry.hpp"
#include "mini_trt_llm/plugins/rmsnorm_plugin.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include "test_reference.hpp"

#include "logger.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace mini_trt_llm {

// TensorRT 的对象统一用 delete 释放，用 unique_ptr 包住避免断言提前返回时泄漏。
namespace {

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const { delete ptr; }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

constexpr int32_t kBatch = 1;
constexpr int32_t kSeqLen = 4;
constexpr int32_t kHidden = 8;
constexpr float kEps = 1e-6f;

}  // namespace

// -----------------------------------------------------------------------------
// PluginRegistry 登记：host 侧即可验证
// -----------------------------------------------------------------------------

TEST(RmsNormPluginRegistryTest, RegisterAllPluginsRegistersRmsNormCreator) {
    PluginRegistry& registry = PluginRegistry::Instance();
    registry.RegisterAllPlugins();

    EXPECT_NE(registry.GetCreator(kRmsNormPluginName), nullptr);

    // 重复调用必须幂等：Registry 用 map 覆盖写，不应破坏已有登记
    registry.RegisterAllPlugins();
    EXPECT_NE(registry.GetCreator(kRmsNormPluginName), nullptr);
}

// -----------------------------------------------------------------------------
// 集成用例：真实 TRT network → engine → 推理，需要真机
// -----------------------------------------------------------------------------

TEST(RmsNormIntegrationTest, SerializedEngineInferenceMatchesCpuReference) {
    MINI_TRT_SKIP_IF_NO_CUDA();

    Logger logger;

    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    ASSERT_NE(builder, nullptr);

    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    ASSERT_NE(network, nullptr);

    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    ASSERT_NE(config, nullptr);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1u << 20);

    nvinfer1::ITensor* input =
        network->addInput("input", nvinfer1::DataType::kFLOAT,
                          nvinfer1::Dims{3, {kBatch, kSeqLen, kHidden}});
    ASSERT_NE(input, nullptr);

    std::vector<float> h_weight(kHidden);
    for (int32_t i = 0; i < kHidden; ++i) {
        // weight 偏离 1.0 才能暴露"忘记乘 weight"这类错误
        h_weight[i] = 0.5f + 0.01f * static_cast<float>(i % 17);
    }
    // nvinfer1::Weights 只持有裸指针，h_weight 必须活到 engine 构建完成
    nvinfer1::IConstantLayer* weight_layer = network->addConstant(
        nvinfer1::Dims{1, {kHidden}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, h_weight.data(), kHidden});
    ASSERT_NE(weight_layer, nullptr);

    // weight 作为 Plugin 第二输入，走常量层让 TRT 做常量折叠
    nvinfer1::ITensor* plugin_inputs[2] = {input, weight_layer->getOutput(0)};
    RmsNormPlugin plugin(kEps, kHidden);
    nvinfer1::IPluginV3Layer* plugin_layer =
        network->addPluginV3(plugin_inputs, 2, nullptr, 0, plugin);
    ASSERT_NE(plugin_layer, nullptr);

    nvinfer1::ITensor* output = plugin_layer->getOutput(0);
    ASSERT_NE(output, nullptr);
    output->setName("output");
    network->markOutput(*output);

    TrtPtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    ASSERT_NE(serialized, nullptr);

    TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);

    // 反序列化会回调 creator.createPlugin(fields)，因此这一步同时验证了
    // serialize/getFieldsToSerialize 与 REGISTER_TENSORRT_PLUGIN 的静态注册是通的
    TrtPtr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    ASSERT_NE(engine, nullptr);

    TrtPtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    ASSERT_NE(context, nullptr);

    const size_t element_count =
        static_cast<size_t>(kBatch) * static_cast<size_t>(kSeqLen) * kHidden;
    std::vector<float> h_input(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] = test_support::DeterministicValue(static_cast<int64_t>(i));
    }

    std::vector<float> reference;
    test_support::CpuRmsNorm(h_input, h_weight, kBatch * kSeqLen, kHidden, kEps, &reference);

    const size_t bytes = element_count * sizeof(float);
    DeviceBuffer d_input(bytes);
    DeviceBuffer d_output(bytes);
    ASSERT_TRUE(d_input.Allocate(bytes));
    ASSERT_TRUE(d_output.Allocate(bytes));
    CUDA_CHECK(cudaMemcpy(d_input.data(), h_input.data(), bytes, cudaMemcpyHostToDevice));

    ASSERT_TRUE(context->setTensorAddress("input", d_input.data()));
    ASSERT_TRUE(context->setTensorAddress("output", d_output.data()));
    ASSERT_TRUE(context->enqueueV3(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output(element_count);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output.data(), bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_TRUE(test_support::WithinTolerance(reference[i], h_output[i], 1e-5f, 1e-6f))
            << "index " << i << " reference=" << reference[i]
            << " actual=" << h_output[i];
    }
}

}  // namespace mini_trt_llm
