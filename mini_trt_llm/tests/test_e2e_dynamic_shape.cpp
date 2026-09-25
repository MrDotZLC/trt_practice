#include "e2e_fixture.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/imodel_builder.hpp"
#include "mini_trt_llm/plugins/rope_plugin.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

constexpr int32_t kHeads = 2;
constexpr int32_t kHeadSize = 8;

// 动态 shape 网络：batch 与 seq_len 都声明为动态（-1），head 与 head_size 静态。
// 选 RoPE 是因为它天然支持动态 batch/seq，且已有数值参考实现可复用。
class DynamicRoPEE2eBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "e2e_dynamic_rope"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader&,
               const ModelConfig&, const BuildOptions&) override {
        nvinfer1::ITensor* query = network->addInput(
            "query", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims{4, {-1, kHeads, -1, kHeadSize}});
        nvinfer1::ITensor* key = network->addInput(
            "key", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims{4, {-1, kHeads, -1, kHeadSize}});
        nvinfer1::ITensor* position_ids = network->addInput(
            "position_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{2, {-1, -1}});
        if (query == nullptr || key == nullptr || position_ids == nullptr) {
            return false;
        }

        nvinfer1::ITensor* inputs[3] = {query, key, position_ids};
        RoPEPlugin plugin(kHeads, kHeads, kHeadSize, kHeadSize, 10000.0f);
        nvinfer1::IPluginV3Layer* layer =
            network->addPluginV3(inputs, 3, nullptr, 0, plugin);
        if (layer == nullptr) {
            return false;
        }
        layer->getOutput(0)->setName("query_out");
        layer->getOutput(1)->setName("key_out");
        network->markOutput(*layer->getOutput(0));
        network->markOutput(*layer->getOutput(1));
        return true;
    }
};

constexpr char kDynamicConfig[] = R"({
    "model_type": "e2e_dynamic_rope",
    "architecture": "decoder_only",
    "hyper_params": {"head_size": 8},
    "weight_map": {}
})";

// 构建带 Prefill/Decode 双 profile 的 engine，返回 engine 文件路径。
std::string BuildDynamicEngine(const std::string& tag) {
    test_support::ModelDirectory directory = test_support::ModelDirectory::Create(tag);
    // 权重内容与用例无关，这里只放一个占位张量让 WeightLoader::Load 能成功
    std::map<std::string, test_support::TensorSpec> tensors;
    tensors["placeholder"] = test_support::TensorSpec{
        {1}, test_support::TensorSpec::Dtype::kF32, std::vector<float>{1.0f}};

    if (!directory.valid() || !directory.WriteConfig(kDynamicConfig) ||
        !directory.WriteWeights(tensors)) {
        return "";
    }

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    // 收窄范围，让"超出范围"的用例有明确边界
    config.min_prefill_batch = 1;
    config.opt_prefill_batch = 1;
    config.max_prefill_batch = 2;
    config.min_prefill_seq_len = 1;
    config.opt_prefill_seq_len = 4;
    config.max_prefill_seq_len = 8;
    config.min_decode_batch = 1;
    config.opt_decode_batch = 1;
    config.max_decode_batch = 2;

    EngineBuilder builder(logger, config);
    builder.RegisterModelBuilder("e2e_dynamic_rope",
                                 std::make_shared<DynamicRoPEE2eBuilder>());

    const std::string engine_path = directory.EnginePath();
    if (!builder.BuildFromConfig(directory.path(), engine_path)) {
        return "";
    }
    const std::string persistent = "/tmp/mini_trt_llm_e2e_" + tag + ".engine";
    std::error_code error;
    std::filesystem::copy_file(engine_path, persistent,
                               std::filesystem::copy_options::overwrite_existing, error);
    return error ? "" : persistent;
}

// 用指定 profile 与 shape 跑一次推理的结果。
struct RunResult {
    bool shape_accepted = false;
    bool enqueue_ok = false;
};

RunResult RunWithShape(Engine* engine, int32_t profile_index, int32_t batch, int32_t seq_len) {
    RunResult result;
    const int64_t elements = static_cast<int64_t>(batch) * kHeads * seq_len * kHeadSize;
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    result.shape_accepted =
        engine->SetOptimizationProfile(profile_index, nullptr) &&
        engine->SetInputShape("query", nvinfer1::Dims{4, {batch, kHeads, seq_len, kHeadSize}}) &&
        engine->SetInputShape("key", nvinfer1::Dims{4, {batch, kHeads, seq_len, kHeadSize}}) &&
        engine->SetInputShape("position_ids", nvinfer1::Dims{2, {batch, seq_len}});
    if (!result.shape_accepted) {
        return result;
    }

    DeviceBuffer d_query(bytes);
    DeviceBuffer d_key(bytes);
    DeviceBuffer d_pos(static_cast<size_t>(batch) * seq_len * sizeof(int32_t));
    DeviceBuffer d_q_out(bytes);
    DeviceBuffer d_k_out(bytes);
    if (!d_query.Allocate(bytes) || !d_key.Allocate(bytes) ||
        !d_pos.Allocate(static_cast<size_t>(batch) * seq_len * sizeof(int32_t)) ||
        !d_q_out.Allocate(bytes) || !d_k_out.Allocate(bytes)) {
        return result;
    }

    std::vector<float> host_query(static_cast<size_t>(elements), 0.5f);
    std::vector<int32_t> host_pos(static_cast<size_t>(batch) * seq_len, 1);
    CUDA_CHECK(cudaMemcpy(d_query.data(), host_query.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key.data(), host_query.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.data(), host_pos.data(), host_pos.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    result.enqueue_ok =
        engine->SetTensorAddress("query", d_query.data()) &&
        engine->SetTensorAddress("key", d_key.data()) &&
        engine->SetTensorAddress("position_ids", d_pos.data()) &&
        engine->SetTensorAddress("query_out", d_q_out.data()) &&
        engine->SetTensorAddress("key_out", d_k_out.data()) &&
        engine->Enqueue(nullptr);
    if (result.enqueue_ok) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
}

}  // namespace

TEST(E2eDynamicShapeTest, PrefillAcceptsVaryingSeqLen) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string engine_path = BuildDynamicEngine("dyn_prefill");
    ASSERT_FALSE(engine_path.empty());

    Logger logger;
    Engine engine(engine_path, logger);
    // min / opt / max 三档都要覆盖：TRT 以 opt 为基准做 kernel 选择，只测 min/max 会漏掉主路径
    for (int32_t seq_len : {1, 4, 8}) {
        const RunResult result = RunWithShape(&engine, /*profile_index=*/0, /*batch=*/1, seq_len);
        EXPECT_TRUE(result.shape_accepted) << "seq_len=" << seq_len;
        EXPECT_TRUE(result.enqueue_ok) << "seq_len=" << seq_len;
    }
}

TEST(E2eDynamicShapeTest, DecodeAcceptsVaryingBatch) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string engine_path = BuildDynamicEngine("dyn_decode");
    ASSERT_FALSE(engine_path.empty());

    Logger logger;
    Engine engine(engine_path, logger);
    // Decode profile 把非 batch 动态维固定为 1
    for (int32_t batch : {1, 2}) {
        const RunResult result = RunWithShape(&engine, /*profile_index=*/1, batch, /*seq_len=*/1);
        EXPECT_TRUE(result.shape_accepted) << "batch=" << batch;
        EXPECT_TRUE(result.enqueue_ok) << "batch=" << batch;
    }
}

TEST(E2eDynamicShapeTest, OutOfRangeShapeIsRejected) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string engine_path = BuildDynamicEngine("dyn_out_of_range");
    ASSERT_FALSE(engine_path.empty());

    Logger logger;
    Engine engine(engine_path, logger);
    // max_prefill_seq_len = 8，超出后必须被拒绝而不是静默出错
    const RunResult result = RunWithShape(&engine, /*profile_index=*/0, /*batch=*/1, /*seq_len=*/16);
    EXPECT_FALSE(result.shape_accepted && result.enqueue_ok);
}

}  // namespace mini_trt_llm
