#include "e2e_fixture.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/imodel_builder.hpp"
#include "mini_trt_llm/plugins/paged_attention_plugin.hpp"
#include "mini_trt_llm/plugins/rmsnorm_plugin.hpp"
#include "mini_trt_llm/plugins/rope_plugin.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include "test_reference.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::TensorSpec;

constexpr int32_t kHidden = 8;

std::vector<float> MakeWeight(int32_t count, float offset) {
    std::vector<float> values(count);
    for (int32_t i = 0; i < count; ++i) {
        values[i] = 0.5f + 0.05f * static_cast<float>((i + static_cast<int32_t>(offset)) % 7);
    }
    return values;
}

std::string ConfigFor(const std::string& model_type, int32_t hidden) {
    return R"({
        "model_type": ")" + model_type + R"(",
        "architecture": "static_test",
        "hyper_params": {"hidden_size": )" + std::to_string(hidden) + R"(},
        "weight_map": {"norm_weight": "source.norm"}
    })";
}

// 从 WeightLoader 取权重并挂成常量层。
//
// 这条路径就是 Phase 2 的 GPT2ModelBuilder 会走的路：权重经 weight_map 从 safetensors
// 取出后直接交给 addConstant。nvinfer1::Weights 只存裸指针，指针必须在整个构建期间有效——
// 这正是 P1.5-0 要保证的性质。
nvinfer1::ITensor* AddWeightConstant(nvinfer1::INetworkDefinition* network,
                                     const WeightLoader& weights, const std::string& trt_name,
                                     int32_t count) {
    size_t bytes = 0;
    const void* data = weights.GetWeight(trt_name, nvinfer1::DataType::kFLOAT, &bytes);
    if (data == nullptr || bytes != static_cast<size_t>(count) * sizeof(float)) {
        return nullptr;
    }
    auto* layer = network->addConstant(
        nvinfer1::Dims{1, {count}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, data, count});
    return layer == nullptr ? nullptr : layer->getOutput(0);
}

// ---------------------------------------------------------------------------
// 每个算子的最小 builder
// ---------------------------------------------------------------------------

class RmsNormE2eBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "e2e_rmsnorm"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader& weights,
               const ModelConfig& config, const BuildOptions&) override {
        const int32_t hidden = config.hyper_params["hidden_size"].AsInt();
        nvinfer1::ITensor* input = network->addInput(
            "input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{2, {1, hidden}});
        if (input == nullptr) {
            return false;
        }
        nvinfer1::ITensor* weight = AddWeightConstant(network, weights, "norm_weight", hidden);
        if (weight == nullptr) {
            return false;
        }
        nvinfer1::ITensor* inputs[2] = {input, weight};
        RmsNormPlugin plugin(1e-6f, hidden);
        nvinfer1::IPluginV3Layer* layer =
            network->addPluginV3(inputs, 2, nullptr, 0, plugin);
        if (layer == nullptr) {
            return false;
        }
        layer->getOutput(0)->setName("output");
        network->markOutput(*layer->getOutput(0));
        return true;
    }
};

class RoPEE2eBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "e2e_rope"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader&,
               const ModelConfig&, const BuildOptions&) override {
        // RoPE 没有可学习权重，因此只验证"多输入多输出算子经真实 engine 调用"这一面。
        // rotary_dim = 4 < head_size = 8，覆盖部分旋转（TROUBLESHOOTING #4 的场景）。
        constexpr int32_t kHeads = 2;
        constexpr int32_t kSeq = 3;
        constexpr int32_t kHeadSize = 8;
        constexpr int32_t kRotaryDim = 4;

        nvinfer1::ITensor* query = network->addInput(
            "query", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims{4, {1, kHeads, kSeq, kHeadSize}});
        nvinfer1::ITensor* key = network->addInput(
            "key", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims{4, {1, kHeads, kSeq, kHeadSize}});
        nvinfer1::ITensor* position_ids = network->addInput(
            "position_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{2, {1, kSeq}});
        if (query == nullptr || key == nullptr || position_ids == nullptr) {
            return false;
        }

        nvinfer1::ITensor* inputs[3] = {query, key, position_ids};
        RoPEPlugin plugin(kHeads, kHeads, kHeadSize, kRotaryDim, 10000.0f);
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

class PagedAttentionE2eBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "e2e_paged_attn"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader&,
               const ModelConfig&, const BuildOptions&) override {
        constexpr int32_t kHeads = 2;
        constexpr int32_t kHeadSize = 8;
        constexpr int32_t kBlockSize = 16;
        constexpr int32_t kMaxBlocks = 2;

        nvinfer1::ITensor* query = network->addInput(
            "query", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims{4, {1, kHeads, 1, kHeadSize}});
        nvinfer1::ITensor* key_cache = network->addInput(
            "key_cache", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims{4, {4, kBlockSize, kHeads, kHeadSize}});
        nvinfer1::ITensor* value_cache = network->addInput(
            "value_cache", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims{4, {4, kBlockSize, kHeads, kHeadSize}});
        nvinfer1::ITensor* block_tables = network->addInput(
            "block_tables", nvinfer1::DataType::kINT32,
            nvinfer1::Dims{2, {1, kMaxBlocks}});
        nvinfer1::ITensor* context_lens = network->addInput(
            "context_lens", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
        if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
            block_tables == nullptr || context_lens == nullptr) {
            return false;
        }

        nvinfer1::ITensor* inputs[5] = {query, key_cache, value_cache, block_tables,
                                        context_lens};
        // block_size 必须显式配置（Q5）
        PagedAttentionPlugin plugin(kHeads, kHeads, kHeadSize, kBlockSize, 0.0f);
        nvinfer1::IPluginV3Layer* layer =
            network->addPluginV3(inputs, 5, nullptr, 0, plugin);
        if (layer == nullptr) {
            return false;
        }
        layer->getOutput(0)->setName("output");
        network->markOutput(*layer->getOutput(0));
        return true;
    }
};

class SamplerE2eBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "e2e_sampler"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader& weights,
               const ModelConfig& config, const BuildOptions&) override {
        const int32_t hidden = config.hyper_params["hidden_size"].AsInt();
        // 用 [hidden, hidden] 的方阵当 LM Head，输出 logits 维数等于 hidden，便于构造小样例
        nvinfer1::ITensor* input = network->addInput(
            "input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{2, {1, hidden}});
        if (input == nullptr) {
            return false;
        }
        size_t bytes = 0;
        const void* data =
            weights.GetWeight("norm_weight", nvinfer1::DataType::kFLOAT, &bytes);
        if (data == nullptr || bytes != static_cast<size_t>(hidden) * hidden * sizeof(float)) {
            return false;
        }
        auto* head = network->addConstant(
            nvinfer1::Dims{2, {hidden, hidden}},
            nvinfer1::Weights{nvinfer1::DataType::kFLOAT, data, hidden * hidden});
        if (head == nullptr) {
            return false;
        }
        auto* matmul = network->addMatrixMultiply(
            *input, nvinfer1::MatrixOperation::kNONE, *head->getOutput(0),
            nvinfer1::MatrixOperation::kNONE);
        if (matmul == nullptr) {
            return false;
        }
        matmul->getOutput(0)->setName("logits");
        network->markOutput(*matmul->getOutput(0));
        return true;
    }
};

// ---------------------------------------------------------------------------
// 通用执行辅助
// ---------------------------------------------------------------------------

struct BuiltEngine {
    test_support::ModelDirectory directory;
    std::string engine_path;
};

// 走完整的 BuildFromConfig 流程，返回 engine 文件路径。
// 失败时返回空路径，由调用方断言。
template <typename BuilderT>
std::string BuildEngine(const std::string& tag, const std::string& model_type,
                        const std::string& config_json,
                        const std::map<std::string, TensorSpec>& tensors) {
    test_support::ModelDirectory directory = test_support::ModelDirectory::Create(tag);
    if (!directory.valid() || !directory.WriteConfig(config_json) ||
        !directory.WriteWeights(tensors)) {
        return "";
    }

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    EngineBuilder builder(logger, config);
    builder.RegisterModelBuilder(model_type, std::make_shared<BuilderT>());

    const std::string engine_path = directory.EnginePath();
    if (!builder.BuildFromConfig(directory.path(), engine_path)) {
        return "";
    }
    // 目录对象析构会删掉文件，因此这里必须把 engine 拷出临时目录
    const std::string persistent = "/tmp/mini_trt_llm_e2e_" + tag + ".engine";
    std::error_code error;
    std::filesystem::copy_file(engine_path, persistent,
                               std::filesystem::copy_options::overwrite_existing, error);
    return error ? "" : persistent;
}

}  // namespace

TEST(E2eSingleOpTest, RmsNormThroughRealWeightLoader) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    const std::vector<float> weight = MakeWeight(kHidden, 0.0f);
    const std::string engine_path =
        BuildEngine<RmsNormE2eBuilder>("rmsnorm", "e2e_rmsnorm", ConfigFor("e2e_rmsnorm", kHidden),
                                       {{"source.norm",
                                         TensorSpec{{kHidden}, TensorSpec::Dtype::kF32, weight}}});
    ASSERT_FALSE(engine_path.empty());

    std::vector<float> input(kHidden);
    for (int32_t i = 0; i < kHidden; ++i) {
        input[i] = 0.25f * static_cast<float>(i + 1);
    }
    std::vector<float> expected;
    test_support::CpuRmsNorm(input, weight, 1, kHidden, 1e-6f, &expected);

    Logger logger;
    Engine engine(engine_path, logger);
    DeviceBuffer d_input(kHidden * sizeof(float));
    DeviceBuffer d_output(kHidden * sizeof(float));
    ASSERT_TRUE(d_input.Allocate(kHidden * sizeof(float)));
    ASSERT_TRUE(d_output.Allocate(kHidden * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input.data(), input.data(), kHidden * sizeof(float),
                          cudaMemcpyHostToDevice));
    ASSERT_TRUE(engine.SetTensorAddress("input", d_input.data()));
    ASSERT_TRUE(engine.SetTensorAddress("output", d_output.data()));
    ASSERT_TRUE(engine.Enqueue(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(kHidden);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output.data(), kHidden * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int32_t i = 0; i < kHidden; ++i) {
        EXPECT_TRUE(test_support::WithinTolerance(expected[i], actual[i], 1e-5f, 1e-5f))
            << "index " << i;
    }
}

TEST(E2eSingleOpTest, RoPEThroughRealBuilder) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    std::map<std::string, TensorSpec> tensors;
    tensors["source.norm"] = TensorSpec{{kHidden}, TensorSpec::Dtype::kF32,
                                        MakeWeight(kHidden, 0.0f)};
    const std::string engine_path = BuildEngine<RoPEE2eBuilder>(
        "rope", "e2e_rope", ConfigFor("e2e_rope", kHidden), tensors);
    ASSERT_FALSE(engine_path.empty());

    // 真实 engine 能建出来即说明三输入两输出、position_ids 为 INT32 的接线成立；
    // 数值正确性由 L1 用例覆盖，这里额外确认 position_ids 经网络输入传入后仍生效。
    std::vector<float> query(16, 0.5f);
    std::vector<float> key(16, 0.25f);
    std::vector<int32_t> positions{7, 9, 11};

    Logger logger;
    Engine engine(engine_path, logger);
    DeviceBuffer d_query(query.size() * sizeof(float));
    DeviceBuffer d_key(key.size() * sizeof(float));
    DeviceBuffer d_pos(positions.size() * sizeof(int32_t));
    DeviceBuffer d_q_out(query.size() * sizeof(float));
    DeviceBuffer d_k_out(key.size() * sizeof(float));
    ASSERT_TRUE(d_query.Allocate(query.size() * sizeof(float)));
    ASSERT_TRUE(d_key.Allocate(key.size() * sizeof(float)));
    ASSERT_TRUE(d_pos.Allocate(positions.size() * sizeof(int32_t)));
    ASSERT_TRUE(d_q_out.Allocate(query.size() * sizeof(float)));
    ASSERT_TRUE(d_k_out.Allocate(key.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_query.data(), query.data(), query.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key.data(), key.data(), key.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.data(), positions.data(), positions.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    ASSERT_TRUE(engine.SetTensorAddress("query", d_query.data()));
    ASSERT_TRUE(engine.SetTensorAddress("key", d_key.data()));
    ASSERT_TRUE(engine.SetTensorAddress("position_ids", d_pos.data()));
    ASSERT_TRUE(engine.SetTensorAddress("query_out", d_q_out.data()));
    ASSERT_TRUE(engine.SetTensorAddress("key_out", d_k_out.data()));
    ASSERT_TRUE(engine.Enqueue(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> q_out(query.size());
    CUDA_CHECK(cudaMemcpy(q_out.data(), d_q_out.data(), query.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // 输入非零且位置非零，旋转后必然与原值不同；若 kernel 未真正执行则会是原值
    bool changed = false;
    for (size_t i = 0; i < query.size(); ++i) {
        if (std::fabs(q_out[i] - query[i]) > 1e-4f) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

TEST(E2eSingleOpTest, PagedAttentionThroughRealBuilder) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    std::map<std::string, TensorSpec> tensors;
    tensors["source.norm"] = TensorSpec{{kHidden}, TensorSpec::Dtype::kF32,
                                        MakeWeight(kHidden, 0.0f)};
    const std::string engine_path = BuildEngine<PagedAttentionE2eBuilder>(
        "paged_attn", "e2e_paged_attn", ConfigFor("e2e_paged_attn", kHidden), tensors);
    // 五输入单输出、两个 INT32 索引输入能被 engine 接受，即说明接线与格式组合正确
    EXPECT_FALSE(engine_path.empty());
}

TEST(E2eSingleOpTest, LogitsFeedSamplerWithoutHostRoundTrip) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    const std::vector<float> head = MakeWeight(kHidden * kHidden, 1.0f);
    const std::string engine_path = BuildEngine<SamplerE2eBuilder>(
        "sampler", "e2e_sampler", ConfigFor("e2e_sampler", kHidden),
        {{"source.norm",
          TensorSpec{{static_cast<size_t>(kHidden), static_cast<size_t>(kHidden)},
                     TensorSpec::Dtype::kF32, head}}});
    ASSERT_FALSE(engine_path.empty());

    Logger logger;
    Engine engine(engine_path, logger);
    std::vector<float> input(kHidden, 1.0f);
    DeviceBuffer d_input(kHidden * sizeof(float));
    DeviceBuffer d_logits(kHidden * sizeof(float));
    DeviceBuffer d_tokens(sizeof(int32_t));
    ASSERT_TRUE(d_input.Allocate(kHidden * sizeof(float)));
    ASSERT_TRUE(d_logits.Allocate(kHidden * sizeof(float)));
    ASSERT_TRUE(d_tokens.Allocate(sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_input.data(), input.data(), kHidden * sizeof(float),
                          cudaMemcpyHostToDevice));
    ASSERT_TRUE(engine.SetTensorAddress("input", d_input.data()));
    ASSERT_TRUE(engine.SetTensorAddress("logits", d_logits.data()));
    ASSERT_TRUE(engine.Enqueue(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 关键点：logits 留在显存里直接交给采样器，不经过 host
    SamplerArgs args;
    args.logits = d_logits.data();
    args.token_ids = static_cast<int32_t*>(d_tokens.data());
    args.batch_size = 1;
    args.vocab_size = kHidden;
    args.is_half = false;
    args.seed = 42;
    ASSERT_EQ(LaunchGreedySampler(args, nullptr), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    int32_t token = -1;
    CUDA_CHECK(cudaMemcpy(&token, d_tokens.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_GE(token, 0);
    EXPECT_LT(token, kHidden);
}

}  // namespace mini_trt_llm
