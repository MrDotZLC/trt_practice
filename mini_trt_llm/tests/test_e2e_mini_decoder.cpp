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
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

// 链路规模：hidden = heads × head_size，保证 QKV 输出能被 reshape 成注意力布局
constexpr int32_t kHeads = 2;
constexpr int32_t kHeadSize = 8;
constexpr int32_t kHidden = kHeads * kHeadSize;  // 16
constexpr int32_t kQkvWidth = 2 * kHidden;       // 只投影 q、k（v 从缓存读，不参与计算）
constexpr int32_t kVocab = 16;
constexpr int32_t kBlockSize = 16;
constexpr int32_t kNumBlocks = 4;
constexpr int32_t kMaxBlocks = 2;
constexpr int32_t kContextLen = 20;  // 跨两个物理块
constexpr float kEps = 1e-6f;

// 全链路：
//   hidden ─► RMSNorm ─► MatMul(QKV) ─► Slice ─► Reshape ─► RoPE ─► PagedAttention
//          ─► Reshape ─► RMSNorm ─► MatMul(LM Head) ─► logits ─► Top-K Sampler
//
// 四个权重全部经 weight_map 从 safetensors 取用，并以 BF16 存储——既是真实使用路径，
// 也是 P1.5-0「多权重转换缓冲区互相覆盖」缺陷的触发条件。
class MiniDecoderBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "e2e_mini_decoder"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader& weights,
               const ModelConfig&, const BuildOptions&) override {
        using nvinfer1::DataType;
        using nvinfer1::Dims;

        nvinfer1::ITensor* hidden =
            network->addInput("hidden", DataType::kFLOAT, Dims{2, {1, kHidden}});
        nvinfer1::ITensor* position_ids =
            network->addInput("position_ids", DataType::kINT32, Dims{2, {1, 1}});
        nvinfer1::ITensor* key_cache = network->addInput(
            "key_cache", DataType::kFLOAT,
            Dims{4, {kNumBlocks, kBlockSize, kHeads, kHeadSize}});
        nvinfer1::ITensor* value_cache = network->addInput(
            "value_cache", DataType::kFLOAT,
            Dims{4, {kNumBlocks, kBlockSize, kHeads, kHeadSize}});
        nvinfer1::ITensor* block_tables = network->addInput(
            "block_tables", DataType::kINT32, Dims{2, {1, kMaxBlocks}});
        nvinfer1::ITensor* context_lens =
            network->addInput("context_lens", DataType::kINT32, Dims{1, {1}});
        if (hidden == nullptr || position_ids == nullptr || key_cache == nullptr ||
            value_cache == nullptr || block_tables == nullptr || context_lens == nullptr) {
            return false;
        }

        struct TensorSpec {
            const char* trt_name;
            const char* source_key;
            Dims dims;
            int32_t count;
        };
        const TensorSpec specs[] = {
            {"norm1_weight", "blk.norm1.weight", Dims{1, {kHidden}}, kHidden},
            {"qkv_weight", "blk.qkv.weight", Dims{2, {kHidden, kQkvWidth}},
             kHidden * kQkvWidth},
            {"norm2_weight", "blk.norm2.weight", Dims{1, {kHidden}}, kHidden},
            {"lm_head", "lm_head.weight", Dims{2, {kHidden, kVocab}}, kHidden * kVocab},
        };

        const void* weight_data[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int i = 0; i < 4; ++i) {
            size_t bytes = 0;
            weight_data[i] = weights.GetWeight(specs[i].trt_name, DataType::kFLOAT, &bytes);
            if (weight_data[i] == nullptr ||
                bytes != static_cast<size_t>(specs[i].count) * sizeof(float)) {
                return false;
            }
            // 指针必须互不别名：否则 addConstant 在构建期读到的是被覆盖后的数据
            for (int j = 0; j < i; ++j) {
                if (weight_data[i] == weight_data[j]) {
                    return false;
                }
            }
        }

        auto constant = [&](int index) -> nvinfer1::ITensor* {
            nvinfer1::IConstantLayer* layer = network->addConstant(
                specs[index].dims, nvinfer1::Weights{DataType::kFLOAT, weight_data[index],
                                                     specs[index].count});
            return layer == nullptr ? nullptr : layer->getOutput(0);
        };

        // 1. RMSNorm
        nvinfer1::ITensor* norm1_in[2] = {hidden, constant(0)};
        RmsNormPlugin norm1_plugin(kEps, kHidden);
        nvinfer1::IPluginV3Layer* norm1 =
            network->addPluginV3(norm1_in, 2, nullptr, 0, norm1_plugin);
        if (norm1 == nullptr) {
            return false;
        }

        // 2. QKV 投影
        nvinfer1::IMatrixMultiplyLayer* qkv = network->addMatrixMultiply(
            *norm1->getOutput(0), nvinfer1::MatrixOperation::kNONE, *constant(1),
            nvinfer1::MatrixOperation::kNONE);
        if (qkv == nullptr) {
            return false;
        }

        // 3. 切出 q / k（v 由 PagedAttention 从缓存读取）
        nvinfer1::ISliceLayer* q_slice = network->addSlice(
            *qkv->getOutput(0), nvinfer1::Dims{2, {0, 0}}, nvinfer1::Dims{2, {1, kHidden}},
            nvinfer1::Dims{2, {1, 1}});
        nvinfer1::ISliceLayer* k_slice = network->addSlice(
            *qkv->getOutput(0), nvinfer1::Dims{2, {0, kHidden}},
            nvinfer1::Dims{2, {1, kHidden}}, nvinfer1::Dims{2, {1, 1}});
        if (q_slice == nullptr || k_slice == nullptr) {
            return false;
        }

        // 4. reshape 成注意力布局 [batch, heads, 1, head_size]
        auto reshape = [&](nvinfer1::ITensor* tensor) -> nvinfer1::ITensor* {
            nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*tensor);
            if (shuffle == nullptr) {
                return nullptr;
            }
            shuffle->setReshapeDimensions(
                nvinfer1::Dims{4, {1, kHeads, 1, kHeadSize}});
            return shuffle->getOutput(0);
        };
        nvinfer1::ITensor* q_4d = reshape(q_slice->getOutput(0));
        nvinfer1::ITensor* k_4d = reshape(k_slice->getOutput(0));
        if (q_4d == nullptr || k_4d == nullptr) {
            return false;
        }

        // 5. RoPE（rotary_dim = head_size，全旋转）
        nvinfer1::ITensor* rope_in[3] = {q_4d, k_4d, position_ids};
        RoPEPlugin rope_plugin(kHeads, kHeads, kHeadSize, kHeadSize, 10000.0f);
        nvinfer1::IPluginV3Layer* rope =
            network->addPluginV3(rope_in, 3, nullptr, 0, rope_plugin);
        if (rope == nullptr) {
            return false;
        }
        rope->getOutput(1)->setName("k_rot");
        // k_rot 作为网络输出保留：否则这条支路会被 TRT 判为死代码消掉，
        // RoPE 的双输出形态就白测了
        network->markOutput(*rope->getOutput(1));

        // 6. 分页注意力（Decoding：query 序列长度为 1）
        nvinfer1::ITensor* attn_in[5] = {rope->getOutput(0), key_cache, value_cache,
                                         block_tables, context_lens};
        PagedAttentionPlugin attn_plugin(kHeads, kHeads, kHeadSize, kBlockSize, 0.0f);
        nvinfer1::IPluginV3Layer* attn =
            network->addPluginV3(attn_in, 5, nullptr, 0, attn_plugin);
        if (attn == nullptr) {
            return false;
        }

        // 7. 展平回 [1, hidden]
        nvinfer1::IShuffleLayer* flatten = network->addShuffle(*attn->getOutput(0));
        if (flatten == nullptr) {
            return false;
        }
        flatten->setReshapeDimensions(nvinfer1::Dims{2, {1, kHidden}});

        // 8. 第二处 RMSNorm
        nvinfer1::ITensor* norm2_in[2] = {flatten->getOutput(0), constant(2)};
        RmsNormPlugin norm2_plugin(kEps, kHidden);
        nvinfer1::IPluginV3Layer* norm2 =
            network->addPluginV3(norm2_in, 2, nullptr, 0, norm2_plugin);
        if (norm2 == nullptr) {
            return false;
        }

        // 9. LM Head
        nvinfer1::IMatrixMultiplyLayer* logits = network->addMatrixMultiply(
            *norm2->getOutput(0), nvinfer1::MatrixOperation::kNONE, *constant(3),
            nvinfer1::MatrixOperation::kNONE);
        if (logits == nullptr) {
            return false;
        }
        logits->getOutput(0)->setName("logits");
        network->markOutput(*logits->getOutput(0));
        return true;
    }
};

std::string ConfigJson() {
    return R"({
        "model_type": "e2e_mini_decoder",
        "architecture": "static_test",
        "hyper_params": {"hidden_size": 16},
        "weight_map": {
            "norm1_weight": "blk.norm1.weight",
            "qkv_weight": "blk.qkv.weight",
            "norm2_weight": "blk.norm2.weight",
            "lm_head": "lm_head.weight"
        }
    })";
}

float RoundToBf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    const uint16_t bf16 = static_cast<uint16_t>((bits + rounding_bias) >> 16);
    const uint32_t widened = static_cast<uint32_t>(bf16) << 16;
    std::memcpy(&bits, &widened, sizeof(float));
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

std::vector<float> MakeWeight(int32_t count, float phase) {
    std::vector<float> values(count);
    for (int32_t i = 0; i < count; ++i) {
        values[i] = std::sin(0.31f * static_cast<float>(i) + phase) * 0.7f;
    }
    return values;
}

std::vector<float> MakeCache(int32_t count, float phase) {
    std::vector<float> values(count);
    for (int32_t i = 0; i < count; ++i) {
        values[i] = std::cos(0.17f * static_cast<float>(i) + phase) * 0.5f;
    }
    return values;
}

struct Weights {
    std::vector<float> norm1;
    std::vector<float> qkv;
    std::vector<float> norm2;
    std::vector<float> lm_head;
};

std::map<std::string, test_support::TensorSpec> ToTensorSpecs(const Weights& weights) {
    std::map<std::string, test_support::TensorSpec> tensors;
    tensors["blk.norm1.weight"] = {{kHidden}, test_support::TensorSpec::Dtype::kBF16,
                                   weights.norm1};
    tensors["blk.qkv.weight"] = {{static_cast<size_t>(kHidden),
                                  static_cast<size_t>(kQkvWidth)},
                                 test_support::TensorSpec::Dtype::kBF16, weights.qkv};
    tensors["blk.norm2.weight"] = {{kHidden}, test_support::TensorSpec::Dtype::kBF16,
                                   weights.norm2};
    tensors["lm_head.weight"] = {{static_cast<size_t>(kHidden),
                                  static_cast<size_t>(kVocab)},
                                 test_support::TensorSpec::Dtype::kBF16, weights.lm_head};
    return tensors;
}

std::string BuildMiniDecoderEngine(const std::string& tag, const Weights& weights) {
    test_support::ModelDirectory directory = test_support::ModelDirectory::Create(tag);
    if (!directory.valid() || !directory.WriteConfig(ConfigJson()) ||
        !directory.WriteWeights(ToTensorSpecs(weights))) {
        return "";
    }

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    EngineBuilder builder(logger, config);
    builder.RegisterModelBuilder("e2e_mini_decoder", std::make_shared<MiniDecoderBuilder>());

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

// 逐算子串联的 CPU 参考，输入与网络完全一致。
std::vector<double> RunReference(const Weights& weights,
                                 const std::vector<double>& hidden,
                                 const std::vector<int32_t>& position_ids,
                                 const std::vector<double>& key_cache,
                                 const std::vector<double>& value_cache,
                                 const std::vector<int32_t>& block_table,
                                 int32_t context_len) {
    using namespace test_support;

    std::vector<double> norm1_weight(kHidden), qkv_weight(kHidden * kQkvWidth);
    std::vector<double> norm2_weight(kHidden), lm_weight(kHidden * kVocab);
    for (int32_t i = 0; i < kHidden; ++i) {
        norm1_weight[i] = weights.norm1[i];
        norm2_weight[i] = weights.norm2[i];
    }
    for (int32_t i = 0; i < kHidden * kQkvWidth; ++i) {
        qkv_weight[i] = weights.qkv[i];
    }
    for (int32_t i = 0; i < kHidden * kVocab; ++i) {
        lm_weight[i] = weights.lm_head[i];
    }

    std::vector<double> x1;
    ReferenceRmsNorm(hidden, norm1_weight, /*rows=*/1, kHidden, kEps, &x1);

    // QKV 投影：qkv = x1 @ W（[1,kHidden] × [kHidden,kQkvWidth]）
    std::vector<double> qkv(kQkvWidth, 0.0);
    for (int32_t c = 0; c < kQkvWidth; ++c) {
        double acc = 0.0;
        for (int32_t r = 0; r < kHidden; ++r) {
            acc += x1[r] * qkv_weight[r * kQkvWidth + c];
        }
        qkv[c] = acc;
    }

    // 切成 q / k 并 reshape 成 [heads, 1, head_size]
    const std::vector<int32_t> positions{position_ids[0]};
    std::vector<double> q3(static_cast<size_t>(kHeads) * kHeadSize);
    std::vector<double> k3(static_cast<size_t>(kHeads) * kHeadSize);
    for (int32_t i = 0; i < kHeads * kHeadSize; ++i) {
        q3[i] = qkv[i];
        k3[i] = qkv[kHidden + i];
    }
    std::vector<double> q_rot;
    std::vector<double> k_rot;
    ReferenceRoPE(q3, positions, /*batch=*/1, kHeads, 1, kHeadSize, kHeadSize, 10000.0,
                  &q_rot);
    ReferenceRoPE(k3, positions, /*batch=*/1, kHeads, 1, kHeadSize, kHeadSize, 10000.0,
                  &k_rot);

    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
    std::vector<double> attn;
    ReferencePagedAttentionDecode(q_rot, key_cache, value_cache, block_table, context_len,
                                  kHeads, kHeads, kHeadSize, kBlockSize, scale, &attn);

    std::vector<double> x2;
    ReferenceRmsNorm(attn, norm2_weight, /*rows=*/1, kHidden, kEps, &x2);

    std::vector<double> logits(kVocab, 0.0);
    for (int32_t v = 0; v < kVocab; ++v) {
        double acc = 0.0;
        for (int32_t r = 0; r < kHidden; ++r) {
            acc += x2[r] * lm_weight[r * kVocab + v];
        }
        logits[v] = acc;
    }
    return logits;
}

}  // namespace

TEST(E2eMiniDecoderTest, FullChainMatchesIndependentCpuReference) {
    MINI_TRT_SKIP_IF_NO_CUDA();

    Weights weights;
    weights.norm1 = MakeWeight(kHidden, 0.0f);
    weights.qkv = MakeWeight(kHidden * kQkvWidth, 1.0f);
    weights.norm2 = MakeWeight(kHidden, 2.0f);
    weights.lm_head = MakeWeight(kHidden * kVocab, 3.0f);

    const std::string engine_path = BuildMiniDecoderEngine("mini_decoder_full", weights);
    ASSERT_FALSE(engine_path.empty());

    // 输入取 BF16/FP16 下都能精确表示的值，避免输入端引入额外舍入
    std::vector<float> hidden(kHidden);
    for (int32_t i = 0; i < kHidden; ++i) {
        hidden[i] = 0.5f + 0.125f * static_cast<float>(i % 5);
    }
    std::vector<int32_t> position_ids{3};
    std::vector<int32_t> block_table{0, 1};

    const size_t cache_elements =
        static_cast<size_t>(kNumBlocks) * kBlockSize * kHeads * kHeadSize;
    std::vector<float> key_cache = MakeCache(static_cast<int32_t>(cache_elements), 0.0f);
    std::vector<float> value_cache = MakeCache(static_cast<int32_t>(cache_elements), 1.3f);

    // 权重以 BF16 存储，参考实现必须用同一批舍入后的数值
    Weights rounded;
    auto round_all = [](std::vector<float>* values) {
        for (float& v : *values) {
            v = RoundToBf16(v);
        }
    };
    rounded.norm1 = weights.norm1;
    rounded.qkv = weights.qkv;
    rounded.norm2 = weights.norm2;
    rounded.lm_head = weights.lm_head;
    round_all(&rounded.norm1);
    round_all(&rounded.qkv);
    round_all(&rounded.norm2);
    round_all(&rounded.lm_head);

    std::vector<double> hidden_d(hidden.begin(), hidden.end());
    std::vector<double> key_cache_d(key_cache.begin(), key_cache.end());
    std::vector<double> value_cache_d(value_cache.begin(), value_cache.end());
    const std::vector<double> expected =
        RunReference(rounded, hidden_d, position_ids, key_cache_d, value_cache_d, block_table,
                     kContextLen);

    Logger logger;
    Engine engine(engine_path, logger);

    DeviceBuffer d_hidden(kHidden * sizeof(float));
    DeviceBuffer d_pos(sizeof(int32_t));
    DeviceBuffer d_key(cache_elements * sizeof(float));
    DeviceBuffer d_value(cache_elements * sizeof(float));
    DeviceBuffer d_table(kMaxBlocks * sizeof(int32_t));
    DeviceBuffer d_context(sizeof(int32_t));
    DeviceBuffer d_logits(kVocab * sizeof(float));
    DeviceBuffer d_k_rot(kHidden * sizeof(float));
    DeviceBuffer d_tokens(sizeof(int32_t));
    ASSERT_TRUE(d_hidden.Allocate(kHidden * sizeof(float)));
    ASSERT_TRUE(d_pos.Allocate(sizeof(int32_t)));
    ASSERT_TRUE(d_key.Allocate(cache_elements * sizeof(float)));
    ASSERT_TRUE(d_value.Allocate(cache_elements * sizeof(float)));
    ASSERT_TRUE(d_table.Allocate(kMaxBlocks * sizeof(int32_t)));
    ASSERT_TRUE(d_context.Allocate(sizeof(int32_t)));
    ASSERT_TRUE(d_logits.Allocate(kVocab * sizeof(float)));
    ASSERT_TRUE(d_k_rot.Allocate(kHidden * sizeof(float)));
    ASSERT_TRUE(d_tokens.Allocate(sizeof(int32_t)));

    const int32_t context_len = kContextLen;
    CUDA_CHECK(cudaMemcpy(d_hidden.data(), hidden.data(), kHidden * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos.data(), position_ids.data(), sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key.data(), key_cache.data(), cache_elements * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value.data(), value_cache.data(), cache_elements * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table.data(), block_table.data(), kMaxBlocks * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_context.data(), &context_len, sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    ASSERT_TRUE(engine.SetTensorAddress("hidden", d_hidden.data()));
    ASSERT_TRUE(engine.SetTensorAddress("position_ids", d_pos.data()));
    ASSERT_TRUE(engine.SetTensorAddress("key_cache", d_key.data()));
    ASSERT_TRUE(engine.SetTensorAddress("value_cache", d_value.data()));
    ASSERT_TRUE(engine.SetTensorAddress("block_tables", d_table.data()));
    ASSERT_TRUE(engine.SetTensorAddress("context_lens", d_context.data()));
    ASSERT_TRUE(engine.SetTensorAddress("logits", d_logits.data()));
    ASSERT_TRUE(engine.SetTensorAddress("k_rot", d_k_rot.data()));
    ASSERT_TRUE(engine.Enqueue(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(kVocab);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_logits.data(), kVocab * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 链路比单算子长，误差按 1e-4 判定（单算子是 1e-5）
    for (int32_t v = 0; v < kVocab; ++v) {
        EXPECT_TRUE(test_support::WithinTolerance(static_cast<float>(expected[v]), actual[v],
                                                  1e-4f, 1e-4f))
            << "logits[" << v << "] expected=" << expected[v] << " actual=" << actual[v];
    }

    // logits 留在显存里直接交给采样器，不经过 host。
    // k = 1 时采样必然等于 argmax，因此可用参考 logits 的 argmax 做断言。
    int32_t expected_argmax = 0;
    for (int32_t v = 1; v < kVocab; ++v) {
        if (expected[v] > expected[expected_argmax]) {
            expected_argmax = v;
        }
    }
    const std::vector<int32_t> host_top_k{1};
    DeviceBuffer d_top_k(sizeof(int32_t));
    const size_t workspace_bytes = TopKSamplerWorkspaceBytes(1, kVocab);
    DeviceBuffer workspace(workspace_bytes);
    ASSERT_TRUE(d_top_k.Allocate(sizeof(int32_t)));
    ASSERT_TRUE(workspace.Allocate(workspace_bytes));
    CUDA_CHECK(cudaMemcpy(d_top_k.data(), host_top_k.data(), sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    TopKSamplerArgs sampler;
    sampler.logits = d_logits.data();
    sampler.token_ids = static_cast<int32_t*>(d_tokens.data());
    sampler.batch_size = 1;
    sampler.vocab_size = kVocab;
    sampler.is_half = false;
    sampler.seed = 42;
    sampler.top_k = static_cast<const int32_t*>(d_top_k.data());
    ASSERT_EQ(LaunchTopKSampler(sampler, nullptr, workspace.data(), workspace_bytes),
              cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    int32_t token = -1;
    CUDA_CHECK(cudaMemcpy(&token, d_tokens.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(token, expected_argmax);
}

TEST(E2eMiniDecoderTest, RejectsMissingWeightFromMap) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // weight_map 指向的 source key 不存在时，builder 必须失败而不是建出错误 engine
    Weights weights;
    weights.norm1 = MakeWeight(kHidden, 0.0f);
    weights.qkv = MakeWeight(kHidden * kQkvWidth, 1.0f);
    weights.norm2 = MakeWeight(kHidden, 2.0f);
    weights.lm_head = MakeWeight(kHidden * kVocab, 3.0f);

    std::map<std::string, test_support::TensorSpec> tensors = ToTensorSpecs(weights);
    tensors.erase("blk.qkv.weight");  // 故意删掉一个被 weight_map 引用的张量

    test_support::ModelDirectory directory = test_support::ModelDirectory::Create("md_missing");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(tensors));

    Logger logger;
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    EngineBuilder builder(logger, config);
    builder.RegisterModelBuilder("e2e_mini_decoder", std::make_shared<MiniDecoderBuilder>());

    const std::string engine_path = directory.EnginePath();
    EXPECT_FALSE(builder.BuildFromConfig(directory.path(), engine_path));
    EXPECT_FALSE(std::filesystem::exists(engine_path));
}

}  // namespace mini_trt_llm
