#include "e2e_fixture.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/gpt2_model_builder.hpp"
#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include "test_gpu_guard.hpp"

#include <NvInfer.h>
#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

// 小配置：只要求结构同构，不追求规模。
constexpr int32_t kLayers = 2;
constexpr int32_t kHidden = 8;
constexpr int32_t kPositions = 16;
constexpr int32_t kVocab = 32;

std::string ConfigJson() {
    return R"({
        "model_type": "gpt2",
        "architecture": "decoder_only",
        "hyper_params": {
            "n_layer": 2, "n_head": 2, "n_embd": 8, "n_positions": 16,
            "vocab_size": 32, "layer_norm_epsilon": 1e-05,
            "tie_word_embeddings": true, "block_size": 4
        },
        "weight_map": {}
    })";
}

std::map<std::string, test_support::TensorSpec> Weights() {
    std::map<std::string, test_support::TensorSpec> tensors;
    const auto add = [&tensors](const std::string& name, std::vector<size_t> shape,
                                float value) {
        size_t count = 1;
        for (size_t dim : shape) {
            count *= dim;
        }
        tensors[name] = test_support::TensorSpec{
            std::move(shape), test_support::TensorSpec::Dtype::kF32,
            std::vector<float>(count, value)};
    };

    const size_t hidden = kHidden;
    add("wte.weight", {static_cast<size_t>(kVocab), hidden}, 0.02f);
    add("wpe.weight", {static_cast<size_t>(kPositions), hidden}, 0.01f);
    for (int32_t i = 0; i < kLayers; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        add(p + "ln_1.weight", {hidden}, 1.0f);
        add(p + "ln_1.bias", {hidden}, 0.0f);
        add(p + "attn.c_attn.weight", {hidden, 3 * hidden}, 0.02f);
        add(p + "attn.c_attn.bias", {3 * hidden}, 0.0f);
        add(p + "attn.c_proj.weight", {hidden, hidden}, 0.02f);
        add(p + "attn.c_proj.bias", {hidden}, 0.0f);
        add(p + "ln_2.weight", {hidden}, 1.0f);
        add(p + "ln_2.bias", {hidden}, 0.0f);
        add(p + "mlp.c_fc.weight", {hidden, 4 * hidden}, 0.02f);
        add(p + "mlp.c_fc.bias", {4 * hidden}, 0.0f);
        add(p + "mlp.c_proj.weight", {4 * hidden, hidden}, 0.02f);
        add(p + "mlp.c_proj.bias", {hidden}, 0.0f);
    }
    add("ln_f.weight", {hidden}, 1.0f);
    add("ln_f.bias", {hidden}, 0.0f);
    return tensors;
}

// 只建网络、不建引擎。
//
// **为什么这条边界值得单独测**：TensorRT 的建网 API 用法（矩阵乘的额外维匹配、
// shuffle 的零占位符、normalization 的 axesMask、softmax 的轴位图、dynamic slice
// 的 size 输入）错了，绝大多数情况下**只在真机上以构建失败或数值错误的形式暴露**。
//
// **环境语义（曾写错，勿再按错误版本改回）**：这里一度写着"建网络不需要 CUDA，
// 可以在沙箱 / CI 里跑"，但 `createInferBuilder` 实测需要 CUDA 初始化——无 GPU 的沙箱里它报
// `CUDA initialization failure with error: 35` 并返回 null。这条错误假设的代价很实在：
// 两条本该在提交时变红的输出数断言被静默 skip 掉，缺陷一路滑到真机（TROUBLESHOOTING #19）。
// 所以下面把两种失败**分开**：没有设备才算"环境不具备"，有设备却建不出 builder 是**真故障**。
class Gpt2NetworkBuildTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // 没有设备 → 跳过（并打印 cudaGetDeviceCount 的探测结果，见 test_gpu_guard.hpp）
        MINI_TRT_SKIP_IF_NO_CUDA();
        builder_.reset(nvinfer1::createInferBuilder(logger_));
        // 设备探测通过却建不出 builder：这是驱动/TensorRT 安装问题，不是"环境不具备"，
        // 必须判失败——否则它又会变成一个安静的 skip（#19 的同款滑梯）。
        ASSERT_NE(builder_, nullptr)
            << "CUDA 设备可用但 createInferBuilder 失败，请检查 TensorRT/CUDA 安装";
    }

    // 返回建好网络的 engine builder（调用方负责析构），失败返回 nullptr。
    nvinfer1::INetworkDefinition* BuildNetwork(BuildStage stage,
                                               const std::string& tag) {
        directory_ = test_support::ModelDirectory::Create(tag);
        if (!directory_.valid() || !directory_.WriteConfig(ConfigJson()) ||
            !directory_.WriteWeights(Weights())) {
            return nullptr;
        }

        if (!weights_.Load(directory_.path())) {
            return nullptr;
        }
        weights_.SetWeightMap(model_config_.weight_map);

        network_.reset(builder_->createNetworkV2(0U));
        if (network_ == nullptr) {
            return nullptr;
        }

        BuildOptions options;
        options.stage = stage;
        options.weight_dtype = nvinfer1::DataType::kFLOAT;
        if (!builder_impl_.Build(network_.get(), weights_, model_config_, options)) {
            return nullptr;
        }
        return network_.get();
    }

    Logger logger_;
    std::unique_ptr<nvinfer1::IBuilder> builder_;
    std::unique_ptr<nvinfer1::INetworkDefinition> network_;
    ModelConfig model_config_;
    WeightLoader weights_;
    GPT2ModelBuilder builder_impl_;
    test_support::ModelDirectory directory_;
};

// Prefill 网络：整段序列前向，并把每层的 K/V 暴露成输出。
TEST_F(Gpt2NetworkBuildTest, PrefillNetworkBuildsWithExpectedIo) {
    ModelConfig config;
    config.model_type = "gpt2";
    config.architecture = "decoder_only";
    config.hyper_params = JsonParser().Parse(R"({
        "n_layer": 2, "n_head": 2, "n_embd": 8, "n_positions": 16,
        "vocab_size": 32, "block_size": 4, "tie_word_embeddings": true
    })");
    model_config_ = config;

    nvinfer1::INetworkDefinition* network =
        BuildNetwork(BuildStage::kPrefill, "gpt2_net_prefill");
    ASSERT_NE(network, nullptr) << "GPT-2 prefill network construction failed";

    ASSERT_EQ(network->getNbInputs(), 2);
    EXPECT_STREQ(network->getInput(0)->getName(), "input_ids");
    EXPECT_STREQ(network->getInput(1)->getName(), "position_ids");

    // 2 层 × (k,v) + logits
    ASSERT_EQ(network->getNbOutputs(), 2 * kLayers + 1);
    bool has_logits = false;
    for (int32_t i = 0; i < network->getNbOutputs(); ++i) {
        has_logits = has_logits || std::string(network->getOutput(i)->getName()) == "logits";
    }
    EXPECT_TRUE(has_logits);

    // logits 的最后一维必须是 vocab；序列维保持动态（-1）。
    for (int32_t i = 0; i < network->getNbOutputs(); ++i) {
        nvinfer1::ITensor* output = network->getOutput(i);
        if (std::string(output->getName()) != "logits") {
            continue;
        }
        const nvinfer1::Dims dims = output->getDimensions();
        ASSERT_EQ(dims.nbDims, 3);
        EXPECT_EQ(dims.d[0], -1);
        EXPECT_EQ(dims.d[1], -1);
        EXPECT_EQ(dims.d[2], kVocab);
    }

    // 逐层触碰 getDimensions()：TensorRT 的 shape 推导错误是**延迟报告**的，
    // addShuffle / addSlice 等在调用时不会返回 nullptr。只有真的读一次维数，
    // 才会把 "reshape 到空张量" 这类问题变成可见的失败（nbDims == -1）。
    // 这条检查覆盖整张图，比逐个算子断言更省事。
    for (int32_t i = 0; i < network->getNbLayers(); ++i) {
        nvinfer1::ILayer* layer = network->getLayer(i);
        ASSERT_NE(layer, nullptr);
        for (int32_t out = 0; out < layer->getNbOutputs(); ++out) {
            const nvinfer1::Dims dims = layer->getOutput(out)->getDimensions();
            EXPECT_GT(dims.nbDims, 0)
                << "层 " << i << " (" << layer->getName() << ") 的输出 " << out
                << " 维数无效，说明建图阶段的 shape 推导出了问题";
        }
    }
}

// stage=kSingle 不导出 K/V（只做数值对拍时不需要这些输出）。
TEST_F(Gpt2NetworkBuildTest, SingleStageOmitsKvOutputs) {
    model_config_.model_type = "gpt2";
    model_config_.architecture = "decoder_only";
    model_config_.hyper_params = JsonParser().Parse(R"({
        "n_layer": 2, "n_head": 2, "n_embd": 8, "n_positions": 16,
        "vocab_size": 32, "block_size": 4, "tie_word_embeddings": true
    })");

    nvinfer1::INetworkDefinition* network =
        BuildNetwork(BuildStage::kSingle, "gpt2_net_single");
    ASSERT_NE(network, nullptr);
    EXPECT_EQ(network->getNbOutputs(), 1);
    EXPECT_STREQ(network->getOutput(0)->getName(), "logits");
}

// decode 网络：单 token 增量前向 + PagedAttention（含当前 token 的 K/V）。
//
// 这里把输入/输出契约钉死，因为它是 runner 侧绑定的依据：
//   block_tables 的宽度必须是 ceil(n_positions / block_size) = 4，
//   key/value cache 的形状必须是 [num_blocks, block_size, heads, head_size]。
// 这几处只要有一处与 PagedKVCache 的配置不一致，真机上就只会表现为"数值不对"。
TEST_F(Gpt2NetworkBuildTest, DecodeNetworkBuildsWithPagedAttentionInputs) {
    model_config_.model_type = "gpt2";
    model_config_.architecture = "decoder_only";
    model_config_.hyper_params = JsonParser().Parse(R"({
        "n_layer": 2, "n_head": 2, "n_embd": 8, "n_positions": 16,
        "vocab_size": 32, "block_size": 4, "tie_word_embeddings": true
    })");

    nvinfer1::INetworkDefinition* network =
        BuildNetwork(BuildStage::kDecode, "gpt2_net_decode");
    ASSERT_NE(network, nullptr) << "GPT-2 decode network construction failed";

    // 输入数 = 两个 token 级输入 + block_tables + context_lens + 每层一对 cache。
    // 每层必须各有一对：插件是单层注意力，共用一个 cache 张量会让所有层都读同一段。
    ASSERT_EQ(network->getNbInputs(), 4 + 2 * kLayers);
    EXPECT_STREQ(network->getInput(0)->getName(), "input_ids");
    EXPECT_STREQ(network->getInput(1)->getName(), "position_ids");
    EXPECT_STREQ(network->getInput(2)->getName(), "block_tables");
    EXPECT_STREQ(network->getInput(3)->getName(), "context_lens");
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const std::string suffix = std::to_string(layer);
        EXPECT_STREQ(network->getInput(4 + 2 * layer)->getName(),
                     ("key_cache_" + suffix).c_str());
        EXPECT_STREQ(network->getInput(5 + 2 * layer)->getName(),
                     ("value_cache_" + suffix).c_str());
    }

    // 单个 token：input_ids / position_ids 的序列维必须是静态 1
    for (const char* name : {"input_ids", "position_ids"}) {
        for (int32_t i = 0; i < network->getNbInputs(); ++i) {
            nvinfer1::ITensor* input = network->getInput(i);
            if (std::string(input->getName()) != name) {
                continue;
            }
            const nvinfer1::Dims dims = input->getDimensions();
            ASSERT_EQ(dims.nbDims, 2);
            EXPECT_EQ(dims.d[0], -1) << name;
            EXPECT_EQ(dims.d[1], 1) << name;
        }
    }

    // cache：n_positions=16 / block_size=4 → 4 个物理块
    const int32_t expected_blocks = kPositions / 4;
    for (int32_t i = 4; i < 4 + 2 * kLayers; ++i) {
        {
            nvinfer1::ITensor* input = network->getInput(i);
            const std::string name = input->getName();
            const nvinfer1::Dims dims = input->getDimensions();
            ASSERT_EQ(dims.nbDims, 4) << name;
            EXPECT_EQ(dims.d[0], expected_blocks) << name;
            EXPECT_EQ(dims.d[1], 4) << name;  // block_size
            EXPECT_EQ(dims.d[2], 2) << name;  // heads
            EXPECT_EQ(dims.d[3], 4) << name;  // head_size = n_embd / n_head
        }
    }

    // block table 宽度 = 每序列最多多少块 = ceil(16 / 4)
    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        nvinfer1::ITensor* input = network->getInput(i);
        if (std::string(input->getName()) != "block_tables") {
            continue;
        }
        const nvinfer1::Dims dims = input->getDimensions();
        ASSERT_EQ(dims.nbDims, 2);
        EXPECT_EQ(dims.d[0], -1);
        EXPECT_EQ(dims.d[1], expected_blocks);
    }

    // 输出：logits + 每层一对 K/V（供 runner 追加到 cache）
    ASSERT_EQ(network->getNbOutputs(), 2 * kLayers + 1);
    bool has_logits = false;
    for (int32_t i = 0; i < network->getNbOutputs(); ++i) {
        nvinfer1::ITensor* output = network->getOutput(i);
        const nvinfer1::Dims dims = output->getDimensions();
        if (std::string(output->getName()) == "logits") {
            has_logits = true;
            ASSERT_EQ(dims.nbDims, 3);
            EXPECT_EQ(dims.d[0], -1);
            EXPECT_EQ(dims.d[1], 1);  // decode 只有一步
            EXPECT_EQ(dims.d[2], kVocab);
        } else {
            // 每层的 K/V：[B, heads, 1, head_size]
            ASSERT_EQ(dims.nbDims, 4) << output->getName();
            EXPECT_EQ(dims.d[0], -1);
            EXPECT_EQ(dims.d[1], 2);
            EXPECT_EQ(dims.d[2], 1);
            EXPECT_EQ(dims.d[3], 4);
        }
    }
    EXPECT_TRUE(has_logits);

    // 同 prefill：逐层触碰维数，把延迟报告的 shape 推导错误变成可见失败
    for (int32_t i = 0; i < network->getNbLayers(); ++i) {
        nvinfer1::ILayer* layer = network->getLayer(i);
        ASSERT_NE(layer, nullptr);
        for (int32_t out = 0; out < layer->getNbOutputs(); ++out) {
            EXPECT_GT(layer->getOutput(out)->getDimensions().nbDims, 0)
                << "层 " << i << " (" << layer->getName() << ") 的输出 " << out
                << " 维数无效";
        }
    }
}

// 权重缺失必须在建网阶段就失败，而不是留到运行期。
TEST_F(Gpt2NetworkBuildTest, MissingWeightFailsTheBuild) {
    model_config_.model_type = "gpt2";
    model_config_.architecture = "decoder_only";
    model_config_.hyper_params = JsonParser().Parse(R"({
        "n_layer": 2, "n_head": 2, "n_embd": 8, "n_positions": 16,
        "vocab_size": 32, "block_size": 4, "tie_word_embeddings": true
    })");

    auto tensors = Weights();
    tensors.erase("h.1.mlp.c_proj.weight");
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_net_missing");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(tensors));

    WeightLoader weights;
    ASSERT_TRUE(weights.Load(directory.path()));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder_->createNetworkV2(0U));
    ASSERT_NE(network, nullptr);

    BuildOptions options;
    options.stage = BuildStage::kSingle;
    options.weight_dtype = nvinfer1::DataType::kFLOAT;
    EXPECT_FALSE(builder_impl_.Build(network.get(), weights, model_config_, options));
}

}  // namespace
}  // namespace mini_trt_llm
