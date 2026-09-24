#include "e2e_fixture.hpp"
#include "mini_trt_llm/core/gpt2_model_builder.hpp"
#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/json.hpp"
#include "mini_trt_llm/utils/safetensors_loader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

// 真实 GPT-2 的规模：weight_map 覆盖性用真材料验，数值用例另算。
constexpr int32_t kRealLayers = 12;
constexpr int32_t kRealHeads = 12;
constexpr int32_t kRealHidden = 768;
constexpr int32_t kRealPositions = 1024;
constexpr int32_t kRealVocab = 50257;

// 用例自造的小 GPT-2：结构同构，但每个维度都缩到能在测试里随便造。
// 保留 "3H / 4H / [in,out]" 这些形状关系才有意义。
GPT2Config SmallConfig() {
    GPT2Config config;
    config.n_layer = 2;
    config.n_head = 2;
    config.n_embd = 8;
    config.n_positions = 16;
    config.vocab_size = 32;
    config.block_size = 4;
    return config;
}

GPT2Config RealConfig() {
    GPT2Config config;
    config.n_layer = kRealLayers;
    config.n_head = kRealHeads;
    config.n_embd = kRealHidden;
    config.n_positions = kRealPositions;
    config.vocab_size = kRealVocab;
    config.block_size = 16;
    return config;
}

// 造出与转换工具产物同构的 config.json。
// with_weight_map = true 时补上"每个权重名指向自己"的映射，正是转换工具在
// "key 不带 transformer. 前缀"时的产物形态。
std::string ConfigJson(const GPT2Config& config, bool with_weight_map) {
    std::string weights = "{}";
    if (with_weight_map) {
        weights = "{";
        bool first = true;
        for (const std::string& name : GPT2WeightNames(config)) {
            if (!first) {
                weights += ",";
            }
            first = false;
            weights += "\"" + name + "\": \"" + name + "\"";
        }
        weights += "}";
    }

    return std::string(R"({
        "model_type": "gpt2",
        "architecture": "decoder_only",
        "hyper_params": {
            "n_layer": )") +
           std::to_string(config.n_layer) + R"(,
            "n_head": )" +
           std::to_string(config.n_head) + R"(,
            "n_embd": )" +
           std::to_string(config.n_embd) + R"(,
            "n_positions": )" +
           std::to_string(config.n_positions) + R"(,
            "vocab_size": )" +
           std::to_string(config.vocab_size) + R"(,
            "layer_norm_epsilon": 1e-05,
            "activation_function": "gelu_new",
            "tie_word_embeddings": )" +
           (config.tie_word_embeddings ? "true" : "false") + R"(,
            "block_size": )" +
           std::to_string(config.block_size) + R"(
        },
        "weight_map": )" +
           weights + R"(,
        "skipped_tensors": []
    })";
}

// 与 ConfigJson 的小配置配套的权重集合（含真实权重里存在但建模不用的 attn.bias）。
std::map<std::string, test_support::TensorSpec> SmallWeights(const GPT2Config& config) {
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

    const size_t hidden = static_cast<size_t>(config.n_embd);
    add("wte.weight", {static_cast<size_t>(config.vocab_size), hidden}, 0.02f);
    add("wpe.weight", {static_cast<size_t>(config.n_positions), hidden}, 0.01f);
    for (int32_t i = 0; i < config.n_layer; ++i) {
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
        add(p + "attn.bias",
            {1, 1, static_cast<size_t>(config.n_positions),
             static_cast<size_t>(config.n_positions)},
            0.0f);
    }
    add("ln_f.weight", {hidden}, 1.0f);
    add("ln_f.bias", {hidden}, 0.0f);
    return tensors;
}

// 真实转换产物的位置：优先取环境变量，其次从几个常见工作目录往上找。
std::string FindRealModelDir() {
    if (const char* env = std::getenv("MINI_TRT_LLM_GPT2_DIR")) {
        if (std::filesystem::exists(std::string(env) + "/config.json")) {
            return env;
        }
    }
    const char* candidates[] = {"models/gpt2", "../models/gpt2", "../../models/gpt2",
                                "../../../models/gpt2"};
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(std::string(candidate) + "/config.json")) {
            return candidate;
        }
    }
    return {};
}

}  // namespace

TEST(Gpt2ConfigTest, ParsesNativeConfig) {
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_cfg_ok");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(
        directory.WriteConfig(ConfigJson(RealConfig(), /*with_weight_map=*/true)));

    const ModelConfig config = ModelConfig::Load(directory.path());
    EXPECT_EQ(config.model_type, "gpt2");
    EXPECT_EQ(config.architecture, "decoder_only");

    GPT2Config gpt2;
    ASSERT_TRUE(GPT2Config::FromModelConfig(config, &gpt2));
    EXPECT_EQ(gpt2.n_layer, kRealLayers);
    EXPECT_EQ(gpt2.n_head, kRealHeads);
    EXPECT_EQ(gpt2.n_embd, kRealHidden);
    EXPECT_EQ(gpt2.n_positions, kRealPositions);
    EXPECT_EQ(gpt2.vocab_size, kRealVocab);
    EXPECT_EQ(gpt2.block_size, 16);
    EXPECT_EQ(gpt2.head_size(), 64);
    EXPECT_EQ(gpt2.num_blocks(), kRealPositions / 16);
    EXPECT_TRUE(gpt2.tie_word_embeddings);
}

// 每个字段的缺失/非法都必须失败：用默认值兜底会把"config 写错"变成
// "静默建出形状不对的网络"，而后者只有在真机跑数值时才会暴露。
TEST(Gpt2ConfigTest, RejectsMissingOrInconsistentHyperParams) {
    const auto parse = [](const std::string& json) {
        GPT2Config out;
        ModelConfig config;
        config.model_type = "gpt2";
        config.architecture = "decoder_only";
        config.hyper_params = JsonParser().Parse(json);
        return GPT2Config::FromModelConfig(config, &out);
    };

    EXPECT_TRUE(parse(R"({"n_layer":1,"n_head":2,"n_embd":8,"n_positions":16,)"
                      R"("vocab_size":32,"block_size":4})"));
    // block_size 缺失：强制显式配置，不给默认值
    EXPECT_FALSE(parse(R"({"n_layer":1,"n_head":2,"n_embd":8,"n_positions":16,)"
                       R"("vocab_size":32})"));
    // n_embd 不能被 n_head 整除
    EXPECT_FALSE(parse(R"({"n_layer":1,"n_head":3,"n_embd":8,"n_positions":16,)"
                       R"("vocab_size":32,"block_size":4})"));
    // 非正数
    EXPECT_FALSE(parse(R"({"n_layer":0,"n_head":2,"n_embd":8,"n_positions":16,)"
                       R"("vocab_size":32,"block_size":4})"));
    EXPECT_FALSE(parse(R"({"n_layer":1,"n_head":2,"n_embd":8,"n_positions":16,)"
                       R"("vocab_size":32,"block_size":0})"));
    // 类型不对
    EXPECT_FALSE(parse(R"({"n_layer":"1","n_head":2,"n_embd":8,"n_positions":16,)"
                       R"("vocab_size":32,"block_size":4})"));
}

// 权重名清单必须与真实 GPT-2 的权重集合一致：
// 12 层 × 12 张 + 4 张顶层 = 148，且不含 attn.bias（mask 缓冲，建模不用）。
TEST(Gpt2ConfigTest, WeightNamesMatchRealModelContract) {
    const std::vector<std::string> names = GPT2WeightNames(RealConfig());
    EXPECT_EQ(names.size(), 148u);

    const std::set<std::string> unique(names.begin(), names.end());
    EXPECT_EQ(unique.size(), names.size()) << "权重名不能重复";
    EXPECT_EQ(unique.count("wte.weight"), 1u);
    EXPECT_EQ(unique.count("wpe.weight"), 1u);
    EXPECT_EQ(unique.count("h.11.mlp.c_proj.weight"), 1u);

    for (const std::string& name : names) {
        EXPECT_EQ(name.find(".attn.bias"), std::string::npos)
            << "因果 mask 缓冲不应出现在建模所需的权重里: " << name;
        // 共享权重时不该去要 lm_head.weight：文件里根本没有这张表
        EXPECT_EQ(name.find("lm_head"), std::string::npos) << name;
    }

    GPT2Config untied = RealConfig();
    untied.tie_word_embeddings = false;
    const std::vector<std::string> with_head = GPT2WeightNames(untied);
    EXPECT_EQ(with_head.size(), 149u);
    EXPECT_EQ(std::count(with_head.begin(), with_head.end(), "lm_head.weight"), 1);
}

// 用真实加载路径（ModelConfig + WeightLoader + weight_map）验证"建模所需的每一个权重
// 都能被解析到"。这一步不需要 GPU：建引擎才需要，而"权重能不能取到"在这里就能判。
TEST(Gpt2WeightContractTest, SyntheticFixtureResolvesEveryWeight) {
    const GPT2Config config = SmallConfig();
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_weights_2l");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(ConfigJson(config, /*with_weight_map=*/true)));
    ASSERT_TRUE(directory.WriteWeights(SmallWeights(config)));

    const ModelConfig model_config = ModelConfig::Load(directory.path());
    GPT2Config parsed;
    ASSERT_TRUE(GPT2Config::FromModelConfig(model_config, &parsed));

    WeightLoader weights;
    ASSERT_TRUE(weights.Load(directory.path()));
    weights.SetWeightMap(model_config.weight_map);

    for (const std::string& name : GPT2WeightNames(parsed)) {
        size_t bytes = 0;
        EXPECT_NE(weights.GetWeight(name, nvinfer1::DataType::kFLOAT, &bytes), nullptr)
            << name;
        EXPECT_GT(bytes, 0u) << name;
    }

    // 映射确实被用上了：指向不存在的 key 时必须取不到（否则"漏映射"会被兜底掩盖）
    JsonValue broken_map = model_config.weight_map;
    broken_map["wte.weight"] = JsonValue(std::string("wte.weight.missing"));
    WeightLoader broken;
    ASSERT_TRUE(broken.Load(directory.path()));
    broken.SetWeightMap(broken_map);
    size_t bytes = 0;
    EXPECT_EQ(broken.GetWeight("wte.weight", nvinfer1::DataType::kFLOAT, &bytes),
              nullptr);
}

// 转换产物落在仓库里（models/gpt2）时做一次真材料校验；没转换过就跳过，
// 这样 CI 不依赖 548MB 的二进制夹具，而本地转换过之后这次校验会真的生效。
TEST(Gpt2WeightContractTest, RealConvertedArtifactIsComplete) {
    const std::string dir = FindRealModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/gpt2 不存在（先跑 hf_to_mini_trt_llm.py 转换）";
    }

    const ModelConfig model_config = ModelConfig::Load(dir);
    ASSERT_EQ(model_config.model_type, "gpt2");
    ASSERT_EQ(model_config.architecture, "decoder_only");

    GPT2Config gpt2;
    ASSERT_TRUE(GPT2Config::FromModelConfig(model_config, &gpt2));
    EXPECT_EQ(gpt2.n_layer, kRealLayers);

    // weight_map 的每个 value 都必须在 safetensors 里真实存在
    ASSERT_TRUE(model_config.weight_map.IsObject());
    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(dir + "/model.safetensors"));
    for (const auto& [trt_name, source_key] : model_config.weight_map.AsObject()) {
        ASSERT_TRUE(source_key.IsString()) << trt_name;
        EXPECT_TRUE(loader.HasTensor(source_key.AsString())) << trt_name;
    }

    // 建模所需的每一个权重都能经 weight_map 解析到（映射无漏项）
    WeightLoader weights;
    ASSERT_TRUE(weights.Load(dir));
    weights.SetWeightMap(model_config.weight_map);
    for (const std::string& name : GPT2WeightNames(gpt2)) {
        EXPECT_TRUE(weights.HasWeight(name)) << name;
    }

    // 关键形状。Conv1D 约定是 [in, out]，写反或转置错了这里会立刻失败，
    // 而不是等到真机上数值对不上才发现。
    const auto expect_shape = [&loader](const std::string& name,
                                        std::vector<size_t> expected) {
        safetensors::dtype dtype{};
        std::vector<size_t> shape;
        ASSERT_TRUE(loader.GetTensorInfo(name, &dtype, &shape)) << name;
        EXPECT_EQ(shape, expected) << name;
    };
    const size_t hidden = kRealHidden;
    expect_shape("wte.weight", {kRealVocab, hidden});
    expect_shape("wpe.weight", {kRealPositions, hidden});
    expect_shape("h.0.attn.c_attn.weight", {hidden, 3 * hidden});
    expect_shape("h.11.attn.c_proj.weight", {hidden, hidden});
    expect_shape("h.0.mlp.c_fc.weight", {hidden, 4 * hidden});
    expect_shape("h.11.mlp.c_proj.weight", {4 * hidden, hidden});
    expect_shape("ln_f.bias", {hidden});

    // 跳过的张量应当恰好是每层一张因果 mask（48MB 缓冲，不该进引擎）
    const JsonValue raw = LoadJson(dir + "/config.json");
    ASSERT_TRUE(raw.Has("skipped_tensors"));
    EXPECT_EQ(raw["skipped_tensors"].Size(), static_cast<size_t>(kRealLayers));
}

}  // namespace mini_trt_llm
