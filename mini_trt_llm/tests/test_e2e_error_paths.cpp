#include "e2e_fixture.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/imodel_builder.hpp"
#include "mini_trt_llm/core/model_config.hpp"
#include "test_gpu_guard.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <memory>
#include <string>

namespace mini_trt_llm {
namespace {

// 合法的 config 模板：model_type 由各用例替换，weight_map 指向 source.weight。
std::string ConfigFor(const std::string& model_type) {
    return R"({
        "model_type": ")" + model_type + R"(",
        "architecture": "decoder_only",
        "hyper_params": {"hidden_size": 8},
        "weight_map": {"norm_weight": "source.weight"}
    })";
}

bool EngineFileExists(const std::string& path) {
    return std::filesystem::exists(path);
}

// 只检查权重能否取到的最小 builder。
//
// 刻意不碰 network：这样"权重缺失"这类失败与 TensorRT 初始化是否成功解耦，
// 在无 GPU 环境下也能定位到真正的原因。
class WeightCheckingBuilder : public IModelBuilder {
 public:
    explicit WeightCheckingBuilder(std::string name) : name_(std::move(name)) {}

    std::string Name() const override { return name_; }

    bool Build(nvinfer1::INetworkDefinition*, const WeightLoader& weights,
               const ModelConfig&, const BuildOptions&) override {
        size_t bytes = 0;
        return weights.GetWeight("norm_weight", nvinfer1::DataType::kFLOAT, &bytes) !=
               nullptr;
    }

 private:
    std::string name_;
};

// 永远失败的最小 builder，用于验证"builder 失败必须让整个构建失败且不产出 engine"。
class FailingBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "e2e_failing"; }
    bool Build(nvinfer1::INetworkDefinition*, const WeightLoader&,
               const ModelConfig&, const BuildOptions&) override {
        return false;
    }
};

EngineBuilder::Config DefaultConfig() {
    EngineBuilder::Config config;
    // 错误路径不应依赖精度设置，用 FP32 避免 FP16 flag 带来的额外分支
    config.precision = Precision::FP32;
    return config;
}

}  // namespace

TEST(E2eErrorPathTest, MissingConfigFailsWithoutProducingEngine) {
    test_support::ModelDirectory dir = test_support::ModelDirectory::Create("err_config");
    ASSERT_TRUE(dir.valid());
    // 只写权重、不写 config.json
    ASSERT_TRUE(dir.WriteWeights({{"source.weight",
                                   test_support::TensorSpec{{8}, test_support::TensorSpec::Dtype::kF32,
                                                            std::vector<float>(8, 1.0f)}}}));

    Logger logger;
    EngineBuilder builder(logger, DefaultConfig());
    builder.RegisterModelBuilder("e2e_missing_config",
                                 std::make_shared<WeightCheckingBuilder>("e2e_missing_config"));

    const std::string engine_path = dir.EnginePath();
    EXPECT_FALSE(builder.BuildFromConfig(dir.path(), engine_path));
    // 半成品 engine 会让下游误判，必须显式断言不存在
    EXPECT_FALSE(EngineFileExists(engine_path));
}

TEST(E2eErrorPathTest, UnregisteredModelTypeFails) {
    test_support::ModelDirectory dir = test_support::ModelDirectory::Create("err_registry");
    ASSERT_TRUE(dir.valid());
    ASSERT_TRUE(dir.WriteConfig(ConfigFor("not_registered")));
    ASSERT_TRUE(dir.WriteWeights({{"source.weight",
                                   test_support::TensorSpec{{8}, test_support::TensorSpec::Dtype::kF32,
                                                            std::vector<float>(8, 1.0f)}}}));

    Logger logger;
    EngineBuilder builder(logger, DefaultConfig());
    // 注册一个别的名字，确保查找确实失败
    builder.RegisterModelBuilder("e2e_other",
                                 std::make_shared<WeightCheckingBuilder>("e2e_other"));

    const std::string engine_path = dir.EnginePath();
    EXPECT_FALSE(builder.BuildFromConfig(dir.path(), engine_path));
    EXPECT_FALSE(EngineFileExists(engine_path));
}

TEST(E2eErrorPathTest, MissingWeightsFileFails) {
    test_support::ModelDirectory dir = test_support::ModelDirectory::Create("err_weights");
    ASSERT_TRUE(dir.valid());
    ASSERT_TRUE(dir.WriteConfig(ConfigFor("e2e_missing_weights")));
    // 不写 model.safetensors

    Logger logger;
    EngineBuilder builder(logger, DefaultConfig());
    builder.RegisterModelBuilder(
        "e2e_missing_weights",
        std::make_shared<WeightCheckingBuilder>("e2e_missing_weights"));

    const std::string engine_path = dir.EnginePath();
    EXPECT_FALSE(builder.BuildFromConfig(dir.path(), engine_path));
    EXPECT_FALSE(EngineFileExists(engine_path));
}

TEST(E2eErrorPathTest, UnknownWeightMapKeyFails) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    test_support::ModelDirectory dir = test_support::ModelDirectory::Create("err_map_key");
    ASSERT_TRUE(dir.valid());
    ASSERT_TRUE(dir.WriteConfig(ConfigFor("e2e_bad_map")));
    // weight_map 指向 source.weight，但文件里只有别的 key
    ASSERT_TRUE(dir.WriteWeights({{"unrelated.weight",
                                   test_support::TensorSpec{{8}, test_support::TensorSpec::Dtype::kF32,
                                                            std::vector<float>(8, 1.0f)}}}));

    Logger logger;
    EngineBuilder builder(logger, DefaultConfig());
    builder.RegisterModelBuilder("e2e_bad_map",
                                 std::make_shared<WeightCheckingBuilder>("e2e_bad_map"));

    const std::string engine_path = dir.EnginePath();
    EXPECT_FALSE(builder.BuildFromConfig(dir.path(), engine_path));
    EXPECT_FALSE(EngineFileExists(engine_path));
}

TEST(E2eErrorPathTest, BuilderFailurePropagatesAndLeavesNoEngine) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // 对应"Plugin 配置非法"这类场景：builder 在构建期发现参数不合法并返回 false。
    test_support::ModelDirectory dir = test_support::ModelDirectory::Create("err_builder");
    ASSERT_TRUE(dir.valid());
    ASSERT_TRUE(dir.WriteConfig(ConfigFor("e2e_failing")));
    ASSERT_TRUE(dir.WriteWeights({{"source.weight",
                                   test_support::TensorSpec{{8}, test_support::TensorSpec::Dtype::kF32,
                                                            std::vector<float>(8, 1.0f)}}}));

    Logger logger;
    EngineBuilder builder(logger, DefaultConfig());
    builder.RegisterModelBuilder("e2e_failing", std::make_shared<FailingBuilder>());

    const std::string engine_path = dir.EnginePath();
    EXPECT_FALSE(builder.BuildFromConfig(dir.path(), engine_path));
    EXPECT_FALSE(EngineFileExists(engine_path));
}

}  // namespace mini_trt_llm
