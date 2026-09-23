#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>
#include <string>

namespace mini_trt_llm {

TEST(ModelConfigTest, LoadValidConfig) {
    const std::string dir = "/tmp/test_mini_trt_model";
    const std::string config_path = dir + "/config.json";
    const std::string json_text = R"({
        "model_type": "gpt2",
        "architecture": "decoder_only",
        "hyper_params": {
            "num_layers": 12,
            "hidden_size": 768
        },
        "weight_map": {
            "embedding": "transformer.wte.weight"
        }
    })";

    std::filesystem::remove_all(dir);
    WriteFile(config_path, json_text.data(), json_text.size());
    auto config = ModelConfig::Load(dir);
    EXPECT_EQ(config.model_type, "gpt2");
    EXPECT_EQ(config.architecture, "decoder_only");
    EXPECT_EQ(config.hyper_params["num_layers"].AsInt(), 12);
    EXPECT_EQ(config.hyper_params["hidden_size"].AsInt(), 768);
    EXPECT_EQ(config.weight_map["embedding"].AsString(),
              "transformer.wte.weight");

    std::filesystem::remove_all(dir);
}

}  // namespace mini_trt_llm
