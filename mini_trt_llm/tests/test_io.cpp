#include "mini_trt_llm/utils/io.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <string>

namespace mini_trt_llm {

TEST(IoTest, ReadWriteFile) {
    const std::string path = "/tmp/test_mini_trt_io.bin";
    const std::string data = "hello mini_trt_llm";

    WriteFile(path, data.data(), data.size());
    auto buffer = ReadFile(path);
    EXPECT_EQ(buffer.size(), data.size());
    EXPECT_EQ(std::string(buffer.begin(), buffer.end()), data);

    std::remove(path.c_str());
}

TEST(IoTest, LoadJson) {
    const std::string path = "/tmp/test_mini_trt_config.json";
    const std::string json_text = R"({
        "model_type": "gpt2",
        "architecture": "decoder_only",
        "hyper_params": {"num_layers": 12}
    })";

    WriteFile(path, json_text.data(), json_text.size());
    auto json = LoadJson(path);
    EXPECT_EQ(json["model_type"].AsString(), "gpt2");
    EXPECT_EQ(json["architecture"].AsString(), "decoder_only");
    EXPECT_EQ(json["hyper_params"]["num_layers"].AsInt(), 12);

    std::remove(path.c_str());
}

}  // namespace mini_trt_llm
