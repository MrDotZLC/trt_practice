#include "cv_test_support.hpp"
#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/core/weight_loader.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/json.hpp"
#include "mini_trt_llm/utils/safetensors_loader.hpp"

#include <NvInfer.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::FindFile;

// 转换产物的**契约**：ResNet18 的解码器侧只有这 21 个带权重的模块。
// 这份清单由结构决定（4 个 stage × 2 个 block + 1 个 stem + 1 个 fc，downsample 只在
// stage 2/3/4 的第一个 block 上），所以它可以独立于转换脚本写出来——
// 两边都写，才能发现命名漂移（只信一边等于没有检查）。
std::vector<std::string> ExpectedModuleNames() {
    std::vector<std::string> names = {"conv1"};
    for (int stage = 1; stage <= 4; ++stage) {
        for (int block = 0; block < 2; ++block) {
            names.push_back("layer" + std::to_string(stage) + "." + std::to_string(block) +
                            ".conv1");
            names.push_back("layer" + std::to_string(stage) + "." + std::to_string(block) +
                            ".conv2");
        }
        if (stage >= 2) {
            names.push_back("layer" + std::to_string(stage) + ".0.downsample.0");
        }
    }
    names.push_back("fc");
    return names;
}

std::string FindModelDir() {
    // 注意返回的是**目录**（ModelConfig::Load 与 SafetensorsLoader 都按目录/文件组合找），
    // 不是 config.json 的路径——第一版返回了文件路径，Load 于是去找 `<...>/config.json/config.json`。
    const std::string config = FindFile({"models/resnet18/config.json",
                                         "../models/resnet18/config.json",
                                         "../../models/resnet18/config.json",
                                         "../../../models/resnet18/config.json"});
    if (config.empty()) {
        return {};
    }
    return std::filesystem::path(config).parent_path().string();
}

// 读 safetensors 的文件头（8 字节小端长度 + 那段 JSON）把键列出来。
//
// 为什么要自己读：`SafetensorsLoader::GetTensorNames()` 是**空实现**（底层库不暴露键遍历，
// 见 PROGRESS §5.2）。而"文件里的键"恰恰是反向覆盖检查要对比的一侧——缺了它，
// 只能单向验"weight_map 里的键都存在"，发现不了"文件里多出没人引用的张量"。
std::set<std::string> ReadSafetensorsKeys(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::set<std::string> keys;
    if (!in) {
        return keys;
    }
    uint64_t header_size = 0;
    in.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!in || header_size == 0 || header_size > (1u << 24)) {
        return keys;
    }
    std::string header(header_size, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_size));
    if (!in) {
        return keys;
    }
    try {
        const JsonValue json = JsonParser().Parse(header);
        if (!json.IsObject()) {
            return keys;
        }
        for (const auto& [key, _value] : json.AsObject()) {
            if (key != "__metadata__") {
                keys.insert(key);
            }
        }
    } catch (const std::exception&) {
        keys.clear();
    }
    return keys;
}

}  // namespace

// R0.2：转换产物的权重名集合与 config.json 的 weight_map **互相覆盖**。
//
// 三个集合必须相等：文件里的键、weight_map 的键、以及由 ResNet18 结构推出的期望名。
// 少一个 → 建模时会取不到权重；多一个 → 说明有张量没人引用（多半是命名漂移或漏转）。
// 全程 host，沙箱可跑。
TEST(ResNet18WeightContractTest, ConvertedArtifactCoversEveryWeight) {
    const std::string dir = FindModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/resnet18 不存在（先跑 tools/convert/onnx_to_mini_trt_llm.py）";
    }

    const ModelConfig config = ModelConfig::Load(dir);
    ASSERT_EQ(config.model_type, "resnet18");
    ASSERT_EQ(config.architecture, "cnn");

    // 结构期望：21 个模块 → weight + bias 共 42 个张量
    std::set<std::string> expected;
    for (const std::string& module : ExpectedModuleNames()) {
        expected.insert(module + ".weight");
        expected.insert(module + ".bias");
    }
    ASSERT_EQ(expected.size(), 42u) << "期望清单本身写错了";

    ASSERT_TRUE(config.weight_map.IsObject());
    std::set<std::string> mapped;
    for (const auto& [logical, source] : config.weight_map.AsObject()) {
        ASSERT_TRUE(source.IsString()) << logical;
        mapped.insert(logical);
        EXPECT_EQ(source.AsString(), logical)
            << "本项目约定 weight_map 是「逻辑名 → safetensors key」；identity 是当前产物形态";
    }

    const std::set<std::string> keys = ReadSafetensorsKeys(dir + "/model.safetensors");
    ASSERT_FALSE(keys.empty()) << "读不出 safetensors 的键（文件损坏或格式变了）";

    EXPECT_EQ(keys, expected) << "safetensors 的键与 ResNet18 的结构期望不一致";
    EXPECT_EQ(mapped, expected) << "weight_map 与结构期望不一致";

    // 经真实路径再验一遍：每个名字都要能通过 WeightLoader（走 weight_map + safetensors）取到
    WeightLoader weights;
    ASSERT_TRUE(weights.Load(dir));
    weights.SetWeightMap(config.weight_map);
    for (const std::string& name : expected) {
        EXPECT_TRUE(weights.HasWeight(name)) << name;
        size_t bytes = 0;
        const void* ptr = weights.GetWeight(name, nvinfer1::DataType::kFLOAT, &bytes);
        EXPECT_NE(ptr, nullptr) << name;
        EXPECT_GT(bytes, 0u) << name;
    }
}

// 形状自检：卷积 weight 必须 4 维且与自身 bias 的输出通道一致；fc 与 config 的类别数一致。
//
// 形状错位是最难从数值上反查的一类问题——它会让"引擎算错"看起来像"权重没学好"。
TEST(ResNet18WeightContractTest, ShapesAreSelfConsistent) {
    const std::string dir = FindModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/resnet18 不存在";
    }
    const ModelConfig config = ModelConfig::Load(dir);
    ASSERT_TRUE(config.hyper_params.Has("num_classes"));
    const int32_t num_classes = config.hyper_params["num_classes"].AsInt();

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(dir + "/model.safetensors"));

    int conv_count = 0;
    for (const std::string& module : ExpectedModuleNames()) {
        std::vector<size_t> weight_shape;
        std::vector<size_t> bias_shape;
        safetensors::dtype dtype = safetensors::kFLOAT32;
        ASSERT_TRUE(loader.GetTensorInfo(module + ".weight", &dtype, &weight_shape)) << module;
        ASSERT_TRUE(loader.GetTensorInfo(module + ".bias", &dtype, &bias_shape)) << module;

        if (module == "fc") {
            ASSERT_EQ(weight_shape.size(), 2u);
            EXPECT_EQ(weight_shape[0], static_cast<size_t>(num_classes));
            ASSERT_EQ(bias_shape.size(), 1u);
            EXPECT_EQ(bias_shape[0], static_cast<size_t>(num_classes));
            std::cout << "[ResNet18] fc.weight = [" << weight_shape[0] << ", "
                      << weight_shape[1] << "]\n";
            continue;
        }
        ++conv_count;
        ASSERT_EQ(weight_shape.size(), 4u) << module;
        ASSERT_EQ(bias_shape.size(), 1u) << module;
        EXPECT_EQ(bias_shape[0], weight_shape[0]) << module << " 的 bias 与输出通道不一致";
    }
    EXPECT_EQ(conv_count, 20) << "ResNet18 应有 20 个卷积层";
}

// `source` 元数据必须与产物自洽。
//
// `ModelConfig` 只解析四个结构性字段，`source` 得直接读 JSON。它值得一验：
// 这份元数据是"这批权重哪来的"的唯一记录（ONNX SHA256、是否折叠 BN、fc 的布局），
// 一份复制粘贴来的 source 块会把后来的人直接带进错误的假设里。
TEST(ResNet18WeightContractTest, SourceMetadataMatchesReality) {
    const std::string dir = FindModelDir();
    if (dir.empty()) {
        GTEST_SKIP() << "models/resnet18 不存在";
    }
    const JsonValue raw = LoadJson(dir + "/config.json");
    ASSERT_TRUE(raw.IsObject());
    ASSERT_TRUE(raw.Has("source")) << "缺少 source：无法回答'这批权重哪来的'";
    const JsonValue& source = raw["source"];

    EXPECT_EQ(source["opset"].AsInt(), 17);
    EXPECT_EQ(source["tensor_count"].AsInt(), 42);
    EXPECT_EQ(source["conv_count"].AsInt(), 20);
    // 这两条最容易被复制粘贴带错：它们描述的正是"消费方该怎么处理权重"
    EXPECT_EQ(source["batch_norm"].AsString(), "folded_into_conv");
    EXPECT_EQ(source["fc_weight_layout"].AsString(), "out_in_transB");

    // fc 的布局声明必须与文件里的实际形状对得上（[out, in]）
    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(dir + "/model.safetensors"));
    std::vector<size_t> fc_shape;
    safetensors::dtype dtype = safetensors::kFLOAT32;
    ASSERT_TRUE(loader.GetTensorInfo("fc.weight", &dtype, &fc_shape));
    ASSERT_EQ(fc_shape.size(), 2u);
    EXPECT_EQ(fc_shape[0], 1000u) << "out_in_transB 要求第 0 维是输出类别数";
}

}  // namespace mini_trt_llm
