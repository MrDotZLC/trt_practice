// REQ-017（路线 C）的集成用例：int8 权重 + 清单 → 原生建图挂 DQ。
//
// 为什么集中在一个文件：它们共用同一套夹具（小模型 + 量化产物），也共享同一个前提——
// **清单里点名的东西必须真的被建进图**。全部需要 GPU / TensorRT，沙箱内显式跳过。

#include "gpt2_test_support.hpp"
#include "engine_layer_info_support.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "test_asset_guard.hpp"
#include "test_gpu_guard.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

// 量化对象按 `design.md` D6 的策略裁剪：只量化 2-D 的 Linear 权重与 `wte`，
// 排除 `wpe`（只被 gather 读一行）与全部 1-D（LayerNorm / bias）。
const std::vector<std::string>& QuantizedNames() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> result{"wte.weight"};
        for (int32_t layer = 0; layer < test_support::kLayers; ++layer) {
            const std::string prefix = "h." + std::to_string(layer) + ".";
            result.push_back(prefix + "attn.c_attn.weight");
            result.push_back(prefix + "attn.c_proj.weight");
            result.push_back(prefix + "mlp.c_fc.weight");
            result.push_back(prefix + "mlp.c_proj.weight");
        }
        return result;
    }();
    return names;
}

struct QuantFixture {
    test_support::ModelDirectory directory;
    std::string manifest_path;
    bool ok = false;
};

// 造"小模型 + int8 权重 + 清单"。公式与产出脚本**必须一致**：
// 对称量化 scale = max|x| / 127，码值 q = round(x / scale)（zero_point = 0）。
//
// `write_int8_file=false` 用来造"清单在、产物缺"；`extra_entry` 用来造"清单点名了模型里
// 没有的张量"。两者都必须让构建**响亮失败**，而不是静默退回 FP32。
QuantFixture MakeQuantFixture(const std::string& tag, bool write_int8_file = true,
                              const std::string& extra_entry = "") {
    QuantFixture fixture;
    fixture.directory = test_support::ModelDirectory::Create(tag);
    if (!fixture.directory.valid()) {
        return fixture;
    }
    const auto fp32 = test_support::SmallGpt2Weights();
    if (!fixture.directory.WriteConfig(test_support::SmallGpt2ConfigJson()) ||
        !fixture.directory.WriteWeights(fp32)) {
        return fixture;
    }

    std::map<std::string, test_support::TensorSpec> int8_weights;
    std::ostringstream entries;
    bool first = true;
    for (const std::string& name : QuantizedNames()) {
        const auto it = fp32.find(name);
        if (it == fp32.end()) {
            return fixture;  // 夹具与 QuantizedNames 不自洽 → ok 保持 false
        }
        const test_support::TensorSpec& spec = it->second;
        float amax = 0.0f;
        for (float value : spec.values) {
            amax = std::max(amax, std::fabs(value));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        std::vector<float> codes(spec.values.size());
        for (size_t i = 0; i < spec.values.size(); ++i) {
            codes[i] = std::round(spec.values[i] / scale);
        }
        int8_weights[name] = test_support::TensorSpec{
            spec.shape, test_support::TensorSpec::Dtype::kI8, std::move(codes)};

        if (!first) {
            entries << ",";
        }
        first = false;
        // scale 用 9 位有效数字写出：6 位会截断出与码值不一致的 scale（用例就会测错东西）。
        entries << "{\"tensor\": \"" << name << "\", \"source_key\": \"" << name
                << "\", \"granularity\": \"per_tensor\", \"axis\": null, \"scales\": ["
                << std::setprecision(9) << scale << "], \"saturate_ratio\": 0.0}";
    }
    if (!first && !extra_entry.empty()) {
        entries << ",";
    }
    entries << extra_entry;

    fixture.manifest_path = fixture.directory.path() + "/quant_int8.json";
    std::ofstream manifest(fixture.manifest_path, std::ios::binary);
    if (!manifest) {
        return fixture;
    }
    manifest << "{\"format_version\": 1,"
                "\"generator\": {\"tool\": \"tests\", \"command\": \"fixture\"},"
                "\"weights_source\": {\"path\": \"model.safetensors\", \"sha256\": \"x\"},"
                "\"int8_weights\": {\"path\": \"model_int8.safetensors\", \"sha256\": \"y\"},"
                "\"scheme\": \"symmetric_per_tensor\", \"zero_point\": 0, \"entries\": ["
             << entries.str() << "]}";
    manifest.close();

    if (write_int8_file &&
        !test_support::WriteSafetensorsFile(fixture.directory.path() + "/model_int8.safetensors",
                                            int8_weights)) {
        return fixture;
    }
    fixture.ok = true;
    return fixture;
}

// 逐层信息的读取已收到共享头 `engine_layer_info_support.hpp`（第三个使用方出现后按既有约定收拢）；
// 里面写明了"没有 `[I8]` 这种标签、要读 `Format/Datatype: Int8`"这条口径，以及为什么该合并。

// 真实 GPT-2 的目录（与既有真机用例同一套候选路径）。
std::string FindRealGpt2Dir() {
    const char* candidates[] = {"models/gpt2", "../models/gpt2", "../../models/gpt2",
                                "../../../models/gpt2"};
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(std::string(candidate) + "/config.json")) {
            return candidate;
        }
    }
    return {};
}

TEST(Gpt2Int8WeightsTest, BuildsBothStagesAndConsumesWholeManifest) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const QuantFixture fixture = MakeQuantFixture("gpt2_int8_build");
    ASSERT_TRUE(fixture.ok) << "夹具构造失败";

    Logger logger;
    EngineBuilder::Config config = test_support::SmallGpt2BuilderConfig();
    config.quant_manifest = fixture.manifest_path;
    EngineBuilder builder(logger, config);

    // prefill 与 decode 两张图都必须能用同一份清单建出来（design.md 的 Module Design）。
    EXPECT_TRUE(builder.BuildFromConfig(fixture.directory.path(),
                                       fixture.directory.EnginePath("int8_prefill.engine"),
                                       BuildStage::kPrefill));
    EXPECT_TRUE(builder.BuildFromConfig(fixture.directory.path(),
                                       fixture.directory.EnginePath("int8_decode.engine"),
                                       BuildStage::kDecode));
}

// **D7 的小模型信号**：若 TRT 在构建期把 DQ 常量折叠掉，int8 常量会变成 FP32 常量，
// 层信息里就看不到 Int8 —— 那意味着收益归零（详见 design.md 的 D7）。
// 体积判据要真实规模才有意义（小模型里权重占比太小），见下面那条 FullGpt2 用例。
TEST(Gpt2Int8WeightsTest, EngineLayerInfoShowsInt8) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const QuantFixture fixture = MakeQuantFixture("gpt2_int8_layers");
    ASSERT_TRUE(fixture.ok) << "夹具构造失败";

    Logger logger;
    EngineBuilder::Config config = test_support::SmallGpt2BuilderConfig();
    config.quant_manifest = fixture.manifest_path;
    config.detailed_profiling = true;  // 逐层精度只在 DETAILED 下写进引擎
    EngineBuilder builder(logger, config);
    const std::string engine_path = fixture.directory.EnginePath("int8_layers.engine");
    ASSERT_TRUE(builder.BuildFromConfig(fixture.directory.path(), engine_path,
                                       BuildStage::kPrefill));

    Engine engine(engine_path, logger);
    const int32_t int8_layers = test_support::InspectLayerInfo(&engine).int8_tensors;
    std::cout << "[REQ-017] int8 引擎的含 Int8 张量层数 = " << int8_layers << "\n";
    EXPECT_GE(int8_layers, 1)
        << "层信息里没有 Int8 张量：int8 常量 + DQ 很可能被构建期折叠成了 FP32（D7 的失败模式）";
}

TEST(Gpt2Int8WeightsTest, RejectsManifestEntryOutsideModel) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // 清单里多一条模型里没有的张量 → 建图收尾的"全消费"校验必须让构建失败。
    const QuantFixture fixture = MakeQuantFixture(
        "gpt2_int8_extra", /*write_int8_file=*/true,
        "{\"tensor\": \"h.9.attn.c_attn.weight\", \"source_key\": \"h.9.attn.c_attn.weight\","
        " \"granularity\": \"per_tensor\", \"axis\": null, \"scales\": [0.5]}");
    ASSERT_TRUE(fixture.ok) << "夹具构造失败";

    Logger logger;
    EngineBuilder::Config config = test_support::SmallGpt2BuilderConfig();
    config.quant_manifest = fixture.manifest_path;
    EngineBuilder builder(logger, config);
    EXPECT_FALSE(builder.BuildFromConfig(fixture.directory.path(),
                                        fixture.directory.EnginePath("int8_extra.engine"),
                                        BuildStage::kPrefill));
}

TEST(Gpt2Int8WeightsTest, RejectsMissingInt8WeightsFile) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const QuantFixture fixture = MakeQuantFixture("gpt2_int8_nofile", /*write_int8_file=*/false);
    ASSERT_TRUE(fixture.ok) << "夹具构造失败";

    Logger logger;
    EngineBuilder::Config config = test_support::SmallGpt2BuilderConfig();
    config.quant_manifest = fixture.manifest_path;
    EngineBuilder builder(logger, config);
    // 清单在、产物缺 → 必须响亮失败，**不得**静默退回纯 FP32 引擎。
    EXPECT_FALSE(builder.BuildFromConfig(fixture.directory.path(),
                                        fixture.directory.EnginePath("int8_nofile.engine"),
                                        BuildStage::kPrefill));
}

// **D7 的正式判据**（design.md）：引擎体积必须相对同配置的 FP32 引擎下降。
//
// 为什么必须用真实规模：小模型的权重只占引擎文件的极小部分，体积差会被元数据噪声淹没；
// 而 weight-only INT8 的收益全部来自"权重按 1 字节读"，只有真实 124M 参数才能体现。
// 引擎路径固定在 /tmp 下**不随临时目录销毁**，这样第二次运行起能命中引擎缓存（首次是分钟级）。
TEST(Gpt2Int8WeightsTest, FullGpt2Int8EngineIsSmallerThanFp32) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::string model_dir = FindRealGpt2Dir();
    if (model_dir.empty()) {
        MINI_TRT_SKIP_IF_MISSING_ASSET("models/gpt2 不存在（先跑 hf_to_mini_trt_llm.py 转换）");
    }
    const std::string manifest = model_dir + "/quant_int8.json";
    if (!std::filesystem::exists(manifest)) {
        MINI_TRT_SKIP_IF_MISSING_ASSET(
            "缺 " + manifest + "（先跑 tools/convert/quantize_gpt2.py --model-dir " + model_dir + "）");
    }

    Logger logger;
    const std::string fp32_engine = "/tmp/mini_trt_llm_gpt2_d7_fp32.engine";
    const std::string int8_engine = "/tmp/mini_trt_llm_gpt2_d7_int8.engine";

    // 真实模型用**默认 profile**（与真机既有用例同口径），只把精度钉成 FP32。
    EngineBuilder::Config fp32_config;
    fp32_config.precision = Precision::FP32;
    EngineBuilder fp32_builder(logger, fp32_config);
    ASSERT_TRUE(fp32_builder.BuildFromConfig(model_dir, fp32_engine, BuildStage::kPrefill));

    EngineBuilder::Config int8_config = fp32_config;
    int8_config.quant_manifest = "quant_int8.json";  // 相对路径按模型目录解释
    EngineBuilder int8_builder(logger, int8_config);
    ASSERT_TRUE(int8_builder.BuildFromConfig(model_dir, int8_engine, BuildStage::kPrefill));

    const auto fp32_bytes = std::filesystem::file_size(fp32_engine);
    const auto int8_bytes = std::filesystem::file_size(int8_engine);
    std::cout << "[REQ-017] D7 引擎体积：FP32 = " << fp32_bytes / 1024 / 1024
              << " MB，INT8 权重 = " << int8_bytes / 1024 / 1024 << " MB\n";
    EXPECT_LT(int8_bytes, fp32_bytes)
        << "INT8 引擎没有变小：DQ 很可能被构建期常量折叠（D7 判据不成立，应回 Gate-A）";
}

}  // namespace
}  // namespace mini_trt_llm
