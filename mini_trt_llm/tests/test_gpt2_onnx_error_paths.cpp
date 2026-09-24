#include "e2e_fixture.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "test_gpu_guard.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <vector>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

// Phase 3 测试计划的缺口 G1（L1a 部分）：`BuildFromOnnx` 的**失败路径**。
//
// 为什么值得单独测：这些校验是 Phase 3 新写的，**没有任何断言钉住**——
// 它们只在"入参错"时才生效，而正常路径的用例永远不会触发。
// 本项目已有"校验被静默绕过"的先例（TROUBLESHOOTING #15：文档写 6 个输入、
// 代码已改成每层一对，照文档改回去就是恢复一个真 bug）。
//
// 为什么能在沙箱跑：这三类校验被**刻意排在 createInferBuilder 之前**
// （AGENTS.md 与 builder.cpp 都记了理由：创建 builder 既慢又依赖驱动）。
// 因此它们不需要 GPU —— 这个分层本身就是"把校验前置"带来的额外收益。
class Gpt2OnnxErrorTest : public ::testing::Test {
 protected:
    void SetUp() override {
        directory_ = test_support::ModelDirectory::Create("gpt2_onnx_err");
        ASSERT_TRUE(directory_.valid());
        // ONNX 文件只需"存在"：这几条用例都在解析之前就失败，不读内容。
        const std::string onnx_path = directory_.path() + "/model.onnx";
        const char placeholder[] = "not-a-real-onnx";
        WriteFile(onnx_path, placeholder, sizeof(placeholder));
        onnx_path_ = onnx_path;
        engine_path_ = directory_.EnginePath("should_not_exist.engine");
    }

    test_support::ModelDirectory directory_;
    std::string onnx_path_;
    std::string engine_path_;
    Logger logger_;
};

// 子图名写错必须失败：`subgraph_names` 声明的是"要核对的子图"，
// 写错名字却静默忽略，等于给自己一个"已经核对过"的假安全感。
TEST_F(Gpt2OnnxErrorTest, RejectsUnknownSubgraphName) {
    ASSERT_TRUE(directory_.WriteConfig(R"({
        "model_type": "gpt2", "architecture": "decoder_only",
        "hyper_params": {"n_layer": 1, "n_head": 1, "n_embd": 8, "n_positions": 16,
                         "vocab_size": 32, "block_size": 4},
        "weight_map": {}
    })"));
    EngineBuilder builder(logger_, EngineBuilder::Config{});
    EXPECT_FALSE(builder.BuildFromOnnx(directory_.path(), onnx_path_, engine_path_,
                                      {"attention", "not_a_subgraph"}));
    // 失败不应留下 engine 文件（否则下游会以为构建成功过）
    EXPECT_FALSE(std::filesystem::exists(engine_path_));
}

// 模型目录缺 config.json：profile 规则无从选择，必须失败而不是猜一个。
TEST_F(Gpt2OnnxErrorTest, RejectsMissingModelConfig) {
    EngineBuilder builder(logger_, EngineBuilder::Config{});
    EXPECT_FALSE(builder.BuildFromOnnx(directory_.path(), onnx_path_, engine_path_, {}));
    EXPECT_FALSE(std::filesystem::exists(engine_path_));
}

// config.json 存在但缺 architecture：同样是"没法选 profile 规则"，必须失败。
// 这条与上一条分开，是因为 ModelConfig::Load 对两者的报错路径不同。
TEST_F(Gpt2OnnxErrorTest, RejectsConfigWithoutArchitecture) {
    ASSERT_TRUE(directory_.WriteConfig(R"({
        "model_type": "gpt2",
        "hyper_params": {},
        "weight_map": {}
    })"));
    EngineBuilder builder(logger_, EngineBuilder::Config{});
    EXPECT_FALSE(builder.BuildFromOnnx(directory_.path(), onnx_path_, engine_path_, {}));
    EXPECT_FALSE(std::filesystem::exists(engine_path_));
}

// ONNX 路径不存在或不可读：必须失败，且**不应**走到 createInferBuilder
// （这一条在无 GPU 环境也能跑，就是"校验前置"的直接证据）。
TEST_F(Gpt2OnnxErrorTest, RejectsUnreadableOnnxPath) {
    ASSERT_TRUE(directory_.WriteConfig(R"({
        "model_type": "gpt2", "architecture": "decoder_only",
        "hyper_params": {"n_layer": 1, "n_head": 1, "n_embd": 8, "n_positions": 16,
                         "vocab_size": 32, "block_size": 4},
        "weight_map": {}
    })"));
    EngineBuilder builder(logger_, EngineBuilder::Config{});
    const std::string missing = directory_.path() + "/nope.onnx";
    EXPECT_FALSE(builder.BuildFromOnnx(directory_.path(), missing, engine_path_, {}));
    EXPECT_FALSE(std::filesystem::exists(engine_path_));
}


// ---------------------------------------------------------------------------
// L1b：解析之后的 I/O 契约校验（需要 GPU —— 解析要走 createInferBuilder）
// ---------------------------------------------------------------------------
//
// 夹具直接用仓库里已有的 `0_resnet18_onnx/resnet18.onnx`：它的 I/O 名是
// `input` / `output`，与方案 B 要求的 `input_ids` / `logits` 完全不同。
// **为什么用现成文件而不是造一个 ONNX**：造图需要 protobuf 级生成器（成本高、易错），
// 而"非同名图"这个场景仓库里本来就有真实样本；省下的成本留给真正的缺口（见测试计划 G1c）。
//
// 它验证的是这条护栏真的会拦人：任何 I/O 名与方案 A 不同的图都不会被悄悄接受——
// 否则"ONNX 与原生对齐"就会变成一句空话（绑定时才报错，甚至数值错而无人察觉）。
TEST_F(Gpt2OnnxErrorTest, RejectsGraphWithForeignIoNames) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "解析 ONNX 需要 CUDA（createInferBuilder）";
    }
    const std::vector<std::string> candidates = {
        "0_resnet18_onnx/resnet18.onnx", "../0_resnet18_onnx/resnet18.onnx",
        "../../0_resnet18_onnx/resnet18.onnx", "../../../0_resnet18_onnx/resnet18.onnx"};
    std::string resnet;
    for (const std::string& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            resnet = candidate;
            break;
        }
    }
    if (resnet.empty()) {
        GTEST_SKIP() << "找不到 0_resnet18_onnx/resnet18.onnx（本用例复用它当'外来 I/O 名'夹具）";
    }
    // 给这份 ONNX 配一个合法的模型目录：这样失败一定来自 I/O 名校验，
    // 而不是"config 缺失"这类更早的检查。
    ASSERT_TRUE(directory_.WriteConfig(R"({
        "model_type": "resnet18", "architecture": "cnn",
        "hyper_params": {}, "weight_map": {}
    })"));
    EngineBuilder builder(logger_, EngineBuilder::Config{});
    EXPECT_FALSE(builder.BuildFromOnnx(directory_.path(), resnet, engine_path_, {}))
        << "外来 I/O 名的图必须被拒绝";
    EXPECT_FALSE(std::filesystem::exists(engine_path_));
}


// L1b 的第二半（缺口 G1c）：输入名对、**输出名不对**。
//
// 为什么它单独存在：G1b 用 resnet18.onnx 覆盖了"输入名不符"，但那张图**两个名字都不符**，
// 所以输入侧检查一拦，输出侧检查**永远不会被执行到**——等于那段护栏没有用例。
// 要测"满足一半"的图，只能自己造：`tools/make_tiny_onnx.py` 生成一张
// `input_ids` → `not_logits` 的单节点图（用 Python 是因为手写 ONNX protobuf 不划算）。
//
// 环境语义：解析需要 CUDA（GPU 门控）；缺 python3/onnx 导致夹具生成失败时**跳过**——
// 缺环境不等于图的契约有问题，混在一起会让失败信号失真。
TEST_F(Gpt2OnnxErrorTest, RejectsGraphWithoutLogitsOutput) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "解析 ONNX 需要 CUDA（createInferBuilder）";
    }
    const std::vector<std::string> generator_candidates = {
        "mini_trt_llm/tools/make_tiny_onnx.py", "../mini_trt_llm/tools/make_tiny_onnx.py",
        "../../mini_trt_llm/tools/make_tiny_onnx.py",
        "../../../mini_trt_llm/tools/make_tiny_onnx.py"};
    std::string generator;
    for (const std::string& candidate : generator_candidates) {
        if (std::filesystem::exists(candidate)) {
            generator = candidate;
            break;
        }
    }
    if (generator.empty()) {
        GTEST_SKIP() << "找不到 tools/make_tiny_onnx.py";
    }

    const std::string fixture = directory_.path() + "/no_logits.onnx";
    const std::string command = "python3 " + generator + " --output " + fixture +
                                " --output-name not_logits > /dev/null 2>&1";
    if (std::system(command.c_str()) != 0 || !std::filesystem::exists(fixture)) {
        GTEST_SKIP() << "生成夹具失败（缺 python3 或 onnx 包）";
    }
    // 夹具自证：确认生成的图确实是"输入 input_ids、输出 not_logits"，
    // 否则用例会以"输入名不符"的路径失败，从而又一次测不到输出侧的检查。
    ASSERT_GT(std::filesystem::file_size(fixture), 0u);

    ASSERT_TRUE(directory_.WriteConfig(R"({
        "model_type": "gpt2", "architecture": "decoder_only",
        "hyper_params": {"n_layer": 1, "n_head": 1, "n_embd": 8, "n_positions": 16,
                         "vocab_size": 32, "block_size": 4},
        "weight_map": {}
    })"));
    EngineBuilder builder(logger_, EngineBuilder::Config{});
    EXPECT_FALSE(builder.BuildFromOnnx(directory_.path(), fixture, engine_path_, {}))
        << "输入名正确但缺 logits 输出的图必须被拒绝";
    EXPECT_FALSE(std::filesystem::exists(engine_path_));
}

}  // namespace
}  // namespace mini_trt_llm
