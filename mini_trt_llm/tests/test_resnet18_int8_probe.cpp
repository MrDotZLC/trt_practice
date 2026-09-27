// P4-INT8-a（`docs/future_iterations.md` + OI-INT8-PERCHANNEL）：per-channel 权重量化的**整网退化根因**。
//
// 设计（D1~D6，出处 `docs/future_iterations_development_plan.md` + OI-INT8-PERCHANNEL-DESIGN）：
//   · D1 标尺 = **ONNX 官方参考实现**执行同一张 Q/DQ 图（不由本工程折 BN）；
//   · D2 只比**量化前**的 float 张量（量化后会被 bin 边界 ±1 格噪声淹没，#30.5/#30.6）；
//   · D3 "探到的是量化前"用 `d_pre ≤ d_post` 自证（TRT 若把量化后再反量化的值交回来，
//        d_post 会反过来更小）；
//   · D4 噪声地板**当场由 PT 臂量出**，不预设绝对阈值；
//   · D6 还要验证"探针图下退化仍然复现"，否则仪器改变了现象。
//
// 用例的验收**不是"绿 / 红"**，而是 §1.5 的"二选一"结论——所以这里的断言只覆盖
// "仪器可信"与"现象复现"两件事，逐层曲线只是一份**报告**（见测试计划 §3）。

#include "cv_test_support.hpp"
#include "diff_stats.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include "test_asset_guard.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::ArgmaxOfRow;
using test_support::ComputeDiffStats;
using test_support::DiffStats;
using test_support::FindFile;
using test_support::kCvChannels;
using test_support::kCvClasses;
using test_support::kCvSize;
using test_support::ReadF32File;

// 逐层曲线只跑一个 batch —— 与参考落盘（`--num-images 8`）严格对齐。
constexpr int32_t kProbeBatch = 8;
const size_t kProbeInputElements =
    static_cast<size_t>(kProbeBatch) * kCvChannels * kCvSize * kCvSize;

// 复现对照（B1-4）与既有 INT8 用例同口径：256 张、batch=8、`margin ≥ 5` 算"有余量"。
constexpr int32_t kReproBatch = 8;
constexpr int32_t kReproImages = 256;
constexpr float kConfidentMargin = 5.0f;
const size_t kReproInputElements =
    static_cast<size_t>(kReproBatch) * kCvChannels * kCvSize * kCvSize;
const size_t kLogitsElements = static_cast<size_t>(kReproBatch) * kCvClasses;

// 越过噪声地带的判据倍数。出处（开发计划 §13.3 D4）：失败幅度是"让 margin≥5 的样本翻类"
// 即 O(1)，而正常逐层差异是 1e-3 量级 —— 两者差 3 个数量级，取 100× 仍低一个数量级。
constexpr double kDivergenceFactor = 100.0;

// 两臂的探针图 / 引擎 / 参考目录。全部可用环境变量覆盖（隔离实验用），默认走本轮产出。
std::string EnvOr(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::string(value) : fallback;
}

std::string ProbeOnnx(const char* env, const std::string& fallback) {
    const std::string path = EnvOr(env, fallback);
    return FindFile({path, "../" + path, "../../" + path, "../../../" + path});
}

std::string ProbeEngine(const char* env, const std::string& fallback) {
    return EnvOr(env, fallback);
}

std::string ProbeRefDir(const char* env, const std::string& fallback) {
    return EnvOr(env, fallback);
}

const char* const kFp32Engine = "/tmp/mini_trt_llm_resnet18_onnx_fp32.engine";

std::string FindModelDir() {
    const std::string config = FindFile({"models/resnet18/config.json",
                                         "../models/resnet18/config.json",
                                         "../../models/resnet18/config.json",
                                         "../../../models/resnet18/config.json"});
    if (config.empty()) {
        return {};
    }
    return std::filesystem::path(config).parent_path().string();
}

std::string FindOnnxPath() {
    return FindFile({"assets/legacy/resnet18_onnx/resnet18.onnx", "../assets/legacy/resnet18_onnx/resnet18.onnx",
                     "../../assets/legacy/resnet18_onnx/resnet18.onnx",
                     "../../../assets/legacy/resnet18_onnx/resnet18.onnx"});
}

std::string FindCalibDir() {
    return FindFile({"assets/legacy/resnet18_onnx/calib_data", "../assets/legacy/resnet18_onnx/calib_data",
                     "../../assets/legacy/resnet18_onnx/calib_data",
                     "../../../assets/legacy/resnet18_onnx/calib_data"});
}

EngineBuilder::Config ProbeConfig() {
    EngineBuilder::Config config;
    config.precision = Precision::INT8;
    // 逐层精度只有建图时打开才读得到（TROUBLESHOOTING #27）。这里一并把 PC/PT 的
    // tactic 分布打出来——**当根因不在 TRT 侧时，这条信息就是"TRT 无辜"的直接证据**。
    config.detailed_profiling = true;
    return config;
}

EngineBuilder::Config Fp32Config() {
    EngineBuilder::Config config;
    config.precision = Precision::FP32;
    return config;
}

// `probe_index.txt` 的一行：`<张量名>\t<元素数>\t<文件名>\t<角色>\t<配对张量>`。
// 角色与配对规则由 `tools/validate/qdq_reference.py` 定义（两侧必须一致）。
struct ProbeIndexRow {
    std::string name;
    size_t elements = 0;
    std::string file;
    std::string role;    // "probe"（也是引擎输出）/ "postquant"（只有参考有）
    std::string paired;  // postquant 行：它属于哪个 probe 张量
};

bool ReadProbeIndex(const std::string& dir, std::vector<ProbeIndexRow>* rows,
                    std::string* error) {
    std::ifstream in(dir + "/probe_index.txt");
    if (!in) {
        *error = "读不到 " + dir + "/probe_index.txt（先用 qdq_reference.py 落盘）";
        return false;
    }
    // 按制表符切分，而不是混用 `>>` 与 `getline`：混用时数字后面的制表符会留在流里，
    // 于是文件名开头多一个 '\t'，而且不报错——这类"静默多一个字符"最难查。
    const auto split_tabs = [](const std::string& text) {
        std::vector<std::string> fields;
        std::string field;
        for (char ch : text) {
            if (ch == '\t') {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(ch);
            }
        }
        fields.push_back(field);
        return fields;
    };
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_tabs(line);
        if (fields.size() != 5) {
            *error = "probe_index.txt 列数不是 5：" + line;
            return false;
        }
        ProbeIndexRow row;
        row.name = fields[0];
        row.file = fields[2];
        row.role = fields[3];
        row.paired = fields[4];
        try {
            row.elements = static_cast<size_t>(std::stoull(fields[1]));
        } catch (const std::exception&) {
            *error = "probe_index.txt 的元素数不是整数：" + line;
            return false;
        }
        rows->push_back(row);
    }
    if (rows->empty()) {
        *error = "probe_index.txt 是空的";
        return false;
    }
    return true;
}

struct EngineRun {
    // 每个 I/O 张量的读回值，按引擎输出顺序。
    std::vector<std::pair<std::string, std::vector<float>>> tensors;

    const std::vector<float>* Find(const std::string& name) const {
        for (const auto& entry : tensors) {
            if (entry.first == name) {
                return &entry.second;
            }
        }
        return nullptr;
    }
};

// 跑一次探针引擎，把**全部** I/O 张量读回。输入张量也必须绑定（TRT 要求所有 I/O 都有地址）。
bool RunProbeEngine(Engine* engine, const std::vector<float>& input, int32_t batch,
                    EngineRun* out, std::string* error) {
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        *error = "engine 为空";
        return false;
    }
    if (!engine->SetOptimizationProfile(0, nullptr)) {
        *error = "setOptimizationProfile(0) 失败";
        return false;
    }
    if (!engine->SetInputShape("input", nvinfer1::Dims4{batch, kCvChannels, kCvSize, kCvSize})) {
        *error = "setInputShape 失败";
        return false;
    }

    // **形状必须问 IExecutionContext，不能问 ICudaEngine**：`ICudaEngine::getTensorShape` 对
    // 动态维返回 **-1**（引擎是"形状无关"的），转成 size_t 就是个天文数字 →
    // 上一版因此在第一张量就报"显存分配失败"。上下文在 `setInputShape` 之后才知道真实形状。
    nvinfer1::IExecutionContext* context = engine->GetContext();
    if (context == nullptr) {
        *error = "engine 没有 execution context";
        return false;
    }
    const int32_t count = cuda->getNbIOTensors();
    std::vector<std::unique_ptr<DeviceBuffer>> buffers;
    std::vector<std::string> names;
    std::vector<bool> is_input;
    for (int32_t i = 0; i < count; ++i) {
        const std::string name = cuda->getIOTensorName(i);
        const auto mode = cuda->getTensorIOMode(name.c_str());
        const auto dims = context->getTensorShape(name.c_str());
        size_t elements = 1;
        for (int32_t d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] <= 0) {
                std::ostringstream message;
                message << "张量 " << name << " 的第 " << d << " 维仍是动态/非法值 (" << dims.d[d]
                        << ")：setInputShape 之后它应当已被解析";
                *error = message.str();
                return false;
            }
            elements *= static_cast<size_t>(dims.d[d]);
        }
        auto buffer = std::make_unique<DeviceBuffer>(elements * sizeof(float));
        if (!buffer->Allocate(elements * sizeof(float))) {
            *error = "显存分配失败：" + name;
            return false;
        }
        if (!engine->SetTensorAddress(name, buffer->data())) {
            *error = "setTensorAddress 失败：" + name;
            return false;
        }
        names.push_back(name);
        is_input.push_back(mode == nvinfer1::TensorIOMode::kINPUT);
        buffers.push_back(std::move(buffer));
    }

    bool saw_input = false;
    for (size_t i = 0; i < names.size(); ++i) {
        if (!is_input[i]) {
            continue;
        }
        const size_t bytes = input.size() * sizeof(float);
        if (buffers[i]->size() < bytes) {
            *error = "输入缓冲小于喂入数据：" + names[i];
            return false;
        }
        CUDA_CHECK(cudaMemcpy(buffers[i]->data(), input.data(), bytes, cudaMemcpyHostToDevice));
        saw_input = true;
    }
    if (!saw_input) {
        *error = "引擎没有任何输入张量";
        return false;
    }
    if (!engine->Enqueue(nullptr)) {
        *error = "enqueueV3 失败";
        return false;
    }
    engine->Synchronize(nullptr);

    out->tensors.clear();
    for (size_t i = 0; i < names.size(); ++i) {
        if (is_input[i]) {
            continue;
        }
        const size_t elements = buffers[i]->size() / sizeof(float);
        out->tensors.emplace_back(names[i], test_support::ReadFloats(buffers[i]->data(), elements));
    }
    return true;
}

// 逐层信息里的 INT8 证据（计数口径与 `test_resnet18_int8.cpp` 一致），外加**逐层落盘**。
//
// 为什么要落盘：探针图多了 21 个图输出，**会改变 TRT 的融合与 tactic 选择**（实测：产物图有
// `i8i8` tactic，探针图上这个计数变成 0）。这正是开发计划 §13.3 D6 要防的那件事，所以
// 不能只看一个计数——把每层的 ONELINE 原文写下来，才能判断"变的是哪几层、变成了什么"。
struct TacticStats {
    int32_t layers = 0;
    int32_t int8_tensors = 0;
    int32_t i8i8_tactics = 0;
    std::vector<std::string> distinct_tactics;
    std::string dump_path;  // 落盘成功时非空
};

TacticStats InspectTactics(Engine* engine, const std::string& dump_path) {
    TacticStats stats;
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        return stats;
    }
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(cuda->createEngineInspector());
    if (inspector == nullptr) {
        return stats;
    }
    std::ofstream dump(dump_path);
    stats.layers = cuda->getNbLayers();
    for (int32_t i = 0; i < stats.layers; ++i) {
        const char* line =
            inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kONELINE);
        if (line == nullptr) {
            continue;
        }
        const std::string text(line);
        if (dump) {
            dump << i << "\t" << text << "\n";
        }
        if (text.find("Format/Datatype: Int8") != std::string::npos) {
            ++stats.int8_tensors;
        }
        if (text.find("i8i8") != std::string::npos) {
            ++stats.i8i8_tactics;
        }
        // TacticName 的原文（供人核对"两臂到底换没换 kernel"）。
        const size_t at = text.find("TacticName: ");
        if (at != std::string::npos) {
            const size_t begin = at + std::string("TacticName: ").size();
            const size_t end = text.find(',', begin);
            const std::string name = text.substr(begin, end - begin);
            if (std::find(stats.distinct_tactics.begin(), stats.distinct_tactics.end(), name) ==
                stats.distinct_tactics.end()) {
                stats.distinct_tactics.push_back(name);
            }
        }
    }
    if (dump) {
        stats.dump_path = dump_path;
    }
    return stats;
}

std::string TacticSummary(const TacticStats& stats, size_t limit) {
    std::string text;
    for (size_t i = 0; i < stats.distinct_tactics.size() && i < limit; ++i) {
        text += (i == 0 ? "" : " | ");
        text += stats.distinct_tactics[i];
    }
    if (stats.distinct_tactics.size() > limit) {
        text += " | …(共 " + std::to_string(stats.distinct_tactics.size()) + " 种)";
    }
    if (text.empty()) {
        text = "(没有 TacticName —— 逐层信息里读不到 tactic)";
    }
    return text;
}

std::vector<float> ReadBatchInput(const std::vector<std::string>& files, int32_t start,
                                  int32_t batch) {
    const size_t per_image = static_cast<size_t>(kCvChannels) * kCvSize * kCvSize;
    std::vector<float> batch_input(static_cast<size_t>(batch) * per_image);
    for (int32_t i = 0; i < batch; ++i) {
        const std::vector<float> image = ReadF32File(files[static_cast<size_t>(start + i)], per_image);
        if (image.size() != per_image) {
            return {};
        }
        std::copy(image.begin(), image.end(),
                  batch_input.begin() + static_cast<ptrdiff_t>(i) * static_cast<ptrdiff_t>(per_image));
    }
    return batch_input;
}

std::vector<std::string> CalibFiles(const std::string& calib_dir) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(calib_dir)) {
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// 一条曲线上的一个点。
struct CurvePoint {
    std::string name;
    double max_abs = 0.0;
    double max_rel = 0.0;
};

// 两臂的公共前置：路径齐了才继续，否则用**带指引的跳过**（不是失败——缺产物不等于缺陷）。
struct ProbeSetup {
    std::string model_dir;
    std::string calib_dir;
    std::string pt_onnx;
    std::string pt_engine;
    std::string pt_ref;
    std::string pc_onnx;
    std::string pc_engine;
    std::string pc_ref;

    bool Complete() const {
        return !model_dir.empty() && !calib_dir.empty() && !pt_onnx.empty() && !pc_onnx.empty() &&
               std::filesystem::exists(pt_ref + "/probe_index.txt") &&
               std::filesystem::exists(pc_ref + "/probe_index.txt");
    }
};

ProbeSetup MakeSetup() {
    ProbeSetup setup;
    setup.model_dir = FindModelDir();
    setup.calib_dir = FindCalibDir();
    setup.pt_onnx = ProbeOnnx("MINI_TRT_PROBE_PT_ONNX",
                              "models/resnet18/resnet18_qdq_probe_per_tensor.onnx");
    setup.pc_onnx = ProbeOnnx("MINI_TRT_PROBE_PC_ONNX",
                              "models/resnet18/resnet18_qdq_probe_per_channel.onnx");
    setup.pt_engine = ProbeEngine("MINI_TRT_PROBE_PT_ENGINE",
                                  "/tmp/mini_trt_llm_resnet18_qdq_probe_pt.engine");
    setup.pc_engine = ProbeEngine("MINI_TRT_PROBE_PC_ENGINE",
                                  "/tmp/mini_trt_llm_resnet18_qdq_probe_pc.engine");
    setup.pt_ref = ProbeRefDir("MINI_TRT_PROBE_PT_REF", "/tmp/mini_trt_llm_int8_probe/pt");
    setup.pc_ref = ProbeRefDir("MINI_TRT_PROBE_PC_REF", "/tmp/mini_trt_llm_int8_probe/pc");
    return setup;
}

const char* const kHowToPrepare =
    "先产出两臂探针图与参考落盘（见开发计划 §13.9）：\n"
    "  python3 mini_trt_llm/tools/convert/quantize_resnet18.py --onnx assets/legacy/resnet18_onnx/resnet18.onnx \\\n"
    "      --calib-dir assets/legacy/resnet18_onnx/calib_data --weight-scope per_channel \\\n"
    "      --output models/resnet18/resnet18_qdq_per_channel.onnx\n"
    "  python3 mini_trt_llm/tools/convert/add_probe_outputs.py --onnx models/resnet18/resnet18_qdq.onnx \\\n"
    "      --output models/resnet18/resnet18_qdq_probe_per_tensor.onnx\n"
    "  python3 mini_trt_llm/tools/convert/add_probe_outputs.py --onnx models/resnet18/resnet18_qdq_per_channel.onnx \\\n"
    "      --output models/resnet18/resnet18_qdq_probe_per_channel.onnx\n"
    "  python3 mini_trt_llm/tools/validate/qdq_reference.py --onnx models/resnet18/resnet18_qdq_probe_per_tensor.onnx \\\n"
    "      --calib-dir assets/legacy/resnet18_onnx/calib_data --num-images 8 --output-dir /tmp/mini_trt_llm_int8_probe/pt\n"
    "  python3 mini_trt_llm/tools/validate/qdq_reference.py --onnx models/resnet18/resnet18_qdq_probe_per_channel.onnx \\\n"
    "      --calib-dir assets/legacy/resnet18_onnx/calib_data --num-images 8 --output-dir /tmp/mini_trt_llm_int8_probe/pc\n";

}  // namespace

// ---------------------------------------------------------------------------
// B1-2：同一引擎、同一输入重复跑 → **逐位相同**。
//
// 为什么先证这个：误差曲线要拿"某一次运行"当事实，如果引擎本身不确定（并行归约顺序、
// 未初始化显存），曲线上的抖动就分不清是"实现差异"还是"运行差异"。
// ---------------------------------------------------------------------------
TEST(Int8ProbeTest, SameEngineSameInputIsBitIdentical) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const ProbeSetup setup = MakeSetup();
    if (!setup.Complete()) {
        MINI_TRT_SKIP_IF_MISSING_ASSET("需要两臂探针图 + 参考落盘。\n" << kHowToPrepare);
    }
    Logger logger;
    EngineBuilder builder(logger, ProbeConfig());
    ASSERT_TRUE(builder.BuildFromOnnx(setup.model_dir, setup.pt_onnx, setup.pt_engine, {}));
    Engine engine(setup.pt_engine, logger);

    const std::vector<std::string> files = CalibFiles(setup.calib_dir);
    ASSERT_GE(static_cast<int32_t>(files.size()), kProbeBatch);
    const std::vector<float> input = ReadBatchInput(files, 0, kProbeBatch);
    ASSERT_EQ(input.size(), kProbeInputElements);

    EngineRun first;
    EngineRun second;
    std::string error;
    ASSERT_TRUE(RunProbeEngine(&engine, input, kProbeBatch, &first, &error)) << error;
    ASSERT_TRUE(RunProbeEngine(&engine, input, kProbeBatch, &second, &error)) << error;

    ASSERT_EQ(first.tensors.size(), second.tensors.size());
    int32_t compared = 0;
    for (size_t i = 0; i < first.tensors.size(); ++i) {
        ASSERT_EQ(first.tensors[i].first, second.tensors[i].first);
        ASSERT_EQ(first.tensors[i].second.size(), second.tensors[i].second.size());
        const bool same = std::equal(first.tensors[i].second.begin(), first.tensors[i].second.end(),
                                     second.tensors[i].second.begin());
        EXPECT_TRUE(same) << "张量 " << first.tensors[i].first << " 两次运行不逐位相同";
        ++compared;
    }
    std::cout << "[Int8Probe] 确定性：比较了 " << compared << " 个张量，全部逐位相同\n";
}

// ---------------------------------------------------------------------------
// B1-1 / B1-3：与 ONNX 官方参考实现逐层对拍。
//
// 断言只覆盖两件事（仪器是否可信），曲线的解释权归 §1.5 的"二选一"结论：
//   ① `d_pre ≤ d_post`：探到的确实是**量化前**张量（D3）；
//   ② 两臂的 conv1 都落在噪声地带上（如果连第一层都对不上，说明标尺或绑定有问题）。
// ---------------------------------------------------------------------------
TEST(Int8ProbeTest, LayerwiseErrorGrowthVsOnnxReference) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const ProbeSetup setup = MakeSetup();
    if (!setup.Complete()) {
        MINI_TRT_SKIP_IF_MISSING_ASSET("需要两臂探针图 + 参考落盘。\n" << kHowToPrepare);
    }
    const std::vector<std::string> files = CalibFiles(setup.calib_dir);
    if (static_cast<int32_t>(files.size()) < kProbeBatch) {
        MINI_TRT_SKIP_IF_MISSING_ASSET("标定图不足 " << kProbeBatch << " 张");
    }
    const std::vector<float> input = ReadBatchInput(files, 0, kProbeBatch);
    ASSERT_EQ(input.size(), kProbeInputElements);

    Logger logger;
    EngineBuilder pt_builder(logger, ProbeConfig());
    EngineBuilder pc_builder(logger, ProbeConfig());
    ASSERT_TRUE(pt_builder.BuildFromOnnx(setup.model_dir, setup.pt_onnx, setup.pt_engine, {}));
    ASSERT_TRUE(pc_builder.BuildFromOnnx(setup.model_dir, setup.pc_onnx, setup.pc_engine, {}));
    Engine pt_engine(setup.pt_engine, logger);
    Engine pc_engine(setup.pc_engine, logger);

    const TacticStats pt_tactics = InspectTactics(&pt_engine, setup.pt_ref + "/layers_pt.txt");
    const TacticStats pc_tactics = InspectTactics(&pc_engine, setup.pc_ref + "/layers_pc.txt");
    std::cout << "[Int8Probe] PT 引擎：层=" << pt_tactics.layers
              << " 含 Int8 张量=" << pt_tactics.int8_tensors
              << " i8i8 tactic=" << pt_tactics.i8i8_tactics << "\n"
              << "[Int8Probe]   tactic 种类：" << TacticSummary(pt_tactics, 6) << "\n"
              << "[Int8Probe]   逐层 ONELINE 落盘：" << pt_tactics.dump_path << "\n"
              << "[Int8Probe] PC 引擎：层=" << pc_tactics.layers
              << " 含 Int8 张量=" << pc_tactics.int8_tensors
              << " i8i8 tactic=" << pc_tactics.i8i8_tactics << "\n"
              << "[Int8Probe]   tactic 种类：" << TacticSummary(pc_tactics, 6) << "\n"
              << "[Int8Probe]   逐层 ONELINE 落盘：" << pc_tactics.dump_path << "\n"
              // 对照：**产物图**（没有探针输出）的 INT8 证据——由既有用例
              // `ResNet18Int8EngineTest.IsActuallyInt8` 给出（44 层 / 38 含 Int8 / 4 个 i8i8）。
              // 探针图上的计数与它不同属于**预期**（多挂了 21 个图输出），但必须让读者看见，
              // 免得把"探针改变了 tactic"误读成"模型没在跑 INT8"。
              << "[Int8Probe]   对照（产物图、不含探针输出）：44 层 / 38 含 Int8 / 4 个 i8i8\n";

    EngineRun pt_run;
    EngineRun pc_run;
    std::string error;
    ASSERT_TRUE(RunProbeEngine(&pt_engine, input, kProbeBatch, &pt_run, &error)) << error;
    ASSERT_TRUE(RunProbeEngine(&pc_engine, input, kProbeBatch, &pc_run, &error)) << error;

    // 参考侧的落盘（probe 行 = 引擎输出；postquant 行只有参考有）。
    std::vector<ProbeIndexRow> pt_rows;
    std::vector<ProbeIndexRow> pc_rows;
    ASSERT_TRUE(ReadProbeIndex(setup.pt_ref, &pt_rows, &error)) << error;
    ASSERT_TRUE(ReadProbeIndex(setup.pc_ref, &pc_rows, &error)) << error;
    ASSERT_EQ(pt_rows.size(), pc_rows.size()) << "两臂的探针清单长度应当一致";

    // postquant 张量按"它属于哪个 probe 张量"归位，供 D3 自证使用。
    std::map<std::string, std::string> pt_post;  // probe name -> file
    std::map<std::string, std::string> pc_post;
    for (const ProbeIndexRow& row : pt_rows) {
        if (row.role == "postquant") {
            pt_post[row.paired] = setup.pt_ref + "/" + row.file;
        }
    }
    for (const ProbeIndexRow& row : pc_rows) {
        if (row.role == "postquant") {
            pc_post[row.paired] = setup.pc_ref + "/" + row.file;
        }
    }

    std::vector<CurvePoint> pt_curve;
    std::vector<CurvePoint> pc_curve;
    std::string first_conv_output;  // 第一层（conv1）的探针张量名
    double pt_pre_of_conv1 = 0.0;
    double pt_post_of_conv1 = 0.0;
    double pc_pre_of_conv1 = 0.0;
    double pc_post_of_conv1 = 0.0;

    for (const ProbeIndexRow& row : pt_rows) {
        if (row.role != "probe") {
            continue;
        }
        const std::vector<float> reference = ReadF32File(setup.pt_ref + "/" + row.file, row.elements);
        const std::vector<float> pc_reference = ReadF32File(setup.pc_ref + "/" + row.file, row.elements);
        ASSERT_EQ(reference.size(), row.elements) << "参考落盘读不出来：" << row.file;
        ASSERT_EQ(pc_reference.size(), row.elements);
        const std::vector<float>* got = pt_run.Find(row.name);
        const std::vector<float>* pc_got = pc_run.Find(row.name);
        ASSERT_NE(got, nullptr) << "引擎里没有输出张量 " << row.name;
        ASSERT_NE(pc_got, nullptr) << "PC 引擎里没有输出张量 " << row.name;
        ASSERT_EQ(got->size(), row.elements);
        ASSERT_EQ(pc_got->size(), row.elements);

        const DiffStats pt_stats = ComputeDiffStats(reference, *got);
        const DiffStats pc_stats = ComputeDiffStats(pc_reference, *pc_got);
        pt_curve.push_back({row.name, pt_stats.max_abs, pt_stats.max_rel});
        pc_curve.push_back({row.name, pc_stats.max_abs, pc_stats.max_rel});

        if (row.name.find("Conv_output_0") != std::string::npos && first_conv_output.empty()) {
            first_conv_output = row.name;
            pt_pre_of_conv1 = pt_stats.max_abs;
            pc_pre_of_conv1 = pc_stats.max_abs;
            // D3：同一层再比一次"量化后"的参考值。引擎交回的若是量化后再反量化的值，
            // d_post 会明显小于 d_pre。
            const auto pt_post_file = pt_post.find(row.name);
            if (pt_post_file != pt_post.end()) {
                const std::vector<float> post = ReadF32File(pt_post_file->second, row.elements);
                if (post.size() == row.elements) {
                    pt_post_of_conv1 = ComputeDiffStats(post, *got).max_abs;
                }
            }
            const auto pc_post_file = pc_post.find(row.name);
            if (pc_post_file != pc_post.end()) {
                const std::vector<float> post = ReadF32File(pc_post_file->second, row.elements);
                if (post.size() == row.elements) {
                    pc_post_of_conv1 = ComputeDiffStats(post, *pc_got).max_abs;
                }
            }
        }
    }
    ASSERT_FALSE(first_conv_output.empty()) << "探针清单里没有 Conv 输出，清单不对";

    std::cout << "[Int8Probe] 逐层 |引擎 − ONNX 参考|（顺序 = 图拓扑序）：\n";
    for (size_t i = 0; i < pt_curve.size(); ++i) {
        std::cout << "[Int8Probe]   " << (i + 1) << " " << pt_curve[i].name << "\n"
                  << "[Int8Probe]        PT max_abs=" << pt_curve[i].max_abs
                  << " max_rel=" << pt_curve[i].max_rel
                  << "   PC max_abs=" << pc_curve[i].max_abs
                  << " max_rel=" << pc_curve[i].max_rel << "\n";
    }

    // D4：噪声地板由 PT 臂（已知健康）当场量出，**不预设绝对阈值**。
    double noise_floor = 0.0;
    for (const CurvePoint& point : pt_curve) {
        noise_floor = std::max(noise_floor, point.max_abs);
    }
    const double threshold = kDivergenceFactor * noise_floor;
    int32_t first_diverged = -1;
    for (size_t i = 0; i < pc_curve.size(); ++i) {
        if (pc_curve[i].max_abs > threshold) {
            first_diverged = static_cast<int32_t>(i);
            break;
        }
    }
    std::cout << "[Int8Probe] 噪声地板 (PT 的逐层 max_abs 上界) = " << noise_floor
              << "；判据 = " << kDivergenceFactor << " × 地板 = " << threshold << "\n";
    // **这条判据问的是"引擎有没有跑偏它自己的图"，不是"两臂谁更准"。**
    // 别把它读成"整网有没有分叉"——那件事由 B1-4（复现对照）与文件级的饱和统计回答
    // （`TROUBLESHOOTING.md` #46 / #47.3）。
    if (first_diverged < 0) {
        std::cout << "[Int8Probe] 两臂都**未越过**判据 → 两个引擎都忠实执行了各自的图。\n"
                     "[Int8Probe]   ⇒ PC 更差**不是引擎造成的**，而在于图（文件）本身："
                     "权重 scale 取自未折 BN 的权重，\n"
                     "[Int8Probe]     导致 16.19% 的 int8 权重被 clamp 饱和、而改源后只有 0.044%"
                     "（见 `TROUBLESHOOTING.md` #46 / #47.3）。\n";
    } else {
        std::cout << "[Int8Probe] PC 臂**首次越界**在第 " << (first_diverged + 1)
                  << " 个张量：" << pc_curve[static_cast<size_t>(first_diverged)].name
                  << "（max_abs=" << pc_curve[static_cast<size_t>(first_diverged)].max_abs
                  << "，是地板的 "
                  << pc_curve[static_cast<size_t>(first_diverged)].max_abs / noise_floor
                  << " 倍）—— 说明**引擎**没跑对那张图，另当别论。\n";
    }

    // ①+D3：探到的必须是**量化前**张量。
    std::cout << "[Int8Probe] 探针自证（" << first_conv_output << "）：PT d_pre=" << pt_pre_of_conv1
              << " d_post=" << pt_post_of_conv1 << "；PC d_pre=" << pc_pre_of_conv1
              << " d_post=" << pc_post_of_conv1 << "\n";
    ASSERT_GT(pt_post_of_conv1, 0.0) << "没有读到量化后的参考值，探针自证无法进行";
    ASSERT_GT(pc_post_of_conv1, 0.0);
    EXPECT_LE(pt_pre_of_conv1, pt_post_of_conv1)
        << "PT 臂的探针更像'量化后'的张量 —— 仪器探错对象了（开发计划 §13.3 D3）";
    EXPECT_LE(pc_pre_of_conv1, pc_post_of_conv1)
        << "PC 臂的探针更像'量化后'的张量 —— 仪器探错对象了（开发计划 §13.3 D3）";

    // ②：首层必须在噪声地带上（对不上就是标尺/绑定错了，不是"发现"）。
    EXPECT_LE(pt_pre_of_conv1, threshold) << "PT 臂第一层就偏离参考，标尺有问题";
    EXPECT_LE(pc_pre_of_conv1, threshold) << "PC 臂第一层就偏离参考，标尺有问题";
}

// ---------------------------------------------------------------------------
// B1-4：**探针图下退化仍复现**（硬门，开发计划 §13.3 D6）。
//
// 挂额外图输出会改变 TRT 的融合。若探针图下 PC 不再更差，说明仪器把现象改掉了，
// B1-3 的曲线不能用来解释原来的退化。口径与正式产物完全一致：256 张、batch=8、
// `margin ≥ 5` 的余量子集一致率（实测 PT 100% / PC 54.5%）。
// ---------------------------------------------------------------------------
TEST(Int8ProbeTest, PerChannelDegradationReproducesUnderProbe) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const ProbeSetup setup = MakeSetup();
    const std::string fp32_onnx = FindOnnxPath();
    if (!setup.Complete() || fp32_onnx.empty()) {
        MINI_TRT_SKIP_IF_MISSING_ASSET("需要两臂探针图 + 参考落盘 + FP32 ONNX。\n" << kHowToPrepare);
    }
    const std::vector<std::string> files = CalibFiles(setup.calib_dir);
    if (static_cast<int32_t>(files.size()) < kReproImages) {
        MINI_TRT_SKIP_IF_MISSING_ASSET("标定图不足 " << kReproImages << " 张");
    }

    Logger logger;
    EngineBuilder fp32_builder(logger, Fp32Config());
    EngineBuilder pt_builder(logger, ProbeConfig());
    EngineBuilder pc_builder(logger, ProbeConfig());
    ASSERT_TRUE(fp32_builder.BuildFromOnnx(setup.model_dir, fp32_onnx, kFp32Engine, {}));
    ASSERT_TRUE(pt_builder.BuildFromOnnx(setup.model_dir, setup.pt_onnx, setup.pt_engine, {}));
    ASSERT_TRUE(pc_builder.BuildFromOnnx(setup.model_dir, setup.pc_onnx, setup.pc_engine, {}));
    Engine fp32_engine(kFp32Engine, logger);
    Engine pt_engine(setup.pt_engine, logger);
    Engine pc_engine(setup.pc_engine, logger);

    int32_t pt_agree = 0;
    int32_t pc_agree = 0;
    int32_t confident = 0;
    int32_t pt_confident_agree = 0;
    int32_t pc_confident_agree = 0;
    for (int32_t start = 0; start + kReproBatch <= kReproImages; start += kReproBatch) {
        const std::vector<float> input = ReadBatchInput(files, start, kReproBatch);
        ASSERT_EQ(input.size(), kReproInputElements);
        EngineRun fp32_run;
        EngineRun pt_run;
        EngineRun pc_run;
        std::string error;
        ASSERT_TRUE(RunProbeEngine(&fp32_engine, input, kReproBatch, &fp32_run, &error)) << error;
        ASSERT_TRUE(RunProbeEngine(&pt_engine, input, kReproBatch, &pt_run, &error)) << error;
        ASSERT_TRUE(RunProbeEngine(&pc_engine, input, kReproBatch, &pc_run, &error)) << error;
        const std::vector<float>* fp32_logits = fp32_run.Find("output");
        const std::vector<float>* pt_logits = pt_run.Find("output");
        const std::vector<float>* pc_logits = pc_run.Find("output");
        ASSERT_NE(fp32_logits, nullptr);
        ASSERT_NE(pt_logits, nullptr);
        ASSERT_NE(pc_logits, nullptr);
        ASSERT_EQ(fp32_logits->size(), kLogitsElements);
        ASSERT_EQ(pt_logits->size(), kLogitsElements);
        ASSERT_EQ(pc_logits->size(), kLogitsElements);

        for (int32_t i = 0; i < kReproBatch; ++i) {
            const bool pt_same = ArgmaxOfRow(*fp32_logits, i) == ArgmaxOfRow(*pt_logits, i);
            const bool pc_same = ArgmaxOfRow(*fp32_logits, i) == ArgmaxOfRow(*pc_logits, i);
            pt_agree += pt_same ? 1 : 0;
            pc_agree += pc_same ? 1 : 0;
            const float* row = fp32_logits->data() + static_cast<size_t>(i) * kCvClasses;
            float best = row[0];
            float second = -1e30f;
            for (int32_t c = 1; c < kCvClasses; ++c) {
                if (row[c] > best) {
                    second = best;
                    best = row[c];
                } else if (row[c] > second) {
                    second = row[c];
                }
            }
            if (best - second >= kConfidentMargin) {
                ++confident;
                pt_confident_agree += pt_same ? 1 : 0;
                pc_confident_agree += pc_same ? 1 : 0;
            }
        }
    }

    const double pt_rate = static_cast<double>(pt_confident_agree) / std::max(confident, 1);
    const double pc_rate = static_cast<double>(pc_confident_agree) / std::max(confident, 1);
    std::cout << "[Int8Probe] 探针图下复现对照（" << kReproImages << " 张）：\n"
              << "[Int8Probe]   整体：PT " << pt_agree << "/" << kReproImages
              << "  PC " << pc_agree << "/" << kReproImages << "\n"
              << "[Int8Probe]   余量子集(margin≥" << kConfidentMargin << ")：PT "
              << pt_confident_agree << "/" << confident << " = " << pt_rate * 100.0
              << "%   PC " << pc_confident_agree << "/" << confident << " = "
              << pc_rate * 100.0 << "%\n"
              << "[Int8Probe]   正式产物口径（TROUBLESHOOTING #29.5）：PT 100% / PC 54.5%\n";

    ASSERT_GT(confident, 0) << "没有任何有余量的样本，这批图不适合做复现对照";
    EXPECT_LT(pc_rate, pt_rate)
        << "探针图下 per-channel 不再更差 —— 仪器改变了被观测的现象（开发计划 §13.3 D6），"
           "本轮结论作废，下一轮退到'单点探针'";
}

}  // namespace mini_trt_llm
