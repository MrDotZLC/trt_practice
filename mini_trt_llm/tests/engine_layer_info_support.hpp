#pragma once

// 引擎逐层信息（Engine Inspector 的 ONELINE）的**共用读取器**。
//
// 为什么提成共享头：同一段"遍历 getNbLayers / getLayerInformation(kONELINE)、统计 Int8 张量与
// tactic 名、可选逐层落盘"的逻辑此前有三份实现（`test_gpt2_int8_weights.cpp` 的 `CountInt8Layers`、
// `test_resnet18_int8_probe.cpp` 的 `InspectTactics/TacticSummary`、以及 2026-10-06 的 GPT-2 FP16
// 诊断用例）。`test_gpt2_int8_weights.cpp` 的注释早就写着"若将来出现第三个使用方，再把这份统计提到
// 共享头，避免两份实现各自演化"——第三个使用方出现后按那条约定收到这里。
//
// **为什么要落盘**（沿自 `test_resnet18_int8_probe.cpp` 的实测结论）：探针图多了若干图输出，
// **会改变 TRT 的融合与 tactic 选择**（实测：产物图有 `i8i8` tactic，探针图上这个计数变成 0）。
// 所以不能只看一个计数——把每层的 ONELINE 原文写下来，才能判断"变的是哪几层、变成了什么"。
//
// **读数口径（勿照直觉改）**：逐层信息里**没有 `[I8]` 这种标签**；判"这一层确实跑了 Int8"
// 要读 `Format/Datatype: Int8`。按标签判会把"确实跑了"误判成"没跑"。

#include "mini_trt_llm/core/engine.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

struct LayerInfo {
    int32_t layers = 0;
    int32_t int8_tensors = 0;                   // 含 `Format/Datatype: Int8` 的层数
    int32_t i8i8_tactics = 0;                   // tactic 名含 `i8i8` 的层数（INT8 专用；其它精度下恒 0）
    std::vector<std::string> distinct_tactics;  // 出现过的 TacticName（去重、保持首次出现序）
    std::string dump_path;                      // 落盘成功时非空
};

// 逐层读一遍。`dump_path` 非空时把 `序号 \t ONELINE` 落盘。
// 引擎 / inspector 为空时返回全零（`layers = 0`）——调用方用 `layers` 判"读到了没有"，别用计数判。
inline LayerInfo InspectLayerInfo(Engine* engine,
                                 const std::string& dump_path = std::string()) {
    LayerInfo info;
    if (engine == nullptr) {
        return info;
    }
    nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
    if (cuda == nullptr) {
        return info;
    }
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(cuda->createEngineInspector());
    if (inspector == nullptr) {
        return info;
    }
    std::ofstream dump(dump_path);
    info.layers = cuda->getNbLayers();
    for (int32_t i = 0; i < info.layers; ++i) {
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
            ++info.int8_tensors;
        }
        if (text.find("i8i8") != std::string::npos) {
            ++info.i8i8_tactics;
        }
        // TacticName 的原文（供人核对"两臂到底换没换 kernel"）。
        const size_t at = text.find("TacticName: ");
        if (at != std::string::npos) {
            const size_t begin = at + std::string("TacticName: ").size();
            const size_t end = text.find(',', begin);
            const std::string name = text.substr(begin, end - begin);
            if (std::find(info.distinct_tactics.begin(), info.distinct_tactics.end(), name) ==
                info.distinct_tactics.end()) {
                info.distinct_tactics.push_back(name);
            }
        }
    }
    if (dump) {
        info.dump_path = dump_path;
    }
    return info;
}

// 把 distinct_tactics 拼成一行；超过 limit 个时省略并给出总数。
inline std::string TacticSummary(const LayerInfo& info, size_t limit) {
    std::string text;
    for (size_t i = 0; i < info.distinct_tactics.size() && i < limit; ++i) {
        text += (i == 0 ? "" : " | ");
        text += info.distinct_tactics[i];
    }
    if (info.distinct_tactics.size() > limit) {
        text += " | …(共 " + std::to_string(info.distinct_tactics.size()) + " 种)";
    }
    if (text.empty()) {
        text = "(没有 TacticName —— 逐层信息里读不到 tactic)";
    }
    return text;
}

}  // namespace test_support
}  // namespace mini_trt_llm
