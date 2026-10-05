#pragma once

#include "mini_trt_llm/utils/json.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mini_trt_llm {

// 一张权重的量化条目，与 `tools/convert/quantize_gpt2.py` 产出的 `quant_int8.json`
// 的 entries[] 一一对应。
struct QuantEntry {
    std::string tensor;         // 建图侧的名字（与 weight_map 的 key 同源）
    std::string source_key;     // 源 safetensors 里的 key，也是 int8 产物里的 key
    std::string granularity;    // 本轮只允许 "per_tensor"（见 design.md 的 D2）
    int32_t axis = -1;          // 仅 per_channel 有意义；本轮恒为 -1
    std::vector<float> scales;  // per_tensor 恒为 1 个元素

    // 本轮的对称量化固定零点是 0，所以反量化就是 y = q * scale。
    float scale() const { return scales.empty() ? 1.0f : scales.front(); }
};

// 量化清单：把"哪些张量量化 + 各自的 scale"固化下来，建图侧只做查表。
//
// 为什么清单要独立于 `config.json`：清单是**脚本产出的产物身份**（含来源与 int8 权重的
// 摘要），而 config.json 描述的是模型结构。分开之后，"换了 scale 但没换模型"这类变化
// 才能被引擎指纹捕捉到（见 design.md 的 D5）。
class QuantSpec {
 public:
    // 从 quant_int8.json 载入。
    //
    // 失败时记录原因并返回 false，**不做部分载入**——半份清单会让建图侧静默退回 FP32
    // 权重，而那正是本 feature 要防的"声明了量化、其实没量化"。
    static bool LoadFromFile(const std::string& path, QuantSpec* out);

    // 按建图侧名字或 source_key 命中。
    //
    // 两个方向都收是刻意的：建图侧只知道 TRT 名字，而 int8 产物里的 key 是源名字，
    // 让调用方自己去查 weight_map 会把"映射写反"的风险散到每个调用点（见
    // `docs/dev/REQ-017-llm-int8-quant/analysis.md` 的 Existing Limitation 第 6 条）。
    const QuantEntry* Find(const std::string& name) const;

    const std::vector<QuantEntry>& entries() const { return entries_; }
    // 已按清单所在目录解析成可直接打开的路径。
    const std::string& int8_weights_path() const { return int8_weights_path_; }
    const std::string& weights_source_path() const { return weights_source_path_; }

 private:
    std::vector<QuantEntry> entries_;
    std::map<std::string, size_t> by_tensor_;
    std::map<std::string, size_t> by_source_key_;
    std::string int8_weights_path_;
    std::string weights_source_path_;
};

}  // namespace mini_trt_llm
