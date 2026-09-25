#pragma once

// CV（Phase 4）用例的共享测试件：路径查找、FP32 张量读写、ramp 公式与 ImageNet 归一化的
// **测试侧唯一来源**。
//
// 为什么必须唯一：`MakeRampInput` 与 `NormalizePixels` 在 **Python 侧也有一份**
// （`scripts/ref_resnet18.py`，用来生成基线）。两边一旦漂移，"引擎与基线对不上"就会
// 伪装成实现 bug——而它们本该是同一套公式。`ResNet18BaselineTest` 里的两条 host 用例
// 就是这条纪律的 meta-test：拿 C++ 的实现去比 Python 落盘的产物。
// 见 docs/PROGRESS.md §2.13（参考实现必须唯一、必须有 meta-test）。

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// ImageNet 归一化参数：与 `0_resnet18_onnx/prepare_calib_data.py` 和
// `scripts/ref_resnet18.py` 必须逐值一致。
inline constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
inline constexpr float kImageNetStd[3] = {0.229f, 0.224f, 0.225f};
inline constexpr int32_t kCvChannels = 3;
inline constexpr int32_t kCvSize = 224;
inline constexpr int32_t kCvClasses = 1000;

inline std::string FindFile(const std::vector<std::string>& candidates) {
    for (const std::string& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

// P4-1 基线产物目录（`models/resnet18/`）；找不到返回空串。
inline std::string FindBaselineDir() {
    const std::string found = FindFile({"models/resnet18/ref_ramp_b8.bin",
                                        "../models/resnet18/ref_ramp_b8.bin",
                                        "../../models/resnet18/ref_ramp_b8.bin",
                                        "../../../models/resnet18/ref_ramp_b8.bin"});
    if (found.empty()) {
        return {};
    }
    return std::filesystem::path(found).parent_path().string();
}

// 读取 P4-1 落盘的 FP32 raw 张量；元素数不符返回空。
inline std::vector<float> ReadF32File(const std::string& path, size_t expected_elements) {
    std::vector<float> data(expected_elements);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(expected_elements * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(expected_elements * sizeof(float))) {
        return {};
    }
    return data;
}

// 与 P4-1 脚本、以及历史工程 `src/main.cpp` 同式的合成 ramp：扁平下标取模，**不归一化**。
//
// 用 float32 除法（而不是先算 double 再转换）以对齐 C++ 的 `float(k) / 255.f`；
// 否则输入本身就会差 1 ulp，对拍时无法区分"输入不同"与"实现不同"。
inline std::vector<float> MakeRampInput(int32_t batch) {
    std::vector<float> data(static_cast<size_t>(batch) * kCvChannels * kCvSize * kCvSize);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 255) / 255.0f;
    }
    return data;
}

// `[0,255]` 像素质（NCHW float32）→ ImageNet 归一化。这是 `CVRunner` 预处理契约的
// 测试侧参考实现；`scripts/ref_resnet18.py` 里有同一公式的 Python 版。
inline std::vector<float> NormalizePixels(const std::vector<float>& pixels) {
    std::vector<float> normalized(pixels.size());
    const size_t plane = static_cast<size_t>(kCvSize) * kCvSize;
    for (size_t i = 0; i < pixels.size(); ++i) {
        const size_t channel = (i / plane) % kCvChannels;
        normalized[i] = (pixels[i] / 255.0f - kImageNetMean[channel]) / kImageNetStd[channel];
    }
    return normalized;
}

// 某一行的 argmax（logits 按 [batch, classes] 行主序）。
inline int32_t ArgmaxOfRow(const std::vector<float>& logits, int32_t row) {
    const size_t begin = static_cast<size_t>(row) * kCvClasses;
    size_t best = begin;
    for (size_t i = begin + 1; i < begin + kCvClasses; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best - begin);
}

}  // namespace test_support
}  // namespace mini_trt_llm
