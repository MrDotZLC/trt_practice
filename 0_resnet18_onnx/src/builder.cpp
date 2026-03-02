#include "builder.hpp"
#include "calibrator.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

void buildEngine(const std::string &onnxPath, const std::string &enginePath,
                 Precision precision, Logger &logger, size_t workspaceBytes) {
    std::cout << "\n[Builder] onnx=" << onnxPath
              << "  precision=" << precisionStr(precision) << "\n";
    // ── 1. 创建 Builder
    // ───────────────────────────────────────────────────────
    //
    // IBuilder 是构建阶段的总入口。
    // 所有构建对象（network、config、profile）都由它创建。
    // createInferBuilder 需要传入 logger，TRT 内部消息通过它回调输出。
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw std::runtime_error("[Builder] createInferBuilder failed");
    }

    // ── 2. 创建网络定义
    // ───────────────────────────────────────────────────────
    //
    // INetworkDefinition 保存从 ONNX 解析出的网络结构（层、张量、连接关系）。
    //
    // kEXPLICIT_BATCH：
    //   显式 batch 模式，batch 维度作为正常维度参与 shape 推导。
    //   TRT 10.x 只支持此模式，旧版 kIMPLICIT_BATCH 已废弃。

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0U));
    if (!network) {
        throw std::runtime_error("[Builder] createNetworkV2 failed");
    }

    // ── 3. 解析 ONNX
    // ──────────────────────────────────────────────────────────
    //
    // IParser 读取 ONNX 文件，将其中的算子逐一转换为 TRT 的 ILayer，
    // 填充到 network 中。
    // 支持的 ONNX opset 版本取决于 TRT 版本，TRT 10.x 支持到 opset 20。
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    if (!parser) {
        throw std::runtime_error("[Builder] createParser failed");
    }
    if (!parser->parseFromFile(
            onnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "[Builder] Parse error: "
                      << parser->getError(i)->desc() << "\n";
        }
        throw std::runtime_error("[Builder] ONNX parse failed");
    }
    std::cout << "[Builder] ONNX parsed OK\n";
    std::cout << "[Builder] Network inputs : " << network->getNbInputs()
              << "\n";
    std::cout << "[Builder] Network outputs: " << network->getNbOutputs()
              << "\n";

    // ── 4. 构建配置
    // ───────────────────────────────────────────────────────────
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("[Builder] createBuilderConfig failed");
    }
    // Workspace（工作区）：
    //   TRT kernel auto-tuning 时需要临时显存来测试候选 kernel。
    //   设置过小会导致部分 kernel 无法测试，影响性能选择。
    //   1 GiB 对 ResNet18 足够。
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1UL << 30);

    // ── 5. 动态 Shape Profile
    // ─────────────────────────────────────────────────
    //
    // Profile 告知 TRT 输入 tensor 的形状范围：
    //   kMIN：最小 batch，TRT 保证此 shape 可正确运行
    //   kOPT：最优 batch，TRT 针对此 shape 选择最快 kernel（最重要）
    //   kMAX：最大 batch，TRT 保证此 shape 可正确运行
    //
    // 推理时 batch 大小必须在 [kMIN, kMAX] 范围内，
    // 越接近 kOPT 性能越好。
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4{8, 3, 224, 224});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4{16, 3, 224, 224});
    config->addOptimizationProfile(profile);

    // ── 6. 精度设置
    // ───────────────────────────────────────────────────────────
    std::unique_ptr<Int8Calibrator> calibrator;
    switch (precision) {
        case Precision::FP16:
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            break;

        case Precision::INT8:
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);  // fallback
            calibrator = std::make_unique<Int8Calibrator>(
                8, 3, 224, 224, "calib_cache.bin",
                std::string(PROJECT_SOURCE_DIR) + "/calib_data");
            config->setInt8Calibrator(calibrator.get());
            break;

        case Precision::FP32:
        default:
            break;
    }

    // ── 7. 构建并序列化 Engine
    // ────────────────────────────────────────────────
    //
    // buildSerializedNetwork 是最耗时的步骤，内部执行：
    //   a. 图优化（层融合、常量折叠等）
    //   b. 对每个算子枚举 CUDA kernel 候选，实际运行计时，选最快的
    //   c. INT8 模式下调用 calibrator 执行校准
    //
    // 返回 IHostMemory：序列化后的二进制 engine 数据（在 CPU 内存中）
    std::cout << "[Builder] Building engine, please wait...\n";

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        throw std::runtime_error("[Builder] buildSerializedNetwork failed");
    }

    // ── 8. 写入文件
    // ───────────────────────────────────────────────────────────
    std::ofstream fout(enginePath, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("[Builder] Cannot write: " + enginePath);
    }

    fout.write(static_cast<const char *>(serialized->data()),
               serialized->size());
    std::cout << "[Builder] Engine saved: " << enginePath
              << "  size=" << serialized->size() / 1024 / 1024 << " MB\n";
}
