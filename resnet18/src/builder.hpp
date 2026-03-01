#pragma once
#include <NvInfer.h>
#include <string>
#include "logger.hpp"

/**
 * 推理精度枚举
 *
 * FP32：全精度浮点，基准精度，无损失
 * FP16：半精度浮点，GTX 1660 Ti 无 FP16 Tensor Core，加速不明显
 * INT8：8位整数，GTX 1660 Ti 有 INT8 Tensor Core，有实质加速，精度略有损失
 */
enum class Precision { FP32, FP16, INT8 };

/**
 * 将 Precision 枚举转为字符串，用于日志输出
 */

inline const char *precisionStr(Precision p) {
    switch (p) {
        case Precision::FP32:
            return "FP32";
        case Precision::FP16:
            return "FP16";
        case Precision::INT8:
            return "INT8";
    }
    return "UNKNOWN";
}

/**
 * 从 ONNX 文件构建 TensorRT Engine 并序列化到磁盘
 *
 * 构建过程（发生在此函数内部）：
 *   1. IBuilder       解析 ONNX，创建网络定义
 *   2. IBuilderConfig 设置精度、workspace、动态 shape profile
 *   3. buildSerializedNetwork()
 *        ├── Layer fusion（算子融合）：如 Conv+BN+ReLU 合并为一个 kernel
 *        ├── Kernel auto-tuning：对每个算子枚举候选 CUDA kernel，选最快的
 *        └── INT8 模式下执行 Calibration，确定每层 scale
 *   4. 序列化为二进制写入磁盘（.engine 文件）
 *
 * 注意：
 *   - engine 与 GPU 架构绑定（SM 7.5 编译的不能在 SM 8.0 运行）
 *   - 构建耗时较长（数秒到数分钟），推理时直接加载 .engine 文件
 *
 * @param onnxPath    输入 ONNX 模型路径
 * @param enginePath  输出 engine 文件路径
 * @param precision   推理精度
 * @param logger      TRT Logger 实例（Builder 和 Runtime 共用）
 */

void buildEngine(const std::string& onnxPath,
                 const std::string& enginePath,
                 Precision precision,
                 Logger& logger);