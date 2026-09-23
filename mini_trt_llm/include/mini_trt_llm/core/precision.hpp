#pragma once

#include <NvInfer.h>
#include <string>

namespace mini_trt_llm {

// 推理精度枚举，与 TensorRT DataType 一一对应。
// Phase 0 仅支持 FP32 / FP16 / INT8，后续可扩展。
enum class Precision { FP32, FP16, INT8 };

// 返回精度的简短字符串表示，主要用于日志输出。
const char* PrecisionStr(Precision p);

// 返回精度的可读字符串，与 PrecisionStr 等价。
std::string PrecisionString(Precision p);

// 将项目内部 Precision 映射为 TensorRT 的 DataType。
nvinfer1::DataType ToTrtDataType(Precision p);

}  // namespace mini_trt_llm
