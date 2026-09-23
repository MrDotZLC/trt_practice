#include "mini_trt_llm/core/precision.hpp"

namespace mini_trt_llm {

const char* PrecisionStr(Precision p) {
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

std::string PrecisionString(Precision p) { return PrecisionStr(p); }

nvinfer1::DataType ToTrtDataType(Precision p) {
    switch (p) {
        case Precision::FP32:
            return nvinfer1::DataType::kFLOAT;
        case Precision::FP16:
            return nvinfer1::DataType::kHALF;
        case Precision::INT8:
            return nvinfer1::DataType::kINT8;
    }
    return nvinfer1::DataType::kFLOAT;
}

}  // namespace mini_trt_llm
