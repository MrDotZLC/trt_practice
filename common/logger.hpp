#pragma once
#include <NvInfer.h>
#include <iostream>

/**
 * TensorRT Logger（日志接口）
 *
 * 背景：
 *   TRT 所有内部消息（构建进度、警告、错误）通过 ILogger 接口回调输出。
 *   必须提供一个 ILogger 实现，并在创建 Builder / Runtime 时传入。
 *
 * Severity 级别（由高到低）：
 *   kINTERNAL_ERROR  TRT 内部错误，通常不可恢复
 *   kERROR           可恢复错误
 *   kWARNING         警告
 *   kINFO            构建进度等信息
 *   kVERBOSE         详细调试信息（非常多）
 *
 * 此处过滤掉 kINFO 和 kVERBOSE，只打印 WARNING 及以上。
 * 调试构建问题时可改为 <= kINFO。
 */

class Logger : public nvinfer1::ILogger {
public:
    // 默认warning级别
    explicit Logger(Severity minSeverity = Severity::kWARNING)
        : mMinSeverity(minSeverity) {}
    
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > mMinSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "[TRT INTERNAL_ERROR] "; break;
        case Severity::kERROR:          std::cerr << "[TRT ERROR]          "; break;
        case Severity::kWARNING:        std::cerr << "[TRT WARNING]        "; break;
        case Severity::kINFO:           std::cout << "[TRT INFO]           "; break;
        case Severity::kVERBOSE:        std::cout << "[TRT VERBOSE]        "; break;
        }

        std::cout << msg << "\n";
    }
    
private:
    Severity mMinSeverity;
};