#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace mini_trt_llm {

enum class LogLevel { kVERBOSE, kINFO, kWARN, kERROR };

inline LogLevel& GlobalLogLevel() {
    static LogLevel level = LogLevel::kINFO;
    return level;
}

inline void SetLogLevel(LogLevel level) { GlobalLogLevel() = level; }

inline const char* LogLevelPrefix(LogLevel level) {
    switch (level) {
        case LogLevel::kVERBOSE:
            return "[VERBOSE] ";
        case LogLevel::kINFO:
            return "[INFO]    ";
        case LogLevel::kWARN:
            return "[WARN]    ";
        case LogLevel::kERROR:
            return "[ERROR]   ";
    }
    return "[UNKNOWN] ";
}

inline void LogMessage(LogLevel level, const std::string& msg) {
    if (level < GlobalLogLevel()) {
        return;
    }
    std::ostream& os = (level >= LogLevel::kWARN) ? std::cerr : std::cout;
    os << LogLevelPrefix(level) << msg << "\n";
}

#define MINI_TRT_LOG_LEVEL(level)                                              \
    do {                                                                       \
        if ((level) >= ::mini_trt_llm::GlobalLogLevel()) {                     \
            std::ostringstream __oss;                                          \
            __oss << __FILE__ << ":" << __LINE__ << " ";                       \
            __oss

#define MINI_TRT_LOG_STREAM(level)                                             \
    ::mini_trt_llm::LogMessage((level), __oss.str());                          \
        }                                                                      \
    } while (0)

#define MINI_TRT_LOG_VERBOSE(msg)                                              \
    do {                                                                       \
        std::ostringstream _oss;                                               \
        _oss << msg;                                                           \
        ::mini_trt_llm::LogMessage(::mini_trt_llm::LogLevel::kVERBOSE,         \
                                   _oss.str());                                \
    } while (0)

#define MINI_TRT_LOG_INFO(msg)                                                 \
    do {                                                                       \
        std::ostringstream _oss;                                               \
        _oss << msg;                                                           \
        ::mini_trt_llm::LogMessage(::mini_trt_llm::LogLevel::kINFO, _oss.str()); \
    } while (0)

#define MINI_TRT_LOG_WARN(msg)                                                 \
    do {                                                                       \
        std::ostringstream _oss;                                               \
        _oss << msg;                                                           \
        ::mini_trt_llm::LogMessage(::mini_trt_llm::LogLevel::kWARN, _oss.str()); \
    } while (0)

#define MINI_TRT_LOG_ERROR(msg)                                                \
    do {                                                                       \
        std::ostringstream _oss;                                               \
        _oss << msg;                                                           \
        ::mini_trt_llm::LogMessage(::mini_trt_llm::LogLevel::kERROR,           \
                                   _oss.str());                                \
    } while (0)

}  // namespace mini_trt_llm
