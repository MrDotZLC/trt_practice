#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <stdexcept>
#include <sstream>
#include <string>

namespace mini_trt_llm {

inline void CudaCheck(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << ": " << cudaGetErrorString(code)
            << " (" << cudaGetErrorName(code) << ")";
        throw std::runtime_error(oss.str());
    }
}

inline void CudaCheckLast(const char* file, int line) {
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA last error at " << file << ":" << line
            << ": " << cudaGetErrorString(code)
            << " (" << cudaGetErrorName(code) << ")";
        throw std::runtime_error(oss.str());
    }
}

inline void NvInferCheck(bool condition, const char* msg,
                         const char* file, int line) {
    if (!condition) {
        std::ostringstream oss;
        oss << "TensorRT error at " << file << ":" << line << ": " << msg;
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(call) ::mini_trt_llm::CudaCheck((call), __FILE__, __LINE__)
#define CUDA_CHECK_LAST() ::mini_trt_llm::CudaCheckLast(__FILE__, __LINE__)
#define NVINFER_CHECK(call)                                                    \
    ::mini_trt_llm::NvInferCheck((call) != nullptr, #call, __FILE__, __LINE__)
#define NVINFER_ASSERT(cond)                                                   \
    ::mini_trt_llm::NvInferCheck((cond), #cond, __FILE__, __LINE__)

}  // namespace mini_trt_llm
