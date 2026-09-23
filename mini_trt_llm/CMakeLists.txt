cmake_minimum_required(VERSION 3.18)
project(mini_trt_llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75) # 本地开发是 SM75
endif()

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build example applications" OFF)

# ---------------------------------------------------------------------------
# Dependencies discovery
# ---------------------------------------------------------------------------
find_path(TRT_INCLUDE_DIR NvInfer.h # TensorRT 的核心头文件
    PATHS /usr/include/x86_64-linux-gnu /usr/local/include
    REQUIRED
)

find_path(CUDA_INCLUDE_DIR cuda_runtime.h # CUDA 的核心头文件
    PATHS /usr/local/cuda/include
    REQUIRED
)

find_library(TRT_LIB     nvinfer        PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(TRT_PLUGIN  nvinfer_plugin PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(TRT_ONNX    nvonnxparser   PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(CUDA_RT_LIB cudart         PATHS /usr/local/cuda/lib64     REQUIRED)

# ---------------------------------------------------------------------------
# Third-party: SentencePiece
# ---------------------------------------------------------------------------
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_NODEJS_BINDINGS OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_PERL_BINDINGS OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_RUBY_BINDINGS OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_JAVA_BINDINGS OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/sentencepiece)

# ---------------------------------------------------------------------------
# Third-party: safetensors-cpp
# ---------------------------------------------------------------------------
set(SAFETENSORS_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SAFETENSORS_CPP_BUILD_C_API OFF CACHE BOOL "" FORCE)
# 启用异常，避免 -fno-exceptions 传播到本项目
set(SAFETENSORS_CPP_CXX_EXCEPTIONS ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/safetensors-cpp)

# ---------------------------------------------------------------------------
# Third-party: GoogleTest (tests only)
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/googletest)
endif()

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece/src
)

# ---------------------------------------------------------------------------
# Collect sources
# ---------------------------------------------------------------------------
file(GLOB_RECURSE MINI_TRT_LLM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------
add_library(mini_trt_llm STATIC ${MINI_TRT_LLM_SRCS})

target_include_directories(mini_trt_llm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece/src
)

target_link_libraries(mini_trt_llm
    ${TRT_LIB}
    ${TRT_PLUGIN}
    ${TRT_ONNX}
    ${CUDA_RT_LIB}
    sentencepiece
    safetensors_cpp
)

target_compile_options(mini_trt_llm PRIVATE
    -Wno-deprecated-declarations
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(BUILD_EXAMPLES)
    # TODO: add examples in future iterations
endif()
