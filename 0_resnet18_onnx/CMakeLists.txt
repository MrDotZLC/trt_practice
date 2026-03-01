cmake_minimum_required(VERSION 3.18)
project(trt_practice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/local/include
    REQUIRED
)

find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    PATHS /usr/local/cuda/include
    REQUIRED
)

include_directories(
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_library(TRT_LIB     nvinfer        PATHS /usr/lib/x86_64-linux-gnu  REQUIRED)
find_library(TRT_PLUGIN  nvinfer_plugin PATHS /usr/lib/x86_64-linux-gnu  REQUIRED)
find_library(TRT_ONNX    nvonnxparser   PATHS /usr/lib/x86_64-linux-gnu  REQUIRED)
find_library(CUDA_RT_LIB cudart         PATHS /usr/local/cuda/lib64      REQUIRED)

file(GLOB SRCS src/*.cpp)
add_executable(trt_resnet18 ${SRCS})

target_link_libraries(trt_resnet18
    ${TRT_LIB}
    ${TRT_PLUGIN}
    ${TRT_ONNX}
    ${CUDA_RT_LIB}
)

# 将源码目录路径编译进可执行文件，避免路径硬编码
target_compile_definitions(trt_resnet18 PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(trt_resnet18 PRIVATE -Wno-deprecated-declarations)