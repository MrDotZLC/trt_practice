cmake_minimum_required(VERSION 3.18)
project(gpt2_trt LANGUAGES CXX)             # ← 改名

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

find_library(TRT_LIB     nvinfer        PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(TRT_PLUGIN  nvinfer_plugin PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(TRT_ONNX    nvonnxparser   PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(CUDA_RT_LIB cudart         PATHS /usr/local/cuda/lib64     REQUIRED)

file(GLOB SRCS src/*.cpp)
add_executable(trt_gpt2 ${SRCS})             # ← 改名

target_link_libraries(trt_gpt2
    ${TRT_LIB}
    ${TRT_PLUGIN}
    ${TRT_ONNX}
    ${CUDA_RT_LIB}
)

target_compile_definitions(trt_gpt2 PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(trt_gpt2 PRIVATE -Wno-deprecated-declarations)

# nsys profile target
add_custom_target(profile_gpt2
    COMMAND nsys profile
        --output ${CMAKE_CURRENT_BINARY_DIR}/nsys_gpt2
        --trace cuda,nvtx,osrt,cudnn,cublas
        --sample=none
        --force-overwrite=true
        $<TARGET_FILE:trt_gpt2>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS trt_gpt2
    COMMENT "Running nsys profile (GPT-2)..."
)