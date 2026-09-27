cmake_minimum_required(VERSION 3.18)
project(trt_practice LANGUAGES CXX)

enable_testing()

# 公共头文件路径
include_directories(${CMAKE_SOURCE_DIR}/common)

# 新框架：mini_trt_llm
add_subdirectory(mini_trt_llm)
