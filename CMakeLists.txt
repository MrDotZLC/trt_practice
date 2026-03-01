cmake_minimum_required(VERSION 3.18)
project(trt_practice LANGUAGES CXX)

# 公共头文件路径
include_directories(${CMAKE_SOURCE_DIR}/common)

add_subdirectory(resnet18)
