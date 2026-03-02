#include "calibrator.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// 构造
Int8Calibrator::Int8Calibrator(int batch_size, int channels, int height,
                               int width, const std::string &cache_file,
                               const std::string& calib_data_dir) 
    : m_batch_size(batch_size),
      m_input_size(batch_size * channels * height * width),
      m_cache_file(cache_file),
      m_calib_data_dir(calib_data_dir) {
    
    // 扫描目录，收集所有 .bin 文件路径，排序保证可复现
    for (auto& entry : fs::directory_iterator(calib_data_dir)) {
        if (entry.path().extension() == ".bin") {
            m_file_list.push_back(entry.path().string());
        }
    }
    std::sort(m_file_list.begin(), m_file_list.end());

    // 每批 batch_size 张，计算总批数
    m_total_batchs = static_cast<int>(m_file_list.size()) / batch_size;

    // 分配 CPU 端数据缓冲区
    m_host_input.resize(m_input_size);

    // 分配 GPU 端数据缓冲区
    cudaMalloc(&m_device_input, m_input_size * sizeof(float));
}

// 析构
Int8Calibrator::~Int8Calibrator() {
    // 释放 GPU 显存，与 cudaMalloc 配对
    if (m_device_input) {
        cudaFree(m_device_input);
    }
}

int Int8Calibrator::getBatchSize() const noexcept {
    return m_batch_size;
}

bool Int8Calibrator::getBatch(void *bindings[], const char *names[],
                             int nbBindings) noexcept {
    // 数据耗尽，通知 TRT 校准结束
    if (m_current_batch >= m_total_batchs) {
        return false;
    }

    // // 生成随机数据
    // // 生产环境：此处替换为从磁盘加载真实图片，并做归一化预处理：
    // //   x = (pixel / 255.0 - mean) / std
    // //   mean = [0.485, 0.456, 0.406]
    // //   std  = [0.229, 0.224, 0.225]
    // std::mt19937 rng(m_current_batch);  // 用 batch 序号做种子，保证可复现
    // std::normal_distribution<float> dist(0.f, 1.f);  // 标准正态分布
    // for (auto &v : m_host_input) {
    //     v = dist(rng);
    // }

    // 每次读取 batch_size 个文件，拼成一个 batch
    int single_size = m_input_size / m_batch_size;  // 单张图片元素数
    for (int i = 0; i < m_batch_size; ++i) {
        int file_idx = m_current_batch * m_batch_size + i;
        if (file_idx >= static_cast<int>(m_file_list.size())) break;

        std::ifstream fin(m_file_list[file_idx], std::ios::binary);
        if (!fin) {
            std::cerr << "[Calibrator] Cannot open: "
                      << m_file_list[file_idx] << "\n";
            return false;
        }
        // 读取单张图片数据到 host_input 的对应位置
        fin.read(reinterpret_cast<char*>(m_host_input.data() + i * single_size),
                 single_size * sizeof(float));
    }

    // CPU → GPU 拷贝（同步）
    // cudaMemcpy(dst, src, bytes, direction)
    cudaMemcpy(m_device_input, m_host_input.data(),
               m_input_size * sizeof(float), cudaMemcpyHostToDevice);

    // 将 GPU 指针写入 bindings[0]（ResNet18 只有一个输入）
    bindings[0] = m_device_input;

    std::cout << "[Calibrator] batch " << m_current_batch + 1 << " / "
              << m_total_batchs << "\n";

    ++m_current_batch;
    return true;
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept {
    m_calibration_cache.clear();

    std::ifstream fin(m_cache_file, std::ios::binary);
    
    if (!fin) {
        length = 0;
        return nullptr;
    }

    m_calibration_cache.assign(std::istreambuf_iterator<char>(fin),
                               std::istreambuf_iterator<char>());
    length = m_calibration_cache.size();

    std::cout << "[Calibrator] Loaded cache: " << m_cache_file
              << " (" << length << " bytes), skipping calibration.\n";

    return m_calibration_cache.data();
}

void Int8Calibrator::writeCalibrationCache(const void* cache,
                               std::size_t length) noexcept {
    std::ofstream fout(m_cache_file, std::ios::binary);
    fout.write(static_cast<const char*>(cache), length);
    
    std::cout << "[Calibrator] Written cache: " << m_cache_file
              << " (" << length << " bytes).\n";
}
