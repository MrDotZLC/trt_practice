#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>

/**
 * INT8 校准器（INT8 Calibrator）
 *
 * 背景：
 *   FP32 有效范围约 ±3.4e38，INT8 只有 [-128, 127]。
 *   量化（Quantization）就是把 FP32 激活值线性映射到 INT8：
 *
 *       x_int8 = clamp( round(x_fp32 / scale), -128, 127 )
 *
 *   scale 的确定需要在真实数据上统计每一层激活值的分布，
 *   这个过程叫做校准（Calibration）。
 *
 * 校准器类型：
 *   TRT 提供 4 种校准器，区别在于 scale 的计算方式：
 *
 *   IInt8MinMaxCalibrator      取激活值的 [min, max]
 * 作为范围，简单但可能有较大误差 IInt8EntropyCalibrator     最小化量化前后的 KL
 * 散度，精度较好（旧版默认） IInt8EntropyCalibrator2    EntropyCalibrator
 * 的改进版，当前推荐 IInt8LegacyCalibrator      旧版兼容，不推荐
 *
 *   本实现继承 IInt8EntropyCalibrator2。
 *
 * 校准流程：
 *   1. TRT 调用 getBatch() 获取一批数据（在 GPU 上）
 *   2. TRT 对该批数据执行前向传播，收集每层激活值分布
 *   3. 重复直到 getBatch() 返回 false
 *   4. TRT 根据统计结果计算每层的 scale，写入 cache
 *
 * 生产环境：
 *   getBatch() 中替换为真实 ImageNet 数据，建议 ≥ 500 张。
 *   此处用随机数据仅验证流程，INT8 精度无参考价值。
 */

class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    /**
     * @param batch_size  每次校准的 batch 大小
     * @param channels   输入通道数
     * @param height     输入高度
     * @param width      输入宽度
     * @param cache_file  校准缓存文件路径，存在则跳过校准直接读取
     */
    Int8Calibrator(int batch_size, int channels, int height, int width,
                   const std::string &cache_file);
    ~Int8Calibrator();
    
    // -- TRT 回调接口--
    int getBatchSize() const noexcept override;

    /**
     * 将下一批数据的 GPU 指针写入 bindings[]。
     * @param bindings  输出数组，bindings[i] = 第 i 个输入 tensor 的 GPU 指针
     * @param names     各输入 tensor 的名称
     * @param nb_bindings tensor 数量
     * @return true 表示有数据；false 表示数据耗尽，校准结束
     */
    bool getBatch(void* bindings[], const char* names[],
                  int nbBindings) noexcept override;

    // 读取已有 cache，返回 nullptr 表示无 cache，TRT 将重新校准
    const void* readCalibrationCache(size_t& length) noexcept override;

    // TRT 校准完成后调用，将 scale 数据写入 cache 文件
    void writeCalibrationCache(const void* cache,
                               std::size_t length) noexcept override;
private:
    int m_batch_size;               // 每批的样本数量
    int m_input_size;               // 每批输入的元素数量（batch_size * channels * height * width）
    int m_current_batch = 0;        // 当前校准批次索引
    int m_total_batchs = 10;        // 校准轮数，生产环境：ceil(500 / batchSize)

    void*               m_device_input = nullptr; 	// GPU 端输入缓冲区
    std::vector<float>  m_host_input;               // CPU 端数据
	
	std::string         m_cache_file;
    std::vector<char>   m_calibration_cache;       // 读入的 cache 数据

};    