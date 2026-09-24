#include "e2e_safetensors_writer.hpp"
#include "mini_trt_llm/utils/safetensors_loader.hpp"

#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cstring>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

// 全部取在 BF16 与 FP16 下都能精确表示的数，这样转换结果可以逐位比较，
// 不需要引入精度容差——容差会掩盖"指数位处理错"这类问题。
const std::vector<float> kExactValues{0.0f,  0.5f,  -1.5f,  2.25f,
                                      -0.75f, 3.0f,  100.0f, -0.125f};

std::string TempPath(const std::string& tag) {
    return "/tmp/mini_trt_llm_st_" + tag + ".safetensors";
}

// 写一个单张量文件并加载，返回 loader（调用方需保证文件先落盘）。
bool WriteSingle(const std::string& path, test_support::TensorSpec::Dtype dtype,
                 const std::vector<float>& values) {
    std::map<std::string, test_support::TensorSpec> tensors;
    tensors["w"] = test_support::TensorSpec{{values.size()}, dtype, values};
    return test_support::WriteSafetensorsFile(path, tensors);
}

std::vector<float> ReadAsFloats(const void* data, size_t count) {
    const auto* floats = static_cast<const float*>(data);
    return std::vector<float>(floats, floats + count);
}

}  // namespace

TEST(SafetensorsLoaderTest, LoadNonExistentReturnsFalse) {
    SafetensorsLoader loader;
    EXPECT_FALSE(loader.LoadFromFile("/tmp/non_existent.safetensors"));
}

TEST(SafetensorsLoaderTest, WritesAndReadsBackFp32) {
    const std::string path = TempPath("f32_roundtrip");
    ASSERT_TRUE(WriteSingle(path, test_support::TensorSpec::Dtype::kF32, kExactValues));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));
    EXPECT_TRUE(loader.HasTensor("w"));

    safetensors::dtype dtype = safetensors::kFLOAT32;
    std::vector<size_t> shape;
    ASSERT_TRUE(loader.GetTensorInfo("w", &dtype, &shape));
    EXPECT_EQ(dtype, safetensors::kFLOAT32);
    ASSERT_EQ(shape.size(), 1u);
    EXPECT_EQ(shape[0], kExactValues.size());

    size_t bytes = 0;
    const void* raw = loader.GetRawData("w", &bytes);
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(bytes, kExactValues.size() * sizeof(float));
    EXPECT_EQ(ReadAsFloats(raw, kExactValues.size()), kExactValues);
}

TEST(SafetensorsLoaderTest, Bf16ConvertsToFp32) {
    const std::string path = TempPath("bf16_to_f32");
    ASSERT_TRUE(WriteSingle(path, test_support::TensorSpec::Dtype::kBF16, kExactValues));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));

    size_t bytes = 0;
    const void* data =
        loader.GetConvertedData("w", nvinfer1::DataType::kFLOAT, &bytes);
    ASSERT_NE(data, nullptr);
    // 回归：SafetensorsToTrtDtype 对 BF16 会回退成 kFLOAT，早期实现据此判定"类型相同"，
    // 直接把 2 字节的 BF16 原始数据当成 4 字节 FP32 返回。字节数必须按目标类型算。
    EXPECT_EQ(bytes, kExactValues.size() * sizeof(float));
    EXPECT_EQ(ReadAsFloats(data, kExactValues.size()), kExactValues);
}

TEST(SafetensorsLoaderTest, Bf16ConvertsToFp16) {
    const std::string path = TempPath("bf16_to_f16");
    ASSERT_TRUE(WriteSingle(path, test_support::TensorSpec::Dtype::kBF16, kExactValues));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));

    size_t bytes = 0;
    const void* data = loader.GetConvertedData("w", nvinfer1::DataType::kHALF, &bytes);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(bytes, kExactValues.size() * sizeof(uint16_t));

    // BF16 与 FP16 的指数位宽度不同（8 vs 5），早期实现靠位截断转换在数值上是错的。
    // 这里逐元素比较：能精确表示说明走的是"先还原 FP32 再降精度"的正确路径。
    const auto* halves = static_cast<const uint16_t*>(data);
    for (size_t i = 0; i < kExactValues.size(); ++i) {
        __half half;
        std::memcpy(&half, &halves[i], sizeof(uint16_t));
        EXPECT_FLOAT_EQ(__half2float(half), kExactValues[i]) << "index " << i;
    }
}

TEST(SafetensorsLoaderTest, Fp16ConvertsToFp32) {
    const std::string path = TempPath("f16_to_f32");
    ASSERT_TRUE(WriteSingle(path, test_support::TensorSpec::Dtype::kF16, kExactValues));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));

    size_t bytes = 0;
    const void* data =
        loader.GetConvertedData("w", nvinfer1::DataType::kFLOAT, &bytes);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(bytes, kExactValues.size() * sizeof(float));
    EXPECT_EQ(ReadAsFloats(data, kExactValues.size()), kExactValues);
}

TEST(SafetensorsLoaderTest, Fp32ConvertsToFp16) {
    const std::string path = TempPath("f32_to_f16");
    ASSERT_TRUE(WriteSingle(path, test_support::TensorSpec::Dtype::kF32, kExactValues));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));

    size_t bytes = 0;
    const void* data = loader.GetConvertedData("w", nvinfer1::DataType::kHALF, &bytes);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(bytes, kExactValues.size() * sizeof(uint16_t));

    const auto* halves = static_cast<const uint16_t*>(data);
    for (size_t i = 0; i < kExactValues.size(); ++i) {
        __half half;
        std::memcpy(&half, &halves[i], sizeof(uint16_t));
        EXPECT_FLOAT_EQ(__half2float(half), kExactValues[i]) << "index " << i;
    }
}

TEST(SafetensorsLoaderTest, SameDtypeRequestIsZeroCopy) {
    const std::string path = TempPath("zero_copy");
    ASSERT_TRUE(WriteSingle(path, test_support::TensorSpec::Dtype::kF32, kExactValues));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));

    size_t raw_bytes = 0;
    size_t converted_bytes = 0;
    const void* raw = loader.GetRawData("w", &raw_bytes);
    const void* converted =
        loader.GetConvertedData("w", nvinfer1::DataType::kFLOAT, &converted_bytes);
    // 类型一致时不应发生拷贝
    EXPECT_EQ(raw, converted);
    EXPECT_EQ(raw_bytes, converted_bytes);
}

TEST(SafetensorsLoaderTest, ConversionResultsForDifferentTensorsDoNotAlias) {
    // 回归：早期实现把转换结果写进一个共享缓冲区，取第二个张量会覆盖第一个。
    // 而 nvinfer1::Weights 只存裸指针、到 buildSerializedNetwork 才读数据，
    // 所以这种覆盖会让先取的权重在构建时变成后取的权重——静默产出错误网络。
    const std::vector<float> first{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> second{-1.0f, -2.0f, -3.0f, -4.0f};
    std::map<std::string, test_support::TensorSpec> tensors;
    tensors["first"] = test_support::TensorSpec{
        {first.size()}, test_support::TensorSpec::Dtype::kBF16, first};
    tensors["second"] = test_support::TensorSpec{
        {second.size()}, test_support::TensorSpec::Dtype::kBF16, second};

    const std::string path = TempPath("no_alias");
    ASSERT_TRUE(test_support::WriteSafetensorsFile(path, tensors));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));

    size_t first_bytes = 0;
    size_t second_bytes = 0;
    const void* first_data =
        loader.GetConvertedData("first", nvinfer1::DataType::kFLOAT, &first_bytes);
    const void* second_data =
        loader.GetConvertedData("second", nvinfer1::DataType::kFLOAT, &second_bytes);

    ASSERT_NE(first_data, nullptr);
    ASSERT_NE(second_data, nullptr);
    EXPECT_NE(first_data, second_data);
    // 关键断言：取完第二个之后，第一个的内容必须仍是它自己的
    EXPECT_EQ(ReadAsFloats(first_data, first.size()), first);
    EXPECT_EQ(ReadAsFloats(second_data, second.size()), second);
}

TEST(SafetensorsLoaderTest, RepeatedRequestHitsCacheAndStaysStable) {
    const std::string path = TempPath("cache_hit");
    ASSERT_TRUE(WriteSingle(path, test_support::TensorSpec::Dtype::kBF16, kExactValues));

    SafetensorsLoader loader;
    ASSERT_TRUE(loader.LoadFromFile(path));

    size_t first_bytes = 0;
    size_t second_bytes = 0;
    const void* first =
        loader.GetConvertedData("w", nvinfer1::DataType::kFLOAT, &first_bytes);
    const void* second =
        loader.GetConvertedData("w", nvinfer1::DataType::kFLOAT, &second_bytes);
    EXPECT_EQ(first, second);
    EXPECT_EQ(first_bytes, second_bytes);
}

}  // namespace mini_trt_llm
