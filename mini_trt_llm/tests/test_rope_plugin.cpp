#include "mini_trt_llm/plugins/rope_kernel.hpp"
#include "mini_trt_llm/plugins/rope_plugin.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include "test_reference.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::WithinTolerance;

// 浮点入参的薄封装：真正的参考实现在 test_support::ReferenceRoPE（唯一来源）。
//
// 之所以不再在本文件里另写一份：曾经存在两份 RoPE 参考实现，其中一份漏了 batch 维度，
// 而另一份是对的——两处独立维护必然漂移（见 docs/TROUBLESHOOTING.md #9）。
void CpuRoPE(const std::vector<float>& input, const std::vector<int32_t>& position_ids,
             int32_t batch_size, int32_t heads, int32_t seq_len, int32_t head_size,
             int32_t rotary_dim, float base, std::vector<float>* output) {
    const std::vector<double> input_d(input.begin(), input.end());
    std::vector<double> output_d;
    test_support::ReferenceRoPE(input_d, position_ids, batch_size, heads, seq_len,
                                head_size, rotary_dim, base, &output_d);
    output->assign(output_d.begin(), output_d.end());
}

float DeterministicValue(int64_t index) {
    return std::sin(0.71f * static_cast<float>(index)) * 0.8f;
}

// 跑一次 kernel，返回 (rotated_query, rotated_key)。
void RunKernel(const std::vector<float>& query, const std::vector<float>& key,
               const std::vector<int32_t>& position_ids, int32_t batch_size,
               int32_t num_heads, int32_t num_kv_heads, int32_t seq_len, int32_t head_size,
               int32_t rotary_dim, float base, std::vector<float>* query_out,
               std::vector<float>* key_out) {
    const size_t query_bytes = query.size() * sizeof(float);
    const size_t key_bytes = key.size() * sizeof(float);
    const size_t position_bytes = position_ids.size() * sizeof(int32_t);

    DeviceBuffer d_query(query_bytes);
    DeviceBuffer d_key(key_bytes);
    DeviceBuffer d_position(position_bytes);
    DeviceBuffer d_query_out(query_bytes);
    DeviceBuffer d_key_out(key_bytes);
    if (!d_query.Allocate(query_bytes) || !d_key.Allocate(key_bytes) ||
        !d_position.Allocate(position_bytes) || !d_query_out.Allocate(query_bytes) ||
        !d_key_out.Allocate(key_bytes)) {
        throw std::runtime_error("RoPE test: failed to allocate device buffers");
    }

    CUDA_CHECK(cudaMemcpy(d_query.data(), query.data(), query_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key.data(), key.data(), key_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_position.data(), position_ids.data(), position_bytes,
                          cudaMemcpyHostToDevice));

    RoPEKernelArgs args;
    args.query = d_query.data();
    args.key = d_key.data();
    args.position_ids = static_cast<const int32_t*>(d_position.data());
    args.query_out = d_query_out.data();
    args.key_out = d_key_out.data();
    args.batch_size = batch_size;
    args.seq_len = seq_len;
    args.num_heads = num_heads;
    args.num_kv_heads = num_kv_heads;
    args.head_size = head_size;
    args.rotary_dim = rotary_dim;
    args.base = base;
    args.is_half = false;
    CUDA_CHECK(LaunchRoPE(args, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    query_out->resize(query.size());
    key_out->resize(key.size());
    CUDA_CHECK(cudaMemcpy(query_out->data(), d_query_out.data(), query_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(key_out->data(), d_key_out.data(), key_bytes,
                          cudaMemcpyDeviceToHost));
}

}  // namespace

TEST(RoPEPluginTest, TypeVersionAndOutputCount) {
    RoPEPlugin plugin(4, 2, 8, 8, 10000.0f);
    EXPECT_STREQ(plugin.getPluginType(), kRoPEPluginName);
    EXPECT_STREQ(plugin.getPluginVersion(), kRoPEPluginVersion);
    EXPECT_EQ(plugin.getNbOutputs(), 2);
}

TEST(RoPEPluginTest, CloneCopiesAttributes) {
    RoPEPlugin plugin(4, 2, 8, 4, 5000.0f);
    std::unique_ptr<IPluginV3Base> copy(plugin.clone());
    ASSERT_NE(copy, nullptr);

    const auto* restored = static_cast<const RoPEPlugin*>(copy.get());
    EXPECT_EQ(restored->num_heads(), 4);
    EXPECT_EQ(restored->num_kv_heads(), 2);
    EXPECT_EQ(restored->head_size(), 8);
    EXPECT_EQ(restored->rotary_dim(), 4);
    EXPECT_FLOAT_EQ(restored->base(), 5000.0f);
}

TEST(RoPEPluginTest, SerializedFieldsRoundTrip) {
    RoPEPlugin plugin(4, 2, 8, 4, 5000.0f);
    const nvinfer1::PluginFieldCollection* fields = plugin.getFieldsToSerialize();
    ASSERT_NE(fields, nullptr);
    ASSERT_EQ(fields->nbFields, 2);

    RoPEPlugin restored(fields);
    EXPECT_EQ(restored.rotary_dim(), 4);
    EXPECT_FLOAT_EQ(restored.base(), 5000.0f);
}

TEST(RoPEPluginCreatorTest, FieldNamesDeclareTwoAttributes) {
    RoPEPluginCreator creator;
    EXPECT_STREQ(creator.getPluginName(), kRoPEPluginName);
    const nvinfer1::PluginFieldCollection* names = creator.getFieldNames();
    ASSERT_NE(names, nullptr);
    ASSERT_EQ(names->nbFields, 2);
    EXPECT_STREQ(names->fields[0].name, "rotary_dim");
    EXPECT_STREQ(names->fields[1].name, "base");
}

TEST(RoPEPluginCreatorTest, CreatePluginFromFields) {
    int32_t rotary_dim = 4;
    float base = 5000.0f;
    nvinfer1::PluginField fields[2] = {
        nvinfer1::PluginField("rotary_dim", &rotary_dim,
                              nvinfer1::PluginFieldType::kINT32, 1),
        nvinfer1::PluginField("base", &base, nvinfer1::PluginFieldType::kFLOAT32, 1)};
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = 2;
    fc.fields = fields;

    RoPEPluginCreator creator;
    std::unique_ptr<nvinfer1::IPluginV3> plugin(
        creator.createPlugin(kRoPEPluginName, &fc, nvinfer1::TensorRTPhase::kBUILD));
    ASSERT_NE(plugin, nullptr);

    const auto* rope = static_cast<const RoPEPlugin*>(plugin.get());
    EXPECT_EQ(rope->rotary_dim(), 4);
    EXPECT_FLOAT_EQ(rope->base(), 5000.0f);
}

TEST(RoPEPluginTest, PositionIdsMustBeInt32WhileOthersMatchQuery) {
    RoPEPlugin plugin(4, 2, 8, 8, 10000.0f);

    // 顺序：query, key, position_ids, rotated_query, rotated_key
    nvinfer1::DynamicPluginTensorDesc in_out[5];
    for (int i = 0; i < 5; ++i) {
        in_out[i].desc.dims = nvinfer1::Dims{4, {1, 4, 3, 8}};
        in_out[i].desc.format = nvinfer1::TensorFormat::kLINEAR;
        in_out[i].desc.type = nvinfer1::DataType::kFLOAT;
    }
    in_out[2].desc.type = nvinfer1::DataType::kINT32;

    for (int pos = 0; pos < 5; ++pos) {
        EXPECT_TRUE(plugin.supportsFormatCombination(pos, in_out, 3, 2)) << "pos=" << pos;
    }

    // position_ids 传成浮点必须被拒绝
    in_out[2].desc.type = nvinfer1::DataType::kFLOAT;
    EXPECT_FALSE(plugin.supportsFormatCombination(2, in_out, 3, 2));

    // key 与 query 类型不一致必须被拒绝
    in_out[2].desc.type = nvinfer1::DataType::kINT32;
    in_out[1].desc.type = nvinfer1::DataType::kHALF;
    EXPECT_FALSE(plugin.supportsFormatCombination(1, in_out, 3, 2));
}

TEST(RoPEPluginTest, ConfigureRejectsInvalidRotaryDim) {
    nvinfer1::DynamicPluginTensorDesc inputs[3];
    inputs[0].desc.dims = nvinfer1::Dims{4, {1, 4, 3, 8}};
    inputs[1].desc.dims = nvinfer1::Dims{4, {1, 4, 3, 8}};
    inputs[2].desc.dims = nvinfer1::Dims{2, {1, 3}};

    // 奇数 rotary_dim 无法成对旋转
    EXPECT_EQ(RoPEPlugin(4, 4, 8, 5, 10000.0f).configurePlugin(inputs, 3, nullptr, 2), 1);
    // 超过 head_size
    EXPECT_EQ(RoPEPlugin(4, 4, 8, 16, 10000.0f).configurePlugin(inputs, 3, nullptr, 2), 1);

    RoPEPlugin valid(4, 4, 8, 8, 10000.0f);
    EXPECT_EQ(valid.configurePlugin(inputs, 3, nullptr, 2), 0);
    EXPECT_EQ(valid.num_kv_heads(), 4);
}

TEST(RoPEPluginTest, ConfigureRejectsNonDivisibleGqaHeads) {
    // kv head 维度声明为动态轴时，num_kv_heads 只能取自属性，GQA 整除约束此时才可被触发；
    // 若形状是静态的，shape 会覆盖属性，也就不会出现"属性与形状不一致"这种情况。
    nvinfer1::DynamicPluginTensorDesc inputs[3];
    inputs[0].desc.dims = nvinfer1::Dims{4, {1, 4, 3, 8}};
    inputs[1].desc.dims = nvinfer1::Dims{4, {1, -1, 3, 8}};
    inputs[2].desc.dims = nvinfer1::Dims{2, {1, 3}};

    RoPEPlugin plugin(4, 3, 8, 8, 10000.0f);
    EXPECT_EQ(plugin.configurePlugin(inputs, 3, nullptr, 2), 1);
}

TEST(RoPEPluginTest, RotaryDimDefaultsToHeadSize) {
    nvinfer1::DynamicPluginTensorDesc inputs[3];
    inputs[0].desc.dims = nvinfer1::Dims{4, {1, 4, 3, 8}};
    inputs[1].desc.dims = nvinfer1::Dims{4, {1, 4, 3, 8}};
    inputs[2].desc.dims = nvinfer1::Dims{2, {1, 3}};

    // rotary_dim = 0 表示未配置，应按 Q3 默认为 head_size
    RoPEPlugin plugin(4, 4, 8, 0, 10000.0f);
    EXPECT_EQ(plugin.configurePlugin(inputs, 3, nullptr, 2), 0);
    EXPECT_EQ(plugin.rotary_dim(), 8);
}

// 回归：TensorRT 会在**反序列化出来的实例**上直接调用 onShapeChange，该实例从未执行过
// configurePlugin，因此 head 配置成员仍是默认值 0（它们不作为序列化属性）。
// 早期实现拿运行期形状去校验这些成员，导致真机上所有 RoPE 端到端用例在 enqueue 时报
// "RoPE: shape change alters head configuration"。这条例用不需要 GPU。
TEST(RoPEPluginTest, DeserializedPluginRefreshesHeadConfigFromShape) {
    RoPEPlugin original(2, 2, 8, 4, 10000.0f);
    const nvinfer1::PluginFieldCollection* fields = original.getFieldsToSerialize();

    // 模拟 engine 反序列化：只恢复 rotary_dim / base
    RoPEPlugin deserialized(fields);
    EXPECT_EQ(deserialized.num_heads(), 0);
    EXPECT_EQ(deserialized.head_size(), 0);

    nvinfer1::PluginTensorDesc inputs[3];
    inputs[0].dims = nvinfer1::Dims{4, {1, 2, 4, 8}};
    inputs[1].dims = nvinfer1::Dims{4, {1, 2, 4, 8}};
    inputs[2].dims = nvinfer1::Dims{2, {1, 4}};

    EXPECT_EQ(deserialized.onShapeChange(inputs, 3, nullptr, 2), 0);
    // 刷新后应与原始插件一致，否则 enqueue 会用到错误的 head 配置
    EXPECT_EQ(deserialized.num_heads(), 2);
    EXPECT_EQ(deserialized.num_kv_heads(), 2);
    EXPECT_EQ(deserialized.head_size(), 8);
    EXPECT_EQ(deserialized.rotary_dim(), 4);
}

TEST(RoPEPluginTest, DeserializedPluginStillRejectsInconsistentShape) {
    RoPEPlugin original(4, 2, 8, 8, 10000.0f);
    const nvinfer1::PluginFieldCollection* fields = original.getFieldsToSerialize();
    RoPEPlugin deserialized(fields);

    // 刷新逻辑不能把真正的矛盾也放过去：GQA 不整除必须仍然失败
    nvinfer1::PluginTensorDesc inputs[3];
    inputs[0].dims = nvinfer1::Dims{4, {1, 4, 4, 8}};
    inputs[1].dims = nvinfer1::Dims{4, {1, 3, 4, 8}};
    inputs[2].dims = nvinfer1::Dims{2, {1, 4}};
    EXPECT_EQ(deserialized.onShapeChange(inputs, 3, nullptr, 2), 1);
}

TEST(RoPEKernelTest, MatchesCpuReferenceWithContiguousPositions) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    constexpr int32_t kBatch = 1;
    constexpr int32_t kHeads = 2;
    constexpr int32_t kSeq = 3;
    constexpr int32_t kHeadSize = 8;
    constexpr float kBase = 10000.0f;

    const size_t query_elements = static_cast<size_t>(kBatch) * kHeads * kSeq * kHeadSize;
    std::vector<float> query(query_elements);
    std::vector<float> key(query_elements);
    std::vector<int32_t> positions(kBatch * kSeq);
    for (size_t i = 0; i < query_elements; ++i) {
        query[i] = DeterministicValue(static_cast<int64_t>(i));
        key[i] = DeterministicValue(static_cast<int64_t>(i) + 1000);
    }
    for (int32_t s = 0; s < kSeq; ++s) {
        positions[s] = s;
    }

    std::vector<float> expected_query;
    std::vector<float> expected_key;
    CpuRoPE(query, positions, kBatch, kHeads, kSeq, kHeadSize, kHeadSize, kBase,
            &expected_query);
    CpuRoPE(key, positions, kBatch, kHeads, kSeq, kHeadSize, kHeadSize, kBase,
            &expected_key);

    std::vector<float> actual_query;
    std::vector<float> actual_key;
    RunKernel(query, key, positions, kBatch, kHeads, kHeads, kSeq, kHeadSize, kHeadSize,
              kBase, &actual_query, &actual_key);

    for (size_t i = 0; i < query_elements; ++i) {
        EXPECT_TRUE(WithinTolerance(expected_query[i], actual_query[i], 1e-5f, 1e-6f))
            << "query index " << i;
        EXPECT_TRUE(WithinTolerance(expected_key[i], actual_key[i], 1e-5f, 1e-6f))
            << "key index " << i;
    }
}

TEST(RoPEKernelTest, HandlesNonContiguousPositionIds) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // KV Cache 续写场景：position_ids 不连续，且 query/key 序列长度不同
    constexpr int32_t kBatch = 1;
    constexpr int32_t kHeads = 2;
    constexpr int32_t kSeq = 2;
    constexpr int32_t kHeadSize = 4;
    constexpr float kBase = 10000.0f;

    const size_t query_elements = static_cast<size_t>(kBatch) * kHeads * kSeq * kHeadSize;
    std::vector<float> query(query_elements);
    std::vector<float> key(query_elements);
    for (size_t i = 0; i < query_elements; ++i) {
        query[i] = DeterministicValue(static_cast<int64_t>(i));
        key[i] = DeterministicValue(static_cast<int64_t>(i) + 500);
    }
    std::vector<int32_t> positions{7, 9};

    std::vector<float> expected_query;
    std::vector<float> expected_key;
    CpuRoPE(query, positions, kBatch, kHeads, kSeq, kHeadSize, kHeadSize, kBase,
            &expected_query);
    CpuRoPE(key, positions, kBatch, kHeads, kSeq, kHeadSize, kHeadSize, kBase,
            &expected_key);

    std::vector<float> actual_query;
    std::vector<float> actual_key;
    RunKernel(query, key, positions, kBatch, kHeads, kHeads, kSeq, kHeadSize, kHeadSize,
              kBase, &actual_query, &actual_key);

    for (size_t i = 0; i < query_elements; ++i) {
        EXPECT_TRUE(WithinTolerance(expected_query[i], actual_query[i], 1e-5f, 1e-6f))
            << "query index " << i;
        EXPECT_TRUE(WithinTolerance(expected_key[i], actual_key[i], 1e-5f, 1e-6f))
            << "key index " << i;
    }
}

TEST(RoPEKernelTest, SupportsPartialRotaryDim) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    // rotary_dim=4 < head_size=8：后半部分维度应保持原值不变
    constexpr int32_t kBatch = 1;
    constexpr int32_t kHeads = 1;
    constexpr int32_t kSeq = 2;
    constexpr int32_t kHeadSize = 8;
    constexpr int32_t kRotaryDim = 4;

    const size_t elements = static_cast<size_t>(kBatch) * kHeads * kSeq * kHeadSize;
    std::vector<float> query(elements);
    std::vector<float> key(elements);
    for (size_t i = 0; i < elements; ++i) {
        query[i] = DeterministicValue(static_cast<int64_t>(i));
        key[i] = DeterministicValue(static_cast<int64_t>(i) + 300);
    }
    std::vector<int32_t> positions{0, 1};

    std::vector<float> expected_query;
    std::vector<float> expected_key;
    CpuRoPE(query, positions, kBatch, kHeads, kSeq, kHeadSize, kRotaryDim, 10000.0f,
            &expected_query);
    CpuRoPE(key, positions, kBatch, kHeads, kSeq, kHeadSize, kRotaryDim, 10000.0f,
            &expected_key);

    std::vector<float> actual_query;
    std::vector<float> actual_key;
    RunKernel(query, key, positions, kBatch, kHeads, kHeads, kSeq, kHeadSize, kRotaryDim,
              10000.0f, &actual_query, &actual_key);

    for (size_t i = 0; i < elements; ++i) {
        EXPECT_TRUE(WithinTolerance(expected_query[i], actual_query[i], 1e-5f, 1e-6f))
            << "query index " << i;
        EXPECT_TRUE(WithinTolerance(expected_key[i], actual_key[i], 1e-5f, 1e-6f))
            << "key index " << i;
    }
}

TEST(RoPEKernelTest, RejectsInvalidArguments) {
    RoPEKernelArgs args;
    EXPECT_NE(LaunchRoPE(args, nullptr), cudaSuccess);

    int32_t position = 0;
    args.query = &position;
    args.key = &position;
    args.position_ids = &position;
    args.query_out = &position;
    args.key_out = &position;
    args.batch_size = 1;
    args.seq_len = 1;
    args.num_heads = 1;
    args.num_kv_heads = 1;
    args.head_size = 8;
    args.rotary_dim = 7;  // 奇数：不可成对旋转
    EXPECT_NE(LaunchRoPE(args, nullptr), cudaSuccess);
}

}  // namespace mini_trt_llm
