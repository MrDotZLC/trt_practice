#include "mini_trt_llm/plugins/paged_attention_kernel.hpp"
#include "mini_trt_llm/plugins/paged_attention_plugin.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include "test_reference.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {
namespace {

using test_support::HasCudaDevice;
using test_support::WithinTolerance;

float DeterministicValue(int64_t index) {
    return std::sin(0.53f * static_cast<float>(index)) * 0.7f;
}

// PagedAttention Decoding 的 CPU 参考实现。
// 逻辑上等价于：对每个 (batch, head)，用 query 与缓存里前 context_len 个 K/V 做
// 标准 scaled dot-product attention，再用块表把逻辑位置映射到物理块。
void CpuPagedAttention(const std::vector<float>& query, const std::vector<float>& key_cache,
                       const std::vector<float>& value_cache,
                       const std::vector<int32_t>& block_tables,
                       const std::vector<int32_t>& context_lens, int32_t batch_size,
                       int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                       int32_t block_size, int32_t max_blocks_per_seq, float scale,
                       std::vector<float>* output) {
    output->assign(static_cast<size_t>(batch_size) * num_heads * head_size, 0.0f);

    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t h = 0; h < num_heads; ++h) {
            const int32_t kv_head = h / (num_heads / num_kv_heads);
            const int32_t context_len = context_lens[b];

            std::vector<float> scores(context_len, 0.0f);
            float max_score = -1e30f;
            for (int32_t t = 0; t < context_len; ++t) {
                const int32_t physical_block = block_tables[b * max_blocks_per_seq + t / block_size];
                const int32_t slot = t % block_size;
                const size_t kv_offset =
                    ((static_cast<size_t>(physical_block) * block_size + slot) * num_kv_heads +
                     kv_head) *
                    head_size;
                double dot = 0.0;
                for (int32_t d = 0; d < head_size; ++d) {
                    dot += static_cast<double>(query[(static_cast<size_t>(b) * num_heads + h) *
                                                         head_size + d]) *
                           key_cache[kv_offset + d];
                }
                scores[t] = static_cast<float>(dot) * scale;
                max_score = std::max(max_score, scores[t]);
            }

            double sum = 0.0;
            for (int32_t t = 0; t < context_len; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum += scores[t];
            }
            if (sum <= 0.0) {
                continue;
            }

            for (int32_t d = 0; d < head_size; ++d) {
                double acc = 0.0;
                for (int32_t t = 0; t < context_len; ++t) {
                    const int32_t physical_block =
                        block_tables[b * max_blocks_per_seq + t / block_size];
                    const int32_t slot = t % block_size;
                    const size_t kv_offset =
                        ((static_cast<size_t>(physical_block) * block_size + slot) *
                             num_kv_heads +
                         kv_head) *
                        head_size;
                    acc += scores[t] * value_cache[kv_offset + d];
                }
                (*output)[(static_cast<size_t>(b) * num_heads + h) * head_size + d] =
                    static_cast<float>(acc / sum);
            }
        }
    }
}

// 构造一个物理块顺序与逻辑顺序不同的缓存，确保 kernel 真的走了块表而不是直接顺序读。
struct AttentionFixture {
    std::vector<float> query;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    std::vector<int32_t> block_tables;
    std::vector<int32_t> context_lens;
    int32_t batch_size = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_size = 0;
    int32_t block_size = 0;
    int32_t max_blocks_per_seq = 0;
    int32_t num_blocks = 0;
    float scale = 0.0f;
};

AttentionFixture MakeFixture(int32_t batch_size, int32_t num_heads, int32_t num_kv_heads,
                             int32_t head_size, int32_t block_size,
                             const std::vector<int32_t>& context_lens,
                             const std::vector<int32_t>& block_tables,
                             int32_t max_blocks_per_seq, int32_t num_blocks) {
    AttentionFixture fixture;
    fixture.batch_size = batch_size;
    fixture.num_heads = num_heads;
    fixture.num_kv_heads = num_kv_heads;
    fixture.head_size = head_size;
    fixture.block_size = block_size;
    fixture.max_blocks_per_seq = max_blocks_per_seq;
    fixture.num_blocks = num_blocks;
    fixture.context_lens = context_lens;
    fixture.block_tables = block_tables;
    fixture.scale = 1.0f / std::sqrt(static_cast<float>(head_size));

    fixture.query.resize(static_cast<size_t>(batch_size) * num_heads * head_size);
    for (size_t i = 0; i < fixture.query.size(); ++i) {
        fixture.query[i] = DeterministicValue(static_cast<int64_t>(i));
    }

    const size_t cache_elements =
        static_cast<size_t>(num_blocks) * block_size * num_kv_heads * head_size;
    fixture.key_cache.resize(cache_elements);
    fixture.value_cache.resize(cache_elements);
    for (size_t i = 0; i < cache_elements; ++i) {
        fixture.key_cache[i] = DeterministicValue(static_cast<int64_t>(i) + 100);
        fixture.value_cache[i] = DeterministicValue(static_cast<int64_t>(i) + 5000);
    }
    return fixture;
}

void RunKernel(const AttentionFixture& fixture, std::vector<float>* output) {
    const size_t query_bytes = fixture.query.size() * sizeof(float);
    const size_t cache_bytes = fixture.key_cache.size() * sizeof(float);
    const size_t table_bytes = fixture.block_tables.size() * sizeof(int32_t);
    const size_t lens_bytes = fixture.context_lens.size() * sizeof(int32_t);
    const size_t output_bytes =
        static_cast<size_t>(fixture.batch_size) * fixture.num_heads * fixture.head_size *
        sizeof(float);

    DeviceBuffer d_query(query_bytes);
    DeviceBuffer d_key_cache(cache_bytes);
    DeviceBuffer d_value_cache(cache_bytes);
    DeviceBuffer d_block_tables(table_bytes);
    DeviceBuffer d_context_lens(lens_bytes);
    DeviceBuffer d_output(output_bytes);
    if (!d_query.Allocate(query_bytes) || !d_key_cache.Allocate(cache_bytes) ||
        !d_value_cache.Allocate(cache_bytes) || !d_block_tables.Allocate(table_bytes) ||
        !d_context_lens.Allocate(lens_bytes) || !d_output.Allocate(output_bytes)) {
        throw std::runtime_error("PagedAttention test: failed to allocate device buffers");
    }

    CUDA_CHECK(cudaMemcpy(d_query.data(), fixture.query.data(), query_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key_cache.data(), fixture.key_cache.data(), cache_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value_cache.data(), fixture.value_cache.data(), cache_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables.data(), fixture.block_tables.data(), table_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_context_lens.data(), fixture.context_lens.data(), lens_bytes,
                          cudaMemcpyHostToDevice));

    PagedAttentionKernelArgs args;
    args.query = d_query.data();
    args.key_cache = d_key_cache.data();
    args.value_cache = d_value_cache.data();
    args.block_tables = static_cast<const int32_t*>(d_block_tables.data());
    args.context_lens = static_cast<const int32_t*>(d_context_lens.data());
    args.output = d_output.data();
    args.batch_size = fixture.batch_size;
    args.num_heads = fixture.num_heads;
    args.num_kv_heads = fixture.num_kv_heads;
    args.head_size = fixture.head_size;
    args.block_size = fixture.block_size;
    args.max_blocks_per_seq = fixture.max_blocks_per_seq;
    args.scale = fixture.scale;
    args.is_half = false;
    CUDA_CHECK(LaunchPagedAttention(args, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    output->resize(output_bytes / sizeof(float));
    CUDA_CHECK(cudaMemcpy(output->data(), d_output.data(), output_bytes,
                          cudaMemcpyDeviceToHost));
}

void ExpectMatchesReference(const AttentionFixture& fixture) {
    std::vector<float> expected;
    CpuPagedAttention(fixture.query, fixture.key_cache, fixture.value_cache,
                      fixture.block_tables, fixture.context_lens, fixture.batch_size,
                      fixture.num_heads, fixture.num_kv_heads, fixture.head_size,
                      fixture.block_size, fixture.max_blocks_per_seq, fixture.scale,
                      &expected);
    std::vector<float> actual;
    RunKernel(fixture, &actual);

    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_TRUE(WithinTolerance(expected[i], actual[i], 1e-4f, 1e-5f))
            << "index " << i << " expected=" << expected[i] << " actual=" << actual[i];
    }
}

}  // namespace

TEST(PagedAttentionPluginTest, TypeVersionAndOutputCount) {
    PagedAttentionPlugin plugin(4, 2, 8, 16, 0.0f);
    EXPECT_STREQ(plugin.getPluginType(), kPagedAttentionPluginName);
    EXPECT_STREQ(plugin.getPluginVersion(), kPagedAttentionPluginVersion);
    EXPECT_EQ(plugin.getNbOutputs(), 1);
}

TEST(PagedAttentionPluginTest, CloneCopiesAttributes) {
    PagedAttentionPlugin plugin(4, 2, 8, 16, 0.5f);
    std::unique_ptr<IPluginV3Base> copy(plugin.clone());
    ASSERT_NE(copy, nullptr);

    const auto* restored = static_cast<const PagedAttentionPlugin*>(copy.get());
    EXPECT_EQ(restored->num_heads(), 4);
    EXPECT_EQ(restored->num_kv_heads(), 2);
    EXPECT_EQ(restored->head_size(), 8);
    EXPECT_EQ(restored->block_size(), 16);
    EXPECT_FLOAT_EQ(restored->scale(), 0.5f);
}

TEST(PagedAttentionPluginTest, SerializedFieldsRoundTrip) {
    PagedAttentionPlugin plugin(4, 2, 8, 16, 0.5f);
    const nvinfer1::PluginFieldCollection* fields = plugin.getFieldsToSerialize();
    ASSERT_NE(fields, nullptr);
    ASSERT_EQ(fields->nbFields, 2);

    PagedAttentionPlugin restored(fields);
    EXPECT_EQ(restored.block_size(), 16);
    EXPECT_FLOAT_EQ(restored.scale(), 0.5f);
}

TEST(PagedAttentionPluginCreatorTest, FieldNamesDeclareTwoAttributes) {
    PagedAttentionPluginCreator creator;
    EXPECT_STREQ(creator.getPluginName(), kPagedAttentionPluginName);
    const nvinfer1::PluginFieldCollection* names = creator.getFieldNames();
    ASSERT_NE(names, nullptr);
    ASSERT_EQ(names->nbFields, 2);
    EXPECT_STREQ(names->fields[0].name, "block_size");
    EXPECT_STREQ(names->fields[1].name, "scale");
}

TEST(PagedAttentionPluginTest, BlockSizeIsRequiredWithoutDefault) {
    nvinfer1::DynamicPluginTensorDesc inputs[5];
    inputs[0].desc.dims = nvinfer1::Dims{4, {1, 4, 1, 8}};
    inputs[1].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[2].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[3].desc.dims = nvinfer1::Dims{2, {1, 4}};
    inputs[4].desc.dims = nvinfer1::Dims{1, {1}};

    // Q5：block_size 未配置时必须直接失败，不能有隐含默认值
    PagedAttentionPlugin unset(4, 2, 8, 0, 0.0f);
    EXPECT_EQ(unset.configurePlugin(inputs, 5, nullptr, 1), 1);

    PagedAttentionPlugin configured(4, 2, 8, 16, 0.0f);
    EXPECT_EQ(configured.configurePlugin(inputs, 5, nullptr, 1), 0);
    // Q6：scale 未显式给出时按 1/sqrt(head_size) 推导
    EXPECT_FLOAT_EQ(configured.scale(), PagedAttentionPlugin::DefaultScale(8));
}

TEST(PagedAttentionPluginTest, RejectsPrefillSequenceLength) {
    nvinfer1::DynamicPluginTensorDesc inputs[5];
    inputs[0].desc.dims = nvinfer1::Dims{4, {1, 4, 3, 8}};  // seq_len = 3 → prefill
    inputs[1].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[2].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[3].desc.dims = nvinfer1::Dims{2, {1, 4}};
    inputs[4].desc.dims = nvinfer1::Dims{1, {1}};

    PagedAttentionPlugin plugin(4, 2, 8, 16, 0.0f);
    EXPECT_EQ(plugin.configurePlugin(inputs, 5, nullptr, 1), 1);
}

TEST(PagedAttentionPluginTest, IndexTensorsMustBeInt32) {
    PagedAttentionPlugin plugin(4, 2, 8, 16, 0.0f);

    // 顺序：query, key_cache, value_cache, block_tables, context_lens, output
    nvinfer1::DynamicPluginTensorDesc in_out[6];
    for (int i = 0; i < 6; ++i) {
        in_out[i].desc.dims = nvinfer1::Dims{4, {1, 4, 1, 8}};
        in_out[i].desc.format = nvinfer1::TensorFormat::kLINEAR;
        in_out[i].desc.type = nvinfer1::DataType::kFLOAT;
    }
    in_out[3].desc.type = nvinfer1::DataType::kINT32;
    in_out[4].desc.type = nvinfer1::DataType::kINT32;

    for (int pos = 0; pos < 6; ++pos) {
        EXPECT_TRUE(plugin.supportsFormatCombination(pos, in_out, 5, 1)) << "pos=" << pos;
    }

    // block_tables 传成浮点必须被拒绝
    in_out[3].desc.type = nvinfer1::DataType::kFLOAT;
    EXPECT_FALSE(plugin.supportsFormatCombination(3, in_out, 5, 1));
}

TEST(PagedAttentionKernelTest, MhaMatchesCpuReferenceAcrossMultipleBlocks) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // 逻辑块顺序 [0,1] 映射到物理块 [5,2]，且两个序列共用同一组物理块，
    // 确保 kernel 真正按块表寻址而不是顺序读缓存。
    const std::vector<int32_t> context_lens{20, 7};
    const std::vector<int32_t> block_tables{5, 2, 2, 7};
    AttentionFixture fixture = MakeFixture(/*batch=*/2, /*heads=*/4, /*kv_heads=*/4,
                                           /*head_size=*/8, /*block_size=*/16,
                                           context_lens, block_tables, /*max_blocks=*/2,
                                           /*num_blocks=*/8);
    ExpectMatchesReference(fixture);
}

TEST(PagedAttentionKernelTest, GqaSharesKvHeadsAcrossQueryHeads) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // num_heads=4 / num_kv_heads=2 → 每 2 个 query head 共享 1 个 kv head
    const std::vector<int32_t> context_lens{13};
    const std::vector<int32_t> block_tables{3, 1};
    AttentionFixture fixture = MakeFixture(/*batch=*/1, /*heads=*/4, /*kv_heads=*/2,
                                           /*head_size=*/16, /*block_size=*/8,
                                           context_lens, block_tables, /*max_blocks=*/2,
                                           /*num_blocks=*/6);
    ExpectMatchesReference(fixture);
}

TEST(PagedAttentionKernelTest, ZeroContextLengthProducesZeros) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // context_len = 0 时不能除零，输出应为 0
    const std::vector<int32_t> context_lens{0};
    const std::vector<int32_t> block_tables{0};
    AttentionFixture fixture = MakeFixture(/*batch=*/1, /*heads=*/2, /*kv_heads=*/2,
                                           /*head_size=*/8, /*block_size=*/16, context_lens,
                                           block_tables, /*max_blocks=*/1,
                                           /*num_blocks=*/1);
    std::vector<float> actual;
    RunKernel(fixture, &actual);
    for (float value : actual) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(PagedAttentionKernelTest, RejectsInvalidArguments) {
    PagedAttentionKernelArgs args;
    EXPECT_NE(LaunchPagedAttention(args, nullptr), cudaSuccess);
}

// 回归：同 RoPE。PagedAttention 只序列化 block_size / scale，head 配置靠形状推导；
// 反序列化后的实例直接进入 onShapeChange 时必须以形状为准刷新。
TEST(PagedAttentionPluginTest, DeserializedPluginRefreshesHeadConfigFromShape) {
    PagedAttentionPlugin original(4, 2, 8, 16, 0.0f);
    const nvinfer1::PluginFieldCollection* fields = original.getFieldsToSerialize();

    PagedAttentionPlugin deserialized(fields);
    EXPECT_EQ(deserialized.num_heads(), 0);
    EXPECT_EQ(deserialized.num_kv_heads(), 0);
    EXPECT_EQ(deserialized.block_size(), 16);  // 序列化属性，必须恢复

    nvinfer1::PluginTensorDesc inputs[5];
    inputs[0].dims = nvinfer1::Dims{4, {1, 4, 1, 8}};
    inputs[1].dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[2].dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[3].dims = nvinfer1::Dims{2, {1, 4}};
    inputs[4].dims = nvinfer1::Dims{1, {1}};

    EXPECT_EQ(deserialized.onShapeChange(inputs, 5, nullptr, 1), 0);
    EXPECT_EQ(deserialized.num_heads(), 4);
    EXPECT_EQ(deserialized.num_kv_heads(), 2);
    EXPECT_EQ(deserialized.head_size(), 8);
    // scale 未显式配置时应由刷新后的 head_size 推导
    EXPECT_FLOAT_EQ(deserialized.scale(), PagedAttentionPlugin::DefaultScale(8));
}

TEST(PagedAttentionPluginTest, DeserializedPluginRejectsCacheBlockMismatch) {
    PagedAttentionPlugin original(4, 2, 8, 16, 0.0f);
    const nvinfer1::PluginFieldCollection* fields = original.getFieldsToSerialize();
    PagedAttentionPlugin deserialized(fields);

    // block_size 与 cache 的 block 维不一致必须失败，不能因为"以形状为准"而放过
    nvinfer1::PluginTensorDesc inputs[5];
    inputs[0].dims = nvinfer1::Dims{4, {1, 4, 1, 8}};
    inputs[1].dims = nvinfer1::Dims{4, {16, 8, 2, 8}};
    inputs[2].dims = nvinfer1::Dims{4, {16, 8, 2, 8}};
    inputs[3].dims = nvinfer1::Dims{2, {1, 4}};
    inputs[4].dims = nvinfer1::Dims{1, {1}};
    EXPECT_EQ(deserialized.onShapeChange(inputs, 5, nullptr, 1), 1);
}

}  // namespace mini_trt_llm
