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

using test_support::WithinTolerance;

float DeterministicValue(int64_t index) {
    return std::sin(0.53f * static_cast<float>(index)) * 0.7f;
}

// PagedAttention Decoding 的 CPU 参考实现。
// 缓存里第 t 个逻辑位置（t < context_len）在 cache 张量里的偏移。
size_t PagedOffset(const std::vector<int32_t>& block_tables, int32_t batch, int32_t t,
                   int32_t block_size, int32_t max_blocks_per_seq, int32_t num_kv_heads,
                   int32_t kv_head, int32_t head_size) {
    const int32_t physical_block =
        block_tables[static_cast<size_t>(batch) * max_blocks_per_seq + t / block_size];
    const int32_t slot = t % block_size;
    return ((static_cast<size_t>(physical_block) * block_size + slot) * num_kv_heads +
            kv_head) *
           head_size;
}

// 当前 token 的 K/V 是独立的一小份张量，不经过 block table。
size_t CurrentTokenOffset(int32_t batch, int32_t num_kv_heads, int32_t kv_head,
                          int32_t head_size) {
    return (static_cast<size_t>(batch) * num_kv_heads + kv_head) * head_size;
}

// 逻辑上等价于：对每个 (batch, head)，用 query 与缓存里前 context_len 个 K/V 做
// 标准 scaled dot-product attention，再用块表把逻辑位置映射到物理块。
//
// key_new / value_new 非空时表示"当前 token 的 K/V 也参与注意力"——它对应 kernel 的
// 第 6/7 个输入，逻辑上排在缓存里的 context_len 个位置之后（第 context_len 个位置）。
void CpuPagedAttention(const std::vector<float>& query, const std::vector<float>& key_cache,
                       const std::vector<float>& value_cache,
                       const std::vector<int32_t>& block_tables,
                       const std::vector<int32_t>& context_lens, int32_t batch_size,
                       int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                       int32_t block_size, int32_t max_blocks_per_seq, float scale,
                       std::vector<float>* output,
                       const std::vector<float>* key_new = nullptr,
                       const std::vector<float>* value_new = nullptr) {
    output->assign(static_cast<size_t>(batch_size) * num_heads * head_size, 0.0f);

    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t h = 0; h < num_heads; ++h) {
            const int32_t kv_head = h / (num_heads / num_kv_heads);
            const int32_t context_len = context_lens[b];
            const bool has_current = key_new != nullptr && value_new != nullptr;
            const int32_t total_len = context_len + (has_current ? 1 : 0);

            std::vector<float> scores(total_len, 0.0f);
            float max_score = -1e30f;
            for (int32_t t = 0; t < total_len; ++t) {
                const size_t kv_offset = t < context_len
                                             ? PagedOffset(block_tables, b, t, block_size,
                                                           max_blocks_per_seq, num_kv_heads,
                                                           kv_head, head_size)
                                             : CurrentTokenOffset(b, num_kv_heads, kv_head,
                                                                  head_size);
                const std::vector<float>& keys =
                    t < context_len ? key_cache : *key_new;
                double dot = 0.0;
                for (int32_t d = 0; d < head_size; ++d) {
                    dot += static_cast<double>(query[(static_cast<size_t>(b) * num_heads + h) *
                                                         head_size + d]) *
                           keys[kv_offset + d];
                }
                scores[t] = static_cast<float>(dot) * scale;
                max_score = std::max(max_score, scores[t]);
            }

            double sum = 0.0;
            for (int32_t t = 0; t < total_len; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum += scores[t];
            }
            if (sum <= 0.0) {
                continue;
            }

            for (int32_t d = 0; d < head_size; ++d) {
                double acc = 0.0;
                for (int32_t t = 0; t < total_len; ++t) {
                    const size_t kv_offset =
                        t < context_len
                            ? PagedOffset(block_tables, b, t, block_size,
                                          max_blocks_per_seq, num_kv_heads, kv_head, head_size)
                            : CurrentTokenOffset(b, num_kv_heads, kv_head, head_size);
                    const std::vector<float>& values =
                        t < context_len ? value_cache : *value_new;
                    acc += scores[t] * values[kv_offset + d];
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
    // 当前 token 的 K/V（插件的第 6/7 个输入）。空表示不连接这两个输入。
    std::vector<float> key_new;
    std::vector<float> value_new;
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

    // 当前 token 的 K/V 只有在 fixture 提供了才分配/传递：
    // 未连接时保持"只按 cache 内容算注意力"的原有路径。
    DeviceBuffer d_key_new;
    DeviceBuffer d_value_new;
    const bool has_current = !fixture.key_new.empty();
    if (has_current) {
        const size_t new_bytes = fixture.key_new.size() * sizeof(float);
        if (!d_key_new.Allocate(new_bytes) || !d_value_new.Allocate(new_bytes)) {
            throw std::runtime_error("PagedAttention test: failed to allocate current token");
        }
        CUDA_CHECK(cudaMemcpy(d_key_new.data(), fixture.key_new.data(), new_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_value_new.data(), fixture.value_new.data(), new_bytes,
                              cudaMemcpyHostToDevice));
    }

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
    if (has_current) {
        args.key_new = d_key_new.data();
        args.value_new = d_value_new.data();
        args.has_current_token = true;
    }
    CUDA_CHECK(LaunchPagedAttention(args, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    output->resize(output_bytes / sizeof(float));
    CUDA_CHECK(cudaMemcpy(output->data(), d_output.data(), output_bytes,
                          cudaMemcpyDeviceToHost));
}

void ExpectMatchesReference(const AttentionFixture& fixture) {
    std::vector<float> expected;
    const bool has_current = !fixture.key_new.empty();
    CpuPagedAttention(fixture.query, fixture.key_cache, fixture.value_cache,
                      fixture.block_tables, fixture.context_lens, fixture.batch_size,
                      fixture.num_heads, fixture.num_kv_heads, fixture.head_size,
                      fixture.block_size, fixture.max_blocks_per_seq, fixture.scale,
                      &expected, has_current ? &fixture.key_new : nullptr,
                      has_current ? &fixture.value_new : nullptr);
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
    MINI_TRT_SKIP_IF_NO_CUDA();
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
    MINI_TRT_SKIP_IF_NO_CUDA();
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
    MINI_TRT_SKIP_IF_NO_CUDA();
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

// 当前 token 必须真的参与注意力。
//
// **最强的一条判据**：让 cache 为空（context_len = 0），只给 key_new / value_new。
// 此时 softmax 只有一个元素、权重恒为 1，输出必须**逐元素等于 value_new**；
// 漏掉当前 token 的实现会走 context_len=0 的兜底分支返回全 0，一眼可辨。
TEST(PagedAttentionKernelTest, CurrentTokenIsAttendedEvenWithEmptyCache) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::vector<int32_t> context_lens{0};
    const std::vector<int32_t> block_tables{0};
    AttentionFixture fixture = MakeFixture(/*batch=*/1, /*heads=*/2, /*kv_heads=*/2,
                                           /*head_size=*/8, /*block_size=*/16, context_lens,
                                           block_tables, /*max_blocks=*/1,
                                           /*num_blocks=*/1);
    // 换掉 cache 内容：即使实现误读 cache，也不会"碰巧"对上
    std::fill(fixture.key_cache.begin(), fixture.key_cache.end(), 0.25f);
    std::fill(fixture.value_cache.begin(), fixture.value_cache.end(), 0.5f);

    fixture.key_new.resize(static_cast<size_t>(fixture.num_kv_heads) * fixture.head_size);
    fixture.value_new.resize(fixture.key_new.size());
    for (size_t i = 0; i < fixture.key_new.size(); ++i) {
        fixture.key_new[i] = DeterministicValue(static_cast<int64_t>(i) + 7);
        fixture.value_new[i] = DeterministicValue(static_cast<int64_t>(i) + 900);
    }

    std::vector<float> actual;
    RunKernel(fixture, &actual);

    ASSERT_EQ(actual.size(), static_cast<size_t>(fixture.batch_size) * fixture.num_heads *
                                fixture.head_size);
    for (int32_t h = 0; h < fixture.num_heads; ++h) {
        const int32_t kv_head = h / (fixture.num_heads / fixture.num_kv_heads);
        for (int32_t d = 0; d < fixture.head_size; ++d) {
            const float expected =
                fixture.value_new[static_cast<size_t>(kv_head) * fixture.head_size + d];
            EXPECT_TRUE(WithinTolerance(
                expected, actual[static_cast<size_t>(h) * fixture.head_size + d], 1e-5f,
                1e-6f))
                << "head " << h << " dim " << d;
        }
    }
}

// 缓存 + 当前 token 一起参与时的数值，与 CPU 参考对比（含 GQA 与 batch>1）
TEST(PagedAttentionKernelTest, CurrentTokenMatchesCpuReferenceWithGqa) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    const std::vector<int32_t> context_lens{5, 3};
    const std::vector<int32_t> block_tables{1, 0, 3, 2, 1, 0};
    AttentionFixture fixture = MakeFixture(/*batch=*/2, /*heads=*/4, /*kv_heads=*/2,
                                           /*head_size=*/8, /*block_size=*/4, context_lens,
                                           block_tables, /*max_blocks=*/3,
                                           /*num_blocks=*/4);
    fixture.key_new.resize(static_cast<size_t>(fixture.batch_size) * fixture.num_kv_heads *
                           fixture.head_size);
    fixture.value_new.resize(fixture.key_new.size());
    for (size_t i = 0; i < fixture.key_new.size(); ++i) {
        fixture.key_new[i] = DeterministicValue(static_cast<int64_t>(i) + 31);
        fixture.value_new[i] = DeterministicValue(static_cast<int64_t>(i) + 77);
    }

    ExpectMatchesReference(fixture);
}

// 只连 key_new 不连 value_new 属于接线错误，必须在 configurePlugin 就拒绝：
// 半连接会让注意力静默少一项，与"忘了给当前 token"是同一类静默错误。
TEST(PagedAttentionPluginTest, RejectsHalfConnectedCurrentToken) {
    nvinfer1::DynamicPluginTensorDesc inputs[6];
    inputs[0].desc.dims = nvinfer1::Dims{4, {1, 4, 1, 8}};
    inputs[1].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[2].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
    inputs[3].desc.dims = nvinfer1::Dims{2, {1, 4}};
    inputs[4].desc.dims = nvinfer1::Dims{1, {1}};
    inputs[5].desc.dims = nvinfer1::Dims{4, {1, 2, 1, 8}};

    PagedAttentionPlugin plugin(4, 2, 8, 16, 0.0f);
    EXPECT_NE(plugin.configurePlugin(inputs, 6, nullptr, 1), 0);
}

// 七个输入全部连上必须接受，并校验当前 token 的形状
TEST(PagedAttentionPluginTest, AcceptsCurrentTokenInputsAndValidatesShape) {
    const auto make_inputs = [](nvinfer1::DynamicPluginTensorDesc* inputs,
                                int32_t key_new_seq) {
        inputs[0].desc.dims = nvinfer1::Dims{4, {1, 4, 1, 8}};
        inputs[1].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
        inputs[2].desc.dims = nvinfer1::Dims{4, {16, 16, 2, 8}};
        inputs[3].desc.dims = nvinfer1::Dims{2, {1, 4}};
        inputs[4].desc.dims = nvinfer1::Dims{1, {1}};
        inputs[5].desc.dims = nvinfer1::Dims{4, {1, 2, key_new_seq, 8}};
        inputs[6].desc.dims = nvinfer1::Dims{4, {1, 2, key_new_seq, 8}};
    };

    nvinfer1::DynamicPluginTensorDesc inputs[7];
    make_inputs(inputs, /*key_new_seq=*/1);
    PagedAttentionPlugin plugin(4, 2, 8, 16, 0.0f);
    EXPECT_EQ(plugin.configurePlugin(inputs, 7, nullptr, 1), 0);

    // seq 维不是 1 → 与 decoding 契约冲突，必须拒绝
    make_inputs(inputs, /*key_new_seq=*/3);
    PagedAttentionPlugin bad(4, 2, 8, 16, 0.0f);
    EXPECT_NE(bad.configurePlugin(inputs, 7, nullptr, 1), 0);
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
