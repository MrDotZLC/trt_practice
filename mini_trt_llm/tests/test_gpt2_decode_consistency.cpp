#include "e2e_fixture.hpp"
#include "logger.hpp"
#include "mini_trt_llm/core/builder.hpp"
#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/core/gpt2_model_builder.hpp"
#include "mini_trt_llm/core/llm_runner_kernel.hpp"
#include "mini_trt_llm/kv_cache/paged_kv_cache.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"
#include "gpt2_test_support.hpp"
#include "test_reference.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace {

// 小模型夹具来自共享头（唯一来源，见 gpt2_small_model.hpp 的说明）
using test_support::kBlockSize;
using test_support::kBlocksPerSeq;
using test_support::kHeadSize;
using test_support::kHeads;
using test_support::kHidden;
using test_support::kLayers;
using test_support::kMultiKeyTol;
using test_support::kPositions;
using test_support::kSingleKeyTol;
using test_support::kSmallGpt2PromptTokens;
using test_support::kVocab;
using test_support::SmallGpt2BuilderConfig;
using test_support::SmallGpt2ConfigJson;
using test_support::SmallGpt2Weights;
using test_support::ComputeDiffStats;
using test_support::DiffStats;
using test_support::ReadFloats;
using test_support::WithinAbs;

}  // namespace

// decode 一步 === prefill 在对应位置的结果。
//
// **这是本阶段最有价值的一条判据**：两条路径的注意力实现完全不同
// （prefill 是手搭的 MatMul+mask+Softmax 子图，decode 走 PagedAttention 的
// online softmax），却能对同一个数学式给出同样结果——分页布局、上下文长度、
// 位置编码、当前 token 是否参与注意力，任何一处写错都会在这里露馅。
// 且它**不依赖外部基线**，因此不受"参考数据是否可靠"的影响。
TEST(Gpt2DecodeConsistencyTest, DecodeStepMatchesPrefillAtSamePosition) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    Logger logger;
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_decode_consistency");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(SmallGpt2ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(SmallGpt2Weights()));

    EngineBuilder builder(logger, SmallGpt2BuilderConfig());
    const std::string prefill_path = directory.EnginePath("prefill.engine");
    const std::string decode_path = directory.EnginePath("decode.engine");
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), prefill_path,
                                        BuildStage::kPrefill))
        << "prefill engine build failed";
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), decode_path,
                                        BuildStage::kDecode))
        << "decode engine build failed";

    Engine prefill(prefill_path, logger);
    Engine decode(decode_path, logger);

    // ---- 参考：用完整 prompt 跑一次 prefill，取最后一个位置的 logits ----
    std::vector<int32_t> prompt_tokens(kSmallGpt2PromptTokens);
    std::vector<int32_t> prompt_positions(kSmallGpt2PromptTokens);
    for (int32_t i = 0; i < kSmallGpt2PromptTokens; ++i) {
        prompt_tokens[i] = i + 3;  // 任意但确定的 token id（< vocab）
        prompt_positions[i] = i;
    }
    DeviceBuffer d_prompt_tokens(kSmallGpt2PromptTokens * sizeof(int32_t));
    DeviceBuffer d_prompt_positions(kSmallGpt2PromptTokens * sizeof(int32_t));
    const size_t prefill_logits_count =
        static_cast<size_t>(kSmallGpt2PromptTokens) * kVocab;
    DeviceBuffer d_prefill_logits(prefill_logits_count * sizeof(float));
    ASSERT_TRUE(d_prompt_tokens.Allocate(kSmallGpt2PromptTokens * sizeof(int32_t)));
    ASSERT_TRUE(d_prompt_positions.Allocate(kSmallGpt2PromptTokens * sizeof(int32_t)));
    ASSERT_TRUE(d_prefill_logits.Allocate(prefill_logits_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_prompt_tokens.data(), prompt_tokens.data(),
                          d_prompt_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prompt_positions.data(), prompt_positions.data(),
                          d_prompt_positions.size(), cudaMemcpyHostToDevice));

    ASSERT_TRUE(prefill.SetOptimizationProfile(0, nullptr));
    ASSERT_TRUE(prefill.SetInputShape("input_ids",
                                      nvinfer1::Dims{2, {1, kSmallGpt2PromptTokens}}));
    ASSERT_TRUE(prefill.SetInputShape("position_ids",
                                      nvinfer1::Dims{2, {1, kSmallGpt2PromptTokens}}));
    ASSERT_TRUE(prefill.SetTensorAddress("input_ids", d_prompt_tokens.data()));
    ASSERT_TRUE(prefill.SetTensorAddress("position_ids", d_prompt_positions.data()));
    ASSERT_TRUE(prefill.SetTensorAddress("logits", d_prefill_logits.data()));

    // prefill 的每层 K/V 输出（本身也要绑定，否则 enqueue 会失败）
    const size_t kv_count = static_cast<size_t>(kHeads) * kSmallGpt2PromptTokens * kHeadSize;
    std::vector<std::unique_ptr<DeviceBuffer>> d_prefill_kv;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        for (const char* tag : {"k_layer", "v_layer"}) {
            auto buffer = std::make_unique<DeviceBuffer>();
            ASSERT_TRUE(buffer->Allocate(kv_count * sizeof(float)));
            ASSERT_TRUE(prefill.SetTensorAddress(
                (std::string(tag) + std::to_string(layer)).c_str(), buffer->data()));
            d_prefill_kv.push_back(std::move(buffer));
        }
    }
    ASSERT_TRUE(prefill.Enqueue(nullptr));
    prefill.Synchronize(nullptr);
    const std::vector<float> ref_logits = ReadFloats(d_prefill_logits.data(),
                                                     prefill_logits_count);

    // ---- 被测：cache 里放 prompt 的前 kSmallGpt2PromptTokens-1 个 token，再 decode 最后一个 ----
    PagedKVCache::Config cache_config;
    cache_config.num_blocks = 8;
    cache_config.block_size = kBlockSize;
    cache_config.num_layers = kLayers;
    cache_config.num_kv_heads = kHeads;
    cache_config.head_size = kHeadSize;
    cache_config.is_half = false;
    cache_config.max_blocks_per_seq = kBlocksPerSeq;
    PagedKVCache cache(cache_config);
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/0, /*max_tokens=*/kSmallGpt2PromptTokens));
    CUDA_CHECK(cache.UploadMetadata(nullptr));

    // 用 prefill 引擎再跑一次"短 prompt"（前 3 个 token），把它的 K/V 写进 cache
    constexpr int32_t kCachedTokens = kSmallGpt2PromptTokens - 1;
    DeviceBuffer d_short_tokens(kCachedTokens * sizeof(int32_t));
    DeviceBuffer d_short_positions(kCachedTokens * sizeof(int32_t));
    const size_t short_logits_count = static_cast<size_t>(kCachedTokens) * kVocab;
    DeviceBuffer d_short_logits(short_logits_count * sizeof(float));
    ASSERT_TRUE(d_short_tokens.Allocate(kCachedTokens * sizeof(int32_t)));
    ASSERT_TRUE(d_short_positions.Allocate(kCachedTokens * sizeof(int32_t)));
    ASSERT_TRUE(d_short_logits.Allocate(short_logits_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_short_tokens.data(), prompt_tokens.data(),
                          d_short_tokens.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_short_positions.data(), prompt_positions.data(),
                          d_short_positions.size(), cudaMemcpyHostToDevice));

    // 顺序很关键：先选 profile 再设形状。反过来（先设形状后选 profile）时
    // TRT 可能按 profile 的 opt 形状覆盖掉显式设置，短 prompt 那趟就会按 4 个 token
    // 执行——写进 cache 的布局假设随之失效，而失败现象只会是"数值差一点"。
    ASSERT_TRUE(prefill.SetOptimizationProfile(0, nullptr));
    ASSERT_TRUE(prefill.SetInputShape("input_ids",
                                      nvinfer1::Dims{2, {1, kCachedTokens}}));
    ASSERT_TRUE(prefill.SetInputShape("position_ids",
                                      nvinfer1::Dims{2, {1, kCachedTokens}}));
    ASSERT_TRUE(prefill.SetTensorAddress("input_ids", d_short_tokens.data()));
    ASSERT_TRUE(prefill.SetTensorAddress("position_ids", d_short_positions.data()));
    ASSERT_TRUE(prefill.SetTensorAddress("logits", d_short_logits.data()));

    // 短 prompt 这一趟必须用**另一组** K/V 缓冲：
    // d_prefill_kv 里存的还是 4-token 那趟的结果，用来当参考；
    // 若复用同一组，3-token 的输出只会覆盖前 24 个 float，
    // 稍后按"第 4 个位置"去读就会读到错位的数据（本用例最初就是这样比错的）。
    std::vector<std::unique_ptr<DeviceBuffer>> d_cache_kv;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        for (const char* tag : {"k_layer", "v_layer"}) {
            auto buffer = std::make_unique<DeviceBuffer>();
            ASSERT_TRUE(buffer->Allocate(kv_count * sizeof(float)));
            ASSERT_TRUE(prefill.SetTensorAddress(
                (std::string(tag) + std::to_string(layer)).c_str(), buffer->data()));
            d_cache_kv.push_back(std::move(buffer));
        }
    }
    ASSERT_TRUE(prefill.Enqueue(nullptr));
    prefill.Synchronize(nullptr);

    // 把每层的 K/V 写进分页 cache（这是 runner 里 prefill 之后的正常动作）
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const void* k = d_cache_kv[static_cast<size_t>(layer) * 2]->data();
        const void* v = d_cache_kv[static_cast<size_t>(layer) * 2 + 1]->data();
        ASSERT_EQ(cache.WritePrefillKV(layer, k, v, kCachedTokens, nullptr),
                  cudaSuccess);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- decode 一步：喂最后一个 prompt token ----
    DeviceBuffer d_decode_token(sizeof(int32_t));
    DeviceBuffer d_decode_position(sizeof(int32_t));
    DeviceBuffer d_decode_logits(static_cast<size_t>(kVocab) * sizeof(float));
    ASSERT_TRUE(d_decode_token.Allocate(sizeof(int32_t)));
    ASSERT_TRUE(d_decode_position.Allocate(sizeof(int32_t)));
    ASSERT_TRUE(d_decode_logits.Allocate(static_cast<size_t>(kVocab) * sizeof(float)));
    const int32_t last_token = prompt_tokens[kSmallGpt2PromptTokens - 1];
    const int32_t last_position = prompt_positions[kSmallGpt2PromptTokens - 1];
    CUDA_CHECK(cudaMemcpy(d_decode_token.data(), &last_token, sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_decode_position.data(), &last_position, sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    ASSERT_TRUE(decode.SetOptimizationProfile(0, nullptr));
    ASSERT_TRUE(decode.SetInputShape("input_ids", nvinfer1::Dims{2, {1, 1}}));
    ASSERT_TRUE(decode.SetInputShape("position_ids", nvinfer1::Dims{2, {1, 1}}));
    ASSERT_TRUE(decode.SetInputShape("block_tables",
                                     nvinfer1::Dims{2, {1, kBlocksPerSeq}}));
    ASSERT_TRUE(decode.SetInputShape("context_lens", nvinfer1::Dims{1, {1}}));
    ASSERT_TRUE(decode.SetTensorAddress("input_ids", d_decode_token.data()));
    ASSERT_TRUE(decode.SetTensorAddress("position_ids", d_decode_position.data()));
    ASSERT_TRUE(decode.SetTensorAddress("block_tables",
                                       const_cast<int32_t*>(cache.block_tables())));
    ASSERT_TRUE(decode.SetTensorAddress("context_lens",
                                       const_cast<int32_t*>(cache.context_lens())));
    // 每层一对 cache 输入：第 L 层绑第 L 段的地址。
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const std::string index = std::to_string(layer);
        ASSERT_TRUE(decode.SetTensorAddress(("key_cache_" + index).c_str(),
                                            cache.key_cache(layer)));
        ASSERT_TRUE(decode.SetTensorAddress(("value_cache_" + index).c_str(),
                                            cache.value_cache(layer)));
    }
    ASSERT_TRUE(decode.SetTensorAddress("logits", d_decode_logits.data()));

    const size_t decode_kv_count = static_cast<size_t>(kHeads) * kHeadSize;
    std::vector<std::unique_ptr<DeviceBuffer>> d_decode_kv;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        for (const char* tag : {"k_layer", "v_layer"}) {
            auto buffer = std::make_unique<DeviceBuffer>();
            ASSERT_TRUE(buffer->Allocate(decode_kv_count * sizeof(float)));
            ASSERT_TRUE(decode.SetTensorAddress(
                (std::string(tag) + std::to_string(layer)).c_str(), buffer->data()));
            d_decode_kv.push_back(std::move(buffer));
        }
    }
    ASSERT_TRUE(decode.Enqueue(nullptr));
    decode.Synchronize(nullptr);

    // decode 的 logits 必须等于 prefill 在最后一个位置上的 logits
    const std::vector<float> decode_logits = ReadFloats(d_decode_logits.data(), kVocab);
    const float* ref_last = ref_logits.data() + static_cast<size_t>(kSmallGpt2PromptTokens - 1) * kVocab;
    const std::vector<float> ref_last_vec(ref_last, ref_last + kVocab);
    const DiffStats logits_diff = ComputeDiffStats(ref_last_vec, decode_logits);

    // 顺带核对：decode 输出的当前 token K/V 应当等于 prefill 在同一位置的 K/V。
    // 这一条能分开"注意力算错"与"K/V 本身算错"。
    // layer0 的 K/V 只经过嵌入 + ln_1 + c_attn（**不含注意力**），
    // layer1 的 K/V 才经过了 layer0 的注意力输出——两者的差异分开看，
    // 就能判断误差是"注意力算错"还是"被放大的累积误差"。
    DiffStats kv_diff[kLayers];
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const std::vector<float> decode_k =
            ReadFloats(d_decode_kv[static_cast<size_t>(layer) * 2]->data(), decode_kv_count);
        // decode 的布局 [heads, 1, head_size]，prefill 的布局 [heads, seq, head_size]；
        // 逐个 head 与 prefill 的最后一个位置比。
        const float* prefill_k =
            static_cast<const float*>(d_prefill_kv[static_cast<size_t>(layer) * 2]->data());
        const std::vector<float> prefill_kv_all =
            ReadFloats(prefill_k, static_cast<size_t>(kHeads) * kSmallGpt2PromptTokens * kHeadSize);
        std::vector<float> expected_k(decode_k.size());
        for (int32_t h = 0; h < kHeads; ++h) {
            for (int32_t d = 0; d < kHeadSize; ++d) {
                expected_k[static_cast<size_t>(h) * kHeadSize + d] =
                    prefill_kv_all[(static_cast<size_t>(h) * kSmallGpt2PromptTokens +
                                    (kSmallGpt2PromptTokens - 1)) *
                                       kHeadSize +
                                   d];
            }
        }
        kv_diff[layer] = ComputeDiffStats(expected_k, decode_k);
    }

    // 诊断汇总：一次跑就能看出误差分布，而不是只知道"第 V 个 logit 差了"。
    std::cout << "[诊断] layer0 K/V  max_abs=" << kv_diff[0].max_abs
              << " max_rel=" << kv_diff[0].max_rel << "（只含嵌入+投影，不含注意力）\n"
              << "[诊断] layer1 K/V  max_abs=" << kv_diff[1].max_abs
              << " max_rel=" << kv_diff[1].max_rel << "（经过 layer0 的注意力）\n"
              << "[诊断] logits      max_abs=" << logits_diff.max_abs
              << " max_rel=" << logits_diff.max_rel << "\n";

    // layer0 不经过注意力：两条路径走到这里必须几乎逐位相同。
    // 若这条不成立，问题根本不在注意力实现上，而是在权重读取或图层结构。
    EXPECT_LT(kv_diff[0].max_abs, kSingleKeyTol)
        << "layer0 的 K/V 不一致 → 问题不在注意力，而在嵌入/投影这一段";
    // 多 key 的 logits 差落在"算法差异"档：阈值由实测确定（D6），
    // 真正的语义判据是下面这条——贪心 token 必须一致。
    EXPECT_LT(logits_diff.max_abs, kMultiKeyTol)
        << "decode 与 prefill 的 logits 差异超出算法差异档";
    int32_t ref_argmax = 0;
    int32_t decode_argmax = 0;
    for (int32_t v = 1; v < kVocab; ++v) {
        if (ref_last_vec[v] > ref_last_vec[ref_argmax]) {
            ref_argmax = v;
        }
        if (decode_logits[v] > decode_logits[decode_argmax]) {
            decode_argmax = v;
        }
    }
    EXPECT_EQ(ref_argmax, decode_argmax)
        << "两条路径的贪心 token 不同：prefill=" << ref_argmax
        << " decode=" << decode_argmax;
}

// 单 key 隔离：把"注意力实现差异"和"cache / 长度 / 位置编码"彻底分开。
//
// 构造 context_lens = 0 的 decode（cache 为空，只有当前 token 参与），
// 与只喂一个 token 的 prefill 对比。此时 softmax 只有一个元素、权重恒为 1，
// 两条路径做的是**同一件几乎没有算法自由度的事**，因此可以卡到 1e-5。
//
// 判读方式：
//   - 这条通过、多 token 那条差 1e-3 → 差异来自累积/算法，不是接线错误；
//   - 这条就挂了 → 问题在 cache 为空时的路径、位置编码或当前 token 的处理上。
TEST(Gpt2DecodeConsistencyTest, DecodeWithEmptyCacheMatchesSingleTokenPrefill) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    Logger logger;
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_decode_single");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(SmallGpt2ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(SmallGpt2Weights()));

    EngineBuilder builder(logger, SmallGpt2BuilderConfig());
    const std::string prefill_path = directory.EnginePath("prefill.engine");
    const std::string decode_path = directory.EnginePath("decode.engine");
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), prefill_path,
                                        BuildStage::kPrefill));
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), decode_path,
                                        BuildStage::kDecode));
    Engine prefill(prefill_path, logger);
    Engine decode(decode_path, logger);

    const int32_t token = 7;
    const int32_t position = 2;  // 非 0，确保位置编码真的被用上
    DeviceBuffer d_token(sizeof(int32_t));
    DeviceBuffer d_position(sizeof(int32_t));
    DeviceBuffer d_prefill_logits(static_cast<size_t>(kVocab) * sizeof(float));
    DeviceBuffer d_decode_logits(static_cast<size_t>(kVocab) * sizeof(float));
    ASSERT_TRUE(d_token.Allocate(sizeof(int32_t)));
    ASSERT_TRUE(d_position.Allocate(sizeof(int32_t)));
    ASSERT_TRUE(d_prefill_logits.Allocate(static_cast<size_t>(kVocab) * sizeof(float)));
    ASSERT_TRUE(d_decode_logits.Allocate(static_cast<size_t>(kVocab) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_token.data(), &token, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_position.data(), &position, sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // prefill：序列长度 1
    const size_t kv_count = static_cast<size_t>(kHeads) * kHeadSize;
    std::vector<std::unique_ptr<DeviceBuffer>> d_prefill_kv;
    ASSERT_TRUE(prefill.SetOptimizationProfile(0, nullptr));
    ASSERT_TRUE(prefill.SetInputShape("input_ids", nvinfer1::Dims{2, {1, 1}}));
    ASSERT_TRUE(prefill.SetInputShape("position_ids", nvinfer1::Dims{2, {1, 1}}));
    ASSERT_TRUE(prefill.SetTensorAddress("input_ids", d_token.data()));
    ASSERT_TRUE(prefill.SetTensorAddress("position_ids", d_position.data()));
    ASSERT_TRUE(prefill.SetTensorAddress("logits", d_prefill_logits.data()));
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        for (const char* tag : {"k_layer", "v_layer"}) {
            auto buffer = std::make_unique<DeviceBuffer>();
            ASSERT_TRUE(buffer->Allocate(kv_count * sizeof(float)));
            ASSERT_TRUE(prefill.SetTensorAddress(
                (std::string(tag) + std::to_string(layer)).c_str(), buffer->data()));
            d_prefill_kv.push_back(std::move(buffer));
        }
    }
    ASSERT_TRUE(prefill.Enqueue(nullptr));
    prefill.Synchronize(nullptr);

    // decode：cache 空（context_lens = 0），只带当前 token
    PagedKVCache::Config cache_config;
    cache_config.num_blocks = 8;
    cache_config.block_size = kBlockSize;
    cache_config.num_layers = kLayers;
    cache_config.num_kv_heads = kHeads;
    cache_config.head_size = kHeadSize;
    cache_config.is_half = false;
    cache_config.max_blocks_per_seq = kBlocksPerSeq;
    PagedKVCache cache(cache_config);
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/0, /*max_tokens=*/kSmallGpt2PromptTokens));
    CUDA_CHECK(cache.UploadMetadata(nullptr));
    // 语境长度保持 0：不写 prefill K/V，让 cache 为空

    std::vector<std::unique_ptr<DeviceBuffer>> d_decode_kv;
    ASSERT_TRUE(decode.SetOptimizationProfile(0, nullptr));
    ASSERT_TRUE(decode.SetInputShape("input_ids", nvinfer1::Dims{2, {1, 1}}));
    ASSERT_TRUE(decode.SetInputShape("position_ids", nvinfer1::Dims{2, {1, 1}}));
    ASSERT_TRUE(decode.SetInputShape("block_tables",
                                     nvinfer1::Dims{2, {1, kBlocksPerSeq}}));
    ASSERT_TRUE(decode.SetInputShape("context_lens", nvinfer1::Dims{1, {1}}));
    ASSERT_TRUE(decode.SetTensorAddress("input_ids", d_token.data()));
    ASSERT_TRUE(decode.SetTensorAddress("position_ids", d_position.data()));
    ASSERT_TRUE(decode.SetTensorAddress("block_tables",
                                       const_cast<int32_t*>(cache.block_tables())));
    ASSERT_TRUE(decode.SetTensorAddress("context_lens",
                                       const_cast<int32_t*>(cache.context_lens())));
    // 每层一对 cache 输入：第 L 层绑第 L 段的地址。
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const std::string index = std::to_string(layer);
        ASSERT_TRUE(decode.SetTensorAddress(("key_cache_" + index).c_str(),
                                            cache.key_cache(layer)));
        ASSERT_TRUE(decode.SetTensorAddress(("value_cache_" + index).c_str(),
                                            cache.value_cache(layer)));
    }
    ASSERT_TRUE(decode.SetTensorAddress("logits", d_decode_logits.data()));
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        for (const char* tag : {"k_layer", "v_layer"}) {
            auto buffer = std::make_unique<DeviceBuffer>();
            ASSERT_TRUE(buffer->Allocate(kv_count * sizeof(float)));
            ASSERT_TRUE(decode.SetTensorAddress(
                (std::string(tag) + std::to_string(layer)).c_str(), buffer->data()));
            d_decode_kv.push_back(std::move(buffer));
        }
    }
    ASSERT_TRUE(decode.Enqueue(nullptr));
    decode.Synchronize(nullptr);

    const std::vector<float> prefill_logits =
        ReadFloats(d_prefill_logits.data(), kVocab);
    const std::vector<float> decode_logits = ReadFloats(d_decode_logits.data(), kVocab);
    const DiffStats logits_diff = ComputeDiffStats(prefill_logits, decode_logits);
    std::cout << "[诊断] 单 token / 空 cache 的 logits  max_abs=" << logits_diff.max_abs
              << " max_rel=" << logits_diff.max_rel << "\n";
    EXPECT_LT(logits_diff.max_abs, kSingleKeyTol)
        << "只有一个 key 时两条路径应当几乎逐位相同；不成立说明接线（cache / "
           "context_lens / position_ids / 当前 token）有问题";

    // layer0 的当前 token K/V 完全不含注意力，必须一致
    const std::vector<float> decode_k0 = ReadFloats(d_decode_kv[0]->data(), kv_count);
    const std::vector<float> prefill_k0 = ReadFloats(d_prefill_kv[0]->data(), kv_count);
    EXPECT_LT(ComputeDiffStats(prefill_k0, decode_k0).max_abs, kSingleKeyTol);
}


// 两步 decode 对拍：定位"decode 输出的 K/V → 追加进 cache → 再读回"这条链。
//
// **为什么需要它**：`LLMRunner` 的生成在第 3 个新 token 分叉（TROUBLESHOOTING #16），
// 而那一步正是第一次读到"由 decode 引擎自己产出、再经 AppendDecodeKV 追加"的 K/V。
// 单步对拍（本文件第一条用例）与 PagedKVCache 的机制测试都通过，说明每一半都对；
// 这条用例把两半串起来，并把误差拆成三个量：
//   (a) 第 1 步 logits —— 只读 prefill 写入的 cache，应当 ~1e-7；
//   (b) 被追加的那份 K/V —— 直接量"追加的数据本身对不对"；
//   (c) 第 2 步 logits —— 第一次读回追加数据，若这里显著变大就坐实了这条链。
TEST(Gpt2DecodeConsistencyTest, TwoStepDecodeMatchesPrefillAfterAppend) {
    if (!test_support::HasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    Logger logger;
    test_support::ModelDirectory directory =
        test_support::ModelDirectory::Create("gpt2_two_step");
    ASSERT_TRUE(directory.valid());
    ASSERT_TRUE(directory.WriteConfig(SmallGpt2ConfigJson()));
    ASSERT_TRUE(directory.WriteWeights(SmallGpt2Weights()));

    EngineBuilder builder(logger, SmallGpt2BuilderConfig());
    const std::string prefill_path = directory.EnginePath("prefill.engine");
    const std::string decode_path = directory.EnginePath("decode.engine");
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), prefill_path, BuildStage::kPrefill));
    ASSERT_TRUE(builder.BuildFromConfig(directory.path(), decode_path, BuildStage::kDecode));
    Engine prefill(prefill_path, logger);
    Engine decode(decode_path, logger);

    const std::vector<int32_t> prompt = {3, 4, 5, 6};
    // 每层 K/V 的缓冲：长度按"最长的一次 prefill"取（6 个 token）
    constexpr int32_t kMaxSeq = 6;
    const size_t kv_elems = static_cast<size_t>(kHeads) * kMaxSeq * kHeadSize;
    std::vector<std::unique_ptr<DeviceBuffer>> kv(static_cast<size_t>(kLayers) * 2);
    for (auto& buffer : kv) {
        buffer = std::make_unique<DeviceBuffer>();
        ASSERT_TRUE(buffer->Allocate(kv_elems * sizeof(float)));
    }
    DeviceBuffer d_tokens(static_cast<size_t>(kMaxSeq) * sizeof(int32_t));
    DeviceBuffer d_positions(static_cast<size_t>(kMaxSeq) * sizeof(int32_t));
    DeviceBuffer d_prefill_logits(static_cast<size_t>(kMaxSeq) * kVocab * sizeof(float));
    DeviceBuffer d_decode_logits(static_cast<size_t>(kVocab) * sizeof(float));

    // 参考路径逐步累积：声明必须放在下面的 lambda 之前（lambda 体在定义点编译）。
    std::vector<int32_t> produced;
    std::vector<std::vector<float>> ref_logits;
    std::vector<std::vector<std::vector<float>>> ref_kv;

    // 用指定长度的前缀跑一次 prefill，返回最后一行 logits 与每层 K/V 的副本。
    const auto run_prefill = [&](int32_t length,
                                 std::vector<float>* last_logits,
                                 std::vector<std::vector<float>>* kv_copy) {
        std::vector<int32_t> tokens(static_cast<size_t>(length));
        std::vector<int32_t> positions(static_cast<size_t>(length));
        for (int32_t i = 0; i < length; ++i) {
            tokens[static_cast<size_t>(i)] = i < static_cast<int32_t>(prompt.size())
                                                 ? prompt[static_cast<size_t>(i)]
                                                 : produced[static_cast<size_t>(i - prompt.size())];
            positions[static_cast<size_t>(i)] = i;
        }
        const int32_t seq = length;
        if (!prefill.SetOptimizationProfile(0, nullptr) ||
            !prefill.SetInputShape("input_ids", nvinfer1::Dims{2, {1, seq}}) ||
            !prefill.SetInputShape("position_ids", nvinfer1::Dims{2, {1, seq}})) {
            return false;
        }
        CUDA_CHECK(cudaMemcpy(d_tokens.data(), tokens.data(), tokens.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_positions.data(), positions.data(),
                              positions.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        if (!prefill.SetTensorAddress("input_ids", d_tokens.data()) ||
            !prefill.SetTensorAddress("position_ids", d_positions.data()) ||
            !prefill.SetTensorAddress("logits", d_prefill_logits.data())) {
            return false;
        }
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            const std::string index = std::to_string(layer);
            if (!prefill.SetTensorAddress(("k_layer" + index).c_str(),
                                          kv[static_cast<size_t>(layer) * 2]->data()) ||
                !prefill.SetTensorAddress(("v_layer" + index).c_str(),
                                          kv[static_cast<size_t>(layer) * 2 + 1]->data())) {
                return false;
            }
        }
        if (!prefill.Enqueue(nullptr)) {
            return false;
        }
        prefill.Synchronize(nullptr);
        const float* row = reinterpret_cast<const float*>(
            static_cast<const char*>(d_prefill_logits.data()) +
            static_cast<size_t>(length - 1) * kVocab * sizeof(float));
        *last_logits = ReadFloats(row, kVocab);
        kv_copy->clear();
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            // 只读该次 prefill 的有效区间：[heads, length, head_size]
            kv_copy->push_back(ReadFloats(
                kv[static_cast<size_t>(layer) * 2]->data(),
                static_cast<size_t>(kHeads) * static_cast<size_t>(length) * kHeadSize));
        }
        return true;
    };

    // ---- 参考路径：逐步 prefill，得到贪心 token 与各步 logits ----
    for (int32_t step = 0; step < 3; ++step) {
        std::vector<float> logits;
        std::vector<std::vector<float>> kv_snapshot;
        ASSERT_TRUE(run_prefill(static_cast<int32_t>(prompt.size()) + step, &logits,
                                &kv_snapshot));
        ref_logits.push_back(logits);
        ref_kv.push_back(kv_snapshot);
        produced.push_back(static_cast<int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()))));
    }
    const int32_t t0 = produced[0];
    const int32_t t1 = produced[1];

    // ---- cache 路径：prefill(4) → 写 cache → decode(t0) → 追加 → decode(t1) ----
    PagedKVCache::Config cache_config;
    cache_config.num_blocks = 8;
    cache_config.block_size = kBlockSize;
    cache_config.num_layers = kLayers;
    cache_config.num_kv_heads = kHeads;
    cache_config.head_size = kHeadSize;
    cache_config.is_half = false;
    cache_config.max_blocks_per_seq = kBlocksPerSeq;
    PagedKVCache cache(cache_config);
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(0, kMaxSeq));
    CUDA_CHECK(cache.UploadMetadata(nullptr));

    std::vector<float> ignored_logits;
    std::vector<std::vector<float>> prompt_kv;
    ASSERT_TRUE(run_prefill(static_cast<int32_t>(prompt.size()), &ignored_logits, &prompt_kv));
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        ASSERT_EQ(cache.WritePrefillKV(layer, kv[static_cast<size_t>(layer) * 2]->data(),
                                      kv[static_cast<size_t>(layer) * 2 + 1]->data(),
                                      static_cast<int32_t>(prompt.size()), nullptr),
                  cudaSuccess);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // decode 两次；每次把当前 token 的 K/V 留下副本后追加进 cache
    std::vector<std::vector<float>> decoded_kv;
    const auto run_decode = [&](int32_t token, std::vector<float>* logits_out) {
        DeviceBuffer d_token(sizeof(int32_t));
        DeviceBuffer d_position(sizeof(int32_t));
        if (!d_token.Allocate(sizeof(int32_t)) || !d_position.Allocate(sizeof(int32_t))) {
            return false;
        }
        CUDA_CHECK(cudaMemcpy(d_token.data(), &token, sizeof(int32_t), cudaMemcpyHostToDevice));
        if (!decode.SetOptimizationProfile(0, nullptr) ||
            !decode.SetInputShape("input_ids", nvinfer1::Dims{2, {1, 1}}) ||
            !decode.SetInputShape("position_ids", nvinfer1::Dims{2, {1, 1}}) ||
            !decode.SetInputShape("block_tables",
                                  nvinfer1::Dims{2, {1, kBlocksPerSeq}}) ||
            !decode.SetInputShape("context_lens", nvinfer1::Dims{1, {1}})) {
            return false;
        }
        if (!decode.SetTensorAddress("input_ids", d_token.data()) ||
            !decode.SetTensorAddress("position_ids", d_position.data()) ||
            !decode.SetTensorAddress("block_tables",
                                     const_cast<int32_t*>(cache.block_tables())) ||
            !decode.SetTensorAddress("context_lens",
                                     const_cast<int32_t*>(cache.context_lens())) ||
            !decode.SetTensorAddress("logits", d_decode_logits.data())) {
            return false;
        }
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            const std::string index = std::to_string(layer);
            if (!decode.SetTensorAddress(("key_cache_" + index).c_str(),
                                         cache.key_cache(layer)) ||
                !decode.SetTensorAddress(("value_cache_" + index).c_str(),
                                         cache.value_cache(layer)) ||
                !decode.SetTensorAddress(("k_layer" + index).c_str(),
                                         kv[static_cast<size_t>(layer) * 2]->data()) ||
                !decode.SetTensorAddress(("v_layer" + index).c_str(),
                                         kv[static_cast<size_t>(layer) * 2 + 1]->data())) {
                return false;
            }
        }
        // position_ids 由设备端 context_lens 填（与 runner 完全一致）
        if (LaunchFillPositionIds(cache.context_lens(),
                                  static_cast<int32_t*>(d_position.data()), 1,
                                  nullptr) != cudaSuccess) {
            return false;
        }
        if (!decode.Enqueue(nullptr)) {
            return false;
        }
        decode.Synchronize(nullptr);
        *logits_out = ReadFloats(d_decode_logits.data(), kVocab);
        // 留下当前 token 的 K/V 副本（就是接下来要被追加的那份数据）
        decoded_kv.clear();
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            decoded_kv.push_back(ReadFloats(
                kv[static_cast<size_t>(layer) * 2]->data(),
                static_cast<size_t>(kHeads) * kMaxSeq * kHeadSize));
        }
        std::vector<const void*> keys(static_cast<size_t>(kLayers));
        std::vector<const void*> values(static_cast<size_t>(kLayers));
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            keys[static_cast<size_t>(layer)] = kv[static_cast<size_t>(layer) * 2]->data();
            values[static_cast<size_t>(layer)] = kv[static_cast<size_t>(layer) * 2 + 1]->data();
        }
        if (cache.AppendDecodeStep(keys, values, nullptr) != cudaSuccess) {
            return false;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return true;
    };

    std::vector<float> step1_logits;
    ASSERT_TRUE(run_decode(t0, &step1_logits));
    const std::vector<std::vector<float>> appended_kv = decoded_kv;  // t0 的 K/V
    std::vector<float> step2_logits;
    ASSERT_TRUE(run_decode(t1, &step2_logits));

    // (a) 第 1 步 logits：decode(t0) vs 5-token prefill 的最后一行
    const DiffStats step1 = ComputeDiffStats(ref_logits[1], step1_logits);
    // (c) 第 2 步 logits：decode(t1) vs 6-token prefill 的最后一行
    const DiffStats step2 = ComputeDiffStats(ref_logits[2], step2_logits);
    // (b) 追加数据的正确性：t0 的 K/V vs 5-token prefill 在位置 4 的 K/V
    //     prefill 布局是 [heads, seq, head], decode 布局是 [heads, 1, head]
    float appended_kv_diff = 0.0f;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        std::vector<float> expected;
        for (int32_t h = 0; h < kHeads; ++h) {
            for (int32_t d = 0; d < kHeadSize; ++d) {
                // ref_kv[1] 是"prompt + t0"（5 个 token）那次 prefill 的快照，
                // 布局是 [heads, 5, head]；步长必须用 5，用 kMaxSeq 会读到错位数据
                // （这正是 §2.14 C 条记的"诊断比错对象"）。
                const size_t seq = prompt.size() + 1;
                expected.push_back(
                    ref_kv[1][static_cast<size_t>(layer)]
                           [(static_cast<size_t>(h) * seq + static_cast<size_t>(prompt.size())) *
                                kHeadSize +
                            static_cast<size_t>(d)]);
            }
        }
        std::vector<float> actual(appended_kv[static_cast<size_t>(layer)].begin(),
                                 appended_kv[static_cast<size_t>(layer)].begin() +
                                     static_cast<size_t>(kHeads) * kHeadSize);
        appended_kv_diff =
            std::max(appended_kv_diff, ComputeDiffStats(expected, actual).max_abs);
    }

    std::cout << "[诊断] 两步 decode 对拍\n"
              << "        (a) 第 1 步 logits max_abs = " << step1.max_abs << "\n"
              << "        (b) 追加的 K/V  max_abs = " << appended_kv_diff << "\n"
              << "        (c) 第 2 步 logits max_abs = " << step2.max_abs << "\n";

    // (b) 与 (a) 都应当接近 FP32 噪声；若它们干净而 (c) 明显变大，
    // 说明账不在"追加的数据"上，而在"追加之后被读回"这一段。
    EXPECT_LT(appended_kv_diff, kSingleKeyTol)
        << "被追加的 K/V 与参考不一致：追加的数据本身就是错的";
    EXPECT_LT(step1.max_abs, kSingleKeyTol) << "第 1 步 decode 就不一致";
    EXPECT_LT(step2.max_abs, kSingleKeyTol)
        << "第 2 步（第一次读回追加数据）不一致";
}

}  // namespace mini_trt_llm
