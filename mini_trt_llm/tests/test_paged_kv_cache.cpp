#include "mini_trt_llm/kv_cache/block_allocator.hpp"
#include "mini_trt_llm/kv_cache/paged_kv_cache.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"
#include "test_gpu_guard.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {
namespace {

constexpr int32_t kBlockSize = 4;
constexpr int32_t kKvHeads = 2;
constexpr int32_t kHeadSize = 4;
constexpr int32_t kNumBlocks = 8;
constexpr int32_t kMaxBlocksPerSeq = 3;  // 每序列最多 12 个 token

PagedKVCache::Config MakeConfig(bool is_half = false) {
    PagedKVCache::Config config;
    config.num_blocks = kNumBlocks;
    config.block_size = kBlockSize;
    config.num_layers = 2;
    config.num_kv_heads = kKvHeads;
    config.head_size = kHeadSize;
    config.is_half = is_half;
    config.max_blocks_per_seq = kMaxBlocksPerSeq;
    return config;
}

// 构造可辨识的 K/V：同一 (batch, head, token, dim) 在 K 与 V 上取不同值，
// 便于区分"写错了张量"与"写错了位置"。
std::vector<float> MakeKV(int32_t batch, int32_t tokens, float base) {
    const size_t count = static_cast<size_t>(batch) * kKvHeads * tokens * kHeadSize;
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = base + static_cast<float>(i);
    }
    return data;
}

// 把某个层的整段 cache 读回主机。
std::vector<float> ReadBack(const void* device_ptr, size_t floats) {
    std::vector<float> host(floats);
    CUDA_CHECK(cudaMemcpy(host.data(), device_ptr, floats * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return host;
}

// 逻辑位置 (b, h, t, d) 在分页 cache 里的偏移。
// 这里刻意按"块表 + 块内槽位"独立算一遍，不复用产品代码的算法。
size_t ExpectedOffset(const std::vector<int32_t>& table, int32_t b, int32_t t, int32_t h,
                      int32_t d) {
    const int32_t physical = table[static_cast<size_t>(b) * kMaxBlocksPerSeq + t / kBlockSize];
    const int32_t slot = t % kBlockSize;
    return ((static_cast<size_t>(physical) * kBlockSize + slot) * kKvHeads + h) * kHeadSize +
           d;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host 侧：块分配器（不需要 GPU）
// ---------------------------------------------------------------------------

TEST(BlockAllocatorTest, AllocatesFreesAndReuses) {
    BlockAllocator allocator(3);
    EXPECT_EQ(allocator.NumFree(), 3u);

    const int first = allocator.Allocate();
    const int second = allocator.Allocate();
    EXPECT_NE(first, second);
    EXPECT_EQ(allocator.NumFree(), 1u);

    allocator.Free(first);
    EXPECT_EQ(allocator.NumFree(), 2u);
    // 释放过的块必须能回到池子里被重新分配出去，否则池子会缓慢漏空。
    // 这里只断言"没有重复发放、释放的块确实回来了"，**不指定**分配顺序——
    // 当前策略是 FIFO（刚释放的块排在队尾），把顺序写进用例等于把实现细节钉死。
    const int third = allocator.Allocate();
    const int fourth = allocator.Allocate();
    EXPECT_EQ(allocator.NumFree(), 0u);
    std::vector<int> handed_out{first, second, third, fourth};
    std::sort(handed_out.begin(), handed_out.end());
    EXPECT_EQ(handed_out, (std::vector<int>{0, 0, 1, 2}));
}

TEST(BlockAllocatorTest, RejectsExhaustionAndDoubleFree) {
    BlockAllocator allocator(1);
    EXPECT_EQ(allocator.Allocate(), 0);
    EXPECT_THROW(allocator.Allocate(), std::runtime_error);
    // 双重释放会让同一个物理块被两个序列同时使用 → 静默串数据
    allocator.Free(0);
    EXPECT_THROW(allocator.Free(0), std::runtime_error);
    EXPECT_THROW(allocator.Free(42), std::runtime_error);
}

// ---------------------------------------------------------------------------
// GPU 侧：分页写入路径
// ---------------------------------------------------------------------------

// prefill 写入必须走块表，而不是按顺序码放。
//
// 用例设计：先让另一个序列吃掉物理块 0/1，于是本序列的块表不是从 0 开始的连续块。
// 如果 kernel 忽略块表、顺序写，第 2 个 batch 的期望偏移会与实测完全错开。
TEST(PagedKVCacheTest, PrefillWritesThroughBlockTable) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PagedKVCache cache(MakeConfig());
    ASSERT_TRUE(cache.valid());

    // 第一个序列先占用两个块，把物理编号往后推
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/7, /*max_tokens=*/8));
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/3, /*max_tokens=*/12));
    ASSERT_EQ(cache.batch_size(), 2);
    ASSERT_EQ(cache.sequence_order()[0], 7);
    ASSERT_EQ(cache.sequence_order()[1], 3);
    CUDA_CHECK(cache.UploadMetadata(nullptr));

    constexpr int32_t kTokens = 7;  // 跨块：4 + 3
    const std::vector<float> key = MakeKV(2, kTokens, 100.0f);
    const std::vector<float> value = MakeKV(2, kTokens, 900.0f);

    DeviceBuffer d_key(key.size() * sizeof(float));
    DeviceBuffer d_value(value.size() * sizeof(float));
    ASSERT_TRUE(d_key.Allocate(key.size() * sizeof(float)));
    ASSERT_TRUE(d_value.Allocate(value.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_key.data(), key.data(), d_key.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value.data(), value.data(), d_value.size(),
                          cudaMemcpyHostToDevice));

    // 显式把第 1 层清零，而不是指望 cudaMalloc 给的显存是干净的
    // （CUDA 并不保证这一点）。这样"第 1 层没被动过"才是可复现的断言：
    // 一旦 kernel 越界写到第 1 层，读回来就是非 0 的 K 值。
    CUDA_CHECK(cudaMemset(cache.key_cache(1), 0, cache.bytes_per_layer()));

    ASSERT_EQ(cache.WritePrefillKV(/*layer=*/0, d_key.data(), d_value.data(), kTokens,
                                   /*stream=*/nullptr),
              cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 期望的块表：seq 7 → 物理块 {0,1}；seq 3 → {2,3,4}
    std::vector<int32_t> table;
    ASSERT_TRUE(cache.GetBlockTable(7, &table));
    ASSERT_EQ(table.size(), 2u);
    std::vector<int32_t> flat_table(kMaxBlocksPerSeq * 2, 0);
    for (size_t i = 0; i < table.size(); ++i) {
        flat_table[i] = table[i];
    }
    ASSERT_TRUE(cache.GetBlockTable(3, &table));
    for (size_t i = 0; i < table.size(); ++i) {
        flat_table[kMaxBlocksPerSeq + i] = table[i];
    }

    // 每层一段：读单层必须用 bytes_per_layer，用整缓冲大小会读穿到分配外，
    // cudaMemcpy 会直接报 invalid argument。
    const size_t layer_floats = cache.bytes_per_layer() / sizeof(float);
    const std::vector<float> key_cache = ReadBack(cache.key_cache(0), layer_floats);

    for (int32_t b = 0; b < 2; ++b) {
        for (int32_t t = 0; t < kTokens; ++t) {
            for (int32_t h = 0; h < kKvHeads; ++h) {
                for (int32_t d = 0; d < kHeadSize; ++d) {
                    const size_t source =
                        ((static_cast<size_t>(b) * kKvHeads + h) * kTokens + t) * kHeadSize + d;
                    const size_t offset = ExpectedOffset(flat_table, b, t, h, d);
                    EXPECT_FLOAT_EQ(key_cache[offset], key[source])
                        << "b=" << b << " t=" << t << " h=" << h << " d=" << d;
                }
            }
        }
    }

    // 第二个层段必须原封不动（每层各占一段，写第一层不能碰第二层）
    const std::vector<float> second_layer =
        ReadBack(cache.key_cache(1), layer_floats);
    for (float v : second_layer) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }

    // 写入完成后 host 侧语境长度应为 tokens
    EXPECT_EQ(cache.SequenceLength(7), kTokens);
    EXPECT_EQ(cache.SequenceLength(3), kTokens);
}

// decode 追加：位置由设备端 context_lens 决定，写完后 context_lens 自增；
// 跨越块边界（第 4 个 token 落到下一个块的第 0 个槽位）必须正确。
TEST(PagedKVCacheTest, AppendCrossesBlockBoundaryAndAdvancesContextLens) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PagedKVCache cache(MakeConfig());
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/0, /*max_tokens=*/8));
    CUDA_CHECK(cache.UploadMetadata(nullptr));

    constexpr int32_t kPrefillTokens = 3;  // 块内前 3 个槽位
    const std::vector<float> prefill_key = MakeKV(1, kPrefillTokens, 10.0f);
    const std::vector<float> prefill_value = MakeKV(1, kPrefillTokens, 20.0f);
    DeviceBuffer d_prefill_key(prefill_key.size() * sizeof(float));
    DeviceBuffer d_prefill_value(prefill_value.size() * sizeof(float));
    ASSERT_TRUE(d_prefill_key.Allocate(prefill_key.size() * sizeof(float)));
    ASSERT_TRUE(d_prefill_value.Allocate(prefill_value.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_prefill_key.data(), prefill_key.data(), d_prefill_key.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prefill_value.data(), prefill_value.data(),
                          d_prefill_value.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cache.WritePrefillKV(0, d_prefill_key.data(), d_prefill_value.data(),
                                   kPrefillTokens, nullptr),
              cudaSuccess);
    // WritePrefillKV 必须自己把长度推到设备：decode 追加的位置取自设备端
    // context_lens，漏掉这一步追加会写回位置 0、静默覆盖第一个 token。
    {
        int32_t device_len = -1;
        CUDA_CHECK(cudaMemcpy(&device_len, cache.context_lens(), sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
        EXPECT_EQ(device_len, kPrefillTokens);
    }

    // 两次追加：第 4 个 token 填满第 0 块，第 5 个落到第 1 块。
    //
    // **必须每层传一对**：AppendDecodeStep 的契约是"一次写全部层、只推进一次长度"
    // （TROUBLESHOOTING #16），少传会被直接拒绝。两层的值刻意取不同基址（layer 0 = 100 段、
    // layer 1 = 200 段），这样"某层写进了别人那一段"这类错误（#15 的形态）才会被读回断言抓住。
    constexpr int32_t kLayers = 2;
    constexpr float kLayerBase[kLayers] = {100.0f, 200.0f};
    for (int32_t step = 0; step < 2; ++step) {
        std::vector<DeviceBuffer> d_key(kLayers);
        std::vector<DeviceBuffer> d_value(kLayers);
        std::vector<const void*> key_ptrs(kLayers);
        std::vector<const void*> value_ptrs(kLayers);
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            const float base = kLayerBase[layer] + static_cast<float>(step);
            std::vector<float> step_kv(kKvHeads * kHeadSize);
            for (size_t i = 0; i < step_kv.size(); ++i) {
                step_kv[i] = base + static_cast<float>(i);
            }
            ASSERT_TRUE(d_key[layer].Allocate(step_kv.size() * sizeof(float)));
            ASSERT_TRUE(d_value[layer].Allocate(step_kv.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_key[layer].data(), step_kv.data(), d_key[layer].size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_value[layer].data(), step_kv.data(), d_value[layer].size(),
                                  cudaMemcpyHostToDevice));
            key_ptrs[layer] = d_key[layer].data();
            value_ptrs[layer] = d_value[layer].data();
        }
        ASSERT_EQ(cache.AppendDecodeStep(key_ptrs, value_ptrs, nullptr), cudaSuccess);
        CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_EQ(cache.SequenceLength(0), kPrefillTokens + step + 1);
    }

    // 设备端 context_lens 必须与 host 记账一致（否则引擎会用错长度）
    int32_t device_len = -1;
    CUDA_CHECK(cudaMemcpy(&device_len, cache.context_lens(), sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(device_len, kPrefillTokens + 2);

    // 追加的数据必须落在逻辑位置 3 与 4（跨块），**两层都要验**：
    // 只读 layer 0 会漏掉"每层写自己那一段"这个语义——#15 正是这样漏过去的。
    std::vector<int32_t> table;
    ASSERT_TRUE(cache.GetBlockTable(0, &table));
    std::vector<int32_t> flat_table(kMaxBlocksPerSeq, 0);
    for (size_t i = 0; i < table.size(); ++i) {
        flat_table[i] = table[i];
    }
    const size_t layer_floats = cache.bytes_per_layer() / sizeof(float);
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const std::vector<float> key_cache = ReadBack(cache.key_cache(layer), layer_floats);
        for (int32_t h = 0; h < kKvHeads; ++h) {
            for (int32_t d = 0; d < kHeadSize; ++d) {
                const float idx = static_cast<float>(h * kHeadSize + d);
                // 第 1 次追加（step=0）落在位置 3，第 2 次（step=1）落在位置 4
                EXPECT_FLOAT_EQ(key_cache[ExpectedOffset(flat_table, 0, 3, h, d)],
                                kLayerBase[layer] + idx);
                EXPECT_FLOAT_EQ(key_cache[ExpectedOffset(flat_table, 0, 4, h, d)],
                                kLayerBase[layer] + 1.0f + idx);
            }
        }
    }
}

// 负例：层数不匹配必须被拒绝，而且**不许留下任何副作用**。
//
// 为什么值得单独一条：拒绝路径若先做了部分写入或推进了长度，一次非法调用就会污染后续推理，
// 表现为"莫名其妙整体错位"，而这种半成品状态极难从下游现象反推回源头。
// 当前实现是先校验、后写入，这条用例把那顺序钉住（把它改成先写后校验就会红）。
TEST(PagedKVCacheTest, AppendDecodeStepRejectsLayerCountMismatch) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PagedKVCache cache(MakeConfig());  // num_layers = 2
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/0, /*max_tokens=*/8));
    CUDA_CHECK(cache.UploadMetadata(nullptr));

    const std::vector<float> one_layer = MakeKV(/*batch=*/1, /*tokens=*/1, 300.0f);
    DeviceBuffer d_key(one_layer.size() * sizeof(float));
    DeviceBuffer d_value(one_layer.size() * sizeof(float));
    ASSERT_TRUE(d_key.Allocate(one_layer.size() * sizeof(float)));
    ASSERT_TRUE(d_value.Allocate(one_layer.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_key.data(), one_layer.data(), d_key.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value.data(), one_layer.data(), d_value.size(),
                          cudaMemcpyHostToDevice));

    // 只传 1 对而配置要求 2 对 → 必须返回错误码，而不是"写一半"
    EXPECT_EQ(cache.AppendDecodeStep({d_key.data()}, {d_value.data()}, nullptr),
              cudaErrorInvalidValue);
    // host 侧记账与设备端 context_lens 都必须停在 0
    EXPECT_EQ(cache.SequenceLength(0), 0);
    int32_t device_len = -1;
    CUDA_CHECK(cudaMemcpy(&device_len, cache.context_lens(), sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(device_len, 0);
}

// 超过预留长度的 prefill 必须被拒绝：块表里未预留的位置在 host 镜像里是 0，
// 越界写入会静默写进物理块 0（通常是别的序列的数据）。
TEST(PagedKVCacheTest, RejectsPrefillBeyondReservedTokens) {
    MINI_TRT_SKIP_IF_NO_CUDA();
    PagedKVCache cache(MakeConfig());
    ASSERT_TRUE(cache.valid());
    ASSERT_TRUE(cache.AllocateSequence(/*seq_id=*/0, /*max_tokens=*/5));
    CUDA_CHECK(cache.UploadMetadata(nullptr));

    const std::vector<float> data(static_cast<size_t>(kKvHeads) * 7 * kHeadSize, 1.0f);
    DeviceBuffer buffer(data.size() * sizeof(float));
    ASSERT_TRUE(buffer.Allocate(data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(buffer.data(), data.data(), buffer.size(), cudaMemcpyHostToDevice));

    // 预留 5 个 token（2 块），写 7 个 → 必须失败
    EXPECT_NE(cache.WritePrefillKV(0, buffer.data(), buffer.data(), 7, nullptr),
              cudaSuccess);
    // 预留范围内的写入应当成功
    EXPECT_EQ(cache.WritePrefillKV(0, buffer.data(), buffer.data(), 5, nullptr),
              cudaSuccess);
}

}  // namespace mini_trt_llm
