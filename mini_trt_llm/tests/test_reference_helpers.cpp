#include "test_reference.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mini_trt_llm {

// 参考实现自身的验证（meta-test）。
//
// **为什么需要这一层**：参考实现是判断被测实现对不对的唯一标尺，它自己错了，
// 测试就会给出错误的裁决。曾有一版 ReferenceRoPE 漏了 batch 维度，调用方在 batch>1 时
// 传入 batch*seq_len 个位置、函数却只用了前 seq_len 个——编译通过、运行通过，
// 直到真机数值对不上才暴露（见 docs/TROUBLESHOOTING.md #9）。
//
// 这些用例全部是纯 host 计算，**不需要 GPU**，因此可以进 CI：
// 参考实现在 CI 里被验证，GPU 侧才只负责"参考 vs 实现"的对比。
// 用例刻意选可手算规模，断言精确值而不是"看起来合理"。

TEST(ReferenceHelpersTest, RopeIsBatchAware) {
    // batch=2 / heads=1 / seq=1 / head_size=2 / rotary_dim=2 / base=10000
    // j=0 时 inv_freq = base^0 = 1，角度就等于 position，可以手算。
    const std::vector<double> input{1.0, 0.0, 1.0, 0.0};  // 两个 batch 的输入相同
    const std::vector<int32_t> positions{0, 1};
    std::vector<double> output;
    test_support::ReferenceRoPE(input, positions, /*batch=*/2, /*heads=*/1, /*seq_len=*/1,
                                /*head_size=*/2, /*rotary_dim=*/2, 10000.0, &output);
    ASSERT_EQ(output.size(), 4u);

    // position = 0 → 角度为 0 → 旋转是恒等变换
    EXPECT_DOUBLE_EQ(output[0], 1.0);
    EXPECT_DOUBLE_EQ(output[1], 0.0);
    // position = 1 → 角度为 1 rad → {cos1, sin1}
    // 旧实现漏了 batch，这里会得到恒等结果，从而被这条用例拦下
    EXPECT_NEAR(output[2], std::cos(1.0), 1e-12);
    EXPECT_NEAR(output[3], std::sin(1.0), 1e-12);
}

TEST(ReferenceHelpersTest, RopeRejectsPositionsShapeMismatch) {
    // 传 4 个位置但声明 batch=1/seq=2：契约不满足必须立刻失败，
    // 而不是静默只读前两个——这正是当初掩盖问题的原因
    const std::vector<double> input(4, 1.0);
    const std::vector<int32_t> positions{1, 2, 3, 4};
    std::vector<double> output;
    EXPECT_THROW(test_support::ReferenceRoPE(input, positions, /*batch=*/1, /*heads=*/1,
                                             /*seq_len=*/2, /*head_size=*/2,
                                             /*rotary_dim=*/2, 10000.0, &output),
                 std::invalid_argument);
}

TEST(ReferenceHelpersTest, RopeRejectsInvalidRotaryDim) {
    const std::vector<double> input{1.0, 2.0};
    const std::vector<int32_t> positions{0};
    std::vector<double> output;
    // 奇数 rotary_dim 无法成对旋转
    EXPECT_THROW(test_support::ReferenceRoPE(input, positions, 1, 1, 1, 2, 1, 10000.0,
                                             &output),
                 std::invalid_argument);
}

TEST(ReferenceHelpersTest, RmsNormMatchesHandComputedValues) {
    // hidden=2, rows=1, eps=0, weight={1,1}
    // mean(x^2) = (9 + 16)/2 = 12.5 → rms = sqrt(12.5)
    const std::vector<double> input{3.0, 4.0};
    const std::vector<double> weight{1.0, 1.0};
    std::vector<double> output;
    test_support::ReferenceRmsNorm(input, weight, /*rows=*/1, /*hidden=*/2, /*eps=*/0.0,
                                   &output);
    ASSERT_EQ(output.size(), 2u);
    const double rms = std::sqrt(12.5);
    EXPECT_NEAR(output[0], 3.0 / rms, 1e-12);
    EXPECT_NEAR(output[1], 4.0 / rms, 1e-12);
}

TEST(ReferenceHelpersTest, PagedAttentionWithSingleTokenReturnsValue) {
    // context_len = 1 时 softmax 只有一个元素，权重必然为 1 → 输出应精确等于该位置的 V
    constexpr int32_t kBlockSize = 4;
    constexpr int32_t kHeadSize = 2;
    std::vector<double> query{0.5, -0.25};
    std::vector<double> key_cache(static_cast<size_t>(kBlockSize) * kHeadSize, 0.0);
    std::vector<double> value_cache(static_cast<size_t>(kBlockSize) * kHeadSize, 0.0);
    key_cache[0] = 1.0;
    key_cache[1] = 2.0;
    value_cache[0] = 7.0;
    value_cache[1] = 9.0;
    const std::vector<int32_t> block_table{0};

    std::vector<double> output;
    test_support::ReferencePagedAttentionDecode(query, key_cache, value_cache, block_table,
                                                /*context_len=*/1, /*num_heads=*/1,
                                                /*num_kv_heads=*/1, kHeadSize, kBlockSize,
                                                /*scale=*/0.5, &output);
    ASSERT_EQ(output.size(), 2u);
    EXPECT_DOUBLE_EQ(output[0], 7.0);
    EXPECT_DOUBLE_EQ(output[1], 9.0);
}

TEST(ReferenceHelpersTest, PagedAttentionSharesKvHeadUnderGqa) {
    // num_heads=2 / num_kv_heads=1：两个 query head 共享同一个 kv head。
    // 两个 query 相同 → 两个输出必须完全相同，否则说明 kv_head 的映射写错了。
    constexpr int32_t kHeadSize = 2;
    constexpr int32_t kBlockSize = 4;
    const std::vector<double> query{0.5, -0.25, 0.5, -0.25};
    std::vector<double> key_cache(static_cast<size_t>(kBlockSize) * kHeadSize);
    std::vector<double> value_cache(static_cast<size_t>(kBlockSize) * kHeadSize);
    for (size_t i = 0; i < key_cache.size(); ++i) {
        key_cache[i] = 0.1 * static_cast<double>(i + 1);
        value_cache[i] = -0.2 * static_cast<double>(i + 1);
    }
    const std::vector<int32_t> block_table{0};

    std::vector<double> output;
    test_support::ReferencePagedAttentionDecode(query, key_cache, value_cache, block_table,
                                                /*context_len=*/3, /*num_heads=*/2,
                                                /*num_kv_heads=*/1, kHeadSize, kBlockSize,
                                                /*scale=*/1.0, &output);
    ASSERT_EQ(output.size(), 4u);
    EXPECT_DOUBLE_EQ(output[0], output[2]);
    EXPECT_DOUBLE_EQ(output[1], output[3]);
}

TEST(ReferenceHelpersTest, PagedAttentionRejectsIndivisibleHeads) {
    const std::vector<double> query(4, 0.0);
    const std::vector<double> cache(8, 0.0);
    const std::vector<int32_t> block_table{0};
    std::vector<double> output;
    EXPECT_THROW(test_support::ReferencePagedAttentionDecode(
                     query, cache, cache, block_table, 1, /*num_heads=*/3,
                     /*num_kv_heads=*/2, 2, 4, 1.0, &output),
                 std::invalid_argument);
}

// 配置预检：复刻 GPU 用例（Fp16PathTest.RoPEHandlesFp16WithGqaAndMultipleBatches）的
// 维度配置，确认参考实现接受它。
//
// **为什么需要**：那次真机失败的根因就是"用例的维度配置"与"参考实现的契约"不一致，
// 而这类不一致在 host 侧完全可以提前发现。把配置预检放进 CI，就不必再花一次真机往返。
TEST(ReferenceHelpersTest, RopeAcceptsFp16TestConfiguration) {
    constexpr int32_t kBatch = 2;
    constexpr int32_t kHeads = 4;
    constexpr int32_t kKvHeads = 2;
    constexpr int32_t kSeq = 2;
    constexpr int32_t kHeadSize = 8;
    constexpr int32_t kRotaryDim = 4;

    const std::vector<int32_t> positions{2, 5, 7, 11};
    const std::vector<double> query(static_cast<size_t>(kBatch) * kHeads * kSeq * kHeadSize,
                                    0.5);
    const std::vector<double> key(static_cast<size_t>(kBatch) * kKvHeads * kSeq * kHeadSize,
                                  0.25);
    std::vector<double> query_out;
    std::vector<double> key_out;

    // 两个调用都必须接受该形状；任一处抛异常就说明用例配置与参考契约不一致
    EXPECT_NO_THROW(test_support::ReferenceRoPE(query, positions, kBatch, kHeads, kSeq,
                                                kHeadSize, kRotaryDim, 10000.0, &query_out));
    EXPECT_NO_THROW(test_support::ReferenceRoPE(key, positions, kBatch, kKvHeads, kSeq,
                                                kHeadSize, kRotaryDim, 10000.0, &key_out));

    // 位置非连续且各 batch 不同，输出必须逐 batch 区分开——
    // 漏 batch 的实现会在这里露馅
    EXPECT_NE(query_out[0], query_out[static_cast<size_t>(kHeads) * kSeq * kHeadSize]);
}

// LayerNorm：与手算结果对齐，并且把"eps 加在方差上"这条钉死。
TEST(ReferenceHelpersTest, LayerNormMatchesHandComputedValues) {
    // 一行 [1, 2, 3, 4]：均值 2.5，方差 1.25
    const std::vector<double> input{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> scale{1.0, 1.0, 1.0, 1.0};
    const std::vector<double> bias{0.0, 0.0, 0.0, 0.0};
    std::vector<double> output;
    test_support::ReferenceLayerNorm(input, scale, bias, 1, 4, 0.0, &output);

    const double inv_std = 1.0 / std::sqrt(1.25);
    const double expected[] = {-1.5 * inv_std, -0.5 * inv_std, 0.5 * inv_std, 1.5 * inv_std};
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-12) << "index " << i;
    }

    // eps 若被错加到标准差上（sqrt(var)+eps），这两组输入会给出不同的比值
    std::vector<double> with_eps;
    test_support::ReferenceLayerNorm(input, scale, bias, 1, 4, 1e-3, &with_eps);
    EXPECT_NE(output[0], with_eps[0]);
    EXPECT_NEAR(with_eps[0], -1.5 / std::sqrt(1.25 + 1e-3), 1e-12);

    // scale / bias 必须逐通道生效，而不是整体缩放
    const std::vector<double> scale2{2.0, 1.0, 1.0, 1.0};
    const std::vector<double> bias2{0.0, 5.0, 0.0, 0.0};
    std::vector<double> output2;
    test_support::ReferenceLayerNorm(input, scale2, bias2, 1, 4, 0.0, &output2);
    EXPECT_NEAR(output2[0], -1.5 * inv_std * 2.0, 1e-12);
    EXPECT_NEAR(output2[1], -0.5 * inv_std + 5.0, 1e-12);
}

TEST(ReferenceHelpersTest, LayerNormRejectsShapeMismatch) {
    const std::vector<double> input(8, 1.0);
    const std::vector<double> scale(4, 1.0);
    const std::vector<double> bias(3, 0.0);
    std::vector<double> output;
    EXPECT_THROW(
        test_support::ReferenceLayerNorm(input, scale, bias, 2, 4, 1e-5, &output),
        std::invalid_argument);
    EXPECT_THROW(
        test_support::ReferenceLayerNorm(input, scale, scale, 3, 4, 1e-5, &output),
        std::invalid_argument);
}

// gelu_new：与 kGELU_TANH 同公式，取几个可手算/可交叉验证的点。
TEST(ReferenceHelpersTest, GeluNewMatchesKnownValues) {
    const std::vector<double> input{-3.0, -1.0, 0.0, 1.0, 3.0};
    std::vector<double> output;
    test_support::ReferenceGeluNew(input, &output);

    // x = 0 处严格为 0；正负对称位置必须单调且异号（tanh 近似的定性特征）
    EXPECT_DOUBLE_EQ(output[2], 0.0);
    EXPECT_LT(output[1], 0.0);
    EXPECT_GT(output[3], 0.0);
    EXPECT_LT(output[1], output[0]);  // 负区间：越深越小
    EXPECT_LT(output[3], output[4]);  // 正区间：越深越大

    // 与标准 gelu(erf) 在 |x|<=1 内的差异应小于 1e-3（tanh 近似本身的精度）
    for (int i = 0; i < 5; ++i) {
        const double x = input[i];
        const double erf_gelu = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        EXPECT_NEAR(output[i], erf_gelu, 1e-3) << "x=" << x;
    }

    // 大正值处趋近恒等（x=3 时 gelu_new(3)=2.9964，偏差 3.6e-3 是该近似本身的固有误差）
    EXPECT_NEAR(output[4], 3.0, 5e-3);
}

}  // namespace mini_trt_llm
