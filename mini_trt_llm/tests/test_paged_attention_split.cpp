// decode 阶段 PagedAttention 的 split-K **纯逻辑**用例（H 层，沙箱可跑）。
//
// 开发计划：`docs/future_iterations_development_plan.md` + OI-FLASHDECODING-PLAN（P2_2-3）
// 测试计划：`docs/future_iterations_test_plan.md` + OI-FLASHDECODING-TESTS（PS-1 ~ PS-7）
//
// 为什么这些用例值得单独存在：分片边界（不整除、片数超过位置数、只有当前 token）
// 与 workspace 布局都是**纯逻辑**，而它们出错的表现是"某些长度下结果错/NaN"或
// "越界写"——前者在真机上极难归因，后者只在真机暴露（`TROUBLESHOOTING` + TS-018 / TS-024）。
// 能在沙箱裁掉的，就不该留给真机（`PROGRESS.md` §2.13）。

#include "mini_trt_llm/plugins/paged_attention_split.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace mini_trt_llm {
namespace {

// PS-1：各片首尾相接、并集恰好覆盖 [0, total_len)。
//
// 判据形式是"逐段断言"而不是"只看片数"：片数对而边界错，会让某些位置被扫两次、
// 另一些位置一次都不扫——后者表现为输出偏小，前者表现为权重被重复计入。
TEST(PagedAttentionSplitPlanTest, CoversRangeWithoutGapOrOverlap) {
    const std::vector<int32_t> lengths = {1, 2, 7, 127, 128, 129, 255, 976, 1024, 4096};
    for (int32_t total_len : lengths) {
        const int32_t effective = PagedAttentionResolveSplits(total_len, 0);
        ASSERT_GE(effective, 1) << "total_len=" << total_len;
        int32_t cursor = 0;
        for (int32_t s = 0; s < effective; ++s) {
            int32_t begin = -1;
            int32_t end = -1;
            PagedAttentionSplitRange(total_len, s, effective, &begin, &end);
            EXPECT_EQ(begin, cursor) << "total_len=" << total_len << " split=" << s;
            EXPECT_GT(end, begin) << "total_len=" << total_len << " split=" << s;
            cursor = end;
        }
        EXPECT_EQ(cursor, total_len) << "total_len=" << total_len;
    }
}

// PS-2：不整除时切法**确定**（余数摊给前几片），且末段 end == total_len。
//
// "确定"是可复现性的前提：同样的输入必须切出同样的区间，否则归并顺序随版本/平台变化，
// 结果会飘（`AGENTS.md` §7）。这里把期望切法写死，等于把规则钉在用例里。
TEST(PagedAttentionSplitPlanTest, HandlesNonDivisibleLength) {
    const int32_t total_len = 100;
    const int32_t effective = 3;  // 100 = 33 + 33 + 34（余数 1 摊给第 0 片）
    const std::vector<std::pair<int32_t, int32_t>> expected = {{0, 34}, {34, 67}, {67, 100}};
    for (size_t s = 0; s < expected.size(); ++s) {
        int32_t begin = -1;
        int32_t end = -1;
        PagedAttentionSplitRange(total_len, static_cast<int32_t>(s), effective, &begin, &end);
        EXPECT_EQ(begin, expected[s].first) << "split=" << s;
        EXPECT_EQ(end, expected[s].second) << "split=" << s;
    }
}

// PS-3：片数超过位置数时，超出的片必须是**空片**（begin == end），而不是负长度或越界区间。
// 空片是合法的输入，但调用方（kernel）必须显式写哨兵，不能让它读未初始化显存。
TEST(PagedAttentionSplitPlanTest, EmptyChunksWhenSplitsExceedLength) {
    const int32_t total_len = 3;
    const int32_t effective = 8;  // 强制（测试开关才会这么用）
    const std::vector<std::pair<int32_t, int32_t>> expected = {
        {0, 1}, {1, 2}, {2, 3}, {3, 3}, {3, 3}, {3, 3}, {3, 3}, {3, 3}};
    for (size_t s = 0; s < expected.size(); ++s) {
        int32_t begin = -1;
        int32_t end = -1;
        PagedAttentionSplitRange(total_len, static_cast<int32_t>(s), effective, &begin, &end);
        EXPECT_EQ(begin, expected[s].first) << "split=" << s;
        EXPECT_EQ(end, expected[s].second) << "split=" << s;
    }
}

// PS-4：`context_len == 0` 且带当前 token 时，整段注意力只有 1 个位置。
// 这条对上了 kernel 侧的最强判据（`CurrentTokenIsAttendedEvenWithEmptyCache`：
// 输出必须逐元素等于 value_new）——分片层先把"确实只有 1 个位置"钉住。
TEST(PagedAttentionSplitPlanTest, SinglePositionWithEmptyCache) {
    EXPECT_EQ(PagedAttentionResolveSplits(0, 0), 0);  // 没有位置
    EXPECT_EQ(PagedAttentionResolveSplits(1, 0), 1);  // 只有当前 token

    int32_t begin = -1;
    int32_t end = -1;
    PagedAttentionSplitRange(1, 0, 1, &begin, &end);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 1);

    // 没有位置（effective == 0）时必须返回空片，而不是 [0, 1) 这种"凭空多一个位置"
    PagedAttentionSplitRange(0, 0, 0, &begin, &end);
    EXPECT_EQ(begin, end);
}

// PS-5：片数选择策略的边界——目标片长两侧各测一次。
//
// 这条锁的是 F2=A 的口径（`kTargetChunk = 128`）：短上下文落在 1 片，长上下文按需增片。
// 边界写错（例如写成 `>` 而不是 `>=`）会让 128 位置这一档多切一片，收益与代价都变。
TEST(PagedAttentionSplitPlanTest, ChoosesSplitsByTargetChunk) {
    EXPECT_EQ(PagedAttentionResolveSplits(1, 0), 1);
    EXPECT_EQ(PagedAttentionResolveSplits(kPagedAttentionTargetChunk - 1, 0), 1);
    EXPECT_EQ(PagedAttentionResolveSplits(kPagedAttentionTargetChunk, 0), 1);
    EXPECT_EQ(PagedAttentionResolveSplits(kPagedAttentionTargetChunk + 1, 0), 2);
    EXPECT_EQ(PagedAttentionResolveSplits(2 * kPagedAttentionTargetChunk, 0), 2);
    EXPECT_EQ(PagedAttentionResolveSplits(2 * kPagedAttentionTargetChunk + 1, 0), 3);
}

// PS-6：片数上限就是 workspace 契约的上界，任何路径都不能越过去。
//
// 为什么连 override 也要钳：测试开关如果能把片数抬到上限之上，而 workspace 是按上限预留的，
// 那么"用例通过"本身就会越界写——把护栏变成事故源。
TEST(PagedAttentionSplitPlanTest, ClampsAtMaxSplits) {
    EXPECT_EQ(PagedAttentionResolveSplits(1 << 20, 0), kPagedAttentionMaxSplits);
    EXPECT_EQ(PagedAttentionResolveSplits(1 << 20, kPagedAttentionMaxSplits + 5),
              kPagedAttentionMaxSplits);
    EXPECT_EQ(PagedAttentionResolveSplits(1 << 20, 3), 3);
    // 没有位置时无论 override 多大都不该开片
    EXPECT_EQ(PagedAttentionResolveSplits(0, 4), 0);
    EXPECT_EQ(PagedAttentionResolveSplits(-1, 4), 0);
}

// PS-7：workspace 的字节数必须覆盖 kernel 实际会写的最大偏移。
//
// 这条是 D3 契约的**可执行形式**：`getWorkspaceSize()` 与 kernel 共用
// `PagedAttentionWorkspaceSlotOffset` / `PagedAttentionWorkspaceBytes`，所以只要
// "布局函数本身自洽 + 字节数 = 最大偏移 + stride"，两边就不会错位。
// 判据里逐 (split, batch, head) 走一遍，而不是只算最后一个槽位——错位的典型形态是
// "batch 与 head 的乘序写反"，那种错在最后一槽上恰好也可能对上。
TEST(PagedAttentionSplitPlanTest, WorkspaceBytesCoverUsedRange) {
    const std::vector<int32_t> head_sizes = {8, 16, 64, 80};
    const std::vector<int32_t> batches = {1, 4};
    const std::vector<int32_t> head_counts = {1, 12};
    const std::vector<int32_t> splits_list = {1, kPagedAttentionMaxSplits};

    for (int32_t batch : batches) {
        for (int32_t heads : head_counts) {
            for (int32_t head_size : head_sizes) {
                for (int32_t splits : splits_list) {
                    const size_t bytes =
                        PagedAttentionWorkspaceBytes(batch, heads, head_size, splits);
                    ASSERT_GT(bytes, 0u);
                    const size_t floats = bytes / sizeof(float);
                    ASSERT_EQ(bytes % sizeof(float), 0u);
                    const size_t stride =
                        static_cast<size_t>(PagedAttentionWorkspaceStride(head_size));
                    EXPECT_EQ(floats, static_cast<size_t>(batch) * heads * splits * stride);
                    for (int32_t s = 0; s < splits; ++s) {
                        for (int32_t b = 0; b < batch; ++b) {
                            for (int32_t h = 0; h < heads; ++h) {
                                const size_t offset = PagedAttentionWorkspaceSlotOffset(
                                    s, b, h, batch, heads, head_size);
                                EXPECT_LT(offset + stride, floats + 1)
                                    << "split=" << s << " batch=" << b << " head=" << h;
                            }
                        }
                    }
                }
            }
        }
    }

    // 非法入参返回 0（调用方据此放弃 split 路径，而不是"按 0 字节预留后又写"）
    EXPECT_EQ(PagedAttentionWorkspaceBytes(0, 12, 64, 8), 0u);
    EXPECT_EQ(PagedAttentionWorkspaceBytes(1, 0, 64, 8), 0u);
    EXPECT_EQ(PagedAttentionWorkspaceBytes(1, 12, 0, 8), 0u);
    EXPECT_EQ(PagedAttentionWorkspaceBytes(1, 12, 64, 0), 0u);
}

// PS-8：两阶段分解的**数学自洽**（纯 host，不依赖 CUDA / 分页）。
//
// 为什么值得单独写：kernel 在沙箱里跑不了，但"把一段位置切成多片 → 各片算局部 (m, l, acc)
// → 按 max-trick 归并"这套**算法**完全可以在这里裁决。它挡的是"归并公式写错 / 余数摊错 /
// 空片没跳过"这类**所有长度都静默偏一点**的错——那种错只在长上下文才明显，靠真机归因极慢。
//
// 裁决对象是**直接 softmax**（独立参考），不是"另一份分片实现"：两份同源实现只会一起错
// （`PROGRESS.md` §2.13）。
TEST(PagedAttentionSplitPlanTest, SplitMergeDecompositionMatchesDirectSoftmax) {
    const auto value_at = [](int64_t index) {
        return std::sin(0.53 * static_cast<double>(index)) * 0.7;
    };

    for (int32_t total_len : {1, 3, 127, 128, 129, 500, 977}) {
        for (int32_t head_size : {8, 64}) {
            for (int32_t override : {0, 1, 3, kPagedAttentionMaxSplits}) {
                const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

                std::vector<double> query(head_size);
                for (int32_t d = 0; d < head_size; ++d) {
                    query[d] = value_at(d + 1);
                }
                std::vector<double> keys(static_cast<size_t>(total_len) * head_size);
                std::vector<double> values(keys.size());
                for (size_t i = 0; i < keys.size(); ++i) {
                    keys[i] = value_at(static_cast<int64_t>(i) + 101);
                    values[i] = value_at(static_cast<int64_t>(i) + 5001);
                }

                // 参考：一次性 softmax（单趟的数学定义）
                std::vector<double> scores(total_len, 0.0);
                double ref_max = -1e30;
                for (int32_t t = 0; t < total_len; ++t) {
                    double dot = 0.0;
                    for (int32_t d = 0; d < head_size; ++d) {
                        dot += query[d] * keys[static_cast<size_t>(t) * head_size + d];
                    }
                    scores[t] = dot * scale;
                    ref_max = std::max(ref_max, scores[t]);
                }
                std::vector<double> reference(head_size, 0.0);
                {
                    double sum = 0.0;
                    for (int32_t t = 0; t < total_len; ++t) {
                        sum += std::exp(scores[t] - ref_max);
                    }
                    for (int32_t d = 0; d < head_size; ++d) {
                        double acc = 0.0;
                        for (int32_t t = 0; t < total_len; ++t) {
                            acc += std::exp(scores[t] - ref_max) *
                                   values[static_cast<size_t>(t) * head_size + d];
                        }
                        reference[d] = acc / sum;
                    }
                }

                // 分片：按被测的 `ResolveSplits` / `SplitRange` 切，逐片算局部三元组
                const int32_t effective = PagedAttentionResolveSplits(total_len, override);
                ASSERT_GE(effective, 1);
                const double neg_inf = -std::numeric_limits<double>::infinity();
                std::vector<double> part_max(effective, neg_inf);
                std::vector<double> part_sum(effective, 0.0);
                std::vector<std::vector<double>> part_acc(
                    effective, std::vector<double>(head_size, 0.0));

                for (int32_t s = 0; s < effective; ++s) {
                    int32_t begin = -1;
                    int32_t end = -1;
                    PagedAttentionSplitRange(total_len, s, effective, &begin, &end);
                    double local_max = -1e30;
                    for (int32_t t = begin; t < end; ++t) {
                        local_max = std::max(local_max, scores[t]);
                    }
                    // 空片（begin == end）保持 -inf 哨兵——与 kernel 写进 workspace 的值同义
                    part_max[s] = (begin < end) ? local_max : neg_inf;
                    double local_sum = 0.0;
                    for (int32_t t = begin; t < end; ++t) {
                        const double w = std::exp(scores[t] - part_max[s]);
                        local_sum += w;
                        for (int32_t d = 0; d < head_size; ++d) {
                            part_acc[s][d] +=
                                w * values[static_cast<size_t>(t) * head_size + d];
                        }
                    }
                    part_sum[s] = local_sum;  // 空片恒为 0
                }

                // 归并：与 kernel 的 merge 步**同一套规则**（含"空片按 l <= 0 跳过"）
                double global_max = neg_inf;
                for (int32_t s = 0; s < effective; ++s) {
                    global_max = std::max(global_max, part_max[s]);
                }
                double merged_sum = 0.0;
                std::vector<double> merged_acc(head_size, 0.0);
                for (int32_t s = 0; s < effective; ++s) {
                    if (part_sum[s] <= 0.0) {
                        continue;
                    }
                    const double weight = std::exp(part_max[s] - global_max);
                    merged_sum += part_sum[s] * weight;
                    for (int32_t d = 0; d < head_size; ++d) {
                        merged_acc[d] += part_acc[s][d] * weight;
                    }
                }

                ASSERT_GT(merged_sum, 0.0);
                for (int32_t d = 0; d < head_size; ++d) {
                    EXPECT_NEAR(merged_acc[d] / merged_sum, reference[d], 1e-9)
                        << "total_len=" << total_len << " head_size=" << head_size
                        << " override=" << override << " d=" << d;
                }
            }
        }
    }
}

}  // namespace
}  // namespace mini_trt_llm
