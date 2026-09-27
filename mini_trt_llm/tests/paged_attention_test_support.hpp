#pragma once

// PagedAttention 测试共享件（FP32 与 FP16 两个测试文件共用）。
//
// 为什么要有这个文件：`SetPagedAttentionNumSplitsOverride` 是**进程级**状态。两个测试文件
// 各写一份 RAII 守卫，等于把"忘记复位"的风险复制成两份——而那种"顺序影响结果"的假故障
// 最难查（`PROGRESS.md` §2.14 C）。共享件只有一份，也方便以后加别的 A/B 开关。

#include "mini_trt_llm/plugins/paged_attention_kernel.hpp"

#include <cstddef>
#include <cstdint>

namespace mini_trt_llm {
namespace test_support {

// 分片数覆盖的 RAII 守卫：构造时设值，析构时**无条件复位为 0（自适应）**。
class ScopedSplitsOverride {
 public:
    explicit ScopedSplitsOverride(int32_t splits) { SetPagedAttentionNumSplitsOverride(splits); }
    ~ScopedSplitsOverride() { SetPagedAttentionNumSplitsOverride(0); }

    ScopedSplitsOverride(const ScopedSplitsOverride&) = delete;
    ScopedSplitsOverride& operator=(const ScopedSplitsOverride&) = delete;
};

// workspace 护栏区的填充值：kernel 若越过了契约（写了 slot 下标 >= 分片上限），
// 护栏区会被改写。这类越界在真机上通常表现为"某些形状下偶发数值错/非法访存"，事后归因极难。
inline constexpr float kWorkspaceGuardSentinel = 12345.0f;

}  // namespace test_support
}  // namespace mini_trt_llm
