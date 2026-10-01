#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <sstream>
#include <string>

namespace mini_trt_llm {
namespace test_support {

// 环境变量 `MINI_TRT_REQUIRE_ASSETS=1` 表示"本环境必须具备测试所需的资产"。
//
// 为什么需要它：缺资产时的默认动作是**跳过**（缺环境 ≠ 代码有 bug），但 ctest 把跳过
// 记成 Passed——实测（2026-09-27，把两个历史示例工程改名后跑沙箱全量）ctest 仍报
// **265 条 / 100% passed / 0 failed**，只有跳过集合变了 3 项。也就是说"删掉资产 → 一批
// 用例不再跑"这件事对 CI **完全不可见**；而真机上的 `MINI_TRT_REQUIRE_GPU=1` 也拦不住它
// （那道闸只管"没有 CUDA 设备"，资产缺失走的是另一条路）。
//
// 所以资产迁移 / 删除这类改动，必须在带本闸门的环境里跑：缺资产即红，覆盖损失无处可藏。
// 计划与验收见 `docs/dev/REQ-009-retire-legacy/phase5_development_plan.md` 阶段 0；资产清单见
// `docs/PROGRESS.md + DEC-ENVIRONMENT`。
inline bool RequireAssets() {
    const char* flag = std::getenv("MINI_TRT_REQUIRE_ASSETS");
    return flag != nullptr && *flag != '\0' && std::string(flag) != "0";
}

}  // namespace test_support
}  // namespace mini_trt_llm

// 资产缺失时跳过本用例；但设了 `MINI_TRT_REQUIRE_ASSETS=1` 时**判失败**。
//
// **必须是宏，不能封装成函数**：`GTEST_SKIP()` / `GTEST_FAIL()` 展开后都是 `return` 语句，
// 封装成函数只会退出那个函数、调用方的测试体会继续执行（`test_gpu_guard.hpp` 里记着这条
// 教训：封装后沙箱里 63 条用例从 Skipped 变成 Failed）。
//
// 用法与 `GTEST_SKIP()` 一致，参数是给用户看的缺失说明：
//   if (onnx.empty()) { MINI_TRT_SKIP_IF_MISSING_ASSET("需要 assets/legacy/resnet18_onnx/resnet18.onnx"); }
#define MINI_TRT_SKIP_IF_MISSING_ASSET(...)                                            \
    do {                                                                               \
        ::std::ostringstream mini_trt_asset_oss_;                                      \
        mini_trt_asset_oss_ << __VA_ARGS__;                                            \
        const ::std::string mini_trt_asset_msg_ = mini_trt_asset_oss_.str();           \
        if (::mini_trt_llm::test_support::RequireAssets()) {                           \
            GTEST_FAIL() << mini_trt_asset_msg_                                        \
                         << "；已设置 MINI_TRT_REQUIRE_ASSETS=1，缺资产在本环境算失败"     \
                            "（资产准备见 docs/PROGRESS.md + DEC-ENVIRONMENT；"          \
                            "闸门由来见 docs/dev/REQ-009-retire-legacy/phase5_development_plan.md 阶段 0）";      \
        }                                                                              \
        GTEST_SKIP() << mini_trt_asset_msg_;                                           \
    } while (false)
