# Test Plan

> **性质**：历史用例回填（2026-10-01）。来源：`docs/future_iterations_development_plan.md` §10.5。
> **判据以 `requirement.md` 为准**。

## Unit Test

采样器既有用例**不改判据、全部通过**（主判据）；解码路径的 Top-K（k=1）与低精度路径的贪心用例通过。

## Integration Test

- **低精度覆盖**：新增 Top-K / Top-P 的低精度用例，判据沿用既有分布口径（与解析概率比较）。
- **大词表覆盖**：在合成大词表与真实词表形状下，
  Top-K 结果必落在解析 top-K 集合内、Top-P 结果必落在解析 nucleus 内。
  **这两条不设阈值**——集合成员关系判定，无阈值即无出处问题。

## Regression Test

语义不变：新路径与旧路径逐 token 一致（同二进制 A/B 的对照入口永久保留）。

## Failure Test

非法参数（k < 1、p 越界）与空指针按既有约定返回错误码，不静默。

## Expected Result

语义回归全绿；低精度与大词表覆盖补齐；性能对冻结基线达标（同二进制 A/B、报中位数与四分位）。

## Actual Result

Top-P 达标；**Top-K 快速路径被撤回**（正确但更慢）；收尾段优化判"无显著差异"，不设判据。

## Status

已关闭（历史条目，部分达标、部分撤回）。
