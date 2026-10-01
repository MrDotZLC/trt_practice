# Test Plan

> **性质**：历史用例回填（2026-10-01）。来源：`docs/future_iterations_development_plan.md` §11.6。
> **判据以 `requirement.md` 为准**。

## Unit Test

分桶脚本带自检（进 ctest）；采集 target 的出现不影响默认构建。

## Integration Test

一键采集在无 GUI 环境下可用：报告文件存在且非空，能打印摘要。

## Regression Test

生产路径不受影响：采集 target 不在默认构建内；逐层精度开关走独立引擎路径。

## Failure Test

- kernel 级无头导出：**本机实测不可用** → 记为**已知限制**，而不是"通过"。
- 低于判别下限的观测 → 结论必须是"无显著差异"。

## Expected Result

三个待决问题都有答案（采样器占比 / 注意力占比 / 显存分配够不够贵），且每个数字都能追溯到命令与构建态。

## Actual Result

三问全部作答；逐 kernel 分解记为能力边界；跨构建对照子项移交独立条目。

## Status

已关闭（历史条目）。
