# Test Plan

> **性质**：历史用例回填（2026-10-01）。来源：`docs/dev/REQ-005-diagnostics-fix/phase2_supplement_plan.md` §2 P2S-4 / §4 验收判据。
> **判据以 `requirement.md` 为准**（A1–A6）。

## Unit Test

- 默认构建的网络输出数不变（A1）。
- 打开开关的仪器路径能列出 4 个诊断输出（A2）。

## Integration Test

- **真机 9 条用例全绿**（6 条先前红 + 3 条对照）：
  建网契约 4 条（含 2 条对照）、解码一致性 3 条、Runner 端到端 1 条、温度拒绝 1 条（对照）。

## Regression Test

- 沙箱 `ctest` 仍 0 失败，且 GPU 跳过条目**打印显式探测原因**（不再静默）。
- 可选：真机全量复跑，确认除"按设计红"外无新增红。

## Failure Test

- 沙箱内"建构建器失败"必须表现为 **FAIL**，不再是 skip（消除假绿滑梯）。
- 若修复后 FP16 复现器仍以 **enqueue 失败**告终，说明开关没穿到底——
  它应当回到"NaN → token 全 0"这一按设计红的形态。

## Expected Result

9 条目标用例全绿；沙箱保持可绿；跳过必须显式。

## Actual Result

全部达成；对照用例证明"开关默认关闭"未改变生产契约。

## Status

已交付（历史条目）。
