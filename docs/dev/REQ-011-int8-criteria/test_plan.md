# Test Plan

> **性质**：历史用例回填（2026-10-01）。来源：`docs/future_iterations_development_plan.md` §2.2 步骤与产物。
> **判据以 `requirement.md` 为准**。

## Unit Test

**护栏自检**：脚本对 7 类非法输入必须拒绝，且每类都有一份故意改坏的输入作为证据。
自检项进 ctest。

## Integration Test

分层统计脚本与运行时侧现役实现在**同一批数据**上口径一致（同一定义、同一分桶来源）。

## Regression Test

- 自检进 ctest 后，"脚本被改坏"这件事可被自动发现。
- 缺 Python / 缺资产时返回跳过码，不污染回归信号。

## Failure Test

缺失必填字段的验收集（无来源 / 无版本 / 无哈希 / 无标签 / 与标定集重叠）必须被**拒绝引用**。

## Expected Result

规格落盘；脚本产出带样本量的分层报告；自检进 ctest；判据表出处指向新规格。

## Actual Result

达成；该规格后来被用于真机侧与脚本侧的**口径交叉校验**。

## Status

已交付（历史条目，**仅离线一半**；整条需求的另一半需联网，仍未完成）。
