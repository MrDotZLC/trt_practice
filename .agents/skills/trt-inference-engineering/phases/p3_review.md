# Phase P3 Review

## Purpose

对设计进行工程审查。

## Entry

读取：

design.md

requirement.md

加载：

[checklists/cpp.md](../checklists/cpp.md)

[checklists/cuda.md](../checklists/cuda.md)

[checklists/tensorrt.md](../checklists/tensorrt.md)

[checklists/llm_runtime.md](../checklists/llm_runtime.md)

规则：

- 必须对四份 checklists 的**每一个 [P0] / [P1] 项逐条回答**，结论逐条记入 review.md。
- 只做"设计事实是否正确"的核对不算完成本阶段。

## Actions

生成：

review.md

## Review Classification

### P0 Blocker

必须修复。

例如：

- TensorRT Context生命周期错误
- CUDA memory ownership不明确
- 数据竞争
- 需求条目在设计中没有落点，或落点是空壳（只有名字，没有流程 / 状态机 / 判据）

结果：

阻止进入P4。

### P1 Risk

需要人工确认。

例如：

- 性能风险
- 异常恢复不足

### P2 Quality

记录即可。

例如：

- 命名风格
- 文档完善

## Exit Gate

### Self Check

review.md必须包含：

| Issue | Level | Action |
||||

以及：

| 需求条目 | 设计落点 | 结论 |
|---|---|---|

### Human Gate

Gate-A（位置：P3 Review 之后）规则：

#### P0存在

暂停，返回 P2。

#### P0不存在，P1存在

暂停。

#### P0/P1不存在

自动进入P4。

## Failure Route

P0失败：

返回P2。

设计重新修改。

## Output

生成：

docs/dev/<feature>/review.md

更新：

STATE.md 字段：

- phase: P4-Baseline（Gate-A 通过后）
- status: waiting-human-gate（Gate-A 未通过前）
