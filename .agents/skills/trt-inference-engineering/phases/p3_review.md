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
- 必须核对 requirement.md 中的每个模糊名词是否已在 analysis.md 的 Terminology 中给出可验证定义；未定义即 P0 Blocker（依据 [p1_analysis.md](p1_analysis.md) 的术语表规则）。
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

review.md 必须包含以下三张表；每张表的数据行数必须等于对应集合的条目数（表一 = 四份 checklists 中 [P0] / [P1] 项总数；表二 = Included 条数 + Acceptance Criteria 条数；表三 = requirement.md 中的模糊名词数），只有表头不算通过。

表一：checklists 结论表，四份 checklists 中每个 [P0] / [P1] 项各占一行。

| Item | Level | Result | Action |
|---|---|---|---|

- Item：checklist 项原文，或可定位到该条目的缩写。
- Level：该项在 checklist 中的等级，取值 P0 / P1。
- Result：通过 / 不通过 / 待确认。
- Action：修复 / 人工确认 / 记录；P0 项必须给出修复落点。

表二：需求落点表，requirement.md 中每条 Included 与每条 Acceptance Criteria 各占一行。

| 需求条目 | 设计落点 | 结论 |
|---|---|---|

- 结论取值：已落点 / 空壳 / 缺失；空壳与缺失即 P0 Blocker（见 Review Classification）。

表三：术语定义表，requirement.md 中每个模糊名词各占一行。

| 模糊名词 | 定义所在 | 结论 |
|---|---|---|

- 结论取值：已定义（给出 analysis.md Terminology 中的条目）/ 未定义（P0 Blocker，见 Entry 规则）。

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
