# Phase P8 Documentation

## Purpose

生成维护文档。

目标：

方便未来开发者理解。

## Entry

> **先复核上一阶段的「判据对照」**（该 feature 的 `STATE.md` 的 `## 判据对照` 一节）：存在
> "未满足（阻塞）"项时**不得开工**，除非作者明确放行并把放行范围记在该节里。

读取：

- requirement.md
- analysis.md
- design.md
- review.md
- benchmark.md

## Actions

生成：

summary.md

## summary.md Structure

必须包含：

### Problem

解决的问题。

### Architecture

整体架构。

### Implementation

关键实现。

### Performance

Benchmark结果。

### Limitation

当前限制。

### Future Work

未来方向。

### Requirement Coverage Result

| 需求条目 | 交付情况 | 去向 / 证据 |
|---|---|---|

规则：

- requirement.md 的每条 Included 与每条 Acceptance Criteria 都必须有一行。
- 交付情况取值：已交付 / 部分交付 / 未交付。
- 部分交付与未交付必须写去向（转入哪条 feature / 哪份文档）与原因。
- 存在未交付且没有去处的条目，本阶段自检不通过。

## Boundary

summary.md：

面向：

项目维护者。

禁止：

生成面试话术。

## Exit Gate

Self Check:

确认：

- 架构完整
- 性能完整
- 限制明确
- 需求逐条有结论（Requirement Coverage Result 无空缺）

## Output

更新 STATE.md 字段：

- phase: P9-Interview
