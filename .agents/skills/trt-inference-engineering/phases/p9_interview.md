# Phase P9 Interview

## Purpose

生成面试复盘材料。

## Entry

> **先复核上一阶段的「判据对照」**（该 feature 的 `STATE.md` 的 `## 判据对照` 一节）：存在
> "未满足（阻塞）"项时**不得开工**，除非作者明确放行并把放行范围记在该节里。

读取：

summary.md

STATE.md

Phase History

## Actions

生成：

interview_notes.md

## Content Requirement

### Project Introduction

包含：

- 背景
- 目标
- 技术栈

### Technical Decisions

记录：

每个Phase关键决策。

例如：

- 为什么选择Paged KV Cache
- 为什么使用Plugin
- 为什么采用某Kernel优化

### Interview Questions

生成Level 1~5问题。

## Difficulty Level

### Level 1

基础概念。

例如：

为什么需要KV Cache？

### Level 2

工程实现。

例如：

如何设计Runtime？

### Level 3

性能优化。

例如：

如何降低Decode latency？

### Level 4

源码分析。

例如：

TensorRT ExecutionContext如何执行？

### Level 5

架构设计。

例如：

如何扩展Continuous Batching？

## Boundary

interview_notes.md：

面向：

- 面试准备
- 技术复盘

不是项目维护文档。

## Exit Gate

Self Check：

确认：

- 问题分级
- 有技术追问
- 有trade-off

## Output

更新 STATE.md 字段：

- phase: P9-Interview
- status: completed
