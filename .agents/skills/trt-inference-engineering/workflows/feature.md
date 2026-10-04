# Feature Workflow

## Purpose

用于：

- 新功能开发
- 性能优化
- 架构增强

原则：

先设计，后实现。

先验证，后优化。

## Workflow State Machine

P0 Requirement  
↓  
P1 Analysis  
↓  
P2 Design  
↓  
P3 Review  
↓  
Gate-A  
↓  
P4 Baseline  
↓  
P5 Implementation  
↓  
P6 Test  
↓  
P7 Benchmark  
↓  
Gate-B  
↓  
P8 Documentation  
↓  
P9 Interview

## Phase Loading

进入Phase：

只加载对应Phase文件。

例如：

进入P3：

读取：

[phases/p3_review.md](../phases/p3_review.md)

禁止：

一次加载全部Phase。

## Gate Rules

两个 Gate 的判据不在此复述。

### Gate-A

位置：

After P3 Review

输入：

design.md + review.md

判断：

以 [phases/p3_review.md](../phases/p3_review.md) 的 Human Gate 细则为唯一权威；摘要见 [SKILL.md](../SKILL.md) 的 Human Gate Protocol Summary。

### Gate-B

位置：

After P7 Benchmark

输入：

benchmark.md

判断：

以 [phases/p7_benchmark.md](../phases/p7_benchmark.md) 的 Performance Decision 为唯一权威（含判别下限前置）。

缺依赖导致 P7 不可执行时，以 [phases/p4_baseline.md](../phases/p4_baseline.md) 的 Dependency Missing 为唯一权威，Gate-B 记 N/A。

## Required Artifacts

Feature Workflow必须产生：

STATE.md

requirement.md

analysis.md

design.md

review.md

benchmark_before.md

benchmark.md

test_plan.md

summary.md

interview_notes.md

## Commit Rules

每个commit：

只完成一个Phase子任务。

推荐格式：

[feature][Phase-X] description

禁止：

一个commit跨越多个无关功能。

提交前：

必须先提醒作者并给出完整 commit msg 与文件清单（含 `git add` 的对象），得到确认后才执行。

禁止：

自动提交（含"add + commit"连做）；未获确认前改动一律留在工作区。
