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
Gate-A  
↓  
P3 Review  
↓  
Gate-B  
↓  
P4 Baseline  
↓  
P5 Implementation  
↓  
P6 Test  
↓  
P7 Benchmark  
↓  
Gate-C  
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

phases/p3_review.md

禁止：

一次加载全部Phase。

## Gate Rules

### Gate-A

位置：

After P2 Design

输入：

design.md

行为：

等待用户确认。

失败：

返回P2。

### Gate-B

位置：

After P3 Review

输入：

review.md

判断：

#### P0 Blocker

立即阻塞。

#### P1 Risk

等待用户确认。

#### P2 Quality

记录并继续。

### Gate-C

位置：

After P7 Benchmark

输入：

benchmark.md

判断：

#### >=5%

Accept。

#### 0~5%

Human Review。

#### <0%

Human Review。

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
