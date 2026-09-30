# Phase P5 Implementation

## Purpose

根据design.md实现功能。

原则：

小步修改，可验证。

# Entry

读取：

design.md

review.md

benchmark_before.md

# Actions

## 1. 创建Implementation Plan

在STATE.md记录：

- 当前修改模块
- 预计文件
- 测试方式

## 2. Code Modification

执行：

- C++修改
- CUDA Kernel修改
- TensorRT Plugin修改
- Runtime修改

# Change Scope Rules

## Soft Constraint

单次commit目标：

Files <=3

Lines <=300

如果超过：

必须在commit说明：

- 原因
- 涉及模块
- 风险

## Hard Constraint

禁止：

- 一个commit跨越两个无关模块
- 未设计新增接口
- 顺手重构

# TensorRT Specific Check

修改涉及TensorRT时：

必须检查：

- Engine ownership
- ExecutionContext生命周期
- Binding一致性
- Dynamic Shape Profile
- CUDA Stream同步

# CUDA Specific Check

修改CUDA Kernel时：

必须检查：

- thread/block配置
- memory access
- synchronization
- boundary condition
- correctness

# Exit Gate

Self Check:

确认：

- 编译通过
- 修改符合design
- 无新增warning

失败：

返回P2或P3重新设计。

# Output

更新：

STATE:

phase: P6-Test
