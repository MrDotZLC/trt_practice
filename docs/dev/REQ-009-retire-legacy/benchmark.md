# Benchmark Report

> **不适用**：无性能目标、无性能采集。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

不适用。

## Measurement

不适用。

## Comparison

| Metric | Before | After | Change |
|---|---|---|---|
| 跳过集合（沙箱） | 迁移前集合 | **逐条相同** ✅ | 0 |
| 跳过集合（真机） | 阶段 0 钉住的基线 | **逐条相同** ✅ | 0 |

## Analysis

"零变化"就是本条目唯一可接受的对比结果：迁移与删除**不允许改变任何用例的覆盖状态**。
删除后仍**只有那些因缺资产/缺设备而跳过**的项目，其余全部照跑。

## Decision

**Keep**（接受删除；判据 = 跳过集合逐条不变）。
