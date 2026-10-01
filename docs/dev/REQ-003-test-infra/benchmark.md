# Benchmark Report

> **不适用**：本条目无性能目标，未做任何性能优化。

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
| — | 不适用 | 不适用 | 不适用 |

## Analysis

本条目真正的成本指标是**测试耗时**（每条端到端用例要建一次引擎），当时用"同一 fixture 合并断言 +
引擎文件缓存复用"来压低它——这是工程效率，不是推理性能。

## Decision

不适用。
