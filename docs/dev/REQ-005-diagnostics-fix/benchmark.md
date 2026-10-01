# Benchmark Report

> **不适用**：本条目无性能改动，无 Before/After 性能对照。

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

唯一与"代价"有关的事实：诊断路径默认关闭，因此**生产路径的引擎构建与运行没有增加任何开销**；
打开开关时才多 4 个输出并需要独立建引擎。

## Decision

不适用。
