# Benchmark Report

> **不适用**：本条目未做性能优化，因此没有 Before/After 对照。采样器的性能优化在
> `docs/dev/REQ-012-sampler-kernel/` 单独立项（那一份有完整的 baseline 与 A/B）。

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

当时唯一被记录的性能相关结论是**定性**的：分页注意力的并行度只有 `(头, 批)`，每块要串行走完
上下文——这条观察后来成为长上下文优化的立项依据（`docs/dev/REQ-014-attention-splitk/`）。

## Decision

不适用。
