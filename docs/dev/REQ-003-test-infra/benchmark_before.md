# Benchmark Before

> **不适用**：本条目的目标是"把端到端链路变成可自动运行的测试，并补齐优化配置能力"，
> **没有性能目标**。

## Commit

不适用。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 不适用　Sequence Length: 不适用　Dtype: FP32（单算子闭环用它做绝对误差判据）

## Measurement

Warmup: 不适用　Iteration: 不适用　Median: 不适用　P95: 不适用

## Result

不适用。本条目唯一与"量"有关的判据是**数值**（FP32 绝对误差 < `1e-4`），不是性能。
