# Benchmark Before

> **不适用**：本条目是**正确性 / 契约修复**，没有性能目标。

## Commit

不适用。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 不适用　Sequence Length: 不适用　Dtype: FP32 / FP16（诊断与复现）

## Measurement

不适用。

## Result

不适用。本条目唯一"量"的代价是**诊断路径要独立建一次引擎**（分钟级，仅诊断时发生）。
