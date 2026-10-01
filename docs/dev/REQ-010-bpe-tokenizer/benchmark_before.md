# Benchmark Before

> **不适用**：本条目的目标是**语义正确**（与独立参考逐 token 全等），不是性能。

## Commit

不适用。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）——**本条目是纯 host 实现**，
不依赖 GPU。TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 不适用　Sequence Length: 不适用　Dtype: 不适用（文本 → token id）

## Measurement

Warmup: 不适用　Iteration: 不适用　Median: 不适用　P95: 不适用

## Result

不适用。本条目唯一的"量"是**正确性**：与独立参考逐 token 全等；以及参考数据自证（词表规模、
每个 id 落在词表内、来源字段齐全）。
