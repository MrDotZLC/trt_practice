# Benchmark Before

> **不适用**：本条目目标是"跑通且自洽"，**没有性能目标**；性能画像在后续条目单独立项
> （`docs/dev/REQ-013-perf-profile/`）。

## Commit

不适用。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 1（当时 Runner 只支持单序列）　Sequence Length: prompt 4 token（对拍基线固定）
Dtype: FP32（推荐精度）

## Measurement

Warmup: 不适用　Iteration: 不适用　Median: 不适用　P95: 不适用

## Result

不适用。当时被记录的是**构建耗时**（大常量 + 多层，分钟级），以及"显存吃紧"这个定性约束。
