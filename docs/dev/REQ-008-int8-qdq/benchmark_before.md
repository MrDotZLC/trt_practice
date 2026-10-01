# Benchmark Before

> **不适用**：本条目的目标是**精度可用与可判**，不是性能。历史工程虽实现过低精度，但
> **没有留下任何性能数字**（只留了方法），因此也不存在可引用的基线。

## Commit

不适用。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，**无 Tensor Core**）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 8（与历史工程一致）　Sequence Length: 不适用　Dtype: INT8（Q/DQ 显式量化）

## Measurement

不适用（未做性能采集）。

## Result

不适用。**本机没有张量核心**，低精度收益来自**显存带宽**而非算力——这一点在立项理由里写清，
但它属于定性判断，本条目没有为它采集数字。
