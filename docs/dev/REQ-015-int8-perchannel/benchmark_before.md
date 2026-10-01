# Benchmark Before

> **不适用**：本条目的目标是**定位数值退化根因**，不是性能。它产出的"读数"是**逐层数值差**，
> 不是延迟。

## Commit

不适用（不改产品代码、不改默认产物）。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）——主证据链还包含**离线**部分
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 8　Sequence Length: 不适用（图像输入）　Dtype: INT8（对照 FP32）

## Measurement

不适用（无延迟测量）。

## Result

不适用。被记录的是**数值判据**：探针读数应满足 `d_pre ≤ d_post`（量化前的差值小于量化后的差值），
以及两臂的**噪声地板**由对照臂当场量出（真机首层：`8.34e-07` vs `0.0398`）。
