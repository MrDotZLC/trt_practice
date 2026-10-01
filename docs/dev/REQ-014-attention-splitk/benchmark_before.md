# Benchmark Before

> **性质**：本条目有真实的 kernel 级与端到端对照，且必须**同二进制 A/B**（旧单趟实现被永久保留
> 作对照入口）。

## Commit

改动前的单趟 decode 注意力（每块串行走完整段上下文）。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 1　Sequence Length: 上下文 32 / 256 / 1024（kernel 级）；4 / 256 / 960（端到端）
Dtype: FP32（另有 FP16 正确性用例）

## Measurement

Warmup: 有　Iteration: kernel 级每轮连发 32 次只同步一次、ABBA、n=7　Median: 报中位数
P95: 报 p25/p75　判别下限: 约 ±400~600 µs（低于它不算差异）

## Result

**同 session、同二进制（旧单趟）**：

| 上下文 | 每层每步（ms） |
|---|---|
| 32 | 0.023600 |
| 256 | 0.174143 |
| 1024 | **0.918948** |

**斜率** = (0.918948 − 0.023600) / (1024 − 32) × 1000 = **0.902569 ms / 1000 位置**。
端到端一臂（同一 session）：prompt 960 时每步 **13.0443 ms**，斜率 **10.795 ms / 1000 位置**。
