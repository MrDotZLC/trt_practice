# Benchmark Before

> **性质**：本条目的性能对照**做过两次、方向相反**，所以"Before"这一侧本身就是两个互相矛盾的
> 观测值。本文件记录**测量条件与已知噪声**，而不是一个可用的基线。

## Commit

阶段 3 的构建产物（外部图引擎 vs 原生引擎，同一 session 内各自构建）。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 1　Sequence Length: 4（prompt）　Dtype: FP32

## Measurement

Warmup: 未固定　Iteration: 单次点值为主　Median: 报的是单次值　P95: 未采集

## Result

| 运行 | 外部图 / 原生（prefill 4 token） | 方向 |
|---|---|---|
| A | 4.545 ms / 3.717 ms | 外部图**慢**约 22% |
| B | 4.716 ms / 6.250 ms | 外部图**快**约 24% |

**两次差异（±25%）小于构建间噪声** → "谁更快"当时**未定**。已知机制：内核 tactic 选择依赖构建时
的机器状态。**这件事直接催生了后续的"可复现测量方法"条目**（`docs/dev/REQ-013-perf-profile/`）。
