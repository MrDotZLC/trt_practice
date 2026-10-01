# Benchmark Report

> **不适用（结论 = 未定）**：本条目**没有**给出可用的性能对照——两次测量方向相反，差异小于构建间
> 噪声。按项目纪律，此时唯一允许的写法是"未定"，不得挑一个方向当成结论。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch 1 / Sequence Length 4 / FP32（两次运行同规格）。

## Measurement

**不满足可复现协议**：未做同 session 的多次交替测量，也没有报极差与判别下限——这正是后续
`docs/dev/REQ-013-perf-profile/` 要建立的东西。

## Comparison

| Metric | Before（原生） | After（外部图） | Change |
|---|---|---|---|
| prefill 4 token 延迟 | 3.717 ms（运行 A）/ 6.250 ms（运行 B） | 4.545 ms（A）/ 4.716 ms（B） | **方向相反，判为未定** |

## Analysis

两次结果互相矛盾且都在噪声量级内 → **不能得出任何方向性结论**。当时的正确处置是：
① 把它记为"未定"；② 先建立可复现的测量方法；③ 之后的同类判断（如"外部图 vs 原生谁快"）
必须先跑跨构建对照（PF-7）。

## Decision

**不适用（未定）**：既不 Keep 也不 Rollback——因为没有可信的差异。
