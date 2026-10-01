# Benchmark Report

> **不适用**：本条目无性能目标、无性能采集，因此没有 Before/After 对照。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Dtype: INT8（Q/DQ）　Batch: 8

## Measurement

不适用。当时被记录的是**与性能无关的两个量**：① 引擎体积（量化图 + 融合带来的变化）；
② 逐层精度与 tactic 名（"真的在跑低精度"的证据）。

## Comparison

| Metric | Before | After | Change |
|---|---|---|---|
| — | 不适用 | 不适用 | 不适用 |

## Analysis

若要回答"低精度到底快多少"，必须按可复现协议单独测（同 session、同轮交替、报中位数与极差，
见 `docs/dev/REQ-013-perf-profile/`）；本条目只负责让它**能跑、能自证、能判**。

## Decision

不适用。
