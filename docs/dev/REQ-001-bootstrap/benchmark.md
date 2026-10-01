# Benchmark Report

> **不适用**：同 `benchmark_before.md`——本条目无性能目标，因此没有 Before/After 对照。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

不适用（骨架阶段没有可测量的推理负载；当时只有测试用的最小网络）。

## Measurement

不适用。

## Comparison

| Metric | Before | After | Change |
|---|---|---|---|
| — | 不适用 | 不适用 | 不适用 |

## Analysis

骨架阶段唯一与"快慢"有关的量是**构建耗时**（依赖编译、引擎构建），它由后续阶段在真机上量过
（见 `docs/dev/REQ-006-gpt2-onnx/` 等条目的 benchmark），不在本条目账上。

## Decision

不适用（无优化可 Keep / Rollback）。
