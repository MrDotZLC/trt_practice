# Benchmark Report

> **不适用**：无性能目标、无性能采集。分词器是 host 侧纯计算，当时没有做吞吐测量。

## Environment

GPU: 不适用（纯 host）　TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

不适用。

## Measurement

不适用。

## Comparison

| Metric | Before | After | Change |
|---|---|---|---|
| — | 不适用 | 不适用 | 不适用 |

## Analysis

唯一与"代价"有关的观察：参考数据的可复现性由一条独立自检承担（重新用参考实现算一遍再比对），
它带来约 3 秒的测试时间——这是**判据成本**，不是性能问题。

## Decision

不适用。
