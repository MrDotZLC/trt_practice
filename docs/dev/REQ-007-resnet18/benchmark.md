# Benchmark Report

> **不适用（无可用对照）**：本条目没有做性能优化，也没有留下可信的历史数据；
> 唯一与性能有关的是"最优批量"这个**配置决策**。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 1 / 8 / 16（推理验收用的档位）　Dtype: FP32 / FP16 / INT8

## Measurement

只做了**端到端的可运行性验证**（运行时封装的推理与基准接口），未按可复现协议采集性能数字。

## Comparison

| Metric | Before | After | Change |
|---|---|---|---|
| — | 无历史数据 | 未采集 | 不适用 |

## Analysis

**不要引用历史工程的性能数字**——它既没留数字，其输入也与真实负载不同。本条目对性能的唯一贡献是
把"最优批量"从 1 改为 8，让后续任何性能测量不再因配置失真而失效。

## Decision

不适用（无优化可 Keep / Rollback；配置决策已生效）。
