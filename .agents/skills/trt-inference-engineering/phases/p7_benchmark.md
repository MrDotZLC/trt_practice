# Phase P7 Benchmark

## Purpose

验证性能变化。

## Entry

必须存在：

benchmark_before.md

test_passed

## Actions

运行After Benchmark。

保持：

Before和After：

- 相同硬件
- 相同输入
- 相同配置

## Measurement Protocol

必须记录：

### Environment

- GPU
- CUDA
- TensorRT
- Driver

### Workload

包含：

- batch size
- sequence length
- dtype
- shape

### Timing

要求：

Warmup:

> =10

Iteration:

> =100

统计：

- median
- P95

## Output

生成：

benchmark.md

格式：

| Metric | Before | After | Change |
|||||

## Performance Decision

### Positive

提升：

> =5%

自动接受。

### Marginal

范围：

0% ~ 5%

需要说明：

- 是否保留
- 是否有其他收益

等待Human确认。

### Negative

性能下降：

必须分析：

- 原因
- 是否回滚

等待Human确认。

## Exit Gate

Self Check：

benchmark.md完整。

Human Gate：

根据收益情况触发。

## Output

更新 STATE.md 字段：

- phase: P8-Documentation
