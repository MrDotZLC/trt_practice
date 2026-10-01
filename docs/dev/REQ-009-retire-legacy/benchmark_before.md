# Benchmark Before

> **不适用**：本条目是**资产迁移与目录下线**，没有性能目标。唯一被"测量"的是**跳过集合**
> （用例覆盖面的度量），它的基线记在 `test_plan.md`。

## Commit

不适用。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

不适用（被度量的是 ctest 的跳过集合，不是推理负载）。

## Measurement

Warmup: 不适用　Iteration: 不适用　Median: 不适用　P95: 不适用

## Result

不适用。**基线 = 一次真机全量的跳过集合**（集合，不是条数），记在
`mini_trt_llm/tests/data/expected_skips.txt`。
