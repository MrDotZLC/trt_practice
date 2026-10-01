# Benchmark Before

> **性质**：本条目**有**与性能相关的决策（最优批量取值），但没有可用的历史性能数据——
> 历史工程只留了方法（计时器 + 批量扫描），**没有留下任何数字**。

## Commit

阶段 4 的 CV 路径构建产物。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 扫描 `{1, 2, 4, 8, 16}`　Sequence Length: 不适用（图像输入，空间维固定）
Dtype: FP32 / FP16 / INT8

## Measurement

Warmup: 未记录　Iteration: 未记录　Median: 未记录　P95: 未记录
计时方式（历史工程）：设备事件计时。

## Result

**无历史数字可引用**。当时被记录下来的只有两条可用信息：

1. **最优批量的取值会影响结论**：历史工程用 8，沿用 1 会让批量 ≥2 走非最优内核 → 默认值由 1 改为 8。
2. 历史工程的基准输入是**常量张量**，与真实推理负载不同 → 其"性能结论"本来也不可迁移。

因此本条目不给性能基线，只给**方法**；真要报性能必须按可复现协议重测（`docs/dev/REQ-013-perf-profile/`）。
