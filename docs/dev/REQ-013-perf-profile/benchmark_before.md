# Benchmark Before

> **性质**：本条目是**测量方法本身**，所以"Before"就是"没有可复现方法"的状态。
> 基线由本条目建立，形态是**协议 + 判别下限**，而不是某个延迟数字。

## Commit

不适用（不改产品代码；采集入口不进默认构建）。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）；WSL2，不能锁时钟
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 1（当时 Runner 只支持单序列）　Sequence Length: 4 / 256 / 960（上下文扫描）
Dtype: FP32

## Measurement

Warmup: 有　Iteration: ≥20（跨构建对照）/ 同轮交替（A/B）　Median: **报中位数**　P95: 报 p25/p75

## Result

**基线不是数字而是三个边界**：

1. **判别下限 ≈ ±400~600 µs**（本平台实测）——低于它的差异**不许**写成"更快/更慢 X%"。
2. **跨 session 不可比**（同实现跨 session 曾差 ±23%）。
3. **本机拿不到 GPU kernel 时间线**（两条 CLI 路径都失败）→ 逐 kernel 分解记为**能力边界**。
