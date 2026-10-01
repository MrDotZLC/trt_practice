# Benchmark Before

> **性质**：本条目**有**真实基线——先测基线，再定"要不要换实现"与性能门槛。

## Commit

改动前的采样器实现（整行降序排序）。

## Environment

GPU: NVIDIA GeForce GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core）
TensorRT: 10.15.1　CUDA: 12.6.85

## Workload

Batch: 1 / 8 行　Sequence Length: 不适用（采样只读 logits 行）
Dtype: FP32 / FP16　词表规模: 50257（真实形状）/ 128000（合成）

## Measurement

Warmup: 有　Iteration: n = 21　Median: **报中位数**（单位 ms）　P95: 未报（另有极差）
口径：真机、`SamplerPerf.ThroughputByShape`。

## Result

| 形状 | 贪心 | Top-K(k=64) | Top-P(p=0.9) | Top-K/贪心 | Top-P/贪心 |
|---|---|---|---|---|---|
| 50257 × 1 | 0.0443 | 0.5671 | **8.1469** | 12.8× | **183.8×** |
| 50257 × 8 | 0.0873 | 0.6685 | 11.4256 | 7.7× | 130.9× |
| 128000 × 1 | 0.0776 | 1.2020 | **15.9150** | 15.5× | **205.0×** |
| 128000 × 8 | 0.1495 | 1.2377 | 20.6862 | 8.3× | 138.4× |

**读出来的三件事**：① 贪心已是单次比较、无排序；② Top-P 是主要瓶颈（相对贪心高两个数量级）；
③ 倍数是**比值**，不受跨 session 漂移影响——所以"先优化 Top-P"是有依据的判断。
