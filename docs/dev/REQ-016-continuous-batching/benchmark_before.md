# Benchmark Before

## Commit

## Environment

GPU:

CUDA:

TensorRT:

## Workload

Batch:

Sequence Length:

Dtype:

## Measurement

Warmup:

Iteration:

Median:

P95:

## Result

N/A: 当前不在 GTX 1660 Ti 环境（沙箱无 GPU、无 nvcc / cmake / 编译器，也无 TensorRT），
P4 baseline 自 2026-10-03 起**搁置**。

这是**延后不是放弃**：环境恢复后必须补齐下面四项，缺一项则"批上限"与 §Performance Consideration
的显存结论都没有依据（`AGENTS.md` §7：阈值与上限不许来路不明）。

环境恢复后必须量（口径：**关闭常驻诊断**，D7）：

1. `batch = 1` 的每步延迟 —— 同 session、同二进制 A/B、逐轮交替，报中位数与四分位，
   **且先声明本次口径的判别下限**（本机这类测量约 ±400~600 µs，`docs/PROGRESS.md` §5.13b）。
2. **prefill logits 的显存占用**（单条 + 按批增长的斜率）。
3. K/V 池的显存占用。
4. 每序列的块使用量。

另需按 D10 准备两种负载（"每步都有新请求进入" / "整批同进同出"）作为 P7 的对照输入。

**三处留痕**（Dependency Missing 的要求）：本文件（已写）+ `STATE.md`（已写）+
`docs/PROGRESS.md` 的"已知问题与坑"（待阶段/批次收口时补，本阶段写不到那里）。