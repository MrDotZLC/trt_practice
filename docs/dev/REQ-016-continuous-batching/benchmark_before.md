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

这是**延后不是放弃**：环境恢复后必须补齐下面五项（2026-10-05 增列第 5 项），缺一项则"批上限"与 §Performance Consideration
的显存结论都没有依据（`AGENTS.md` §7：阈值与上限不许来路不明）。

环境恢复后必须量（口径：**关闭常驻诊断**，D7）：

1. `batch = 1` 的每步延迟 —— 同 session、同二进制 A/B、逐轮交替，报中位数与四分位，
   **且先声明本次口径的判别下限**（本机这类测量约 ±400~600 µs，`docs/PROGRESS.md` §5.13b）。
2. **prefill logits 的显存占用**（单条 + 按批增长的斜率）。
3. K/V 池的显存占用。
4. 每序列的块使用量。
5. **S5 的 chunk 维度对照**（2026-10-05 新增）：分块修订后"分块真的启用"，恢复后必须量两条 ——
   ① 同一条长 prompt 的"**分块 vs 不分块**"每步延迟；② 不同 `max_prefill_seq_len`
   （**1 / 中间值 / ≥ prompt_len**）的对照。**理由**：AC9 的"不浪费"与 P7 的收益判据都要以它为基准，
   否则 AC6 / AC7 无从结。

另需按 D10 准备两种负载（"每步都有新请求进入" / "整批同进同出"）作为 P7 的对照输入。

## Dependency Missing 的三处留痕（R1 口径：引用出处 + 逐项打勾，行数 = 权威清单行数）

权威清单 = `phases/p4_baseline.md` 的 `### Dependency Missing`（不得转述、不得替换清单项）。

| # | 留痕处（技能原文点名） | 状态 | 证据 |
|---|---|---|---|
| 1 | `benchmark_before.md` 记 `N/A: <原因>` 并通过本自检 | ✅ 已写 | 本文件 `## Result` |
| 2 | `STATE.md` 的 Next Action / Current Blockers 写 `Gate-B: N/A（缺依赖：<原因>）` | ✅ 已写（2026-10-05 补字面行） | `STATE.md` 的 Current Blockers（P4 / P7 搁置那条） |
| 3 | `summary.md` 的 Performance 一节写 N/A + 原因 | ⏳ **未满足（阻塞）**：该文件是 P8 产物 | `STATE.md` 的 Next Action 第 5 条（义务锚点）；按 `AGENTS.md` §5 第 5 条已登记+上报 |

> **2026-10-05 更正（R1 实例）**：本节原先写的是"**本文件 + STATE + PROGRESS**"——**把技能点名的
> `summary.md` 换成了本文件**，于是第 3 处义务在记录里消失；`PROGRESS` 那处又被推到了"阶段/批次
> 收口"。现按权威清单逐项列出，并保留 `docs/PROGRESS.md` §5.16 作为**附加**留痕。
