# Benchmark Before

## Commit

`e1ce430`（`e1ce4300511f4cf4e06c2e45d3927a13b4ab818e`）。本次基线**未取数**，此 hash 只用来
标记"代码在这个前提下开工"。

## Environment

GPU: 未取得 —— 本设备暂不支持真机测试（作者 2026-10-05 指示真机测试搁置）

CUDA: 未取得 —— 目标环境见 `AGENTS.md` §1（12.6.85）

TensorRT: 未取得 —— 目标环境见 `AGENTS.md` §1（10.15.1）

## Workload

Batch: 1（沿用 `LLMRunner` 的 `batch = 1` 口径，与 PF-9 可对照）

Sequence Length: prefill prompt 三档 4 / 256 / 960（与 PF-9 的 3.055 / 6.062 / 14.705 ms 同口径，
便于"固定项主导"这条判断在真机上被直接检验）

Dtype: FP32（基线臂）；对照臂 = int8 权重 + 同一个精度档位（激活精度按 D7 在真机上先定）

## Measurement

Warmup: ≥10（协议要求；本次未执行）

Iteration: ≥100（协议要求；本次未执行）

Median: 未取得

P95: 未取得

## Result

N/A: 本设备暂不支持真机测试（无 GPU、无 nvcc / cmake / TensorRT），无法构建引擎与取数；
作者 2026-10-05 指示"真机测试搁置、先完成开发工作"。

按 `phases/p4_baseline.md` 的 Dependency Missing，P7 随之不可执行：**Gate-B 记 N/A（缺依赖）**，
既不算通过也不算失败。环境恢复后至少要补下面五项，且必须先于任何"收益"结论：

1. **D7 的激活精度实验**（P5 第一步的欠账）：最小图（单个 Linear + int8 常量 + DQ）在
   "无 flag（FP32）"与"kFP16"两种构建下各建一份，读**引擎体积**与逐层精度——体积不降就说明
   DQ 被常量折叠，收益归零，必须回 Gate-A。
2. **FP32 基线四项**：预填/解码两种引擎的体积、显存占用、每步延迟（三档 prompt 各一次）。
3. **INT8 引擎同四项** + 逐层精度自证（`detailed_profiling` + Inspector）。
4. **P7 的 A/B**：同 session、同二进制、逐轮交替、报中位数与四分位，先声明判别下限
   （本机约 ±400~600 µs，出处 `docs/PROGRESS.md` §5.13b）。
5. **AC3 的数值对照**：同 prompt 同采样策略下逐 token 一致率 + prefill logits 的相对/绝对偏差，
   阈值按 D4"先量再定"——口径由 P6 的 `test_plan.md` 定义。

## 三处留痕（`phases/p4_baseline.md` 的 Dependency Missing）

行数 = 该条点名的三处；逐项打勾，不转述条文。

| # | 留痕 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| ① | 本文件的 `## Result` 记 `N/A: <原因>` | 满足 | 见上（原因含设备与作者指示） |
| ② | `STATE.md` 的 Next Action / Current Blockers 写 `Gate-B: N/A（缺依赖：<原因>）` | 满足 | `STATE.md` 的 `## Current Blockers` 首条 |
| ③ | `summary.md` 的 Performance 一节写 N/A + 原因 | **未满足（阻塞）** | `summary.md` 是 P8 产物，P4 时点不可能产出。按 `AGENTS.md` §5 第 5 条登记并上报，**不改写成别的产物**；义务锚定在 `STATE.md` 的 Next Action（P8 必清） |
| ④ | `docs/PROGRESS.md` 的"已知问题与坑"留一条 | 满足 | 作者 2026-10-05 放行后追加：`docs/PROGRESS.md` §5.17（附加留痕，不在技能点名的三处里） |
