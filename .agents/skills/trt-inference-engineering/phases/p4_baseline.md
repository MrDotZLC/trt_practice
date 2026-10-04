# Phase P4 Baseline

## Purpose

在修改代码前建立性能基线。

## Entry

> **先复核上一阶段的「判据对照」**（该 feature 的 `STATE.md` 的 `## 判据对照` 一节）：存在
> "未满足（阻塞）"项时**不得开工**，除非作者明确放行并把放行范围记在该节里。

条件：

design通过。

review无Blocker。

## Rule

如果改动涉及性能：

必须执行Baseline。

Baseline必须发生在：

Phase P5 Implementation  
之前

## Actions

运行当前版本。

记录：

### Environment

必须包含：

- Git commit hash
- GPU型号
- CUDA版本
- TensorRT版本

### Workload

必须固定：

- batch size
- sequence length
- dtype
- input shape

### Measurement

要求：

Warmup：

> =10

Iteration：

> =100

统计：

- median
- P95

## Output

生成：

benchmark_before.md

## Failure Handling

### Timeout / OOM

调整workload。

必须记录原因。

### Dependency Missing

本例外只适用于以下情形（封闭列表，不得扩大）：

- 环境无 GPU。
- 驱动或 CUDA 运行时不接受真机任务。
- 无法访问真机（例如只有沙箱环境）。

其余"跑不起来"（脚本报错、参数写错、超时）不属于本条，按 Timeout / OOM 或 Continuous Failure 处理；禁止用本条跳过判据（AGENTS §7）。**本条款不得用于跳过下面『三处留痕』里的任何一处，也不得把某一处改写成别的产物**（2026-10-05 补：REQ-016 的 P4 正是把 `summary.md` 换成了"本文件"、又把 `PROGRESS` 推到收口）。

写进 benchmark_before.md 的 `## Result` 一节：

N/A: <原因>

`<原因>` 是占位符，必须替换成本次的具体原因（一句话即可）；不允许原样保留 `reason` 之类的字面量。

此时 P7 不可执行：

- 不产生 P7 读数。
- Gate-B 记 N/A（缺依赖），既不算通过也不算失败。
- 三处必须同时留痕：
  - `STATE.md` 的 Next Action 或 Current Blockers 写 `Gate-B: N/A（缺依赖：<原因>）`。
  - `summary.md` 的 Performance 一节写 N/A + 原因。
  - `docs/PROGRESS.md` 的"已知问题与坑"留一条：本条 feature 的性能未验证及原因。
  - **时点冲突的处理（2026-10-05 补）**：若某处留痕在当前阶段**不可能产出**（例：`summary.md` 属 P8），
    按 `AGENTS.md` §5 第 5 条记"**未满足（阻塞）**"，并在当轮回复里列出"冲突双方 + 建议"，等作者裁决；
    **不得**改写成别的产物，**不得**悄悄延到"阶段/批次收口"。
- 禁止用"外部参考数据"自行替代 before / after；除非用户明确指定了可比来源与口径，并把来源、口径、可比性理由写进 benchmark.md。

### Continuous Failure

连续3次失败：

暂停。

请求用户介入。

## Exit Gate

Self Check:

benchmark_before.md 必须包含：

- environment
- workload
- measurement

缺依赖时按 N/A 判定：benchmark_before.md 记 `N/A: <原因>` 即视为通过本自检，但必须同时满足 Dependency Missing 的三处留痕要求（STATE.md / summary.md / PROGRESS.md），否则不通过。

Human Gate:

无。

## Output

更新 STATE.md 字段：

- phase: P5-Implementation
