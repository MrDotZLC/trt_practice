# Project Summary

> **性质**：历史交付总结回填（2026-10-01）。来源：本目录 `phase4_int8_plan.md` §7 执行结果。

## Problem

隐式校准路线已废弃；默认量化配置非对称、被运行时拒绝；低精度"怎么算合格"没有出处。

## Solution

自己产**对称** Q/DQ 量化图，复用外部图构建入口；用逐层精度自证"真的在跑低精度"；
把判据写成有出处的纪律。

## Architecture

```text
产图脚本 → 对称 Q/DQ 量化图 → 外部图构建入口 → 引擎（构建期融合 Q/DQ 与卷积）
                              ↓
                    逐层精度 / tactic 可读（自证）
```

## Implementation

工具链调研（4 个现场实验）→ 产图脚本 → 逐层精度开关 → 精度用例 → 文档收口；
交付期间发现并修掉"尺度来源与量化对象不一致"的根因。

## Performance

**不适用**（无性能采集；收益在立项理由里是定性的"省带宽"）。

## Limitation

- **默认产物为 per-tensor**；per-channel 的整网退化根因后来单独结案（`REQ-015`）。
- 绝对误差界**未定**：当前只用"余量子集一致率"作判据，且该子集无真值标签 → 开放项。

## Future Work

- 带真值标签的验收集与绝对误差判据 → `docs/future_iterations.md` §1.6。
- per-channel 根因 → `docs/dev/REQ-015-int8-perchannel/`。
