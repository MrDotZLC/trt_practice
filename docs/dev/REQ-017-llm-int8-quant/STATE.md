# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-017-llm-int8-quant |
| phase | P3-Review |
| phase_index | 3 |
| status | waiting-human-gate |
| updated | 2026-10-01 |
| owner | Codex |

---

## Completed Artifacts

- `requirement.md`（P0-Requirement，2026-10-01）
- `analysis.md`（P1-Analysis，2026-10-01）
- `design.md`（P2-Design，2026-10-01）

---

## Current Blockers

- **Gate-A 未通过**：路线选择（原生 DQ vs ONNX Q/DQ）与"KV cache 是否在本轮做"需要作者拍板。
- **判据阈值尚无出处**：INT8 的绝对误差界需要先量再定（`AGENTS.md` §7 禁止来路不明的阈值）。
- 真机验收需批准；若走 ONNX 路线还需先解决 `docs/future_iterations.md` §10.1 的 I/O 契约。

---

## Next Action

等待 Gate-A 确认。确认后：P3 Review → P4 Baseline（量 FP32 基线的引擎体积 / 显存 / 每步延迟，
作为 INT8 的对照），再进入实现。

---

## Phase History

- 2026-10-01: P0 -> P1
- 2026-10-01: P1 -> P2
- 2026-10-01: P2 -> Gate-A（`status = waiting-human-gate`）

---

## Recovery Notes

- **必须先读 `docs/PROGRESS.md` §3.0j**：ResNet18 的 INT8 per-channel 退化根因是产图脚本取错
  权重源（未折 BN），这条教训直接适用于 LLM 权重量化——**量化对象与 scale 来源必须是同一份张量**。
- **依赖关系**：本 feature 的 ONNX 路线依赖 `REQ-019-onnx-subgraph`（ONNX 路径目前没有
  decode 图与 KV cache）；原生路线不依赖任何其他 feature。
- ~~**文档矛盾待修**~~ **已修（2026-10-01，作者确认）**：`docs/future_iterations.md` §1.2 与
  `docs/phase4_development_plan.md` D2 / 依据表里的"有 INT8 Tensor Core"已更正为
  "TU116 无 Tensor Core、收益来自显存带宽"（冻结文档按 `docs/README.md` §7 用日期批注保留原文）。
