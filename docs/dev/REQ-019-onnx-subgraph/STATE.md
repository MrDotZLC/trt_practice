# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-019-onnx-subgraph |
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

- **Gate-A 未通过**：路线（Python 侧重写 vs C++ 侧图变换）与"是否真的做替换"需要拍板。
- **前置 PF-7 未跑**：ONNX 与原生 prefill 的跨构建对照做过两次、**方向相反**
  （一次 ONNX 慢 22%、一次 ONNX 快 24%），差异小于构建间噪声 → **"替换能带来什么收益"
  目前无法论证**。PF-7 必须在真机跑（≥3 次构建 / ≥20 次推理），按 `AGENTS.md` §0.3 需批准。
- `docs/future_iterations.md` §6.2（自定义算子导出）是本条的实现前置，尚未做。

---

## Next Action

等待 Gate-A 确认。确认后**第一步不是写代码，而是跑 PF-7**（命令与口径见
`docs/future_iterations_development_plan.md` §11.9 与 `docs/future_iterations_test_plan.md` §10.2）。

PF-7 的结论决定这条 feature 的走向：

- 若外部图确实显著更慢 → 替换有明确收益目标，按 S1 → S2 → S3 推进；
- 若两者无显著差异 → **"替换的收益"不成立**，应改为记录结论并考虑关闭本条
  （这同样是有价值的结论，不是失败）。

---

## Phase History

- 2026-10-01: P0 -> P1
- 2026-10-01: P1 -> P2
- 2026-10-01: P2 -> Gate-A（`status = waiting-human-gate`）

---

## Recovery Notes

- 本条与 `REQ-017-llm-int8-quant` 的 ONNX 路线耦合：若两者都做，外部图路径的 decode 图应在这里一并补齐，
  否则 INT8 走 ONNX 路线时要再改一次同一段代码。
- 与 `REQ-016-continuous-batching` 的写冲突点在运行时入口（外部图路径要接进运行时）。
- 前置事实来源：`docs/future_iterations.md` §10.1 / §10.2、§0.1 的 PF-7 行、
  `docs/TROUBLESHOOTING.md` + TS-017（两条路的 I/O 契约差异）。
