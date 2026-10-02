# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-016-continuous-batching |
| phase | P3-Review |
| phase_index | 3 |
| status | waiting-human-gate |
| updated | 2026-10-02 |
| owner | Codex |

---

## Completed Artifacts

- `requirement.md`（P0-Requirement，2026-10-02 重做）
- `analysis.md`（P1-Analysis，2026-10-02 重做）
- `design.md`（P2-Design，2026-10-02 重做）

上一版（2026-10-01）的三件产物已由本轮重做覆盖，旧版可用
`git show 2a61b22:docs/dev/REQ-016-continuous-batching/<file>` 取回。

---

## Current Blockers

- **Gate-A 未通过**：设计需作者确认后才能进入 P3 Review 与 P4 Baseline。
  本轮新增 4 条待确认决策：D6（块池与引擎 cache 维度）、D7（常驻诊断开关）、
  D8（profile 归属校验）、D9（块分配失败语义），另 D5 的顺序拍板尚未由作者确认。
- 数值 / 性能验收必须在真机执行（沙箱无 GPU），按 `AGENTS.md` §0.3 需先获批准。

---

## Next Action

等待 Gate-A 确认。确认后：P3 Review（按 `checklists/cpp.md` / `checklists/llm_runtime.md` 自检）
→ P4 Baseline（先量 `batch = 1` 的对照数据：每步延迟、**prefill logits 与 K/V 各自的显存占用**、
块使用量；口径按 D7 关闭常驻诊断）。

P5 按里程碑分批实现与提交：S1 批量执行 → S2 资源生命周期 → S3 请求级调度，
不把三个里程碑压成一笔提交。

---

## Phase History

- 2026-10-01: P0 -> P1（上一版）
- 2026-10-01: P1 -> P2（上一版）
- 2026-10-01: P2 -> Gate-A（上一版，未通过）
- 2026-10-02: 作者批准重做 P0–P2；三件产物按代码核实结果重写
- 2026-10-02: P0 -> P1 -> P2 -> Gate-A（本轮，`status = waiting-human-gate`）

---

## Recovery Notes

- 设计要点：`design.md` 的 S1/S2/S3 里程碑、D1~D9 决策，以及 §不变量（4 条）。
- **本轮重做修正的内容**（下一会话不要改回去）：
  1. 元数据缓冲**不是**"每次请求都重新分配"——设备缓冲容量够时会复用；只有批大小增长才重分配。
     当前正确性依赖"decode 每步重绑"，不是"指针不变"。
  2. 引擎把 cache 张量第 0 维声明成 `ceil(n_positions / block_size)`，**同一个数**又当块表宽度用；
     运行时池可以更大（现有用例即是），但这条关系此前没有被写进任何契约。
  3. 随批增长的显存大头是 **prefill logits**（`S=512` 时约 98 MiB/条），不是 K/V 池（64 块约 72 MiB）。
  4. 建图代码不进引擎指纹，**改图必须手工 bump `graph_version`**。
- **写冲突**：本 feature 与 `REQ-017-llm-int8-quant`、`REQ-019-onnx-subgraph` 都要改同一段
  `LLMRunner` 代码。三者不能并行改同一个文件；D5 拍板为先做本 feature。
- 只要设计没有变化，恢复时不必重读 `docs/future_iterations.md`，看本文 + `analysis.md` 即可。
