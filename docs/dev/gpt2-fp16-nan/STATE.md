# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | bugfix |
| feature | gpt2-fp16-nan |
| phase | B2-MinimalFix |
| phase_index | 2 |
| status | waiting-human-gate |
| updated | 2026-10-01 |
| owner | Codex |

---

## Completed Artifacts

- `requirement.md`（B0-Reproduce：问题、复现资产、验收判据，2026-10-01）
- `analysis.md`（B1-Diagnose：现有证据链、已否证方向、待查方向，2026-10-01）
- `design.md`（B2 候选修复方案与改动面 / 预算判定，2026-10-01）

---

## Current Blockers

- **未定位到具体算子**：5 轮真机往返只定位到"性质"（构建相关的不稳定、幅值远未溢出），未定位到"哪一层、哪个算子"。
- **B2 预算可能超限**：Bugfix 默认预算是"文件 ≤ 3 / 行数 ≤ 100"；三条候选路线（尤其"逐算子
  二分"与"激活缩放"）**预计超限** → 必须停下来做 **Bugfix → Feature Decision**（技能禁止自动转换）。
- **策略冲突**：本条此前被决定为"**按政策不修**"（`PROGRESS.md` §5.11、`TROUBLESHOOTING.md` + TS-018.1）；本次立项等于改主意，需作者明确同意。
- 复现与验证都需要真机（沙箱无 GPU），按 `AGENTS.md` §0.3 需先获批准。

---

## Next Action

等待作者裁决三件事：

1. 是否同意撤销"按政策不修"的决定；
2. 是否批准 B1 的下一轮真机诊断（探针从"只导第 0 层"扩到全部层）；
3. B2 若超预算：立为 Feature（走完整设计流程）还是停在诊断结论。

---

## Phase History

- 2026-10-01: B0 -> B1（现有复现器与诊断仪器已存在，补文档）
- 2026-10-01: B1 -> B2（根因未定位，B2 预算待判定）
- 2026-10-01: B2 -> Human Gate（`status = waiting-human-gate`）

---

## Recovery Notes

- **路由**：本条走 Bugfix Workflow（技能把"结果错误"归 Bugfix；现象 = 贪心恒为 0、logits 全 NaN）。
- 历史证据在 `docs/TROUBLESHOOTING.md` + TS-018.1（5 轮真机定位表），结论索引在 `PROGRESS.md` §5.11；
  本文档不复制那些数据，只写"下一步查什么"。
- 复现器：`mini_trt_llm/tests/test_gpt2_generate.cpp` 的
  `Gpt2GenerateTest.RealGpt2Fp16GreedyMatchesReferenceTokens`（当前**按设计红**）与
  `Fp16PrefillOutputsDiagnostic`（诊断打印，按设计通过）。
- **改这条会动测试基线**：它目前是全量里唯一的红，修好之后红数归零，
  需要同步回填 `PROGRESS.md` 的当前基线 / §5.11 / `interview_summary.md`。
