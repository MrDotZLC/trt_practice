# Phase 4 原文中的 INT8 相关条目（逐字摘出）

> **STATUS: ARCHIVED｜阶段已交付（2026-09-26，Phase 4）**
> 本文只作设计与判据出处，**不代表现状**；现状见 `PROGRESS.md`，排查见 `TROUBLESHOOTING.md`。
> 冻结后不再更新；确需修订时另开文档，并在 `docs/README.md` §2 登记。

> **性质**：2026-10-01 按"需求归属"从 Phase 4 的两份原文中**逐字摘出**（只搬行、不改字）。
> 原位置各留一行指针；本节内容即 `REQ-008` 的**原文级证据**，与 `requirement.md` / `design.md` / `test_plan.md` 的现行陈述配套。

## 来自 `phase4_development_plan.md`

### §0.5 待拍板决策 —— D2（原第 61 行）

| **D2** | INT8 是否纳入 Phase 4 | **①** 不纳入；**②** 纳入，Q/DQ 显式量化；**③** 纳入，沿用隐式量化 + Calibrator | **✅ ②（纳入，Q/DQ）**。~~1660 Ti 无 FP16 Tensor Core、有 INT8 Tensor Core（历史 README 结论）→ FP16 只省带宽，INT8 才是真加速~~；**更正（2026-10-01，作者确认）**：1660 Ti（TU116）**没有 Tensor Core**（FP16 / INT8 都没有）→ 两者的收益都来自**显存带宽**（INT8 是 4→1 字节、FP16 是 4→2 字节），不是张量核心吞吐。**决策不变**：仍走 Q/DQ——理由是隐式量化在 TRT 10.15 已废弃，与 Tensor Core 无关。**代价**：Q/DQ 要单列任务 **P4-7** |

### §2 目标与范围 —— "按 D2 决定"（原第 130 行）

| **按 D2 决定** | INT8（P4-7） |

### §4 任务分解 —— P4-7（原第 210 行）

| **P4-7** | **（按 D2 条件式）INT8，走 Q/DQ 显式量化** | P4-6 | 见 **`docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md`**（独立计划：技术前提、工具链 A/B/C、任务 P4-7-0~5、判据纪律） | 高：TRT 10.15 已把 `kINT8`/`setDynamicRange`/Calibrator 全线废弃并指向"strong typing"；弱类型网络能否吃 Q/DQ 是**必须先验的前提** |

### §6 验收标准 —— INT8 达标项（原第 275 行）

- [x] （D2 选 ②）**INT8 达标**——但判据按实测改成了**分层**：主判据 = FP32 余量子集一致率（≥90%，实测 12/12 = 100%），整体一致率只作下界（≥30%，实测 38.3%）；ramp 判据**作废**（分布外输入，退化量随方案变）。理由见 `TROUBLESHOOTING.md` #29.3 / #29.4。

### §7 风险 —— INT8 走错路线（原第 288 行）

| INT8 走错路线（隐式量化已废弃） | 建了将来要拆的东西 | D2 推荐 Q/DQ；若选隐式，文档里写明折旧风险 |

### §10 执行结果 —— P4-7 回填（原第 334 行）

| P4-7 | ✅ 完成（2026-09-26） | INT8（Q/DQ 显式量化）全流程落地，子计划见 `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §7：P4-7-0 调研（S1/S2/S3 全闭环）→ P4-7-1 量化脚本 + 产物 `models/resnet18/resnet18_qdq.onnx`（**`prequant_dq` 形态，13.3 MB**，60 对 Q/DQ、zero_point 全 0）→ P4-7-2 `detailed_profiling` 开关 + 层信息自证（43 层 / 38 层含 Int8 / 4 层 `i8i8` tactic）→ P4-7-3 用例 R2.6（**分层判据**：余量子集 12/12 = 100%）→ P4-7-4 真机全量 182 条 / 1 红 / 0 跳过。**开放项**：per-channel 整网退化（原因未知，P4-INT8-a） |

## 来自 `phase4_test_plan.md`

### §2 用例表 —— R2.6（原第 84 行）

| R2.6 | `ResNet18Int8EngineTest.IsActuallyInt8` + `ResNet18Int8AccuracyTest.{RampInputIsOutOfDistribution,Top1AgreementOnRealImages}`（D2 选②） | Q/DQ INT8 引擎：① 层信息自证在跑 INT8；② ramp 只记录不判（分布外）；③ 真实图按**分层**判 | ① ≥20 层含 `Format/Datatype: Int8` 且 ≥1 个 `i8i8` tactic（对照 FP32 引擎为 0）；③ **主判据：FP32 余量子集（margin≥5）一致率 ≥90%**（实测 12/12），整体一致率 ≥30% 仅作下界（实测 38.3%） |

### §3 判据与出处 —— INT8 行（原第 106 行）

| INT8 | 需先测再定（argmax 一致 + top-1 一致率） | 校准集用现成的 500 张真实图；**不引用历史数字**（历史没留） |

### §7 结果回填 —— R2.6（原第 226 行）

| R2.6 | ✅ | 真机：QDQ 引擎 43 层 / **38 层含 Int8** / **4 层 `i8i8` tactic**（对照 FP32 引擎 0 层 Int8）；真实图 256 张整体一致 38.3%、**余量子集 12/12 = 100%**、`max_abs = 21.6`；ramp 只记录（0/8 不一致）。产物形态 `prequant_dq`（13.3 MB）。**开放项**：per-channel 整网退化（`TROUBLESHOOTING.md` #29/#30/#31，P4-INT8-a）。**"整体 38.3%"的成因已用交叉统计固化进用例输出**：按 FP32 余量分层 → `<1: 23.6%`、`1~2: 39.7%`、`2~5: 73.7%`、`5~10: 100%`、`>10: 100%`（58% 的样本余量<1，即类别本身不可判） |

### §7 结果回填 —— R2.6 的后续（原第 227 行）

| R2.6 的后续（判据升级） | 未开始 | **验收集到位后要补报的两项**：① 带真值标签的 top-1 **正确率**（不只是与 FP32 的一致率）；② 绝对误差在**余量子集**上的 p50 / p95 / p99（不报全样本 max）。规格与脚本：`mini_trt_llm/tools/validate/README.md` + `int8_eval.py`（`--self-test` 已进 ctest）；**唯一来源**：`docs/future_iterations.md` §1.6（P4-INT8-b，前置 = 联网取带标签验收集，须先获批） |
