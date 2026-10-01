# docs/dev 索引（统一编号）

> 本目录是技能 `trt-inference-engineering` 规定的开发 artifact 落点（流程见 `AGENTS.md` §5）。
> **每个目录的 `STATE.md` 是该条目状态的唯一落点**；本文件只做一览，**不复制状态数字**
> （基线 / 产物尺寸的唯一出处仍是 `PROGRESS.md`）。

## 0. 编号规则

- **命名**：`REQ-NNN-<slug>`；目录名与编号一一对应。**编号只承载身份、不承载状态**
  （状态写在 `STATE.md` 与本表）。理由：按状态分块编号会让条目一开工就得换号，而换号会打断所有引用。
- **编号范围 = 已立项或已交付的需求**；编号一经分配不再更改，新增条目按顺序往后取号。
- **历史条目**（`REQ-001`~`015`）的工作早已交付，目录只归档"当时要求什么"；
  `status = completed` **不代表仍在推进**。需求要变更 → 新立条目，不改写历史 `requirement.md`。
- **历史条目带四个产物**：`requirement.md`（要求什么）/ `design.md`（怎么设计）/
  `test_plan.md`（判据与用例）/ `STATE.md`（状态）。**每份文件首部都写明它从哪份历史文档的哪一节抽取**
  （本文不复制这些映射）。判据原本被切成两半（目标在阶段计划、判据在测试计划），
  现在分别落进 `requirement.md` 与 `test_plan.md`。
- **归档原文**：`REQ-001`~`009` 还收着**从 `docs/` 根迁入的原始文档**（如
  `phase2_development_plan.md`），**文件名保持不变**，因此原文里的锚点 ID（`PH2-*` 等）继续有效。
  它们是**不再更新的完整记录**（执行回填、逐条判据状态、原始测量表都在里面），
  **不受"单文档 ≤200 行"约束**——该约束只针对活文档。
- **每个条目自包含**：需求、设计、判据、状态、原始记录**都在同一个目录里**；
  跨条目引用才写完整路径。

## 1. 编号总表（19 条）

| 编号 | 目录 | 类别 | phase | status | 一句话 |
|---|---|---|---|---|---|
| `REQ-001` | `REQ-001-bootstrap` | 历史 | P9-Interview | completed | 骨架、依赖接入与基础设施 |
| `REQ-002` | `REQ-002-plugins` | 历史 | P9-Interview | completed | 插件基类 + 三个 Plugin + 采样器 kernel |
| `REQ-003` | `REQ-003-test-infra` | 历史 | P9-Interview | completed | 全流程测试基建与收尾 |
| `REQ-004` | `REQ-004-gpt2-native` | 历史 | P9-Interview | completed | GPT-2 原生构建与自回归循环 |
| `REQ-005` | `REQ-005-diagnostics-fix` | 历史 | P9-Interview | completed | 诊断输出契约修复 + CUDA 环境显式判定 |
| `REQ-006` | `REQ-006-gpt2-onnx` | 历史 | P9-Interview | completed | ONNX 路径与子图识别 |
| `REQ-007` | `REQ-007-resnet18` | 历史 | P9-Interview | completed | CV 路径（ONNX + 原生 + Runner） |
| `REQ-008` | `REQ-008-int8-qdq` | 历史 | P9-Interview | completed | INT8 走显式 Q/DQ |
| `REQ-009` | `REQ-009-retire-legacy` | 历史 | P9-Interview | completed | 下线两个历史示例工程 |
| `REQ-010` | `REQ-010-bpe-tokenizer` | 历史 | P9-Interview | completed | GPT-2 原生 BPE Tokenizer |
| `REQ-011` | `REQ-011-int8-criteria` | 历史 | P9-Interview | completed | INT8 判据的离线口径定义（仅离线一半） |
| `REQ-012` | `REQ-012-sampler-kernel` | 历史 | P9-Interview | completed | 采样器高性能 kernel（部分撤回） |
| `REQ-013` | `REQ-013-perf-profile` | 历史 | P9-Interview | completed | decode 性能画像与可复现测量方法 |
| `REQ-014` | `REQ-014-attention-splitk` | 历史 | P9-Interview | completed | 长上下文 attention 的上下文维切分 |
| `REQ-015` | `REQ-015-int8-perchannel` | 历史 | P9-Interview | completed | per-channel 整网退化根因 |
| `REQ-016` | `REQ-016-continuous-batching` | 进行中 | P3-Review | waiting-human-gate | 批量 > 1 + 每序列 K/V 与最小调度 |
| `REQ-017` | `REQ-017-llm-int8-quant` | 进行中 | P3-Review | waiting-human-gate | 语言模型权重量化；KV 缓存量化另立里程碑 |
| `REQ-018` | `REQ-018-gpt2-fp16-nan` | 进行中（bugfix） | B2-MinimalFix | waiting-human-gate | 唯一的按设计红：FP16 端到端 NaN |
| `REQ-019` | `REQ-019-onnx-subgraph` | 进行中 | P3-Review | waiting-human-gate | 外部图子图替换；前置 = 先跑跨构建对照 |

## 2. 不参与编号的内容

**`future_iterations.md` 的未触发条目与冻结条目保持原样、不进本编号体系**（作者 2026-10-01 定向）。
它们由该文件自己保存：目标 / 触发条件 / 前置 / 判据 / 优先级都在那里，唯一事实来源不变。

不预建目录的两个理由：① 技能把 `status` 限定为 `in-progress` / `waiting-human-gate` /
`completed` —— **"未开始"没有合法取值**；② 建目录就要复制一份内容，必然漂移。

条目被点名开工时，才建 `REQ-0NN-<slug>/`，并把需求正文从该文件搬进 `requirement.md`。

## 3. 过渡期权威与删除

同一需求若**同时**存在于本目录与历史文档（即迁入本目录各条目下的 `phaseN_*.md`），
**以本目录为准**；原文只作"当时是怎么写的"快照。

**旧文档的删除是最后一步**（这是规则）；**当前的待办状态与引用影响清单要求见
`PROGRESS.md` + `DEC-REQ-NUMBERING`**（本文件只写规则，不写状态）。

## 4. 旧锚点 → 新落点（删除旧文档时的改指映射）

仓外的代码 / 工具 / 脚本引用历史文档时通常写成"路径 + 锚点"。锚点搬到这里之后，改指按下表
（**这是删除那一步的输入**）：

| 旧锚点 | 原文位置（已迁入下表条目） | 现在的现行落点 |
|---|---|---|
| `PH1-RISKS` | `docs/dev/REQ-002-plugins/phase1_development_plan.md` §9 | `REQ-002-plugins/design.md` |
| `PH1-DECISIONS`（D1–D5，含 D4 阈值） | `docs/dev/REQ-002-plugins/phase1_development_plan.md` §10.1 | `REQ-002-plugins/design.md` |
| `PH3-ASSETS` | `docs/dev/REQ-006-gpt2-onnx/phase3_development_plan.md` §0.1 | `REQ-006-gpt2-onnx/design.md` |
| `PH3-GAPS`（G1c / G5 / G6） | `docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` §5 | `REQ-006-gpt2-onnx/test_plan.md`（G6 另见 `REQ-013`） |
| `PH4-LEGACY-FINDINGS` / `PH4-LEGACY-INHERIT` | `docs/dev/REQ-007-resnet18/phase4_development_plan.md` §1.1–§1.3 | `REQ-007-resnet18/design.md` |
| `PH4-ONNX-SOURCE-DECISION` / `PH4-CV-PROFILE` | 同上 §0.5 / §3.1 / §3.5 | `REQ-007-resnet18/design.md` |
| `PH4-TEST-INPUTS` / `PH4-CRITERIA` | `docs/dev/REQ-007-resnet18/phase4_test_plan.md` §3 / §4 | `REQ-007-resnet18/test_plan.md` |
| `PH4-INT8-TOOLCHAIN` | `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §1.3 | `REQ-008-int8-qdq/design.md` |
| `PH4-INT8-CRITERIA` | 同上 §4 | `REQ-008-int8-qdq/`（`requirement.md` + `test_plan.md`） |
| 阶段 0 的改名实验依据（无 ID） | `docs/dev/REQ-009-retire-legacy/phase5_development_plan.md` §3 | `REQ-009-retire-legacy/test_plan.md` |
| G6 测量协议 | `docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` §5 / 迭代开发计划 §11.3 | `REQ-013-perf-profile/design.md` |

## 5. 每个目录的内容清单（与目录实际内容一致）

| 条目 | 活文档 | 迁入的归档原文（文件名不变） |
|---|---|---|
| `REQ-001` | STATE / requirement / design / test_plan | `phase0_development_plan`、`phase0_code_review_plan`、`phase0_model_loading_test_plan` |
| `REQ-002` | 同上 | `phase1_development_plan`、`phase1_test_plan` |
| `REQ-003` | 同上 | `phase1_5_development_plan`、`phase1_5_test_plan` |
| `REQ-004` | 同上 | `phase2_development_plan`、`phase2_test_plan` |
| `REQ-005` | 同上 | `phase2_supplement_plan` |
| `REQ-006` | 同上 | `phase3_development_plan`、`phase3_test_plan` |
| `REQ-007` | 同上 | `phase4_development_plan`、`phase4_test_plan`（两者的 **INT8 条目已按需求摘出**，原位留指针） |
| `REQ-008` | 同上 | `phase4_int8_plan` + `phase4-excerpts`（从 REQ-007 的两份原文**逐字摘出**的 INT8 条目） |
| `REQ-009` | 同上 | `phase5_development_plan` |
| `REQ-010`~`015` | 同上 | 无（来源是迭代计划与迭代测试计划，二者仍是活文档） |
| `REQ-016`~`019` | STATE / requirement / analysis / design | 无（进行中，尚无归档原文） |

**执行回填明细、逐条判据状态、原始测量表**都在上表的"归档原文"里（不再单独上浮）；
活文档里只在 `test_plan.md` 的 `Actual Result` 保留结论摘要。
