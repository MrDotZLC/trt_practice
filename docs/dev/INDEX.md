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
- **判据有两半**：历史条目的目标在计划侧、判据的另一半在测试计划侧。逐条出处写在各自
  `requirement.md` 的首部（本文不复制）。

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

同一需求若**同时**存在于本目录与历史文档（`docs/phaseN_*.md` 等），**以本目录为准**；
历史文档在删除前只作"当时是怎么写的"快照。

**旧文档的删除是最后一步**（这是规则）；**当前的待办状态与引用影响清单要求见
`PROGRESS.md` + `DEC-REQ-NUMBERING`**（本文件只写规则，不写状态）。
