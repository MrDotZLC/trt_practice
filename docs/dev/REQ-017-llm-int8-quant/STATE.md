# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-017-llm-int8-quant |
| phase | P5-Implementation |
| phase_index | 5 |
| status | in-progress |
| updated | 2026-10-05 |
| owner | Codex |

---

## Completed Artifacts

产物清单与计数口径以 `.agents/skills/trt-inference-engineering/workflows/feature.md` 的
`## Required Artifacts`（10 件）为唯一来源，本节只登记现状，不复述定义。

**实到 6 / 10**：`STATE.md`、`requirement.md`、`analysis.md`、`design.md`、`review.md`、
`benchmark_before.md`。

缺 4 件，且分属后续阶段：`test_plan.md`（P6）、`benchmark.md`（P7）、`summary.md`（P8）、
`interview_notes.md`（P9）。**开发前的文档部分（P0–P3）至此齐套。**

- `requirement.md`（P0-Requirement，2026-10-01；2026-10-05 按作者裁决把 K/V 缓存量化的
  可行性评估移入 Excluded）
- `analysis.md`（P1-Analysis，2026-10-05 重做：补 `Terminology`，修正 INT8 取数路径的描述，
  按 REQ-016 之后的运行时现状重写）
- `design.md`（P2-Design，2026-10-05 重做：补 `Requirement Coverage`，D1 改为路线 C，
  新增 D5 / D6 / D7，登记三条假定与核对义务）
- `review.md`（P3-Review，2026-10-05 新产出：四份 checklist 的 52 条 [P0]/[P1] 逐条结论 +
  三张对齐表 + "已定 / 必须拒绝"条目的来源核对；**P0 = 0，P1 = 7**）
- `benchmark_before.md`（P4-Baseline，2026-10-05：记 `N/A: <原因>` + 恢复后必补的五项 +
  "三处留痕"逐项打勾；其中 `summary.md` 与 `PROGRESS.md` 两处按阻塞上报，见该文件末节）

---

## Current Blockers

- **Gate-A 待作者裁决**（`status = waiting-human-gate`）。7 条 P1 见 `review.md`，其中需要你拍板的
  三条是：① D6 的默认量化清单（本设计按"每步读取量"推导，未经你点名）；② D7 的激活精度与
  DQ 融合（若只有 FP16 激活才融合，会依赖 `REQ-018` 的 FP16 NaN）；③ `models/gpt2/` 权重资产
  的存在性与规格。
- **写冲突（2026-10-03 登记，仍有效）**：本 feature 与 `REQ-016-continuous-batching`（仍在
  `P5-Implementation / in-progress`）、`REQ-019` 都要改同一段运行时入口。顺序由作者定：
  **REQ-016 先做**（它定义批量契约）。它交付前，本 feature 不得开代码。
- **`Gate-B: N/A（缺依赖：本设备暂不支持真机测试，无 GPU / 无 nvcc·cmake·TensorRT；作者
  2026-10-05 指示真机测试搁置）`** —— P4 未取基线、P7 随之不可执行，既不算通过也不算失败。
- **P4 的"三处留痕"有两处按阻塞上报**（细节与口径见 `benchmark_before.md` 的末节）：
  ③ `summary.md` 的 Performance 一节 = P8 产物，P4 时点不可能产出；④ `docs/PROGRESS.md` 的
  "已知问题与坑"一条属"结构类文档"（`AGENTS.md` §0.2），**先确认再改**，本轮未动。
- **Gate-A 的 7 条 P1 仍未逐条确认**：作者 2026-10-05 的指令（"真机测试搁置、先完成开发工作"）
  被当作**放行**处理，放行范围与不适用条款见 `## 判据对照`。**D6 的量化清单**与 **D7 的激活
  精度**仍是未确认项——本轮把它们做成"清单驱动 + 可配置"，代码不替你锁定取值。
- **P4 / P6 / P7 的真机项当前不可执行**：本机沙箱无编译器 / 无 GPU。`CPP-P0-8`（编译）与
  `CUDA-P0-6`（benchmark 证明）两条 P0 判据的验证时点分别在 P5 与 P7，已按"P3 不适用"记入
  `## 判据对照`（见 `review.md` 的 P1-5）。

---

## Next Action

1. **继续 P5**：按 `## Implementation Plan` 的顺序把剩下的两处落完
   （`builder.*` 的清单入口 / 指纹 / `graph_version`，`gpt2_model_builder.*` 的 int8 常量 + DQ）。
2. **P6**：写 `test_plan.md` 与 host/GPU 用例，并把 `quantize_gpt2.py --self-test` 注册进 ctest。
3. **待你点头的两件**：③ `summary.md` 的 Performance N/A（P8 补）与 ④ `docs/PROGRESS.md` 的
   已知问题一条——后者属结构类文档，需你确认后我再写。
4. **真机窗口恢复后**：先做 `benchmark_before.md` 里列的 D7 最小图实验（它决定这条路线的收益
   是否成立），再按 Gate-B 的口径做 A/B。
5. **P8 必清**：`summary.md` 的 Performance 一节写 `N/A + 原因`（P4 的留痕 ③）。

---

## Implementation Plan

- **当前修改模块**：Python 离线量化脚本（新）；量化清单与权重取数（新）；GPT-2 原生建图
  （扩展）；引擎构建入口与指纹（扩展）。**不改运行时、不改 kernel、不改插件、不改 cache**（D3）。
- **预计文件与落地顺序**（✅ = 已落，⏳ = 未落）：

  | # | 文件 | 内容 | 状态 |
  |---|---|---|---|
  | 1 | `tools/convert/quantize_gpt2.py`（新） | 算 scale → 写 int8 权重 + 清单；含 4 道护栏自检 | ✅ |
  | 2 | `include/.../core/quant_spec.hpp` + `src/core/quant_spec.cpp`（新） | 清单解析与拒绝面 | ✅ |
  | 3 | `src/utils/safetensors_loader.cpp` + 头 | 只多开 int8→int8 的零拷贝 | ✅ |
  | 4 | `include/.../core/weight_loader.hpp` + `src/core/weight_loader.cpp` | 独立 loader 读量化产物；`ResolveSourceKey` 供来源核对 | ✅ |
  | 5 | `include/.../core/imodel_builder.hpp` | `BuildOptions::quant` 透传 | ✅ |
  | 6 | `include/.../core/builder.hpp` + `src/core/builder.cpp` | `Config::quant_manifest`；载入清单；清单与 int8 权重进指纹；`graph_version` bump | ⏳ |
  | 7 | `include/.../core/gpt2_model_builder.hpp` + `src/core/gpt2_model_builder.cpp` | int8 常量 + DQ；清单"全消费"校验 | ⏳ |
  | 8 | `tests/test_quant_spec.cpp`（新，host）+ `tests/CMakeLists.txt` | 清单解析/拒绝面；注册 Python 自检 | ⏳ |
  | 9 | `tests/test_gpt2_int8_weights.cpp`（新，GPU） | 引擎构建 + 逐层精度自证 + 与 FP32 对照（沙箱跳过） | ⏳ |

- **测试方式**：
  - **host（沙箱可跑，本轮已跑通一条）**：`python tools/convert/quantize_gpt2.py --self-test`
    → 4 道护栏（重复生成 / 缺张量 / 非 2-D / 产物被改坏）+ scale 与摘要的身份自检全部通过；
    `QuantSpecTest.*` 与加载器 int8 零拷贝用例待补。
  - **GPU（真机，本次搁置）**：int8 引擎构建、逐层精度自证、与 FP32 的数值对照、引擎体积。
  - **编译**：本环境无编译器 → P5 Exit Gate 的"编译通过 / 无新增 warning"**未验证**，
    与 `REQ-016` 同源。

---

## Phase History

- 2026-10-01: P0 -> P1
- 2026-10-01: P1 -> P2
- 2026-10-01: P2 -> Gate-A（曾被写成 `waiting-human-gate`，但 `review.md` 缺失；见 Recovery Notes）
- 2026-10-05: 作者裁决（D1 = C / D3 维持只做权重 / 头文件与权重按环境当既有前提）→
  P1 + P2 重做
- 2026-10-05: P2 -> P3（产出 `review.md`：P0 = 0，P1 = 7）→ P3 -> Gate-A
  （`status = waiting-human-gate`）
- 2026-10-05: 作者指令"真机测试搁置、先完成开发工作" → 记为 Gate-A 放行（放行范围见
  `## 判据对照`）→ P4（Dependency Missing，记 N/A）→ P5-Implementation（`status = in-progress`）

---

## Recovery Notes

- **本轮修正的五处偏差**（指针，不重抄正文）：`analysis.md` 末尾的"文档与现状的矛盾"表（4 行）
  + `design.md` 开头的"修正记录"（含缺 `Requirement Coverage` 一处）。
- **作者裁决记录（2026-10-05）**：① D1 采纳 C；② K/V 缓存量化维持 D3、另立里程碑；③ 设备信息与
  产物规格按 `AGENTS.md` §1 + `docs/PROGRESS.md` §6.5 当作既有前提。第 ③ 条已按"假定 + 核对
  义务"落进 `design.md`，**不得**当成已验证结论引用。
- **D6 的清单是设计推导的默认值，不是作者点名**（`design.md` D6 的"来源登记"）：清单内容在
  `requirement.md` 与既有文档里没有可引用的依据，因此不作为"已定"条目使用，改由 Gate-A 裁决
  （`review.md` 的 P1-1）。这是本 feature 唯一一处"默认值待确认"，**不要**在下个会话当成已定。
- **旧 STATE 的流程缺口**：`phase = P3-Review / status = waiting-human-gate` 曾与"`review.md`
  不存在"并存，即"没有评审也能挂 Gate-A"。这类"状态与产物不一致"按 `AGENTS.md` §5 第 4 条属
  必须当场修的项，本轮已补齐 `review.md` 并把阶段历史写清。
- **必须先读**：`docs/PROGRESS.md` §3.0j（ResNet18 INT8 per-channel 的根因是"scale 取自未折 BN
  的权重、量化对象是已折 BN 的权重"）——这条教训直接适用于 LLM 权重量化：**量化对象与 scale
  来源必须是同一份张量**。本 feature 把它落成 `design.md` 的文件级判据。
- **依赖关系**：ONNX 路线依赖 `REQ-019-onnx-subgraph`（该路径没有 decode 图与 K/V cache）；
  本轮采纳的路线 C 不依赖任何其他 feature；但 D7 在"只有 FP16 激活才融合"的情形下会依赖
  `REQ-018`。
- **与后续 feature 的接口面**：`REQ-016` 的 `design.md` D11 已给出"元素宽度只在一处映射"的
  目标口径；本轮核对后的现状是它尚未成立（见 `analysis.md` 的 Current Architecture §6），
  S2 开工时先读那一节再对齐。
- **顺序偏差（本轮自曝，按 SKILL 的会话收尾规则落载体）**：`AGENTS.md` §5 要求"改代码的前置 =
  先把 Implementation Plan 记进 STATE.md"，而我在补本节之前**已经落了两批代码**
  （`quantize_gpt2.py` 与 quant_spec / loader / weight_loader / imodel_builder）。代码内容没有
  偏离 design（D1 路线 C），偏差在**顺序**：先写后记。已在本轮把 Implementation Plan 补进
  `## Implementation Plan`，并把这条记在这里，避免下个会话把它当成"计划本来就在"。
- **Gate-A 的放行口径**：作者 2026-10-05 的"真机测试搁置、先完成开发工作"被当作放行使用，
  但它**不等于** `review.md` 里 7 条 P1 已逐条确认。因此 D6（量化清单）与 D7（激活精度）在
  代码里被实现成**清单驱动 + 可配置**：脚本与构建器都不替你锁死取值，改单只需换清单 + 重建。

---

## 判据对照

口径：只在异常路径填写，清单只引用出处、逐项打勾，不转述条文
（`.agents/skills/trt-inference-engineering/SKILL.md` 的 Mandatory #8 / #10）。
本轮发生了**阶段切换**（P1 / P2 重做、P3 首次产出）与四条**不适用 / 放行**，故对表一次。

| 判据（出处） | 状态 | 证据 / 放行 |
|---|---|---|
| P1 Exit Gate 1：修改入口在哪里（`phases/p1_analysis.md`） | 满足 | `analysis.md` 的 Extension Point 与 Relevant Code Path |
| P1 Exit Gate 2：数据如何流动（同上） | 满足 | `analysis.md` 的 Data Flow |
| P1 Exit Gate 3：哪些模块受影响（同上） | 满足 | `analysis.md` 的 Module Structure |
| P2 Exit Gate 1：Requirement Coverage 表存在且无空落点（`phases/p2_design.md`） | 满足 | `design.md` 的 Requirement Coverage（9 行 = Included 4 + AC 5） |
| P2 Exit Gate 2：是否明确接口（同上） | 满足 | `design.md` 的 Module Design 与 Data Structure 的契约表 |
| P2 Exit Gate 3：是否说明资源生命周期（同上） | 满足 | `design.md` 的 Resource Lifecycle |
| P2 Exit Gate 4：是否说明性能影响（同上） | 满足 | `design.md` 的 Performance Consideration |
| P3 表一：checklists 结论表行数 = [P0]/[P1] 项总数（`phases/p3_review.md` 的 Exit Gate） | 满足 | `review.md` 的 Checklist Result：52 行（cpp 12 + cuda 14 + tensorrt 13 + llm_runtime 13） |
| P3 表二：需求落点表行数 = Included + AC（同上） | 满足 | `review.md` 的 Requirement Coverage Result：9 行 |
| P3 表三：术语定义表行数 = 模糊名词数（同上） | 满足 | `review.md` 的 Terminology Check：10 行 |
| K/V 缓存量化的可行性评估（原 Included 5） | 不适用（移出本轮） | **放行记录**：作者 2026-10-05 裁决"另立里程碑"；`requirement.md` 已把该项移入 Excluded（含理由）。**放行范围** = 本轮不交付该评估的独立产物；**不得**用于跳过 Included 1–4 与 AC1–AC5 |
| D6 量化对象清单 | 满足（默认值待 Gate-A 裁决） | `design.md` D6 的默认清单 + `review.md` 的 P1-1；来源口径 = 按"每步读取量"推导（**非**作者点名） |
| `CPP-P0-8`（Debug/Release 编译）与 `CUDA-P0-6`（benchmark 证明）在 P3 的可判性 | 不适用 | 理由：验证时点分别在 P5 的 Exit Gate 与 P7；已登记为 `review.md` 的 P1-5。**不得**用于跳过这两条判据本身，也不得用于跳过 `review.md` 的条款④ |
| `design.md` 的「例外与不得跳过」三条（S2 不纳入本轮 / 运行时本轮不改 / K/V 移出本轮） | 不适用 | **放行记录**：作者 2026-10-05 裁决"只做权重、缓存另立里程碑"；**放行范围** = 本轮不为 S2 出实现落点、运行时契约不动。**不得**用于跳过各条款点名的判据（原文见 `design.md` 该节） |
| Gate-A 的 7 条 P1（`review.md` 的 P1 Risks） | 未满足（阻塞） | **放行记录**：作者 2026-10-05 指令"真机测试搁置、先完成开发工作"。**放行范围** = 允许在 P1 未逐条确认的前提下进入 P4 / P5；**不得**用于跳过 P4 的三处留痕、**不得**用于跳过 AC1–AC5，也不得把 D6 / D7 当成已确认 |
| P4 留痕 ①：`benchmark_before.md` 记 `N/A: <原因>`（`phases/p4_baseline.md` 的 Dependency Missing） | 满足 | `benchmark_before.md` 的 `## Result` |
| P4 留痕 ②：本文件写 `Gate-B: N/A（缺依赖：<原因>）` | 满足 | `## Current Blockers` 首条 |
| P4 留痕 ③：`summary.md` 的 Performance 一节写 N/A + 原因 | 未满足（阻塞） | P8 产物，P4 时点不可能产出。按 `AGENTS.md` §5 第 5 条登记并上报（`benchmark_before.md` 的留痕表第 ③ 行）；义务锚定在 `## Next Action` 第 5 条，**P8 必清** |
| P4 附加留痕：`docs/PROGRESS.md` 的"已知问题与坑"一条 | 未满足（阻塞） | 属 `AGENTS.md` §0.2 的"结构类文档"→ 先确认再改，本轮未动；等你点头后追加。**不得**因为它没写就当作 P4 已完成 |
| P5 的 Implementation Plan 前置（`phases/p5_implementation.md` 的 Actions 1） | 满足（补记） | 本轮补进 `## Implementation Plan`；顺序偏差见 `## Recovery Notes` 的同名条目 |
