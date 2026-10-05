# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-017-llm-int8-quant |
| phase | P6-Test |
| phase_index | 6 |
| status | in-progress |
| updated | 2026-10-05 |
| owner | Codex |

---

## Completed Artifacts

产物清单与计数口径以 `.agents/skills/trt-inference-engineering/workflows/feature.md` 的
`## Required Artifacts`（10 件）为唯一来源，本节只登记现状，不复述定义。

**实到 7 / 10**：`STATE.md`、`requirement.md`、`analysis.md`、`design.md`、`review.md`、
`benchmark_before.md`、`test_plan.md`。

缺 3 件，且分属后续阶段：`benchmark.md`（P7）、`summary.md`（P8）、`interview_notes.md`（P9）。

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
- `test_plan.md`（P6-Test，2026-10-05：四类用例清单 17 行 + `Requirement Traceability` 9 行 +
  "待真机窗口的数值对照"口径；结果一律记"未验证"）

---

## Current Blockers

- **Gate-A 已逐条裁决（2026-10-05）**：7 条 P1 的处置见 `## 判据对照` 的 `P1-1` ~ `P1-7` 行
  （D6 默认清单同意、D7 同意做、与 `REQ-018` 不合并、权重资产按设备纪律、收益量级真机后更新、
  编译 / benchmark 维持绑 P5 Exit Gate / P7）。**D6 的最终确认**放在 D7 之后（见
  `## Implementation Plan` 的"后续执行项"第 1–2 条）。
- **写冲突（2026-10-03 登记；2026-10-05 更新）**：本 feature 与 `REQ-016-continuous-batching`
  （仍在 `P5-Implementation / in-progress`）、`REQ-019` 共用运行时入口。原定"REQ-016 先做、
  本 feature 不得开代码"，实际按作者"真机测试搁置、先完成开发工作"的指令**并行推进**——该指令
  被当作对写冲突的解除（当轮已标明这是假设，作者未叫停）。若你要严格串行，请指出，我按你给的
  顺序退。
- **`Gate-B: N/A（缺依赖：本设备暂不支持真机测试，无 GPU / 无 nvcc·cmake·TensorRT；作者
  2026-10-05 指示真机测试搁置）`** —— P4 未取基线、P7 随之不可执行，既不算通过也不算失败。
- **P4 的"三处留痕"还剩一处按阻塞上报**（细节与口径见 `benchmark_before.md` 的末节）：
  ③ `summary.md` 的 Performance 一节 = P8 产物，P4 时点不可能产出；④ 的附加留痕已由作者
  2026-10-05 放行后追加（`docs/PROGRESS.md` §5.17）。
- **P4 / P6 / P7 的真机项当前不可执行（作者已确认）**：**作者 2026-10-05 确认"现在的设备没有
  编译和真机测试的条件"**——按设备纪律（`requirement.md` 的 `## Constraints`）落档，这不再是我
  的环境观察。因此：P5 Exit Gate 的编译项、P6 Exit Gate 的四类用例、P7 的基准，在本设备上都
  **无法收口**。`CPP-P0-8`（编译）与 `CUDA-P0-6`（benchmark 证明）两条 P0 判据的验证时点分别在
  P5 与 P7，已按"P3 不适用"记入 `## 判据对照`（见 `review.md` 的 P1-5）。
- **P5 未收口（实现面已提交；未完成项全部依赖环境）**：P5 的 Exit Gate 只剩"编译通过 / 无新增
  warning"与真机四项。**未完成项与原因的唯一来源** = `docs/PROGRESS.md` §4.7；本文件
  `## Implementation Plan` 的"真机必查"三条给出其中的技术细节。
- **P6 未收口（用例与判据已落，结果未取）**：`test_plan.md` 的四类用例要求**实际通过**，而本
  环境无编译器 / 无 GPU → 只落了"用例 + 判据 + 口径"，**不写通过**。解除条件 = 真机窗口
  （见 `## Next Action` 第 1 条）。

---

## Next Action

1. **真机窗口（把 P5 的 Exit Gate 与 P6 的 Exit Gate 一起收口）**：① `cmake --build build -j`；
   ② 按 `test_plan.md` 逐条跑 Unit / Integration / Regression / Failure 并回填 `## Actual Result`
   与 `## Status`；③ 顺带消掉"真机必查"三条（`addDequantize` 的约束、`wte` 新路径在 FP32/FP16 下
   的行为、D7 的引擎体积判据）——它们与 `test_plan.md` 的 I2 / I3 是同一件事。
2. **P7 / P8 / P9**：`benchmark.md` / `summary.md` / `interview_notes.md`，均等真机出数后写。
3. **待你点头的一件**：③ `summary.md` 的 Performance N/A（P8 时补；④ 已随 `PROGRESS` §5.17
   落地）。
4. **真机窗口恢复后**：先做 `benchmark_before.md` 里列的 D7 最小图实验（它决定这条路线的收益
   是否成立），再按 Gate-B 的口径做 A/B。
5. **P8 必清**：`summary.md` 的 Performance 一节写 `N/A + 原因`（P4 的留痕 ③）。
6. **`REQ-018` 依赖的分支处置（已裁决：不合并）**：作者 2026-10-05 定**不合并 feature**。
   若 D7 落在"只有 FP16 激活才融合"一支，处置 = 在 `REQ-018` 的 requirement 里加一条联合验收
   （"INT8 权重 + FP16 激活的 GPT-2 端到端可跑且数值达标"）——**该动作在那个时点再向作者点名**；
   `REQ-018` 的 requirement 本轮不动。

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
  | 6 | `include/.../core/builder.hpp` + `src/core/builder.cpp` | `Config::quant_manifest`；载入清单；清单与 int8 权重进指纹；`graph_version` 3→4 / 6→7 | ✅ |
  | 7 | `include/.../core/gpt2_model_builder.hpp` + `src/core/gpt2_model_builder.cpp` | int8 常量 + DQ；清单"全消费"校验；`source_key` 同源核对 | ✅ |
  | 8 | `tests/test_quant_spec.cpp`（新，host）+ `tests/CMakeLists.txt` | 清单解析/拒绝面；注册 Python 自检 | ⏳ |
  | 9 | `tests/test_gpt2_int8_weights.cpp`（新，GPU） | 引擎构建 + 逐层精度自证 + 与 FP32 对照（沙箱跳过） | ⏳ |

- **测试方式**：
  - **host（沙箱可跑，本轮已跑通一条）**：`python tools/convert/quantize_gpt2.py --self-test`
    → 4 道护栏（重复生成 / 缺张量 / 非 2-D / 产物被改坏）+ scale 与摘要的身份自检全部通过；
    `QuantSpecTest.*` 与加载器 int8 零拷贝用例待补。
  - **GPU（真机，本次搁置）**：int8 引擎构建、逐层精度自证、与 FP32 的数值对照、引擎体积。
  - **编译**：本环境无编译器 → P5 Exit Gate 的"编译通过 / 无新增 warning"**未验证**，
    与 `REQ-016` 同源。

- **后续执行项（2026-10-05 作者裁决"应该做就纳入开发计划"）**——顺序即优先级：

  1. **D7 最小图实验**（真机第一步）：单 Linear + int8 常量 + DQ，在"无 flag / kFP16"两种构建下
     各建一份；**判据 = 引擎体积必须下降**（不降 = DQ 被常量折叠 → 收益归零 → 回 Gate-A）。
  2. **D6 清单最终确认**：默认值已获同意（48 Linear + `wte`，`wte` / `lm_head` 为可摘除项）；
     确认时点放在 D7 之后。
  3. **P5 Exit Gate 的编译项** + 既有用例回归（真机窗口）。
  4. `addDequantize` 的 API / 广播约束核对，以及 `wte` 新路径在 FP32 / FP16 下的确认——与第 1 项
     同一个窗口做。
  5. **P6**：`test_plan.md` + 四类用例（含 P1-6 定的"短 + 长各一档"口径）——**需作者点名**。
  6. 真机出数后：把 P1-5 的两条结果**更新回 P3**（回填 `review.md` 对应行），并按 P1-4 用实测
     替换收益量级。

  环境依赖项的完整状态表仍是 `docs/PROGRESS.md` §4.7（**唯一来源**）。

- **P5 落地后的"真机必查"清单**（都不影响本轮的文档结论，但决定这条路线的成败）：

  1. **`addDequantize` 的 API 与约束**：本机没有 `NvInfer.h`，代码按作者"假设有头文件"的
     指令写（`network->addDequantize(input, scale, zeroPoint)`，scale / zeroPoint 取
     "与权重同秩、各维为 1"的构建期常量）。真机第一步核对签名与广播约束。
  2. **`wte` / lm_head 那条新路径**（TS-056 的发现 4 已改）：原来是"先 DQ 再转置"，
     转置折不了常量 → 每步真的转一遍 154 MB 权重。现在拆成
     `gather(int8) → DQ` 与 `transpose(int8) → reshape → DQ → MatMul`，形状操作都落在常量上。
     真机要确认两件事：① 这两种形状在 FP32 与 FP16 下都建得出来；② 转置确实被折成常量
     （逐层信息里不该出现每步执行的 Transpose，引擎体积也应随之下降）。
  3. **逐层精度自证要覆盖 DQ 层**：`detailed_profiling` 下确认量化层确实带 Int8，
    而不是被静默折叠。

### P5 设计符合性对账（静态，2026-10-05；P5 Exit Gate 的"修改符合 design"一支）

口径：逐条把 `design.md` 的落点与代码对账，**只记结论与证据指针**，不重抄 design 条文
（`SKILL.md` 的 Mandatory #8）。`CPP-P0-8`（编译）与 `CUDA-P0-6`（benchmark）**不在此表**——
它们的验证时点在 P5 真机窗口与 P7，已按"P3 不适用"记入 `## 判据对照`。

| design 落点 | 代码证据 | 结论 |
|---|---|---|
| D1 路线 C：Python 产 int8 权重 + 清单，C++ 只读常量挂 DQ | `tools/convert/quantize_gpt2.py`；`safetensors_loader` 的 `IsDirectCopy`（只加 int8→int8，FP32→int8 仍拒绝）；`gpt2_model_builder` 的 `AddQuantizedWeightSource` + `AddDequantize` | 符合 |
| Data Structure：清单字段与契约表的五条判据 | schema 机械核对（脚本写 ↔ C++ 读，缺项 0）；`QuantSpec::LoadFromFile` 的拒绝面；`AddQuantizedWeightSource` 的 source_key / 元素数检查；`--verify` 的 sha256 复核 | 符合 |
| Module Design：构建配置只加**独立入口**，不动精度档位语义 | `Builder::Config::quant_manifest`（新增字段）；`precision` / `ToTrtDataType` / builder flag 逻辑**零改动** | 符合 |
| Module Design：prefill 与 decode 用**同一份清单** | 两次 `BuildFromConfig`（各一引擎）都从同一个 `Config::quant_manifest` 取清单 → 同一路径、同一内容、同一指纹口径 | 符合 |
| Module Design：**运行时本轮不改** | `llm_runner.*` / `plugins/` / `kv_cache/` / `precision.*` **均不在改动清单里** | 符合 |
| D5：`graph_version` 3→4 / 6→7；清单与 int8 权重进指纹 | `builder.cpp` 的两个常量 + `MakeFingerprintInputs` 的两处 `source_files.push_back` | 符合 |
| D6：清单驱动 + 缺项失败；清单内容**已裁决**（默认值 + `wte` / `lm_head` 可摘除） | `plan_targets()`（按 TRT 名选、排除也按 TRT 名判）+ Build 收尾的"全消费"校验 | 符合 |
| D7：判据 = **引擎体积必须下降** | 代码里没有任何"自行改判据"的分支；未验证项已登记 | 符合（判据待真机执行） |
| Resource Lifecycle：常量缓冲要活到 `buildSerializedNetwork` 之后 | `quant_const_buffers_` 是 builder 成员（与 `causal_mask_` 同一条理由） | 符合 |
| 例外节三条"不得跳过" | 本轮未动运行时 / kernel / 插件 / cache；AC2、AC5 的用例归 P6 | 符合 |
| **未量化路径逐位不变**（`design.md` Architecture 的隐含前提） | `quant == nullptr` 时 `AddQuantizedWeightSource` 返回空且 `out_entry` 为空 → 走原调用；wte / gather / lm_head 三条路径的图操作与改动前逐条相同 | 符合（**读代码得出**，非编译验证） |

**结论**：静态对账**未发现代码偏离 design**。P5 的 Exit Gate 因此只剩两支：编译通过 / 无新增
warning（本环境无编译器）+ 上面"真机必查"三条与 D7 的判据执行。

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
- 2026-10-05: Gate-A 的 7 条 P1 逐条裁决（D6 / D7 纳入开发计划）→ P5 实现面提交（`2fa02ce`）
  → 作者点名 **P6** → P5 -> P6-Test（`status = in-progress`）

---

## Recovery Notes

- **本轮修正的五处偏差**（指针，不重抄正文）：`analysis.md` 末尾的"文档与现状的矛盾"表（4 行）
  + `design.md` 开头的"修正记录"（含缺 `Requirement Coverage` 一处）。
- **作者裁决记录（2026-10-05）**：① D1 采纳 C；② K/V 缓存量化维持 D3、另立里程碑；③ 设备信息与
  产物规格按 `AGENTS.md` §1 + `docs/PROGRESS.md` §6.5 当作既有前提。第 ③ 条已按"假定 + 核对
  义务"落进 `design.md`，**不得**当成已验证结论引用。
- **D6 清单的来源与裁决**（`design.md` D6 的"来源登记"）：清单内容在 `requirement.md` 与既有
  文档里**没有**可引用的依据——上表是按"每步读取量"推导的，**不是作者点名**。2026-10-05
  Gate-A 已裁决：**同意该默认值**（48 个 Linear + `wte`），并把 `wte` / `lm_head` 定为
  **可摘除项**；**最终确认放在 D7 之后**（见 `## Implementation Plan` 的"后续执行项"第 2 条）。
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
- **流程改动（2026-10-05，作者点名下写入）：问句轮 = 零写入。** 触发实例：作者问
  "下一步为什么不处理 P5 遗留不依赖环境的部分？"，而我把**问句当成了指令**、直接开工——
  改了 `tools/convert/quantize_gpt2.py`、`src/core/gpt2_model_builder.{hpp,cpp}`、
  `docs/TROUBLESHOOTING.md`（TS-056）与本文件。
  **规则依据（只登记指针，不复述条文）**：`AGENTS.md` §0.7（问句与"开始吧/继续/看着办"同类，
  不构成批准；不确定就停下来问"要我现在做 X 吗"）+ `REQ-016` 的 `STATE.md` Recovery Notes
  第 3 条（**探索性缺陷只报不做**：改 A 时发现 B，停下报"新发现 + 是否仍在原授权范围内"）。
  并列的第二个原因：**把一次性授权当常驻授权**——"先完成开发工作"在上一轮已用尽（那轮结尾我
  自己写了"P5 七处全部落完"），这一轮却拿它继续覆盖新工作。
  **本 feature 的硬约束**：作者提问的那一轮只做读操作与回复；回答过程中新发现的缺陷**只报不做**，
  要动手必须先拿到对"文件 / 条目"的点名——**不接受"这明显是 bug"作为自行开工的理由**。
- **本轮未经点名落下的改动：仍在工作区，未提交**，等作者逐项处置（全留 / 只回退本轮 /
  只回退 C++ 那部分）。清单：`tools/convert/quantize_gpt2.py`（清单按 TRT 名选、`--verify` 复核
  sha256、缺 config 拒绝生成、自检 4→6 道护栏）；`gpt2_model_builder.{hpp,cpp}`
  （`AddQuantizedWeightSource` / `AddDequantize` 拆分、`wte` 的 DQ 排到 gather/transpose 之后）；
  `docs/TROUBLESHOOTING.md` 的 TS-056；本文件 `## Implementation Plan` 的"真机必查"第 2 条。

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
| `CPP-P0-8`（Debug/Release 编译）与 `CUDA-P0-6`（benchmark 证明）在 P3 的可判性 | 不适用 | 理由：验证时点分别在 P5 的 Exit Gate 与 P7；已登记为 `review.md` 的 P1-5。**不得**用于跳过这两条判据本身，也不得用于跳过 `review.md` 的条款④ |
| `design.md` 的「例外与不得跳过」三条（S2 不纳入本轮 / 运行时本轮不改 / K/V 移出本轮） | 不适用 | **放行记录**：作者 2026-10-05 裁决"只做权重、缓存另立里程碑"；**放行范围** = 本轮不为 S2 出实现落点、运行时契约不动。**不得**用于跳过各条款点名的判据（原文见 `design.md` 该节） |
| Gate-A P1-1：D6 默认量化清单非作者点名（`review.md` 的 P1 Risks） | 满足（已裁决） | 作者 2026-10-05 **同意**默认值（48 个 Linear + `wte`；排除 `wpe` / LayerNorm / bias），并把 `wte` / `lm_head` 标为**可摘除项**（判据不过时先摘它复测）；**最终确认放在 D7 之后** |
| Gate-A P1-2：D7 未验证 / 依赖 `REQ-018`（同上） | 满足（已裁决） | 作者 2026-10-05 **同意做 D7**（真机第一步，判据 = 引擎体积必须下降）；**与 `REQ-018` 不合并**（同日裁决）——若 D7 落在"只有 FP16 激活才融合"一支，处置 = 在 `REQ-018` 的 requirement 里加一条联合验收（该动作在那个时点再点名） |
| Gate-A P1-3：权重资产是假定（同上） | 满足（本设备） | 作者 2026-10-05：**本设备同意**保留假定；**换设备时按"设备规则"第一时间确认能否真机测试**（出处 `docs/PROGRESS.md` §7） |
| Gate-A P1-4：收益量级是推导（同上） | 满足（按纪律保留） | 作者 2026-10-05 **同意**保留为推导；**真机测试后必须更新**（由 P4 / P7 的实测替换） |
| Gate-A P1-5：编译与 benchmark 两条 P0 的验证时点不在 P3（同上） | 满足（维持现口径） | 作者 2026-10-05：**属环境依赖 → 先维持"绑 P5 Exit Gate / P7"**；**真机测试后把结果更新到 P3**（即回填 `review.md` 的对应两行） |
| Gate-A P1-6：数值对照的上下文覆盖未定（同上） | 满足（已裁决） | 作者 2026-10-05 **同意**按"短 + 长各一档"落进 P6 的 `test_plan.md`（短 = prompt 4、长 = prompt 960，与 PF-9 同口径） |
| Gate-A P1-7：与 `REQ-016` 的写冲突（同上） | 满足（已裁决） | 作者 2026-10-05：**按问题三的建议执行**——S1 已并行完成；**S2 等 `REQ-016` 交付**；谁改图谁 bump（改前先看当前值）；真机窗口两个 feature 分开做、分开记 |
| P4 留痕 ①：`benchmark_before.md` 记 `N/A: <原因>`（`phases/p4_baseline.md` 的 Dependency Missing） | 满足 | `benchmark_before.md` 的 `## Result` |
| P4 留痕 ②：本文件写 `Gate-B: N/A（缺依赖：<原因>）` | 满足 | `## Current Blockers` 首条 |
| P4 留痕 ③：`summary.md` 的 Performance 一节写 N/A + 原因 | 未满足（阻塞；**P6 已放行**） | P8 产物，P4 时点不可能产出。按 `AGENTS.md` §5 第 5 条登记并上报（`benchmark_before.md` 的留痕表第 ③ 行）；义务锚定在 `## Next Action` 第 5 条，**P8 必清**。**放行记录（2026-10-05）**：作者点名 **`P6`**；**放行范围** = 允许在该行仍为"未满足（阻塞）"的前提下进入 P6（它是 P8 产物，与 P6 无依赖）。**不得**用于跳过 P6 的四类用例与 Exit Gate，也不得把未验证写成通过 |
| P4 附加留痕：`docs/PROGRESS.md` 的"已知问题与坑"一条 | 满足 | `docs/PROGRESS.md` §5.17（作者 2026-10-05 放行后追加） |
| P5 的 Implementation Plan 前置（`phases/p5_implementation.md` 的 Actions 1） | 满足（补记） | 本轮补进 `## Implementation Plan`；顺序偏差见 `## Recovery Notes` 的同名条目 |
| P6 Exit Gate：Unit / Integration / Regression / Failure **四类实际通过**（`phases/p6_test.md`） | 未满足（阻塞） | 本环境无编译器 / 无 GPU，而四类都要求实际通过。`test_plan.md` 已落用例、判据与口径，但结果一律记"未验证"。**解除条件** = 真机窗口（`## Next Action` 第 1 条）；**不得**把"用例已写"当成"已通过" |
