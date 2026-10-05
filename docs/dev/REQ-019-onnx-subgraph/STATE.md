# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-019-onnx-subgraph |
| phase | P2-Design |
| phase_index | 2 |
| status | waiting-human-gate |
| updated | 2026-10-06 |
| owner | Codex |

---

## Completed Artifacts

- `requirement.md`（P0-Requirement，2026-10-01）
- `analysis.md`（P1-Analysis，2026-10-01；2026-10-06 补 `## Terminology` 与两条现状事实）
- `design.md`（P2-Design，2026-10-06 **重设计：乙框架**；上一版 2026-10-01 已被本次覆盖）
- `s0_contract_interface_spec.md`（P2 接口细化，2026-10-06）
- `s1_topology_interface_spec.md`（P2 接口细化，2026-10-06）
- `s2_custom_op_interface_spec.md`（P2 接口细化，2026-10-06）
- `review.md`（P3-Review，2026-10-06；**结论 BLOCK**——2 条 P0 都在 S2，S0 / S1 设计不受影响）

> 三份 spec 是 S0 / S1 / S2 的**接口与改动点唯一来源**（技能 Mandatory #7 的指针式登记）；
> `design.md` 只保留里程碑与决策，不复述它们的细节。**S3 的 spec 待 PF-7 判定后再出**。

## 判据对照

> 触发条件（技能全局规则 10）：本条存在**依赖缺失 + 降级放行**。下一阶段（P4）的 Entry 必须先复核本表。

| 判据 | 状态 | 证据 / 谁放行、放行范围 |
| --- | --- | --- |
| Included 第 1 条（PF-7 前置测量） | 未满足（阻塞） | 需 sm_75 目标真机；当前设备为 GTX 960（2026-10-06 核对）。作者 2026-10-06 接受"S3 长期标待定" |
| Included 第 3 条 + AC 1 / 3 / 4（替换落地） | 未满足（阻塞） | 同上；重开条件见 `design.md` 的「待定项（阻塞）」 |
| PF-7 的用例前提（薄用例） | **已授权（2026-10-06）** | `OnnxVsNative.PerfPerBuildMedian` 原先不存在；作者 2026-10-06 授权新写（属新代码）。落码后可编译，但**出数仍需真机** |
| Included 第 2 / 4 / 6 条（S1 / S2 / S0） | 不受阻塞 | `design.md` 的 Milestones 表；D0 把性能门收窄为只挡 S3 |
| P3 评审 P0-1：`createParser` 带 `IPluginRegistry` 的重载是否存在 | 未满足（阻塞） | 本机无 TRT 头、全仓检索无既有证据。**作者 2026-10-06 裁定"暂时不真机" → 核实推迟**；下次真机窗口读 `NvOnnxParser.h`（不成立则回 P2 选路） |
| P3 评审 P0-2：creator 命名空间为空串 ↔ ONNX 自定义域必须非空 | **已裁决（放行）** | 作者 2026-10-06 "接受"：namespace 设为 `mini_trt_llm` + 接受连带重建（重建范围待 §3 第 3 条核实后写死）。落码在 S2 / P5 |
| P3 评审 P1-5：S0 的识别基线归属 | **已裁决（放行）** | 作者 2026-10-06 选 **B**：`--check` 只对源图生效，重写图由重写器自检 + S0 的 host 用例覆盖 |
| **S1 开工放行**（技能全局规则 10 的放行记录） | **已放行（2026-10-06）** | 作者点名"放行 S1；批准装 onnx；授权写 PF-7 薄用例；夹具用 RmsNorm"。**放行范围 = S1 的落码 + 夹具级自检**（不改运行时入口、不碰 S0 / S2 / S3）。`onnx` 已装（1.23.1）。其余阻塞项（T1 / T5 / T6 / T7 / T8 / T9 / T12）不受此放行影响 |

## Current Blockers

- **Gate-A 未过（唯一障碍 = P0-1）**：`review.md` 已产出（52 / 12 / 12 三张表），结论 BLOCK；
  剩下的 P0 是"`createParser` 重载是否存在"，受作者 **"暂时不真机"** 的约束。
- **S3 阻塞**：PF-7 需目标真机；`docs/future_iterations.md` §6.2（自定义算子导出）是 S2 的实现前置。
- **PF-7 薄用例已授权、待真机**：代码可写，但它按协议要"同 session ≥3 次构建 × ≥20 次推理"才有数，
  所以真正的读数仍要等真机窗口。
- **本阶段不真机**（作者 2026-10-06）：所有需要 GPU / TRT 的核实与验证推迟。受影响的"待核实"项：
  ① S0：真实图的位置编码形态（A 常量 / B `Range`）与 `input_ids` 的 INT64 消费点——需 652MB 资产；
  ② S1：真实图里注意力块的实际边界形态——同上；
  ③ S2：`createParser` 重载是否存在、属性类型映射——需 TRT 头（即 **P0-1**）。
  例外：S1 的**夹具级**三个用例（F1 / F2 / F3）只依赖 `onnx` 包、不依赖资产与真机，
  是本阶段唯一能在沙箱推进的验证项——作者 2026-10-06 已放行 S1 并批准装 `onnx`（已装 1.23.1），
  当前正在落码。

## 待定与待决清单

> **口径**：本表是本条**所有**未决事项的唯一登记处（技能 Mandatory #7：细节的唯一来源在各自文件，
> 本表只做"指针 + 状态 + 解除条件"）。任一项解除后，**同时回填它的出处文件**，本表同步删行。
> "是否挡 Gate-A"只对评审关口有意义；其余项挡的是落码或后续里程碑。

| # | 事项 | 类型 | 出处（细节的唯一来源） | 解除条件 / 谁决定 | 挡 Gate-A？ |
|---|---|---|---|---|---|
| T1 | `createParser` 带 `IPluginRegistry` 的重载是否存在 | 待核实（**阻塞**） | `review.md` 的 P0-1；`s2_custom_op_interface_spec.md` §3 第 1 条 | 真机读一次 `NvOnnxParser.h`；不成立则回 P2 选路（`IPluginFactory` 路线未设计） | **是** |
| T3 | S3 的接口细化（`s3_*_interface_spec.md`）尚未产出 | 按约定不写 | `design.md` 的 Milestones 节 | PF-7 判"可判"后再出；走向由作者裁决 | 否 |
| T4 | PF-7 的薄用例 | **已授权 + 已落码（未编译）**；待真机出数 | `design.md` 的 Performance Consideration；`docs/future_iterations_test_plan.md` 的 PF-7 行 | 授权已给、用例已写（`OnnxVsNative.PerfPerBuildMedian`）；剩"编译 + 真机窗口" | 否（挡 S3） |
| T5 | S3 子图替换本身 | 未满足（阻塞，作者已接受长期） | `design.md` 的「待定项（阻塞）」 | 真机 + PF-7 判"可判" | 否 |
| T6 | S0 的真实图形态（位置编码 A / B）与 `input_ids` 的 INT64 消费点清单 | 待核实 | `s0_contract_interface_spec.md` §2 / §7 第 2、3 条 | 真机跑 `inspect_onnx.py` 探针（需 652MB 资产） | 否（落码前置） |
| T7 | S1 的真实注意力块边界形态 | 待核实（**部分降级**） | `s1_topology_interface_spec.md` §1 / §7 第 2 条 / §7.2 | 夹具级（F1/F2/F3）已在沙箱验过；**真实图的串联形态仍要资产**（`inspect_onnx.py <资产图> --check-topology`） | 否 |
| T8 | 属性 → `PluginField` 的映射细节 | 待核实 | `s2_custom_op_interface_spec.md` §3 第 4 条 | 真机（读头文件 + 最小复现） | 否 |
| T9 | P0-2 的"连带重建范围" | 待写死 | `s2_custom_op_interface_spec.md` §3.1；`review.md` 的 P0-2 | 先核实既有引擎是否受影响（§3 第 3 条），再由作者定范围 | 否 |
| T12 | `BuildFromOnnx` 签名变更与运行时入口的写冲突串行顺序 | 待排期 | `design.md` 的「耦合与写冲突」 | S3 落码前由作者指定串行顺序 | 否 |

> **T10 已于 2026-10-06 解除**（作者点名同步）：`docs/future_iterations_development_plan.md` 第 1278 行
> 的"无新代码"已更正为"P4 需补一个薄用例"。按本表口径删行；**编号不重排**，以保持既有引用稳定。
>
> **同日再解除两项**：**T2**（夹具算子裁决 = `MiniTrtLlmRmsNorm`，已回填 `s2_custom_op_interface_spec.md`）、
> **T11**（`onnx` 已装 1.23.1）。**T4 转为"已授权、待真机"**（不再需要授权动作）。

## 真机窗口执行清单

> 目标：**一次窗口把"不建引擎就能拿到的事实"全部取回**，需要建引擎 / 需要授权的动作留到最后一档。
> 本清单只排**顺序、命令与回填位置**；判据细节在各自文件，不在此转述。

| 次序 | 动作 | 命令 / 看什么 | 判据 | 回填到哪 |
|---|---|---|---|---|
| 前置 | 确认当前设备是**目标真机** | `nvidia-smi`；对照项目级硬件基线（sm_75 / GTX 1660 Ti） | 不是目标真机 → 按 `requirement.md` 的「设备纪律」当轮上报，本窗口作废 | 当轮回复 + 本文件 |
| 1 | 读 `NvOnnxParser.h`（**T1 / P0-1**，唯一挡 Gate-A 的） | 找 `createParser` 的重载声明 | 存在带 `IPluginRegistry&` 的重载 → **抄签名原文**；不存在 → **回 P2 选路**（现场不改接口） | `s2_custom_op_interface_spec.md` §3 第 1 条；`review.md` 的 P0-1；本文件 `## 判据对照` |
| 2 | S0 的图形态探针（**T6**） | `python mini_trt_llm/tools/inspect_onnx.py <资产图> --json` | 位置编码属 **A**（常量 initializer 驱动 `Gather`）还是 **B**（图内 `Range`）；列出 `input_ids` 的 INT64 消费点 | `s0_contract_interface_spec.md` §2 / §7 第 2、3 条 |
| 3 | S1 的真实块边界（**T7**） | 复用上一条的输出（节点与连接） | 12 个注意力块能否按 T1–T4 切出边界；与 §1 的骨架是否一致 | `s1_topology_interface_spec.md` §1 / §7 第 2 条（形态不符时**先改 spec 再改代码**） |
| 4 | 既有引擎反序列化探针（**T9 的前提**） | 编译并运行 `mini_trt_llm/tools/inspect_engine.cpp`，对一个既有 `.engine` | 能反序列化 → 改 namespace 的影响留到 S2 落码后再判；不能 → 读出失败原因 | `s2_custom_op_interface_spec.md` §3 第 3 条 |
| 5 | S2 的属性 → `PluginField` 映射（**T8**） | 需要 S2 代码先落 → 本窗口只能做"读头文件记结论"那半 | 记录 `eps`(float32) / `hidden_size`(int32) 的映射规则与不符时的失败形态 | `s2_custom_op_interface_spec.md` §3 第 4 条 |
| 6 | PF-7 的薄用例（**T4**，已授权，代码已落） | 新用例（按 §10.2 口径：同 session ≥3 次构建 × ≥20 次推理） | 极差 < 中位数差 = **可判**；否则 = **未定** | `design.md` 的 Performance Consideration；P6 时进 `test_plan.md` |

**顺序理由与窗口纪律**：

1. 第 1 项最便宜、且是唯一挡 Gate-A 的，所以排第一；第 2/3 项只读图不建引擎；第 4/5 项需要编译或代码；
   第 6 项最贵（6 次全新构建）——按**代价递增**排。
2. **本窗口不重建引擎、不改图版本、不动代码**；产出的是**事实与原文**（签名、节点形态、反序列化结果），
   不是结论。
3. 跨 session 的数字不可比 → 第 6 项若要做，必须在**同一 session 内跑完**。
4. 任一项的结论落地时，**同时回填它的出处文件**（上表末列），并删掉 `## 待定与待决清单` 的对应行。

## Next Action

> **设计文档已备齐**：P0 `requirement` + P1 `analysis` + P2 `design` + 三份里程碑接口细化（S0 / S1 / S2）。
> **P3 评审已跑，结论 BLOCK**（`review.md`）：52 行 checklist 结论、12 行需求落点、12 行术语定义。

> **2026-10-06 三条裁决已落档**：① 暂时不真机；② 接受（P0-2 的修法 + 连带重建）；③ 基线归属 = B。
> 因此 **P0-2 与 P1-5 已闭合**，当前唯一障碍是 **P0-1**（下次真机窗口读一次头文件即可）。
>
> **2026-10-06 第二批放行**：放行 S1、批准装 `onnx`（已装 1.23.1）、授权写 PF-7 薄用例、夹具算子定为
> `MiniTrtLlmRmsNorm`。因此 **S1 已进入落码**（沙箱可自检），PF-7 用例可写但出数仍待真机。
>
> **同日落码结果**：S1 三件（探针开关 / 夹具脚本 / ctest 自检）**已完成并在沙箱跑绿**；
> PF-7 用例**已落码、未编译**。**`mini_trt_llm/` 自此不再是零文档改动**——
> 本次改了 `tools/inspect_onnx.py`、新增 `tools/make_topology_fixture.py`、`tests/CMakeLists.txt`、
> `tests/test_gpt2_onnx.cpp`。

1. **（已完成，沙箱自检通过）S1 落码**：`inspect_onnx.py --check-topology`（T1–T4）+ 夹具
   （F1 / F2 / F3）+ `onnx_topology_selftest`（ctest 条目 +1）。自检结果见
   `s1_topology_interface_spec.md` §7.2。**真实图的串联形态仍待资产（T7）**——那是 S1 唯一没闭的那半条。
2. **（已落码，未编译）PF-7 薄用例**：`OnnxVsNative.PerfPerBuildMedian`（每边 3 次构建 × 20 次推理，
   纯打印不设阈值）。写完**只能编译、不能出数**——真机窗口才见数。
3. **（真机窗口，恢复后第一件）清 T1 / P0-1**：读 `NvOnnxParser.h` 确认 `createParser` 是否接受
   `IPluginRegistry`；结论回填 `s2_custom_op_interface_spec.md` §3，再重跑 `review.md` 受影响的行。
4. **（真机窗口）清两处落码前置**：S0 的图形态探针、S1 的真实块边界（各 spec §7）。
5. 以上清完后按 `design.md` 的 Milestones 推进 **S0 → S2**（S1 落码中，S3 仍待 PF-7）。
  每个里程碑落码前先清各自 spec 末节的"待核实"项，**来源缺失时停在原地上报**。

## Phase History

- 2026-10-01: P0 -> P1
- 2026-10-01: P1 -> P2
- 2026-10-01: P2 -> Gate-A（`status = waiting-human-gate`）
- 2026-10-06: 作者裁定**乙框架**（性能门只挡替换）→ 回 P2 重设计 → P3-Review（`status = in-progress`）
- 2026-10-06: P3 复评 → **BLOCK（2 条 P0，均在 S2）** → 按技能的 P3 Failure Route 返回 P2-Design
  （`status = waiting-human-gate`）；S0 / S1 的设计**不需返工**，只欠落码前置探针
- 2026-10-06: 作者三条裁决落档（`review.md` 的「作者裁决」）：① **暂时不真机**（P0-1 推迟）；
  ② **接受** P0-2 的修法与连带重建；③ S0 基线归属 **= B**。→ P0-2 / P1-5 闭合，P0-1 保持未闭环
- 2026-10-06: 作者第二批放行（**S1 开工** / 装 `onnx` / 授权 PF-7 薄用例 / 夹具算子 = `MiniTrtLlmRmsNorm`）
  → 按技能规则 10 的放行记录写进 `## 判据对照`；S1 进入落码，脚本与夹具**未编译依赖、沙箱可自检**
- 2026-10-06: S1 落码完成（`--check-topology` + 三夹具 + `onnx_topology_selftest`）→ **沙箱自检通过**
  （详见 `s1_topology_interface_spec.md` §7.2）；PF-7 用例 `OnnxVsNative.PerfPerBuildMedian` **已落码未编译**

## Recovery Notes

- 本条与 `REQ-017-llm-int8-quant` 的 ONNX 路线耦合：若两者都做，外部图路径的 decode 图应在这里一并补齐，
  否则 INT8 走 ONNX 路线时要再改一次同一段代码。
- 与 `REQ-016-continuous-batching` 的写冲突点在运行时入口（`BuildFromOnnx` 的签名变更会波及同一批调用方）。
- 前置事实来源：`docs/future_iterations.md` §10.1 / §10.2 / §6.2、§0.1 的 PF-7 行；
  `docs/TROUBLESHOOTING.md` + TS-017（两条路的 I/O 契约差异）。
- 2026-10-06 的动作边界：PF-7 薄用例**在第二批放行后开始落码**（此前未授权，故此前未动）。
- 2026-10-06 更正 PF-7 的"无新代码"表述，共两处：`docs/future_iterations_test_plan.md`（PF-7 行）
  与 `docs/future_iterations_development_plan.md`（第 1278 行，作者当日点名同步）。两处都改为
  "待真机 + 待补用例"并写明原因（该用例在代码里不存在，现有测量达不到协议口径）。
- 2026-10-06 作者点名同步 **8 处**下游口径（`PROGRESS.md` §6.6 两行；`future_iterations.md` 的 §0.1 §6.2 行 /
  G5 行 / **§0.1 的 §10.1 行** / **§0.3 执行序列的 §10.1 行** / §10.1 正文；`future_iterations_development_plan.md`
  第 1278 行）：把"PF-7 决定整条 REQ-019"收窄为"只决定 S3"；把 §6.2 / G5 改成"已由 S2 / S1 立项"；
  把 §10.1 的"先加 Cast 即可"更正为"`input_ids` 改 INT32 且提升 `position_ids`"。
