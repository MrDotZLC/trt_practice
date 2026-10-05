# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | bugfix |
| feature | REQ-018-gpt2-fp16-nan |
| phase | B2-MinimalFix |
| phase_index | 2 |
| status | waiting-human-gate |
| updated | 2026-10-06 |
| owner | Codex |

---

## Completed Artifacts

- `requirement.md`（B0-Reproduce：问题、复现资产、验收判据，2026-10-01）
- `analysis.md`（B1-Diagnose：现有证据链、已否证方向、待查方向，2026-10-01；
  **2026-10-05 起同时承载 B2 的候选修复方案与预算判定**——原独立 `design.md` 按作者指令
  并入该文件的 `## Candidate Fixes（B2）` 一节后删除；这是 Bugfix 规则的产物口径：
  `requirement.md` + `analysis.md` + `summary.md`（B4），候选方案不单独建文件；
  **2026-10-06：候选方案整节换成"重设计方案 v2"**，并补出 `## Terminology` 与"候选方案节例外注记"）
- `review.md`（P3 Review：52 + 8 + 6 三张表、P0 / P1 / P2、Decision = BLOCK，2026-10-06）

---

## Current Blockers

- **设备条件缺失（阻塞，2026-10-06 作者裁决）**：当前设备**无编译与真机条件** → 诊断（B1'）、
  修复验证（B2）、回归（B3）、以及评审 P0-3（读 `NvInfer.h` 核实 Softmax 是否有精度接口）
  全部**搁置**。本条现在只到"方案与判据就绪、待真机窗口"。
- **评审 P0-3 未关闭**：`addSoftMax`（`gpt2_model_builder.cpp:948`）今天没有精度设置，
  `ISoftMaxLayer` 在 TRT 10 是否有等价接口**本机查不到**（无 `NvInfer.h`）→ 判据来源缺失，
  Gate-A 不通过。它随真机一并搁置，不得用"推测 TRT 会做 FP32"来代替核实。
  **离线取头文件的四条路都试过（2026-10-06，逐条留痕）**：① 沙箱 `/usr/include/**` 无 `NvInfer*.h`；
  ② Windows 侧没有 CUDA/TensorRT 安装（`C:\Program Files\NVIDIA GPU Computing Toolkit` 不存在、
  `System32` 无 `nvinfer*.dll`，只有驱动 `nvidia-smi.exe`）；③ 本环境**没有可用的 WSL 发行版**
  （`wsl.exe true` 退出码 1、只回帮助文本）→ 真机那套 TRT 头文件在这里够不着；
  ④ pip 侧无 tensorrt 包、无本地 wheel 缓存。**联网取 wheel 这条 2026-10-06 也试过了**：
  PyPI 的 10.15.1.29 只有元包 sdist（16 KB，`tensorrt` 依赖 `tensorrt_cu13`），
  `tensorrt_cu13` 全线同样只有元包 sdist，`tensorrt_cu13_bindings` 的 wheel 共 20 个文件、
  全是 `.py`/`.pyd`，**没有任何发行包带 `NvInfer.h`**。→ 只剩真机读头文件这一条路
  （若接受更低可信度，另一条是 GitHub 镜像源，需另行批准）。
- **未定位到具体算子**：5 轮真机往返只到"性质"（构建相关的不稳定、幅值远未溢出），
  没到"哪一层、哪个算子"；定位手段（三图对照）已写进设计，等真机窗口执行。
- 真机窗口内要跑的用例与引擎构建，按 `AGENTS.md` §0.3 仍需先获批准（本次已整体搁置）。

> 已解除的两条（保留在此只为交接时不被重新当成 blocker）：
> **策略冲突**已由作者 2026-10-06 裁决**撤销"按政策不修"**（来源与口径见 `requirement.md` 的 Constraints）；
> **B2 预算可能超限**已由重设计方案解掉——修复面收窄为"定点一处显式精度"（≤1 文件 / ≤30 行，在预算内），
> 旧的"逐算子二分"降级为兜底、"激活缩放"移出本轮。

---

## Next Action

- **当前没有可动手的事**（阻塞在设备）。三件待裁决事项已全部落地：策略（已撤销）、真机
  （已搁置）、预算（已在范围内）。
- **真机窗口出现时**，按 `analysis.md` 的 `## Candidate Fixes（B2）` 的 `### Runtime Flow` 执行：
  图 A / 图 B / 图 C 一次建完、六条读数、判定 `L_A` 与 `L_B`；顺带读 `NvInfer.h` 关掉 P0-3。
  **执行顺序（点名到用例）**：
  1. `Gpt2GenerateTest.Fp16PlainPrefillLayersDiagnostic`（**图 A**：生产图、无探针）→ 记 `L_A` 与生产图的 tactic 指纹；不要跳过它——图 B 挂了 85 个输出，只有图 A 能说"仪器没扰动被测对象"；
  2. `Gpt2GenerateTest.Fp16PrefillOutputsDiagnostic`（**图 B**：全探针）→ 记 `L_B`、层内首个非有限张量、逐层 RMS / 幅值；
  3. `Gpt2GenerateTest.Fp32PrefillOutputsDiagnostic`（**图 C**：对照臂）→ **必须全有限**，并给出健康幅值剖面；
  4. 读 `NvInfer.h` 关 P0-3；`RealGpt2Fp16GreedyMatchesReferenceTokens` 作复现器（此时应仍红）。
  判据：`L_A == L_B` 才能直接用图 B 的层内读数；不等则走 `### Runtime Flow` 的兜底分支
  （上限 = 1 次全导 + 3 次切层二分）。
- **用例条目数的影响**：本次新增 2 条用例（图 A / 图 C）。按 `gtest_discover_tests`
  （`mini_trt_llm/tests/CMakeLists.txt:22`）"一条 case = 一条 ctest 条目"的口径，**沙箱条目数预计 +2**
  → 从 §0 的 269 变 **271**，仍标"待复跑"。`PROGRESS.md` §0 的那个数因此需要同步——
  PROGRESS 属结构类文档，**等作者点名后再改**（本条只记录影响，不代改）。
- **H4 的静态对账已做**（2026-10-06，零真机成本）→ `analysis.md` 的 `#### D1a`：
  产物路径**排除**；留下一条窄假设 = 逐层 K/V 声明精度是否全层一致（已在诊断用例里逐层打印，等真机）。
- 真机窗口内除三图对照外，还要顺带跑：逐层 dtype 打印（D1a 的窄假设）、`NvInfer.h` 与
  `getTensorShape` 的动态维语义确认。

---

## Implementation Plan

<!--
Bugfix 没有 P5，这里按 `AGENTS.md` §5 的"改代码前置"落 B1' 的仪器改动。
本次**只改仪器**（诊断探针 + 诊断用例），不改任何产品路径的行为。
-->

- **当前模块**：GPT-2 建图器的诊断探针 + 诊断用例的绑定与读数；引擎指纹的开关项。
- **实际文件**（3 个，均在 Bugfix 预算内；**已落码，未编译验证**）：
  1. `mini_trt_llm/src/core/gpt2_model_builder.cpp`：探针由"仅第 0 层、4 个切点"改为
     "**全部层 × 7 切点**（`attn_ctx` / `attn_out` / `attn_res` / `ln2_out` / `mlp_fc` /
     `mlp_gelu` / `mlp_res`）+ `block_in_0`"，仍只在 `export_diagnostics` 打开时挂、且只对 prefill 生效。
  2. `mini_trt_llm/src/core/builder.cpp`：加 `kDiagnosticProbeVersion`，**只在开诊断时**并入指纹
     —— 否则改了探针集会安静复用旧引擎（TS-040 同类）。
  3. `mini_trt_llm/tests/test_gpt2_generate.cpp`：诊断用例的输入按**声明 dtype** 填（`padding_bias` 全 0、
     形状 `[1,1,1,S]`）、输出遇**未识别名字即失败**（不再落 `hidden` 兜底 = 越界写）、
     逐张量打印 `max|v|` / RMS / 首个非有限下标、逐层 ONELINE 落盘、逐层 K/V 声明精度打印。
  4. `mini_trt_llm/src/core/llm_runner.cpp`（2026-10-06 加，属**护栏**不属修复）：启动期检查
     **逐层 K/V 与逐层 cache 的声明精度必须一致**——把 D1a 的"假定全层相同"变成"不一致就拒绝启动"。
     全层一致时是 no-op；命名写错（`getTensorDataType` 对不存在的名字按 TRT 约定返回 kFLOAT）
     也会在这里响亮失败。
  5. `mini_trt_llm/tests/engine_layer_info_support.hpp`（**新增**，2026-10-06，属 P2 收口不属修复）：
     把"遍历逐层 ONELINE、统计 Int8 张量与 tactic 名、可选落盘"收到一处，并改造两个既有使用方
     `test_resnet18_int8_probe.cpp` / `test_gpt2_int8_weights.cpp`（原各有一份实现）。
- **测试方式**：本机**无编译器 / 无 GPU**，只能做静态自检（符号与参数逐个核对 + 名字/尺寸分类表逐条走查），
  编译与运行**全部记"未编译验证"**，等真机窗口一次性执行。
  **归类口径（作者 2026-10-06 裁决）**：护栏与仪器按"非修复"归类，**不计入 B2 的修复预算**
  → 真修复面仍是 0 文件（等真机）；因此本条目累计 4 个代码文件不触发 Bugfix → Feature Decision。
  放行范围 = 本条的仪器与护栏改动，不含其它。

---

## Phase History

- 2026-10-01: B0 -> B1（现有复现器与诊断仪器已存在，补文档）
- 2026-10-01: B1 -> B2（根因未定位，B2 预算待判定）
- 2026-10-01: B2 -> Human Gate（`status = waiting-human-gate`）
- 2026-10-06: 作者裁决撤销"按政策不修"（来源口径 = 正确性，非性能）；同轮出"重设计方案 v1"
- 2026-10-06: B2 内部评审（P3 Review 口径）→ Decision = BLOCK（4 条 P0；P0-1 / 2 / 4 同轮关闭，P0-3 开放）
- 2026-10-06: 作者裁决当前设备无真机条件 → 真机相关部分整体搁置；`status` 维持 `waiting-human-gate`
- 2026-10-06: H4 的静态对账完成（产物路径排除，留一条窄假设）→ `analysis.md` 的 `#### D1a`
- 2026-10-06: B1' 的仪器改动落码（建图器探针 / 指纹开关项 / 诊断用例自证，共 3 文件），**未编译验证**
- 2026-10-06: P0-3 的离线取头文件路径**全部试过并排除**（含联网取 wheel：cu13 线只有元包 sdist、无任何包带 `NvInfer.h`）→ 只剩真机
- 2026-10-06: D1a 的窄假设落成**启动检查**（`llm_runner.cpp`，属护栏；未编译验证）
- 2026-10-06: 补齐设计里的**图 A / 图 C 两臂**（原 B1' 漏项）：抽公共辅助函数 + 新增 2 条用例（未编译验证）
- 2026-10-06: 评审 P2 收口（C）：逐层信息读取提到共享头 `engine_layer_info_support.hpp`，三处实现收拢为一处（未编译验证）

---

## Recovery Notes

- **路由**：本条走 Bugfix Workflow（技能把"结果错误"归 Bugfix；现象 = 贪心恒为 0、logits 全 NaN）。
- 历史证据在 `docs/TROUBLESHOOTING.md` + TS-018.1（5 轮真机定位表），结论索引在 `PROGRESS.md` §5.11；
  本文档不复制那些数据，只写"下一步查什么"。
- **来源登记（为什么它是 REQ-018）**：它不是新立的需求，而是**归档编号**——2026-09-26 该问题被
  按政策登记为"已知限制"并保留红色复现器、另立三条路线（`future_iterations.md` §1.4）；
  2026-10-01 采用本技能为流程权威时（提交 `ddec282`）把 4 个开放中的历史遗留项回落成 `docs/dev/`
  条目，同年同日的归档提交 `2a61b22` 按顺序给它编号 018。编号只承载身份、不承载状态
  （`docs/dev/INDEX.md` §0）。**不要把它当成"某天新提的需求"来追溯。**
- **策略口径（易被误读）**：2026-10-06 的撤销依据是**正确性**（默认构建精度不可用 = 产品缺陷），
  不是 `future_iterations.md` §1.4 写的"低精度推理**性能**"触发条件；旧口径的收益结论
  （无 Tensor Core、收益有限）**仍然成立**，只是不再作为策略前提。
- **候选方案节的例外注记**：`analysis.md` 的 `## Candidate Fixes（B2）` 是 `phases/p1_analysis.md`
  "analysis.md 禁止出现修改计划"的**唯一例外**，依据 = `workflows/bugfix.md` 的 Required Artifacts
  （作者 2026-10-06 裁决：以 bugfix.md 为准）。该注记已写在 `analysis.md` 顶部。
- 复现器：`mini_trt_llm/tests/test_gpt2_generate.cpp` 的
  `Gpt2GenerateTest.RealGpt2Fp16GreedyMatchesReferenceTokens`（当前**按设计红**）与
  `Fp16PrefillOutputsDiagnostic`（诊断打印，按设计通过）。
- **改这条会动测试基线**：它目前是全量里唯一的红，修好之后红数归零，
  需要同步回填 `PROGRESS.md` 的当前基线 / §5.11 / `interview_summary.md`。

---

## 判据对照

说明：本节只在**异常路径**填写（Mandatory #10）。本次阶段切换（B2 内部：方案重设计 + 评审）里
出现"不适用 / 未满足 / 放行"的判据逐项留行；"行数 = 该阶段判据数"取**本阶段实际触发异常的那些判据**，
条文出处只引用、不复述（SKILL.md 的 Mandatory #8）。

| 判据（出处） | 状态 | 证据 / 放行 |
|---|---|---|
| P3 Entry：先复核上一阶段的「判据对照」 | 满足 | 本节（此前缺失，2026-10-06 补出）；补出前该 Entry 未正式满足，由作者 2026-10-06 点名直接评审 |
| P3 Entry：`requirement.md` 的模糊名词须在 `analysis.md` 的 Terminology 有可验证定义 | 满足 | `analysis.md` + `## Terminology`（本次补出，6 条） |
| p3_review 规则：条目要求的输入 / 接口在系统里查不到即 P0 | **未满足（阻塞）** | `review.md` 的 P0-3（Softmax 精度接口，本机无 `NvInfer.h`）；解除条件 = 真机窗口读头文件或建最小图 |
| 技能 B2：改动预算（文件 ≤3 / 行数 ≤100），超限触发 Bugfix → Feature Decision | 满足（**作者 2026-10-06 放行**） | 冲突双方与建议见本行的历史记录（① 设计 v2：B1' 不占修复预算；② 累计 4 个代码文件，按"全部改动"计会超 ≤3）。**作者裁决 = 采纳建议**：护栏/仪器按"非修复"归类，真修复面 = 0 文件 → 不触发 Feature Decision。**放行范围** = 本条的仪器（B1' 3 文件）与护栏（`llm_runner.cpp`），不含其它改动 |
| `AGENTS.md` §0.7 / §0.3：真机任务需作者点名批准 | **未满足（阻塞）** | 作者 2026-10-06 裁决**当前设备无真机条件 → 真机部分搁置**；放行范围 = 无（不批准任何真机动作） |
| 技能 Bugfix Required Artifacts：`requirement` / `analysis` / `summary`(B4) | 满足 | 前两件在；`summary.md` 属 B4，**未到阶段**，不计缺 |
| 规则冲突：`workflows/bugfix.md`（候选方案写在 analysis.md）与 `phases/p1_analysis.md`（analysis.md 禁止修改计划） | 满足 | 作者 2026-10-06 裁决：以 bugfix.md 为准 + 在 analysis.md 顶部加例外注记（§5.5 的"冲突双方 + 建议"已上报） |
