# Review Report

<!--
审查对象：2026-10-06 的 REQ-018「重设计方案 v1」（评审时尚未落盘）。
阶段：bugfix `B2-MinimalFix`；口径按技能的 `phases/p3_review.md` 执行。
三张表的行数口径（Exit Gate 要求行数 = 对应集合条目数）：
  表一 = 四份 checklists 的 [P0] / [P1] 项 = 35 + 17 = 52 行；
  表二 = `requirement.md` 的 Included（4）+ Acceptance Criteria（4）= 8 行；
  表三 = `requirement.md` 中需要可验证定义才能判定"做到没做到"的模糊名词 = 6 行。
条目文字只写到"可定位到该条目"的粒度，条文原文以四份 checklists 与 `requirement.md` 为准（不复述）。
-->

## Summary

重设计方案 v1 的**方向**成立：一次真机窗口内用三张图（生产 / 全探针 / FP32 对照）做对照实验，
把"观测是否扰动了被测对象"变成可判定的读数；修复面从"逐算子二分"收窄到"定点一处显式精度"，
预算落在 Bugfix 内（≤1 文件 / ≤30 行）。

但 v1 作为设计有 4 条 P0：术语表缺失、图 B 不复现时无失败分支、出口 A 的 Softmax 分支无可兑现接口、
`padding_bias` 的动态维绑定未写。**Decision = BLOCK**，按 Gate-A 暂停。

处置现状（2026-10-06 同轮修订）：P0-1 / P0-2 / P0-4 已在设计 v2 里关闭；**P0-3 仍开放**——
本机没有 `NvInfer.h`，"Softmax 能否设计算精度"查不到来源，只能等有头文件的机器核实。
另：作者同日裁决**当前设备无真机条件 → 真机部分搁置**，所以 v2 的 B1' / B2 / B3 全部转为
"待真机窗口"执行，本次不产出任何实测读数。

## Checklist Result

**checklists/cpp.md**

| Item | Level | Result | Action |
|---|---|---|---|
| RAII 管理资源生命周期 | P0 | 通过 | 记录：探针缓冲与引擎沿用 `DeviceBuffer` / `Engine` 的 RAII |
| Ownership 明确 | P0 | 通过 | 记录：诊断引擎与其缓冲归诊断用例所有 |
| 悬空引用风险 | P0 | 待确认 | 人工确认：探针层集合若做成入参需按值传，不得持有外部容器引用 |
| 异常路径释放资源 | P1 | 通过 | 记录：沿用现有守卫 |
| 数据竞争 | P0 | 不适用（单线程诊断路径，设计不含并发） | 记录 |
| Mutex / Lock 生命周期 | P0 | 不适用（设计不含锁） | 记录 |
| 不必要锁竞争 | P1 | 不适用（同上） | 记录 |
| API 输入输出是否明确 | P0 | 不通过 | 修复：写明 `export_diagnostics` 语义由"仅第 0 层"改为"全部层"及其对指纹、输出计数断言的影响（v2 已补） |
| 错误处理方式是否统一 | P0 | 通过 | 记录：沿用 `MINI_TRT_LOG_ERROR` / `ASSERT` / `CUDA_CHECK` |
| 接口是否容易扩展 | P1 | 待确认 | 人工确认：探针层集合是否需要配置入口 |
| Debug / Release 是否均可编译 | P0 | 待确认 | 人工确认：本机无编译器；按设备纪律不在本机验收 |
| 是否引入额外依赖 | P1 | 通过 | 记录：不新增第三方依赖 |

**checklists/cuda.md**

| Item | Level | Result | Action |
|---|---|---|---|
| Kernel 边界条件是否正确 | P0 | 不适用（本轮不改 kernel） | 记录 |
| 是否存在越界访问 | P0 | 通过 | 记录：修复落点 = 绑定尺寸从引擎读，取代按名字猜（v2 已补） |
| Synchronization 是否正确 | P0 | 不适用（不改 kernel） | 记录 |
| 不同 shape 是否覆盖测试 | P1 | 不适用（固定单 prompt 形状，目标是定位而非形状覆盖） | 记录 |
| Global Memory 访问是否安全 | P0 | 不适用（不改 kernel） | 记录 |
| 是否存在 race condition | P0 | 不适用（不改 kernel） | 记录 |
| Memory coalescing 是否优化 | P1 | 不适用（不改 kernel） | 记录 |
| Shared Memory 是否存在 bank conflict | P1 | 不适用（不改 kernel） | 记录 |
| 优化是否有 benchmark 证明 | P0 | 不适用（本轮不追性能；出口 A 只要求如实报告代价） | 记录 |
| 是否分析 compute bound / memory bound | P1 | 不适用（同上） | 记录 |
| 是否考虑 occupancy | P1 | 不适用（同上） | 记录 |
| Stream 生命周期是否正确 | P0 | 通过 | 记录：沿用现有同步路径 |
| Event 同步是否正确 | P0 | 通过 | 记录 |
| Async API 是否正确使用 | P1 | 通过 | 记录 |

**checklists/tensorrt.md**

| Item | Level | Result | Action |
|---|---|---|---|
| ICudaEngine 生命周期是否明确 | P0 | 通过 | 记录：`Engine` RAII |
| IExecutionContext 生命周期是否明确 | P0 | 通过 | 记录：随 `Engine` 持有 |
| Runtime / Engine / Context ownership 是否正确 | P0 | 通过 | 记录 |
| Tensor shape 是否明确 | P0 | 通过 | 记录：落点 = 从引擎读 shape（v2 已补） |
| Dynamic shape profile 是否覆盖输入范围 | P0 | 不通过 | 修复：补 `padding_bias` 的 S 维绑定与 profile 上下界口径（v2 已补） |
| Tensor dtype 转换是否正确 | P1 | 通过 | 记录：按引擎声明精度读写 |
| Plugin creator 注册是否正确 | P0 | 不适用（本轮不改插件） | 记录 |
| Plugin serialize / deserialize 是否完整 | P0 | 不适用（不改插件） | 记录 |
| Plugin enqueue 中的 stream 是否正确 | P0 | 不适用（不改插件） | 记录 |
| Plugin workspace 管理是否合理 | P1 | 不适用（不改插件） | 记录 |
| Binding index 是否正确 | P0 | 通过 | 记录：按名绑定 + 按声明形状 |
| CUDA stream 是否传递正确 | P0 | 通过 | 记录 |
| Async execution 是否正确 | P1 | 通过 | 记录 |

**checklists/llm_runtime.md**

| Item | Level | Result | Action |
|---|---|---|---|
| Request 生命周期是否明确 | P0 | 不适用（诊断用例自建绑定，不走 runner） | 记录 |
| Prefill / Decode 流程是否区分 | P0 | 通过 | 记录：图 A / B / C 全为 prefill |
| Batch 状态是否一致 | P0 | 不适用（固定 batch = 1） | 记录 |
| KV Cache ownership 是否明确 | P0 | 不适用（诊断不走 PagedKVCache，K/V 只作逐层读数） | 记录 |
| Block 管理是否正确 | P0 | 不适用（同上） | 记录 |
| Cache eviction 策略是否明确 | P0 | 不适用（同上） | 记录 |
| Memory fragmentation 是否考虑 | P1 | 不适用（同上） | 记录 |
| Long context 是否测试 | P1 | 不适用（固定短 prompt） | 记录 |
| Dynamic request 加入 / 退出是否安全 | P0 | 不适用（不涉及调度） | 记录 |
| Scheduler 状态是否一致 | P0 | 不适用（同上） | 记录 |
| Batch 调度策略是否合理 | P1 | 不适用（同上） | 记录 |
| Sampling 结果是否正确 | P0 | 通过 | 记录：AC1 用逐 token 语义判据；现象本身即 NaN → 贪心恒 0 |
| CUDA kernel 是否验证 | P1 | 不适用（不改采样 kernel） | 记录 |

## Terminology Check

选取口径：`requirement.md` 中需要可验证定义才能判定"做到没做到"的名词；定义落点为 `analysis.md` 的
`## Terminology`（本轮补出）。

| 模糊名词 | 定义所在 | 结论 |
|---|---|---|
| 具体算子 | `analysis.md` + `## Terminology` 第 1 条 | 已定义 |
| 最小修复 / 最小改动 | `analysis.md` + `## Terminology` 第 2 条 | 已定义 |
| 可复核的结论 | `analysis.md` + `## Terminology` 第 3 条 | 已定义 |
| 与 FP32 基线一致 | `analysis.md` + `## Terminology` 第 4 条 | 已定义 |
| 不回归 / 新红 / 沙箱全绿 | `analysis.md` + `## Terminology` 第 5 条 | 已定义 |
| 非有限值 | `analysis.md` + `## Terminology` 第 6 条 | 已定义 |

## Requirement Coverage Result

| 需求条目 | 设计落点 | 结论 |
|---|---|---|
| Included-复现 | 沿用现有 failure test，v2 不改 | 已落点 |
| Included-诊断 | v2 `### Runtime Flow` 的三图六读数（含失败分支与代价上限） | 已落点（v1 为空壳，v2 补失败分支） |
| Included-修复 | v2 `### Trade-off` 的出口 A / B / C（含各自的判据来源） | 已落点（v1 的 Softmax 分支为空壳，v2 拆开写并标出待核实来源） |
| Included-回归 | v2 出口通过后的 B3 全量 | 已落点 |
| AC1 原问题消失 | 出口 A / B 的判据（逐 token 一致，见 Terminology 第 4 条） | 已落点 |
| AC2 不回归 | B3 全量 + FP32 / INT8 对照（见 Terminology 第 5 条） | 已落点 |
| AC3 可解释 | v2 `### Trade-off` 的假设表（每条带判别量与阈值） | 已落点 |
| AC4 判据有出处 | 不新增数值阈值；仅有的两条算术线给出推导 | 已落点 |

## P0 Blockers

| # | P0 | 证据 | 处置（2026-10-06） |
|---|---|---|---|
| P0-1 | 术语表缺失：`analysis.md` 无 `## Terminology`，表三 6 个词全部未定义（p3_review 的 Entry 硬规则） | 评审时 `analysis.md` 的小节标题清单 | **已关闭**：v2 补出 `## Terminology` 六条可验证定义 |
| P0-2 | 图 B 不复现生产图 NaN 时没有失败分支 → Included-诊断 可能整条落空 | v1 只写了 `L_A == L_B` 的处置 | **已关闭**：v2 `### Runtime Flow` 写明 `L_A != L_B` 的分支与构建次数上限，超限即按出口 C 出结论 |
| P0-3 | 出口 A 的 Softmax 分支无可兑现接口：`addSoftMax`（`gpt2_model_builder.cpp:948`）无精度设置，`ISoftMaxLayer` 在 TRT 10 是否有等价接口**查不到**（本机无 `NvInfer.h`） | `PROGRESS.md` §4.7 第 2 行、`TROUBLESHOOTING.md` 第 3472 行 | **仍开放（阻塞）**：来源 = 真机读 `NvInfer.h` 或建最小图；设备条件缺失，随真机部分一并搁置 |
| P0-4 | `padding_bias` 的动态维绑定未写进设计：诊断用例今天给它设 rank-2 形状、写入 position id 位型（`test_gpt2_generate.cpp:470` 起）；FP16 下恰好下溢成 0，FP32 对照臂会真加进分数 | 代码对照 + `llm_runner.cpp:461-493` 的既有口径 | **已关闭**：v2 `### Resource Lifecycle` 与 `### Data Structure` 写明按声明形状填 0、S 维与 `input_ids` 对齐 |

## P1 Risks

1. **探针集合变化不进引擎指纹** → 诊断引擎路径固定，改了探针集却复用旧引擎 = 安静拿到旧读数
   （`TROUBLESHOOTING.md` TS-040 同类）。设计 v2 要求：探针集合变化必须 bump 图版本或并入指纹。
2. **`setComputePrecision` 的语义未核实**——它承诺的是"避免溢出"还是"覆盖归约本身"不清楚；
   若读到 FP32 仍 NaN，需要有第二控制（显式 Cast 子图）。v2 已把该控制写进出口 A 的第二步。
3. **H2 成立后的手段只有方向**（钉 tactic / timing cache），没有落到具体 API；届时可能又是一轮设计。
4. **出口 A 的代价没有量化口径**——只写了"如实报告"，没写怎么量（实测 vs 估算）。v2 要求按
   "同一次构建的引擎"对照（`PROGRESS.md` §6.4 的口径）。
5. **真机窗口的设备前提**：必须带 `MINI_TRT_REQUIRE_GPU=1`，否则 GPU 用例静默跳过 = 白跑；
   三份引擎各自重建属预期。
6. **策略依赖已裁决**：作者 2026-10-06 撤销"按政策不修"，来源口径 = 正确性（默认路径不可用），
   不是 `future_iterations.md` §1.4 的"低精度性能"触发。旧收益结论本身未被推翻，只是不再作为前提。
7. **P3 入口条件**：本条此前缺 `## 判据对照`，本次已补；补后仍含"未满足（阻塞）"项（真机依赖、P0-3），
   所以下一次进入任何阶段前必须先复核该节。

## P2 Quality

- 探针命名扩到逐层后，测试里硬编码的 768 / 4×768 应改为按模型 config 推导。
- `InspectTactics` 在本方案里是**第三个使用方**（前两个 = `test_gpt2_int8_weights.cpp` 与
  `test_resnet18_int8_probe.cpp`）；后者第 130 行的注释自己写着"若将来出现第三个使用方，再把这份统计
  提到共享头"。按仓库自己的约定，这次该提。
- 设计里"全探针图的输出个数"要与代码里实际存在的 4 个导出点（`*_0`）对齐说明，
  避免读者以为扩容是免费的。

## Decision

**BLOCK**——P0-3 未关闭，Gate-A 不通过，阶段停在 `B2-MinimalFix` / `waiting-human-gate`。
关闭 P0-3 需要真机（读 `NvInfer.h` 或建最小图），当前设备无真机条件 → 该条与 B1' / B2 / B3 一并搁置。
P0-1 / P0-2 / P0-4 的关闭方式已记在 `analysis.md` 的设计 v2 里；关闭后需按本模板**重评一次**
（行数口径同上）。
