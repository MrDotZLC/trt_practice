# Review Report

<!--
本版 2026-10-03 重做，对应重做后的 design.md；**2026-10-04 追加 S3 路线修正后的复评**
（D12 由"固定槽位 + 全批定长"推翻为"活跃批 + 压实"，见文末那一节）。
规则（phases/p3_review.md）：四份 checklists 的每个 [P0] / [P1] 项逐条回答；
三张表齐全（checklists 结论表 / 需求落点表 / 术语定义表）；行数等于对应集合的条目数。
-->

## Summary

设计本身是自洽的：三个里程碑各有独立判据，批量契约（形状、行号同源、块表宽度）写清了，
常驻诊断与 profile 归属这两个"隐性错配源"都被提成显式决策。

三处需要作者拍板才能往下走：**D5**（三条 feature 的串行顺序）、**D7**（诊断开关的默认值）、
**D10**（S3 两步式的收益是否成立——决定 S3 做不做）。另有一处流程缺口：本表"Result"取值集
（通过 / 不通过 / 待确认）没有"不适用"，而本轮不改 kernel，cuda 一份的多数条目只能答"不适用"。

## Checklist Result

| Item | Level | Result | Action |
|---|---|---|---|
| C++-RAII：资源生命周期 | P0 | 通过 | 缓冲与 cache 均由 RAII 持有；S2 预分配后生命周期收敛到构造/析构 |
| C++-Ownership 是否明确 | P0 | 通过 | 见 §Resource Lifecycle 表 |
| C++-悬空引用风险 | P0 | 通过 | S2 之后设备指针恒定；不变量 5 把它列为断言项 |
| C++-异常路径是否释放资源 | P1 | 通过 | D9：调度入口先检查后分配，避免 `BlockAllocator::throw` 穿透；块归还与移除同步 |
| C++-多线程数据竞争 | P0 | 不适用 | 本轮单线程驱动，不引入多线程 |
| C++-Mutex / Lock 生命周期 | P0 | 不适用 | 同上 |
| C++-不必要锁竞争 | P1 | 不适用 | 同上 |
| C++-API 输入输出是否明确 | P0 | 通过 | 批量入口的输入是请求列表，输出是每序列 token 段（§Data Structure） |
| C++-错误处理方式统一 | P0 | 通过 | 沿用"返回是否成功 + 失败路径记录日志"，不新增异常路径（D9） |
| C++-接口是否容易扩展 | P1 | 通过（2026-10-04 复评） | 接口形态已落地：`RunScheduler(SchedulerRequest[])` + 写回带 `rows`/`row_lengths`、采样带 per-row `offsets`；后续 feature（REQ-017 的元素宽度、REQ-019 的 ONNX 路径）按 D11 在同一批量契约上做加法 |
| C++-Debug/Release 可编译 | P0 | 待确认 | 实现阶段验证（P5） |
| C++-是否引入额外依赖 | P1 | 通过 | 不新增第三方依赖 |
| CUDA-Kernel 边界条件 | P0 | 不适用 | 本轮不改 kernel（§Module Design 已声明） |
| CUDA-越界访问 | P0 | 不适用 | 同上；但 S1 依赖"池 ≥ 块表宽度"（不变量 1） |
| CUDA-Synchronization | P0 | 不适用 | 同上；循环内零同步由不变量 3 保证 |
| CUDA-不同 shape 覆盖测试 | P1 | 通过 | P6 计划覆盖 (B,S) 组合与批内长度不齐 |
| CUDA-Global Memory 安全性 | P0 | 不适用 | 本轮不改 kernel |
| CUDA-race condition | P0 | 不适用 | 同上 |
| CUDA-Memory coalescing | P1 | 不适用 | 同上 |
| CUDA-Shared Memory bank conflict | P1 | 不适用 | 同上 |
| CUDA-优化是否有 benchmark 证明 | P0 | 通过 | AC6 要求先声明判别下限再判定；禁止按百分比自动接受 |
| CUDA-compute/memory bound | P1 | 通过 | §Performance Consideration 已定性：decode 访存受限，收益来自权重读取摊薄 |
| CUDA-occupancy | P1 | 不适用 | 本轮不改 kernel |
| CUDA-Stream 生命周期 | P0 | 通过 | 沿用默认流；不变量 3 |
| CUDA-Event 同步 | P0 | 不适用 | 本轮不引入 Event |
| CUDA-Async API 是否正确使用 | P1 | 通过 | 循环内只用 async + 同一 stream 串行；同步只发生在每请求一次的位置 |
| TRT-ICudaEngine 生命周期 | P0 | 通过 | 沿用现有 `shared_ptr<Engine>` 持有方式，不新增所有权 |
| TRT-IExecutionContext 生命周期 | P0 | 通过 | 同上；本轮不改执行上下文管理 |
| TRT-Runtime/Engine/Context ownership | P0 | 通过 | 同上 |
| TRT-Tensor shape 是否明确 | P0 | 通过 | §Data Structure 给出 `[B,S]` / `[B,1]` 与 per-batch 1-D 成员的完整形状 |
| TRT-Dynamic shape profile 覆盖范围 | P0 | 待确认 | 批上限取决于 P4 实测的 prefill logits 显存；未跑 P4 前不能定 |
| TRT-Tensor dtype 转换 | P1 | 通过 | 沿用现有"按引擎实际声明精度分配缓冲"的做法，不新增假定 |
| TRT-Plugin creator 注册 | P0 | 不适用 | 本轮不改插件 |
| TRT-Plugin serialize/deserialize | P0 | 不适用 | 同上 |
| TRT-Plugin enqueue stream | P0 | 不适用 | 同上 |
| TRT-Plugin workspace 管理 | P1 | 不适用 | 同上 |
| TRT-Binding index 正确性 | P0 | 通过 | 每层 cache / K/V 仍按名字绑定；不变量 4 保证行号同源 |
| TRT-CUDA stream 传递正确 | P0 | 通过 | 沿用默认流；同一 stream 上 engine → 写 K/V → 采样串行 |
| TRT-Async execution | P1 | 通过 | 循环内 async；同步点仅在每请求开始/结束 |
| LLM-Request 生命周期 | P0 | 通过 | §Data Structure 的状态机 + §Resource Lifecycle 的块生命周期 |
| LLM-Prefill / Decode 是否区分 | P0 | 通过 | §Runtime Flow 明确两段，且 S3 用两步式避免混批 |
| LLM-Batch 状态是否一致 | P0 | 通过 | 不变量 4（行号同源） |
| LLM-KV Cache ownership | P0 | 通过 | 仍由运行时持有，调度只通过分配/回收接口操作 |
| LLM-Block 管理正确性 | P0 | 通过 | S2：先检查后分配 + 归还与移除同步 |
| LLM-Cache eviction 策略 | P0 | 通过（本轮不做） | requirement 的 Excluded 明确排除抢占与换出；无 eviction 即无该风险 |
| LLM-Memory fragmentation | P1 | 待确认 | 块池是定长块，预期无碎片；需 S2 后按"跑 N 轮"实测空闲块水位 |
| LLM-Long context 是否测试 | P1 | 待确认 | 依赖真机；`S=512` 下 logits 显存是主要约束 |
| LLM-Dynamic request 加入/退出是否安全 | P0 | 通过（设计 + 实现，2026-10-04 复评） | 五步流程 + 块归还与移除同步 + `SequenceScope` 兜底失败路径；用例 `SequenceRetiresAndRowCompacts` / `BlocksReturnAtEnd`（含失败路径）待真机跑绿才算闭环 |
| LLM-Scheduler 状态是否一致 | P0 | 通过（2026-10-04 复评） | 活跃表 + 每步重建 + 不变量 4 的显式校验（`RowOf(seq) == generation_rows + j`）；结果按请求槽位聚集、行号每步压实 |
| LLM-Batch 调度策略是否合理 | P1 | 待确认 | D10：两步式是否吃掉收益要用 P4 数据判（结论不变） |
| LLM-Sampling 结果是否正确 | P0 | 通过（设计） | D3 把参数扩到 per-batch；正确性由 AC1 的对拍判 |
| LLM-CUDA kernel 是否验证 | P1 | 不适用 | 本轮不改采样 kernel |

## Terminology Check

| 模糊名词 | 定义所在 | 结论 |
|---|---|---|
| 批量 > 1 / 批 | analysis.md Terminology 第 1 行 | 已定义 |
| 静态批 | 同上 第 2 行 | 已定义 |
| 最小连续批 | 同上 第 3 行 | 已定义（并明确"一步之内不变化"） |
| 每序列独立的采样参数 | 同上 第 4 行 | 已定义 |
| 跨请求重复调用时不泄漏 | 同上 第 5 行 | 已定义（引用相等，并含失败路径） |
| 完全一致 | 同上 第 6 行 | 已定义（逐位相同，且写明"top-1 相同不算通过"） |
| 逐序列正确 | 同上 第 7 行 | 已定义 |
| 空闲块数回到初始值 | 同上 第 8 行 | 已定义 |
| 无显著差异 | 同上 第 9 行 | 已定义（绑定本机判别下限与出处） |
| 可复现且不自证 | 同上 第 10 行 | 已定义 |

## Requirement Coverage Result

| 需求条目 | 设计落点 | 结论 |
|---|---|---|
| Included 1：批量 > 1 的 prefill 与 decode | §Architecture + §Runtime Flow（S1） | 已落点 |
| Included 2：每序列长度记账 / 位置编码 / 结果收集 | §Data Structure + §Runtime Flow（S1） | 已落点 |
| Included 3：每序列独立采样参数 | §Data Structure + D3 | 已落点 |
| Included 4：每序列 K/V 分配回收与不泄漏 | §Resource Lifecycle + §Runtime Flow（S2） | 已落点 |
| Included 5：请求级调度 | §Runtime Flow（S3 状态机）+ D10 | 已落点 |
| Included 6：批量 == 逐条单跑 | §验证策略 | 已落点（验证型） |
| AC1 数值一致性 | §验证策略 | 已落点（验证型） |
| AC2 长度不齐 | §Runtime Flow（S3）+ §验证策略 | 已落点 |
| AC3 资源回收 | §Resource Lifecycle + §Runtime Flow（S2） | 已落点 |
| AC4 不回归 | §验证策略 | 已落点（验证型） |
| AC5 单序列语义不变 | §验证策略 | 已落点（验证型） |
| AC6 性能可复现且不自证 | §验证策略 + §Performance Consideration | 已落点（验证型） |

## P0 Blockers

无。

## P1 Risks

| 风险 | 需要谁确认 |
|---|---|
| D5：本 feature / REQ-017 / REQ-019 共用运行时入口，串行顺序未拍板 | 作者 |
| D7：常驻诊断开关的**默认值**（建议默认关，P4/P7 强制关） | 作者 |
| D10：S3 两步式可能吃掉批量收益，收益成立与否要 P4 数据 | 作者（并影响 S3 是否交付） |
| ~~AC2 / Included 5 完全依赖 S3~~ | 已确认（2026-10-03）：S3 交付，判据 2 不下修 |
| 批上限与 profile 范围依赖 P4 实测（prefill logits 显存） | 真机 P4 |
| AC1 / AC3 / AC5 只能在真机验证；沙箱无 GPU | 真机 P6 |
| S3 需要改图（加 padding mask）→ 必须 bump `graph_version` 且与图改动同提交 | 实现阶段 |

## P2 Quality

| 项 | 说明 |
|---|---|
| 本表 Result 取值缺"不适用" | 已处置（2026-10-03）：技能 `phases/p3_review.md` 把取值集补为 通过 / 不通过 / 待确认 / 不适用，并要求"不适用"必须写明理由 |
| design.md 的 `## 验证策略` 是无模板章节 | 为承载验证型判据而新增；若技能后续补了同类章节，应合并 |

## Gate-A 确认记录（2026-10-03）

| 待确认项 | 作者结论 | 落点 |
|---|---|---|
| D5 三条 feature 的串行顺序 | 本 feature 先做（`REQ-017` / `REQ-019` 均未开工，顺序天然成立） | design.md D5 |
| D7 常驻诊断开关默认值 | **默认关闭**；P4 / P7 强制关闭 | design.md D7 |
| D10 S3 两步式的收益门槛 | **做 S3**；P4 的两种负载对照只用于把收益结论写准，不作为 S3 的交付门槛 | design.md D10 |
| AC2 / Included 5 是否下修 | 不下修（S3 交付，判据 2 保持原样） | design.md Requirement Coverage |
| 与后续 feature 的接口面 | 按"可维护性 / 可扩展性"补齐，见 design.md D11 | design.md D11 |

其余 P1（真机验证、批上限实测、改图 bump）按 STATE.md 的 Next Action 执行。

## S3 路线修正后的复评（2026-10-04）

**触发**：D12 在 2026-10-04 被作者推翻——批的形态由"固定槽位 + 全批定长"改为"**活跃批 + 压实**"，
并新增 D13（S3 padding 为生产路径、S4 packed 为默认路径）。10-03 那版评审对应的是旧路线，
因此对受影响的条目做一次增量复评（术语表与需求落点表不受影响，未重列）。

**复评结论（受影响条目）**

| 条目 | 旧结论 | 复评结论 | 依据 |
|---|---|---|---|
| LLM-Scheduler 状态是否一致（P0） | 待确认 | 通过 | 活跃表五步（retire/admit/context/generation/sample&flag）+ 压实行号；不变量 4 落成显式校验 |
| LLM-Dynamic request 加入/退出是否安全（P0） | 待确认 | 通过（闭环待真机） | 准入"先查后分配"（D9）+ `SequenceScope` 保证任何出口归还；用例覆盖失败路径 |
| C++-接口是否容易扩展（P1） | 待确认 | 通过 | 见上表 |
| LLM-Prefill / Decode 是否区分（P0） | 通过 | 通过（前提加强） | 由"分两段"加强为"**每次引擎调用只装一种相**"——这是不出现"覆盖 running 行 K/V"的必要条件 |
| LLM-Batch 状态是否一致（P0） | 通过 | 通过（口径修正） | 不变量 4 的"结果第 i 段"改为**按请求槽位聚集**（行号每步变），行↔结果的映射由每步散射显式维持 |
| C++-Debug/Release 可编译（P0） | 待确认 | 待确认 | 本环境无编译器，P5 Exit Gate 仍待真机 |

**复评新发现（10-03 版设计与 spec 未覆盖，实现阶段才暴露）**

| 发现 | 为什么是设计缺口 | 处置 |
|---|---|---|
| 写回必须带**逐行真实长度** | padding 路径下每行真实 prompt 长度不同，而 `WritePrefillKV` 只有全局 `tokens`（= 写入 stride）→ 语境长度会被写成 stride，decode 从填充位置起算、并把填充位置纳入注意力（AC2 不成立） | 加 `row_lengths`，签名与理由回填 `p5_s3_interface_spec.md` §3 |
| `AppendDecodeStep` 必须能**只追加前 N 行** | 生成段只装活跃表前缀，按 `batch_size()` 追加会把本步刚入批的 context 行的长度也推一格、并往它们的块里写 K/V | 加 `row_count`，回填 spec §3 |
| 随机步号必须**逐行** | 调度下同一次引擎调用里各行"已生成计数"不同；标量 offset 会让"到得晚"的请求换一条随机流 → AC1 不成立 | 加 per-row `offsets`，回填 spec §5 |
| 结果不能按行号存放 | 退出即压实行号，按行存的 token 会被喂错行 | 结果按请求槽位聚集（每步按行聚集/散射） |

**两条判据的可观测性（复评时的 P1 → 已处置）**："EOS 是否**在下一步**退出"与
"context 段是否只装新入批的行"在公开接口上都不可观测
（`EosRetiresImmediately` 的结果口径对"提前退出"和"跑满再截断"是等价的）。
已加 `LLMRunner::SchedulerStats`（只读）并用 `steps` / `context_rows` 把两条判据固定进用例（见 test_plan.md）。

**复评顺带修掉的一处配置缺陷**：S1 的批量用例声明 `max_batch = 2` 却复用 1/1/1 的 profile
（批 2 的形状落在 profile 之外，`SetInputShape` 会失败）。公共 fixture
`SmallGpt2BuilderConfig(max_batch = 1)` 现在接受批上限，批量用例显式传入。

**Decision（复评）**：PASS——无 P0，无新增待确认 P1；上面四处发现已回填 spec 并落码，
真机编译与 P6 用例仍是唯一未闭环项。

## S4 / S5 设计的 P3 增量复评（2026-10-04）

**触发与范围**：S4（packed 混合批：一个张量装两相、attention 按段分派）的 P2 设计草案出齐
（`p5_s4_interface_spec.md`），S5（chunked prefill）同日本作者改判立项。10-03 那版评审对应的是
S1/S2/S3 的 padding 路线，S4 把执行形态从"每步两次调用"改成"一次调用装两相"、S5 又引入第三种
计算模式，因此按 P3 规则逐条回答**受影响的** [P0]/[P1] 项。

**一个必须点明的口径变化**：10-03 版里多数 CUDA 条目答的是"不适用（本轮不改 kernel）"——
**S4/S5 要改 kernel（新增插件）**，所以这些项从"不适用"转为"必须回答"，本表就是它们的新答案。

### 表一（增量）：受影响的 checklists 条目

| Item | Level | Result | Action |
|---|---|---|---|
| CUDA-Kernel 边界条件 | P0 | 通过（设计层） | §3 的下标纪律 + §4 的段内/段间映射写死；实现时逐条自检（段边界、`cu_seqlens` 单调、每段 token 区间） |
| CUDA-越界访问 | P0 | 通过（设计层） | packed 的 `T` 边界 + block table 宽度；沿用"越界立即返回、不读不写"的既有纪律（paged 插件同款） |
| CUDA-race condition | P0 | 通过 | 两段在**同一 kernel 内**按位置区间分派，不共享写目标；两段的 K/V 写回都在图外同 stream 串行 |
| CUDA-Synchronization | P0 | 通过 | 图内无同步；沿用 S3 的"每步重建缓冲 + 同流有序"；不引入 Event |
| CUDA-不同 shape 覆盖测试 | P1 | 通过 | profile 的三个极端（`T=1` / 纯 context / 纯 generation）进用例 `CuSeqlensBoundaryCases` |
| CUDA-Async API 是否正确使用 | P1 | 通过 | 与 S3 同一纪律：循环内只用 async + 同 stream 串行 |
| CUDA-Stream 生命周期 / Event 同步 | P0 | 通过 / 不适用 | 沿用默认流；不引入 Event |
| TRT-Plugin creator 注册 / serialize / enqueue stream | P0 | 通过（设计层） | 新插件按 `IPluginV3` 三能力拆分；注册与序列化沿用现有插件的约定 |
| TRT-Plugin workspace 管理 | P1 | 通过 | 走 `getWorkspaceSize()`，**禁止** enqueue 内 `cudaMalloc`（AGENTS.md §3.B.3） |
| TRT-Binding 一致性 | P0 | 通过 | §3 列了全部输入/输出；`graph_version` 必须 bump，且与 S3 的两套图区分 |
| TRT-Tensor shape 是否明确 | P0 | 通过 | §3 的契约表（`input_ids[T]` / `position_ids[T]` / 段边界标量 / 每段 `cu_seqlens` / cache 输入） |
| TRT-Dynamic shape profile 覆盖范围 | P0 | 通过（待实现核对） | packed 的 `T ∈ [1, max_batch × max_prefill_seq_len]`；沿用 D8 的构造期校验 |
| TRT-CUDA stream 传递正确 | P0 | 通过 | 沿用默认流 |
| TRT-ICudaEngine / IExecutionContext 生命周期 | P0 | 通过 | 沿用 `shared_ptr<Engine>`；S4 多一张图，不新增所有权模型 |
| C++-API 输入输出是否明确 | P0 | 通过 | §3 + §4 的**下标纪律**（段内下标独立、禁止跨段混用、按行输入按 packed 行序） |
| C++-RAII / Ownership / 悬空引用 | P0 | 通过 | 映射数组由 runner 持有；引擎沿用 `shared_ptr`；S3 的"预分配 + 指针恒定"不变 |
| C++-异常路径是否释放资源 | P1 | 通过 | S4/S5 不新增异常路径；沿用 `SequenceScope` 的"任何出口都归还" |
| C++-接口是否容易扩展 | P1 | 通过 | `prefill_mode` 开关 + 不改调用方接口（AC8）；S5 在 packed 契约上是加法 |
| C++-Debug/Release 可编译 | P0 | 待确认 | 真机（本环境无编译器） |
| LLM-Batch 状态是否一致 | P0 | 通过（口径改写） | 不变量 4 从"下标天然相同"改为"**显式映射数组**是唯一依据"（packed 行序 ≠ 缓存行序） |
| LLM-Sampling 结果是否正确 | P0 | 通过 | 三段行序（packed / cache / 结果）都走映射；末位定位按段各写一遍 |
| LLM-Scheduler 状态是否一致 | P0 | 通过 | 调度**策略**不分叉（准入 / 退出 / 压实 / D9 预算） |
| LLM-Dynamic request 加入 / 退出是否安全 | P0 | 通过（S4）；S5 另有待办 | S5 要在活跃表加"prompt 进度"字段 → 那条属于 S5 自己的设计 |
| LLM-Long context 是否测试 | P1 | 待确认 | 这是 S5 的动机；判据 AC9 + 真机长 prompt 场景 |
| LLM-Memory fragmentation | P1 | 待确认 | S4/S5 改变显存占用形态（logits 由 `B·S_max·V` 变 `T·V`）→ P4 量 |
| LLM-Batch 调度策略是否合理 | P1 | 待确认 | D10 的两种负载对照 → P4/P7；S4 的目的就是消掉"每步第二次调用"的代价 |
| CUDA-优化是否有 benchmark 证明 | P0 | 通过（口径） | AC6 不变：先声明判别下限再 A/B；"默认用 S4"已在 D13 写明是**设计决定**、落地后仍要 A/B |

### 表二（增量）：需求落点

| 需求条目 | 设计落点 | 结论 |
|---|---|---|
| Included 7：按真实长度计费 + 两条路径（打包为默认） | `p5_s4_interface_spec.md` §3–§8 + D13 / D14 | 已落点（S4） |
| Included 8（**新增**）：长 prompt 分块推进（S5） | design.md D15（范围 / 代价 / 依赖 / 判据）+ requirement AC9 | 已落点（**里程碑级**）；S5 自己的接口细化（P2）待补 |
| AC7 不浪费 | S4 的 packed 计费 + §6 的显存账（`T·V` vs `B·S_max·V`）+ 用例 `PackedShortSequenceNotPenalized` | 已落点 |
| AC8 两条路径各自成立且可回退 | §8 的 `prefill_mode` 开关（不改调用方接口）+ §9 的用例 | 已落点 |
| AC9（**新增**）：分块与不分块逐位相同 | D15；S5 的用例待其设计细化时补 | 已落点（里程碑级） |

### 复评发现（按 P2 记录，但两条与实现阶段的 P0 检查挂钩）

1. **段内下标混用**（作者指出）：把两段的序列下标当成同一个 `i` 会让分派与写回错位 —— 且是**静默**错。
   已落成 §3 的下标纪律 + §4 的按段公式；实现时必须有入口校验 + 用例 `ContextTokensPrecedeGeneration`。
2. **packed 行序 ≠ 缓存行序**：映射数组成为行号同源的唯一依据（不变量 4 口径改写）。
3. **`T` 的 profile 上限**：`max_batch × max_prefill_seq_len`，与 `max_batch` 必须一起校验（D8 同款入口）。
4. **单插件内两份 attention 实现（A1）**：插件复杂度上升 → S3 路径保留为对照，A2 留作退路。
5. **S5 尚无自己的设计**：按"改代码前必须有对应设计 artifact"，S5 开工前必须先补 P2（接口细化）。

### Decision（复评）

- **P0：无**（S4 设计层面）；表一 Action 列出的实现期检查项作为实现阶段的 P0 自检清单。
- **P1（两条，需要作者确认）**：① 性能类判据（AC6 / AC7 / D10 的负载对照）仍绑真机，环境不可用；
  ② S5 的 P2 何时补（建议排在 S4 实现之后、S5 开工之前）。
- **结论：PASS（设计层面）**。S4 可在真机窗口进入 P5 实现；S5 的代码在它自己的 P2 补齐前不开工。

### S4 文档的第二遍复评（2026-10-04，作者要求）

**为什么再评一遍**：第一遍是对着**设计裁决**评的（口径 / 引擎形态 / 插件形态）；这一遍是对着**文档本身**评的，
专找内部不一致与实现级缺口。结果：**6 处陈旧措辞 + 5 处实现级缺口 + 2 条新待定**，均不改变设计裁决。

**A. 内部不一致（当场修）**

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| A1 | `p5_s4` §2 段边界行 | 仍写 `cu_seqlens[B_ctx]`（单数组时代的名字） | 改 `cu_seqlens_ctx[B_ctx]` |
| A2 | §2 打包行 | 仍写"重建 `cu_seqlens[B_total+1]`" | 改"`cu_seqlens_ctx[B_ctx+1]` + 段边界标量" |
| A3 | §2 采样行 | 仍写"末位 = `cu_seqlens[i+1]-1`" —— **正是作者指出"两段 i 会撞"的那句**（§4 改了、§2 漏了） | 改"按 §4 的按段公式" |
| A4 | §5 写回 | `cu_seqlens[i]` / `cu_seqlens[i+1]` 用同一个 `i` 表两段 | 改段内下标 `j`，generation 段写绝对式 `cu_seqlens_ctx[B_ctx] + j` |
| A5 | §6 优化行 | `Gather(cu_seqlens[i+1]-1)` 同名问题 | 改"按 §4 的末位下标" |
| A6 | §11 标题 | 仍写"待作者确认（进入 P3 之前）"，而 5 条已定 | 改"确认记录与待定项" |

**教训（写进这里，避免复发）**：改公式必须**全篇搜旧名字** —— 这次是 §4 改了、§2/§5/§6 漏了；
这与 10-03 那次"口头确认没回到文档"是同一类问题（表述落后于裁决）。

**B. 实现级缺口（已补进 §3/§5/§6/§7 + 一条用例 + 两条风险）**

| # | 缺口 | 后果（不补就会踩） | 处置 |
|---|---|---|---|
| B1 | S3 的 `UploadMetadata` 是"缓存行序镜像**直传**"；S4 的引擎行是 packed 行序 | 引擎行与缓存行错位 → **静默算错** | §3 写明每步按 packed 行序重建 `block_tables`/`context_lens`；用例 `PackedMetadataFollowsPackedOrder` 直接锁 |
| B2 | generation 段 `position_ids` 的取法 | 照 S3 去设备端搬（多一步）；或误以为 host 长度不可信 | §3 写明走 **host 长度镜像**（S3 已把它维护成准确的），并点明这条依赖 |
| B3 | 写回/追加的源寻址变了 | 实现者按 S3 的"每层一段连续 `[row_count,H,1,D]`"接，直接错位 | §5 写明：写回要多收**设备端** `cu_seqlens`（源偏移数据相关）；追加要收"源基址 + 行集 + 行映射" |
| B4 | 块预留口径 | 照抄 S3 的 stride 口径会白占块（并发度被 D9 预算卡住） | §5 写明 packed 下按真实长度预算；并在 `p5_s3` §3 标注"按 stride 只属于 padding 路径" |
| B5 | 采样前聚集的实现方式 | 可能去新写 kernel 或把 logits 读回主机（后者破"循环内零同步"） | §6 写明复用 S3 的**逐行 async D2D**（host 算偏移） |

**C. 第二遍复评新增的两条 —— 作者 2026-10-04 按推荐确认（已闭环）**

1. **`SchedulerStats` 的口径**：**按路径分别定义** —— 跨路径口径 = `steps` / `max_active` / `context_rows` /
   `generation_rows`（两条路径的用例只依赖这四个）；`prefill_calls` / `decode_calls` **仅 S3** 的两段式
   （S4 下不读、也不去凑语义）。S3 代码已补 `generation_rows`，头文件写明这条口径。
2. **packed 引擎 profile 的 `opt` 值**：**P4 实测后定**；实现时先用显式标注"待实测"的保守值 ——
   现在拍板等于把它变成隐性契约。

**结论**：设计裁决不变（第一遍的 PASS 仍成立）；第二遍把"文档级不一致"与"实现级缺口"补齐，
并在 §9 补了一条直接锁 B1 的用例。

### Gate-A 确认记录（S4 / S5，2026-10-04）

| 待确认项 | 作者结论 | 落点 |
|---|---|---|
| S4 设计 5 条待确认 | 全部确认（分路径前提 / `p5_s3` §10 限定 / 插件 A1 / 映射与不变量 4 口径 / 段内下标纪律） | `p5_s4_interface_spec.md` §11；design.md S4 小节与 D14 |
| S5（chunked prefill）是否纳入范围 | **纳入，立项为里程碑 S5**（不阻塞 S4） | requirement Included 8 / AC9；design.md D15 |
| 第二遍复评的两条 | `SchedulerStats` **按路径分别定义**（跨路径四个量）；profile 的 `opt` **P4 实测后定** | `p5_s4_interface_spec.md` §6/§7；`SchedulerStats` 的字段注释 |
| S4 是否开工 / 范围 | **通过；全做（含新插件与图），实机测试先搁置** | STATE.md 的 P5-S4 节与 Phase History |

**P0：无。P1：一条** —— 性能类判据（AC6 / AC7 / D10 的负载对照）仍绑真机，环境不可用；
S4 的代码按"未编译验证"记账，真机窗口的第一件事是编译（P5 Exit Gate）。

## S5 设计的 P3 增量复评（2026-10-04）

**触发与范围**：chunked prefill 立项为 S5（requirement Included 8 / AC9），P2 草案出齐
（`p5_s5_interface_spec.md` + design.md D15/D16），作者同日确认了 §8 的四条。
本节的增量点：**S5 会第一次修改 attention 的计算语义**（context 段从"自包含 varlen"变成"分页因果"），
所以 CUDA 侧的 P0/P1 项要重新回答；TRT 侧反而**不动**（同图、同 profile 区间、同 `graph_version`）。

### 表一（增量）：受影响的 checklists 条目

| Item | Level | Result | Action |
|---|---|---|---|
| CUDA-Kernel 边界条件 | P0 | 通过（设计层） | query 数 > 1 + 按段因果 + 末块变短，全部由 `cu_seqlens_ctx` 分段表达；实现期逐条自检 |
| CUDA-越界访问 | P0 | 通过 | 写回位置从 `prompt_done` 起、预留仍按 `prompt_len + max_new`（S4 已改成按真实长度）→ 不会越出预留 |
| CUDA-race condition | P0 | 通过 | 与 S4 同一分派结构：各行的 K/V 写目标互不相交（按块表）+ 图外按 stream 串行 |
| CUDA-Synchronization | P0 | 通过 | 不新增同步点；沿用"每步重建 + 同流有序" |
| CUDA-不同 shape 覆盖测试 | P1 | 通过 | 两个极端（全对齐步 / 含末块的步）进 `ChunkedEqualsWholePrompt`（`chunk_limit` 取 1 / 中间值 / ≥ prompt_len） |
| CUDA-优化是否有 benchmark 证明 | P0 | **待确认** | "fused kernel 的收益"只能由 P4/P7 判（环境搁置）；本设计只保证**不静默回退**（§3 兜底纪律） |
| TRT-Tensor shape / Dynamic profile | P0 | 通过 | **S5 不动图**：chunk 只让每步的 `T` 变小，仍在 S4 已定的 `[1, max_batch × max_prefill_seq_len]` 内 |
| TRT-Binding 一致性 / `graph_version` | P0 | 通过 | 同图同绑定，**不需要 bump**（S4 契约的直接红利） |
| TRT-Plugin workspace 管理 | P1 | 通过 | 仍走 `getWorkspaceSize()`；拿不到 workspace 时按 §3 的兜底纪律**显式拒绝**（不再沿用 paged 插件"退单趟"那种静默降级） |
| C++-API 输入输出是否明确 | P0 | 通过 | **不新增开关**、不改调用方签名（`chunk_limit` 由 profile 推导） |
| C++-RAII / 异常路径 | P0/P1 | 通过 | 不新增资源类型；失败路径仍由 `SequenceScope` 兜底归还 |
| LLM-Scheduler 状态是否一致 | P0 | 通过 | 活跃表新增 `prompt_done`，与行号/压实同源（不变量 4 的口径不变） |
| LLM-Dynamic request 加入 / 退出是否安全 | P0 | 通过 | 分块中途不抢占、不换出；退出判据（EOS / `max_new`）只在 prefill 完成后参与 |
| LLM-Long context 是否测试 | P1 | 待确认 | 这是 S5 的动机：用例 + 真机长 prompt 场景 |
| AC9 需求落点 | — | 已落点 | 判据写成"分块与不分块逐位相同"（`p5_s5_interface_spec.md` §6） |

### 复评发现

1. **`chunk_limit` 必须落在 fused kernel 支持的常量集合内**（作者口径：由 profile 推导 + 显式拒绝）——
   实现时要在**构造期**校验并给出可操作的错误信息，否则"推导值"会在运行期撞上 kernel 的假设。
2. **非末块对齐 / 末块按实际长度**要写成一条规则的**两半**（不是两种模式）：对齐是为了形状与 kernel
   假设稳定，末块变短由同一个 kernel 按 `cu_seqlens` 处理 —— 这正是 §3"不静默回退"的边界。
3. **`SchedulerStats` 的口径要跟着改**：分块后 `context_rows` 变成"Σ 本步 chunk 行数"，而一条序列可能
   **跨多步**出现在 context 段（S4 下它只出现一次）。用例断言与 `test_plan` 的行要按新口径写，
   免得把"同一序列多次出现在 context 段"误判成缺陷。
4. **退出判据的计时**：`max_new` 从 prefill 完成起算 —— 这条要落成活跃表字段语义（`generated` 只在
   完成后才增长），否则分块会改变可见语义；用例 `ChunkProgressStateIsCorrect` 锁住。

### Decision（复评）

- **P0：无**（S5 设计层面）；表一的 Action 列作为实现阶段的 P0 自检清单。
- **P1（两条）**：① fused kernel 的收益判据绑 P4/P7（环境不可用）；② 真机长 prompt 场景。
- **结论：PASS（设计层面）**。**实现顺序：作者 2026-10-04 改判** —— S4 的真机测试先搁置，
  **S5 的代码先做**（两者共用同一张 packed 图，S5 只在它之上加 chunk 语义，
  不改图 / profile）。原写的"排在 S4 真机收口之后"这条前提**作废**；
  真机窗口恢复后按 S4 → S5 的顺序一起验证（S5 的用例与 S4 的回归同一批跑）。
  （2026-10-04 第二遍复评改判：`graph_version` 不能再写"不改"，见下节 P1-1。）

## S5 设计的第二遍复评（2026-10-04，作者要求）

**触发与范围**：作者要求对 S5 的设计方案再做一遍评审。判据与上一节相同（四个 checklists 的
[P0] / [P1] 逐条回答 + 需求落点表 + 术语表），但**核对对象从"设计是否自洽"改成"设计与代码现状
是否对得上"**：上一节的 15 条增量项多数只回答了"设计怎么说"，没有对照
`packed_attention_plugin.cu` / `paged_attention_split.hpp` / `paged_kv_cache_kernels.cu` /
`llm_runner.cpp` 的真实形态。按 `AGENTS.md` §5 的「反向也要查」，与代码矛盾处必须当场修，
因此这一节改判了上一节的结论。

### 核对通过的既有能力（S5 的便宜处，先记下来）

| 说法 | 代码证据 | 结论 |
|---|---|---|
| 读缓存不需要新输入 | packed 插件的输入已含 `block_tables[5]` / `context_lens[6]` / `cu_seqlens_ctx[7]`（`packed_attention_plugin.hpp` §输入契约） | 成立：不动绑定、不动图拓扑 |
| 注意力时刻的"已写入长度"就是 `prompt_done` | runner 顺序是 ②上传元数据 → ③跑引擎 → ④写回；上传时 `host_context_lens[j] = SequenceLength(seq_id)`（`llm_runner.cpp:1458`） | 成立：分页部分天然是 `[0, prompt_done)` |
| 写回起点可白拿 | 写回发生在引擎调用之后，设备端 `context_lens` 仍是本步之前的值 | 成立：kernel 用 `context_lens[row] + t`，不需要新输入 |
| 压实不丢已写 chunk 的 K/V | `FreeSequence` 按各序列自己的块表重建行镜像（`paged_kv_cache.cpp:157-168`） | 成立：块归属跟着 sequence |
| 首块自动退化 | `prompt_done == 0` ⇒ 分页部分长度为 0 | 成立（前提是新 kernel 显式处理 cache 长度 0） |

### 表一（增量，含改判）：受影响的 checklists 条目

标注「改判」的行是本节相对上一节改了结论的项；未列出的条目在本轮不受 S5 影响（沿用上一节的结论）。

| Item | Level | Result | Action |
|---|---|---|---|
| CUDA-Kernel 边界条件 | P0 | **不通过（改判）** | 上一节写"由 `cu_seqlens_ctx` 分段表达"，但没写每 query 的 key 数上界与新 kernel 的形态；现补为"新写 chunked context kernel + `key 数 <= 1024`"（D16 / spec §3） |
| CUDA-越界访问 | P0 | **待确认（改判）** | 写回从 `prompt_done` 起是对的，但 host 侧记账必须同时从"赋值"改"累加"、预留量校验改按累计长度；只改 kernel 会静默错（spec §4） |
| CUDA-Synchronization | P0 | 通过 | 不新增同步点，沿用"每步重建 + 同流有序" |
| CUDA-Global Memory 访问是否安全 | P0 | **待确认（改判）** | 新增"按块表读缓存前缀"的访问面；越界保护要覆盖"累计长度 vs 预留量"（原稿只按单次 chunk 长度） |
| CUDA-是否存在 race condition | P0 | 通过 | 各行的 K/V 写目标互不相交（按块表）+ 图外按 stream 串行 |
| CUDA-Stream 生命周期 | P0 | 通过 | 沿用默认流 |
| CUDA-Event 同步 | P0 | 不适用 | 本轮不引入 Event |
| CUDA-优化是否有 benchmark 证明 | P0 | 待确认 | 收益判据仍绑 P4/P7（环境搁置）；本轮补记"首块也吃分页开销"这个代价 |
| CUDA-不同 shape 是否覆盖测试 | P1 | 通过 | 全对齐步 / 含末块步 + `chunk_limit` 取 1 / 中间值 / ≥ prompt_len |
| TRT-Tensor shape 是否明确 | P0 | 通过 | chunk 只让 `T` 变小，仍在 `[1, max_batch × max_prefill_seq_len]` 内 |
| TRT-Dynamic shape profile 是否覆盖 | P0 | 通过 | 同 S4 区间；真实约束是 `n_positions <= 1024` 与 `prompt_len <= n_positions` |
| TRT-Plugin creator / serialize / enqueue stream / Binding 一致性 | P0 | 通过 | 输入个数与顺序不变 |
| TRT-Plugin workspace 管理 | P1 | **不通过（改判）** | "不改 workspace"未被证明过：若走 split-K 扩 query 的方案，`getWorkspaceSize` 会变 → 必须 bump `graph_version`。现采用"新写 kernel、score 留在 shared"的方案把这条约束**变成可证的**（D16 方案 A/C 对比） |
| C++-API 输入输出是否明确 | P0 | 通过 | 不新增开关、不改调用方签名 |
| C++-错误处理方式是否统一 | P0 | 通过 | 新路径的越界改为入口显式拒绝（兜底纪律的检查对象已落成真实常量） |
| C++-Debug / Release 是否均可编译 | P0 | 待确认 | 实现阶段验证（P5，本环境无 nvcc） |
| LLM-Prefill / Decode 是否区分 | P0 | **不通过（改判）** | S5 引入第三种"分块中的 prefill"行；"本步完成的行"在活跃表里可能不连续，而采样器只吃连续 `[count]` → 需要显式行列表 + 紧凑暂存（spec §2/§4） |
| LLM-Scheduler 状态是否一致 | P0 | **待确认（改判）** | ① `prompt_done` 与 `PagedKVCache::Sequence::length` 是同一事实的第二份拷贝，应单源；② 活跃表要能表达"生成中 / 分块中 / 新准入"三类并存 |
| LLM-Batch 状态是否一致（不变量 4） | P0 | 待确认 | 采样行集不得靠重排活跃表实现（会破行号同源），故选显式行列表 |
| LLM-Dynamic request 加入 / 退出是否安全 | P0 | 通过 | 分块中途不抢占、不换出；退出判据只在 prefill 完成后参与 |
| LLM-Long context 是否测试 | P1 | 待确认 | S5 的动机本身；用例 + 真机长 prompt 场景 |
| LLM-Batch 调度策略是否合理 | P1 | 待确认 | D10 的结论不变（两步式收益仍要 P4 数据） |
| AC9 需求落点 | — | **已落点（本轮补）** | 判据写成"分块与不分块逐位相同"，并增补位置与采样行集两个用例 |

### 表二（增量）：需求落点

| 需求条目 | 设计落点 | 结论 |
|---|---|---|
| Included 7：不等长 prefill 按真实长度计费 + 两条路径（打包为默认） | §Runtime Flow（S3/S4）+ D10 + D13 | 已落点（`design.md` 的 Requirement Coverage 表本轮补登） |
| Included 8：长 prompt 的分块推进（S5） | §Runtime Flow（S5）+ D15 + D16 | 已落点（本轮补登） |
| AC7 不浪费 | §验证策略（AC7 行）+ D10 | 已落点（本轮补登） |
| AC8 两条路径各自成立且可回退 | §Runtime Flow（S3/S4）+ D13 + D14 | 已落点（本轮补登） |
| AC9 分块与不分块等价 | §Runtime Flow（S5）+ D16 + `p5_s5_interface_spec.md` §6 | 已落点（本轮补登；含位置与采样行集两条） |

### 表三（增量）：术语定义

| 模糊名词 | 定义所在 | 结论 |
|---|---|---|
| 分块 / chunk（Included 8、AC9） | `p5_s5_interface_spec.md` §2（`chunk_len = min(prompt_len - prompt_done, chunk_limit)`、非末块对齐 / 末块按实际长度） | **已定义（指针式登记，作者 2026-10-04 定）**：`analysis.md` 的 Terminology 里新增了条目，但**不复制正文**，指向 spec §2 作唯一来源 |
| 末块 / 非末块 | 同上 §2 | 已定义（同"分块"一条，指针式登记） |
| chunk 的绝对位置 | 同上 §2（chunk 内第 i 个 token 的 `position_ids` = `prompt_done + i`） | 已定义（指针式登记；判伪口径见该行） |
| 分块与不分块逐位相同 | `analysis.md` Terminology 的"完全一致"（逐位相同，top-1 相同不算通过） | 已定义（AC9 复用该口径） |

### P0 Blockers

1. **position_ids 的绝对位置在 S5 侧没有落点（数值正确性）。** `llm_runner.cpp:1426` 现在写的是
   `host_positions[offset + i] = i`（注释："段内位置从 0 起"）；模型是绝对位置查表
   （`gpt2_model_builder.cpp` 的 `addGather(wpe, position_ids)`）。第二块 chunk 必须是
   `prompt_done + i`，而 spec §4 / D16 / 交接 prompt 都没提这一条 → 照文档实现会静默算错，AC9 必红。
   **已修**：`design.md` D16 与 spec §2/§4 补规则，spec §6 增用例 `ChunkedPositionsAreAbsolute`。
2. **"统一成分页因果"在实现层是空壳。** generation 段的 split-K 是 decode 专用（每行 1 个 query、
   workspace 布局写死 `[split][batch][head][m,l,acc]`、`has_current_token` 单 token），
   context kernel 又完全不接 `block_tables`/`context_lens` 且越界时**静默 return 不写输出**。
   "统一成同族、只是 query 数 > 1"于是没有可实现的落点。
   **已修**：spec §3 写清 chunked context kernel 的形态（网格同形、K/V 两段、score 留 shared、
   `getWorkspaceSize` 不变、禁止静默跳过），D16 给出 A/B/C 三个方案与选定理由。
3. **采样行集的表达没有落点。** 采样器只吃连续 `[count]`（`llm_runner.hpp` 的 `SampleBatch` 契约
   与 `d_step_tokens_` 注释），而"本步完成 prefill 的行"在活跃表里可能被未完成的行隔开；
   重排活跃表会破不变量 4。
   **已修**：spec §2/§4 定"显式行列表 + 紧凑暂存"，不动采样器签名与行号纪律；spec §6 增用例
   `ChunkedSamplingRowSetIsCompacted`。

### P1 Risks

1. **`graph_version` 的"不需要 bump"没有依据，且与项目自己的规则 / 先例冲突。**
   `engine_cache.hpp` 写的是"任何改动建图 / 精度 / **插件行为**的代码变更都要 +1"；
   `builder.cpp` 记着 4 = packed 图、以及 1 → 2 正是因 `PagedAttentionPlugin::getWorkspaceSize`
   从 0 变正数。技术上新引擎可以复用（kernel 由运行期插件解析加载）**当且仅当** I/O 与
   `getWorkspaceSize` 都不变 —— 但那是"有条件的豁免"，原稿只写了结论。
   **处置**：作者 2026-10-04 复核后**确定** `kPackedPrefillGraphVersion` 4 → 5（见本节末的
   "作者确认记录"；不再保留"沿用 4 + 写豁免条件"的分支）。
2. **"fused kernel 支持的常量集合"在代码里不存在。** 可当检查对象的只有
   `kPackedAttentionMaxContextSeqLen = 1024`、`configurePlugin` 对 `max_seq_len` 的校验、
   `kPagedAttentionMaxSplits = 8`、`kMaxHeadSize = 1024`；`chunk_limit` 是**数据**不是模板常量。
   **已修**：改成 `n_positions <= 1024` ＋ `prompt_len <= n_positions` 两条。
3. **写回位置与记账的口径没落到设计。** 现有 kernel 从 0 覆盖写（`paged_kv_cache_kernels.cu` 的
   packed 写回用 `t / block_size`、`t % block_size`），host 侧是赋值
   （`paged_kv_cache.cpp:323-324`）。S5 必须同时改 kernel 与记账。
   **已修**：spec §4 增"cache 记账（累加）"行，并在 §7 增对应风险行。

### P2 Quality

| 项 | 说明 |
|---|---|
| `prompt_done` 与 `PagedKVCache::Sequence::length` 双源 | spec §2 已注明优先单源（S4 的 `host_context_lens` 就是取 `SequenceLength`），避免第二份拷贝漂移 |
| `design.md` 的 `## Requirement Coverage` 漏登 Included 7/8 与 AC7/8/9 | 本轮补齐（5 行），`## 验证策略` 同步补 AC7/AC8/AC9 行 |
| 现有 S5 复评表一写"workspace 拿不到时显式拒绝、不再沿用静默降级"，但 packed 插件此刻仍会静默退单趟 | 该纪律针对"配置不可用"，运行期 workspace 缺失是另一类；需要作者写清边界（本轮只在 P1-1 里记下） |
| 本节的表一是"增量"口径，而非技能 Exit Gate 要求的具体条目数 | 沿用 S3/S4 复评的既有先例；若作者要严格口径，可另出一版全量表（52 项） |

### Decision（第二遍复评）

- **P0：3 条**（position_ids 落点、chunked context kernel 形态、采样行集表达）→ **BLOCK**，返回 P2 补设计。
  本轮已把这 3 条的落点写进 `design.md`（D15/D16 + 验证策略 + Requirement Coverage）与
  `p5_s5_interface_spec.md`（§2/§3/§4/§5/§6/§7）；作者同日给出第三轮确认（见本节末），
  重开 Gate-A 只差"确认这轮修订文档"这一步。
- **P1：3 条**（`graph_version` 裁决、常量集合口径、写回记账口径）。三条都已收口 ——
  第 1 条由作者 2026-10-04 确认为 **bump 4 → 5**，另两条落成文档（见本节末的"作者确认记录"）。
- **术语表**：`分块` / `末块` / `chunk 的绝对位置` 已按作者口径**指针式登记**进 `analysis.md`
  （表里只放指针与判伪口径，定义以 spec §2 为唯一来源）。
- **与上一节的关系**：上一节的"P0：无 / PASS"**被本节改判**（原因见开头"触发与范围"）。
  本节的结论不追溯覆盖 2026-10-03 的整体 Gate-A 记录，只针对 S5 这一增量。

### 作者确认记录（2026-10-04 第二轮）

| # | 事项 | 作者决定 | 落到哪 |
|---|---|---|---|
| 1 | `graph_version`（本节 P1-1） | **确定 bump 4 → 5**（不再保留"沿用 4 + 写豁免条件"的分支） | `design.md` D16；`p5_s5_interface_spec.md` §4/§5 |
| 2 | "分块"的术语口径（本节表三） | **指针式登记**：`analysis.md` 的 Terminology 里登记条目，定义指向 spec §2 作唯一来源，不复制正文 | `analysis.md`；`p5_s5_interface_spec.md` §2 |
| 3 | 第二遍复评查出的三条缺口 | **全部折进设计**（见下两行的具体形态） | 见下 |
| 4 | 缺口之一：`chunk_limit` 的来源 | 给 **`Engine` 加只读 profile 查询**并由 runner 在构造期推导，**不新增 `LLMRunner::Config` 字段**；查询失败即构造期报错（依据"按对方查询、不按配置假定"）。**具体机制由本轮折入时选定，作者可否决** | `design.md` D16；`p5_s5_interface_spec.md` §2/§4/§5 |
| 5 | 缺口之二 / 之三：入口拒绝的信息与兜底纪律的边界 | 配置 / 形状类**显式拒绝**（错误信息带实际值与上界）；运行期资源类（TRT 未给 workspace）保留既有降级但只降速、打 WARN，且 **S5 新增路径不得引入新的静默降级** | `design.md` D16；`p5_s5_interface_spec.md` §3 |

**收口状态**：本节的 3 条 P0 与 3 条 P1 均已落到文档（P0 落在 design/spec，P1 的
`graph_version` 与常量口径、记账口径同）；P0-1 的实现（`Engine` 的 profile 查询接口）与
`graph_version` 常量本身属**代码**，按 §0.7 留到 S5-1 开工时再动。
Gate-A 的重开等作者确认这轮修订文档。

## Decision

PASS（无 P0；P1 已由作者于 2026-10-03 确认，Gate-A 通过）

**范围注**：上面这行是 2026-10-03 对"本 feature 整体设计"的 Gate-A 记录，保持不动。
**S5 这一增量的 Gate-A 在 2026-10-04 的第二遍复评里被改判为 BLOCK**（见上一节：P0 三条已补落点，
P1 三条已收口 —— `graph_version` 由作者确认为 bump 4 → 5，术语按指针式登记，另两条落成文档）。
作者确认这轮修订文档后，在这里记一次复评结论、Gate-A 重开。
