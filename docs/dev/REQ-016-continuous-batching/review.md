# Review Report

<!--
本版 2026-10-03 重做，对应重做后的 design.md。
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
| C++-接口是否容易扩展 | P1 | 待确认 | S3 的准入/退出接口形态取决于 D10 |
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
| LLM-Dynamic request 加入/退出是否安全 | P0 | 待确认 | 设计已给状态机与四步流程；安全性需 P6 用例固定（含异常路径归还块） |
| LLM-Scheduler 状态是否一致 | P0 | 待确认 | 同上一行；S3 未实现前无法判"通过" |
| LLM-Batch 调度策略是否合理 | P1 | 待确认 | D10：两步式是否吃掉收益要用 P4 数据判 |
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

## Decision

PASS（无 P0；P1 已由作者于 2026-10-03 确认，Gate-A 通过）