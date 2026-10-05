# Review Report

<!--
阶段：P3-Review（2026-10-06，乙框架重设计后的复评）。
输入：`design.md` + `requirement.md` + 四份 checklists + 三份里程碑接口细化。
口径：对四份 checklists 的每个 [P0] / [P1] 项逐条回答；不适用必须写理由。
-->

## Summary

**设计主体成立，但有一条 P0**：乙框架把性能门收窄到只挡 S3 之后，S0 / S1 / S2 的设计是完整的
（接口、失败语义、判据来源齐全，且三条判据来源都已核实存在）。问题全部集中在 **S2**：
"让自定义算子在外部图上被执行"这条落点依赖两个**当前系统里不成立**的前提——
解析器拿不到注册表、creator 的命名空间为空串。

**S3 不计入本轮**：它是被作者明确接受的"未满足（阻塞）"项，已登记在 `STATE.md` 的 `## 判据对照`，
按技能全局规则 10 不重复计入。

**统计**：checklist 结论表 52 行（P0 35 + P1 17）；需求落点表 12 行；术语定义表 12 行。

## Checklist Result

> 出处：`.agents/skills/trt-inference-engineering/checklists/` 的 `cpp.md` / `cuda.md` /
> `tensorrt.md` / `llm_runtime.md`。**行数 = 四份清单里 [P0] + [P1] 项总数 = 52**（P0 35 / P1 17）。
> `不适用` 一律附理由。

### 表一（cpp.md，12 行）

| Item | Level | Result | 证据 / 理由 | Action |
|---|---|---|---|---|
| cpp / Resource Management / RAII 管理资源生命周期 | P0 | 通过 | S2 适配器只查表、不持有 creator 生命周期（`s2_custom_op_interface_spec.md` §1）；设计未新增需手工释放的资源 | 记录 |
| cpp / Resource Management / Ownership 是否明确 | P0 | 通过 | 同上一行；creator 指针来源与既有 `PluginRegistry` 约定一致 | 记录 |
| cpp / Resource Management / 悬空引用风险 | P0 | 通过 | 返回静态实例地址；`OnnxIoContract` 用字符串字面量，无生命周期问题 | 记录 |
| cpp / Resource Management / 异常路径是否释放资源 | P1 | 通过 | 重写脚本失败即非零退出且**不产出半成品图**（`s0_contract_interface_spec.md` §1/§4） | 记录 |
| cpp / Concurrency / 多线程数据竞争 | P0 | 不适用 | 本轮不新增线程；适配器只在解析期单线程使用 | — |
| cpp / Concurrency / Mutex / Lock 生命周期 | P0 | 不适用 | 同上：本轮不引入锁 | — |
| cpp / Concurrency / 不必要锁竞争 | P1 | 不适用 | 同上 | — |
| cpp / Interface Design / API 输入输出是否明确 | P0 | 通过 | 三份 spec 给出签名、dtype、形状与返回语义 | 记录 |
| cpp / Interface Design / 错误处理方式是否统一 | P0 | 通过 | 沿用 `MINI_TRT_LOG_ERROR` + 非零退出；三份 spec 都有失败语义表 | 记录 |
| cpp / Interface Design / 接口是否容易扩展 | P1 | 待确认 | `BuildFromOnnx` 的签名变更留给 S3（`design.md` D0 之后的待定项），与 REQ-016 的写冲突已登记 | 人工确认 |
| cpp / Build / Debug / Release 是否均可编译 | P0 | 不适用 | 设计阶段无代码改动；由 P5 的 Exit Gate 负责 | — |
| cpp / Build / 是否引入额外依赖 | P1 | 通过 | 只用既有 `onnx` 依赖（`requirements.txt` 已声明） | 记录 |

### 表一（cuda.md，14 行）

| Item | Level | Result | 证据 / 理由 | Action |
|---|---|---|---|---|
| cuda / Kernel Correctness / 边界条件 | P0 | 不适用 | **本轮不新增、也不修改任何 kernel** | — |
| cuda / Kernel Correctness / 越界访问 | P0 | 不适用 | 同上 | — |
| cuda / Kernel Correctness / Synchronization | P0 | 不适用 | 同上 | — |
| cuda / Kernel Correctness / 不同 shape 覆盖 | P1 | 不适用 | 不改 kernel；dtype / 形状覆盖由 S0 / S2 的引擎用例承担 | 记录 |
| cuda / Memory / Global Memory 访问安全 | P0 | 不适用 | 不新增 kernel | — |
| cuda / Memory / race condition | P0 | 不适用 | 同上 | — |
| cuda / Memory / Memory coalescing | P1 | 不适用 | 同上 | — |
| cuda / Memory / Shared Memory bank conflict | P1 | 不适用 | 同上 | — |
| cuda / Performance / 优化是否有 benchmark 证明 | P0 | 通过 | 本条**未声称任何性能提升**：唯一性能主张（S3 替换）被 PF-7 卡住，`design.md` 已记为待定 | 记录 |
| cuda / Performance / compute / memory bound 分析 | P1 | 不适用 | 不新增 kernel；端到端口径见 `design.md` 的 Performance Consideration | 记录 |
| cuda / Performance / occupancy | P1 | 不适用 | 同上 | — |
| cuda / CUDA Runtime / Stream 生命周期 | P0 | 不适用 | 未改 stream 使用；S2 的插件调用仍由 TRT 的 enqueue 传入 stream | 记录 |
| cuda / CUDA Runtime / Event 同步 | P0 | 不适用 | 未改 | — |
| cuda / CUDA Runtime / Async API 是否正确使用 | P1 | 不适用 | 本轮无异步 API 改动（图重写是离线 Python） | — |

### 表一（tensorrt.md，13 行）

| Item | Level | Result | 证据 / 理由 | Action |
|---|---|---|---|---|
| tensorrt / Engine Lifecycle / ICudaEngine 生命周期 | P0 | 通过 | 沿用既有 Builder / Engine 路径；判据含 `tools/inspect_engine.cpp` 的反序列化探针 | 记录 |
| tensorrt / Engine Lifecycle / IExecutionContext 生命周期 | P0 | 通过 | 未改 | 记录 |
| tensorrt / Engine Lifecycle / Runtime / Engine / Context ownership | P0 | 通过 | 未改 | 记录 |
| tensorrt / Tensor / Tensor shape 是否明确 | P0 | 通过 | `s0_contract_interface_spec.md` §1 给出 `[batch, seq]` + dtype | 记录 |
| tensorrt / Tensor / Dynamic shape profile 是否覆盖输入范围 | P0 | **通过（已核实）** | 非 packed 路径按 dim 下标套范围 → 新增的 `position_ids[B,S]` **自动获得** batch / seq 两组范围（`builder.cpp` 的 `AddLlmOptimizationProfiles`） | 记录 |
| tensorrt / Tensor / Tensor dtype 转换是否正确 | P1 | 待确认 | R1 规则已定，但"图内 INT64 消费点能否穷尽"需资产探针（`s0_contract_interface_spec.md` §7 第 3 条） | 人工确认 |
| tensorrt / Plugin / Plugin creator 注册是否正确 | P0 | **不通过**（设计侧已裁决，代码未落） | 见 P0-1 / P0-2 与「作者裁决」：解析器未挂注册表；四个 creator 的 `getPluginNamespace()` 返回空串，修法已定 | 修复（落码在 S2 / P5） |
| tensorrt / Plugin / Plugin serialize / deserialize 是否完整 | P0 | 待确认 | 既有插件已具备；**改 namespace 对既有引擎反序列化的影响未核实**（`s2_..._spec.md` §3 第 3 条） | 人工确认 |
| tensorrt / Plugin / Plugin enqueue 中的 stream | P0 | 不适用 | 未改 enqueue 实现 | — |
| tensorrt / Plugin / Plugin workspace 管理 | P1 | 不适用 | 未改 `getWorkspaceSize` | — |
| tensorrt / Execution / Binding index 是否正确 | P0 | 通过 | S0 后为两输入两绑定，由改造后的对拍用例覆盖 | 记录 |
| tensorrt / Execution / CUDA stream 是否传递正确 | P0 | 不适用 | 未改 | — |
| tensorrt / Execution / Async execution 是否正确 | P1 | 不适用 | 未改；PF-7 的测量协议单列在 `design.md` | — |

### 表一（llm_runtime.md，13 行）

| Item | Level | Result | 证据 / 理由 | Action |
|---|---|---|---|---|
| llm_runtime / Request Scheduling / Request 生命周期 | P0 | 不适用 | 本轮不改 runner；与 REQ-016 的写冲突点已登记 | — |
| llm_runtime / Request Scheduling / Prefill / Decode 区分 | P0 | 待确认 | 外部图路径目前**只有 prefill**（无 K/V 输入）；decode 图是 S3 与 REQ-017 的耦合项，S3 待定 | 记录 |
| llm_runtime / Request Scheduling / Batch 状态一致 | P0 | 不适用 | 不改 runner | — |
| llm_runtime / KV Cache / KV Cache ownership | P0 | 不适用 | 外部图路径不接 K/V（S3 待定） | — |
| llm_runtime / KV Cache / Block 管理 | P0 | 不适用 | 不改 kv_cache | — |
| llm_runtime / KV Cache / Cache eviction 策略 | P0 | 不适用 | 同上 | — |
| llm_runtime / KV Cache / Memory fragmentation | P1 | 不适用 | 同上 | — |
| llm_runtime / KV Cache / Long context 是否测试 | P1 | 不适用 | 不改 attention / context 路径 | — |
| llm_runtime / Continuous Batching / Dynamic request 加入退出 | P0 | 不适用 | 不改 scheduler | — |
| llm_runtime / Continuous Batching / Scheduler 状态一致 | P0 | 不适用 | 同上 | — |
| llm_runtime / Continuous Batching / Batch 调度策略 | P1 | 不适用 | 同上 | — |
| llm_runtime / Sampling / Sampling 结果是否正确 | P0 | 不适用 | 不改采样器 | — |
| llm_runtime / Sampling / CUDA kernel 是否验证 | P1 | 不适用 | 不新增 kernel | — |

## Terminology Check

> 出处：`requirement.md` 逐段提取的模糊名词，**行数 = 12**。定义为
> `analysis.md` 的 `## Terminology` 条目。

### 表三（术语定义表，12 行）

| 模糊名词 | 定义所在 | 结论 |
|---|---|---|
| 外部图（ONNX）路径 | `analysis.md`「外部图路径（方案 B）」 | 已定义（含可判据：单一 INT64 输入 + 只挂 prefill） |
| 原生路径 / 原生实现 | `analysis.md`「原生路径（方案 A）」 | 已定义（输入为 INT32 双输入） |
| 自研算子 | `analysis.md`「自定义算子 / 自定义域算子」 | 已定义（本 feature 里两者同指 `src/plugins` 的实现） |
| 计数级识别 | `analysis.md`「计数级识别」 | 已定义 |
| 拓扑级识别 | `analysis.md`「拓扑级识别」 | 已定义（含"能拒绝计数相同连接不同"的可判据） |
| 子图替换 | `analysis.md`「子图替换」 | 已定义（含"边界张量名不变"与四项引擎可用性） |
| 自定义算子的导出方式 | `analysis.md`「自定义算子 / 自定义域算子」 | 已定义（判据 = 解析期由插件注册表找到创建器） |
| 可复现对照 / PF-7 | `analysis.md`「PF-7 / 可复现对照」 | 已定义（含次数与报数口径） |
| 中位数与极差 | `analysis.md`「PF-7 / 可复现对照」+「可判 / 未定」 | 已定义 |
| 契约统一 / 同一套调用方式 | `analysis.md`「契约统一（同一套调用方式）」 | 已定义（判据 = 同一份输入绑定两条路） |
| 替换前后数值一致 / 有出处的判据 | `analysis.md`「有出处的判据」 | 已定义（含"不得跨精度复用"） |
| 引擎可用性（构建 / 序列化 / 反序列化 / 推理） | `requirement.md` 原文自述四步 + `analysis.md`「替换生效」 | 已定义（四步为 requirement 自述；"确实执行"由「替换生效」补） |

## Requirement Coverage Result

### 表二（需求落点表，12 行）

| 需求条目 | 设计落点 | 结论 |
|---|---|---|
| Included：前置测量 | `design.md` Performance Consideration（PF-7 口径与工具缺口） | 已落点（执行待定，作者已接受） |
| Included：拓扑级识别 | `s1_topology_interface_spec.md` §1 / §2 + `design.md` Interfaces | 已落点 |
| Included：子图替换的路线选型与落地 | `design.md` Trade-off D2 / D3 / D6 + Data Structure | 已落点（S3 执行待定；接口细化按约定暂缺） |
| Included：自定义算子的导出方式 | `design.md` D4 + `s2_custom_op_interface_spec.md` §1 | 已落点（但依赖 P0-1 / P0-2） |
| Included：一致性判据 + 引擎可用性 | `design.md` 判据与来源 + `s2_..._spec.md` §5 | 已落点 |
| Included：两条路输入契约统一 | `design.md` D5 + `s0_contract_interface_spec.md` | 已落点 |
| AC 1：前置有依据 | `design.md` Overview（PF-7 判据口径） | 已落点（执行待定） |
| AC 2：识别到拓扑级 | `s1_..._spec.md` §1（T1–T4）+ §4 反例夹具 | 已落点 |
| AC 3：替换可跑 | `s2_..._spec.md` §5 生效判据 | 已落点 |
| AC 4：数值一致 | `design.md` 判据与来源 + 两份 spec 的阈值出处 | 已落点 |
| AC 5：契约统一 | `design.md` D5 + `s0_..._spec.md` §1 | 已落点 |
| AC 6：不回归 | `design.md` 判据与来源（回归行）+ 三份 spec 的测试方式 | 已落点 |

## 来源核对（"已定 / 必须拒绝"逐条走一遍）

> 规则（`phases/p3_review.md` 的 Entry）：每条"已定 / 必须拒绝"要回答
> "兑现它需要什么输入或接口 → 系统里到底有没有"；**找不到来源的记 P0**。

| 已定条目 | 兑现它需要什么 | 系统里有没有 | 结论 |
|---|---|---|---|
| D0 乙框架 | `requirement.md` 的 Goal / Excluded 原文 | 有 | 通过 |
| D2 Python 侧图重写 | `onnx` 包 + 造图 / 探针工具 | 有（`make_tiny_onnx.py` / `inspect_onnx.py` / `requirements.txt`） | 通过 |
| D3 只替换注意力 | "GPT-2 无 RoPE / RMSNorm"的事实 | 有（`analysis.md` 已知问题 7 + `BASELINE.absent_ops`） | 通过 |
| D5 契约统一改外部图 | 图重写能力 | S0 交付（新工具）；**真实图形态未核实** | P1-1 |
| R1 dtype 统一 | 图内 INT64 消费点清单 | **查不到**（本机无资产） | P1-1 |
| R2 `position_ids` 提升 | 位置编码来源形态（A 常量 / B `Range`） | **查不到**（同上） | P1-1 |
| S1 的 T1–T4 | 真实图的块边界形态 | **查不到**（同上） | P1-2 |
| S2 解析器接注册表 | `nvonnxparser::createParser` 带 `IPluginRegistry` 的重载 | **查不到**（本机无 TRT 头；全仓检索无既有证据） | **P0-1** |
| S2 自定义算子能被查到 | creator 的 namespace 非空且与节点域匹配 | **不存在**（已核实：`getPluginNamespace()` 返回空串） | **P0-2 → 已裁决（2026-10-06）**，修法见 §3.1 |
| S2 属性 → `PluginField` 映射 | `eps`(float32) / `hidden_size`(int32) 的类型契约 | 有（creator 的 `getFieldNames()` 声明） | 通过（映射细节待真机确认） |
| 数值一致判据 | 同精度阈值出处 | 有（`tests/test_gpt2_onnx.cpp`） | 通过 |
| 替换生效判据 | ONELINE 读取器 + `detailed_profiling` | 有（`tests/engine_layer_info_support.hpp` / `builder.hpp`） | 通过 |
| PF-7 判据（可判 / 未定） | 能跑 PF-7 的真机 + 用例 | **用例不存在、真机不可用** | 已按作者裁决记"未满足（阻塞）"，不计本轮 P0 |

## P0 Blockers

### P0-1 解析器接注册表的接口**未核实**

`design.md` 与 `s2_custom_op_interface_spec.md` 把"自定义算子进图"落在
`nvonnxparser::createParser(*network, *logger_, *plugin_registry)` 上。**这个重载是否存在，
本次查不到**：本机无 TensorRT 头文件，全仓检索也没有既有证据。

**修复落点**：真机读 `NvOnnxParser.h` 的函数声明（分钟级）。若不存在 → **回 P2 选路**
（备选是自定义 `IPluginFactory`，当前**未设计**）。`s2_..._spec.md` §3 第 1 条已登记这条判据与后果。

**2026-10-06 裁决：本阶段不真机** → 该核实**推迟**，P0-1 保持在"未闭环"。这不是设计缺陷，
而是环境安排；它同时也是 S2 不得落码的原因。

### P0-2 creator 的命名空间与自定义域不匹配（**已核实的代码事实**）

四个 creator 的 `namespace_` 从未赋值，`getPluginNamespace()` 返回**空串**
（`src/plugins/rmsnorm_plugin.cu` 的构造函数 + `iplugin_v3_base.cpp` 的
`plugin_namespace_ = pluginNamespace ? pluginNamespace : ""`）。而 ONNX 的自定义域**必须非空**
（空域等于 `ai.onnx`）——也就是说，按当前实现，**图上不可能出现一个能被查到的自定义算子**，
该落点在系统里不成立。

**修复落点**：`s2_..._spec.md` §1 的域名常量 + §2 的 creator namespace 改动。
但这条修复有一个**未吸收的连带影响**：既有已序列化引擎里若记录了插件命名空间，
改 namespace 会让它们反序列化失败 → 需要连带重建。该影响**未核实**，且属于扩大影响面，
必须由作者单独裁决（`s2_..._spec.md` §7 第 3 条）。

**2026-10-06 裁决：接受。** 作者裁定 creator 的 namespace 设为 `kMiniTrtLlmPluginNamespace`，
并接受由此带来的连带重建。因此这条 P0 的**设计与授权都已闭合**，剩下的只是落码（S2 / P5），
重建范围待 §3 第 3 条核实后写死。

## P1 Risks

| # | 风险 | 处置 |
|---|---|---|
| P1-1 | S0 的两条图形态未核实（A 常量 / B `Range`；INT64 消费点清单） | 落码前置：真机跑 `inspect_onnx.py` 探针；两条形态都不成立时停止并上报 |
| P1-2 | S1 的真实块边界未核实 | 落码前置：同上；形态与骨架不符时先改 spec 再改代码 |
| P1-3 | 改 creator namespace 可能让既有引擎失效 | 需作者裁决重建范围（与 P0-2 同源） |
| P1-4 | S3 / PF-7 阻塞 | 已按作者裁决记"未满足（阻塞）"；重开条件在 `design.md` 待定项 |
| P1-5 | S0 的识别基线归属（`BASELINE.inputs` 写死单输入，S0 后必红） | **已裁决（作者 2026-10-06）= B**：`--check` 只对源图生效，重写图由重写器自检与 S0 的 host 用例覆盖 |
| P1-6 | `BuildFromOnnx` 签名变更会让 S3 与 REQ-016 撞同一批调用方 | 已登记写冲突；开工前需指定串行顺序 |
| P1-7 | 外部图路径只有 prefill，没有 decode 图与 K/V 输入 | 与 REQ-017 的耦合项；S3 若做需一并补齐（`design.md` 耦合与写冲突节） |

## P2 Quality

- 三份 spec 的"待作者确认"小节里各有 2–3 条选择项（夹具算子选型、反例夹具落点、核实窗口安排）。
- `docs/future_iterations_development_plan.md` 第 1278 行那处"无新代码"表述**已于 2026-10-06 更正**
  （作者点名；与 `future_iterations_test_plan.md` 的 PF-7 行同批）。
- 本文的 P0-1 / P1-* 与各 spec 的未决项已汇总登记在 `STATE.md` 的 `## 待定与待决清单`（T1–T12），
  本文件只保留"评审结论与证据"，不复述那份清单。

## Decision

**BLOCK**（P0 剩 1 条：P0-1）。

按 `phases/p3_review.md` 的 Human Gate：**P0 存在 → 暂停，返回 P2**。
经 2026-10-06 作者裁决后，两条 P0 的状态是：

| P0 | 状态 | 何时能清 |
|---|---|---|
| P0-1（`createParser` 重载是否存在） | **未闭环**（本阶段不真机，无法核实） | 下次真机窗口，读一次 `NvOnnxParser.h` |
| P0-2（creator namespace 与自定义域不匹配） | **已闭合**（裁决 = 设为 `mini_trt_llm` + 接受连带重建） | 落码在 S2，重建范围待核实后写死 |

**S0 / S1 的设计本身不需要返工**：它们的落点、判据与来源都已核实，剩下的只是两处落码前置探针（P1-1 / P1-2）。
因此本条的**唯一放行障碍是 P0-1**，而它受"本阶段不真机"约束。

## 作者裁决（2026-10-06）

| # | 裁决 | 影响 |
|---|---|---|
| 1 | **暂时不真机** | P0-1 保持未闭环；S2 不得落码；S0 的引擎侧验证同样推迟（见 `STATE.md` 的 Next Action） |
| 2 | **接受**（P0-2 的修法与连带重建） | creator namespace 定为 `mini_trt_llm`；既有引擎若因此失效则连带重建，范围待核实后写死 |
| 3 | **B**（S0 的基线归属） | `inspect_onnx.py --check` 只对源图生效；重写图由重写器自检与 S0 的 host 用例覆盖 |
