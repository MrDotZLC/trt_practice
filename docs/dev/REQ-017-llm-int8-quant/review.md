# Review Report

## Summary

- **评审对象**：2026-10-05 重做后的 `requirement.md` / `analysis.md` / `design.md`
  （路线 C：Python 离线量化产出 int8 权重 + 清单，C++ 原生建图只读常量并挂 DQ）。
- **评审输入**：四份 checklists（cpp / cuda / tensorrt / llm_runtime）、`requirement.md`、
  `design.md`、`analysis.md` 的 Terminology。
- **本轮 S1 的改动面**：新增 Python 量化脚本与两份产物；权重加载器多开一条 int8 零拷贝；
  语言模型构建器按清单建 int8 常量 + DQ；`graph_version` +1 与指纹加项。**不新增 / 不修改任何
  kernel、插件与 cache 逻辑，也不改运行时契约**（D3 + Module Design）。
- **结论**：**P0 Blocker = 0**；P1 Risk = 7（送 Gate-A 裁决）；Decision = **PASS**。

### N/A 与例外条款（表一适用）

以下条款只说明"某些 checklist 项不适用于本轮"，每一条都点名**不得用于跳过哪些判据**
（依据 `SKILL.md` 的 Mandatory #9）：

- **条款①（kernel）**：S1 不新增 / 不修改任何 CUDA kernel，不改 kernel 的边界与形状处理。
  **不得用于跳过**：AC2（自证低精度）、AC5（不回归）、D7（引擎体积判据）。
- **条款②（plugin）**：S1 不新增 / 不修改任何 TensorRT plugin（creator / serialize / enqueue /
  workspace 都不动）。**不得用于跳过**：AC2、AC5、D7。
- **条款③（调度与 cache）**：S1 不改请求调度、连续批与 cache 块管理 / 淘汰策略（那些是
  `REQ-016` 的契约面）。**不得用于跳过**：AC1（端到端可跑）、AC5。
- **条款④（编译）**：本评审发生在沙箱内，无编译器 / 无 GPU，编译与运行不可执行。
  **不得用于跳过**：表一的 `CPP-P0-8`（Debug/Release 编译）与 P5 的 Exit Gate。
- **条款⑤（并发形态）**：S1 不引入多线程与锁（脚本与构建期都是单线程）。
  **不得用于跳过**：AC1、AC5。

## Checklist Result（表一）

行数 = 四份 checklists 的 [P0] / [P1] 项总数 =（cpp 8 + 4）+（cuda 8 + 6）+（tensorrt 10 + 3）
+（llm_runtime 9 + 4）= **52**。

| Item | Level | Result | Action |
|---|---|---|---|
| CPP-P0-1 RAII 是否用于管理资源生命周期 | P0 | 通过 | 记录 —— 落点 Resource Lifecycle：int8 权重与清单只在构建期存在，随引擎 / 加载器析构释放；运行时零新增资源 |
| CPP-P0-2 Ownership 是否明确 | P0 | 通过 | 记录 —— Module Design 已写"清单与 int8 权重必须与权重加载器同生命周期" |
| CPP-P0-3 是否存在悬空引用风险 | P0 | 通过 | 记录 —— 常量为 mmap / 转换缓存指针，沿用现有零拷贝语义；无跨生命周期引用 |
| CPP-P1-1 异常路径是否释放资源（Exception 路径） | P1 | 通过 | 记录 —— 构建失败沿用 `EngineBuilder` 的 `unique_ptr` 链；脚本侧失败即非零退出、不落半成品 |
| CPP-P0-4 多线程访问是否存在数据竞争 | P0 | 不适用（条款⑤） | 记录 |
| CPP-P0-5 Mutex / Lock 生命周期是否正确 | P0 | 不适用（条款⑤） | 记录 |
| CPP-P1-2 是否存在不必要锁竞争 | P1 | 不适用（条款⑤） | 记录 |
| CPP-P0-6 API 输入输出是否明确 | P0 | 通过 | 记录 —— Module Design + Data Structure 的契约表 |
| CPP-P0-7 错误处理方式是否统一 | P0 | 通过 | 记录 —— 缺项 / 清单越界一律响亮失败，沿用 `MINI_TRT_LOG_ERROR`，无静默回落 |
| CPP-P1-3 接口是否容易扩展 | P1 | 通过 | 记录 —— 清单已有 `granularity` / `axis` 字段为 per-channel 留位 |
| CPP-P0-8 Debug / Release 是否均可编译 | P0 | 待确认 | 人工确认 —— 沙箱无编译器（条款④）；绑定 P5 真机窗口的第一步 |
| CPP-P1-4 是否引入额外依赖 | P1 | 通过 | 记录 —— C++ 侧零新依赖；Python 侧只用既有的 safetensors / numpy 类工具链 |
| CUDA-P0-1 Kernel 边界条件是否正确 | P0 | 不适用（条款①） | 记录 |
| CUDA-P0-2 是否存在越界访问 | P0 | 不适用（条款①） | 记录 |
| CUDA-P0-3 Synchronization 是否正确 | P0 | 不适用（条款①） | 记录 |
| CUDA-P1-1 不同 shape 是否覆盖测试 | P1 | 不适用（条款①） | 记录 —— S1 不改 kernel 的形状处理；既有用例的回归由 AC5 覆盖 |
| CUDA-P0-4 Global Memory 访问是否安全 | P0 | 不适用（条款①） | 记录 |
| CUDA-P0-5 是否存在 race condition | P0 | 不适用（条款①） | 记录 |
| CUDA-P1-2 Memory coalescing 是否优化 | P1 | 不适用（条款①） | 记录 |
| CUDA-P1-3 Shared Memory 是否存在 bank conflict | P1 | 不适用（条款①） | 记录 |
| CUDA-P0-6 优化是否有 benchmark 证明 | P0 | 待确认 | 人工确认 —— 本轮不出性能结论；验证时点在 P7（当前无 GPU） |
| CUDA-P1-4 是否分析 compute bound / memory bound | P1 | 通过 | 记录 —— Performance Consideration 已给分解（固定项 vs 上下文斜率）与 179 GB/s 推导，并标注为**推导**；实测归 P4 / P7 |
| CUDA-P1-5 是否考虑 occupancy | P1 | 不适用（条款①） | 记录 |
| CUDA-P0-7 Stream 生命周期是否正确 | P0 | 不适用（条款①） | 记录 —— 构建期不涉及 stream；运行时路径不变 |
| CUDA-P0-8 Event 同步是否正确 | P0 | 不适用（条款①） | 记录 |
| CUDA-P1-6 Async API 是否正确使用 | P1 | 不适用（条款①） | 记录 |
| TRT-P0-1 ICudaEngine 生命周期是否明确 | P0 | 通过 | 记录 —— 不变；D5 只改指纹与 `graph_version` |
| TRT-P0-2 IExecutionContext 生命周期是否明确 | P0 | 通过 | 记录 —— 不变 |
| TRT-P0-3 Runtime / Engine / Context ownership 是否正确 | P0 | 通过 | 记录 —— 不变 |
| TRT-P0-4 Tensor shape 是否明确 | P0 | 通过 | 记录 —— 量化不改任何张量形状；`[1, in, out]` 的 rank-3 声明已在 `analysis.md` 记录 |
| TRT-P0-5 Dynamic shape profile 是否覆盖输入范围 | P0 | 通过 | 记录 —— D5 明确不动 profile 区间 |
| TRT-P1-1 Tensor dtype 转换是否正确 | P1 | 待确认 | 人工确认 —— **这就是 D7**（int8 常量 + DQ 的 dtype 路径与是否被融合）；P5 第一步用最小图 + 引擎体积判据 |
| TRT-P0-6 Plugin creator 注册是否正确 | P0 | 不适用（条款②） | 记录 |
| TRT-P0-7 Plugin serialize / deserialize 是否完整 | P0 | 不适用（条款②） | 记录 |
| TRT-P0-8 Plugin enqueue 中的 stream 是否正确 | P0 | 不适用（条款②） | 记录 |
| TRT-P1-2 Plugin workspace 管理是否合理 | P1 | 不适用（条款②） | 记录 |
| TRT-P0-9 Binding index 是否正确 | P0 | 通过 | 记录 —— I/O 集合与绑定顺序不变 |
| TRT-P0-10 CUDA stream 是否传递正确 | P0 | 通过 | 记录 —— 不变 |
| TRT-P1-3 Async execution 是否正确 | P1 | 通过 | 记录 —— 不变 |
| LLM-P0-1 Request 生命周期是否明确 | P0 | 通过 | 记录 —— 不变 |
| LLM-P0-2 Prefill / Decode 流程是否区分 | P0 | 通过 | 记录 —— Module Design 已写"两张图必须用同一份清单"，避免两侧数值口径分叉 |
| LLM-P0-3 Batch 状态是否一致 | P0 | 通过 | 记录 —— 不变（REQ-016 契约） |
| LLM-P0-4 KV Cache ownership 是否明确 | P0 | 通过 | 记录 —— 不变 |
| LLM-P0-5 Block 管理是否正确 | P0 | 通过 | 记录 —— 不变 |
| LLM-P0-6 Cache eviction 策略是否明确 | P0 | 不适用（条款③） | 记录 |
| LLM-P1-1 Memory fragmentation 是否考虑 | P1 | 不适用（条款③） | 记录 |
| LLM-P1-2 Long context 是否测试 | P1 | 待确认 | 人工确认 —— 数值对照的上下文覆盖范围未定（见 P1-6）；建议至少短 + 长两档 |
| LLM-P0-7 Dynamic request 加入 / 退出是否安全 | P0 | 通过 | 记录 —— 不变（REQ-016 契约） |
| LLM-P0-8 Scheduler 状态是否一致 | P0 | 通过 | 记录 —— 不变 |
| LLM-P1-3 Batch 调度策略是否合理 | P1 | 不适用（条款③） | 记录 —— 判据归 `REQ-016` |
| LLM-P0-9 Sampling 结果是否正确 | P0 | 通过 | 记录 —— 采样器不变；正确性由 AC1 / AC3 的端到端对照覆盖 |
| LLM-P1-4 CUDA kernel 是否验证 | P1 | 不适用（条款①） | 记录 —— 不改采样器与注意力 kernel |

## Terminology Check（表三）

行数 = `requirement.md` 的模糊名词数 = **10**。

| 模糊名词 | 定义所在 | 结论 |
|---|---|---|
| INT8 权重（weight-only） | `analysis.md` 的 Terminology 第 1 行 | 已定义 |
| 自证低精度 | 同表第 2 行 | 已定义 |
| 逐层精度可读 | 同表第 3 行 | 已定义 |
| 量化对象与 scale 同源 | 同表第 4 行 | 已定义 |
| per-tensor | 同表第 5 行 | 已定义 |
| per-channel | 同表第 6 行 | 已定义 |
| 逐 token 一致率 | 同表第 7 行 | 已定义 |
| logits 相对偏差 | 同表第 8 行 | 已定义 |
| 数值判据有出处 | 同表第 9 行 | 已定义 |
| K/V 缓存元素宽度 | 同表第 10 行 | 已定义 |

## Requirement Coverage Result（表二）

行数 = Included 条数 + Acceptance Criteria 条数 = 4 + 5 = **9**
（K/V 缓存量化的可行性评估已按作者 2026-10-05 裁决移入 `requirement.md` 的 Excluded）。

| 需求条目 | 设计落点 | 结论 |
|---|---|---|
| Included 1 路线选型与实现 | Module Design + Trade-off D1 | 已落点 |
| Included 2 自证低精度 | Module Design 的"精度自证" + Data Structure 的契约表 | 已落点 |
| Included 3 与 FP32 的数值对比 | Trade-off D4 + Runtime Flow 的 P6 段 | 已落点 |
| Included 4 显存 / 体积对比 | Performance Consideration + D7 的引擎体积判据 | 已落点 |
| AC1 端到端可跑 | Architecture + D5 | 已落点 |
| AC2 自证低精度 | Module Design 的"精度自证" | 已落点 |
| AC3 数值判据有出处 | D4（先量再定） | 已落点 |
| AC4 资源下降 | Performance Consideration + D7 | 已落点 |
| AC5 不回归 | D5 + Resource Lifecycle 的"运行时缓冲不变" + 例外节 | 已落点 |

## 已定 / 必须拒绝 条目的来源核对

规则（`phases/p3_review.md` 的 Entry）：每条"已定 / 必须拒绝"的条目，都要走一遍
"兑现它需要什么输入或接口 → 系统里到底有没有"；找不到来源记 P0。

| 已定 / 必须拒绝 的条目 | 兑现它需要的输入或接口 | 系统里有吗 | 结论 |
|---|---|---|---|
| 清单里每个 tensor 都能取到 int8 权重 | `model_int8.safetensors` 的 key 集合 | 有（本轮新增的产物，脚本是落点） | 通过 |
| int8 权重元素数 = 源张量元素数 | safetensors 头里的 shape + 清单 `source_key` | 有（库提供） | 通过 |
| `scales` 长度符合粒度 | 清单 + 声明轴长度 | 有 | 通过 |
| scale 与量化对象同源 | `weights_source.sha256` + 源文件 | 有 | 通过 |
| 清单缺项 / 清单外被要求量化 → 失败 | 清单 + 权重 key 集合 | 有 | 通过 |
| 引擎边界精度与运行时一致 | `ICudaEngine::getTensorDataType` | 有（现有实现） | 通过 |
| per-channel 的轴 = rank-3 第 2 维 | 建图侧的 `Dims3(1, in, out)` 声明 | 有（已核对源码） | 通过 |
| 改图必须 bump `graph_version` | `builder.cpp` 的手工代次约定 | 有（现有约定） | 通过 |
| 清单与 int8 权重必须进指纹 | `MakeFingerprintInputs` 的 `source_files` | 有（现有接口） | 通过 |
| DQ 必须被 TRT 吸收（不被常量折叠） | TRT 10.15.1 的 DQ / 融合行为 | **假定**：作者 2026-10-05 指令"假设有头文件"+ `AGENTS.md` §1 环境；带 P5 核对义务 | 通过（假定 + 核对义务，见 P1-2） |

## P0 Blockers

**无。** 每条 Included / AC 都有非空壳落点；术语全部已定义；"已定 / 必须拒绝"条目都能追到输入
或接口（唯一的假定是 TRT 的 DQ 行为，已按作者指令登记为"假定 + 核对义务"并送 P1-2）。

## P1 Risks

1. **D6 的量化对象清单是本设计推导出的默认值，不是作者点名**（`design.md` D6 已标明）。需要
   你在 Gate-A 上确认或改单：默认 = 48 个 Linear 权重 + `wte` / `lm_head`，排除 `wpe` 与
   LayerNorm / bias。
2. **D7：激活精度与 DQ 是否被吸收尚未验证**，且"只有 FP16 激活才融合"的情形会依赖
   `REQ-018`（GPT-2 FP16 端到端 NaN，仍是开放中的 bugfix）。P5 第一步用最小图 + 引擎体积判据
   消掉；不成立则回 Gate-A。
3. **权重资产的假定**：`models/gpt2/model.safetensors`（548 MB）按 `docs/PROGRESS.md` §6.5 的
   规格存在，但当前不在盘上。P4 之前必须核对存在性与 SHA256；缺失即阻塞并上报。
4. **收益量级是推导，不是实测**（固定项 ≈2.75 ms、179 GB/s、预估 2–4×）。当前环境无 GPU，
   无法在本阶段消掉；归 P4 / P7。
5. **两条 P0 级判据的验证时点不在 P3**：`CPP-P0-8`（编译）与 `CUDA-P0-6`（benchmark 证明）。
   它们分别绑定 P5 的 Exit Gate 与 P7；本评审按"待确认（人工确认）"记，不作为 P3 的 P0 Blocker
   ——若你认为这类也该按 P0 处理，请指出，我改判。
6. **数值对照的上下文覆盖范围未定**（`LLM-P1-2`）：建议至少短 + 长两档。需要你在 Gate-A 上定
   口径（或授权我在 P6 的 `test_plan.md` 里按"短 + 长各一档"落点）。
7. **写冲突**：`REQ-016-continuous-batching` 仍在 `P5-Implementation / in-progress`，本 feature
   与 `REQ-019` 都要改同一段运行时入口。顺序已定（REQ-016 先），但**它交付前本 feature 不得开
   代码**——这是排期风险，不是设计缺陷。

## P2 Quality

- `test_plan.md`（P6 产物）里每个数值阈值必须挨着写出处；P4 未量之前不得先填阈值。
- 新产物命名建议与既有 `resnet18_qdq*.onnx` / `*.meta.json` 的风格对齐（`model_int8.safetensors`
  / `quant_int8.json`），便于将来的资产闸门识别。
- `STATE.md` 的 `phase` 在 Gate-A 通过后要置 `P4-Baseline`，勿沿用旧版"没评审也挂 Gate-A"的写法。

## Decision

**PASS**（P0 Blocker = 0；7 条 P1 送 Gate-A 裁决）。

按 `phases/p3_review.md` 的 Human Gate：P0 不存在、P1 存在 → **暂停，等待作者确认**。

## Gate-A 裁决回填（2026-10-05）

口径：**上面的 `## P1 Risks`、表一的 4 行"待确认"与 `## Decision` 保持 P3 时点的原样，不回改**
——那是"当时评的是什么"的记录；本节只登记裁决结果，**唯一来源**是 `STATE.md` 的 `## 判据对照`。

| P1 | 裁决（2026-10-05） | 落地 |
|---|---|---|
| P1-1 D6 默认量化清单非作者点名 | **同意**默认值（48 个 Linear + `wte`），`wte` / `lm_head` 定为**可摘除项** | 最终确认放 D7 之后；`STATE.md` 的 P1-1 |
| P1-2 D7 未验证 / 依赖 `REQ-018` | **同意做 D7**；**与 `REQ-018` 不合并** | 真机第一步；分支预案 = 若"仅 FP16 才融合"，在 `REQ-018` 的 requirement 加联合验收（届时再点名）；`STATE.md` 的 P1-2 |
| P1-3 权重资产是假定 | 本设备**同意**保留；**换设备时按设备纪律第一时间确认真机测试能力** | 设备纪律落在 `requirement.md` 的 `## Constraints`；`STATE.md` 的 P1-3 |
| P1-4 收益量级是推导 | **同意**保留为推导，**真机测试后必须更新** | 由 P4 / P7 实测替换；`STATE.md` 的 P1-4 |
| P1-5 编译 / benchmark 两条 P0 的时点不在 P3 | **属环境依赖 → 维持绑 P5 Exit Gate / P7**；**真机后把结果更新到 P3** | 见下表两行；`STATE.md` 的 P1-5 |
| P1-6 数值对照的上下文覆盖未定 | **同意**按"短 + 长各一档"（prompt 4 / 960，与 PF-9 同口径） | 落进 P6 的 `test_plan.md`；`STATE.md` 的 P1-6 |
| P1-7 与 `REQ-016` 的写冲突 | 按评审建议执行 | S1 已并行完成；**S2 等 `REQ-016` 交付**；谁改图谁 bump；真机窗口两 feature 分开做、分开记；`STATE.md` 的 P1-7 |

**表一中 4 行"待确认"的回填**（原文见上表；本节只给新状态）：

| Item | 新状态 | 依据 |
|---|---|---|
| `CPP-P0-8` Debug / Release 编译 | 待确认（**时点已定**：P5 Exit Gate；真机后把结果更新回本表） | P1-5 |
| `CUDA-P0-6` benchmark 证明 | 待确认（**时点已定**：P7；真机后把结果更新回本表） | P1-5 |
| `TRT-P1-1` Tensor dtype 转换 | 待确认（= **D7**，已同意做；真机第一步，判据 = 引擎体积必须下降） | P1-2 |
| `LLM-P1-2` Long context 是否测试 | 待确认（**口径已定**：短 + 长各一档，落进 P6 的 `test_plan.md`） | P1-6 |

裁决落地后，本 feature 的 `phase` 由 `waiting-human-gate` 变为 `P5-Implementation / in-progress`
（依据作者同日指令"真机测试搁置、先完成开发工作"）。
