# Design

## Overview

本条按 **乙框架** 组织：**性能门只挡"子图替换"这一段**，不挡契约统一 / 拓扑识别 / 自定义算子进图。

**为什么改**（上一版设计 2026-10-01 的问题，记为 D0）：`requirement.md` 的 Goal 只说"先有可复现的
性能依据再决定是否推进**替换**"，Excluded 也只说"若测量表明没有收益，结论就是不做替换"——
两处都只约束替换；`## Problem` 第 3 条的矛头同样指向"替换值不值得做"。
上一版把 PF-7 提升成**整条 feature** 的开门条件，这一步放大没有 requirement 依据，
后果是三项与"替换收益"无因果关系的可完成工作被无限期停摆（当前设备非目标真机，PF-7 不可执行）。

**PF-7 的判据口径（本次澄清）**：PF-7 回答的不是"谁更快"，而是
**"本平台的构建间噪声是否小到后续 A/B 有意义"**。

```text
PF-7（同 session：≥3 次构建 × ≥20 次推理，报中位数与极差）
   ├─ 极差 < 中位数差 → 可判 → 替换进入 S3，后续 A/B 的结果可信
   └─ 极差 ≥ 中位数差 → 未定 → 任何小于噪声的效果不可观测
                        → 按 Excluded，"替换"不做（记为未满足（阻塞））
```

依据：`requirement.md` 的 Problem 3 / Goal / Excluded；`docs/future_iterations.md` §10.2
（两次测量方向相反：ONNX 慢 22% / 快 24%）；`docs/PROGRESS.md` §5.13b（判别下限约 ±400~600 µs）。

## Milestones

分期规则（`phases/p2_design.md`）：每个里程碑至少完整承担一条需求条目，不得把需求条目悬空。

**接口细化（指针式登记，细节的唯一来源在各自文件）**：S0 → `s0_contract_interface_spec.md`；
S1 → `s1_topology_interface_spec.md`；S2 → `s2_custom_op_interface_spec.md`；
S3 的细化**待 PF-7 判定后再出**（理由见「待定项（阻塞）」，避免为一个可能不做的里程碑先行设计）。

| 里程碑 | 交付 | 依赖 | 需真机 | 验证 Phase | 状态 |
|---|---|---|---|---|---|
| **S0 契约统一** | 外部图输入 dtype 统一为 INT32；`position_ids` 提升为显式输入；两路共用同一套绑定调用 | 图重写工具（Python） | 是（引擎重建） | P6 | 设计就绪；落码与验证需真机 |
| **S1 拓扑级识别** | 识别从"计数"升级到"连接级"，能拒绝"计数相同、连接不同"；计数基线与 `absent_ops` 护栏保留 | 无（纯 host） | 否 | P6 | 设计就绪；夹具级用例可在沙箱验证（需 `onnx` 包） |
| **S2 自定义算子进图并被执行** | 导出侧产出 `mini_trt_llm` 域算子；解析器接 `IPluginRegistry`；夹具图上插件真正执行且数值一致 | 与 S0 共用重写工具 | 是 | P6 | **阻塞**（P0-1 未闭环 + 本阶段不真机，见 `review.md`） |
| **S3 子图替换** | 注意力子图重写 + 替换生效判据 + 前后一致性 | S0 + S1 + S2 + **PF-7（P4）** | 是 | P4 / P6 / P7 | **待定（PF-7 待真机）** |

前置：`docs/future_iterations.md` §6.2（自定义算子导出）是 S2 的实现前置；PF-7 是 S3 的 P4 前置。

## Architecture

```text
（现状：解析即构建，没有任何图变换环节）

ONNX ──解析──→ 网络定义 ──原样──→ 引擎          自研算子实现：零使用者

（本次：在图侧插入识别与重写）

ONNX ──① 拓扑级识别（S1）───────────────┐
     └──② 图重写（S0 契约统一 / S2 自定义算子 / S3 子图替换）→ 新图
                                              │
                                              ↓ 解析（挂 plugin registry，S2）
                                            网络定义 ──→ 引擎
                                              ↑
                                    ③ 生效判据（S3：层信息里出现插件层）
```

新增 / 扩展的四个环节：

1. **识别（扩展 `tools/inspect_onnx.py`）**：从"算子计数 + 固定基线"升级为"计数 **且** 连接形态"，
   连接形态不符即拒绝。
2. **图重写（新增 `tools/rewrite_onnx.py`）**：S0 的契约统一与 S2 / S3 的算子替换共用同一工具。
3. **解析器接注册表（扩展 `BuildFromOnnx`）**：现状是 `createParser(*network, logger_)`，
   **没有传 plugin registry** → 即使图上出现自定义域算子也找不到创建器（见 `analysis.md`）。
4. **生效判据**：复用 `tests/engine_layer_info_support.hpp` 与 `BuildOptions::detailed_profiling`。

## Module Design

| Module | Responsibility | Dependency |
|---|---|---|
| `tools/inspect_onnx.py`（扩展） | 拓扑级识别：子图边界 + 连接判据；保留计数基线与 `absent_ops` 护栏 | `onnx` 包 + 图资产 |
| `tools/rewrite_onnx.py`（新增） | 图重写：S0 `--contract-unify`、S2 / S3 `--replace-subgraph`；产出新图与重写报告 | `onnx` 包 |
| `EngineBuilder::BuildFromOnnx`（扩展） | 参数从"子图名白名单"改为"子图规格"；解析器接 `IPluginRegistry` | 既有 builder / TRT |
| 插件注册表 / 插件本体 | 提供三个自研算子的创建器与（反）序列化 | 已具备，当前无使用者 |
| 一致性对拍用例 | 替换前后、两条路在同一输入上的数值对比 | `tests/test_gpt2_onnx.cpp` |
| 生效判据 | 读引擎逐层 ONELINE，证明插件层真的进了引擎 | `IEngineInspector`（既有） |
| 引擎缓存 | 两条路（替换前 / 替换后）独立命名，避免互相判过期 | 既有指纹机制 |

## Data Structure

```text
SubgraphBoundary {
    name;              // attention / layernorm / position_embedding
    internal_nodes;    // 要被替换掉的算子节点集合
    inputs;            // 边界输入张量名（保持原名，禁止改）
    outputs;           // 边界输出张量名（保持原名，禁止改）
    target_domain;     // "mini_trt_llm"
    target_op;         // 目标算子名——**只允许指向已有算子实现**
    plugin_version;    // 解析器查创建器时用
}

RewriteReport {
    boundary;          // 本次替换的边界
    renamed_tensors;   // **必须为空**；非空即失败（边界名一变，下游引用与对拍基准全断）
    io_signature;      // 重写前后 I/O 名字与 dtype 的对照（S0 用它锁契约）
}

ContractPlan {         // S0 专用
    input_retype;      // input_ids: INT64 -> INT32（图输入本身，不是图内插 Cast）
    hoisted_input;     // position_ids: 新增图输入（INT32，形状 [batch, seq]）
    cast_sites;        // 图内仍需要 INT64 的消费点（按输入的 Cast 补）
}
```

要点：**替换只动内部，不改边界张量名**——名字一改，外部契约与所有下游引用都要跟着改，
也会让"替换前后对拍"失去共同基准。

## Interfaces

### 图重写（Python 侧）

```text
python mini_trt_llm/tools/rewrite_onnx.py <in.onnx> <out.onnx> \
    [--contract-unify]                 # S0：改输入 dtype + 提升 position_ids
    [--replace-subgraph attention]     # S2 / S3：用自定义域算子替换子图
    [--report <path.json>]             # 落 RewriteReport，供断言与留痕
```

契约：**只写新产物，绝不原地改输入图**；`renamed_tensors` 非空时以非零退出码收场。

### 识别（Python 侧）

```text
python mini_trt_llm/tools/inspect_onnx.py <onnx> --check            # 计数级（保留，不变）
python mini_trt_llm/tools/inspect_onnx.py <onnx> --check-topology   # 新增：连接级
```

`--check` 与 `--check-topology` 是**两条都要过**的护栏（旧基线不得因升级而被替换掉）。

### 构建入口（C++ 侧）

```text
// 现状：subgraph_names 只做名字白名单校验，不产生任何图变换
bool BuildFromOnnx(model_dir, onnx_path, engine_path,
                   const std::vector<std::string>& subgraph_names);

// 设计：同一入口扩成"子图规格"；空 vector = 保持现状（只核对不替换）
bool BuildFromOnnx(model_dir, onnx_path, engine_path,
                   const std::vector<SubgraphSpec>& subgraphs);
```

**签名变更会波及调用方**（`tests/test_gpt2_onnx.cpp`，以及 `REQ-016` 点名过的运行时入口写冲突点），
必须与调用方同轮改，不允许留下编译不过的中间态。

自定义算子要被解析出来，解析器必须拿到插件注册表：

```text
nvonnxparser::createParser(*network, *logger_, *nvinfer1::getPluginRegistry());
```

**来源核对（S2 开工前必做）**：上面的写法是**待验证形态**——TRT 是否提供带 `IPluginRegistry` 的
`createParser` 重载，本机没有 TRT 头文件，**本次未核对**（`review.md` 记为 **P0-1**）。
S2 的第一步是在 `NvOnnxParser.h` 上核对；若签名不符，本设计要回到 P2 重新选路
（不允许现场临机改接口）。

### 生效判据（S3）

```text
engine_layer_info_support.hpp 的 InspectLayerInfo(engine, dump_path)
    + BuildOptions::detailed_profiling = true   // 默认关，只有"需要自证"的场合才开
```

读数口径参照该共享头记下的教训（"逐层信息里没有 `[I8]` 这类标签，要读 `Format/Datatype`"）：
**先落盘 ONELINE 原文，再据此定断言**，不凭直觉猜标签。

## Runtime Flow

```text
             ┌──────────────── S1 识别（host，不占真机）────────────────┐
 gpt2.onnx ──┤ 计数基线 + 连接判据 → 通过 / 拒绝                      │
             └────────────────────────────────────────────────────────┘
                                    │ 通过
                                    ↓
                     S0 契约统一（重写 → 新图 + RewriteReport）
                                    ↓
              ┌─────────────── 引擎 A / 引擎 B ────────────────┐
              │ 同一份输入（input_ids + position_ids，INT32）  │
              │   → 数值对拍（相对界，注明出处与精度）         │
              │   → 同 session 延迟对比（PF-7 同款协议）       │
              └───────────────────────────────────────────────┘
                                    ↑
                        S3 生效判据：ONELINE 里出现插件层
```

## Resource Lifecycle

| 资源 | 变化 |
|---|---|
| 重写后的图文件 | 新产物，与模型一起落盘（不入库，按既有规则） |
| 引擎 A / B | **两条缓存路径必须分开命名**，否则互相判过期、来回重建（项目踩过这个坑） |
| 插件实例 | 由解析器创建，随网络 / 引擎生命周期管理；反序列化路径必须能重建 |
| 临时张量 | 替换不引入额外运行时分配 |
| `RewriteReport` | 每次重写落盘，作为"边界名未变"的留痕证据 |

## Performance Consideration

- **compute**：替换本身不改变计算量，改变的是"由谁执行"（原生层 → 自研算子）以及融合机会。
  自研算子会**打断融合**，可能更慢——这正是必须先测的原因。
- **memory**：自研算子引入的中间张量可能抬高峰值占用。
- **measurement**：必须沿用既有 G6 口径（同 session、同二进制、≥3 次构建 / ≥20 次推理、
  报中位数与极差、先给判别下限）。**跨 session 的数字不可比**（项目已有实测教训）。
- **本机能力边界**：拿不到逐 kernel 时间线 → 只能用端到端与斜率口径，
  不能用"某算子省了多少"来论证。
- **PF-7 的用例缺口**：`OnnxVsNative.PerfPerBuildMedian` 这个用例名**全仓只在测试计划里出现**
  （代码里没有）；`tests/test_gpt2_onnx.cpp` 现有的测量是"各跑 5 次取平均、单次构建、不出极差"，
  **达不到 PF-7 自己的协议**。→ P4 需补一个薄用例（**新代码，须作者点名**）。
  测试计划原文曾记"无新代码"，该表述已于 2026-10-06 更正（见 `docs/future_iterations_test_plan.md` 的 PF-7 行）。

## Trade-off

### D0 要不要把 PF-7 当作整条 feature 的开门条件

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 上一版：PF-7 卡住一切 | 收益未证明前什么都不做 | 与 requirement 的 Goal / Excluded 的范围不符；把三项无因果关系的工作一起停摆 |
| **B. 性能门只挡替换（推荐，本次采用）** | S0 / S1 / S2 先做，S3 等 PF-7 | 需要显式声明"S3 长期待定"的阻塞状态（已获作者接受） |

**决策：B。**

### D1 要不要做替换

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 先跑 PF-7 再决定（推荐）** | 用可复现对照决定方向 | 多一次真机往返；但避免为一个不存在的收益写代码 |
| B. 直接做替换 | 认为"自研算子一定更快" | 与两次反向测量矛盾；做完可能整体更慢 |

**决策：A。** 若 PF-7 判"未定"，"不做替换"是合法且应当记录的结论。

### D2 替换在哪一侧做

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. Python 侧图重写（推荐）** | 在图上完成替换，导出带自定义算子的新图 | 工具链成熟、可离线、可进 host 用例；替换逻辑可见可测 |
| B. C++ 侧网络变换 | 在解析后的网络上定位并重建算子 | 没有通用子图替换原语，要按名找层、逐张量重连；**脆弱且难测** |

**决策：A。** B 只在"只有引擎、没有源图"的场合才有意义，本项目不需要。

### D3 替换哪些子图

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 三个自研算子都替换 | 听起来最完整 | GPT-2 里**不存在** RoPE 与 RMSNorm；LayerNorm 用原生层已足够 → 大部分是空转 |
| **B. 只替换注意力子图（推荐）** | 目标模型上唯一有实际意义的可替换面 | 范围小，但"为什么只替换这一个"能答得清楚 |

**决策：B。**

### D4 自定义算子怎么进图

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 导出时注册符号映射 | 由导出脚本产出自定义域算子 | 需要维护一套符号映射；对上层模型代码透明 |
| **B. 图重写时直接构造自定义算子节点（推荐）** | 在重写阶段产出 | 不依赖上游导出行为，替换逻辑自包含 |

**决策：B。**

### D5 契约统一改哪一侧

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 改外部图（推荐）** | 图输入 dtype 改 INT32，并按输入提升 `position_ids` | 只影响 ONNX 一侧；代价是要引入图重写工具（S2 / S3 也要用） |
| B. 改原生路径去迁就 INT64 | 把原生输入也改成 INT64 | 原生路是项目自己的契约，改它会波及全部调用方与 `REQ-016` 的运行时入口 |
| C. 只在图内插 Cast | 不动图输入类型 | **做不到**：图输入的 dtype 由声明决定，图内 Cast 改不了调用方要喂的类型 |

**决策：A。** 依据：`requirement.md` 的 AC 第 5 条要求"同一套调用方式可直接切换"，
而两条路的差异不只是 dtype（外部图还缺 `position_ids`），所以必须动图。

### D6 重写工具落点

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 独立脚本 `tools/rewrite_onnx.py`（推荐）** | 与只读探针分开 | ctest 只调探针；重写有产物，职责与失败语义都不同 |
| B. 并入 `inspect_onnx.py` | 少一个文件 | 让一个被测试调用的只读探针变成会写图的工具，角色混乱 |

**决策：A。**

## Requirement Coverage

> 落点为本章节名；验证 Phase 取 P4 / P6 / P7。

| 需求条目 | 设计落点（章节） | 交付里程碑 | 验证 Phase |
|---|---|---|---|
| Included：前置测量（外部图 vs 原生可复现对照） | Performance Consideration（PF-7 口径与工具缺口） | S3 前置（P4） | P4 |
| Included：拓扑级识别（保留现有护栏） | Interfaces（识别小节）、Module Design | S1 | P6 |
| Included：子图替换的路线选型与落地 | Trade-off D2 / D3 / D6、Data Structure（`SubgraphBoundary`） | S3 | P6 |
| Included：自定义算子的导出方式 | Trade-off D4、Interfaces（图重写小节） | S2 | P6 |
| Included：替换前后一致性判据 + 引擎可用性 | Runtime Flow、判据与来源 | S3 | P6 |
| Included：两条路输入契约统一 | Trade-off D5、Data Structure（`ContractPlan`） | S0 | P6 |
| AC 1：前置有依据（PF-7 可复现 + 能回答"为什么替换"） | Overview（PF-7 判据口径） | S3 前置（P4） | P4 |
| AC 2：识别到拓扑级（拒绝"计数相同连接不同"） | Interfaces（识别小节，两条护栏都要过） | S1 | P6 |
| AC 3：替换可跑（构建 / 序列化 / 反序列化 / 推理） | Module Design（生效判据行）、Interfaces（生效判据小节） | S3 | P6 |
| AC 4：数值一致（阈值须有出处） | 判据与来源（数值一致行） | S3 | P6 |
| AC 5：契约统一（同一套调用方式可直接切换） | Trade-off D5、Interfaces（图重写小节） | S0 | P6 |
| AC 6：不回归（沙箱全绿、真机无新红） | 判据与来源（回归行） | 全部里程碑 | P6 |

## 判据与来源

> 每条判据必须能回答"凭什么这么定"。**来源不存在的判据不算落点**（`phases/p3_review.md`）。

| 判据 | 来源 |
|---|---|
| 拓扑识别基线（计数 + `absent_ops`） | `tools/inspect_onnx.py` 的 `BASELINE`（首次实测 2026-09-25）；不得因升级而被替换 |
| 连接级识别判据 | 同一图资产上的节点 / 边事实，由 `--check-topology` 在被拒绝的图上报出具体差异；反例由 `make_tiny_onnx.py` 造的夹具图提供 |
| 契约统一（两路可互换调用） | `tests/test_gpt2_onnx.cpp` 现有对拍用例：S0 后应能用**同一份输入**绑定两路（现状是按各引擎声明的契约分别准备输入） |
| 数值一致（替换前后） | 复用**同精度**下的既有出处：FP32 用该用例的 `cosine > 0.999999` / `max_abs ÷ max|ref| < 1e-5`；FP16 用其自身出处。**禁止跨精度复用**（`AGENTS.md` §7） |
| 替换生效 | `IEngineInspector::getLayerInformation(kONELINE)` 的原文（工具与开关均已存在）；断言以 S3 首次实测的原文为准 |
| 引擎可用性 | 构建 → 序列化 → `deserializeCudaEngine`（`tools/inspect_engine.cpp` 已是最小反序列化探针） |
| 性能可判性（PF-7） | 同 session ≥3 次构建 × ≥20 次推理、中位数与极差；判别下限约 ±400~600 µs（`PROGRESS.md` §5.13b） |
| 回归 | `ctest` 沙箱全绿；真机既有用例不出现新红 |

## 待定项（阻塞）

| 项 | 状态 | 阻塞原因 | 重开条件 |
|---|---|---|---|
| S3 子图替换（含 Included 第 1 / 3 / 5 条与 AC 1 / 3 / 4 的落地） | **未满足（阻塞）** | PF-7 需要 sm_75 目标真机，当前设备为 GTX 960（非目标机） | 拿到目标真机 → 先补 PF-7 薄用例 → 跑 PF-7；判"可判"则进 S3，判"未定"则按 Excluded 记录不做 |

作者 2026-10-06 已明确接受"S3 长期标待定（PF-7 待真机）"，故 **S0 / S1 不受此项阻塞**；
S2 另受 `review.md` 的 **P0-1** 阻塞（`createParser` 重载未核实），与本项无关。

> 本条**全部**未决事项（含 S0 / S1 / S2 的落码前置探针、待授权项、待裁决项）已汇总登记在
> `STATE.md` 的 `## 待定与待决清单`（T1–T12）。本节只保留与 S3 直接相关的那一条。

## Risk

| 风险 | 影响 | 缓解 |
|---|---|---|
| 解析器接 `IPluginRegistry` 的 API 签名与本设计不符 | S2 无法落地 | S2 第一步核对 `NvOnnxParser.h`；不符则回 P2（不允许现场改接口） |
| 图重写改动了边界张量名 | 下游引用全断、对拍失去基准 | `RewriteReport.renamed_tensors` 必须为空，用 host 用例锁住 |
| 识别升级后基线变化 | 现有护栏误报 | 保留原计数基线，新增拓扑判据，**二者都要过** |
| 两条引擎缓存互相判过期 | 来回重建，分钟级浪费 | 两条路径独立命名（已有先例与教训） |
| PF-7 结论是"未定" | S3 失去目标 | 提前接受；按 Excluded 记"不做"，并写清重开条件 |
| 自研算子打断融合后更慢 | 做了负收益的改动 | 先 PF-7，再逐子图 A/B；不达标不上生产路径 |
| 契约统一改动了既有的对拍与调用方 | 出现新红 | S0 与调用方同轮改；回归按 AC 6 判 |

## 耦合与写冲突

- 与 `REQ-017-llm-int8-quant`：若它走 ONNX 路线，外部图路径的 decode 图应在这里一并补齐，
  否则 INT8 走 ONNX 路线时要再改一次同一段代码。
- 与 `REQ-016-continuous-batching`：写冲突点在运行时入口——外部图路径要接进运行时，
  且 `BuildFromOnnx` 的签名变更会波及同一批调用方。按技能 `Multi Feature Handling`，
  两方同时要改同一文件时须由作者指定串行顺序。
