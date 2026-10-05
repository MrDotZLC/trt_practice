# Design

## 修正记录（2026-10-05）

作者 2026-10-05 裁决：**D1 采纳 C**（Python 离线量化产出 int8 权重 + scale，C++ 原生建图只负责
读常量并挂反量化）；**K/V 缓存量化维持 D3**（只做权重，缓存量化另立里程碑）。本轮同时补上 P2 的
硬性缺口 `## Requirement Coverage`，并新增 D5（图版本与引擎指纹契约）、D6（量化对象清单）与
D7（激活精度与 DQ 融合的存在性判据）。

上一版设计的三处表述与现状不符（依据见 `analysis.md` 的"文档与现状的矛盾"）：INT8 权重没有来源、
没有 Requirement Coverage、没有把新增产物的指纹影响写出来。

## Overview

| 里程碑 | 内容 | 是否本轮 |
|---|---|---|
| **S1 权重量化（weight-only INT8）** | GPT-2 以 int8 权重端到端跑通，含自证与有出处的数值判据 | 是 |
| **S2 K/V 缓存量化** | cache 元素宽度可配 + 注意力插件支持低精度 cache | 否（作者 2026-10-05 确认，另立里程碑） |

S1 内部顺序：先做"能跑 + 能自证 + 有判据"，再做"省了多少"的对照。

## Architecture

S1 采用路线 C（D1 = C）：

```text
models/gpt2/model.safetensors（FP32，HF 原样布局）
        │
        │ ① Python 离线量化（新增脚本）
        ▼
models/gpt2/model_int8.safetensors   ← int8 权重（与源同布局，不重排）
models/gpt2/quant_int8.json          ← scale / 粒度 / 轴 / 来源哈希 / 清单
        │
        │ ② C++ 原生建图（按清单驱动）
        ▼
 addConstant(kINT8) → addDequantize(scale, zeroPoint) → MatMul / Gemm
        │                                    （期望 TRT 把 DQ 融进 GEMM）
        ▼
 引擎（`detailed_profiling` 打开时可读逐层精度）
        ▼
 运行时（契约不变：仍向引擎查询边界精度，仍按实际精度分配缓冲）
```

## Module Design

| Module | Responsibility | Dependency |
|---|---|---|
| `tools/convert/quantize_gpt2.py`（新增，Python） | 从 FP32 权重算 scale、产出 int8 权重与清单；复用视觉侧的来源自检 / 饱和比例统计 / 产物身份（SHA256） | 现有 `tools/convert/` 与 `models/gpt2/` |
| 量化清单（新产物） | 声明哪些张量量化、粒度、轴、scale、来源哈希 | 由上面的脚本产出 |
| 权重加载器（扩） | **只多开一条零拷贝**：源与目标同为 int8 时直接返回；其余"转成 int8"的请求继续拒绝。量化清单与 int8 权重必须与它**同生命周期**（常量指针在 build 结束前有效，沿用现有零拷贝语义） | 现有 `SafetensorsLoader` |
| 语言模型构建器（扩） | 按清单把对应权重建成 int8 常量并挂 DQ；清单里有而模型缺 → 失败。**prefill 与 decode 两张图必须用同一份清单**（否则两侧数值口径不同，runner 的 K/V 精度一致性校验也会失去意义） | 权重加载器 + 量化清单 |
| 构建配置（扩） | 新增"量化清单路径"入口；**不动现有精度档位语义** | — |
| 精度自证（复用） | 把"量化层确实以 int8 执行"变成用例断言 | `detailed_profiling` + `IEngineInspector` |
| 运行时 | **本轮不改**（weight-only 下 cache 与 logits 的契约不变） | — |
| 分页 cache / 注意力插件 | 留给 S2 | — |

## Data Structure

### 量化清单（`quant_int8.json`）

```text
QuantSpec {
    format_version;                     // 产物格式版本（为将来扩 per-channel 留位）
    generator { tool; command; };       // 可复现
    weights_source { path; sha256; };   // 量化对象的来源身份
    int8_weights  { path; sha256; };    // 量化结果的产物身份
    scheme;                             // 本轮固定为对称量化
    zero_point;                         // 本轮固定 0
    entries: [
        { tensor; source_key; granularity; axis; scales[]; saturate_ratio; }
    ]
}
```

### 契约与失败面（"必须拒绝 / 必须判等"的判定依据来源）

| 判据 | 判定依据的来源 |
|---|---|
| 清单里每个 tensor 都能取到 int8 权重 | 清单 `entries` ↔ `model_int8.safetensors` 的 key 集合（都是我们自己的产物） |
| int8 权重的元素数 = 源张量元素数 | 清单的 `source_key` + safetensors 头里的 shape（库提供） |
| `scales` 长度符合粒度 | 本轮 = 1；per-channel（后续）= 声明轴的长度 |
| scale 与量化对象同源 | 清单的 `weights_source.sha256` + `source_key`（文件级证据） |
| 引擎边界精度与运行时一致 | 现有 `getTensorDataType` 查询（口径不变） |

### 布局与轴

权重在产物里保持 HF 的 `[in, out]`（**不经任何重排**），建图声明成 rank-3 `[1, in, out]`。
因此 **per-channel 的轴是 rank-3 的第 2 维（输出通道）**，不是 HF 习惯的轴 0。本轮 per-tensor
不涉及轴，但清单字段与这条约定先写死，避免将来按 HF 习惯接错。

## Runtime Flow

```text
[P4 基线] FP32：引擎体积 / 显存 / 每步延迟
   ↓
[P5 实现] ① 真机核对 TRT 的 DQ API（见"假设与核对义务"）
          ② 跑量化脚本 → 核对清单自检与饱和比例
          ③ 建 int8 引擎 → 用 detailed_profiling 读逐层精度
   ↓
[P6 测试] 同 prompt、同采样策略：INT8 vs FP32
          ├ 逐 token 一致率（含样本量）
          └ prefill logits 相对偏差 + 绝对差（同处给出比较对象的形状 / 布局）
   ↓
[P7 基准] 同 session、同二进制、逐轮交替 → 延迟 / 显存 / 引擎体积
```

## Resource Lifecycle

| 资源 | 何时创建 | 何时释放 |
|---|---|---|
| FP32 权重（量化脚本的输入） | 脚本运行期 | 脚本退出 |
| int8 权重 / 清单（磁盘） | 脚本产出后常驻模型目录 | 手动删除 |
| 权重数据（host） | 加载器零拷贝指向 mmap / 转换缓存 | 加载器析构 |
| int8 常量 + DQ 层（TRT） | 建图期 | 引擎析构 |
| 引擎缓存（磁盘） | 构建后落盘；指纹含精度与清单文件身份 → 换配置 / 换量化产物自动重建 | 手动删除 |
| 运行时缓冲 | **不变**（仍按引擎声明的边界精度分配） | 析构 |

要点：运行时**不应**因为权重量化而新增任何缓冲逻辑——这是选"让 TRT 在图上做 DQ"的理由之一。

## Performance Consideration

- **compute**：本机 TU116 **无 Tensor Core**。视觉侧实测过 `i8i8` tactic 存在（那是 Q / DQ
  全量化路径），但 weight-only 是否会被选进 INT8 GEMM **未知**——本设计不假设。
- **memory（主要收益）**：权重字节数 4 → 1，decode 每步的权重读取量降到约 1/4。这是本机唯一
  能提前立住的收益来源。
- **测量口径**：沿用既有协议（同 session、同二进制 A/B、逐轮交替、报中位数与四分位、先声明
  判别下限）。本机这类测量的判别下限约 ±400~600 µs（出处 `docs/PROGRESS.md` §5.13b）。
  **必须报"每步延迟"**，不能只报每 token 吞吐。
- **不融合的代价**：若 DQ 不能与 GEMM 融合，多出的类型转换可能让延迟变差——那时结论就写
  "收益不成立"，按实测写。

## Trade-off

### D1 落地路线

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. C++ 内量化 | 扩取数路径把 FP32 直接转 int8，再用 TRT 的 DQ API 建图 | 量化逻辑落在 C++，scale 来源与产图不可分离，排查成本高（#46 的教训正是"尺子量 A、裁剪 B"） |
| B. ONNX Q/DQ | 复用视觉侧成熟工具链 | GPT-2 的 ONNX 路径没有 decode 图与 K/V 输入 → 依赖 `REQ-019`，且 PF-7 尚未跑出可复现对照 |
| **C. Python 产 int8 权重 + scale，C++ 只读常量挂 DQ（采纳）** | 量化、来源自检、饱和统计、产物身份都在 Python 侧；C++ 只多"读 int8 + 挂 DQ" | 需要新增一个产物格式与两处小改动（loader 零拷贝 + 构建器按清单建图）；换来"可按文件级证据判对错" |

**决策：C（作者 2026-10-05）。** 依据：与项目既有分工一致（Python 产图 / C++ 读图），且能在
不碰运行时的前提下，把 #46 那类"来源不一致"变成文件级可判的问题。

### D2 量化粒度

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. per-tensor（先做）** | 每张权重一个标量 scale | 简单、可解释；精度损失略大 |
| B. per-channel | 每个输出通道一个 scale（rank-3 的轴 2） | 精度更好；多一份通道轴处理，且必须先证明 scale 与量化对象同源 |

**决策：先 A 后 B**，B 必须在 A 的判据体系跑通之后做。清单格式已为 B 留字段
（`granularity` / `axis`）。

### D3 K/V 缓存量化

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 本轮只做权重 | 改动面限于构建期 | 收益集中在权重读取；cache 仍是 2 字节 |
| **B. 权重 + 缓存分两阶段（采纳）** | S2 单独做缓存 | 缓存量化要动元素宽度的表示、插件白名单、写 K/V 的转换路径、运行时精度校验——**四处**，且每处都影响数值 |

**决策：本轮只做权重，缓存量化另立里程碑（作者 2026-10-05 确认）。** 理由：两者混在一起会让
数值偏差无法归因。

### D4 判据

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 复用视觉模型的阈值 | 省事 | **禁止**：跨精度 / 跨口径复用（`AGENTS.md` §7） |
| **B. 先量基线再定（采纳）** | P4 量出 FP32 自身的构建间波动，P6 用同 session 对照 | 慢一点，但判据能回答"凭什么这么定" |

### D5 图版本与引擎指纹

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 只加图、不动身份 | 省事 | 加了 DQ 就是改了图，但指纹看不见 → 复用旧引擎（静默）；清单是独立文件，改了 scale 也不会失效（静默） |
| **B. 图版本 +1，且清单与 int8 权重进 `source_files`（采纳）** | 一次性的引擎重建 | 代价是一次重建（分钟级，按 `docs/PROGRESS.md` §6.5 的口径属预期）；换来"改了量化产物一定重建" |

**决策：B。** 两处都要做：`kEngineGraphVersion` +1；`MakeFingerprintInputs` 的源文件清单加入
量化清单与 int8 权重文件。

### D6 量化对象清单（哪些张量进 INT8）

**默认值（按"每步读取量"推导，非作者点名；2026-10-05 Gate-A 已裁决：同意）**：

| 张量组 | 参数量 | FP32 字节 | decode 每步是否整块读 | 默认处置 |
|---|---|---|---|---|
| 12 层 × 4 个 Linear（`c_attn` / `c_proj` / `c_fc` / `mlp.c_proj`） | 84.93 M | 340 MB | 是 | **进 INT8** |
| `wte.weight`（`lm_head` 与它绑定） | 38.60 M | 154 MB | 是（生成 logits 时读整块） | **进 INT8** |
| `wpe.weight` | 0.79 M | 3.1 MB | 否（只 gather 一行） | 留 FP16 / FP32 |
| LayerNorm 权重与全部 bias | 0.12 M | 0.5 MB | 是但极小 | 留 FP16 / FP32 |

- 参数量按 `models/gpt2/config.json` 的 `n_layer=12 / n_embd=768 / vocab_size=50257 /
  n_positions=1024` 算出，逐项与 `weight_map` 对得上。
- **`wpe` 排除的理由**：它只被 gather 读一行，量化它省不到带宽，反而多一条"gather 出 int8 还要
  DQ"的路径。
- **`wte` / `lm_head` 的风险与退路**：它是所有张量里对 logits 最敏感的一项。若 P6 判据不过，
  **从清单里摘掉它重建一次即可**（清单进指纹 → 自动失效重建），不必改代码。
- **来源登记（`AGENTS.md` §5 第 5 条）**：清单内容在 `requirement.md` 与既有文档里**没有**可
  引用的依据——上表是设计按"每步读取量"推导的默认值，**不是作者点名**。
- **裁决（2026-10-05，Gate-A）**：作者**同意**上表默认值，并把 `wte` / `lm_head` 定为
  **可摘除项**（判据不过时先摘它复测——清单进指纹，改单只需一次重建）；**最终确认放在 D7 之后**。
  处置记录见 `STATE.md` 的 `## 判据对照`（P1-1）。
- 机制仍是清单驱动：脚本产出、构建器按清单执行；**清单里有而权重取不到、或清单外仍被要求量化
  → 失败**（不静默回落）。

### D7 激活精度与 DQ 融合的存在性判据

**这是本路线的存在性前提，不是实现细节。** weight-only INT8 的收益全部来自"权重按 1 字节读"；
若 TRT 在构建期把 `DQ` **常量折叠**掉（例如激活为 FP32 且挑不到 int8 × fp32 的 GEMM），引擎里
留的就是 FP32 权重——引擎体积不降、带宽不降，收益归零**且不会报错**。

| 事项 | 内容 |
|---|---|
| 观测判据 | **引擎体积相对同一配置的 FP32 引擎必须下降**（最便宜、最直接的融合证据）；次级证据 = 逐层信息里 GEMM 层是否出现 int8 |
| 何时判 | **P5 第一步**：最小图（单个 Linear + int8 常量 + DQ）在"无 flag（FP32）"与"kFP16"两种构建下各建一份 |
| 失败时的退路 | 不成立 → **回 Gate-A**，不自行改判据继续 |
| 与其它 feature 的耦合 | 若"只有 FP16 激活才融合"，就会依赖 `REQ-018`（GPT-2 的 FP16 端到端 NaN，仍是开放中的 bugfix）→ 必须在 Gate-A 上让作者知道 |
| 禁止项 | **不许**把"与 FP16 基线对比"当作绕开 NaN 的手段——那等于用已知坏的基线当尺子（`AGENTS.md` §7） |

AC4（资源下降）与本节共用同一个观测：引擎体积 / 显存是 P7 的交付项，同时也是 D7 的判定证据。

## Requirement Coverage

| 需求条目 | 设计落点（章节） | 交付里程碑 | 验证 Phase |
|---|---|---|---|
| Included 1 路线选型与实现 | Module Design + Trade-off D1 | S1 | P2 → Gate-A |
| Included 2 自证低精度 | Module Design 的"精度自证" + Data Structure 的契约表 | S1 | P6 |
| Included 3 与 FP32 的数值对比 | Trade-off D4 + Runtime Flow 的 P6 段 | S1 | P4 → P6 |
| Included 4 显存 / 体积对比 | Performance Consideration | S1 | P4 + P7 |
| AC1 端到端可跑 | Architecture + D5 | S1 | P6 |
| AC2 自证低精度 | Module Design 的"精度自证" | S1 | P6 |
| AC3 数值判据有出处 | D4 | S1 | P4 → P6 |
| AC4 资源下降 | Performance Consideration | S1 | P7 |
| AC5 不回归 | D5 + Resource Lifecycle 的"运行时缓冲不变" | S1 | P6 |

## 例外与不得跳过

- **S2 不纳入本轮**（Included 5 / D3）：不得用于跳过 Included 1–4 与 AC1–AC5 的任何一条。
- **运行时本轮不改**：不得用于跳过 AC2（自证）与 AC5（不回归）——"不改"是设计判断，
  "没问题"要实测。
- **K/V 缓存量化移出本轮**（`requirement.md` 的 Excluded + 本文 D3）：不得用于跳过 Included 1–4
  与 AC1–AC5 的任何一条，也不得用于跳过上面的 Requirement Coverage 表
  （行数 = Included 4 + AC 5）。

## Risk

| 风险 | 影响 | 缓解 |
|---|---|---|
| TRT 的 DQ API 与假定不符（签名 / per-channel 约束） | P5 第一步就卡住 | 见"假设与核对义务"；不成立则回 Gate-A |
| scale 与量化对象不同源 | 精度静默崩坏，看起来像"TRT 不行" | 文件级证据（清单的 sha256 + 饱和比例）+ 探针对照，沿用 #46 的排查法与 §3.0j 的教训 |
| DQ 未与矩阵乘融合（或**被常量折叠**） | 轻则变慢，重则收益归零且无报错 | **D7**：引擎体积判据 + P7 延迟对照；不成立就回 Gate-A |
| 精度档位语义被稀释 | 现有 FP32 / FP16 行为被改坏 | 新增独立入口，不动现有档位语义 |
| 判据阈值来路不明 | 违反 `AGENTS.md` §7 | 阈值出处写进用例与测试计划 |
| 量化清单未进指纹 | 改了 scale 却复用旧引擎（静默） | D5 |
| 6 GB 显存不足（大上下文构建失败） | 验收受阻 | P4 先量 FP32 的显存占用 |

## 假设与核对义务

作者 2026-10-05 指令：按 `AGENTS.md` §1 的设备信息与既有产物规格当作前提。下面三条是**假定**，
不是已验证结论——每条都带核对动作与失败时的退路：

| 假定 | 依据（谁给的） | 核对义务（何时 / 怎么消掉） |
|---|---|---|
| TRT 10.15.1 提供显式 DQ 层，且 per-channel 可以用轴表达 | 作者 2026-10-05 指令 + `AGENTS.md` §1 的环境信息 | **P5 第一步**：真机读 `NvInfer.h` 确认签名与约束，并把结论写进本文档的决策记录；不成立 → 回 Gate-A |
| `models/gpt2/model.safetensors`（548 MB）按 `docs/PROGRESS.md` §6.5 的规格存在 | 作者 2026-10-05 指令 + `docs/PROGRESS.md` §6.5 | P4 之前核对存在性与 SHA256；缺失即阻塞并上报，**不猜来源** |
| int8 常量 + DQ 会被 TRT 融合进 GEMM | 视觉侧 ONNX Q/DQ 的实测先例（**不同路径**） | P5 用逐层精度自证，P7 用延迟对照；不融合 → 结论写"收益不成立" |
