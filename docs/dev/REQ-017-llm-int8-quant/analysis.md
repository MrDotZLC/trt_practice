# Analysis

<!--
只描述当前系统，不写设计方案。
-->

## Current Architecture

### 精度是怎么决定的

- 构建配置里只有一个精度档位（FP32 / FP16 / INT8）。FP16 会设一个"整网倾向 FP16"的
  builder flag；**INT8 不设任何 flag**——按注释与实测，`kINT8` 自 TensorRT 10.12 起废弃，
  引擎精度由网络里的显式量化 / 反量化节点决定。
- 权重以常量层的形式进入网络，其元素类型由构建选项单独给出（不是由 builder flag 决定）。
  也就是说：**"整网倾向"与"这个常量是什么类型"是两件事**，代码里已经这么区分了。
- 弱类型网络下，边界张量的实际精度由 TRT 决定，不由配置决定。运行时会**查询**引擎声明的
  精度来分配缓冲（这是本项目踩过坑之后确立的规则，见 `docs/TROUBLESHOOTING.md` + TS-018）。

### 视觉模型侧已有的 INT8 闭环（可复用的部分）

视觉模型走的是**外部图量化**：Python 工具在 ONNX 图上插入对称量化 / 反量化节点，再由
ONNX 解析路径建引擎；配套还有判据脚本与交叉校验脚本。它已经解决了两件事：

1. **"引擎真的在跑 INT8"的自证**：把逐层精度写进引擎，再用引擎检查器把每层的实际精度读出来。
2. **判据与真值的分层**：区分"整体一致率"（测噪声）与"有判别力子集的一致率"（测质量）。

**但它的阈值与本 feature 不可复用**：那套阈值是在图像分类网 + 校准图集上量出来的，
口径不同（`AGENTS.md` §7 明确禁止跨精度 / 跨口径复用）。

### 语言模型侧的现状

- 语言模型只有**原生建图**一条端到端路径：两张图（prefill / decode），decode 用分页 K/V 与
  自研注意力插件。ONNX 路径（`docs/future_iterations.md` §10）目前只有整段前向，
  **没有 K/V 缓存输入、没有 decode 图、不进运行时**。
- 原生建图把权重直接作为常量加进网络，**没有任何量化节点**。
- 权重加载支持从 safetensors 读入并转换到目标类型，包括 INT8 的映射已经存在。
- 引擎缓存的指纹覆盖精度档位与构建开关 → 换精度会自动重建，不存在"复用旧引擎"的假象。

## Module Structure

| 模块 | 与量化的关系 | 现状 |
|---|---|---|
| 精度枚举与映射 | 精度档位 → TRT 数据类型 | 已有 FP32 / FP16 / INT8 三档 |
| 构建配置（flag） | 整网倾向 | FP16 设 flag；INT8 无 flag（显式节点决定） |
| 权重加载器 | 读出 safetensors → 目标类型 | 已支持多源类型，含 INT8 目标类型映射 |
| 语言模型构建器 | 把权重作为常量加进图 | **无量化节点** |
| ONNX 构建路径 | 解析并建引擎 | 只挂整段前向的 profile；无 K/V 输入 |
| 注意力插件 | decode 注意力，读写分页 cache | **只接受 FP32 / FP16 元素** |
| 分页 K/V 缓存 | 缓存元素宽度 | 只有 `is_half` 一个布尔（2 或 4 字节） |
| 运行时 | 校验 cache 输入精度与配置一致 | 精度契约写死为 FP32 / FP16 |
| 引擎检查器 | 读逐层精度 | 已具备（视觉模型侧在用） |
| Python 量化工具链 | 图量化 + 判据 + 探针 | 已有，针对视觉模型 |

## Data Flow

当前语言模型路径的数据流（与量化相关的部分）：

```text
safetensors（FP32/FP16/BF16 权重）
   ↓ 权重加载（按目标类型转换、缓存裸指针）
构建期：权重 → 常量层（目标类型 = 构建选项给的类型）
   ↓
网络（无量化节点）→ 引擎（精度档位只影响 flag）
   ↓
运行时：按**引擎声明的**边界精度分配缓冲 → 前向 → 采样
```

关键事实：**没有任何一步在做权重量化**；`Precision::INT8` 这个枚举值当前无法端到端使用。

## Runtime Flow

与量化有关的运行时行为：

1. 构造期校验：decode 引擎的 cache 输入精度必须与配置一致，否则**拒绝启动**并打印两侧精度。
2. 缓冲分配：按各张量的实际声明精度分别分配（`logits` 与 K/V 可能不同）。
3. K/V 写入：按"源精度 → cache 精度"做转换，四种组合显式分发。
4. 采样：按 logits 的实际声明精度读。

也就是说，运行时的精度契约是"**向引擎查询**"而不是"按配置假定"——这条已经固化，
扩展 INT8 时必须沿用。

## Relevant Code Path

| 位置 | 与本次相关的行为 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/core/precision.hpp`、`src/core/precision.cpp` | 三档精度枚举与 TRT 类型映射 |
| `mini_trt_llm/src/core/builder.cpp` | 构建 flag 的设置（仅 FP16）；逐层精度开关；ONNX 路径解析 |
| `mini_trt_llm/include/mini_trt_llm/core/imodel_builder.hpp` | 构建选项里的"权重常量目标类型" |
| `mini_trt_llm/src/core/gpt2_model_builder.cpp` | 权重 → 常量层；无量化节点 |
| `mini_trt_llm/src/plugins/paged_attention_plugin.cu` | 元素类型只接受 FP32 / FP16 |
| `mini_trt_llm/include/mini_trt_llm/kv_cache/paged_kv_cache.hpp` | cache 元素宽度只有 `is_half` |
| `mini_trt_llm/src/core/llm_runner.cpp` | cache 输入精度校验；按实际精度分配缓冲 |
| `mini_trt_llm/tools/convert/quantize_resnet18.py` | 视觉模型的图量化工具（路线的参考实现） |
| `mini_trt_llm/tools/validate/` | 判据脚本与自检（口径参考） |

## Existing Limitation

1. **语言模型没有量化节点**：`Precision::INT8` 目前是"能枚举、不能用"。
2. **ONNX 路径不是一等公民**：只有整段前向、无 K/V 输入、input_ids 类型与原生路径不同、
   缺 position_ids 输入 → 它现在**接不进运行时**（这正是 `docs/future_iterations.md` §10.1 的触发条件）。
3. **注意力插件不接受低精度 cache**：元素类型白名单是 FP32 / FP16，且插件的 workspace 与
   归约按这两种类型实例化。
4. **cache 元素宽度不可配**：缓存侧只有"半精度与否"一个布尔，无法表达第三种宽度。
5. **运行时精度校验写死两档**：cache 输入精度只认 FP32 / FP16，扩展时要同步改。
6. **量化工具链只覆盖视觉模型**：语言模型的权重命名 / 结构（多层、QKV 合并、词表嵌入）
   都与视觉模型不同，工具不能直接套用。

## Extension Point

- **原生建图路径**：在"权重 → 常量层"这一步之后插入反量化节点，是改动面最小、且**不依赖
  任何其他 feature** 的入口（TensorRT 提供显式的量化 / 反量化层 API，且支持指定输出类型）。
- **ONNX 路径**：已有解析与插件注册表，但要有 decode 图与 K/V 输入才谈得上端到端
  → 依赖 `REQ-019-onnx-subgraph`。
- **注意力插件**：元素类型白名单 + 模板实例化 + workspace 计算，是 cache 量化的改动面。
- **分页 K/V 缓存**：把"半精度与否"扩成元素类型，是 cache 量化的前置。
- **引擎检查器**：逐层精度已可读，可直接用作"真的在跑 INT8"的护栏。

## 文档与现状的矛盾（须当场修，`AGENTS.md` §5 第 4 条）

`docs/future_iterations.md` §1.2 写"**sm_75 有 INT8 Tensor Core**，LLM INT8 可显著提升吞吐"，
而 `AGENTS.md`、`docs/PROGRESS.md`、`docs/interview_summary.md` 一致写本机是
**GTX 1660 Ti（TU116）无 Tensor Core**。TU116 确实没有张量核心单元，`sm_75` 的指令集支持
INT8 张量指令但该芯片不具备对应硬件。

**影响**：这是本 feature 立项理由的一句话依据，写错会让"为什么 INT8 有收益"的论证立不住。
**处置（已完成，2026-10-01）**：作者确认后，`docs/future_iterations.md` §1.2 已改为
"收益来自**显存带宽**（decode 访存受限），不是张量核心吞吐"；`docs/phase4_development_plan.md`
的 D2 与依据表同源那句已按冻结文档的"冲突修正"口径加日期批注（保留原文 + 删除线）。
**决策未变**：INT8 仍走显式 Q/DQ——理由是隐式量化在 TRT 10.15 已废弃，与 Tensor Core 无关。
