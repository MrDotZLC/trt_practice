# Analysis

<!--
只描述当前系统，不写设计方案。
证据口径：本文所有"现状"均在 2026-10-05 逐条撞过源码；引用的行号以当时工作区为准。
-->

## Current Architecture

### 1. 精度档位、builder flag、权重常量类型是三件事

- `Precision{FP32, FP16, INT8}` 与 TRT 类型的映射在 `include/mini_trt_llm/core/precision.hpp`
  与 `src/core/precision.cpp`。
- `EngineBuilder::Config::precision` 只驱动两个动作：FP16 时 `setFlag(kFP16)`；
  `detailed_profiling` 时 `setProfilingVerbosity(kDETAILED)`。**INT8 不设任何 flag**
  （`src/core/builder.cpp:198-212`），依据写在同处注释：`kINT8` 自 TRT 10.12 起废弃，
  精度由网络里的显式量化节点决定。
- 权重常量用什么 dtype 是另一个字段：`BuildOptions::weight_dtype`
  （`include/mini_trt_llm/core/imodel_builder.hpp`），与 builder flag 无关。
- 弱类型网络里**边界张量的实际精度由 TRT 决定**，不由配置决定：`LLMRunner` 构造期用
  `ICudaEngine::getTensorDataType` 查询后按实际精度分配缓冲
  （`src/core/llm_runner.cpp:96-140`；出处 `docs/TROUBLESHOOTING.md` + TS-018）。

### 2. 权重如何进入原生图（量化的落点在这一步）

链路：`WeightLoader::GetWeight(trt_name, target_type, &bytes)`
→ `SafetensorsLoader::GetConvertedData` → `ConvertTensorData` / `GetRawData`
（`src/core/weight_loader.cpp`、`src/utils/safetensors_loader.cpp`）。

取数路径的实际转换面（逐条读源码得到）：

- 零拷贝只有两条：源 FLOAT32→目标 kFLOAT、源 FLOAT16→目标 kHALF（`IsDirectCopy`）；
- 逐元素转换：源 FLOAT32 / FLOAT16 / BFLOAT16 / FLOAT64 → 目标 **FP32 / FP16**，
  累加走 FP32；
- **目标不是 kFLOAT / kHALF 时直接失败**：`GetConvertedData` 命中
  `"Conversion target must be FP32 or FP16"` 分支后返回 nullptr；
- `SafetensorsToTrtDtype` 里**有** `safetensors::kINT8 → nvinfer1::DataType::kINT8` 的枚举映射
  （`src/utils/safetensors_loader.cpp` 的该函数），但它不在取数路径上，也不解锁任何用法；
  `IsDirectCopy` 同样不含 kINT8 —— 连"文件本来就是 INT8、目标也是 INT8"这条零拷贝都没开。

建图侧：`AddWeightConstant` 调 `weights.GetWeight(name, dtype, &bytes)` 成功后
`addConstant(dims, Weights{dtype, data, actual})`（`src/core/gpt2_model_builder.cpp:74-110`）。
因此 `weight_dtype = kINT8` 时**每个权重都会在取数处失败**（日志会打成
`GPT-2 weight not found`，但真实原因是被取数路径拒绝）。

图上没有任何量化节点：全仓库对 `addDequantize` / `IDequantizeLayer` / `setDynamicRange`
的引用为 0（`rg` 复核）。

**结论（现状陈述）**：`Precision::INT8` 目前是"能枚举、不能建图、不能端到端"。

### 3. GPT-2 权重的布局与来源

- 转换脚本**不做任何重排**：`convert()` 直接 `save_file(tensor_dict, ...)`，
  全文件无 transpose / permute / reshape（`tools/convert/hf_to_mini_trt_llm.py`）。
- 布局作为显式契约写进产物：`_build_gpt2_native_config` 输出
  `"source": {"conv1d_layout": "in_out", ...}`（同文件 `:153-186`）——HF GPT-2 的 Conv1D
  权重本来就是 `[in, out]`。
- 建图按 rank-3 声明 `[1, in, out]`：`c_attn.weight → Dims3(1, hidden, 3*hidden)` 等
  （`gpt2_model_builder.cpp:546-562`）；`AddLinear` 用 `kNONE × kNONE` 的
  `addMatrixMultiply`（`:208-222`），**不做转置**。
- `lm_head` 与 `wte` 绑定时没有独立常量：靠 `AddTranspose(wte)` 拿到 `[out, in]`
  再 reshape 成 `[1, hidden, vocab]`（`gpt2_model_builder.cpp:912-925`）。
- 资产现状：`models/gpt2/` 目前只有 `config.json`；548 MB 的 `model.safetensors` 由转换脚本
  生成、被 `.gitignore` 忽略（`docs/PROGRESS.md` §6.5）。

### 4. 视觉模型侧的 INT8 闭环（可复用的部分，且在另一条路径上）

- 路线：Python 侧**把权重预先量化成 int8 initializer**，图上只保留 `DequantizeLinear`
  （`tools/convert/quantize_resnet18.py:297-343`）。
- 自证：`detailed_profiling` 打开后，用
  `IEngineInspector::getLayerInformation(i, kONELINE)` 读每层文本，统计
  `"Format/Datatype: Int8"` 的层数与 `i8i8` tactic 数
  （`tests/test_resnet18_int8.cpp:166-200` 的 `InspectEngine`，用例 `IsActuallyInt8`）。
  该处注释明确写着：**没有 `[I8]` 这类标签**，按标签判会把"确实跑了 INT8"误判成"没跑"。
- 这条路线走 ONNX 解析路径，不是原生建图；配套的来源自检 / 饱和比例统计 / 判据分层见
  `docs/PROGRESS.md` §3.0j 与 `docs/TROUBLESHOOTING.md` #46。

### 5. ONNX 路径接不进运行时

- `BuildFromOnnx` 校验 I/O 名（`OnnxIoContractFor(architecture)`），非 cnn 时只挂
  **prefill 一组 profile**，图上没有 K/V 输入（`src/core/builder.cpp:568-610`）。
- ONNX 侧 `input_ids` 是 INT64，原生侧是 INT32 的 `input_ids` + `position_ids`
  （`docs/future_iterations.md` §10.1 / TS-017）。
- 因此 ONNX 路径承担不了 decode 与端到端生成。

### 6. 运行时的精度契约（REQ-016 落地后的现状）

- `LLMRunner::Config::is_half` 是 bool（`include/mini_trt_llm/core/llm_runner.hpp:55`）。
  构造期 `expected = is_half ? kHALF : kFLOAT`，decode 的 `key_cache_0` 必须与之一致，
  否则拒绝启动（`src/core/llm_runner.cpp:99-121`）；报错文案只区分 kHALF 与"其他"
  （`:110-113`，非 kHALF 一律印成 FP32）。
- 引擎边界精度逐个查询并记录：prefill / decode 各自的 K/V 与 logits（`:124-140`）；
  两侧 K/V 精度不一致 → 拒绝启动。
- cache 宽度是**两个 bool**：`PagedKVCache::Config::is_half`（cache 元素）+
  `source_is_half`（引擎导出的 K/V）（`include/mini_trt_llm/kv_cache/paged_kv_cache.hpp:36-40`）；
  写入内核按四种"源 / 目标"组合显式分发（`src/kv_cache/paged_kv_cache_kernels.cu:175-205`）。
- "元素宽度 → 字节数"**不是一个函数点**：`ElementSize(bool)` 在
  `src/core/llm_runner.cpp:24` 与 `src/kv_cache/paged_kv_cache.cpp:14` 各有一份。
- REQ-016 之后同一段入口还承载批量契约：padded / packed mixed 两套 prefill 图
  （`BuildOptions::packed_mixed`、`Builder::Config::packed_mixed_prefill`）、活跃批行号与
  写回 `rows` 映射（`paged_kv_cache.hpp` 的公开入口说明；设计口径见
  `docs/dev/REQ-016-continuous-batching/design.md` 的 D11 / D13 / D14）。

### 7. 引擎身份与缓存指纹的覆盖面

- 指纹字段：stage / precision / source_kind / graph_version / TRT 版本 / CUDA runtime 版本 /
  源文件身份（`config.json` + `model.safetensors`，或 ONNX 图）/ numeric_params（profile 区间、
  `n_positions`）/ flags（`src/core/builder.cpp` 的 `MakeFingerprintInputs`）。
- 建图代码**不进指纹**：`kEngineGraphVersion` / `kPackedPrefillGraphVersion` 是手工代次，
  改图必须 +1（同文件顶部注释）。
- 现状事实：**只有落在 `source_files` 里的产物**才会让引擎失效重建。

## Module Structure

| 模块 | 与量化的关系 | 现状 |
|---|---|---|
| 精度枚举与映射 | 档位 → TRT dtype | 三档齐全 |
| 构建配置（flag） | 整网倾向 | 只有 FP16 设 flag；INT8 无 flag |
| 权重加载器 | safetensors → 目标类型 | **目标只支持 FP32 / FP16**；INT8 取数被拒 |
| 语言模型构建器 | 权重 → 常量层 | 无量化节点；声明形状为 rank-3 `[1, in, out]` |
| ONNX 构建路径 | 解析 + 建引擎 | 只挂 prefill profile；无 K/V 输入 |
| 注意力插件 / 分页 cache | 元素宽度 | 两个 bool（2 / 4 字节）；内核四组合分发 |
| 运行时 | cache 精度校验与缓冲分配 | 契约按 `is_half` 写死为两档 |
| 引擎检查器 | 读逐层精度 | 已具备（视觉侧在用，需 `detailed_profiling`） |
| Python 量化工具链 | 图量化 + 判据 + 探针 | 已有，**针对视觉模型的 ONNX 图** |
| Python 权重转换 | HF → safetensors + config | 已有；无任何量化能力 |

## Data Flow

现有语言模型路径（与量化相关的部分）：

```text
safetensors（FP32 / FP16 / BF16）
   ↓ WeightLoader::GetWeight(name, weight_dtype, &bytes)
   ↓ （目标不是 FP32 / FP16 → 取数直接失败）
构建期：addConstant(dims, Weights{dtype, data, count})
   ↓
网络（无 Q / DQ 节点）→ builder（只有 FP16 才有 flag）→ 引擎
   ↓
运行时：getTensorDataType 查询边界精度 → 按实际精度分配缓冲 → 前向 → 采样
```

## Runtime Flow

1. 构造期：由 `is_half` 推出 `expected`；decode 的 `key_cache_0` 必须等于它，否则拒绝启动。
2. 逐个查询 prefill / decode 的 K/V 与 logits 精度并记录；两侧 K/V 不一致 → 拒绝启动。
3. cache 建 `is_half` + `source_is_half`；宽度换算按 bool 取 2 / 4 字节。
4. 写入 K/V 时按"源精度 → cache 精度"四组合分发；同一份逻辑也服务 REQ-016 的批量写回。
5. 采样按 logits 的**声明**精度读；`logits_half` 是运行期查询值。

## Relevant Code Path

| 位置 | 与本次相关的行为 |
|---|---|
| `include/mini_trt_llm/core/precision.hpp` / `src/core/precision.cpp` | 三档枚举与 dtype 映射 |
| `src/core/builder.cpp:198-212` | flag 设置（仅 FP16）；INT8 无 flag 的依据 |
| `src/core/builder.cpp` 的 `MakeFingerprintInputs` | 指纹字段与源文件清单 |
| `src/core/builder.cpp:568-610` | ONNX 路径的 I/O 契约与 profile |
| `include/mini_trt_llm/core/imodel_builder.hpp` | `BuildOptions::weight_dtype` |
| `src/core/weight_loader.cpp` | trt 名 → source key → 取数 |
| `src/utils/safetensors_loader.cpp` 的 `GetConvertedData` / `IsDirectCopy` | 取数与转换；**INT8 目标被拒** |
| `src/core/gpt2_model_builder.cpp:74-110` | 权重 → 常量层（含形状 / 数量校验） |
| `src/core/gpt2_model_builder.cpp:208-222, 546-562, 912-925` | Linear 布局、`[1,in,out]` 声明、wte / lm_head 绑定 |
| `src/core/llm_runner.cpp:96-140` | 边界精度查询与拒绝启动 |
| `src/core/llm_runner.cpp:165-180` | cache 的两个 bool 与 `max_batch` |
| `include/mini_trt_llm/kv_cache/paged_kv_cache.hpp:30-45` | cache 精度的表示 |
| `src/kv_cache/paged_kv_cache_kernels.cu:170-210` | 源 / 目标四种组合 |
| `tests/test_resnet18_int8.cpp:166-200` | 逐层精度自证的现成实现 |
| `tools/convert/quantize_resnet18.py:297-343` | 视觉侧预量化 + 只留 DQ 的先例 |
| `tools/convert/hf_to_mini_trt_llm.py` | GPT-2 权重与 config 产物（无重排、无量化） |

## Existing Limitation

1. 语言模型没有量化节点；`Precision::INT8` 无法建图。
2. **INT8 权重没有来源**：取数路径拒绝非 FP32 / FP16 目标，连"文件即 INT8、目标也 INT8"的
   零拷贝都没开。
3. ONNX 路径不是一等公民：没有 decode 图与 K/V 输入，接不进运行时。
4. 注意力插件与 cache 只表达"2 字节 / 4 字节"，无法表达第三种宽度。
5. 运行时精度契约是 bool（`is_half`），报错文案也只区分两档。
6. 量化工具链只覆盖视觉模型的 ONNX 图；GPT-2 的命名（多层、`[in,out]` Conv1D、wte / lm_head
   绑定）与它不同，不能直接套。
7. 引擎指纹只看文件身份：**新增的量化产物若不在 `source_files` 里，改了它不会触发重建**。
8. 当前环境（本机沙箱）无编译器 / 无 GPU；`models/gpt2/` 目前没有权重文件。

## Extension Point

- **权重取数**：`GetConvertedData` / `IsDirectCopy` 是"能不能拿到 INT8 权重"的唯一闸门。
- **量化产物**：`tools/convert/` 与 `models/gpt2/` 是 Python 侧产物的落点；`config.json` 的
  `source` 段已有"把布局约定写进产物"的先例可循。
- **建图**：`AddWeightConstant` 与 `AddLinear` 之间是插入量化 / 反量化语义的位置；
  `AddFloatConstant` 已有"先建 FP32 常量再显式 Cast"的处理范式可参照。
- **精度自证**：`detailed_profiling` + `IEngineInspector` 是现成护栏。
- **引擎身份**：`MakeFingerprintInputs` 的 `source_files` 与 `graph_version` 是"新增产物
  必须被看见"的两处。
- **运行时**：`is_half` 与 `source_is_half` 是 cache 宽度扩展的入口（属另一里程碑）。

## Terminology

| 模糊名词（出自 requirement.md） | 本项目的可验证定义 | 怎样算没做到 |
|---|---|---|
| INT8 权重（weight-only） | 权重以 int8 常量进入网络，激活与边界张量仍是 FP32 / FP16 | 权重仍是 FP32 / FP16 常量；或激活也被量化 |
| 自证低精度 | 从引擎（`detailed_profiling`）读出的层信息里，量化层带 `Format/Datatype: Int8`，且层数与清单一致 | 只凭"传了 INT8 配置"或构建成功就宣称在跑 INT8 |
| 逐层精度可读 | 引擎 ONELINE 层信息里能定位到量化层及其 dtype 文本 | 引擎是 `kLAYER_NAMES_ONLY`，读不出 dtype |
| 量化对象与 scale 同源 | 算 scale 的张量与写进图的 int8 张量是同一份（同 key、同布局、可逐字节核对） | 用另一份（例如没做同一变换的）张量算 scale |
| per-tensor | 每张权重一个标量 scale | scale 数 ≠ 1 |
| per-channel | 每张权重沿指定轴一组 scale，且轴与建图声明的形状对应 | 轴与 `[1, in, out]` 声明不一致，或 scale 数 ≠ 该轴长度 |
| 逐 token 一致率 | 同 prompt、同采样策略下逐位置 token id 相等的比例，且必须同时报样本量 | 只报"生成结果看起来对" |
| logits 相对偏差 | 同输入下 INT8 与 FP32 logits 的相对差；须同一处给出比较对象的形状 / 布局与绝对差 | 只给相对差，或只给单一极大值 |
| 数值判据有出处 | 阈值旁写明它来自哪次实测 / 哪个分位 / 哪份文档 | 复用视觉模型阈值，或"看着定" |
| K/V 缓存元素宽度 | cache 单个元素的字节数（当前 2 或 4） | 用"精度档位"代替元素宽度描述 cache |

## 文档与现状的矛盾（AGENTS.md §5 第 4 条要求的反向查）

本轮重做分析时逐条核对旧 `analysis.md`，结果与处置：

| 原表述 | 与源码 / 现状的冲突 | 处置 |
|---|---|---|
| "权重加载支持……包括 INT8 的映射已经存在" | 只有枚举映射；取数路径对非 FP32 / FP16 目标直接失败 | 改写为"没有 INT8 权重的来源"，列入 Existing Limitation 第 2 条 |
| 缺 `## Terminology` 一节 | P1 要求该节，且规定未定义的模糊名词不得进入 design | 本轮补上（见上表） |
| 运行时"精度契约写死为 FP32 / FP16" | 结论成立，但落点已随 REQ-016 变动（bool + 两处 `ElementSize` + 四组合内核分发） | 按现状改写为 Current Architecture §6 |
| 只提"两张图（prefill / decode）" | REQ-016 之后同一入口另有 padded / packed mixed 两套 prefill 图与活跃批行号契约 | 补进 Current Architecture §6 的末条 |

（旧 `design.md` 缺 `## Requirement Coverage` 一节的处置记在 `design.md` 的修正记录里。）
