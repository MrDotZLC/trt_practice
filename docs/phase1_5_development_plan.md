# Phase 1.5 开发计划：全流程测试基建与收尾

> **定位**：Phase 1 已交付算子层（RMSNorm / RoPE / PagedAttention / Sampler）并通过真机验证，
> 但「模型加载 → builder 分发 → Plugin 挂载 → engine 构建/反序列化 → 推理 → 采样」这条
> **真实使用路径**从未被自动化验证过。Phase 1.5 补的是这一段，不是新功能开发。
>
> **为什么单独立一个阶段**：这段工作既不属于 Phase 1（算子已交付），也不能塞进 Phase 2
> （GPT-2 的构建要依赖它产出的防线与基建）。立成 1.5 是为了让归属清晰，避免又一次
> 「Phase 0 宣布完成、实际有漏项」的情况。
>
> **用例设计细节见** `docs/phase1_test_plan.md`（E1–E4 的设计背景与链路图）；
> 本文档定义的是**任务、顺序、产出与验收**。

---

## 0. 执行结果（2026-09-24 完成，已通过真机验证）

| ID | 状态 | 实际产出 |
|---|---|---|
| P1.5-0 | ✅ 完成 | 修复 `SafetensorsLoader` 转换路径的 3 个缺陷 + 缓存按 key 隔离 + 映射注释，详见 `docs/TROUBLESHOOTING.md` #5 |
| P1.5-1 | ✅ 完成 | `tests/e2e_safetensors_writer.{hpp,cpp}` |
| P1.5-2 | ✅ 完成 | `tests/test_e2e_error_paths.cpp`：5 条用例（**3 条可进 CI**，2 条需 GPU） |
| P1.5-3 | ✅ 完成 | `tests/test_e2e_single_op.cpp`：4 条（RMSNorm / RoPE / PagedAttention / Sampler） |
| P1.5-4 | ✅ 完成 | `tests/test_e2e_mini_decoder.cpp`：多算子全链路串联 + 缺失权重用例。FP16 改为 L1 分层覆盖（见 §0.1 / §0.3） |
| P1.5-5 | ✅ 完成 | `AddCvOptimizationProfile` / `AddLlmOptimizationProfiles` 实现并接入；`Engine::SetOptimizationProfile` |
| P1.5-6 | ✅ 完成 | `tests/test_e2e_dynamic_shape.cpp`：3 条 |
| P1.5-7 | ✅ 完成 | 文档归位 4/4（含修正 `phase0_development_plan.md` 的验收判据） |

**真机验证记录**：E1 / E2 / E3 与 E4 的 2 条 GPU 用例、以及 FP16 覆盖用例
（`--gtest_filter='Fp16PathTest.*:E2eMiniDecoderTest.*'`，共 5 条）**全部通过**。
过程中共修复 4 个产品缺陷（`docs/TROUBLESHOOTING.md` #5 / #6 / #8 / #9）与 1 处校验顺序问题（#7）。

### 0.1 P1.5-4 的定位（含一次结论修正）

**本阶段的定位**：在进入 Phase 2 的模型加载之前，把**组件与集成验证全部收口**——
插件、采样器、权重加载、动态 shape、端到端测试骨架都在 Phase 1.5 内完成并验证，
Phase 2 只负责"加载模型 + 接线"。

**一次结论修正（重要）**：此前曾把 E2 描述为"GPT-2 子图的缩微版"，**该类比是错的**。
对 `1_gpt2_onnx/gpt2.onnx` 的算子统计为：

```
LayerNormalization × 25    （2/block × 12 + 最终 1 层）
Tanh × 12                  （gelu_new，tanh 近似）
MatMul × 25 / Gemm × 48 / Softmax × 12
```

GPT-2 用的是 **LayerNorm + 学习式位置编码**，不含 RMSNorm、不含 RoPE。因此
`RMSNorm → QKV → RoPE → PagedAttention → RMSNorm → LM Head` 是一条 **LLaMA 风格**的组合，
不是 GPT-2 的结构。

**那为什么仍然要覆盖它**：因为 Phase 1.5 的目标是"把组件写完并验证能组合"，
而不是"预演某个具体模型"。这条链路的作用是验证**多算子相邻时的形状 / dtype / 布局契约**，
以及多权重经真实路径取用的正确性——这些契约与模型无关，是插件本身的公共性质。
至于"某个模型该用哪些算子"，属于语义选择（RMSNorm ≠ LayerNorm、RoPE 只是位置编码的一种），
由具体 `IModelBuilder` 决定，不由插件决定。

**交付内容**：

- 完整链路 `RMSNorm → QKV(MatMul) → Slice → Reshape → RoPE → PagedAttention(decode) → Reshape → RMSNorm → LM Head → Top-K Sampler`；
- 四个权重全部经 `weight_map` 从 safetensors 取用，并以 BF16 存储（P1.5-0 缺陷的触发条件）；
- 配套用例：`FullChainMatchesIndependentCpuReference`（与独立 CPU 参考对比 + k=1 采样等于 argmax）、
  `RejectsMissingWeightFromMap`（`weight_map` 指向不存在的 key 时必须失败）。

**关于链路的 FP16 变体**：原计划在链路上再跑一遍 FP16。实际改为**分层覆盖**——
链路的职责是验证算子间的形状 / dtype / 布局契约，这些契约与计算精度无关；
而 FP16 分支已在 L1 逐算子覆盖（见 §0.3）。在链路上再验一遍属于重复投入，
且需要把整个 builder 按 dtype 参数化，维护成本不低。若后续 Phase 2 的 GPT-2 在 FP16 下暴露
链式精度问题，再补也不迟。

**参考实现**：用 C++ 双精度逐算子串联实现，不复用被测实现的公式推导；
不引入 `scripts/ref_mini_block.py`——该脚本的价值是把参考实现与 HF 算子绑定，
而本链路的算子约定（half-split RoPE、online softmax attention）已由
`scripts/ref_rope.py` 与独立 CPU 参考分别验证过，再加一层 Python 不增加独立性。

### 0.2 P1.5-7 完成情况

4 项全部完成：

1. `future_iterations.md` 新增 §9，迁入三项有意延后的能力；
2. `PROGRESS.md` §6 收敛为单入口，不再重复罗列条目；
3. `phase1_test_plan.md` 改为引用式，进度归属以本计划为准；
4. `phase0_development_plan.md` §4 改写为可执行判据，并补上 Phase 0 遗漏的
   「Optimization profile 能力可用」一条（详见该文档 §4.1 / §4.2）。

### 0.3 一并收口的覆盖缺口

`docs/phase1_test_plan.md` §8 列的缺口，凡属于「Phase 2 建模前应具备的组件能力」，本次一并处理：

| 缺口 | 处理 |
|---|---|
| RoPE / PagedAttention / Sampler 缺 FP16 覆盖 | ✅ 新增 `tests/test_fp16_paths.cpp`：RoPE（同时覆盖 **GQA + batch>1 + 部分旋转**）、PagedAttention、Greedy Sampler |
| RoPE 缺 GQA 与 batch>1 数值用例 | ✅ 由上面的 FP16 用例一并覆盖 |
| RoPE / PagedAttention 缺 L2 集成 | ✅ 由 E2 完整链路覆盖（两者都在链路内） |
| Top-K / Top-P 的 FP16 覆盖 | ⬜ 未做——Greedy 已覆盖 FP16；Top-K/Top-P 的 FP16 分支同属采样器同一处 dtype 分派，风险低，留待 Phase 2 真实采样路径一并验证 |
| 采样器分布数据固化 | 仍留在 `future_iterations.md` §9.3（属增强，非组件能力） |

---

## 1. 阶段目标

1. 让「配置 + 权重 → engine → 推理」这条链路有**可自动运行**的端到端测试。
2. 把 Phase 1 算子按**真实调用方式**（经 `IModelBuilder` + `WeightLoader` + `EngineBuilder`）串起来验证，
   而不是测试里手工 `addPluginV3`。
3. 补齐 Phase 2 的硬前置：动态 shape 的 optimization profile。
4. 修掉本次盘点发现的两个会污染 Phase 2 的底层缺陷（见 P1.5-0）。

**非目标**：`LLMRunner` 的多步 decode 循环（属 Phase 2）、PagedAttention Prefill、Sampler 性能优化。

---

## 2. 任务总览

| ID | 任务 | 类型 | 依赖 | 预估 |
|---|---|---|---|---|
| **P1.5-0** | 修 `WeightLoader` 的权重指针生命周期与映射注释 | 缺陷修复 | 无 | 0.5 天 |
| **P1.5-1** | G2：Safetensors 写入 helper（测试基建） | 基建 | 无 | 0.5 天 |
| **P1.5-2** | E4：错误路径用例（host 侧，进 CI） | 测试 | 无（与 P1.5-1 并行） | 0.5 天 |
| **P1.5-3** | E1：单算子最小模型闭环（4 条） | 测试 | P1.5-0, P1.5-1 | 1 天 |
| **P1.5-4** | E2：mini decoder step 串联（含参考脚本） | 测试 | P1.5-3 | 1.5 天 |
| **P1.5-5** | G1：Optimization profile 实现 | 能力补齐 | P1.5-4 | 1 天 |
| **P1.5-6** | E3：Prefill / Decode 双 profile 测试 | 测试 | P1.5-5 | 0.5 天 |
| **P1.5-7** | 文档归位（见 §3.8） | 收尾 | 全部 | 0.5 天 |

**并行关系**：P1.5-0 / P1.5-1 / P1.5-2 三者互不依赖，可并行开工；P1.5-3 起严格串行。

---

## 3. 任务详细说明

### 3.1 P1.5-0：修 `WeightLoader` 的权重指针生命周期与映射注释

**为什么排在最前**：E2 需要 4 个以上权重、Phase 2 的 GPT-2 需要上百个。这个缺陷会让
`addConstant` 拿到的指针指向已被覆盖的数据，**静默构建出错误网络**——比崩溃更难查。

**缺陷 1：共享转换缓冲区导致多权重互相覆盖**

- 现状：`SafetensorsLoader::GetConvertedData` 在源 dtype 与目标 dtype 不一致时，把结果写进
  **同一个成员 `conversion_buffer_`** 并返回其指针；头文件已注明"会被下一次 `GetConvertedData`
  调用覆盖"。而 `nvinfer1::Weights` 只持有裸指针，要到 `buildSerializedNetwork` 才真正读数据。
- 后果：`GetWeight(w1)` → `addConstant(w1)` → `GetWeight(w2)` 之后，w1 的常量层指向的数据已变成 w2。
- 触发条件：**仅当源权重需要转换**（目前是 BF16 → FP32/FP16）。FP32/FP16 源且目标 dtype 相同时
  走 `GetRawData`、返回稳定指针，因此这个坑不会在日常小样例里暴露。
- **修法**：`WeightLoader` 内部按 key 缓存每一份转换结果，使返回指针在 `WeightLoader` 生命周期内稳定；
  同一 key 重复请求命中缓存。
- **验收**：新增单测——构造含两个 BF16 权重的最小 safetensors，连续取两次后断言两个指针不相等、
  且各自内容与源数据一致。该用例是纯加载器行为，**不需要 GPU**，可进 CI。

**缺陷 2：FP16 源无法上转到 FP32**

- 现状：`GetConvertedData` 只实现了 BF16 → FP32/FP16；FP16 源请求 `kFLOAT` 会直接报错返回 nullptr。
- 影响：用 FP32 精度构建一个 FP16 权重的模型时会失败——Phase 2 排查精度问题时很可能撞上。
- **修法**：补 FP16 → FP32 的上转，顺带补齐 FP32 → FP16，让「F32/F16/BF16 源 × F32/F16 目标」
  六种组合都有确定行为。
- **验收**：参数化单测覆盖 dtype 组合矩阵，断言转换值与 CPU 参考一致（同样不需要 GPU）。

**缺陷 3：`weight_loader.hpp` 的映射方向注释写反**

- 实现是 `weight_map_[trt_name] = source_key`（TRT 名 → safetensors key），
  但头文件注释写的是 `source_key -> trt_name`；`model_config.hpp` 的注释才是对的。
- **修法**：只改注释并在 `SetWeightMap` 上补一句说明；**不改实现**（实现与 config 语义一致）。

---

### 3.2 P1.5-1：Safetensors 写入 helper（G2）

**目标**：让测试能在临时目录里自包含地造出 `model.safetensors`，不提交二进制夹具、不依赖 Python。

**新增文件**：`mini_trt_llm/tests/e2e_safetensors_writer.hpp` / `.cpp`

**实现要点**：

- 用第三方库已有的写入 API（`safetensors.hh` 的 `safetensors::save_to_file`，用法见同目录
  `serialize-example.cc`）：先把数据追加进 `safetensors_t::storage`，再逐个填 `tensor_t` 的
  `dtype` / `shape` / `data_offsets`。
- **实现宏 `SAFETENSORS_CPP_IMPLEMENTATION` 只允许在一个 TU 里定义**。本项目静态库已由
  `third_party/safetensors-cpp/safetensors.cc` 提供实现，因此 helper 只 include 头文件、
  不得定义该宏，否则重复定义符号。
- 接口设计成「给定 `{张量名 → {dtype, shape, 数据}}`，写出一份合法文件」；数据用确定性伪随机
  （复用 `tests/test_reference.hpp` 的生成方式），保证与参考实现输入一致。
- 放在 `tests/` 而非产品代码：写入能力**只有测试需要**，放进 `include/` 会污染产品 API。
  `tests/CMakeLists.txt` 的 `*.cpp` glob 会自动收录，无需改构建脚本。

**验收标准**：

- helper 写出的文件能被 `SafetensorsLoader::LoadFromFile` 读回，`HasTensor` / `GetTensorInfo` /
  `GetRawData` 的结果与写入前一致（host 侧用例，可进 CI）。
- 至少覆盖 FP32 与 BF16 两种 dtype（BF16 用于驱动 P1.5-0 的缓存用例）。

---

### 3.3 P1.5-2：错误路径用例（E4）

**目标**：确认失败是**可控失败**，而不是崩溃、产出半成品 engine、或静默成功。
这组是 Phase 1.5 里唯一完全不需要 GPU 的端到端用例，先拿到它就有了 CI 防线。

**新增文件**：`mini_trt_llm/tests/test_e2e_error_paths.cpp`

| 用例 | 场景 | 期望 |
|---|---|---|
| `MissingConfigFails` | `model_dir` 下无 `config.json` | 返回 false，且**不产出** engine 文件 |
| `UnregisteredModelTypeFails` | `config.json` 的 `model_type` 未注册 | 返回 false，日志含 `No registered builder for model type` |
| `MissingWeightsFails` | 目录下缺 `model.safetensors` | 返回 false |
| `UnknownWeightMapKeyFails` | `weight_map` 指向不存在的 source key | builder 取权重失败 → 返回 false（不得建出错误 engine） |
| `InvalidPluginConfigFails` | 例如 PagedAttention 未配 `block_size` | 构建失败，错误信息指向具体插件 |

**实现要点**：

- 用 `tests/e2e_fixture.hpp` 的临时目录 helper（`mkdtemp` + 测试结束清理），避免残留污染。
- 「不产出 engine 文件」要显式断言文件不存在，而不是只看返回值——半成品文件会让下游误判。
- 第四条要求测试 builder 在 `GetWeight` 返回 nullptr 时**主动失败**，这本就是 Phase 2 真实 builder
  应有的行为，测试同时把这条约定固化下来。

**验收标准**：五条用例在沙箱内全绿，且都不因缺少 GPU 而跳过。

---

### 3.4 P1.5-3：单算子最小模型闭环（E1）

**目标**：验证每个 Phase 1 算子能通过**真实使用路径**被调用——权重从 safetensors 取出、
经 `weight_map` 映射、挂成 Plugin 的常量输入。

**新增文件**：

- `mini_trt_llm/tests/e2e_fixture.hpp` / `.cpp`：临时目录 + `config.json` + `model.safetensors` 组装 helper
- `mini_trt_llm/tests/test_e2e_single_op.cpp`：4 条用例 + 4 个测试专用 `IModelBuilder`

| 用例 | Builder 注册名 | 网络结构 | 关键验证点 |
|---|---|---|---|
| `RmsNormEndToEnd` | `e2e_rmsnorm` | input + 常量 weight → RmsNormPlugin | `weight_map` 命中；`eps` / `hidden_size` 经 engine 序列化往返后仍正确 |
| `RoPEndToEnd` | `e2e_rope` | query + key + position_ids → RoPEPlugin（**无权重**） | `position_ids` 作为网络输入；`rotary_dim < head_size` 时尾部保持原值（TROUBLESHOOTING #4 的真实路径回归） |
| `PagedAttentionEndToEnd` | `e2e_paged_attn` | q + k_cache + v_cache + block_tables + context_lens → 插件 | 跨多个物理块的块表寻址 |
| `SamplerEndToEnd` | `e2e_sampler` | input + 常量权重 → MatMul → logits → `LaunchTopKSampler` | 算子**之外**的衔接段：engine 输出 logits 直接喂采样器，固定 seed 可复现 |

**实现要点**：

- 测试 builder 在 `Build()` 里用 `weights.GetWeight(trt_name, dtype, &bytes)` 取权重并挂 `addConstant`
  ——这正是 P1.5-0 必须先行修复的原因。
- 先用**全静态 shape + FP32**，把「链路通不通」和「精度对不对」分开判断；FP16 变体放到 P1.5-4 之后。
- 采样器不是 Plugin，因此 E1.4 验证「engine 输出 → 采样器」这一段，不走 `addPluginV3`。
- 输入输出张量名用 `setName` 固定，不依赖 TRT 自动命名（`GetTensorNames()` 返回空，不能靠它枚举）。

**验收标准**：4 条用例在真机全绿；输出与 `tests/test_reference.hpp` 的 CPU 参考对比，
FP32 绝对误差 < `1e-4`；沙箱内 `GTEST_SKIP`。

---

### 3.5 P1.5-4：mini decoder step 串联（E2）

**目标**：真正的「全流程」——一条用例同时验证算子之间的**接口契约**（shape / dtype / 布局）
与整体数值，也是 Phase 2 之前的最后一道集成防线。

**新增文件**：`mini_trt_llm/tests/test_e2e_mini_decoder.cpp`、`scripts/ref_mini_block.py`

**网络结构**（解码一步，全静态 shape）：

```
hidden[B,1,H] ─► RMSNorm ─► QKV 线性层(MatMul+Add) ─► RoPE(q,k) ─► PagedAttention(decode)
key_cache / value_cache（网络输入）───────────────────────────────────┘
                                            │
                              RMSNorm ◄─────┘ ─► LM Head(MatMul) ─► logits[B,V] ─► Top-K Sampler ─► token_ids
```

**实现要点**：

- 权重数量 ≥ 4（两处 RMSNorm + QKV + LM Head），全部经 `weight_map` 从 safetensors 取——
  这条用例是 P1.5-0 修复的主要受益者，也是它的回归防线。
- **参考实现必须独立**：`ref_mini_block.py` 用 PyTorch/HF 算子组装，不复用我们的公式推导。
  理由是串联场景下若参考与被测犯同一个错（例如都搞错 QKV 布局），逐元素对比会双双通过。
  该原则已在 `scripts/ref_rope.py` 上验证有效（与 HF 差异为 0）。
- 参考数据落盘为 float32 `.bin`，C++ 侧载入比对；不要在测试里内嵌大数组。
- 数值判据：FP32 绝对误差 < `1e-4`（链路比单算子长，故比 E1 的 `1e-5` 放宽一档）；
  另补一条 FP16 变体，按 D4（相对误差 < `1e-3` + 小值绝对误差 Guardrail）判定。

**验收标准**：真机全绿；logits 与 Python 参考在阈值内；采样结果的候选集合与 Python 一致。

---

### 3.6 P1.5-5：Optimization profile 实现（G1）

**目标**：打通动态 shape。这是 Phase 2 的**硬前置**——Prefill/Decode 双 profile 是设计文档里
已确认的前提，不是可选优化。

**现状**：`EngineBuilder` 声明了 `AddCvOptimizationProfile` 与 `AddLlmOptimizationProfiles`，
但 `src/core/builder.cpp` 里**没有任何定义**，`BuildFromConfig` 中只有一句
`// TODO: 根据 architecture 添加 optimization profile`；`Config` 里的 CV / Prefill / Decode
三组 `min/opt/max_*` 字段目前完全未被使用。

**实现要点**：

- 每个 profile 显式 `setDimensions` 输入张量的 min / opt / max；输入张量名从 `network->getInput(i)`
  按索引取，**不能**写死名字。
- 按 `model_config.architecture` 选择：`"cnn"` → CV 单 profile（只动态 batch）；
  `"decoder_only"` → Prefill + Decode 双 profile。
- 多 profile 时必须显式指定 profile 索引（`setOptimizationProfileAsync`），否则运行期可能拿到
  错的那组形状——这是 TRT 的常见坑，测试里要覆盖。
- profile 与输入张量数量不匹配、或某维 min > max 时要明确报错，不要静默跳过。
- 同时补 `Engine` 侧的 profile 选择能力（当前 `SetInputShape` 只设 shape，多 profile 下需要能切换）。

**验收标准**：E3 用例全绿；未配置 profile 但输入含动态维时，`BuildFromConfig` 给出可诊断的错误，
而不是抛 TRT 内部异常。

---

### 3.7 P1.5-6：Prefill / Decode 双 profile 测试（E3）

**新增文件**：`mini_trt_llm/tests/test_e2e_dynamic_shape.cpp`

| 用例 | 输入 | 期望 |
|---|---|---|
| `PrefillAcceptsVaryingSeqLen` | `[B, S]`，S 在 `[min_prefill_seq_len, max_prefill_seq_len]` 内取多个值 | 同一 engine 接受不同 S，输出正确 |
| `DecodeAcceptsVaryingBatch` | `[B, 1]`，B 在 `[min_decode_batch, max_decode_batch]` 内变化 | 同一 engine 接受不同 B |
| `OutOfRangeShapeFails` | 超出 profile 范围 | `SetInputShape` 失败并给出明确错误，不静默出错 |

**实现要点**：以 E2 的网络为基础换成动态输入维，不重复造网络；`opt` 形状也要各测一次，
因为 TRT 以 opt 为基准做 kernel 选择，只测 min/max 会漏掉主路径。

**验收标准**：真机全绿，每条用例断言输出数值而不只是"没崩"。

---

### 3.8 P1.5-7：文档归位

避免文档变成多个互相冲突的事实来源：

1. 把 `PROGRESS.md` §6 里的 **PagedAttention Prefill**、**Sampler 手写高性能 kernel**、
   **采样器分布数据固化** 三项迁入 `docs/future_iterations.md`——它们才是"有意延后"的能力。
2. `PROGRESS.md` §6 改为只指向下一阶段入口，不重复罗列条目。
3. 更新 `docs/phase1_test_plan.md`：E1–E4 的 phase 归属改为引用本计划，不再单独维护一份清单。
4. 修正 `phase0_development_plan.md` 的验收判据：原文以"文件存在"通过（`AddCvOptimizationProfile`
   等只声明未定义也算过），改为可执行判据（"该方法有定义且被调用"），避免同类漏项重演。

---

## 4. 文件清单

```text
mini_trt_llm/
├── include/mini_trt_llm/core/
│   └── weight_loader.hpp               # P1.5-0：修注释 + 缓存接口
├── src/core/
│   ├── weight_loader.cpp               # P1.5-0：按 key 缓存转换结果
│   ├── builder.cpp                     # P1.5-5：实现并接入 optimization profile
│   └── engine.cpp                      # P1.5-5：多 profile 选择
└── tests/
    ├── e2e_safetensors_writer.hpp/.cpp # P1.5-1：写入 helper
    ├── e2e_fixture.hpp/.cpp            # P1.5-3：临时模型目录组装
    ├── test_e2e_error_paths.cpp        # P1.5-2：E4
    ├── test_e2e_single_op.cpp          # P1.5-3：E1
    ├── test_e2e_mini_decoder.cpp       # P1.5-4：E2
    └── test_e2e_dynamic_shape.cpp      # P1.5-6：E3

scripts/
└── ref_mini_block.py                   # P1.5-4：独立 PyTorch 参考
```

`mini_trt_llm/tests/CMakeLists.txt` 用 `*.cpp` glob，新增测试文件无需改构建脚本；
`include/` 与 `src/` 新增的 `.cu` 已被现有 glob 覆盖。

---

## 5. 验收标准（Phase 1.5 整体）

- [ ] P1.5-0 的三个缺陷各有针对性回归用例，且都不需要 GPU 即可运行。
- [ ] `ctest` 在沙箱内全绿，GPU 用例以 `GTEST_SKIP` 跳过、不产生噪声。
- [ ] E4 五条用例在沙箱内实际执行并通过（CI 防线建立）。
- [ ] E1 四条用例在真机通过，FP32 绝对误差 < `1e-4`。
- [ ] E2 的 logits 与 PyTorch 独立参考在阈值内，FP16 变体按 D4 判定。
- [ ] E3 三条用例在真机通过，且覆盖 opt 形状。
- [ ] 未配置 profile 而输入含动态维时，报可诊断错误。
- [ ] P1.5-7 的四项文档动作完成。

---

## 6. 注意事项与风险

1. **P1.5-0 不可后置**。E2 与 Phase 2 都会撞上共享转换缓冲区的问题，且症状是"静默错误"
   而非崩溃。若排期紧张要砍任务，也不能砍这一条。
2. **engine 构建耗时**。每个 E2E 用例都要跑一次 `buildSerializedNetwork`（数秒到数十秒）。
   同一 fixture 的多个断言应合并到一条用例，engine 文件按 `model_type + shape` 缓存复用。
3. **测试用 builder 必须与真实 builder 同构**。若为了省事绕开 `WeightLoader`，
   全流程测试就失去意义——必须真实走 `GetWeight` + `addConstant`。
4. **静态 shape 先行**。E1/E2 在 P1.5-5 之前全部用静态 shape，避免把 profile 未实现的问题
   混进"链路是否打通"的判断。
5. **`LLMRunner` 仍是 stub**。E2 只覆盖单步 decode，完整自回归循环留给 Phase 2；
   不要为了让 E2 "更像端到端"而顺手实现 runner，那属于 Phase 2 范围。

---

## 7. 与 Phase 2 的衔接

- P1.5-5（profile）是 Phase 2 的**第一块地基**：`GPT2ModelBuilder` 的 Prefill / Decode 双引擎
  与动态 batch 都直接依赖它。
- E2 的 mini decoder 是 GPT-2 的缩微版：`GPT2ModelBuilder` 可以直接复用它的
  「RMSNorm → QKV → RoPE → PagedAttention → LM Head」接线方式与测试骨架。
- P1.5-4 确立的「参考实现必须独立编写」原则，同样适用于 Phase 2 的 `ref_output.bin` 对比。

---

*文档版本：v1.0*  
*关联文档：`docs/phase1_test_plan.md`（用例设计）、`docs/PROGRESS.md`、`docs/TROUBLESHOOTING.md`*
