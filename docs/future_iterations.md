# 后续迭代计划

> 本文档记录 `mini_trt_llm` 中已明确延后、但未来需要实现的能力，防止遗忘。  
> 每一项标注预计优先级、大致阶段、关键依赖。**优先级定义与排序见 §0——章节顺序是主题分类，不是优先级。**
> **执行安排见 `docs/future_iterations_development_plan.md`（怎么做）与
> `docs/future_iterations_test_plan.md`（怎么验）**——本文件只保留"是什么 / 为什么 / 触发条件 / 验收判据"。

---

## 0. 优先级定义与排序规则（优先级的唯一排序来源）

> **先纠一个容易误读的地方：章节顺序是主题分类，不是优先级。**
> 章节内部原本**恰好**大致从高到低写（本次重定前：`§2` 是 P1→P2、`§5` 是 P1→P2→P3），
> 所以整篇读起来像排期表；但跨章节从来不单调——`§1.3`(P3) 排在 `§1.6`(P2) 前面，
> `§5.1`(P1) 排在 `§4.3`(P3) 后面。也就是说，"标题顺序 = 优先级"**看起来成立、实际不成立**。
> 排序请查 §0.1 的表。**章节编号保持稳定、不按优先级重排**：全仓有 **40 处**
> `docs/future_iterations.md §x.y` 形式的引用（2026-09-26 用 `rg` 统计，分布在 `PROGRESS.md`、
> `phase1_5_*` / `phase2_*` / `phase3_*` / `phase4_*` / `TROUBLESHOOTING.md` 等 10 份文档里），
> 重排编号会让这些引用全部指向错误章节。

**为什么用"触发驱动"而不是待办队列**：Phase 0~4 已全部完成、没有下一阶段（`docs/PROGRESS.md` §6），
所以不存在"当前该做哪个"的排期。优先级表达的是**触发条件成立时的相对顺序**，
不等于"现在就该动手"；判定只采用条目自己写的触发条件，与现状逐条核对。

| 级别 | 定义 |
|---|---|
| **P0** | 触发条件已成立，且**阻塞已交付能力** → 立刻做。**当前为空**（核对结果见 §0.1） |
| **P1** | 无外部前置（不联网 / 不需新硬件 / 不需新需求），且是"让已交付模型真正可用"的门槛 |
| **P2** | 触发条件明确，但有外部前置：联网 / 新数据 / 新硬件 / 先补测量 / 需求口径未定 |
| **P3** | 当前硬件（sm_75、单卡、无 Tensor Core）收益有限，或研究性 / 探索性 |
| **冻结** | 已被别的路线取代，或触发条件已被现状否证。保留备查、不排期；解冻需重新评估 |

> **文档归位规则（2026-09-26 作者决定）**：后续迭代中**每个事项的开发计划与测试计划，一律并入
> `docs/future_iterations_development_plan.md` 与 `docs/future_iterations_test_plan.md`**，
> **不再单独新建 per-item 文件**（例：§9.2 的两份计划已并入 §10 / §9）。

### 0.1 优先级总表（按建议优先级排序）

| 建议 | 条目 | 原标签 | 触发条件 vs 现状（依据） | 开工前提 / 备注 |
|---|---|---|---|---|
| ~~P1~~ **已完成** | §5.1 BPE Tokenizer | P1 | **✅ 已交付（2026-09-26，批次 A1）**；交付内容与实测见 `PROGRESS.md` §3.0e、用例见 `future_iterations_test_plan.md` §2.1（15 条，含"文本 → prompt token"桥接用例）。原触发理由（能力门槛）：`LLMRunner` 只收/还 token id，`tokenizer_` 仅在构造时校验 → 目前没有"文本进 / 文本出"的端到端路径 | 纯 host 代码、可离线、可进 CI。**前置已核实（2026-09-26 更正）**：tokenizer 数据文件在本地 HF 缓存里（`~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d.../{vocab.json,merges.txt,tokenizer.json}`），且 `transformers 4.44.0` 可 `local_files_only=True` 离线加载并给出参考 token → **不需要联网**。（上一版这里写"本地没有、降为 P2"是**错的**：当时只查了 `models/gpt2/`，把"仓库里没有"当成了"机器上没有"） |
| ~~P1~~ **已完成** | §1.6 的**离线子项**（口径定义：验收集规格 / meta 字段 / 分层统计脚本） | P2（整条） | **✅ 已交付（2026-09-26，批次 A2）**：规格 `tools/validate/README.md`、脚本 `int8_eval.py`（3 项分层数学 + 7 道护栏自检）、ctest 项 `int8_eval_selftest`；C 批已用它在真机完成与 C++ 侧统计的口径交叉校验 | **下载验收集那一半仍是 P2**（需联网，见本表下一档） |
| **P2** | §1.2 LLM INT8 / INT4 | P2 | 触发 = 需要 LLM 低精度吞吐；未触发 | 前置依赖已核实：`PagedAttentionPlugin` 目前只接受 `kFLOAT` / `kHALF`（`paged_attention_plugin.cu:265`）→ 先要 INT8 KV cache；INT4 需先评估 sm_75 kernel 可行性 |
| **P2** | §1.6 验收集与绝对误差判据（整条） | P2 | 同上 | **联网下载**带真值标签的验收集，按 `AGENTS.md` §0.2 须先获批；采样须排除与 `calib_data` 重叠的图 |
| **P2** | §2.1 显存池替换 | **P1 → P2** | 触发 = "频繁分配有性能开销"；**没有量过**，且现行设计已避开 decode 循环内分配（`AGENTS.md` §3.A.3 本来就不允许） | 先量一次分配开销再动手，否则优化对象都没确定；`DeviceBuffer` 仍是直接 `cudaMalloc`（`memory_pool.cpp:46`） |
| **P2** | §2.3 Continuous Batching | P2 | 触发 = 服务化需求；未触发 | 与缺口 **G2-3**（`LLMRunner` 有意限定 `batch = 1`）耦合，要先扩 batch |
| **P2** | §2.4 CV 动态分辨率 | P2 | 触发 = CV 需要变分辨率输入；未触发（`AddCvOptimizationProfile` 目前显式拒绝） | 主体是 shape 配置，但会牵动 CV profile 与引擎缓存路径 |
| **P2** | §3.1 Encoder-Decoder | P2 | 触发 = 接 seq2seq 模型；未触发 | 架构接口已预留（`architecture = "encoder_decoder"`） |
| **P2** | §3.2 Vision Transformer | P2 | 触发 = 接 ViT；未触发 | 可复用现有 Transformer block 建图逻辑 |
| **P2** | §4.1 归一化 Plugin（**范围已缩小**） | **P1 → P2** | `LayerNormPlugin` 这一半**已被现状取代**：GPT-2 用 TRT 原生 `addNormalizationV2`（`gpt2_model_builder.cpp:245`），不需要自研 | 只剩 `GroupNorm` / `InstanceNorm`（CV 模型需要），等真接这类模型时再做 |
| **P2** | §4.2 激活函数 Plugin | P1 → P2 | 触发 = 接 LLaMA / GLM 系列；未触发 | `SiLU` / `SwiGLU` 是那些 FFN 的必需件；`GELU` 只在原生实现性能不足时才考虑 |
| **P2** | §6.1 转换工具增强 | P1 → P2 | 触发 = 出现新的权重来源 / 需要 ONNX 导出；未触发 | 按需扩，避免为不存在的来源先写代码 |
| **P2** | §6.2 ONNX 导出与 custom op | P2 | 触发 = 真做子图替换（§10.2）；未触发 | 是 §10.2 的前置件 |
| **P2** | §6.3 Nsight 一键 Profile Target | P2 | 触发 = 要按 **G6** 口径做可复现测量时；未触发 | 属测量基建：没有它，性能类结论只能靠单次点值（`phase3_test_plan.md` §3.1 已吃过一次） |
| **P2** | §6.4 CI / 自动化测试 | P2 | 触发 = 需要自动化回归；未触发 | **GitHub Actions 属联网操作，须先获批**；本地脚本部分不受限 |
| **P2** | §9.2 Sampler 高性能 kernel | P1 → P2 | 触发 = "`nsys` / `ncu` 定位到 sampler 占比显著"；**该 profile 至今没做过**（`PROGRESS.md` §3.0d 只记录了 `CVRunner` 的 benchmark）→ 触发条件未被验证 | 先做一次 decode 性能 profile，再决定要不要重写 sampler |
| **P2** | §9.3 采样器参考数据固化 | P2 | 触发 = 需要采样统计正确性回归；未触发 | 顺带覆盖 `p` 接近 1 的边界 |
| **P2** | §10.1 ONNX / 原生 I/O 契约统一 | P2 | 触发 = 要让 ONNX 路径接进 `LLMRunner`；未触发（Phase 3 D3 明确本阶段不做） | 先加 `Cast` 即可统一 dtype；`position_ids` 提升为输入是更大的改动 |
| **P3** | §1.5 per-channel 整网退化（P4-INT8-a） | P3 | 触发 = "需要更高 INT8 精度"；未触发（per-tensor 余量子集已 100%） | **触发即升 P1**；它是唯一**无外部前置**的立项条目（做法 / 验收 / 成本校准都已写好，前 3 步不联网）→ 若你想在无新需求时推进一件实事，它是首选 |
| **P3** | §1.3 BF16 原生计算 | P3 | 触发 = 迁到 Ampere+；硬件不支持 | 保持 P3，无需行动 |
| **P3** | §1.4 GPT-2 FP16 端到端 | **补标签** | 触发 = 真要 GPT-2 低精度推理；未触发 | sm_75 无 Tensor Core，收益有限；成本已按低→高列了三条路线 |
| **P3** | §2.2 FlashAttention / FlashDecoding | P2 → P3 | 触发 = 量出 attention 占比显著；未量过 | sm_75 **无官方 FlashAttention**，属 hand-tune 研究性工作，先要有基线 |
| **P3** | §3.3 多模态（CLIP / LLaVA） | P3 | 未触发 | 成本高，且依赖 §3.1 / §3.2 打底 |
| **P3** | §3.4 Audio（Whisper） | P3 | 未触发 | 依赖 §3.1 |
| **P3** | §4.3 MoE | P3 | 未触发 | 研究性 |
| **P3** | **SP-1**（`SentencePieceTokenizer` 未验证） | 新增（2026-09-26） | 触发 = 要用 SentencePiece 系模型；**未触发**。`BaseTokenizer` 的另一个实现，但零用例、无资产（`.model`/`.spm`）、无调用方 → 正确性从未被裁决 | **先造参考再动实现**（照 BPE 那套：golden + 来源 SHA256 + meta 自证 + 负例）；资产需联网取或作者提供 |
| **P3** | §5.2 Tiktoken Tokenizer | P2 → P3 | 触发 = 接 GPT-4 类模型；未触发 | 与 §5.1 相比，它没有当前模型支撑 |
| **P3** | §5.3 多模态 Prompt Template | P3 | 未触发 | 依赖 §3.3 |
| **P3** | §7.1 HTTP / gRPC 服务 | P3 | 未触发 | 依赖 §2.3（调度）与 `G2-3`（batch） |
| **P3** | §7.2 多 GPU / Tensor Parallel | P3 | 未触发 | 开发机为单卡，无验证环境 |
| **P3** | §10.2 ONNX 子图替换 | P3 | 触发 = 有明确性能目标；**前提取决于 G6** | 两次测量方向相反（±25% < 构建间噪声）→ 先建可复现测量方法，再谈收益 |
| **冻结** | §1.1 ResNet18 INT8 校准 | **P1 → 冻结** | **触发已消失**：ResNet18 的 INT8 已由 **Q/DQ 显式量化**交付（`PROGRESS.md` §3.0d）；且 TRT 10.12 起 `kINT8` / `IInt8Calibrator` 路线弃用 | `mini_trt_llm` 内**没有任何 calibrator 代码**（已核实），现行路线是 `tools/convert/quantize_resnet18.py`。保留备查：若将来要重拾 implicit calibration，须先重审（`TROUBLESHOOTING.md` #27） |
| **冻结** | §9.1 PagedAttention 的 Prefill 阶段 | **P1 → 冻结** | **触发条件已被现状否证**：原文说"Phase 2 若走单引擎就必须先补"，而 Phase 2 之后的 `LLMRunner` 构造收 **prefill + decode 两个引擎**（`llm_runner.hpp:56`），双引擎路径一直成立 | 保留备查：只有将来要合成单引擎时才需要补 Prefill kernel |

### 0.2 与旧标签的差异（本次重定，2026-09-26）

| 条目 | 旧 | 新 | 理由 |
|---|---|---|---|
| §1.1 ResNet18 INT8 校准 | P1 | **冻结** | 已被 Q/DQ 路线取代，且该 API 路线在 TRT 10.12 起弃用（详见 §0.1 行） |
| §1.4 GPT-2 FP16 端到端 | **（缺标签）** | P3 | 文档头声明"每一项标注优先级"，但它漏了 → 补齐；P3 依据是 sm_75 无 Tensor Core |
| §2.1 显存池 | P1 | P2 | P1 应留给"能力门槛"；本条是性能优化且收益未测 |
| §2.2 FlashAttention / FlashDecoding | P2 | P3 | 未量基线、无官方支持，属探索性 |
| §4.1 归一化 Plugin | P1 | P2（范围缩小） | `LayerNormPlugin` 被 TRT 原生层取代；只剩 CV 侧的 `GroupNorm` / `InstanceNorm` |
| §4.2 激活函数 Plugin / §6.1 转换工具 | P1 | P2 | 均为"接新模型时才需要"的按需件，不阻塞现行能力 |
| §9.1 PagedAttention Prefill | P1 | **冻结** | 触发条件已被双引擎现状否证 |
| §9.2 Sampler 高性能 kernel | P1 | P2 | 触发条件（profile 结果）从未被验证过，不能算"已触发" |
| §5.2 Tiktoken | P2 | P3 | 没有任何现行模型需要它 |
| §1.5 / §1.6 | P3 / P2 | 不变 | 分别保留"触发即升 P1" 与"离线子项升 P1"的口径，见 §0.1 |

**结论（2026-09-26 刷新）**：**P0 / P1 均为空** —— P1 的两条（§5.1 BPE Tokenizer、§1.6 的离线子项）
已经在批次 A 交付。下一档是 **P2**；P2 内部按"**前置可否立即满足 → 解锁广度 → 成本**"排序，
具体顺序见 **§0.3 执行序列**。
**每个条目内部的「优先级」行必须与 §0.1 一致**；不一致时以 §0.1 为准并当场改回一致。

### 0.3 执行序列（"按优先级逐个完成"的建议顺序，2026-09-26 刷新）

排序规则（**同级别内**）：**前置能否立即满足 → 解锁多少下游 → 成本**。
"需批准"列指按 `AGENTS.md` §0.2 / §0.3 必须先取得许可的动作。

| # | 事项 | 级别 | 前置 / 成本 | 需批准 |
|---|---|---|---|---|
| **1** | **§9.2 的前置测量**：按 **G6** 口径做一次 decode 性能 profile；顺带落地 §6.3 的 profile target。**计划见** `future_iterations_development_plan.md` §10 / `future_iterations_test_plan.md` §9 | P2 | 无外部资源；真机 1~2 往返。**当前 decode 侧零性能数据**，且它一次性解锁 §9.2（sampler 要不要重写）、§2.2（attention 分块是否值得）、§10.2/§6.2（子图替换所需的可复现测量方法） | **是**（单测外的测试任务，§0.3） |
| 2 | §9.3 采样器参考数据固化（`scripts/ref_sampler.py` 输出落成 `.bin` 供 C++ 载入） | P2 | 无外部资源；纯 host | 否 |
| 3 | §10.1 ONNX / 原生 I/O 契约统一（ONNX 侧加 `Cast` 把 `input_ids` 降到 INT32） | P2 | 无外部资源；为"ONNX 路径接进 `LLMRunner`"铺路 | 否 |
| 4 | §2.1 显存池（**先量分配开销**；可与第 1 步同一次真机往返顺带量） | P2 | 无外部资源 | 否（测量）；若真改实现则需确认 |
| 5 | §2.4 CV 动态分辨率 | P2 | 无外部资源，但**需要一个明确需求口径**（当前无人要求变分辨率） | 是（需求确认） |
| 6 | §2.3 Continuous Batching（含前置 G2-3：把 `LLMRunner` 从 `batch = 1` 扩开） | P2 | 无外部资源；工作量偏大 | 否（代码） |
| 7 | §1.2 LLM INT8 / INT4（含前置：`PagedAttentionPlugin` 支持 INT8 KV cache） | P2 | 无外部资源；INT4 需先评估 sm_75 kernel 可行性 | 否（代码） |
| 8 | §6.4 CI 自动化 | P2 | 本地脚本部分无外部资源；**GitHub Actions 属联网** | 部分 |
| 9 | §1.6 整条（INT8 绝对误差判据） | P2 | **需联网**下载带真值标签的验收集 | **是** |
| 10 | §3.1 / §3.2 / §3.3 / §3.4（seq2seq / ViT / 多模态 / Audio）、§4.1 / §4.2（CV 归一化 / LLaMA 类激活） | P2 | 需目标模型权重（多需联网） | **是** |
| 11 | §6.1 / §6.2（转换工具增强 / ONNX custom op） | P2 | 按需；§6.2 是 §10.2 的前置 | 视情况 |
| 12 | **§1.5**（P4-INT8-a：per-channel 整网退化根因） | P3（触发即升 P1） | 无外部资源；真机 ≤3 往返；**做法 / 验收 / 成本校准已写全**（第 1 步是修仪器：探"量化前"张量并与 torch 已折叠 BN 对拍） | 是（改探针脚本 + 真机） |
| 13 | P3 其余（§1.3 BF16 / §1.4 GPT-2 FP16 / §2.2 FlashAttention / §5.2 tiktoken / §5.3 多模态模板 / §7.1 / §7.2 / §10.2）与 **SP-1** | P3 | 等触发：硬件 / 需求 / 测量方法 | —— |
| —— | §1.1（implicit 校准）、§9.1（PagedAttention Prefill） | 冻结 | 已被取代 / 触发条件已被现状否证 | —— |

**读法**：第 1~4 步是"不需要任何外部资源、现在就能做完"的一档；第 5~8 步需要代码工作量但无外部资源；
第 9~11 步需要联网或新模型资产；第 12 步是 P3 里唯一"做法写全、可立即执行"的一条；
第 13 步与冻结项**不主动开工**。

---

## 1. 量化与精度优化

### 1.1 ResNet18 INT8 校准

- **优先级**：**冻结**（原 P1，2026-09-26 重定为冻结，理由见 §0.1）
- **背景**：现有 `0_resnet18_onnx` 已支持 INT8，含 `calib_data/` 与 `Int8Calibrator`。`mini_trt_llm` 替换后需补齐该能力。
- **现状更新（2026-09-26）**：本条的"实现 `nvinfer1::IInt8Calibrator` 封装"路线**已被取代**——
  Phase 4 用 **Q/DQ 显式量化**（`tools/convert/quantize_resnet18.py`）交付了 ResNet18 INT8，
  且 TRT 10.12 起 `kINT8` / implicit calibration 路线弃用（`docs/TROUBLESHOOTING.md` #27）。
  下面"工作内容"保留原文仅作备查；`mini_trt_llm` 内当前没有任何 calibrator 代码。
- **工作内容**：
  - 实现 `core/int8_calibrator.hpp/.cpp`，封装 `nvinfer1::IInt8Calibrator`。
  - 支持 `CalibrationDataReader` 读取 `calib_data/*.bin`。
  - `EngineBuilder` 增加 INT8 配置路径。
  - 在 `ModelConfig` 中增加 `quantization` 字段描述每层/全局校准策略。
- **关键依赖**：`0_resnet18_onnx` 的 `calibrator.cpp` 可直接参考。

### 1.2 LLM INT8 / INT4 量化

- **优先级**：P2
- **背景**：sm_75 有 INT8 Tensor Core，LLM INT8 可显著提升吞吐。
- **工作内容**：
  - 支持 INT8 weight-only 或 SmoothQuant。
  - 支持 GPTQ/AWQ 需评估 sm_75 兼容性（可能不支持部分 INT4 kernel）。
- **关键依赖**：PagedAttention Plugin 需支持 INT8 KV Cache。

### 1.3 BF16 原生计算支持

- **优先级**：P3
- **背景**：当前 sm_75 无 BF16 Tensor Core，BF16 权重会转换为 FP32/FP16。
- **工作内容**：若未来迁移到支持 BF16 的 GPU（Ampere+），可直接启用 BF16 engine。

---

### 1.4 GPT-2 的 FP16 端到端（需要激活缩放 / 关键算子保 FP32）

- **优先级**：P3（2026-09-26 补标签：本条原先漏标；依据 = sm_75 无 Tensor Core，收益有限，
  且触发条件是"真要推进低精度推理"——见 §0.1）

**现状（已实测）**：真实 GPT-2 在本项目的**弱类型 FP16** 引擎下端到端产生 NaN，
且出现 NaN 的层随构建变化（0/1/2），而激活幅值远未触及 FP16 上限。FP32 端到端完全正确。
完整证据链与"为什么按政策不修"见 `docs/TROUBLESHOOTING.md` §18.1。

**要解决它，可行方向**（按成本从低到高）：

1. **关键算子显式保 FP32**：残差相加、softmax、归一化（LN 已做）设 `setPrecision(kFLOAT)`，
   FP16 只用于 matmul；代价是每层多几组 cast；
2. **逐算子二分到具体算子**：把第 0 层式的"切点导出"扩到前若干层（仪器已具备），
   先定位到算子再决定改哪个——预计 2 轮以上真机往返；
3. **激活缩放**（outlier 处理）：属研究性质，与 INT8/低精度量化同一课题（见 §1.2）。

**触发条件**：真要推进 GPT-2 的 FP16/低精度推理性能时（目标硬件无 Tensor Core，
收益有限，所以不急）。

### 1.5 per-channel 权重量化的整网退化根因（P4-INT8-a 立项）

- **优先级**：P3（不阻塞任何现行路径——正式产物已定为 per-tensor 权重；只在"需要更高 INT8 精度"
  时升为 P1）
- **现状（全部为实测，出处见条目末尾）**：

  | 层面 | per-channel | per-tensor |
  |---|---|---|
  | 单卷积（合成权重） | 与 numpy 模拟差 `1.9e-6` | 同 |
  | 单卷积（conv1 真实权重，scale 跨度 2.88e13） | 与模拟差 `8e-7` | 同 |
  | 最小残差 block（真实权重） | 与模拟差 `0.0134`（**更好**） | `0.0457` |
  | **整网**（256 张） | 余量子集 **54.5%** / 整体 25.0% | 余量子集 **100%** / 整体 37.9%（A/B 那一轮的口径；最终 `prequant_dq` 产物实测整体 38.3%，见 `docs/PROGRESS.md` §3.0d） |
  | 整网模拟（同批 scale） | 余量子集 100% | 余量子集 100% |

  即：**算子级与 block 级都证明 per-channel 至少不差，整网级却明显更差**，机制未知。
  已否证 11 条假设（写法错 / 死通道 scale 跨度 / 模拟不忠实 / 残差融合 / `axis` 属性类型 /
  2-D 广播形状 / 权重只留 DQ / 布局 / 舍入方式 / 量化 step 与 scale 不符 / 探针工具自身——详见
  `docs/TROUBLESHOOTING.md` #29 / #30 / #31）。
- **目标**：给出**结论**，二选一即可，但不允许"下次再看"：
  1. 定位到**从第几层开始分叉**并说明机制（该机制还要能被算子级 / block 级最小复现解释）；
  2. 或证明所有可查方向均已查空，逐条写明"为什么这条路不能再查"。
- **做法（按顺序；仪器在前次排查中已经定型）**：
  1. **探"量化前"的 float 张量**（`QuantizeLinear` 之前那一个），挂成图的额外输出，与 torch
     **已折叠 BN** 的模型同点对拍。**为什么这样做**：上一次探"量化后"的张量失败了——量化台阶
     `0.0796` 会把 FP32 kernel 的正常差异（`1e-3` 量级）在 bin 边界附近放大成 ±1 格，
     噪声与待查信号同量级（见 #30.5）。量化前的张量量级是 `1e-3`，可直接判读。
  2. 同一引擎**重复跑同一输入**确认确定性，然后画"逐层误差增长曲线"——**不做单点比较**
     （#30.6 的第 2 条）。
  3. 覆盖前次最小复现**没覆盖**的两段：3 个**下采样卷积**（1×1/s2）与 **GAP + fc** 段
     （#30.3 第 2 条）。
  4. 需要更硬结论时换用 §1.6 的验收集（本地只有 tiny-imagenet 放大图，余量子集仅 11~12 张）。
     注意方向：样本量放大同时也会放大"per-channel 更差"的统计显著性，但要先排除验收集自身
     分类退化带来的混淆。
- **验收判据**：
  - 要么"第 N 层、机制 X"且被算子级 / block 级最小复现支持；
  - 要么"已否证假设清单"每条都附**判定实验与实测值**（沿用 #30.1 的表格体例）。
  - 无论哪种，结论都要写回 `docs/TROUBLESHOOTING.md`（新增节），并回头改本条目的状态。
- **前置依赖**：无。第 1~3 步不需要联网、不需要新数据（第 4 步依赖 §1.6）。
- **成本校准（前次实测）**：全流程是**每轮 1 次真机往返**级别；前次 #30.5 用掉 3 轮，其中 2 轮
  耗在**探针自身的 BN 折叠错**上——这次先修仪器再取数。
- **出处**：`docs/TROUBLESHOOTING.md` #29.2 / #30（#30.1 否证表、#30.3 未排除方向、#30.5 仪器坑、
  #30.6 收口决定）；`docs/phase4_int8_plan.md` §7 的 P4-7-3 行。

### 1.6 INT8 精度判据与带真值标签的验收集（P4-INT8-b 立项）

- **优先级**：P2
- **现状（实测）**：当前判据是**分层的**——主判据 = "FP32 有余量"（`margin >= 5`）子集的
  top-1 一致率 `>= 90%`（实测 12/12 = 100%）；整体一致率 `>= 30%` 只作"没崩坏"下界（实测 37.9%）。
  三个已知弱点：
  1. `calib_data` 是 **tiny-imagenet 放大图**，分类本身退化（8 张里 5 张同类）——**没有真值标签**，
     所以现在只能判"与 FP32 是否一致"，判不了"对不对"；
  2. 有判别力的样本只有 **11~12 张**，率的分辨力弱（Fisher 精确检验 ≈0.03，勉强算显著）；
  3. **绝对误差界未定**：实测 `max_abs ≈ 21.6` 被少数样本放大，所以**故意不拿它当判据**
     （`docs/phase4_int8_plan.md` §4 的"数值上界"行）。
- **目标**：把 INT8 的验收从"一致率"升级为能回答"凭什么这么定"的两条判据：
  (a) 有样本量依据的 top-1 一致率；(b) 在有判别力样本上量出的绝对 / 相对误差上界。
- **做法**：
  1. 固定一个 **500–1000 张、带真值标签**的 ImageNet 验证子集，并把它变成**可复现的产物**：
     记来源、版本与 SHA256；采样时**排除与 `calib_data` 重叠**的图——否则标定集污染验证集，
     "一致率"会被系统性高估。口径（含重叠排除规则）必须写进 meta。
  2. 沿用同一套 `margin` 分层统计：余量子集与全体**分别**报率，并**报样本量**
     （只报率不报 n 的结论不可复核）。
  3. 绝对误差只在余量子集上量 **per-sample `max_abs` / 相对误差的分布**（报 p95 / p99，不报全样本
     max），阈值取该分布的合理倍数，**阈值必须挨着写它的出处**（哪次实测、哪个分位）。
  4. 阈值定稿后回写 `docs/phase4_int8_plan.md` §4 与 `docs/phase4_test_plan.md` 的 R2.6，
     并用**同一把尺子**重测一次 per-channel vs per-tensor（这对 §1.5 也是判据输入）。
- **验收判据**：
  - 每个阈值旁边能回答"凭什么这么定"（本条目实测的分布 + 样本量 + 分层口径）；
  - 验收集本身可复现（meta 里查得到来源与 SHA256）；
  - 明确写出**本判据不覆盖什么**（例：只覆盖该验证集分布内的分类一致率，不构成对任意输入的
    数值保证）。
- **前置依赖**：**需要联网下载数据集**（`AGENTS.md` §0.2：联网操作必须另行单独获批）。
  未获批时可先做本条的"口径定义"部分（验收集规格 + meta 字段 + 分层统计脚本），不下载数据。
- **不许做的事**：不许为了"绿"而调阈值或删断言（`AGENTS.md` §7）；不许把本条的阈值套到
  FP16 / FP32 上（那是跨精度复用，见 `docs/PROGRESS.md` §2.14 A）。
- **出处**：`docs/phase4_int8_plan.md` §4（判据表）/ §5（风险与回退）/ §7 的 P4-7-3 行；
  `docs/TROUBLESHOOTING.md` #29.4 / #30.5。

> **为什么拆成两条**：§1.5 是**仪器 / 机制**问题（第 1~3 步不依赖新数据），§1.6 是**判据 / 数据**
> 问题（依赖联网）。两者共用一份验收集，但开工前提不同；合成一条会让"不联网就一步都动不了"。

## 2. 性能优化

### 2.1 显存池替换简单封装

- **优先级**：P2（原 P1，2026-09-26 下调：收益未测，且现行设计已避开 decode 循环内分配）
- **背景**：Phase 0 的 `DeviceBuffer` 仅封装 `cudaMalloc/cudaFree`，频繁分配有性能开销。
- **触发条件**：先量出"分配开销确实显著"。当前 `DeviceBuffer` 仍是直接 `cudaMalloc`
  （`src/utils/memory_pool.cpp`），但热点路径上没有循环内分配 → 没有测量就没有依据。
- **工作内容**：
  - 实现基于 freelist 或 arena 的 `MemoryPool`。
  - 支持按大小分桶、按 stream 隔离。
  - 替换 `DeviceBuffer` 内部实现，保持接口不变。

### 2.2 FlashAttention / FlashDecoding

- **优先级**：P3（原 P2，2026-09-26 下调：未量基线、sm_75 无官方支持，属探索性）
- **背景**：Turing sm_75 无官方 FlashAttention 优化，但可 hand-tune 基础 attention kernel。
- **工作内容**：
  - 在 `PagedAttentionPlugin` 中实现针对 sm_75 的分块 attention。
  - 支持 decode 阶段的 batching 优化。

### 2.3 Continuous Batching / In-Flight Batching

- **优先级**：P2
- **背景**：服务化部署时，continuous batching 可大幅提升 GPU 利用率。
- **工作内容**：
  - `LLMRunner` 支持多请求队列调度。
  - `PagedKVCache` 支持跨请求动态分配与回收。

### 2.4 CV 动态分辨率

- **优先级**：P2
- **背景**：Phase 0 CV 只支持动态 batch，后续需支持输入分辨率变化。
- **工作内容**：
  - `CVRunner` 支持 `H/W` 动态轴。
  - `EngineBuilder::Config` 增加 CV 动态分辨率 profile。
  - ResNet18 全局池化层天然支持动态分辨率，主要工作在 shape 配置。

---

## 3. 模型与架构扩展

### 3.1 Encoder-Decoder 模型

- **优先级**：P2
- **背景**：架构已预留接口（`ModelConfig::architecture = "encoder_decoder"`）。
- **工作内容**：
  - 实现 `EncoderDecoderModelBuilder`。
  - `LLMRunner` 扩展为 `Seq2SeqRunner`，支持 encoder prefill + decoder cross-attention。
  - KV Cache 需同时管理 self-attention 与 cross-attention cache。

### 3.2 Vision Transformer（ViT）

- **优先级**：P2
- **背景**：ViT 结构与 LLM 高度相似，只是输入为 image patches。
- **工作内容**：
  - 实现 `ViTModelBuilder`。
  - Patch Embedding Plugin。
  - 复用现有 Transformer Block 构建逻辑。

### 3.3 多模态模型（CLIP / LLaVA）

- **优先级**：P3
- **背景**：需要 CV encoder + text encoder/decoder 联合推理。
- **工作内容**：
  - 扩展 `BaseTokenizer` 支持多模态 prompt 模板。
  - `EngineBuilder` 支持多输入网络（image + text）。
  - 跨模态投影层 Plugin。

### 3.4 Audio 模型（Whisper）

- **优先级**：P3
- **背景**：Encoder-Decoder 的特例。
- **工作内容**：
  - 音频特征提取（log-mel spectrogram）预处理。
  - Whisper encoder 与 decoder 构建。

---

## 4. Plugin 与 Kernel 扩展

### 4.1 更多归一化 Plugin

- **优先级**：P2（原 P1，2026-09-26 下调并**缩小范围**：`LayerNormPlugin` 已被 TRT 原生层取代）
- **内容**：
  - ~~`LayerNormPlugin`~~：**已作废**——GPT-2 用 TRT 原生 `addNormalizationV2`
    （`src/core/gpt2_model_builder.cpp`），不需要自研；将来若原生层有性能问题再重开。
  - 当前已有 `RMSNormPlugin`（Phase 1 交付）。
  - `GroupNormPlugin`、`InstanceNormPlugin`（CV 模型需要）。

### 4.2 更多激活函数 Plugin

- **优先级**：P2（原 P1，2026-09-26 下调：属于"接 LLaMA 系列时才需要"的按需件）
- **内容**：
  - `SiLUPlugin`、`SwiGLUPlugin`（LLaMA/GLM 系列 FFN 需要）。
  - `GELUPlugin` 若 TRT 原生实现性能不足时自定义。

### 4.3 MoE（Mixture of Experts）支持

- **优先级**：P3
- **内容**：
  - Expert 路由 Plugin。
  - 稀疏门控与专家并行。

---

## 5. Tokenizer 扩展

### 5.1 BPE Tokenizer（GPT-2 原生）

- **优先级**：P1
- **背景**：GPT-2 使用 BPE，SentencePiece 行为可能与 Python `tiktoken`/`transformers` 不完全一致。
- **接口事实（2026-09-26 补）**：byte-level BPE 需要**两个文件**（`vocab.json` + `merges.txt`），
  而 `BaseTokenizer::Load` 只有一个路径参数 → 实现时把入参语义定为**目录**（不改基类，见开发计划 F2）。
- **工作内容**：
  - 实现 `BpeTokenizer : public BaseTokenizer`。
  - 从 `vocab.json` + `merges.txt` 加载。
  - 与 Python GPT-2 tokenizer 逐 case 对比。

### 5.2 Tiktoken Tokenizer

- **优先级**：P3（原 P2，2026-09-26 下调：没有任何现行模型需要它）
- **背景**：GPT-4 / ChatGLM 等模型使用 tiktoken。
- **工作内容**：
  - 接入 `tiktoken` C++ 实现或自研。

### 5.3 多模态 Prompt Template

- **优先级**：P3
- **背景**：LLaVA 等模型需要 image token 占位与特殊模板。
- **工作内容**：
  - `BaseTokenizer` 增加 `ApplyChatTemplate` 接口。

---

## 6. 工具链与工程效率

### 6.1 Python 转换工具增强

- **优先级**：P2（原 P1，2026-09-26 下调：按需扩，不阻塞现行能力）
- **内容**：
  - 支持更多来源：PyTorch `.pt`、HuggingFace、Meta 原始 checkpoint。
  - 支持 INT8 校准数据自动生成。
  - 支持 ONNX 导出（方案 B 需要）。

### 6.2 ONNX 导出与 Plugin Custom Op

- **优先级**：P2
- **背景**：方案 B 需要 ONNX 中带有 `mini_trt_llm` domain 的 custom op。
- **工作内容**：
  - 提供 PyTorch `torch.onnx.register_custom_op_symbolic` 示例。
  - 或提供 `torch.export` + custom decomposition 脚本。

### 6.3 Nsight 一键 Profile Target

- **优先级**：P2
- **内容**：
  - CMake 增加 `profile_gpt2`、`profile_resnet18` 自定义 target。
  - 支持 `nsys profile` 与 `ncu` 导出 `.ncu-rep`。

### 6.4 CI / 自动化测试

- **优先级**：P2
- **内容**：
  - GitHub Actions 或本地脚本：
    - 代码格式检查（`clang-format`）。
    - `cmake -DBUILD_TESTS=ON && ctest`。
    - 下载 dummy model 并跑端到端测试。

---

## 7. 服务化与部署

### 7.1 HTTP / gRPC 推理服务

- **优先级**：P3
- **内容**：
  - 基于 `LLMRunner` 提供 OpenAI-compatible API。
  - 支持流式输出。

### 7.2 多 GPU / Tensor Parallel

- **优先级**：P3
- **内容**：
  - 评估 sm_75 单机多卡价值（GTX 1660 Ti 通常为单卡）。
  - 如未来迁移到多卡环境，再实现 TP/PP。

---

## 8. 已知问题记录

> **本节只是索引**：事实与处置口径以 §1 / §2 / §5 / §9 / §10 与 `docs/TROUBLESHOOTING.md` 为准，
> 本表不新增口径（两处维护必然漂移）。

| 问题 | 影响 | 状态 | 计划解决阶段 |
|---|---|---|---|
| BF16 在 sm_75 下需转换 | 无功能影响，有轻微构建时开销 | 已知 | Phase 0 已处理 |
| SentencePiece 与 GPT-2 BPE 不完全一致 | 可能导致 tokenizer 结果偏差 | 已知 | 后续实现 BpeTokenizer |
| PagedAttention 无 FlashAttention 优化 | Decode 延迟较高 | 已知 | 后续 hand-tune |
| INT8 的 implicit 校准（`IInt8Calibrator`）未实现 | **不影响现行能力**——ResNet18 的 INT8 已由 Q/DQ 显式量化交付（`docs/PROGRESS.md` §3.0d） | **冻结**（TRT 10.12 起该路线弃用） | 见 §0.1 / §1.1 |
| CV 动态分辨率未实现 | 输入尺寸固定 | 已知 | 后续迭代 |

---

## 9. Phase 1 明确延后的能力

以下三项在 Phase 1 实现算子层时**有意**未做，属于"能力延后"而非"漏项"，故从 `docs/PROGRESS.md`
的下一步计划中迁出、归档到这里。

### 9.1 PagedAttention 的 Prefill 阶段

- **优先级**：**冻结**（原 P1，2026-09-26 重定，理由见 §0.1 与下方"现状更新"）
- **背景**：Phase 1 只实现了 Decoding 阶段（query 序列长度为 1，决策 D1）。
  Prefill 需要对 query 序列做因果 mask 与按位置分块，kernel 结构与 Decode 路径差异较大，
  混在一起会同时拖慢两条路径。
- **工作内容**：新增 Prefill kernel（因果 mask + 分块 softmax），Plugin 侧按 `seq_len` 分派；
  去掉 `configurePlugin` 中"拒绝 `seq_len > 1`"的校验。
- **触发时机**：Phase 2 的 GPT-2 若走单引擎（不分 Prefill/Decode 两个 engine），就必须先补。
- **现状更新（2026-09-26）**：**上述触发条件已被现状否证**——`LLMRunner` 的构造签名收
  **prefill + decode 两个引擎**（`include/mini_trt_llm/core/llm_runner.hpp:56`），
  Phase 2 / Phase 3 / Phase 4 全程都是双引擎路径。本条转为冻结备查，
  只有将来真要做"单引擎含 Prefill"时才需要重新评估。

### 9.2 Sampler 的手写高性能 kernel

- **计划（2026-09-26 成文）**：开发计划见 `docs/future_iterations_development_plan.md` **§10**、
  测试计划见 `docs/future_iterations_test_plan.md` **§9**（含计划对账 / 任务分解 P9_2-0~8 / 验收判据 /
  破坏性动作预告 / 真机清单）。**状态：计划已成文；P9_2-0 的性能仪器已实现（`SamplerPerf.ThroughputByShape`），基线数据待真机运行。**
- **优先级**：P2（原 P1，2026-09-26 下调：触发条件从未被验证过，见下）
- **背景**：Phase 1 用 CUB 分段排序保证正确性（决策 Q15），Top-K / Top-P 目前每步都要对整行
  `vocab_size` 做一次降序排序，是明显的性能瓶颈（`vocab_size` 可达 128K）。
- **工作内容**：warp-level Top-K 选择（无需全排序）、bitonic sort、以及与后续 continuous batching 的配合。
- **触发时机**：Phase 2 跑通端到端吞吐后，用 `nsys` / `ncu` 定位到 sampler 占比显著时。
- **触发条件现状（2026-09-26 核实）**：**这个 profile 至今没做过**——
  `docs/PROGRESS.md` §3.0d 记录的性能数字只有 `CVRunner` 的 benchmark（`mean≈8.4 ms`），
  没有任何 decode 阶段的 sampler 占比。触发条件"未验证"不等于"已成立"，故不能按 P1 对待。

### 9.3 采样器参考数据固化为数据文件

- **优先级**：P2
- **背景**：D3 要求 Generation 层"对比分布"。Phase 1 已落地两件事：C++ 侧的统计检验
  （词频收敛到解析 softmax 概率）与 `scripts/ref_sampler.py`（打印 HF 风格截断语义）。
  尚未做的是把 Python 输出落盘成 `.bin` 供 C++ 测试直接载入，因此**与 HF 截断语义的自动化交叉验证缺失**。
- **工作内容**：脚本导出 nucleus 集合与概率，测试读取并比对；顺带覆盖 `p` 接近 1 的边界。

---

*文档版本：v1.0*  
*关联文档：`docs/mini_trt_llm_design.md`、`docs/phase0_development_plan.md`*

## 10. Phase 3 明确延后的能力

### 10.1 ONNX 路径与原生路径的 I/O 契约统一

- **优先级**：P2（2026-09-26 补标签：原文漏标；触发条件 = ONNX 路径要接进 `LLMRunner`，见 §0.1）

**现状**（`docs/TROUBLESHOOTING.md` #17）：两条路的输入契约不一致——

| | 输入张量 | 类型 |
|---|---|---|
| 原生构建（方案 A） | `input_ids` + `position_ids` | 均为 INT32 |
| ONNX（方案 B，torch 导出） | 只有 `input_ids` | **INT64** |

**为什么要统一**：ONNX 路径要成为"可互换的一等公民"，调用方就不该按引擎类型分别准备输入。
（当前对拍用例已按各引擎**声明的**契约准备输入，因此判据仍然有效。）

**可能的做法**：在 ONNX 路径加一个 `Cast` 把 `input_ids` 降到 INT32（保持 `input_ids` 语义不变）；
或把 `position_ids` 提升为显式输入（同时要把图内的常量位置编码改成按输入 gather）——
后者改动更大，只有在"两条路必须共用同一个 runner"时才值得。

**判定依据**：先做前者（Cast）即可让 dtype 一致；是否需要后者，取决于将来是否要把
ONNX 路径接进 `LLMRunner`（`docs/phase3_development_plan.md` D3 已明确本阶段不做）。

### 10.2 ONNX 图的子图替换（D1=B）

- **优先级**：P3（2026-09-26 补标签：原文漏标；前提取决于缺口 G6 的可复现测量方法，见 §0.1）

**为什么现在还不能判断**（**已修正**：本条原先写的是"ONNX 慢约 22%、故替换无收益"，
那是从**单次测量**里读出的结论，已被后续运行推翻）：

Phase 3 的两次测量方向相反（`docs/phase3_test_plan.md` §3.1）：

| 运行 | prefill(4 token) ONNX / 原生 |
|---|---|
| A | 4.545 ms / 3.717 ms（ONNX 慢 22%） |
| B | 4.716 ms / 6.250 ms（ONNX 快 24%） |

差异（±25%）小于**构建间噪声**——已知机制是 TRT 的 kernel tactic 选择依赖构建时的机器状态。
因此**"ONNX 更快还是更慢"目前是未定**，也不能据此判断子图替换的收益。
要做这个判断，先得建立可复现的测量方法（见 `docs/phase3_test_plan.md` 缺口 G6）。

**若将来要做，前置条件**：(1) 有明确的性能目标与基线；(2) 先量出"替换能省掉多少 kernel 时间"
（用 nsys/ncu 的实测，而非推断）；(3) 替换前后保留两份 engine 做数值对照。

**保持不变的部分**：`tools/inspect_onnx.py --check` 的子图识别断言应当保留——
它是"图变了要知道"的护栏（含 `absent_ops`：RMSNorm / RoPE / Attention 等一旦出现，
说明模型或导出路径变了，需要重审结论）。

## 11. 测试覆盖缺口索引（默认不排期，按触发条件决定；已立项的在 §1 / §2）

**说明**：本节**只做索引**，不复制细节——缺口的事实与背景在各阶段的测试计划 / 故障记录里
（`docs/PROGRESS.md` §2.13「唯一来源」原则）。放在这里的原因：缺口若只留在已过阶段的文档里，
换阶段后必然失传。**各条目的优先级与排序见 §0.1**（本节只写"是什么 / 触发条件"）。

**索引里已经"立项"的条目**（§1 / §2 有正式排期条目 = 目标 / 做法 / 验收判据 / 前置依赖）：
**P4-INT8-a → §1.5**、**P4-INT8-b → §1.6**。其余仍是"记着但没排"。
两处不复制内容：本节只写"是什么 / 触发条件"，正式条目只写"怎么做 / 怎么验收"，
事实与实测值仍在 `docs/TROUBLESHOOTING.md`（这是刻意的，避免三处漂移）。

| 缺口 | 出处（唯一来源） | 是什么 | 触发条件（什么时候做） |
|---|---|---|---|
| ~~G1c~~ | 测试计划 §5 | ~~ONNX 输出名护栏无用例~~ **已关闭**（2026-09-25）：新增 `tools/make_tiny_onnx.py` 生成 `input_ids → not_logits` 夹具 + `RejectsGraphWithoutLogitsOutput` 用例 | —— |
| ~~G4b~~ | 测试计划 §5 | ~~ONNX 路径 batch 维未覆盖~~ **已关闭**（2026-09-25）：`RunEngine` 改为 batch 感知，覆盖 `(batch,seq) ∈ {(1,1),(1,64),(1,512),(2,4),(2,64)}` | —— |
| **G5** | 同上 | 子图识别只做"计数"，未做拓扑/邻接级 | 仅当真要做子图替换时（见 §10.2）——计数相同但连接不同是识别不出来的 |
| **G6** | 同上 | L3 性能**无可复现测量方法**（两次运行方向相反，±25% < 构建间噪声） | 仅当真要优化 ONNX 路径性能或要回答"要不要做子图替换"时。做法：固定机器状态 + 同一 session 内 ≥3 次构建 / ≥20 次推理，报中位数与极差 |
| ~~G2-1~~ | `docs/phase2_test_plan.md` §5 | ~~真实 GPT-2 的 FP16 端到端未测~~ **已执行**：FP16 端到端出 NaN（缓冲问题已修；数值问题按政策不修，登记为已知限制） | 解决路径见本文件 §1.4；复现器与诊断仪器保留在 `tests/test_gpt2_generate.cpp` |
| **G2-3** | 同上 | `LLMRunner` 只支持 `batch = 1`（有意限定） | 需要批处理时再扩（同时引入多序列 block 分配、各自 `context_lens` 与采样参数） |
| **G2-4** | 同上 | EOS 无法在循环内早停（已知 workaround，语义正确） | 见 `docs/PROGRESS.md` §5.0；若要真早停，需设备侧 stop flag + 条件图 |
| ~~G7~~ | 开发计划 §4 | ~~探针未接入 ctest~~ **已关闭**（2026-09-25）：注册为 `onnx_graph_probe`，缺环境返回 77 → ctest 报 Skipped | —— |
| **P1.5-a** | `docs/phase1_5_test_plan.md` §5 | **Top-K / Top-P 的 FP16 分支未覆盖**（Greedy 已覆盖；三者同属采样器同一处 dtype 分派，风险低） | 真正跑非贪心采样时（`temperature` / `top_p` 一旦进入产品路径） |
| **P1.5-b** | 同上 | E2 的**完整链路**（`RMSNorm → QKV → RoPE → PagedAttention → LM Head`）与 `ref_mini_block.py` 有意留后（P1.5-4 缩减完成） | 要往 LLaMA 风格链路继续做时（Phase 4 之后），或怀疑"多算子相邻契约"出问题时 |
| **P1.5-c** | 同上 | E3 只验"接受/拒绝"，未验**同 engine 内多次切换 profile 后的数值一致性** | 真的依赖多 profile 混用时（当前 runner 每步只用 profile 0） |
| **P4-INT8-a** | `docs/TROUBLESHOOTING.md` #29 / #30 | **权重 per-channel 量化在整网上比 per-tensor 差得多**（余量子集 54.5% vs 100%），而单卷积与"真实权重+残差"的最小 block 上它都**不差**（甚至更好）→ **原因仍未找到**。已排除 11 条假设（写法错 / 死通道 scale 跨度 / 模拟不忠实 / 残差融合 / `axis` 类型 / 2-D 广播 / 权重只留 DQ / 布局 / 舍入 / step 与 scale 不符 / 探针自身） | 需要更高 INT8 精度时。**已在 §1.5 立项**（目标 / 做法 / 验收判据 / 前置依赖在那里；第 1~3 步不依赖联网）。关键方法：探"**量化前**"的 float 张量，而不是量化后的——后者被 bin 边界 ±1 格噪声主导，分辨率不够（#30.5 / #30.6） |
| **P4-INT8-b** | `docs/phase4_int8_plan.md` §4 / §5 | INT8 的**数值上界判据未定**（当前只在"有判别力子集"上用一致率判，且该子集无可核对的真值标签） | 需要给出 INT8 的绝对误差保证时。**已在 §1.6 立项**（目标 / 做法 / 验收判据在那里；**前置依赖 = 联网下载验收集，须先获批**） |
| **P4-FP16-a** | `docs/phase4_int8_plan.md` §1.1 | **FP16 路径仍使用已废弃的 `BuilderFlag::kFP16`**（TRT 10.12 起废弃，指向 strong typing）；实测可用 | 真要迁到强类型网络时（两条 builder 的每个算子都要显式设类型，代价大） |
| **P1.5-d** | 同上 | 采样器**分布级数据未固化**（`scripts/ref_sampler.py` 只打印，输出没落成测试数据） | 要做采样的统计正确性回归时（属增强，见本文件 §9.3） |
| **SP-1** | `docs/PROGRESS.md` §3.0e（"未验证件"） | `SentencePieceTokenizer` 是**无用例、无资产、无调用方的未验证件**：`BaseTokenizer` 的另一个实现，但测试目录零引用、仓库里没有 `.model`/`.spm`、`LLMRunner` 只存基类指针（测试传 `nullptr`）→ `Load`/`Encode`/`Decode` 从未被任何参考裁决过 | 要真正用 SentencePiece 系模型（LLaMA 类）时。**处置顺序**：先造参考与用例再动实现（照 BPE 那套：golden + 来源 SHA256 + meta 自证 + 负例）；资产需作者提供或联网取（**联网须先获批**） |

**判定原则**：这些是"覆盖不足"，不是"已知缺陷"——已发现的缺陷一律进
`docs/TROUBLESHOOTING.md` 并配回归用例；缺口是"还没被盯住的地方"，处置方式不同。
