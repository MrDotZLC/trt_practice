# 后续迭代计划

> 本文档记录 `mini_trt_llm` 中已明确延后、但未来需要实现的能力，防止遗忘。  
> 每一项标注预计优先级、大致阶段、关键依赖。**优先级定义与排序见 §0——章节顺序是主题分类，不是优先级。**
> **执行安排见 `docs/future_iterations_development_plan.md`（怎么做）与
> `docs/future_iterations_test_plan.md`（怎么验）**——本文件只保留"是什么 / 为什么 / 触发条件 / 验收判据"。

---

## 0. [OI-PRIORITY-DEF] 优先级定义与排序规则（优先级的唯一排序来源）

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

> **文档归位规则（2026-10-01 修订；原 2026-09-26 决定已被取代）**：后续迭代中**每个事项的
> 开发状态与设计落点改为 `docs/dev/<feature>/`**（入口 = 该目录的 `STATE.md`；阶段链与 Gate
> 见 `AGENTS.md` §5 与技能 `trt-inference-engineering`）。本文件（触发条件 / 优先级 / 判据）
> 与 `docs/future_iterations_development_plan.md`、`docs/future_iterations_test_plan.md`
> **保留触发条件、判据与执行回填**，不再承载新 feature 的设计正文。
> （例：§9.2 的两份计划当年并入 §10 / §9，属老口径存量，不回改。）

### 0.1 [OI-PRIORITY-TABLE] 优先级总表（按建议优先级排序）


<details><summary>展开：0.1 [OI-PRIORITY-TABLE] 优先级总表（按建议优先级排序） 全文</summary>

| 建议 | 条目 | 原标签 | 触发条件 vs 现状（依据） | 开工前提 / 备注 |
|---|---|---|---|---|

| ~~P1~~ **已完成** | §5.1 BPE Tokenizer | P1 | **✅ 已交付（2026-09-26，批次 A1）**；交付内容与实测见 `PROGRESS.md` §3.0e、用例见 `future_iterations_test_plan.md` §2.1（15 条，含"文本 → prompt token"桥接用例）。原触发理由（能力门槛）：`LLMRunner` 只收/还 token id，`tokenizer_` 仅在构造时校验 → 目前没有"文本进 / 文本出"的端到端路径 | 纯 host 代码、可离线、可进 CI。**前置已核实（2026-09-26 更正）**：tokenizer 数据文件在本地 HF 缓存里（`~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d.../{vocab.json,merges.txt,tokenizer.json}`），且 `transformers 4.44.0` 可 `local_files_only=True` 离线加载并给出参考 token → **不需要联网**。（上一版这里写"本地没有、降为 P2"是**错的**：当时只查了 `models/gpt2/`，把"仓库里没有"当成了"机器上没有"） |
| ~~P1~~ **已完成** | §1.6 的**离线子项**（口径定义：验收集规格 / meta 字段 / 分层统计脚本） | P2（整条） | **✅ 已交付（2026-09-26，批次 A2）**：规格 `tools/validate/README.md`、脚本 `int8_eval.py`（3 项分层数学 + 7 道护栏自检）、ctest 项 `int8_eval_selftest`；C 批已用它在真机完成与 C++ 侧统计的口径交叉校验 | **下载验收集那一半仍是 P2**（需联网，见本表下一档） |
| **P2** | §1.2 LLM INT8 / INT4 | P2 | 触发 = 需要 LLM 低精度吞吐；未触发 | 前置依赖已核实：`PagedAttentionPlugin` 目前只接受 `kFLOAT` / `kHALF`（`paged_attention_plugin.cu:265`）→ 先要 INT8 KV cache；INT4 需先评估 sm_75 kernel 可行性 |
| **P2** | §1.6 验收集与绝对误差判据（整条） | P2 | 同上 | **联网下载**带真值标签的验收集，按 `AGENTS.md` §0.2 须先获批；采样须排除与 `calib_data` 重叠的图 |
| ~~P2~~ **触发未获支持** | §2.1 显存池替换 | **P1 → P2 → 不排期** | **触发已量过（PF-5，2026-09-27）**：全程 `cudaMalloc` **63 次 / 3.335 ms**、`cudaFree` 69 次 / 249.7 ms（后者是拆除期成本）→ **"频繁分配有性能开销"不成立** | 出处 `PROGRESS.md` §3.0h 第 3 条。**这一行原写"没有量过"是过期表述**（§11 交付后就没再回改）→ 2026-09-27 修正。真要重开：先证明"除分配外的热点"或"长驻服务的分配抖动"确实咬人 |
| **P2** | §2.3 Continuous Batching | P2 | 触发 = 服务化需求；未触发 | 与缺口 **G2-3**（`LLMRunner` 有意限定 `batch = 1`）耦合，要先扩 batch |
| **P2** | §2.4 CV 动态分辨率 | P2 | 触发 = CV 需要变分辨率输入；未触发（`AddCvOptimizationProfile` 目前显式拒绝） | 主体是 shape 配置，但会牵动 CV profile 与引擎缓存路径 |
| **P2** | §3.1 Encoder-Decoder | P2 | 触发 = 接 seq2seq 模型；未触发 | 架构接口已预留（`architecture = "encoder_decoder"`） |
| **P2** | §3.2 Vision Transformer | P2 | 触发 = 接 ViT；未触发 | 可复用现有 Transformer block 建图逻辑 |
| **P2** | §4.1 归一化 Plugin（**范围已缩小**） | **P1 → P2** | `LayerNormPlugin` 这一半**已被现状取代**：GPT-2 用 TRT 原生 `addNormalizationV2`（`gpt2_model_builder.cpp:245`），不需要自研 | 只剩 `GroupNorm` / `InstanceNorm`（CV 模型需要），等真接这类模型时再做 |
| **P2** | §4.2 激活函数 Plugin | P1 → P2 | 触发 = 接 LLaMA / GLM 系列；未触发 | `SiLU` / `SwiGLU` 是那些 FFN 的必需件；`GELU` 只在原生实现性能不足时才考虑 |
| **P2** | §6.1 转换工具增强 | P1 → P2 | 触发 = 出现新的权重来源 / 需要 ONNX 导出；未触发 | 按需扩，避免为不存在的来源先写代码 |
| **P2** | §6.2 ONNX 导出与 custom op | P2 | 触发 = 真做子图替换（§10.2）；未触发 | 是 §10.2 的前置件 |
| ~~P2~~ **已交付** | §6.3 Nsight 一键 Profile Target | P2 → **已交付（2026-09-27）** | 四个 target（`profile_{gpt2,resnet18}[_ncu]`）+ 一键脚本 + 分桶脚本都跑通；**§11 已关闭**（开发计划 §11.5.2） | 属测量基建：没有它，性能类结论只能靠单次点值（`docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` §3.1 已吃过一次）。落地口径见开发计划 §11.3。**能力边界**：本机（WSL2）拿不到 GPU kernel 时间线 → 逐 kernel 分解改为"同 session 比值法 + 上下文扫描"（#41） |
| **P2** | §6.4 CI / 自动化测试 | P2 | 触发 = 需要自动化回归；未触发 | **GitHub Actions 属联网操作，须先获批**；本地脚本部分不受限 |
| ~~P2~~ **已关闭** | §9.2 Sampler 高性能 kernel | P1 → P2 → **已交付（2026-09-27）** | ✅ P9_2-0~8 全部落地：Top-P 改成"保留 CUB 排序 + 行内并行 kernel"，配对（净）`legacy/parallel` = **12.63 / 22.43 / 15.73 / 17.57×**（4/4 达标）；Top-K 快速路径正确性通过但**性能不达标已撤出生产**；P9_2-5b 经同二进制 A/B 判定**效果无显著差异（保留代码）**、P9_2-5c **不做**。细节见 `PROGRESS.md` §3.0g、开发计划 §10、`TROUBLESHOOTING.md` #35/#37/#38 | **它留下的那一问已补上**（2026-09-27）：sampler 在整步 decode 里的占比由 **PF-8** 在同 session 内量出并与 `SamplerPerf` 互校一致——greedy **1.25%** / top-k **17.7%** / top-p **20.1%**（开发计划 §11.5.1、测试计划 §10.5、`PROGRESS.md` §3.0h）。**读法**：这是**短上下文**下的比例；长上下文分母变大（每步 14.7 ms），同样的 sampler 只占 ≈3% |
| **P2** | §9.3 采样器参考数据固化 | P2 | 触发 = 需要采样统计正确性回归；未触发 | 顺带覆盖 `p` 接近 1 的边界 |
| **P2** | §10.1 ONNX / 原生 I/O 契约统一 | P2 | 触发 = 要让 ONNX 路径接进 `LLMRunner`；未触发（Phase 3 D3 明确本阶段不做） | 先加 `Cast` 即可统一 dtype；`position_ids` 提升为输入是更大的改动 |
| ~~**P3**~~ **已结案** | §1.5 per-channel 整网退化（P4-INT8-a） | P3 → **已开工并结案（2026-09-27）** | 作者点名开工。**根因已定位、离线反证、真机确认**：权重 scale 取自**未折 BN** 的权重、而量化对象是**已折 BN** 的权重（`TROUBLESHOOTING` #46）；改 `--weight-range-source onnx` 后 per-channel 余量子集 **54.5% → 100%**；真机 B1 四条全绿（含"引擎 vs 自己的图"逐层最大 0.2714 → **TRT 忠实**、复现对照 PT 12/12 / PC 6/12）。**产物身份已钉死、两件后续事项暂不做**（触发条件 = §1.6 的验收集到位；见开发计划 §13.11） | 全程**0 次联网**；真机 1 次往返（含首跑修绑定 bug，见 #47） |
| **P3** | §1.3 BF16 原生计算 | P3 | 触发 = 迁到 Ampere+；硬件不支持 | 保持 P3，无需行动 |
| **P3** | §1.4 GPT-2 FP16 端到端 | **补标签** | **2026-10-06 已转 `REQ-018` 并撤销"按政策不修"**（依据 = 正确性，见该节状态行）；本行的触发条件（低精度**性能**）**仍未触发** | sm_75 无 Tensor Core，收益有限；成本已按低→高列了三条路线 |
| ~~P2~~ **已交付** | §2.2 FlashAttention / FlashDecoding | P3 → P2 → **已交付（2026-09-27）** | **触发成立**：prompt 4→960 每步 3.055→14.705 ms，长上下文下 attention 约占每步 80%（`PROGRESS.md` §3.0h / PF-9） | 方向 = **把上下文维切开**（FlashDecoding 式 split-K），已落地并在真机验证：**斜率降幅 88.85%（PP-1 kernel 级）/ 88.20%（PP-2 端到端，两次复现 88.12 / 88.20）**、三档 prompt 的 token 与单趟完全一致、`max_rel=1.5e-06`；计划 = 开发计划 **§12** / 测试计划 **§11**，验收逐条对照见 §12.4。**F1 的 B 半句（短上下文退化）经作者决定改为观测项、不设判据**（实测 +1.877%），理由见 `TROUBLESHOOTING` #45 / #45.1 |
| **P3** | §3.3 多模态（CLIP / LLaVA） | P3 | 未触发 | 成本高，且依赖 §3.1 / §3.2 打底 |
| **P3** | §3.4 Audio（Whisper） | P3 | 未触发 | 依赖 §3.1 |
| **P3** | §4.3 MoE | P3 | 未触发 | 研究性 |
| **P3** | **SP-1**（`SentencePieceTokenizer` 未验证） | 新增（2026-09-26） | 触发 = 要用 SentencePiece 系模型；**未触发**。`BaseTokenizer` 的另一个实现，但零用例、无资产（`.model`/`.spm`）、无调用方 → 正确性从未被裁决 | **先造参考再动实现**（照 BPE 那套：golden + 来源 SHA256 + meta 自证 + 负例）；资产需联网取或作者提供 |
| **P3** | §5.2 Tiktoken Tokenizer | P2 → P3 | 触发 = 接 GPT-4 类模型；未触发 | 与 §5.1 相比，它没有当前模型支撑 |
| **P3** | §5.3 多模态 Prompt Template | P3 | 未触发 | 依赖 §3.3 |
| **P3** | §7.1 HTTP / gRPC 服务 | P3 | 未触发 | 依赖 §2.3（调度）与 `G2-3`（batch） |
| **P3** | §7.2 多 GPU / Tensor Parallel | P3 | 未触发 | 开发机为单卡，无验证环境 |
| **P3** | §10.2 ONNX 子图替换 | P3 | 触发 = 有明确性能目标；**前提取决于 PF-7**（可复现测量方法本身已就绪——G6 已由 §11 交付） | 两次测量方向相反（±25% < 构建间噪声）→ 先跑 **PF-7**（ONNX vs 原生 prefill，≥3 次构建 / ≥20 次推理）拿到可复现对照，再谈收益。PF-7 是**从 §11 移交过来**的（2026-09-27），命令见开发计划 §11.9 / 测试计划 §10.2 |
| **冻结** | §1.1 ResNet18 INT8 校准 | **P1 → 冻结** | **触发已消失**：ResNet18 的 INT8 已由 **Q/DQ 显式量化**交付（`PROGRESS.md` §3.0d）；且 TRT 10.12 起 `kINT8` / `IInt8Calibrator` 路线弃用 | `mini_trt_llm` 内**没有任何 calibrator 代码**（已核实），现行路线是 `tools/convert/quantize_resnet18.py`。保留备查：若将来要重拾 implicit calibration，须先重审（`TROUBLESHOOTING.md` #27） |
| **冻结** | §9.1 PagedAttention 的 Prefill 阶段 | **P1 → 冻结** | **触发条件已被现状否证**：原文说"Phase 2 若走单引擎就必须先补"，而 Phase 2 之后的 `LLMRunner` 构造收 **prefill + decode 两个引擎**（`llm_runner.hpp:56`），双引擎路径一直成立 | 保留备查：只有将来要合成单引擎时才需要补 Prefill kernel |

</details>

### 0.2 [OI-LABEL-DIFF] 与旧标签的差异（本次重定，2026-09-26）


<details><summary>展开：0.2 [OI-LABEL-DIFF] 与旧标签的差异（本次重定，2026-09-26） 全文</summary>

| 条目 | 旧 | 新 | 理由 |
|---|---|---|---|

| §1.1 ResNet18 INT8 校准 | P1 | **冻结** | 已被 Q/DQ 路线取代，且该 API 路线在 TRT 10.12 起弃用（详见 §0.1 行） |
| §1.4 GPT-2 FP16 端到端 | **（缺标签）** | P3 | 文档头声明"每一项标注优先级"，但它漏了 → 补齐；P3 依据是 sm_75 无 Tensor Core |
| §2.1 显存池 | P1 | P2 | P1 应留给"能力门槛"；本条是性能优化且收益未测。**2026-09-27 补：已测（PF-5）→ 触发不成立 → 不排期**（见 §0.1 与 §2.1 正文） |
| §2.2 FlashAttention / FlashDecoding | P3 | **P2**（2026-09-27 再调） | **触发条件已由实测成立**：长上下文下 attention ≈ 每步 80%，且机制定位为"每层只有 12 个 block、串行走完上下文"→ 延迟受限。原 P3 的理由（未量基线）不再成立；仍属自研 kernel（sm_75 无官方实现），故未直接升 P1 |
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

</details>

### 0.3 [OI-EXEC-ORDER] 执行序列（"按优先级逐个完成"的建议顺序，2026-09-27 刷新）

排序规则（**同级别内**）：**前置能否立即满足 → 解锁多少下游 → 成本**。
"需批准"列指按 `AGENTS.md` §0.2 / §0.3 必须先取得许可的动作。

<details><summary>展开：0.3 [OI-EXEC-ORDER] 执行序列（"按优先级逐个完成"的建议顺序，2026- 全文</summary>


| # | 事项 | 级别 | 前置 / 成本 | 需批准 |
|---|---|---|---|---|
| ~~**1**~~ **已交付（2026-09-27）** | ~~**decode 端到端性能画像**（按 **G6** 口径，`nsys`）+ 顺带落地 **§6.3** 的 profile target~~ —— **§11 已关闭**，它要回答的三个问题**都有答案**：sampler 占比（**PF-8**：greedy 1.25% / top-k 17.7% / top-p 20.1%）、attention 占比（**PF-9**：长上下文每步 ≈80% → 触发 §2.2）、显存分配（**PF-5** → 见第 4 步）。**剩下的**：GPU 时间线（`nsys`/`ncu`）| P2 | 无外部资源；真机 1~2 往返。**能力边界**：本机（WSL2）拿不到 GPU kernel 时间线 → 逐 kernel 分解（attention vs MLP）记为能力边界（`TROUBLESHOOTING.md` #41）。落地口径见开发计划 §11.3 / `PROGRESS.md` §3.0h | ~~是~~ 已执行 |
| 2 | §9.3 采样器参考数据固化（`scripts/ref_sampler.py` 输出落成 `.bin` 供 C++ 载入） | P2 | 无外部资源；纯 host | 否 |
| 3 | §10.1 ONNX / 原生 I/O 契约统一（ONNX 侧加 `Cast` 把 `input_ids` 降到 INT32） | P2 | 无外部资源；为"ONNX 路径接进 `LLMRunner`"铺路 | 否 |
| ~~4~~ **已量完** | ~~§2.1 显存池~~ —— **已经跟着第 1 步同一次真机往返量完了**（PF-5：`cudaMalloc` 63 次 / 3.335 ms），结论 = **触发未获支持、不排期** | 已量完 | 唯一的成本已在第 1 步里出过 | 否 |
| 5 | §2.4 CV 动态分辨率 | P2 | 无外部资源，但**需要一个明确需求口径**（当前无人要求变分辨率） | 是（需求确认） |
| 6 | §2.3 Continuous Batching（含前置 G2-3：把 `LLMRunner` 从 `batch = 1` 扩开） | P2 | 无外部资源；工作量偏大 | 否（代码） |
| 7 | §1.2 LLM INT8 / INT4（含前置：`PagedAttentionPlugin` 支持 INT8 KV cache） | P2 | 无外部资源；INT4 需先评估 sm_75 kernel 可行性 | 否（代码） |
| 8 | §6.4 CI 自动化 | P2 | 本地脚本部分无外部资源；**GitHub Actions 属联网** | 部分 |
| 9 | §1.6 整条（INT8 绝对误差判据） | P2 | **需联网**下载带真值标签的验收集 | **是** |
| 10 | §3.1 / §3.2 / §3.3 / §3.4（seq2seq / ViT / 多模态 / Audio）、§4.1 / §4.2（CV 归一化 / LLaMA 类激活） | P2 | 需目标模型权重（多需联网） | **是** |
| 11 | §6.1 / §6.2（转换工具增强 / ONNX custom op） | P2 | 按需；§6.2 是 §10.2 的前置 | 视情况 |
| ~~12~~ **已结案（2026-09-27）** | ~~**§1.5**（P4-INT8-a：per-channel 整网退化根因）~~ —— **根因已定位 + 离线反证 + 真机确认**，见 `TROUBLESHOOTING` #46 / #47、开发计划 §13 | 已结案 | 实际成本：**0 次联网**；离线全在 CPU 上跑完，真机只 1 次往返（B1 四条全绿）。**产物身份已钉死**；切默认源 / 重生成 per-channel 两件**暂不做**（触发 = §1.6 验收集到位，见 §13.11） | 将来做那两件时**是**；本轮生产代码与默认产物**未动** |
| 13 | P3 其余（§1.3 BF16 / §1.4 GPT-2 FP16 / §5.2 tiktoken / §5.3 多模态模板 / §7.1 / §7.2 / §10.2）与 **SP-1** | P3 | 等触发：硬件 / 需求 / 测量方法 | —— |
| ~~**14**~~ **已交付** | ~~**§2.2 长上下文 attention**~~ —— 已落地并真机验证：斜率降幅 **88.85%（PP-1）/ 88.12%（PP-2）**、token 与单趟一致、`max_rel=1.5e-06`；计划 = 开发计划 **§12** / 测试计划 **§11** | 已交付 | 图版本已 bump 到 2（旧引擎失效一次，属预期）；`kPagedAttentionPluginVersion` 1→2 | —— |
| —— | §1.1（implicit 校准）、§9.1（PagedAttention Prefill） | 冻结 | 已被取代 / 触发条件已被现状否证 | —— |

**读法（2026-09-27 刷新）**：第 **1**（decode 性能画像 + §6.3 target，**已交付**）、
**2**（§9.3 采样器参考数据）、**3**（§10.1 I/O 契约统一）是"不需要任何外部资源"的一档；
第 **4** 步（§2.1 显存池）**已随第 1 步量完并判定"不排期"**；
第 5~8 步需要代码工作量但无外部资源；第 9~11 步需要联网或新模型资产；
第 12 步（§1.5）**已结案**；第 13 步与冻结项**不主动开工**。
也就是说：**这一档里还剩第 2 / 3 步**（零前置、纯 host，但**不服务任何现存需求**——
§10.1 的真正受益者是 §10.2，而 §10.2 要先跑 PF-7），其余**全部等触发**。

---

</details>

### 0.4 [OI-ID-HOME] 存量 `OI-` 编号的落点（新增编号的唯一归属）

**规则**（出处 `docs/README.md` §4.3 的存量例外）：**新增**的开放项编号只在本文件定义。
下面 16 个是 2026-09-26~27 把"计划段 / 用例段 / 验收段 / 结果段"也各自编号留下的**存量**，
**不改名**（§4.2 的存量豁免）；改这些标题时同步本表。

| 存量 `OI-` 编号 | 定义落点 | 对应本文件的条目 |
|---|---|---|
| `OI-RUNBOOK` | 开发计划 §8 | §0.1 全表（真机执行基建） |
| `OI-RUNBOOK-RESULTS` | 开发计划 §9 | 同上（执行结果回填） |
| `OI-RED-ARGMAX-CASE` | 开发计划 §8.5 | 对应 `PROGRESS.md` §5.13 的历史红 |
| `OI-SAMPLER-KERNEL-PLAN` | 开发计划 §10 | §9.2 [OI-SAMPLER-KERNEL] |
| `OI-SAMPLER-KERNEL-ACCEPTANCE` | 开发计划 §10.5 | 同上 |
| `OI-SAMPLER-TOPP-TAIL` | 开发计划 §10.12 | 同上（P9_2-5b） |
| `OI-SAMPLER-KERNEL-TESTS` | 测试计划 §9 | 同上 |
| `OI-PERF-PROFILE-PLAN` | 开发计划 §11 | §6.3 [OI-NSIGHT-TARGET] |
| `OI-PERF-PROFILE-DECOMPOSITION` | 开发计划 §11.4 | 同上 |
| `OI-PERF-PROFILE-TESTS` | 测试计划 §10 | 同上 |
| `OI-FLASHDECODING-PLAN` | 开发计划 §12 | §2.2 [OI-FLASHDECODING] |
| `OI-FLASHDECODING-TESTS` | 测试计划 §11 | 同上 |
| `OI-INT8-PERCHANNEL-PLAN` | 开发计划 §13 | §1.5 [OI-INT8-PERCHANNEL] |
| `OI-INT8-PERCHANNEL-DESIGN` | 开发计划 §13.3 | 同上 |
| `OI-INT8-PERCHANNEL-ARTIFACTS` | 开发计划 §13.11 | 同上 |
| `OI-BPE-TOKENIZER-TESTS` | 测试计划 §2.1 | §5.1 [OI-BPE-TOKENIZER] |

---

## 1. 量化与精度优化

### 1.1 [OI-INT8-CALIB] ResNet18 INT8 校准

- **优先级**：**冻结**（原 P1，2026-09-26 重定为冻结，理由见 §0.1）
- **背景**：现有 `0_resnet18_onnx` 已支持 INT8，含 `calib_data/` 与 `Int8Calibrator`。`mini_trt_llm` 替换后需补齐该能力。

<details><summary>展开：1.1 [OI-INT8-CALIB] ResNet18 INT8 校准 全文</summary>

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

</details>

### 1.2 LLM INT8 / INT4 量化

- **优先级**：P2
- **背景**：本机 GPU 为 **TU116（GTX 1660 Ti Mobile）**——`sm_75` **指令集**含 INT8 张量指令，
  但该芯片**没有 Tensor Core 硬件单元**（`AGENTS.md` §1）。因此 LLM INT8 的收益来自
  **显存带宽**（decode 访存受限，权重 4 字节 → 1 字节），不是张量核心吞吐。
  **更正记录（2026-10-01，作者确认）**：原文写"sm_75 有 INT8 Tensor Core"，与该芯片事实不符；
  **决策不变**（仍走显式 Q/DQ），只改收益来源的表述。
- **工作内容**：
  - 支持 INT8 weight-only 或 SmoothQuant。
  - 支持 GPTQ/AWQ 需评估 sm_75 兼容性（可能不支持部分 INT4 kernel）。
- **关键依赖**：PagedAttention Plugin 需支持 INT8 KV Cache。

### 1.3 BF16 原生计算支持

- **优先级**：P3
- **背景**：当前 sm_75 无 BF16 Tensor Core，BF16 权重会转换为 FP32/FP16。
- **工作内容**：若未来迁移到支持 BF16 的 GPU（Ampere+），可直接启用 BF16 engine。

---

### 1.4 [OI-GPT2-FP16] GPT-2 的 FP16 端到端（需要激活缩放 / 关键算子保 FP32）

- **优先级**：P3（2026-09-26 补标签：本条原先漏标；依据 = sm_75 无 Tensor Core，收益有限，
  且触发条件是"真要推进低精度推理"——见 §0.1）
- **2026-10-06 状态**：**已转正式条目 `REQ-018`，并撤销"按政策不修"**——依据是**正确性**
  （默认构建精度不可用即产品缺陷），**不是本节的"低精度性能"触发条件**；本节的三条路线与
  "收益有限"的判断本身仍然成立。执行与现状以 `docs/dev/REQ-018-gpt2-fp16-nan/STATE.md` 为准，
  追加留痕见 `TROUBLESHOOTING.md` + 18.2。

<details><summary>展开：1.4 [OI-GPT2-FP16] GPT-2 的 FP16 端到端（需要激活缩放 / 关 全文</summary>


**现状（已实测）**：真实 GPT-2 在本项目的**弱类型 FP16** 引擎下端到端产生 NaN，
且出现 NaN 的层随构建变化（0/1/2），而激活幅值远未触及 FP16 上限。FP32 端到端完全正确。
完整证据链与"为什么按政策不修"见 `docs/TROUBLESHOOTING.md` + TS-018-FP16-NAN。
（**该"不修"已于 2026-10-06 由作者撤销**，见本节上方的状态行。）

**要解决它，可行方向**（按成本从低到高）：

1. **关键算子显式保 FP32**：残差相加、softmax、归一化（LN 已做）设 `setPrecision(kFLOAT)`，
   FP16 只用于 matmul；代价是每层多几组 cast；
2. **逐算子二分到具体算子**：把第 0 层式的"切点导出"扩到前若干层（仪器已具备），
   先定位到算子再决定改哪个——预计 2 轮以上真机往返；
3. **激活缩放**（outlier 处理）：属研究性质，与 INT8/低精度量化同一课题（见 §1.2）。

**触发条件**：真要推进 GPT-2 的 FP16/低精度推理性能时（目标硬件无 Tensor Core，
收益有限，所以不急）。

</details>

### 1.5 [OI-INT8-PERCHANNEL] per-channel 权重量化的整网退化根因（P4-INT8-a 立项）

> **2026-09-27 状态：根因已定位，并已在本机（CPU）用独立标尺完成"复现 → 修复 → 反证"。**
> 结论：**根因在产图脚本，不在 TRT** —— `quantize_resnet18.py` 的权重范围取自

<details><summary>展开：1.5 [OI-INT8-PERCHANNEL] per-channel 权重量化的整网退化 全文</summary>

> **未折 BN 的 torchvision 权重**，而 Q/DQ 插在**已折 BN 的 ONNX 权重**上（折叠系数逐通道跨度
> 实测 **0.05 ~ 19.9**）；per-tensor 只错一个全局倍率（后果轻），per-channel 错的是**逐通道倍率**
> → 系数 >1 的通道被 clamp 饱和。把 scale 改成取自"**被量化那张张量**"（`--weight-range-source onnx`）
> 后，per-channel 在 FP32 余量子集上从 **54.5% → 100%**。
> 完整证据链与逐层曲线见 `docs/TROUBLESHOOTING.md` **#46**（含被推翻的旧结论两条）；
> 执行计划见 `future_iterations_development_plan.md` **§13**。
> **文件级证据（与后端无关）**：per-channel 的 int8 权重常量是 Python 侧算好写进 ONNX 的，
> 读文件数被 clamp 到 ±127 的比例——**PT 3.919% / PC(错源) 16.188% / PC(改源) 0.044%**
> （对称量化下每通道约 1 个才对）。**坏值已经烘进文件**，任何忠实后端都会复现同样的退化。
> **真机 B1 四条：全绿（2026-09-27）**——确定性 22 个张量逐位相同；探针自证 `conv1`
> `d_pre=8.34e-07` vs `d_post=0.0398`；**引擎 vs 自己的图**逐层最大 `max_abs` 0.2714（未越界 ⇒
> **TRT 忠实执行了那张图**）；探针图复现对照 **PT 12/12 = 100% / PC 6/12 = 50%**（与正式产物
> 口径一致）。（首跑曾 3 条全红，红在**用例自身的绑定**——`ICudaEngine::getTensorShape` 对动态维
> 返回 -1；同次运行还量到**探针图会把 `i8i8` tactic 从 4 变成 0**。见 `TROUBLESHOOTING.md` #47。）
> **产物身份已钉死（2026-09-27 作者指示）**：`resnet18_qdq.onnx` = **正式产物**；
> `resnet18_qdq_per_channel.onnx`（及其探针图）= **#46 的复现样本，不是候选基线**（按"错源"生成，
> 16.19% 的 int8 权重被 clamp 饱和；B1-4 要的就是"它更差"）。
> **两件事项暂不做**（都无外部前置、也都不是默认行为）：① 把 `--weight-range-source` 默认切到
> `onnx`（实测把饱和权重从 3.919% 降到 0.000%，但判据与一致率**完全没变** → 换默认的理由只能是
> "更对"、不能是"更准"，且要连带重测回填全套 INT8 数字）；② 重生成 per-channel 产物
> （零功能收益，且它是 B1-4 的承重件，重生成会让 B1-4 变红，必须与"退役/改写 B1-4"打包做）。
> 触发条件 = **先有 §1.6 的验收集**（当前批判别力不足）。展开见开发计划 **§13.11**、
> `PROGRESS.md` §3.0j、`TROUBLESHOOTING.md` §46.4。

- **优先级**：**P3 → 已开工并基本结案**（不阻塞任何现行路径——正式产物仍定为 per-tensor 权重，
  且默认行为**逐字节未变**。剩下的两件"要作者拍板"的事：① 是否把产图脚本的默认
  `--weight-range-source` 切到 `onnx`；② 是否重生成 per-channel 产物）
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
  `docs/TROUBLESHOOTING.md` + TS-029 / #30 / #31）。
  - **2026-09-27 修订**：上表"整网模拟（同批 scale）：余量子集 100% / 100%"这一行是**错的**——
    那个"模拟"拿未折 BN 的权重做量化，与图里被量化的张量不是同一个（#46.3）。
    用 ONNX 官方参考实现直接执行同一张图，per-channel 的余量子集一致率就是 **54.5%**
    （与"真机 TRT"那行的数字逐位相同）；把 scale 的来源改成被量化的张量后回到 **100%**。
    "已否证 11 条"里的**第 3 条（模拟不忠实）应当作废**——当时用来否证它的模拟同样不忠实。
- **目标**：给出**结论**，二选一即可，但不允许"下次再看"：
  1. 定位到**从第几层开始分叉**并说明机制（该机制还要能被算子级 / block 级最小复现解释）；
  2. 或证明所有可查方向均已查空，逐条写明"为什么这条路不能再查"。
> **2026-09-27 执行说明**：下面第 1 条把标尺写成"与 torch **已折叠 BN** 的模型同点对拍"；
> 实际执行时换成了**更强的形式**——用 ONNX 官方参考实现**直接执行那张 Q/DQ 图**
> （图里 BN 已折好，压根没有"折叠"这一步，从源头消掉 #30.5 那类探针自身的错）。
> 第 1~3 条均已执行完；第 4 条（更大验收集）未做，因为根因已经用前 3 条定位到了。

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

  **2026-09-27 判定：走上分支一，且比预期更强——分叉从第 0 层（`conv1`）就开始，
  机制是"尺子量 A、裁剪 B"**（权重范围来自未折 BN 的权重，量化对象是已折 BN 的权重）。
  它同时满足两条验收：① 机制被**算子级最小复现**支持——只要 scale 与被量化张量一致，
  单卷积/最小 block/整网三级的 per-channel 都不差（`--weight-range-source onnx` 的
  A/B 就是这条最小复现的整网版）；② 结论已写进 `TROUBLESHOOTING.md` #46，本节状态已改。
- **前置依赖**：无。第 1~3 步不需要联网、不需要新数据（第 4 步依赖 §1.6）。
- **成本校准（前次实测）**：全流程是**每轮 1 次真机往返**级别；前次 #30.5 用掉 3 轮，其中 2 轮
  耗在**探针自身的 BN 折叠错**上——这次先修仪器再取数。
- **出处**：`docs/TROUBLESHOOTING.md` #29.2 / #30（#30.1 否证表、#30.3 未排除方向、#30.5 仪器坑、
  #30.6 收口决定）；`docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` + PH4-INT8-RESULTS 的 P4-7-3 行。

</details>

### 1.6 [OI-INT8-ABS] INT8 精度判据与带真值标签的验收集（P4-INT8-b 立项）

- **优先级**：P2
- **现状（实测）**：当前判据是**分层的**——主判据 = "FP32 有余量"（`margin >= 5`）子集的

<details><summary>展开：1.6 [OI-INT8-ABS] INT8 精度判据与带真值标签的验收集（P4-INT8- 全文</summary>

  top-1 一致率 `>= 90%`（实测 12/12 = 100%）；整体一致率 `>= 30%` 只作"没崩坏"下界（实测 37.9%）。
  三个已知弱点：
  1. `calib_data` 是 **tiny-imagenet 放大图**，分类本身退化（8 张里 5 张同类）——**没有真值标签**，
     所以现在只能判"与 FP32 是否一致"，判不了"对不对"；
  2. 有判别力的样本只有 **11~12 张**，率的分辨力弱（Fisher 精确检验 ≈0.03，勉强算显著）；
  3. **绝对误差界未定**：实测 `max_abs ≈ 21.6` 被少数样本放大，所以**故意不拿它当判据**
     （`docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` + PH4-INT8-CRITERIA 的"数值上界"行）。
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
  4. 阈值定稿后回写 `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` + PH4-INT8-CRITERIA 与 `docs/dev/REQ-007-resnet18/phase4_test_plan.md` 的 R2.6，
     并用**同一把尺子**重测一次 per-channel vs per-tensor（这对 §1.5 也是判据输入）。
- **验收判据**：
  - 每个阈值旁边能回答"凭什么这么定"（本条目实测的分布 + 样本量 + 分层口径）；
  - 验收集本身可复现（meta 里查得到来源与 SHA256）；
  - 明确写出**本判据不覆盖什么**（例：只覆盖该验证集分布内的分类一致率，不构成对任意输入的
    数值保证）。
- **前置依赖**：**需要联网下载数据集**（`AGENTS.md` §0.2：联网操作必须另行单独获批）。
  未获批时可先做本条的"口径定义"部分（验收集规格 + meta 字段 + 分层统计脚本），不下载数据。
- **不许做的事**：不许为了"绿"而调阈值或删断言（`AGENTS.md` §7）；不许把本条的阈值套到
  FP16 / FP32 上（那是跨精度复用，见 `docs/PROGRESS.md` + DEC-EVIDENCE-DISCIPLINE A）。
- **出处**：`docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` + PH4-INT8-CRITERIA（判据表）/ §5（风险与回退）/ §7 的 P4-7-3 行；
  `docs/TROUBLESHOOTING.md` #29.4 / #30.5。

> **为什么拆成两条**：§1.5 是**仪器 / 机制**问题（第 1~3 步不依赖新数据），§1.6 是**判据 / 数据**
> 问题（依赖联网）。两者共用一份验收集，但开工前提不同；合成一条会让"不联网就一步都动不了"。

</details>

## 2. 性能优化

### 2.1 显存池替换简单封装

- **优先级**：P2 → **不排期**（原 P1；2026-09-26 下调为 P2：收益未测、且现行设计已避开 decode
  循环内分配；2026-09-27 **已测（PF-5）→ 触发不成立**，见下表）

<details><summary>展开：2.1 显存池替换简单封装 全文</summary>

- **背景**：Phase 0 的 `DeviceBuffer` 仅封装 `cudaMalloc/cudaFree`，频繁分配有性能开销。
- **触发条件（2026-09-27 已量，结论 = 不成立）**：原定"先量出分配开销确实显著"。**已量**——
  §11 的 **PF-5** 用 `summarize_nsys.py --api` 在真实 decode 上量到：全程 `cudaMalloc`
  **63 次 / 3.335 ms**、`cudaFree` 69 次 / 249.7 ms（后者含隐式同步，是**拆除期**成本）。
  → **触发条件未获支持**，本条目**不排期**（`PROGRESS.md` §3.0h 第 3 条是这组数的出处）。
  真要动它的前提变了：得先证明"除分配外的热点"或"长驻服务的分配抖动"确实咬人。
- **现状**：`DeviceBuffer` 仍是直接 `cudaMalloc`（`src/utils/memory_pool.cpp:46`），
  但热点路径上没有循环内分配（`AGENTS.md` §3.A.3 本来就不允许）。
- **工作内容**：
  - 实现基于 freelist 或 arena 的 `MemoryPool`。
  - 支持按大小分桶、按 stream 隔离。
  - 替换 `DeviceBuffer` 内部实现，保持接口不变。

</details>

### 2.2 [OI-FLASHDECODING] FlashAttention / FlashDecoding

- **优先级**：~~P2~~ **已交付**（2026-09-27 收口；P3 → P2 → 已交付，与 §0.1 一致）——
  上调的依据是触发条件**已由实测成立**：长上下文下 attention 占每步约 80%；
  原 P3 的理由"未量基线"已不成立。见下方"现状"。

<details><summary>展开：2.2 [OI-FLASHDECODING] FlashAttention / FlashD 全文</summary>

- **执行计划（2026-09-27 成文）**：开发计划见 `docs/future_iterations_development_plan.md` **§12**
  （计划对账 / 设计决策 D1~D5 / 任务分解 P2_2-0~9 / 验收判据 / 风险与回退 / 破坏性动作清单 /
  待拍板 F1~F4 / 真机清单），测试计划见 `docs/future_iterations_test_plan.md` **§11**
  （用例 → 判据 → 出处 → 环境 → 状态）。
  **状态（2026-09-27 收口）：已交付。** 逐条对照见开发计划 §12.4——第 5 条拆成
  5A（斜率降幅，✅ **88.85%（PP-1）/ 88.20%（PP-2，两次复现）**）与
  5B（短上下文退化，**经作者决定改为观测项、不设判据**，实测 +1.877%，见下）。
  已拍板 F1~F4 全按 A、`future_iterations_development_plan.md` §12.8 的 7 条破坏性动作全批；P2_2-1 ~ P2_2-7 已落盘
  （`paged_attention_split.hpp` + 两个 kernel + 17 条新用例，含 PP-1/PP-2 两把同 session 尺子），
  沙箱 **259 条 / 0 失败**（**这是当时（2026-09-27 收口）的套件总数**；现为 **264**，
  多出的是 §1.5 的 2 条 host 自检 + 3 条 `Int8Probe*`，见 `PROGRESS.md` §3.0j），
  图版本已 bump 到 2。
  **真机验证（2026-09-27）**：交付时是 **259 条 / 1 红 / 1 跳过**（当时总数）；
  **同日整轮全量复跑：264 条 / 1 红 / 1 跳过**（449 s，`沙箱同为 264 / 0`），唯一红 = 按设计的
  `RealGpt2Fp16GreedyMatchesReferenceTokens`（`PROGRESS.md` §5.11 的 FP16 NaN）、跳过 = `int8_crosscheck` 缺报告。
  → **2026-09-27 复跑（provenance 改造成 `路径 + ID` 形式之后）：264 条 / 1 红 / 0 跳过 / 310 s**——
  先跑 §8.3 的 C-1/C-2 产出报告后，`int8_crosscheck` 执行并通过。
  本轮的新用例**全部通过**：
  ① **FP32 多片**：`PagedAttentionSplitKernelTest.*`（MHA / GQA / MQA / batch>1 / 空片 /
     `context_len=0`）对 double 参考全绿 → **否证了"归并丢片（只算一部分上下文）"**；
  ② **FP16 多片**：`Fp16PathTest.PagedAttentionSplitMatchesFp16Reference`（2 片与强制 8 片）；
  ③ **端到端**：`Gpt2DecodeConsistency.*` 与 FP32 8-token 冻结基线通过 → 生产路径经 TRT 正确；
  ④ **两把同 session 尺子**（PP-1 kernel 级 / PP-2 端到端 A/B）已实现并在真机跑通
     （0.71 s / 11.66 s）。
  首轮那 4 条红的根因**全在测试侧**（自相矛盾夹具 + 一条写反的 GQA 断言），
  修法与教训见 `TROUBLESHOOTING` #44；**产品代码一行未改**。
  **F1 性能判据（真机出数）**：两把**独立**的同 session 尺子都远超 40% 的达标线——
  PP-1（kernel 级）斜率 0.902569 → 0.100609 ms/1000 位置 = **降幅 88.85%**；
  PP-2（端到端、同引擎逐轮 ABBA）10.8048 → 1.28379 = **降幅 88.12%**，且三档 prompt 的
  **生成 token 与单趟完全一致**。二者互校差约 5%（PP-1 单趟 ctx=1024 0.918948×12 = 11.03 ms/步
  vs PP-2 单趟斜率 10.8048×0.976 = 10.5 ms/步）。
  **机制证据**：PP-1 显示 ctx=32 时 split **慢 8.4%**（多一次归并发射）、256 时 1.836×、
  1024 时 **7.318×** —— 正是"上下文越长、切开越划算"的交叉曲线，7.3× 贴近 8 片的并行度上限。
  数值差异 `max_abs=3.58e-07` / `max_rel=1.52e-06`（比判据低约 65×）→ 没有任何阈值被放宽。
  **F1 的 B 半句（短上下文退化）最终定为"观测项、不设判据"**（作者 2026-09-27 决定）：
  原措辞"退化不超过同 session 漂移"三重不成立——① "漂移"有 `max`/`min`/配对差三种读法，
  前两种给出相反结论（而 `max` 那个 12.29% 来自整轮第一档、机器未进稳态，锚点本身不可信）；
  ② 曾议的绝对线"≤2%"其唯一数值输入（单次发射 3~6 µs）**无出处**；③ PP-1（kernel 级推出
  +0.95%）与 PP-2（端到端实测 +1.877%）对同一笔代价差 **1.97× 且未解释**。
  保留的观测记录：**+1.877%**，机制 = 每层多一次归并 kernel 发射（`future_iterations_development_plan.md` §12.3 D2 **在拿到数据前**
  已预估并接受该代价）；若将来要恢复成硬判据，走"**先量后定**"（先单独校出发射开销），
  **不要回头捡那个 2%**。完整账见 `TROUBLESHOOTING` **#45 / #45.1**。
  当前进度明细见开发计划 §12.4 的"执行状态"。
- **背景**：Turing sm_75 无官方 FlashAttention 优化，但可 hand-tune 基础 attention kernel。
- **现状（2026-09-27 实测，出处：`PROGRESS.md` §3.0h / 测试计划 PF-9 / 开发计划 §11.4.1）**：
  GPT-2 原生引擎、batch=1、FP32、greedy，用**上下文长度扫描**测得每步 decode 耗时
  （attn 与其他算子对上下文长度的敏感性不同，差值即 attention 的边际成本）：

  | prompt | 平均上下文 | 每步 |
  |---|---|---|
  | 4 | 20.5 | **3.055 ms** |
  | 256 | 272.5 | **6.062 ms** |
  | 960 | 976.5 | **14.705 ms** |

  两段斜率 11.93 / 12.28 **ms per 1000 位置**（差 3%→ **线性**），外推 1024 位置 +12.2 ms
  → **attention 占长上下文每步的 ~80%**。
  **机制已定位**：`LaunchPagedAttention` 的 `grid=(num_heads, batch)`
  （`src/plugins/paged_attention_plugin.cu:141`）→ GPT-2 每层只有 **12 个 block**，
  而本机有 **24 个 SM**（一半闲置）；每个 block 用 64 线程**串行**走完整段上下文
  （976 次迭代）→ **延迟受限**：有效带宽仅 ≈6 GB/s（约峰值 192 GB/s 的 3%）。
  **结论**：要动的是**把上下文维切开并行**（FlashDecoding 式 split-K + 两阶段归约），
  单纯"tile 一下"不够。sm_75 无官方实现，仍需自研 kernel。
- **工作内容（2026-09-27 按现状收敛；**交付前**的口径，保留备查）**：
  - 在 `PagedAttentionPlugin` 中把**上下文维**切成多个 block 并行，再做两阶段 softmax 归约；
    `num_splits == 1` 时仍走原单趟 kernel，短上下文不退化成两次发射（开发计划 §12.3 D2）。
  - ~~支持 decode 阶段的 batching 优化（与 `G2-3` 的 batch 扩展一起考虑更省事）~~ ——
    **这句与现状矛盾，已作废（2026-09-27）**：`PagedAttentionDecodeKernel` 的
    `grid.y = args.batch_size`，**kernel 本来就支持任意 batch**
    （`PagedAttentionKernelTest.MhaMatchesCpuReferenceAcrossMultipleBlocks` 已覆盖 `batch=2`）；
    真正把 batch 卡在 1 的是 `LLMRunner`（缺口 **G2-3**，属 §2.3），**不在本条范围**。
    本条范围因此收敛为"只做上下文维 split-K，不扩 batch"——把两个变量混在一起会让
    性能读数无法归因（`PROGRESS.md` §2.14 C）。改动原因记在开发计划 §12.1 的"反向查"。
- **验收判据**：形式见开发计划 **§12.6**（数值判据**不放宽**·语义不变·长上下文斜率下降），
  其中性能"达标线"的数值由开发计划 §12.9 的 **F1** 拍板 —— **已拍板**：达标线 ≥40%，
  实测 **88.85%（PP-1）/ 88.20%（PP-2）**；F1 的 B 半句（短上下文退化）经作者决定改为观测项、
  不设判据（实测 +1.877%），完整账见 `docs/TROUBLESHOOTING.md` #45 / #45.1。

</details>

### 2.3 Continuous Batching / In-Flight Batching

- **优先级**：P2
- **背景**：服务化部署时，continuous batching 可大幅提升 GPU 利用率。
- **工作内容**：
  - `LLMRunner` 支持多请求队列调度。
  - `PagedKVCache` 支持跨请求动态分配与回收。

### 2.4 [OI-CV-DYNAMIC-RES] CV 动态分辨率

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

### 5.1 [OI-BPE-TOKENIZER] BPE Tokenizer（GPT-2 原生）

- **优先级**：**已完成**（2026-09-26 批次 A1 交付；原 P1）——交付内容与实测见 `PROGRESS.md` §3.0e，
  用例见 `future_iterations_test_plan.md` §2.1（含"文本 → prompt token"桥接用例）
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

### 6.3 [OI-NSIGHT-TARGET] Nsight 一键 Profile Target

- **优先级**：**已交付**（2026-09-27；原 P2，理由与状态见 §0.1 与开发计划 **§11.5.2**）
- **内容**：

<details><summary>展开：6.3 [OI-NSIGHT-TARGET] Nsight 一键 Profile Targe 全文</summary>

  - CMake 增加 `profile_gpt2`、`profile_resnet18` 自定义 target。
  - 支持 `nsys profile` 与 `ncu` 导出 `.ncu-rep`。
- **交付情况（2026-09-27）**：四个 target（`profile_{gpt2,resnet18}[_ncu]`）+ 一键脚本
  `tools/profile/run_profile.sh` + 分桶脚本 `tools/profile/summarize_nsys.py`（含自检）。
  执行安排与收口见开发计划 **§11** / 测试计划 **§10**。
- **能力边界（2026-09-27 实测更正）**：导出本身可行（target / 脚本 / 分桶都能跑），
  但**本机（WSL2）采不到 GPU kernel 数据**——`nsys` 报告里只有 CUDA API / OSRT / NVTX，
  `ncu` 直接报 `Unknown Error on device 0`；加 `sudo` 与显式 `--trace=cuda` 都无效；
  **把 `.nsys-rep` 拷到 Windows 宿主用 GUI 打开也没用**（数据压根没被采集，不是查看器的问题）。
  要 kernel 级数据得换一台有 GPU 跟踪能力的机器。逐 kernel 分解在本机改用
  **"同 session 比值 + 上下文扫描"**（开发计划 §11.4.1、`TROUBLESHOOTING.md` #41）。
  **只导出、不调优**这条不变。

  > **更正记录（2026-09-27）**：本条本日早些时候加的一句"无头导出（`-o`）+ 宿主机 GUI 是主路径"
  > 已按上条实测**作废**——那份报告里没有 GPU 数据，换查看器救不了（#41）。

</details>

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

## 9. [OI-PHASE1-DEFERRED] Phase 1 明确延后的能力

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

### 9.2 [OI-SAMPLER-KERNEL] Sampler 的手写高性能 kernel

- **计划（2026-09-26 成文）**：开发计划见 `docs/future_iterations_development_plan.md` **§10**、
  测试计划见 `docs/future_iterations_test_plan.md` **§9**（含计划对账 / 任务分解 P9_2-0~8 / 验收判据 /

<details><summary>展开：9.2 [OI-SAMPLER-KERNEL] Sampler 的手写高性能 kernel 全文</summary>

  破坏性动作预告 / 真机清单）。
  **状态（2026-09-27 收口）：P9_2-0 ~ P9_2-8 全部落地**——
  - **Top-P（P9_2-5，主要收益）**：保留 CUB 排序、把行内数学块内并行；**真机全量 235 条 / 1 红**
    （唯一红 = 按设计的 GPT-2 FP16 NaN）；**性能对 `future_iterations_development_plan.md` §10.5 冻结基线 4/4 达标（13.0 / 18.9 / 13.4 / 15.0 ×）**，
    同 session 口径 `50257×1 = 9.99×`（差 0.08%，机制已定位 = 收尾段串行重扫的延迟）→
    剩余部分记作可选优化 **P9_2-5b**（见 §11）。
  - **Top-K 快速路径（P9_2-2~4）**：正确性真机通过（与 legacy 逐 token 相同），但**性能不达标**
    （慢 6~9×）→ **已从生产路径撤下**，库内保留待重做（根因与重做方向见开发计划 §10.10）。
  - **判据缺口（P9_2-6/7）**：FP16（S-12/S-13）与大词表（S-14/S-15）覆盖补齐，
    **`§11` 的 P1.5-a 据此关闭**。
  数据与出处：`docs/future_iterations_development_plan.md` + OI-SAMPLER-KERNEL-ACCEPTANCE（基线）/ §10.9.1（复测与两个口径）/
  `docs/TROUBLESHOOTING.md` + TS-035（性能归因）、#36（一条判据红的定位）。
- **优先级**：**已关闭 / 已交付**（2026-09-27；原 P1 → P2 → 已交付）。
  **下调时给的触发条件（"profile 从未做过"）已经补上**：sampler 在整步 decode 里的占比
  已由 PF-8 在同 session 内量出并与 `SamplerPerf` 互校一致（greedy **1.25%** /
  top-k **17.7%** / top-p **20.1%**，见开发计划 §11.5.1、测试计划 §10.5、`PROGRESS.md` §3.0h）。
  下面"背景 / 工作内容 / 触发时机"保留原文，仅作历史记录
- **背景**：Phase 1 用 CUB 分段排序保证正确性（决策 Q15），Top-K / Top-P 目前每步都要对整行
  `vocab_size` 做一次降序排序，是明显的性能瓶颈（`vocab_size` 可达 128K）。
- **工作内容**：warp-level Top-K 选择（无需全排序）、bitonic sort、以及与后续 continuous batching 的配合。
- **触发时机**：Phase 2 跑通端到端吞吐后，用 `nsys` / `ncu` 定位到 sampler 占比显著时。
- **触发条件现状（2026-09-26 核实）**：**这个 profile 至今没做过**——
  `docs/PROGRESS.md` 里查不到任何 decode 阶段的 sampler 占比（当时唯一的端到端数字是
  `CVRunner` 的 benchmark `mean≈8.4 ms`，见 `docs/dev/REQ-007-resnet18/phase4_development_plan.md` 的 P4-3 行）。
  触发条件"未验证"不等于"已成立"，故不能按 P1 对待。

</details>

### 9.3 [OI-SAMPLER-REFDATA] 采样器参考数据固化为数据文件

- **优先级**：P2
- **背景**：D3 要求 Generation 层"对比分布"。Phase 1 已落地两件事：C++ 侧的统计检验
  （词频收敛到解析 softmax 概率）与 `scripts/ref_sampler.py`（打印 HF 风格截断语义）。
  尚未做的是把 Python 输出落盘成 `.bin` 供 C++ 测试直接载入，因此**与 HF 截断语义的自动化交叉验证缺失**。
- **工作内容**：脚本导出 nucleus 集合与概率，测试读取并比对；顺带覆盖 `p` 接近 1 的边界。

---

*文档版本：v1.0*  
*关联文档：`docs/mini_trt_llm_design.md`、`docs/dev/REQ-001-bootstrap/phase0_development_plan.md`*

## 10. Phase 3 明确延后的能力

### 10.1 [OI-IO-CONTRACT] ONNX 路径与原生路径的 I/O 契约统一

- **优先级**：P2（2026-09-26 补标签：原文漏标；触发条件 = ONNX 路径要接进 `LLMRunner`，见 §0.1）


<details><summary>展开：10.1 [OI-IO-CONTRACT] ONNX 路径与原生路径的 I/O 契约统一 全文</summary>

**现状**（`docs/TROUBLESHOOTING.md` + TS-017）：两条路的输入契约不一致——

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
ONNX 路径接进 `LLMRunner`（`docs/dev/REQ-006-gpt2-onnx/phase3_development_plan.md` D3 已明确本阶段不做）。

</details>

### 10.2 [OI-ONNX-SUBGRAPH] ONNX 图的子图替换（D1=B）

- **优先级**：P3（2026-09-26 补标签：原文漏标）。**前置已变更（2026-09-27）**：
  原先写的是"前提取决于缺口 G6 的可复现测量方法"——**测量方法已经就绪**（§11 已关闭，

<details><summary>展开：10.2 [OI-ONNX-SUBGRAPH] ONNX 图的子图替换（D1=B） 全文</summary>

  交付了协议 + 工具链），所以真正还缺的是**跑一次 PF-7**（ONNX vs 原生 prefill，
  同 session ≥3 次构建 / ≥20 次推理，报中位数与极差）。PF-7 原本挂在 §11 的收口范围里，
  因为它服务的是本条、而非"decode 时间花在哪"，已于 2026-09-27 移交到这里。
  命令见开发计划 §11.9 / 测试计划 §10.2。

**为什么现在还不能判断**（**已修正**：本条原先写的是"ONNX 慢约 22%、故替换无收益"，
那是从**单次测量**里读出的结论，已被后续运行推翻）：

Phase 3 的两次测量方向相反（`docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` + PH3-L3-MEASUREMENTS）：

| 运行 | prefill(4 token) ONNX / 原生 |
|---|---|
| A | 4.545 ms / 3.717 ms（ONNX 慢 22%） |
| B | 4.716 ms / 6.250 ms（ONNX 快 24%） |

差异（±25%）小于**构建间噪声**——已知机制是 TRT 的 kernel tactic 选择依赖构建时的机器状态。
因此**"ONNX 更快还是更慢"目前是未定**，也不能据此判断子图替换的收益。
要做这个判断，先得**跑 PF-7**（可复现的测量方法本身已经建好——§11 已交付协议与工具；
PF-7 是它的正例，见测试计划 §10.2）。

**若将来要做，前置条件**：(1) 有明确的性能目标与基线；(2) 先量出"替换能省掉多少 kernel 时间"
（用 nsys/ncu 的实测，而非推断）；(3) 替换前后保留两份 engine 做数值对照。

**保持不变的部分**：`tools/inspect_onnx.py --check` 的子图识别断言应当保留——
它是"图变了要知道"的护栏（含 `absent_ops`：RMSNorm / RoPE / Attention 等一旦出现，
说明模型或导出路径变了，需要重审结论）。

</details>

## 11. [OI-COVERAGE-GAPS] 测试覆盖缺口索引（默认不排期，按触发条件决定；已立项的在 §1 / §2）

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
| ~~**G6**~~ **方法已交付** | 同上 | ~~L3 性能**无可复现测量方法**~~ **已解决（2026-09-27）** | **测量方法已建立并被实际使用**：同 session、同轮交替（ABBA）、斜率 `(T4−T1)/3`、判别下限 ±400~600 µs、报告 n/温度/构建态——协议在开发计划 §11.3，工具链在 §11.5.2，PF-8/PF-9 两次用它出结论（还自己抓到一次测量错误 #42）。**§11 已关闭**。**唯一没跑的是 PF-7**（G6 的"正例"：ONNX vs 原生的跨构建极差）→ 它服务 §10.2，**已移交到 §10.2 的前置**。另有一条能力边界：本机（WSL2）拿不到 GPU kernel 时间线，逐 kernel 分解改用"同 session 比值 + 上下文扫描"（#41） |
| ~~G2-1~~ | `docs/dev/REQ-004-gpt2-native/phase2_test_plan.md` §5 | ~~真实 GPT-2 的 FP16 端到端未测~~ **已执行**：FP16 端到端出 NaN（缓冲问题已修；数值问题**当时**按政策不修并登记为已知限制，**该决定已于 2026-10-06 撤销**） | 解决路径见本文件 §1.4（其状态行记着撤销）；复现器与诊断仪器保留在 `tests/test_gpt2_generate.cpp` |
| **G2-3** | 同上 | `LLMRunner` 只支持 `batch = 1`（有意限定） | 需要批处理时再扩（同时引入多序列 block 分配、各自 `context_lens` 与采样参数） |
| **G2-4** | 同上 | EOS 无法在循环内早停（已知 workaround，语义正确） | 见 `docs/PROGRESS.md` §5.0；若要真早停，需设备侧 stop flag + 条件图 |
| ~~G7~~ | 开发计划 §4 | ~~探针未接入 ctest~~ **已关闭**（2026-09-25）：注册为 `onnx_graph_probe`，缺环境返回 77 → ctest 报 Skipped | —— |
| ~~P1.5-a~~ | `docs/dev/REQ-003-test-infra/phase1_5_test_plan.md` §5 | ~~Top-K / Top-P 的 FP16 分支未覆盖~~ **已关闭（2026-09-27，真机）**：Greedy 早已覆盖；Top-P = S-13（`Fp16PathTest.TopPSamplingDistributionMatchesAnalyticProbabilitiesInFp16`）、Top-K = S-12（同 suite），两条**真机均通过**。判据：k=3/6（真的发生截断）的词频各按 3σ 对解析参考，且 FP32/FP16 互相在 3√2σ 内 | —— |
| **P1.5-b** | 同上 | E2 的**完整链路**（`RMSNorm → QKV → RoPE → PagedAttention → LM Head`）与 `ref_mini_block.py` 有意留后（P1.5-4 缩减完成） | 要往 LLaMA 风格链路继续做时（Phase 4 之后），或怀疑"多算子相邻契约"出问题时 |
| **P1.5-c** | 同上 | E3 只验"接受/拒绝"，未验**同 engine 内多次切换 profile 后的数值一致性** | 真的依赖多 profile 混用时（当前 runner 每步只用 profile 0） |
| ~~**P4-INT8-a**~~ **已结案（2026-09-27）** | `docs/TROUBLESHOOTING.md` #46 / #47 | ~~权重 per-channel 在整网上比 per-tensor 差得多（余量子集 54.5% vs 100%）、原因未知~~ → **根因 = 权重 scale 取自未折 BN 的权重、量化对象是已折 BN 的权重**（"尺子量 A、裁剪 B"）；改 `--weight-range-source onnx` 后 **54.5% → 100%**，真机 B1 四条全绿（含"引擎忠实"与"现象复现"两条硬证据）。完整结论见本文件 **§1.5** | —— |
| **P4-INT8-b** | `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §4 / §5 | INT8 的**数值上界判据未定**（当前只在"有判别力子集"上用一致率判，且该子集无可核对的真值标签） | 需要给出 INT8 的绝对误差保证时。**已在 §1.6 立项**（目标 / 做法 / 验收判据在那里；**前置依赖 = 联网下载验收集，须先获批**） |
| **P4-FP16-a** | `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §1.1 | **FP16 路径仍使用已废弃的 `BuilderFlag::kFP16`**（TRT 10.12 起废弃，指向 strong typing）；实测可用 | 真要迁到强类型网络时（两条 builder 的每个算子都要显式设类型，代价大） |
| **P1.5-d** | 同上 | 采样器**分布级数据未固化**（`scripts/ref_sampler.py` 只打印，输出没落成测试数据） | 要做采样的统计正确性回归时（属增强，见本文件 §9.3） |
| ~~**P9_2-5b**~~ | `docs/TROUBLESHOOTING.md` #35 / #38 / 开发计划 **§10.12** | 收尾段串行重扫候选优化：**已实施、真机正确性通过，A/B 判定 = 效果无显著差异**（同轮交替测两版：中位数 +4.8 / +27.3 / −330.5 / −81.3 µs，p25/p75 全部跨 0；预期效应 27~68 µs 低于本平台 ±400~600 µs 的判别下限）。**已关闭**：代码保留（无任何一行显示显著更慢；最坏串行尾 500→32），`LaunchTopPSamplerTwoLevel` 留作永久对照入口 | —— |
| ~~**P9_2-5c**~~ | `docs/TROUBLESHOOTING.md` #38 / 开发计划 **§10.12.8** | ~~第一趟的访存/MLP~~ **不做**：同一轮里量到 **greedy（本来就完全合并访存、只读一遍行）净成本 35~155 µs**，而 `top-p 净 − top-k 净` @50257×1 仅 **60.1 µs**（p25=56.4 / p75=64.0）→ 采样内核净开销仅"裸读一遍行"的 ~1.7 倍，**优化空间见底**；其余 ~89% 是保留的 CUB 排序 | —— |
| **SP-1** | `docs/PROGRESS.md` §3.0e（"未验证件"） | `SentencePieceTokenizer` 是**无用例、无资产、无调用方的未验证件**：`BaseTokenizer` 的另一个实现，但测试目录零引用、仓库里没有 `.model`/`.spm`、`LLMRunner` 只存基类指针（测试传 `nullptr`）→ `Load`/`Encode`/`Decode` 从未被任何参考裁决过 | 要真正用 SentencePiece 系模型（LLaMA 类）时。**处置顺序**：先造参考与用例再动实现（照 BPE 那套：golden + 来源 SHA256 + meta 自证 + 负例）；资产需作者提供或联网取（**联网须先获批**） |

**判定原则**：这些是"覆盖不足"，不是"已知缺陷"——已发现的缺陷一律进
`docs/TROUBLESHOOTING.md` 并配回归用例；缺口是"还没被盯住的地方"，处置方式不同。
