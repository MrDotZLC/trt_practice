# mini_trt_llm 项目进度交接文档

> 最后更新：2026-09-27（`PROGRESS.md` + `DEC-INT8-WEIGHT-SOURCE` 结案）。
>
> **当前基线（唯一现状口径；以下几段历史快照与各节里的旧数字都不得当作现状）**：
> - **测试（2026-09-27 整轮复跑，最新）**：沙箱 **264 / 0 失败**；真机整轮 **264 / 1 红 / 0 跳过 / 310 s**。
>   其后新增 1 条 host 回归用例（`EngineCacheTest.SourceFileIdentityIgnoresPathSpelling`，见 `TS-048`）
>   → **沙箱现为 265 / 0 失败（已实测）**；**真机总数随之为 265，待下一次真机整轮确认**（未跑过的不写"通过"）。
>   唯一红 = `RealGpt2Fp16GreedyMatchesReferenceTokens`（GPT-2 FP16 NaN，按设计，见 §5.11）。
>   `int8_crosscheck` 这次是 **Passed**——它只在**缺报告**时按设计跳过（77）；报告由 §8.3 的 C-1/C-2 产出。
>   **真机必须带 `MINI_TRT_REQUIRE_GPU=1`**，否则等于白跑。
>   同轮：INT8 交叉校验复跑通过（`overall 98/256`、**余量子集 12/12**、`max_abs 21.5985`），
>   报告里的 provenance 已换成 `路径 + ID` 形式、C++ 与 Python 两侧一致。
> - **阶段**：Phase 0 / 1 / 1.5 / 2 / 3 / 4 全部完成；Phase 5 永久取消（§4.6）。**没有下一阶段**。
> - **精度**：GPT-2 推荐 **FP32**（FP16 端到端 NaN，按政策不修，§5.11）；ResNet18 的 FP32/FP16
>   健康；INT8 走 Q/DQ 显式量化，判据 = "FP32 余量子集一致率"（实测 12/12），默认 per_tensor。
> - **最近一轮交付**：per-channel 整网退化的根因 = 产图脚本的权重 scale 取自**未折 BN** 的权重
>   （`--weight-range-source onnx` 后 54.5% → 100%）；**默认行为与正式产物逐字节未变**。见 §3.0j。
>
> **接手必读（入口，不在此复述）**：
> 1. 已知限制与坑 → §5；开放项 → §6.6（事实与触发条件的唯一来源是 `future_iterations.md` §11）。
> 2. 跑真机 → `future_iterations_development_plan.md` + `OI-RUNBOOK`；命令与前置检查都在那里。
> 3. 开工纪律（点名才开工 / 破坏性动作先列清单）→ `AGENTS.md` 与 §2.16。
> 4. 本地产物与再生命令 → §6.5；环境版本 → §7；测试口径 → §5.10 / §2.13。

<details><summary>展开：2026-09-27 及更早的头部快照原文（保留备查；现状以上面「当前基线」为准）</summary>

> 最后更新：2026-09-27（**`future_iterations.md` §1.5 / P4-INT8-a 结案**，结论见 **§3.0j**、
> 排查过程见 `docs/TROUBLESHOOTING.md` **#46 / #47**）。
> **[历史快照，勿当现状] 当天记的基线**：沙箱 264 条 / 0 失败；真机整轮全量 264 条 / 1 红 / 1 跳过
> （红 = 按设计的 GPT-2 FP16 NaN、跳过 = `int8_crosscheck` 缺报告；真机整轮 **449 s**）。
> 两边的**总数相同**，差别只在 GPU 用例是跑还是跳过。下面这一段头部里的 242 / 243 / 235 / 259
> 都是**当天的历史快照**（每段都标了日期），要现状请以本行为准，明细见"接手必读"第 4 条与 §3.0j。
>
> **本轮（2026-09-27）交付：`future_iterations.md` §1.5（P4-INT8-a）结案**，一句话 ——
> **per-channel 的整网退化是产图脚本的错，不是 TRT 的错**：`quantize_resnet18.py` 的权重 scale
> 取自**未折 BN** 的 torchvision 权重，而 Q/DQ 插在**已折 BN** 的 ONNX 权重上（逐通道折叠系数
> 0.05~19.9）→ 16.19% 的 int8 权重被 clamp 饱和；改 `--weight-range-source onnx` 后 per-channel
> 的余量子集一致率 **54.5% → 100%**。
> **三条互相咬合的真机证据**：① 引擎**忠实**（vs 自己的图逐层最大 `max_abs` **0.2714**）；
> ② 探针**探对了对象**（`d_pre` 8.34e-07 vs `d_post` 0.0398）；③ 现象**复现**（探针图下
> PT **12/12 = 100%** / PC **6/12 = 50%**）。另有**文件级**证据（饱和 int8 权重：PT 3.919% /
> PC **16.188%** / 改源 **0.044%**——与后端无关，坏值已烘进 ONNX）。
> 交付物：2 个新工具（`tools/validate/qdq_reference.py`、`tools/convert/add_probe_outputs.py`，
> 各带 `--self-test` 并进 ctest）、1 个新用例文件（`tests/test_resnet18_int8_probe.cpp`，3 条）、
> `--weight-range-source` 开关 + `fake_quant_check` 的忠实性修复。
> **默认行为与正式产物逐字节未变**；**没有待办**，两件"暂不做"的触发条件见开发计划 **§13.11**。

> **以下各段是 2026-09-27 当天更早几次收口的记录（按时间倒序，保留备查）**——
> 最新状态以本节开头的"当前测试基线"与本轮交付为准。
>
> 交付：profile target ×4（`profile_{gpt2,resnet18}[_ncu]`）、`tools/profile/run_profile.sh`
> （一键 nsys/ncu + 温度时钟 + 日志落文件只回摘要）、`tools/profile/summarize_nsys.py`
> （三层分桶 + `--api` 分配统计 + 自检）、`tests/test_decode_perf.cpp`（两个 P 层用例）；
> 计划 = 开发计划 **§11** / 测试计划 **§10**。
> **它要回答的三个问题都有答案**：sampler 占比（PF-8：greedy **1.25%** / top-k **17.7%** /
> top-p **20.1%**，与 `SamplerPerf` 同 session 互校一致）、attention 占比（PF-9：长上下文
> ≈ **每步 80%** → `§2.2` 触发成立、**P3 → P2**）、显存分配（`§2.1` 触发**未获支持**）。
> **两条边界**：① 本机（WSL2）**拿不到 GPU kernel 时间线**（nsys 只有 API / OSRT / NVTX、
> ncu 报 `Unknown Error on device 0`、加 sudo 与显式 `--trace=cuda` 都无效、"拷到 Windows"
> 也救不了，见 **#41**）→ 逐 kernel 分解记为**能力边界**（工具与自检都在，换机器即可用），
> 本机改用"同 session 比值 + 上下文扫描"；② **PF-7**（ONNX vs 原生 prefill 的跨构建对照）
> **已移交 `future_iterations.md` §10.2**，不在 §11 账上。
> 同轮补齐 §3.0f 漏改的 **3 处"存在性门"**（`test_gpt2_generate.cpp`，见 **#40**），
> 并把 `future_iterations.md` 的 **§5.1 / §6.3 / §9.2** "优先级"行与 §0.1 对齐。
> 沙箱 **242 条 / 0 失败**；**真机复跑应为 243 条**（P9_2-5b 之后全量一直没复跑）。
> —— **这一段已被紧接下来的 §2.2 交付说明取代**（全量已复跑：259 条 / 1 红 / 1 跳过）。
>
> **更新（同日稍后）：§2.2「长上下文 attention」已交付并真机验证**，见 **§3.0i** ——
> 斜率降幅 **88.85%（PP-1 kernel 级）/ 88.20%（PP-2 端到端，两次复现 88.12 / 88.20）**、
> 三档 prompt 的生成 token 与单趟**完全一致**、`max_abs=3.58e-07 / max_rel=1.52e-06`。
> 图版本 bump 到 **2**、`kPagedAttentionPluginVersion` **1 → 2**（旧引擎全部失效一次，**属预期**）。
> **沙箱 259 条 / 0 失败；真机 259 条 / 1 红 / 1 跳过**（唯一红 = 按设计的 GPT-2 FP16 NaN；
> 跳过 = `int8_crosscheck` 缺报告）。F1 的 B 半句（短上下文退化）**经作者决定改为观测项、
> 不设判据**——理由与账目见 `TROUBLESHOOTING.md` **#45 / #45.1**。
>
> 同日：**`future_iterations.md` §9.2 采样器迭代收口并关闭**（Top-P 改"保留 CUB 排序 + 行内并行"；P9_2-5b 判"无显著
> 差异"、P9_2-5c 不做）。**真机最近一次全量 235 条 / 1 红**（唯一红 = 按设计的 GPT-2 FP16 NaN；
> 该全量在 P9_2-5b 之前、也在本轮新增 8 项之前）→ ~~全量待复跑~~ **已复跑**：
> 259 条 / 1 红 / 1 跳过（见上）。
>
> **开工纪律（2026-09-27 收紧，见 `AGENTS.md` §0.1 / §0.7）**：改 `AGENTS.md` 需要口令
> **"授权修改AGENTS.md一次"**；**其他任何工作**（写文档 / 改代码 / 跑实验 / 真机任务）都要作者
> **点名到具体条目**才开工，笼统的"开始吧"不算批准。**为什么**：作者要控制节奏与验收范围——
> 抢跑产生的改动一旦落盘，评审与回滚成本由他承担。
> 2026-09-26：**Phase 4 收口**（CV 路径打通 + INT8 落地）。
> 当前阶段：**Phase 4 已完成**（ResNet18：ONNX 路径 / 原生路径 / `CVRunner` / 转换工具 / FP16 / INT8）；
> **没有下一阶段**——**Phase 5（清理旧模块）已永久取消**，旧模块由作者自行处理（见 §6.6）。
>
> **接手必读五件事**：
> 1. **GPT-2 的推荐精度是 FP32** —— FP16 端到端数值不稳定（NaN，层数随构建变化），
>    按政策不修，见 §5.11 与 `docs/TROUBLESHOOTING.md` §18.1；
> 2. Phase 2 的残余缺口（含已定位的已知限制）见 §5.12 与 `docs/phase2_test_plan.md` §5；
> 3. Phase 3 的缺口只剩 **G5**（ONNX 子图识别只做计数）——**G6 的"可复现测量方法"已于
>    2026-09-27 交付并随 §11 关闭**（见 §3.0h / 开发计划 §11.5.2）；`docs/future_iterations.md` §11。
> 4. **测试基线（2026-09-27 整轮复跑）**：**沙箱 264 条 / 0 失败；真机整轮全量 264 条 / 1 红 / 0 跳过**
>    （整轮 **310 s**；两边**总数相同**，差别只在 GPU 用例跑还是跳过）。
>    唯一红 = `RealGpt2Fp16GreedyMatchesReferenceTokens`（GPT-2 FP16 NaN，按设计，§5.11）。
>    `int8_crosscheck` 在报告齐备时**执行并通过**，只在缺报告时按设计跳过（77）。
>    上一轮（449 s）有 1 条跳过，就是因为当时还没跑 §8.3 的 C-1/C-2。
>    **历史演进（保留备查，勿当现状）**：204 → 215 → 235 → 242 → 259 → **264**；
>    笔记里那句"复跑应为 243"的**外推不准**——ctest 总数在沙箱与真机是**同一个数**。
>    过程红两条都已结案且**都不是产品缺陷**：`Gpt2OnnxTest.MatchesAcrossProfileShapes`
>    （两实现差异之下的并列 → 判据钉行号，§5.13 / #34）与 S-14 首版（参考在并列行上不良定义 → 换
>    `TopKSetByValue`，`TROUBLESHOOTING.md` #36）。
>    B（文本端到端）与 C（INT8 口径交叉校验）两批真机用例**都通过**（见 §3.0e / §3.0f）。
>    **引擎缓存指纹已上线**（`<engine>.fingerprint`）：升级后**第一次真机会重建全部引擎**（旧引擎无指纹），
>    第二次起打 `Engine cache hit` —— **真机已确认**：第一遍 stale ×2 → 重建（623/709 MB），
>    第二遍 hit ×2 且不再重建（见 §3.0f、`TROUBLESHOOTING.md` #34.10）。
>    跑真机时请带 `MINI_TRT_REQUIRE_GPU=1`：否则 GPU 用例会静默跳过，等于白跑。
> 5. **Phase 4 的精度现状**：ResNet18 的 **FP32 / FP16 都健康**（argmax 全一致）；
>    **INT8 走 Q/DQ 显式量化**，判据是"**FP32 有余量子集的一致率**"（实测 12/12 = 100%），
>    权重默认 **per_tensor**。**per-channel 的整网退化已于 2026-09-27 结案**（§3.0j）：
>    根因在**产图脚本**——权重 scale 取自**未折 BN** 的 torchvision 权重，而 Q/DQ 插在**已折 BN**
>    的 ONNX 权重上（逐通道折叠系数 0.05~19.9）。改 `--weight-range-source onnx` 后 per-channel
>    的余量子集一致率 **54.5% → 100%**。**正式产物与默认行为均未改动**（逐字节复核）。
>    真机 B1 四条**全绿**；完整证据链见 §3.0j 与 `TROUBLESHOOTING.md` **#46 / #47**。
>    **产物身份（勿误用）**：`resnet18_qdq.onnx` = **正式产物**；
>    `resnet18_qdq_per_channel.onnx`（及其探针图）= **#46 的复现样本，不是候选基线**
>    （B1-4 的"PC 更差"要的就是它）。两件"暂不做"（切默认源 / 重生成 per-channel）见开发计划 **§13.11**。
>    细节见 §3.0d / §3.0j 与 `docs/phase4_int8_plan.md`。
>    **另一条开放项**：P4-INT8-b → `docs/future_iterations.md` **§1.6**（INT8 绝对误差判据 +
>    带真值标签的验收集；**前置依赖 = 联网下载，须先获批**）。
>    §6.6 与 `future_iterations.md` §11 只保留索引，目标 / 做法 / 验收判据在 §1.5 / §1.6。

</details>

---

## 1. 项目目标

构建一个面向 NVIDIA Turing / `sm_75` 的极简多模态 TensorRT 推理框架，以 `mini_trt_llm` 统一替换现有 `0_resnet18_onnx` 与 `1_gpt2_onnx` 两个独立 ONNX 加载模块。

---

## 2. 当前架构与关键决策

### 2.1 定位：多模态推理框架，而非单一 LLM 引擎

- **为什么**：旧项目同时包含 CV（ResNet18）和 LLM（GPT-2）两个示例。若只替换 LLM，ResNet18 会 orphaned；若各做各的，代码重复。
- **结果**：`mini_trt_llm` 统一承载 CV + LLM，未来预留 Encoder-Decoder / ViT / 多模态接口。

### 2.2 模型构建：配置驱动 + 模型注册表

- **决策**：用 JSON `config.json` 描述模型结构，通过 `ModelRegistry` 分发到具体 `IModelBuilder`（`GPT2ModelBuilder`、`ResNet18ModelBuilder` 等）。
- **为什么选 A 不选 B**：
  - **A（配置驱动 + 注册表）**：每新增模型只需写一个 builder 子类并注册，不修改 `EngineBuilder` 类。
  - **B（硬编码 `BuildGPT2` / `BuildResNet18`）**：扩展性差，每加模型都要改核心 builder。

### 2.3 模型来源：双模式支持

- **方案 A（原生构建）**：从 Safetensors + JSON config 用 TRT Network API 手搭网络。
- **方案 B（ONNX + Plugin）**：解析 ONNX，按子图模式替换为自定义 Plugin。
- **为什么同时支持**：方案 A 是完全替换 ONNX 加载的最终目标；方案 B 是过渡路径，便于复用现有 ONNX 模型并逐步替换算子。
- **命名冲突规避**：所有 Plugin 层名加 `mini_trt_llm_` 前缀，Creator 名用 `MiniTrtLlmXxxPluginCreator`，避免与 TRT 原生层重名。

### 2.4 权重格式：Safetensors

- **为什么选 Safetensors 不选 PyTorch `.bin`**：
  - Safetensors 加载安全（无 pickle 反序列化风险）。
  - 支持 mmap，Header 与张量元数据分离，便于 C++ 解析。
  - 已成 HuggingFace 标准导出格式。
- **解析库**：`syoyo/safetensors-cpp`（轻量，不拉整个官方库）。

### 2.5 Tokenizer：SentencePiece 源码嵌入 + 多模态抽象

- **为什么选 SentencePiece 先接入**：首批模型 GPT-2 虽用 BPE，但 SentencePiece 是更通用的子词分词器，LLaMA/T5 等后续模型直接可用。
- **为什么源码嵌入**：避免依赖系统包版本不一致，构建自包含。
- **抽象接口**：`BaseTokenizer` 预留，未来可扩展 BPE / Tiktoken / CLIP tokenizer。

### 2.6 Plugin API：IPluginV3（TRT 10.x）

- **为什么选 IPluginV3 不选 IPluginV2DynamicExt**：
  - 当前环境 TensorRT 为 **10.15.1**，V3 是推荐接口。
  - V2 虽兼容性好，但在 TRT 10.x 下属于 legacy，长期维护成本高。

### 2.7 日志：独立业务日志宏

- **决策**：新增 `MINI_TRT_LOG_INFO/WARN/ERROR`，不直接复用 `common/logger.hpp`（TRT `ILogger` 接口）。
- **为什么**：TRT logger 绑定 TensorRT 回调接口，打业务日志语义别扭；独立宏可灵活桥接或独立输出。

### 2.8 测试框架：GoogleTest 源码嵌入

- **为什么选源码嵌入不选系统包**：系统未安装 `libgtest-dev`，且网络受限无法 `FetchContent`；源码嵌入最稳定。

### 2.9 构建选项

- `BUILD_TESTS=OFF`（默认），`BUILD_EXAMPLES=OFF`（默认）。
- **为什么默认 OFF**：日常编译更快；CI 显式 `-DBUILD_TESTS=ON`。

### 2.10 INT8 与性能优化延后

- **INT8 校准**：ResNet18 / LLM 的 INT8 支持放到后续迭代。
- **内存池**：Phase 0 仅 `cudaMalloc/cudaFree` RAII 封装，池化实现后续迭代。
- **为什么延后**：Phase 0 优先搭骨架和可编译性，过早做 INT8/池化会拖慢基础验证。

### 2.11 代码注释规范

- **决策**：启用 `cpp-comment-style` skill，要求注释写"Why 而非 What"，必须覆盖魔数、workaround、非直观逻辑和公共 API；禁止复述代码、逐行翻译和遗留调试注释。
- **为什么**：项目代码会长期维护并交给多轮会话接力，清晰的"Why"注释比代码本身更能降低接手成本。

### 2.12 [DEC-IFACE-PHASE1] Phase 1 算子层的接口约定（实现时确立，勿回改）

以下五条是实现 Phase 1 Plugin 时定下的约定，其中前三条与 `phase1_development_plan.md` 的早期表述**不一致**，
是经用户确认后的有意偏离，后续会话不要按 plan 原文"修正"回去：

<details><summary>展开：2.12 [DEC-IFACE-PHASE1] Phase 1 算子层的接口约定（实现时 全文</summary>


- **形状优先、属性兜底**：`configurePlugin` 一律以输入形状为权威来源推导 `num_heads` / `head_size` 等；
  只有对应维度是动态轴（`<= 0`）时才回退到属性值。因此针对属性的校验用例必须构造动态轴。
- **RoPE 采用 half-split 约定**（`out[j] = x[j]cos - x[j+h]sin`），而非 plan §4.2 字面写的"相邻两维配对"。
  **为什么**：plan 同节指定的测试参考是 HuggingFace `apply_rotary_pos_emb`，而它是 half-split；
  已用 `scripts/ref_rope.py` 交叉验证，最大差异 `0.000e+00`。GPT-NeoX 风格留作后续属性开关。
- **RoPE 的 head 配置不做序列化属性**：只序列化 `rotary_dim` 与 `base`，`num_heads` / `num_kv_heads` /
  `head_size` 全部从形状推导。**为什么**：做成属性会多一份可能与形状失配的状态。
- **PagedAttention 不含 `max_context_len` 输入**：plan §4.3 列了该标量输入，但 online softmax 按
  per-batch `context_lens[b]` 循环，全局上界用不到，保留会成为死参数。
- **采样器只有设备侧 API**：`sampler/sampler_common.hpp` 的 `Launch*Sampler` 是唯一接口，token 结果直接写回
  device。**为什么**：Phase 0 遗留的「标量 k/p + host `std::vector` 输出」与 Q7（per-batch tensor）及
  「Decode 全程驻留显存」冲突，已删除（见 §5.9）。实现文件落在 `src/sampler/sampler_kernels.cu`,
  与 plan §3 写的 `src/plugins/sampler/*.cu` 不同——采样器不是 TensorRT Plugin，放在 `src/sampler/` 更贴合实际分层。

另有一条 kernel 实现纪律（来自 `docs/TROUBLESHOOTING.md` #4）：
**输出与输入分离的 kernel，只要存在"部分写入"路径，就必须显式处理未覆盖区间**。

</details>

### 2.13 [DEC-TEST-CONVENTIONS] 测试与验证约定（Phase 1 / 1.5 沉淀，后续沿用）

- **参考实现必须唯一、必须自带断言、必须有 host 侧 meta-test**。
  **为什么**：参考实现是裁决对错的标尺，标尺错了会给出错误裁决——Phase 1.5 就吃过一次

<details><summary>展开：2.13 [DEC-TEST-CONVENTIONS] 测试与验证约定（Phase 1  全文</summary>

  （`ReferenceRoPE` 漏了 batch 维度，把实现正确的 kernel 判成错的，见 #9）。
  "参考与被测必须独立"针对的是参考 vs 实现；同一算子的两份参考彼此只会漂移，必须合并。
  参考实现是纯 host 代码，进 CI 的成本远低于一次真机往返。
- **带 batch 维的算子必须覆盖 `batch > 1`**。
  **为什么**：Phase 1.5 的两个缺陷（C++ 参考、Python 交叉验证脚本）在 `batch=1` 时**都表现为通过**。
- **验证分层**：host 侧用例（含参考实现 meta-test）进沙箱 / CI；GPU 用例在用户 WSL2 真机跑，
  并在本文档 §3 留痕。**沙箱内 host 全绿不代表真机没问题**——Phase 1 的 RoPE 部分旋转缺陷、
  Phase 1.5 的反序列化缺陷都只在真机暴露。
- **每个 Phase 结束必须走一遍真机验证**，不留给下个阶段。
- **失败时的判别方法**：先看"错误从哪个维度边界开始"——这通常直接指向是哪个维度的处理写错了；
  再用"某个配置下能过"反推可以排除哪些代码路径。两者都比逐行读代码快。
- **`pipeline` 组件要区分构建期与运行期**：凡是"从形状推导"的状态，运行期入口
  （`onShapeChange`）都必须能自行推导，不能依赖只在构建期发生的初始化（见 #8）。
- **GPU 用例的"跳过"必须显式、且不许把真故障伪装成跳过**（2026-09-25 补）：
  - 判定统一走 `test_support::ProbeCudaDevice()`（主判定 `cudaGetDeviceCount`），
    跳过信息里必须打印探测结果；`MINI_TRT_REQUIRE_GPU=1` 时**跳过即失败**；
  - "环境不具备"（无设备）才允许跳过；"有设备但 `createInferBuilder` 失败"是**故障**，必须红；
  - 为什么：`test_gpt2_network_build.cpp` 曾把两者合成一个 `GTEST_SKIP`，让两条本该红的断言
    一路滑到提交（TROUBLESHOOTING #19）。但**不能**因此把无 GPU 沙箱的跳过整体改成失败——
    那会让 CI 永远不绿（§5.7）。
  - 实现细节：跳过必须用 **`MINI_TRT_SKIP_IF_NO_CUDA` 宏**。`GTEST_SKIP()`/`GTEST_FAIL()` 都是
    `return` 语句，封装成函数只会退出那个函数、测试体继续执行（实测：63 条用例从 Skipped
    变成 Failed）。
- **凡是做精度比较的用例，必须在构建配置里显式写出目标精度并打印它**（2026-09-25 补）。
  **为什么**：`EngineBuilder::Config::precision` 默认是 **FP16**，而弱类型网络的 I/O 会被 TRT
  声明成 FP32——"看 I/O 精度"看不出内部是 FP16 计算。实测代价：ResNet18 对拍第一版写成
  `Config{}`，拿 FP16 引擎比 FP32 基线，量到 `max_abs = 0.033`（真值 9.5e-6），
  差点被当成"TRT 实现差异大"而放宽阈值。见 `TROUBLESHOOTING.md` #21。
- **借别的模型的产物当负例夹具，等于把两边的契约耦合起来**（2026-09-25 补）。
  **为什么**：`Gpt2OnnxErrorTest.RejectsGraphWithForeignIoNames` 借 `resnet18.onnx` 当"外来
  I/O 名"样本，而 P4-2 把 `cnn` 契约定义成 `input`/`output` 之后，这份夹具就悄悄从"外来名"
  变成了"完全合规"。借的时候必须在注释里点明依赖，并**保证真机全量跑得到它**
  （沙箱里它会被跳过）。见 `TROUBLESHOOTING.md` #22.1。
- **改完接口/契约必须跑真机全量**（2026-09-25 补，由 #22 再次验证）。
  这次一次抓到两条：旧夹具失效 + 一条**单跑通过、全量 SEGFAULT** 的越界切片。
  "单跑过了"不等于对——越界读属于看运气的缺陷，只能靠完整套件与不同堆布局暴露。
- **护栏必须有用例证明它会拦人**（2026-09-26 补，Phase 4 P4-4 落地）。
  转换脚本 / 校验器里的每条"拒绝逻辑"都要有一份**故意改坏**的输入把它打出来
  （例：`resnet18_convert_selftest` 把真模型改坏 5 次，逐个确认自检生效）。
  **为什么**：没有这条证据，"护栏"与"注释里的祈使句"没有区别——项目已经吃过
  "声明了却没人核对"的亏（`phase3_test_plan.md` §5 的 G1c：子图名核对一栏当年没有用例）。
- **新增源文件后必须重新 configure**（2026-09-26 补，**第二次踩**）。
  `mini_trt_llm/CMakeLists.txt` 用 `file(GLOB ...)` 收源文件，GLOB 只在 configure 时求值；
  不重新跑 cmake 的话新文件根本不参与构建，症状是链接期 `undefined reference to vtable for ...`。
  排查捷径：先看 `cmake --build` 输出里**有没有这个文件的编译行**（见 #24.2）。
- **TRT 的 element-wise 要求两侧 rank 相同**（不是 NumPy 广播）（2026-09-26 补）。
  典型写法是把偏置常量声明成带前导 1 的 `[1, N]` 而不是 `[N]`；
  写错时 `IModelBuilder::Build()` 会**返回 true**，错误拖到引擎构建期才报
  （`Assertion x.nbDims == y.nbDims failed`）——见 #24.1，也见 `test_gpt2_network_build.cpp`
  关于"shape 推导延迟报告"的既有警告。
- **测试里的路径 helper 必须返回它名字声称的东西**（2026-09-26 补，由 #25 换来）。
  返回文件却叫 `FindXxxDir()`，会让调用点拼出的路径永远不存在 → 用例**静默跳过**，
  而"跳过"看起来像"跑过了"。**静默跳过比失败更贵**。
  查覆盖时要看 `[ OK ]` / `[ SKIPPED ]` 行本身，别只看 `N passed`——
  总数比预期少一条就是信号。
- **文档只记用例名，不记 ctest 序号**（2026-09-26 补，由 #34 换来）：序号随用例增加而漂移
  （`RealGpt2Fp16...` 从 `#50` 变成 `#61`），引用序号会让文档在下一次加用例后立刻变成假话。
  要指代一条用例，用 `--gtest_filter=<Suite>.<Case>` 或 `ctest -R <name>`。
- **跨实现比较时，"精确判据"必须带"可判性"前提**（2026-09-26 补，由 #34 换来）。
  两条**独立实现**的数值只做到"有界接近"，而 argmax 判等是**精确**判据：在"并列间距小于
  两侧差异"的行上，结果只由误差的**符号**决定，会随构建态翻绿翻红（Phase 3 记过绿、今天红，
  **两次都成立且都不指向缺陷**）。规定：精确判据只作用在**可判行**上（`m > 2d`，`d` = 该行
  两侧逐元素最大差，`2d` 是**可证**门槛），不可判行**必须在真机实测后钉到行号**
  （`ExpectedUndecidableRows()`）——**新增行即红**，改登记表必须给实测依据。判据唯一实现与推导见
  `tests/gpt2_test_support.hpp` 的 `CompareArgmaxByDecidability`，事故全记录见
  `docs/TROUBLESHOOTING.md` #34。

</details>

### 2.15 [DEC-IFACE-PHASE23] Phase 2 / 3 确立的接口约定（实现时确立，勿回改）

与 §2.12 同性质：这些是实现时定下、且**与直觉写法相反**的约定。后续会话不要按"更自然"的写法改回去。


<details><summary>展开：2.15 [DEC-IFACE-PHASE23] Phase 2 / 3 确立的接口约定 全文</summary>

| 约定 | 为什么（写错会怎样） |
|---|---|
| `IModelBuilder::Build` 带 `BuildOptions{stage, weight_dtype}` | builder 需要显式知道目标精度与"建哪个切面"；`BuildStage::kSingle` = 单引擎挂预填/解码两组 profile（Phase 1.5 的 E3 语义），`kPrefill` / `kDecode` 各只挂一组 |
| decode 网络的 cache 输入是**每层一对**（`key_cache_<layer>` / `value_cache_<layer>`） | `PagedAttentionPlugin` 是"单层注意力"，只接收一个 4-D cache。共用一张张量 → **每层都去读第 0 段 cache**，层数越多错得越离谱（真机 2 层差 `1.2e-3`，12 层完全失真）。见 TROUBLESHOOTING #15 |
| `PagedKVCache`：`AppendDecodeKV`（只写、**不推进长度**）+ `AppendDecodeStep`（一次写全部层、**只推进一次**长度） | 长度是"每 token 一个"的量，挂在"每层一次"的接口上会被推进 `n_layer` 倍 → 第 3 个新 token 起发散。见 TROUBLESHOOTING #16 |
| `BuildFromOnnx(model_dir, onnx_path, engine_path, subgraph_names)` | profile 规则要按 `config.json` 的 `architecture` 选，与方案 A **同一套**（否则"两条路对齐"失去意义）；`subgraph_names` 是"要核对的子图名"，**写错即失败**；图必须有 `input_ids` 输入与 `logits` 输出 |
| ONNX 图的 I/O 与原生**不同**（`input_ids` 是 INT64、且无 `position_ids`） | 调用方必须按**引擎声明的**契约准备输入；统一两者见 `docs/future_iterations.md` §10.1 |
| `LLMRunner::Generate` 约定 | 返回**新生成**的 token（不含 prompt）；失败返回**空 vector**（成功至少 1 个 token）；`temperature != 1.0` 显式失败；循环内零 H2D/D2H（`position_ids` 由 kernel 从设备端 `context_lens` 填） |
| `LLMRunner` 按引擎**声明的**边界精度分配缓冲（`prefill_kv_half_` / `*_logits_half_`），**不按 `Config::is_half`** | 弱类型网络下 K/V 与 logits 的输出类型**由 TRT 决定**（FP16 引擎实测为 FP32；`addCast` 钉不住，已实测否证）。按 config 假定宽度 → TRT 按 4 字节写进 2 字节缓冲 → **越界写 → 非法访存**，而 FP32 下永远不暴露（2 与 4 恰好一致）。见 TROUBLESHOOTING #18 |
| `PagedKVCache::Config::source_is_half`（源 = 引擎导出的 K/V）与 `is_half`（目标 = cache 布局）是**两个独立**字段 | `WriteKVKernel` 因此是双模板 `<SrcT, DstT>` 并按源×目标四种组合显式分发；假定二者一致，则 FP32 源写进 FP16 cache 时宽度与数值全错。其中 `is_half` 仍跟引擎 **cache 输入**的声明精度走（不随源精度变），两件事别"顺便统一" |
| `LLMRunner` 启动时校验 decode `key_cache_0` 的声明精度 == `Config::is_half`，不一致**拒绝构造**（`ok() == false`） | cache 宽度是引擎与 `PagedKVCache` 之间的契约，错了 PagedAttention 会按错误宽度读。这道闸是把"内存越界"变成"一条可读启动错误"的机制，**无论将来走强类型还是消费方适配都要保留** |
| 诊断输出（`mlp_fc_0` 等中途张量）必须由 `BuildOptions::export_diagnostics` **显式打开，默认关** | 建图侧 `markOutput` = 改 I/O 契约：每个消费方都要多分配并绑定，TRT 对未绑定输出**直接拒绝 enqueue**。默认带上它，曾经在真机上打挂 6 条用例（见 TROUBLESHOOTING #19）。要加新诊断输出，先 `grep` 全部绑定方与输出计数断言 |
| 引擎缓存路径不得在"两种 I/O 契约"之间共用（如诊断开 / 诊断关） | 两种契约是**两种不同的图**。现在指纹能区分它们（`export_diagnostics` 进指纹），但共用一条路径会让**每次切换都触发重建**（分钟级）；诊断仪器因此单独用 `..._diag.engine` |
| ONNX 的 I/O 契约**按 `architecture` 选**（`OnnxIoContractFor`：`cnn` → `input`/`output`；其余 → `input_ids`/`logits`），且该映射必须能被 host 用例直接测到 | 契约映射内联在 `BuildFromOnnx` 里时，它只在"解析 ONNX + `createInferBuilder`"之后才执行——那两步都要 CUDA，等于护栏只在真机才验证得到。抽成函数后沙箱即可覆盖（`OnnxIoContractTest.*`） |
| CV 的 profile `opt_batch = 8`（历史工程口径），**不要改回 1** | TRT 针对 kOPT 形状挑最快 kernel；沿用 1 会让 batch ≥ 2 的推理走非最优 kernel、性能结论失真 |
| `CVRunner` 的输入契约 = **float32、NCHW、`[0,255]` 像素质**，归一化由 Runner 自己做；失败一律返回**空 vector / 零值统计**并打日志 | 该形态与 P4-1 的契约输入一致（可直接对拍）；HWC→CHW 留给调用方——形参名 `image_nchw` 就是这么定的。失败约定与 `LLMRunner::Generate` 保持一致（异常只用于构造期与底层库） |
| `CVRunner` 的维度与 batch 范围**必须向引擎查询**：输入用 `getProfileShape`、输出用 `getTensorShape` | 写死 224/1000/16 等于把"模型是什么"焊进 Runner；而两个查询 API 分工不同——`getProfileShape` **只对输入有效**（对输出返回 `Dims{-1,{}}`），用错的表现很像"契约不合法"（#23.2） |
| `CVRunner` 的前处理入参是 **`pixels_per_channel`（=H*W）显式传入**，不从总长度反推 | NCHW 的通道下标是 `(i/(H*W))%C`；用"总元素数/C"当分母在 **batch=1 时恰好等价**、batch>1 才错（#23.1）。凡是按 NCHW 拆下标的代码都要覆盖 `batch>1` |
| **建图侧 `markOutput` = 改 I/O 契约**：新增诊断输出必须由 `BuildOptions::export_diagnostics` 显式打开（默认关），且给独立引擎路径 | 默认带上诊断输出曾在真机打挂 6 条用例（#19）；引擎缓存只按路径名区分、不随开关失效 |
| **凡是做精度比较的用例，必须显式写出目标精度并打印**（`Config::precision` 默认 FP16，弱类型引擎的 I/O 却常被 TRT 定成 FP32） | 第一版 ResNet18 对拍把 FP16 引擎当 FP32 用，量到 0.033 差点被当成"TRT 差异大"（#21） |
| **INT8 走 Q/DQ 显式量化**：`SetupBuilder` 里 **不设任何 INT8 flag**，精度由图中的 Q/DQ 决定；Q/DQ 必须**对称**（zero_point 恒为 0），否则 TRT 解析期直接拒 | `kINT8` 自 TRT 10.12 废弃（由 strong typing / Q/DQ 取代）；非对称图报 `Non-zero zero point is not supported`（#27）。另外 Q/DQ **不受我们传的 precision 影响**——同一张 QDQ 图在 FP32/INT8 配置下都跑 INT8（层信息才是证据） |
| **`BpeTokenizer::Load` 的入参是"目录"**，不是单个文件 | byte-level BPE 要 `vocab.json` + `merges.txt` **两个**文件，而 `BaseTokenizer::Load` 只有一个路径参数；基类原话是"具体路径含义由子类决定"，所以把入参解释成目录，避免为它改已交付的基类接口 |
| **预切分里的 ` ?` 只吃字面空格 U+0020**（tab / 换行 / 不换行空格都不算） | 写成"任何空白都能当前导空格"会把 `"a\tb"` 切成 `["a", "\tb"]`（HF 是 `["a", "\t", "b"]`）、把 `"line1\nline2"` 切成 `["line1", "\nline2"]`。同类错误还踩过一次：把它写进"非空白类"分支 → `" quick"` 被拆成 `" "` + `"quick"`（见 #33.2 / #33.3） |
| **引擎缓存必须带构建指纹**（`<engine>.fingerprint`）：`BuildFromConfig` / `BuildFromOnnx` 在入口比指纹，一致才复用，**缺指纹一律视为不可信并重建** | 缓存过去只按路径名复用、不随代码/配置失效，只能靠人记得删 `/tmp`（`TROUBLESHOOTING.md` #34 的牵连因素）。指纹覆盖 stage / 精度 / 源文件身份（size+mtime）/ 全部建图数值参数 / 建图开关 / TRT·CUDA 版本 / **手工图版本 `kEngineGraphVersion`**；**图代码变了必须 bump 它**（自动察觉只能靠编译时间戳，代价不成比例）。测试里各文件自己判断 `exists()` 的门全部取消——复用与否只能由 builder 决定（见 #34.10） |
| **INT8 的判据是"FP32 有余量子集的一致率"**（阈值 ≥90%），整体一致率只作"没崩坏"下界；**能用 `IEngineInspector` 自证在跑 INT8**（需 `Config::detailed_profiling = true`，且判 `Format/Datatype: Int8`，**不是** `[I8]` 标签） | 这批图 FP32 自身摇摆（55% 样本 margin<2），整体一致率主要在测测试集噪声；不设 `kDETAILED` 则读不出逐层精度，会误判成"没跑 INT8"（#29.4 / #30.5 / `phase4_int8_plan` §4） |

</details>

### 2.14 [DEC-EVIDENCE-DISCIPLINE] 证据纪律与操作纪律（Phase 2 沉淀，后续沿用）

本节记录两条**流程级**教训，源自 Phase 2 的两次实际事故：
阈值放宽（技术事故，详见 `docs/TROUBLESHOOTING.md` #15）与擅自改写 Phase 0 文件

<details><summary>展开：2.14 [DEC-EVIDENCE-DISCIPLINE] 证据纪律与操作纪律（Pha 全文</summary>

（操作事故，记录即本节 B 条）。它们不是技术缺陷，但代价比技术缺陷更高：
一次是让用户承担了本可以避免的决策负担，一次差点让一个真 bug 以"全绿"的形态留下来。

#### A. 禁止用"放宽期望值"换取通过

当实测与期望不一致时，**允许的动作只有三种**：


<details><summary>展开：A. 禁止用"放宽期望值"换取通过 全文</summary>

1. 继续查，不给结论（状态写成"原因未知"）；
2. 证明**期望值本身**错——必须给出独立依据（参考实现、实测敏感性数据、设计文档出处），
   改的同时把推导过程留在代码注释与文档里；
3. 把用例标成"已知失败 + 原因未知"，**保持红色**。

**不允许**：调大阈值、删断言、把断言降级成打印、skip 掉用例。

配套要求（可检查）：

- **每个数值阈值旁边必须写清出处**：来自哪次实测、哪个标准、哪份设计文档。
  阈值可以紧、可以松，但不能"来路不明"。
- **阈值不跨精度复用**：D4 的 `rel < 1e-3` 是 FP16 标准，套到 FP32 上等于把尺子放宽 1000 倍。
- **放宽阈值前必须先量"无关差异"**：把与正确性无关的差异（算法不同、累加顺序不同、
  kernel 不同）量出来，阈值放在它的合理倍数上。观测值若比"无关差异"高出几个数量级，
  说明有别的东西在起作用——**此时唯一的动作是查**。
  Phase 2 的实测：一次性 softmax 与 online softmax 在 float32 下差 `6e-8`，
  模型对扰动的放大倍数 ≈ 1；而当时观测到 `1.25e-3`，高 4 个数量级 → 确有真 bug。
- **要求"它凭什么通过"**：只看"绿了没有"会漏掉整类问题；每个 Phase 验收时，
  对关键判据要能回答"这个阈值凭什么这么定"。

</details>

#### B. 批准目标 ≠ 批准手段

- **涉及删除/覆盖现有文件、改配置文件、动 git 历史、联网**的操作，**每一个具体动作都要单独确认**，
  即使计划文档里已经写过"要重写 X"。计划批准的是目标，不是这批破坏性动作。
- **动手前列一份"破坏性动作清单"**一次性交用户确认**，不要每步问一次（那会拖慢节奏），
  也不要因为"清单太琐碎"而省略（Phase 2 就是省掉了这一步）。
- **不要自己当"这文件没人用"的裁判**。判断依据只从代码里找（引用、依赖）不够——
  作者脑子里可能还有别的用法。这类判断属于用户。
- 违反的代价不是"文件被删"，而是**把本可以一句提问解决的事，变成用户事后的回滚决策**。

#### C. 诊断代码也必须自证

- **诊断输出必须说明比较对象是什么**（比了哪两个东西、各自的布局/形状是什么）。
  Phase 2 出现过诊断本身比错对象、输出 `13.8` / `175` 这种"看起来像真故障"的数字——

<details><summary>展开：C. 诊断代码也必须自证 全文</summary>

  **比没有诊断更危险**，因为它会把人引向错误的方向。
- 读回/对拍时先确认**读取范围落在同一段分配内**（那次 `cudaMemcpy` 越界报 invalid argument
  就是这么来的）。
- 关键诊断要同时给**绝对差与相对差**：相对差在小值上会放大，单看相对差会误判严重程度。
- **仪器覆盖不够时，别把"观测缺口"当成现象**：Phase 2 排查 FP16 NaN 时，
  三次运行的"首个 NaN 层"分别是 1/0/2，一度被读成"边界随机漂移"；
  真实原因是中间切点只给第 0 层导出了——**看不到的地方，现象会假装在移动**。
  加仪器之前先问："我现在能看见哪几层／哪几个量？"
- **每轮只改一个变量**：被否证的改动不是白做（LN 精度那次排除了一整个方向），
  但同时改多个变量会让读数无法归因。

</details>

</details>

### 2.16 协作与权限规则（2026-09-27 收紧）

- **改 `AGENTS.md`**：必须由作者**原话**给出许可口令 **"授权修改AGENTS.md一次"**（`AGENTS.md` §0.1）。
  含糊的表达（"编辑一条规则""顺手改一下"）**不构成**许可——Agent 只出草稿，等口令。
- **其他任何工作**（**写计划文档 / 改代码 / 跑实验 / 真机任务**）：必须由作者**点名到具体条目**
  （条目号 / 文件 / 任务号）才开工（`AGENTS.md` §0.7）；笼统的"开始吧 / 继续"不算批准；
  **批准范围不外扩**（批准写计划 ≠ 批准改代码；批准改代码 ≠ 批准 bump 图版本 / 删缓存 / 跑真机）。
- **为什么**：作者要控制节奏与验收范围。抢跑产生的改动一旦落盘（尤其产品代码与文档结论），
  评审与回滚成本由作者承担。

### 2.17 §2.2 确立的接口约定（同 §2.12 / §2.15 的性质：**勿按"更自然"的写法改回去**）

| 约定 | 为什么（写错会怎样） |
|---|---|
| **片数由设备端按各自 `context_lens[b]` 推导**，stage-1 的 grid z 恒为上限、超出有效片数的 block 立即返回不读不写 | 宿主侧**拿不到**当前上下文长度（`block_tables` 形状固定、`context_lens` 在设备上）→ 想在宿主侧判"要不要切"只能走被禁止的 D2H 或改 runner / profile 语义。写成"宿主判定"会让 split 路径永远走分支的另一侧 |
| **`getWorkspaceSize()` 按 `DynamicPluginTensorDesc.max` 报上界**，不是 `desc.dims` | 动态轴在 `desc.dims` 里是 **-1**（TRT 头文件原话）→ 按它算出的 workspace 偏小、kernel 照写 → **越界写**（与 #18 同类，只在真机暴露） |
| **分片切法与 workspace 布局只有一份实现**（`paged_attention_split.hpp`），`getWorkspaceSize` 与 kernel 都调它 | 两边各写一份 → "改了片数 / 改了 head_size"后静默错位，而这类错在真机之前看不出来 |
| **`SetPagedAttentionNumSplitsOverride`：`>0` 强制片数（钳到上限）、`0` 自适应、`<0` 强制旧单趟路径** | `<0` 是**同二进制 A/B 开关**（#37 / #38：跨 session 的差值不可直接比）。把负数钳成 0 会让 A/B 静默失效 |
| **旧单趟 kernel 永久保留**（A/B 入口 + workspace 不可用时的兜底） | 删掉它 → ① A/B 只能靠 git；② workspace 拿不到时没有"正确但慢"的退路 |
| **`kPagedAttentionPluginVersion` 与 `kEngineGraphVersion` 同批 bump** | workspace 需求由 0 变正数，复用旧引擎 = 往 0 字节缓冲里写。指纹**看不见插件源码的变化**（只看模型/配置的 size+mtime）→ 必须手工声明代次 |
| **短上下文多一次归并发射是已知代价**（实测 +1.877%，观测项） | 它来自上面那条"宿主判定不可行"；想消掉得先解决宿主可见性，不是"顺手优化" |

### 2.18 `future_iterations.md` §1.5 确立的仪器 / 产物纪律（2026-09-27；**下个会话按这个来，不要回退**）

| 纪律 | 为什么（违反时会怎样） |
|---|---|
| **量化类转换：scale 必须取自"被量化那张张量"本身** | `quantize_resnet18.py` 曾用**未折 BN** 的 torchvision 权重统计 scale，却把 Q/DQ 插在**已折 BN** 的 ONNX 权重上（折叠系数逐通道 0.05~19.9）→ per-channel 16.19% 的权重被 clamp 饱和、整网余量子集一致率 54.5%（修好后 100%）。**"尺子量 A、裁剪 B"是本项目最贵的一次教训**（`TROUBLESHOOTING.md` #46）。脚本已加**来源自检**：不一致就 `[WARN]` 并写进 meta（只报不拦——默认路径的历史产物要能逐位复现） |
| **标尺必须独立于被测实现，且自己先被校准** | `future_iterations.md` §1.5 的标尺 = **ONNX 官方参考实现**（`tools/validate/qdq_reference.py`），不是"自己折 BN 的 torch 模型"：图里 BN 已折好 → **没有"折叠"这一步可错**。它带 `--self-test`（最小 Q/DQ 图**逐位**比手算语义）并进 ctest。反面教材：#30.3 曾用一个**同样不忠实**的模拟去"否证"模拟不忠实 |
| **探针图必须可证"= 产物图 + 探针"** | `tools/convert/add_probe_outputs.py` **只追加 `graph.output`**，并断言 `node` / `initializer` / `input` / `opset` **逐字节不变**。若改成"重新标定 + 顺手加输出"，探针图与产物图就绑在两次独立标定上，两图不可比 |
| **探针要探"量化前"的 float 张量，不探量化后** | 量化台阶（conv1 是 0.0796）会把 FP32 kernel 的正常差异（1e-3）在桶边界放大成 **±1 格**——噪声与待查信号同量级（#30.5）。落地自证用 `d_pre ≤ d_post`（真机：8.34e-07 vs 0.0398） |
| **产物身份必须钉死：正式产物 vs 复现样本** | `resnet18_qdq.onnx` = **正式产物**；`resnet18_qdq_per_channel.onnx` 及其探针图 = **#46 的复现样本、不是候选基线**（生成"错源"产物时脚本会直接打印这句）。B1-4 的"PC 更差"**要的就是它**——重生成它会让 B1-4 变红，那不是故障 |
| **能离线验的别上真机** | `future_iterations.md` §1.5 整条（含修复的反证）在本机 CPU 上 4 分钟跑完；同一条排查上一轮花了 3 次真机往返（#30.5）。**先找"不依赖后端行为的证据"**（本例：直接读 ONNX 里的 int8 权重常量，数被 clamp 到 ±127 的比例） |
| **形状只能问 `IExecutionContext`，不能问 `ICudaEngine`** | 引擎上动态维是 **-1**，转 `size_t` 就是天文数字 → 报错会伪装成"显存分配失败"（#47.1，遍历 I/O 张量时必踩） |

---

## 3. 已完成的部分

### 3.0a [DEC-PHASE2-DELIVERY] Phase 2 交付（GPT-2 原生构建，2026-09-25）


<details><summary>展开：3.0a [DEC-PHASE2-DELIVERY] Phase 2 交付（GPT-2  全文</summary>

| 文件 / 模块 | 说明 |
|---|---|


<details><summary>展开：3.0a [DEC-PHASE2-DELIVERY] Phase 2 交付（GPT-2 原生 全文</summary>

| `core/gpt2_model_builder.{hpp,cpp}` | GPT-2 原生建图；`kSingle` / `kPrefill` / `kDecode` 三种切面共用同一份代码，只有注意力分支不同 |
| `core/llm_runner.{hpp,cpp}` + `core/llm_runner_kernel.{hpp,cu}` | Prefill→Decode→Sampler 自回归循环；循环内零 H2D/D2H（`position_ids` 由设备端 `context_lens` 填） |
| `kv_cache/paged_kv_cache.{hpp,cpp}` + `paged_kv_cache_kernels.{hpp,cu}` | 分页 cache：块池、序列预留、prefill 覆盖写、decode 追加（`AppendDecodeStep`） |
| `plugins/paged_attention_plugin.*` | 扩展为 5 / 7 输入两形态（第 6/7 个输入是当前 token 的 K/V） |
| `tools/convert/hf_to_mini_trt_llm.py` | 产出 mini_trt_llm 原生 `config.json`（`weight_map` / `skipped_tensors` / 布局声明 / `block_size`） |
| `models/gpt2/` | 转换产物（`model.safetensors` 被 .gitignore 忽略，`config.json` 入库） |
| 用例 | `test_gpt2_config` / `test_gpt2_network_build` / `test_gpt2_decode_consistency` / `test_gpt2_generate` / `test_gpt2_prefill_accuracy` / `test_paged_kv_cache` / `tests/gpt2_test_support.hpp` |
| `tools/inspect_engine.cpp` | 引擎 I/O 探针：反序列化任意 `.engine` 并打印各 I/O 的名字 / 方向 / **声明精度** / 维数（不建 context、不推理）。**不进构建流程**，编译命令写在文件头；用途与限制见 §6.5 |
| `tests/test_gpt2_generate.cpp` 的 FP16 补测 | 复现器 `RealGpt2Fp16GreedyMatchesReferenceTokens`（**真机预期失败**，见 §6.5）+ 纯打印诊断仪器 `Fp16PrefillOutputsDiagnostic`（逐输出给 `max\|v\|` 与 NaN 标记） |

**真机验证结果**（Phase 2 收工时的快照；沙箱内 132 用例 0 失败、GPU 用例自动跳过，
**当前总数见 §3.5**。注意：`625939c` 之后真机已有 6 条实测失败（见 §5.11 / TROUBLESHOOTING #19），
下面这些数字是**该缺陷引入之前**的快照）：

- 真实 GPT-2 贪心 8 token 与 HF 基线**逐 token 一致**：
  `[274, 389, 257, 1049, 835, 284, 651, 257]`（prompt = `"The quick brown fox"`）；
- prefill logits 对拍 `ref_output.bin`：`max_abs = 9.92e-05`、`max_abs/max|ref| = 9.19e-07`、
  `cosine = 1.0`、逐位置 argmax 一致；
- 解码一致性（decode 一步 == prefill 对应位置）在严格阈值 `1e-5` 下通过。

**过程中修掉的 5 个真缺陷**（详见 `docs/TROUBLESHOOTING.md`）：
#13 粘性 CUDA 错误被误读、#14 KV Cache 写入路径两处、#15 decode 各层共用同一 cache 张量、
#16 追加按层推进语境长度、#18（前半）FP16 缓冲按**假定**精度分配 → 越界写。
同一条 #18 的**后半**是另一码事：FP16 图本身产生 NaN，已登记为已知限制（§5.11），按政策不修。

</details>

</details>

### 3.0b [DEC-PHASE3-DELIVERY] Phase 3 交付（GPT-2 ONNX 路径，2026-09-25）


<details><summary>展开：3.0b [DEC-PHASE3-DELIVERY] Phase 3 交付（GPT-2  全文</summary>

| 文件 / 模块 | 说明 |
|---|---|


<details><summary>展开：3.0b [DEC-PHASE3-DELIVERY] Phase 3 交付（GPT-2 ON 全文</summary>

| `core/builder.{hpp,cpp}` | `BuildFromOnnx(model_dir, onnx_path, engine_path, subgraph_names)`：复用方案 A 的 profile/精度语义、I/O 契约校验、parse 错误逐条打印、只挂 prefill 一组 profile |
| `tools/inspect_onnx.py` | 图结构探针：基线比对 + `absent_ops` 护栏 + 三类子图识别断言（**人工执行，未接入 ctest**） |
| `tests/test_gpt2_onnx.cpp` | 三方对拍（ONNX/原生/HF）、FP16 对照、`seq ∈ {1,64,512}` 覆盖；`RunEngine` 按引擎**声明的** I/O 与精度读取（不假定） |
| `tests/test_gpt2_onnx_error_paths.cpp` | 失败路径：4 条沙箱可跑（子图名/缺 config/缺 architecture/ONNX 不可读）+ 1 条真机（外来 I/O 名，复用 `resnet18.onnx` 当夹具） |
| `requirements.txt` | 补 `onnx>=1.16` |

**真机实测**：ONNX vs 原生相对偏差 `5.66e-07`（阈值 `1e-5`）；ONNX/原生各自对 HF 参考
`7.6e-05 ~ 9.9e-05`（随构建的 tactic 变化，相对量级稳定在 1e-6）；FP16 与
`seq ∈ {1,64,512}` 对照均通过（实测值未采集，阈值维持 D6/D2 冻结口径）。

**口径与残余缺口**：见 `docs/phase3_test_plan.md`（G1c / G4b / G5 / G6）。
**性能结论未定**：两次测量的方向相反（±25%，小于构建间噪声），不能据此判断 ONNX 路径
是否更优，更不能据此决定是否做子图替换——见 `docs/future_iterations.md` §10.2。

</details>

</details>

### 3.0c [DEC-PHASE2-PATCH] Phase 2 补丁（诊断输出开关 + CUDA 环境判定，2026-09-25）


<details><summary>展开：3.0c [DEC-PHASE2-PATCH] Phase 2 补丁（诊断输出开关 +  全文</summary>

| 文件 / 模块 | 说明 |
|---|---|


<details><summary>展开：3.0c [DEC-PHASE2-PATCH] Phase 2 补丁（诊断输出开关 + CU 全文</summary>

| `core/imodel_builder.hpp` + `core/builder.{hpp,cpp}` | `BuildOptions::export_diagnostics` / `EngineBuilder::Config::export_diagnostics`，**默认关** |
| `core/gpt2_model_builder.cpp` | 4 处诊断 `markOutput` 改由开关控制 → 默认构建的输出数回到 `2*n_layer + 1` |
| `tests/test_gpu_guard.hpp` | `ProbeCudaDevice()`（主判定 `cudaGetDeviceCount`）+ `MINI_TRT_SKIP_IF_NO_CUDA` 宏 + `MINI_TRT_REQUIRE_GPU` 闸门 |
| `tests/test_cuda_check.cpp` | `GpuEnvProbe.ReportsCudaAvailability`：**永不跳过**，每次运行都打印环境事实 |
| `tests/test_gpt2_network_build.cpp` | SetUp 三分支："无设备"跳过、"有设备但建不出 builder"**判失败** |
| `tests/*.cpp`（19 个文件、59 处） | GPU 跳过统一走显式探测宏，跳过信息里带 `cudaGetDeviceCount` 原始结果 |

| `tests/test_paged_kv_cache.cpp` | P2S-6：追加用例改为**每层一对**并补两层读回断言；新增负例 `AppendDecodeStepRejectsLayerCountMismatch`（见 `TROUBLESHOOTING.md` #20） |

计划与验收见 `docs/phase2_supplement_plan.md`；缺陷与实测见 `docs/TROUBLESHOOTING.md` #19 / #20。
**真机全量结果**：146 条，0 跳过（`MINI_TRT_REQUIRE_GPU=1`），**1 条红 = FP16 NaN 复现器（按设计红）**。

</details>

</details>

### 3.0d [DEC-PHASE4-DELIVERY] Phase 4 交付（ResNet18 / CV 路径，2026-09-26）

计划与测试计划：`docs/phase4_development_plan.md`、`docs/phase4_test_plan.md`；
INT8 子计划：`docs/phase4_int8_plan.md`；缺陷与排查：#21 ~ #31。

<details><summary>展开：3.0d [DEC-PHASE4-DELIVERY] Phase 4 交付（ResNet 全文</summary>


<details><summary>展开：3.0d [DEC-PHASE4-DELIVERY] Phase 4 交付（ResNet18 全文</summary>


| 文件 / 模块 | 说明 |
|---|---|
| `core/resnet18_model_builder.{hpp,cpp}` | **原生建图**（零 Plugin：BN 已折叠；conv/relu/add/maxpool/GAP/flatten/gemm）；I/O 名复用 `OnnxIoContractFor("cnn")`；`stage != kSingle` 显式失败；注册进 `EngineBuilder` |
| `core/cv_runner.{hpp,cpp}` | `CVRunner`：输入契约 **NCHW float `[0,255]`**、前处理归 Runner、维度与 batch 范围**向引擎查询**、失败返回空 vector/零值统计 |
| `core/builder.{hpp,cpp}` 的改动 | `OnnxIoContractFor(architecture)`（`cnn → input/output`，可 host 测）；CV `opt_batch` 默认 1→8；`Config::detailed_profiling` |
| `tools/convert/onnx_to_mini_trt_llm.py` | ONNX → `models/resnet18/{config.json, model.safetensors}`（42 张量；逻辑名从 **Conv/Gemm 节点名**推导；含 5 道自检 + `--self-test`，已接入 ctest） |
| `tools/convert/quantize_resnet18.py` | ONNX → **对称 int8 Q/DQ** 图（PTQ：torch hook 收直方图 → 99.9 分位裁剪 → 每个 Conv 插输入/权重/输出三处 Q/DQ）；自检（60 对、zero_point 全 0、checker）+ fake-quant 预估 |
| `scripts/ref_resnet18.py` | torchvision FP32 外部基线（ramp / pixels 两套输入；自证：两次运行逐位一致） |
| 用例 | `test_resnet18_baseline` / `test_resnet18_weights` / `test_resnet18_onnx` / `test_resnet18_native` / `test_resnet18_fp16` / `test_resnet18_int8` / `test_cv_runner`（对应测试计划里的 R0.x / R1.x / R2.x / R3.x） |
| 共享测试件 | `tests/cv_test_support.hpp`（CV 侧公式/路径/读写的唯一来源）、`tests/diff_stats.hpp`（差异口径的唯一来源，GPT-2 侧改为包含它） |

**真机实测（关键数字）**：

| 对拍 | 结果 |
|---|---|
| ONNX 路径 vs torchvision 基线 | ramp `max_abs = 9.54e-6`、pixels `1.34e-5`，argmax 全一致 |
| **原生 vs ONNX**（同一份权重） | `max_abs = 1.07e-6`（阈值 `1e-5`） |
| 原生 vs torchvision 基线 | `1.05e-5`（阈值 `1e-4`） |
| **FP16** | ONNX-FP16 vs FP32 基线 `0.031`、原生-FP16 vs ONNX-FP16 `0.0076`、CVRunner+FP16 `0.062`、原生-FP16 vs 基线 `0.033`；**argmax 全一致、无 NaN**（阈值两档 `0.1` / `0.05`，出处 `TROUBLESHOOTING` #26） |
| **INT8** | Q/DQ 引擎 **43 层 / 38 层含 Int8 张量 / 4 层 `i8i8` tactic**（对照 FP32 引擎 0 层 Int8）；**FP32 余量子集一致率 12/12 = 100%**、整体 38.3%（判据出处 `phase4_int8_plan.md` §4）。产物形态为 **`prequant_dq`**（预量化 int8 权重 + 只留 DQ，ONNX 44.7 MB → **13.3 MB**，实测与 `Q→DQ` 数值等价，见 `TROUBLESHOOTING.md` #31.3） |
| `CVRunner` 端到端 | batch 1/8 对基线 `1.34e-5`；超范围 batch / 尺寸不符 / `ok()==false` 均显式失败；benchmark `mean≈8.4 ms`、`≈950 img/s`（**仅记录，非判据**） |
| **真机全量** | **182 条 / 1 红 / 0 跳过**（Phase 4 当时的快照；唯一红 = GPT-2 的 FP16 已知限制。之后新增了批次 A 的 host 用例与脚本项、以及 B / C 的真机用例 → 真机总量应为 204，**待复验**，见 §3.0e / §3.5） |

**过程中的真缺陷/真问题（全部留痕）**：#21（把引擎建成 FP16 却比 FP32）、#22（旧夹具失效 + 越界切片）、
#23（前处理通道下标在 batch>1 时算错、`getProfileShape` 用错 API）、#24（element-wise 的 rank 匹配、GLOB 需重跑 configure）、
#25（路径 helper 返回文件 → 用例静默跳过）、#26（FP16 阈值不能用 FP32 的尺子）、
#27 ~ #31（INT8：非对称 Q/DQ 被拒、per-channel 整网退化及其 11 条被否证的假设）。

**开放项**：见 §6.6（P4-INT8-a / P4-INT8-b / P4-FP16-a 等）。

</details>

</details>

### 3.0e [DEC-BATCH-A-DELIVERY] future_iterations 批次 A 交付（2026-09-26）

按 `docs/future_iterations_development_plan.md` 的分批，**批次 A（可立即开工）**两项已落地；
计划 / 用例 / 判据出处见该文件与 `docs/future_iterations_test_plan.md`。

<details><summary>展开：3.0e [DEC-BATCH-A-DELIVERY] future_iteration 全文</summary>


<details><summary>展开：3.0e [DEC-BATCH-A-DELIVERY] future_iterations  全文</summary>


| 文件 / 模块 | 说明 |
|---|---|
| `tools/validate/README.md` + `int8_eval.py` | **A2（`future_iterations.md` §1.6 的离线子项）**：INT8 判据的验收集规格（meta 必需字段 / 重叠排除规则 / 率必带 n）与评估脚本（分层报告、余量子集 p50·p95·p99、带真值标签时另报 top-1 正确率）。含 `--self-test`（3 项分层数学 + 7 道护栏），已注册 ctest 项 `int8_eval_selftest`。**不下载任何数据** |
| `tokenizer/bpe_tokenizer.{hpp,cpp}` | **A1（§5.1）**：GPT-2 byte-level BPE。`Load` 入参 = **目录**（`vocab.json` + `merges.txt`）；`Encode` 先按 GPT-2 正则语义做预切分、再按 merge rank 合并；`Decode` 走 byte 回退并在非法 UTF-8 处替换 U+FFFD；额外提供 `PreTokenizeForTesting` / `ByteEncodedPiecesForTesting` 供排错 |
| `utils/json.hpp`（改动） | 补 `\uXXXX`（含 UTF-16 代理对）——GPT-2 的 `vocab.json` 全是这种转义，原先直接抛 `unknown escape sequence`；覆盖见 `tests/test_json.cpp`（6 条） |
| `tools/make_tokenizer_golden.py` + `tests/data/gpt2_tokenizer_golden.json` | 参考数据的生成与校验。golden 存每个样本的 `text` / **`pieces`（HF 预切分结果）** / `ids` / `decoded` 与来源文件 SHA256；`--check` 已注册 ctest 项 `tokenizer_golden_check`（重新用 HF 算一遍再比对）。**脚本 `local_files_only=True`，不联网** |
| 用例 | `tests/test_bpe_tokenizer.cpp`（8 条 host）、`BpeTokenizerReferenceTest.GoldenIsSelfConsistent`（参考自证）、`BpeTokenizerWithRunnerTest.TextPromptMatchesReferenceTokens`（放在 `test_gpt2_generate.cpp`，把"文本 → prompt token"与既有真机基线常量接起来） |

**实测（2026-09-26）**：`Encode` 与 HF **逐 token 全等**（21 个样本：basic 3 / whitespace 5 /
utf8 10 / long 1 / edge 2；长文本 306 token）。utf8 一组刻意压在"Unicode 分类是近似"这个已知限制上：
中文 / emoji / 中英混排 / 重音拉丁 / 日文 / 西里尔 / 全角字母数字 / CJK 标点 / ZWJ 家庭 emoji / 货币符号
——**全部一次通过**（分类区间表按实测有效，但这仍是近似，新增语种要按同一办法加样本验证）。
`Decode` 与 HF 解码一致；`Load` 的 4 条负例（目录不存在 /
缺 merges / 坏 JSON / merges 行无空格）全部按预期拒绝。沙箱全量 `ctest` **204 条 / 0 失败**。

**同批还有两条用例（当时标注"待真机"；**均已在 2026-09-26 真机执行通过**，结果见本节末尾）**：

- **B：文本端到端** —— `Gpt2GenerateTest.RealGpt2TextPromptEndToEnd`，把
  `文本 → Encode → LLMRunner → Decode → 文本` 接成一条链（判据三段：分词 / 生成 / 解码文本）。
- **C：INT8 口径交叉校验（A2-5）** —— C++ 侧 `DumpsLogitsAndCppReportForCrossCheck` 落 logits + 报告，
  Python 侧 `int8_eval.py --legacy-mode` 出报告，`crosscheck_reports.py` 比对两侧的 n 与分子
  （不一致 = 口径漂移，**不许改阈值**）。

**执行清单（命令、顺序、期望值、失败语义）写在 `docs/future_iterations_development_plan.md` §8，
由作者在真机执行**——Agent 侧无 GPU。

**真机执行结果（2026-09-26，作者执行）**：

- **B 文本端到端：PASSED**——`文本 → Encode → LLMRunner → Decode → 文本` 四段判据全过，
  即"文本进 / 文本出"这条链路在真机成立（此前只有 token 入口的端到端）。
- **C INT8 口径交叉校验：PASSED**（C-1 落 artefact → C-2 Python 出报告 → C-3 比对）——
  C++ 侧现役统计与 Python 侧 `int8_eval.py` 对同一批 logits 给出**同一组 n 与分子**
  （口径一致；注意 C 走 `--legacy-mode`：当前测试图与标定集同源，报告已标注"一致率会被高估"）。
- **A 真机全量（当时快照）：204 条 / 2 红**——除按设计的 FP16 红外，出现 1 条**新红**；
  该条已按方案 B 结案，**全量重跑后为 215 条 / 1 红**（见 §3.0f / §5.13）。

**过程中的真缺陷**：`TROUBLESHOOTING.md` **#33**（公共 JSON 解析器不支持 `\u`；
预切分的"可选前导空格"两次写错——一次是放错分支、一次是把它当成"任何空白"）。
两次都是**先看诊断表里的 pieces 对照**再改代码，而不是逐行读实现。

**未验证件（2026-09-26 现场核实，登记为开放项 SP-1）**：`SentencePieceTokenizer` 是
`BaseTokenizer` 的实现之一，但**没有任何用例引用它、仓库里没有 `.model`/`.spm` 资产、
也没有生产调用方**（`LLMRunner` 只存 `BaseTokenizer` 指针，测试传 `nullptr`），
等于它的 `Load` / `Encode` / `Decode` 从未被任何参考裁决过。
核实命令：`rg -n "SentencePieceTokenizer" mini_trt_llm/`（除自身实现外无引用）、
`find . -name "*.model" -o -name "*.spm"`（除 third_party 外为空）。
**处置**：要真正用 SentencePiece 系模型时，**先造参考与用例再动实现**（照 BPE 的做法：
golden + 来源 SHA256 + meta 自证 + 负例）；资产需联网取或由作者提供（联网须先获批）。
索引见 `docs/future_iterations.md` §11 的 **SP-1**。

> **原"A2-5 尚未申请"已不成立**：C 批（C-1 / C-2 / C-3）已在真机跑通并把 artefact 落盘，口径交叉校验完成；
> 收尾事项见 §3.0f 末尾。

</details>

</details>

### 3.0f [DEC-ENGINE-FINGERPRINT] 判据修正 + 引擎缓存指纹（2026-09-26，承接真机新红 #34）

**背景**：真机全量出现一条新红 `Gpt2OnnxTest.MatchesAcrossProfileShapes`（seq=512 逐行 argmax）。
机制已被**定量**为"两实现差异之下的并列"——数据与三个可核对事实见 `TROUBLESHOOTING.md` **#34**

<details><summary>展开：3.0f [DEC-ENGINE-FINGERPRINT] 判据修正 + 引擎缓存指纹（ 全文</summary>


<details><summary>展开：3.0f [DEC-ENGINE-FINGERPRINT] 判据修正 + 引擎缓存指纹（20 全文</summary>

（`TROUBLESHOOTING.md` §34.6 定量、`TROUBLESHOOTING.md` §34.8 为何 Phase 3 曾绿、`TROUBLESHOOTING.md` §34.9 判据、`TROUBLESHOOTING.md` §34.10 指纹）。

| 文件 / 模块 | 说明 |
|---|---|
| `tests/gpt2_test_support.hpp` | `CompareArgmaxByDecidability`：判据的**唯一实现**——可判行（`m > 2d`）必须 argmax 全等；不可判行允许不同。附 `2d` 可证门槛的推导 |
| `tests/test_gpt2_onnx.cpp` | 换用该判据；每个形状打印"可判 / 不可判（含行号与 9 位有效数字取值）"；`ExpectedUndecidableRows()` 把**不可判行钉到行号**（实测 `(1,512) = {118}`，其余形状为空） |
| `tests/test_argmax_criterion.cpp`（新，6 条 host） | 锁语义与边界（含严格 `>` 的边界、混合多行的计数与行号），并用实测数字复现 #34 的 `IncidentRow118IsClassifiedUndecidable` |
| `core/engine_cache.{hpp,cpp}`（新） | 构建指纹：stage / 精度 / 来源 / 源文件身份（size+mtime）/ 建图数值参数 / 开关 / TRT·CUDA 版本 / 手工 `kEngineGraphVersion`；落 `<engine>.fingerprint` |
| `core/builder.cpp` | `BuildFromConfig` / `BuildFromOnnx` **入口比指纹**：一致 → `Engine cache hit`；不一致或缺指纹 → `Engine cache stale` 并重建 |
| 8 个测试文件里的 16 处"存在性门" | 全部去掉——复用与否**只由 builder 决定**，否则会绕过指纹检查（老坑的来源）。**2026-09-27 复核补齐**：`test_gpt2_generate.cpp` 里当时还剩 3 处（FP16 复现器 ×2、FP16 诊断仪器 ×1），当日去掉，见 `TROUBLESHOOTING.md` #40 |
| `tests/test_engine_cache.cpp`（新，5 条 host） | 确定性 / 每个字段都会改变指纹 / 缺指纹不可信 / 往返与规范化文本 / 源文件身份随内容变化 |

**真机实测（单测复跑，2026-09-26）**：`Gpt2OnnxTest.MatchesAcrossProfileShapes` **PASSED（4.58 s）**；
不可判行 `(1,1)=0`、`(1,64)=0`、`(1,512)=1`（行 118）、`(2,4)=0`、`(2,64)=0`；
数值项全过（相对 `1.05e-06 ~ 2.09e-06`、`cosine = 1`）。那一行两个引擎各自都只"看到"约 `1.5e-05` 的间距、
**方向相反**，而它们对该行的分歧是 `3.8e-05`/`6.9e-05` —— 即该问题在两边精度下都没有答案。

**收尾确认（2026-09-26 真机，已完成）**：

1. **全量重跑：215 条 / 1 红** —— 唯一红是 `Gpt2GenerateTest.RealGpt2Fp16GreedyMatchesReferenceTokens`
   （GPT-2 FP16 NaN 的按设计红，签名与 §5.11 逐项一致）。**"只剩 1 红"已确认**。
   附带教训：ctest 序号会随用例增加而漂移（这条从 `#50` 变成 `#61`）——**文档只记用例名、不记序号**。
2. **引擎缓存指纹的真机行为已确认**：第一遍 `Engine cache stale` ×2 → `Engine saved`（623 MB / 709 MB，
   旧引擎无 sidecar → 按不可信重建，**慢是预期的**）；第二遍 `Engine cache hit` ×2 且**没有** `Engine saved`
   → 指纹生效。这是真机上唯一能证明该机制的观察点。
3. 仍待触发：`docs/future_iterations.md` §11 的 **SP-1**（SentencePieceTokenizer 未验证）。

</details>

</details>

### 3.0g [DEC-SAMPLER-KERNEL] `future_iterations.md` §9.2 采样器高性能 kernel（P9_2-0 ~ P9_2-5b，2026-09-26 ~ 27，**已关闭**）

**计划落点**：开发计划 `future_iterations_development_plan.md` **§10**（P9_2-5b 见 §10.12）、
测试计划 `future_iterations_test_plan.md` **§9**；归因与测量协议的教训在 `TROUBLESHOOTING.md`

<details><summary>展开：3.0g [DEC-SAMPLER-KERNEL] `future_iterations 全文</summary>


<details><summary>展开：3.0g [DEC-SAMPLER-KERNEL] `future_iterations.m 全文</summary>

**#35 / #37 / #38**（本节只留决策与状态）。

| 落点 | 说明 |
|---|---|
| `src/sampler/sampler_kernels.cu` | ① `TopPParallelSampleKernel<kSubChunked>`：一行一个 256 线程 block，各线程算"自己那段连续元素"的块和（`true` 时再切 16 个子块和）→ thread 0 做"块 → 子块 → 元素"三级交叉定位；② `LaunchTopPSampler` 走 `true` = **生产路径**；③ `LaunchTopPSamplerTwoLevel` 走 `false` = **P9_2-5b 之前的形态，只作 A/B**（1 KB shared / 48 寄存器，对照生产版 17.4 KB / 51 寄存器）；④ `LaunchTopPSamplerLegacy`（旧的逐行串行实现，只作对照）；⑤ `FastTopKSampleKernel` + `LaunchTopKSamplerFast`（**契约 `k ≤ 64`，越界写哨兵 -1；性能不达标、已撤出生产**） |
| `include/.../sampler/nucleus_cutoff.hpp`（新） | 交叉点定位的**唯一实现**：`FindCrossingSegment` / `ScanSegmentForCrossing` / `FindCrossingByLevels`，`__host__ __device__` → kernel 与 host 用例共用（避免参考漂移） |
| `include/.../sampler/sampler_common.hpp` | 只加声明与语义说明；**API 形态与 workspace 需求未变**（`LLMRunner` 无需改动即生效） |
| `tests/sampler_test_support.hpp`（新） | 采样器解析参考的唯一实现（排序顺序 / nucleus / 截断后归一化分布 / **并列安全**的 top-k 集合） |
| `tests/test_sampler.cpp` | perf harness（**中位数 + 配对 + 斜率 + ABBA + 同二进制 A/B**）、S-15/S-16/S-21/S-24、`NucleusCutoffTest.*`（4 条 host）、`SamplerReferenceTest.*`（3 条 host） |
| `tests/test_fp16_paths.cpp` | S-12 / S-13（Top-K / Top-P 的 FP16 分布） |

**最终验收（口径 = 配对 + 斜率净成本，n=15/31，2026-09-27 真机）**

| 形状 | 配对（净）`legacy/parallel` | top-p 净 | top-k 净 | greedy 净 | `top-p − top-k` |
|---|---|---|---|---|---|
| 50257 × 1 | **12.63×** | 0.499 ms | 0.438 | 0.035 | **60.1 µs**（p25 56.4 / p75 64.0） |
| 50257 × 8 | **22.43×** | 0.614 | 0.588 | 0.052 | −0.5 µs |
| 128000 × 1 | **15.73×** | 1.225 | 1.225 | 0.098 | 14.3 µs |
| 128000 × 8 | **17.57×** | 1.471 | 1.377 | 0.155 | 130.7 µs |

- **判据（`future_iterations_development_plan.md` §10.5 第 4 条；冻结后从未调低）：top-p 对同类 legacy ≥10× → 4/4 达标**。
- **Top-P 的语义差异（唯一一处，勿按"更自然"的写法改回去）**：legacy 逐元素累加 `exp/total` 再与 `p` 比，
  新实现累加 `exp` 再与 `p·Σexp` 比（先除后加 vs 先加后除）→ **极端并列处 cutoff 可能差一格**；
  随机数消费、`>=`、稳定项取 top-1、前缀内重新归一化全部不变。因此 Top-P 的判据是**分布级 + 集合级**，
  **不能**要求"逐 token 与 legacy 相同"（登记在测试计划 §9.4）。
- **P9_2-5b（收尾段子块级并行化）：效果无显著差异 → 保留代码、这条优化线关闭**。同二进制 A/B 中位数
  （子块版 − 两级版）= +4.8 / +27.3 / −330.5 / −81.3 µs，**p25/p75 全跨 0** → 效应低于本平台
  **±400~600 µs** 的判别下限。**P9_2-5c（第一趟访存/MLP）不做**：`greedy`（本来就完全合并访存）
  净成本 35~155 µs，而 `top-p 净 − top-k 净` @50257×1 只有 60.1 µs —— 采样内核净开销 ≈ 裸读一遍行的
  1.7 倍，**优化空间见底**（其余 ~89% 是保留的 CUB 排序）。
- **Top-K 快速路径（P9_2-2~4）：正确性通过**（与 legacy **逐 token 相同**），**性能不达标**（慢 6~9×，
  根因见开发计划 §10.10）→ **已从 `LLMRunner` 撤下**（生产走旧 CUB 路径）；达标前**不要接回**。
- **测试覆盖**：本轮新增 6 条 GPU 用例（S-12 Top-K FP16 / S-13 Top-P FP16 / S-14 Top-K 大 vocab /
  S-15 Top-P 大 vocab / S-16 p=1 大 nucleus / S-21 分布级截断）与 10 条 host 用例
  （`NucleusCutoffTest.*` 4 / `SamplerReferenceTest.*` 3 / S-24 3）→ **`future_iterations.md` §11 的
  P1.5-a 据此关闭**（Top-K / Top-P 的 FP16 分支都有真机通过的用例）。
- **过程红 1 条（非产品缺陷）**：S-14 首版——128000 第 64 名有 **4 个 token 精确并列**，参考用不稳定
  的 `partial_sort` 取"前 k 个"会把合法的采样结果排掉。改法 = **并列安全**的 `TopKSetByValue`
  （`value ≥ 第 k 大值`）+ 事故数据固化为 host 回归；**产品代码一行未改**（`TROUBLESHOOTING.md` #36）。
- **仍未采集**：sampler 在**整步 decode** 中的占比（`nsys` 那一腿）→ 见 §6 的下一步建议。
- **测量纪律（本轮最贵的教训，务必沿用）**：① 分段测量对"差百分之几"没有判别力（#37）；
  ② 跨协议 / 跨 session 的差值**不能直接比**（#38）；③ 判"改动有没有用"要**同二进制、同轮交替测**，
  并用**斜率** `(T4−T1)/3` 扣掉每窗口固定开销；④ 这台机器对这类问题的**判别下限约 ±400 µs**。

</details>

</details>

### 3.0h [DEC-PERF-PROFILE] decode 性能画像基建（2026-09-27，**已出首份数据；kernel 时间线待宿主机**）

**计划落点**：`future_iterations.md` §6.3 与 §11 的 **G6**；开发计划 **§11**（P6_3-0 ~ P6_3-7）、
测试计划 **§10**（PF-1 ~ PF-7）。本节只记"落点 + 状态"，口径与判据在开发计划 §11.3 / §11.4。

<details><summary>展开：3.0h [DEC-PERF-PROFILE] decode 性能画像基建（2026-0 全文</summary>


<details><summary>展开：3.0h [DEC-PERF-PROFILE] decode 性能画像基建（2026-09- 全文</summary>


| 落点 | 说明 |
|---|---|
| `tests/CMakeLists.txt` | 四个**显式** target：`profile_gpt2` / `profile_resnet18` / `profile_gpt2_ncu` / `profile_resnet18_ncu`；**不进默认构建、不进 ctest**；报告默认落 `/tmp/mini_trt_llm_profiles/`（**不入库**） |
| `tools/profile/run_profile.sh`（新） | 一键 nsys / ncu：时间戳命名、采样前后记录温度 / 时钟、nsys 跑完自动导出 `cuda_gpu_kern_sum` 与 `cuda_api_sum` 摘要并分桶；工具缺失 / 缺文件给可读错误 |
| `tools/profile/summarize_nsys.py`（新） | **三层分桶**（① 我们的 kernel / ② CUB / ③ TRT 内部）+ sampler / attention / KV 占比；`--api` 模式报 `cudaMalloc` / `cudaFree` 的次数与总耗时（§2.1 的供数）。`--self-test` 已注册 ctest 项 `profile_summary_selftest` |
| `tests/test_decode_perf.cpp`（新） | `Gpt2DecodePerf.StepLatencyByPhase`（P 层：只打印；用斜率 `(T32−T1)/31` 分离 prefill 与 decode；**唯一 assert** 是"同 session 两次测量的中位数漂移 < 判别下限"）+ `PerfStatsTest.*`（5 条 host） |
| `tests/perf_stats.hpp`（新） | 中位数 / **最近秩分位（不插值）** / 极差 / 斜率 的**唯一实现**，被上面那个用例自证 |

**沙箱实测（2026-09-27）**：`ctest --test-dir build` → **242 条 / 0 失败**。
新增 8 项 = 5 条 `PerfStatsTest.*`（实跑通过）+ 2 条 `Gpt2DecodePerf.*`（无 GPU 显式跳过）
+ 1 条 `profile_summary_selftest`（通过）。

**事实边界（写在这一轮开工时；真机结果已见本节后面的"真机首跑 / 第一次成功出数 / 第 5 条"）**：

- **真机一次都没跑过**：Agent 沙箱里 `nsys` 启动即报 `open: Operation not permitted`（无 perf 权限），
  所以 P6_3-0（仪器自证）与 P6_3-4 ~ P6_3-6（出数）**只能由作者在 WSL2 真机执行**（命令见开发计划 §11.9）；
- **后续进展（同日）**：`profile_gpt2` 真机跑通并出数；**sampler 占比**（PF-8）与
  **attention 占比**（PF-9 上下文扫描）都有了；`§2.1`（显存池）的触发条件**未获支持**；
  `§2.2` 触发成立 → 升 P2；只有 **``future_iterations.md` §10.2`（ONNX 子图替换）仍缺可复现性能对照（PF-7 未跑）**；
- **逐 kernel 分解仍未拿到**（WSL2 的 nsys 采不到 GPU 活动、ncu 不可用、加 sudo 与显式 trace 都无效，见 #41），
  但它已不再阻塞 §2.2（改用上下文扫描回答）。
- 引擎缓存：`Gpt2DecodePerf` 与 `RealGpt2Greedy...` **共用** engine 路径与形状参数，
  目的是命中缓存；换形状参数会让指纹失效并触发分钟级重建。

**真机首跑（2026-09-27，作者）：三个发现，两个当场修、一个记为环境限制**（全过程见
`docs/TROUBLESHOOTING.md` **#39**；开发计划 §11.5.1 有同样的记录）：

1. **`profile_gpt2` target 失败 = 被测用例红了，不是 nsys 坏了**。`nsys` 会**透传被 profile
   程序的退出码**；后用 `nsys stats --report cuda_api_sum` 后处理那份已生成的报告（**无需 GPU**）
   看到 30921 次 `cudaLaunchKernel` → 用例其实跑完了全部测量。红在我加的
   `EXPECT_LT(|Δmedian|, 0.6 ms)`：**那是采样器类的判别下限，被我套到量级大一到两个数量级的
   整步 decode 上**（`AGENTS.md` §7 禁止的阈值跨场景复用）。**已改成只打印绝对 + 相对漂移**。
2. **"假 CSV"**：`nsys stats` 自己的 `Generating SQLite...` / `Processing [...]` 走 stdout，与
   CSV 混流；resnet18 那份所谓 `kern_sum.csv` 只有 415 B 的消息。**已改成**"确认有 `Total Time`
   表头才落盘，否则报警并删除"。
3. **本机两条 CLI profiling 路径都拿不到 kernel 时间线**：nsys 报告不含 GPU kernel 数据
   （`cuda_gpu_kern_sum` 连表头都没有），`profile_gpt2_ncu` 也失败
   （`==ERROR== Unknown Error on device 0.`、无 `.ncu-rep`）；**"拷到 Windows 看"没用**
   ——数据压根没被采集（#41 更正了 #39 的说法）。**CUDA API 摘要是好的** → `future_iterations_development_plan.md` §11.4 的
   "三层 kernel 分解"仍未拿到；绕法见 #41。**PF-5（分配开销）不受影响**，已拿到初步数据
   （GPT-2：`cudaMalloc` 63 次 / 3.335 ms、`cudaFree` 69 次 / 249.7 ms；
   **`cudaFree` 含隐式同步，不等于纯分配器成本**）。

另修两个可用性问题：被 profile 进程的输出一律落 `<报告>.app.log`、终端只回 ≤20 行摘要
（不再"满屏看不到原因"）；去掉 `nsys profile --stats=true`。

**第一次成功出数（2026-09-27，重跑；引擎 `cache hit` ×2）**——`Gpt2DecodePerf`（batch=1、
FP32、greedy、prompt 4 token、生成 32、n=15 + warmup 3、每轮 ABBA）：

| 量 | median | p25 / p75 |
|---|---|---|
| T(1)（prefill 4 token + 1 decode 步） | **4.947 ms** | 4.730 / 6.298 |
| T(32) | **93.190 ms** | 90.109 / 103.367 |
| 派生 decode 每步（斜率 `(T32−T1)/31`） | **2.847 ms** | —— |
| 派生 prefill（4 token） | ≈ **2.100 ms** | —— |

读法（细节在开发计划 §11.5.1 / 测试计划 §10.5）：

1. **同 session 漂移 10~18%**（T(1) 0.917 ms / 18.5%，T(32) 9.478 ms / 10.2%），
   同期 GPU 72→78 °C、44.5→65.7 W → **机器未进稳态**；decode 量级的比较**必须**同轮交替，
   且**不能**套用 `future_iterations.md` §9.2 的 ±400~600 µs 下限（#39 的教训）。
2. **batch=1 的 decode 是每步固定开销主导**：prefill 4 个 token 约 2.10 ms，decode 每个 token
  却要 2.85 ms，全程 ~30921 次 `cudaLaunchKernel`（~26 次/步）→ 假设是 launch/固定开销占大头，
  **待 kernel 时间线定论**。**（同日修正：这只在短上下文成立——第 5 条的上下文扫描显示，
  上下文一长 attention 就变成主导项；两条不矛盾，是"谁主导"随上下文长度切换。）**
3. **§2.1（显存池）的触发条件未获支持**：全程 `cudaMalloc` 63 次 / 3.335 ms，
   `cudaFree` 69 次 / 249.7 ms（含隐式同步，是拆除期成本）。
4. **sampler 占比已出数并完成互校（#41 绕法 1，测试计划 PF-8；#42 已定位）**：
   decode 步 **2.458 ms**；greedy **0.0308 ms（1.25%）**、top-k(64) **0.4360 ms（17.7%）**、
   top-p(0.9) **0.4938 ms（20.1%）**。与 `SamplerPerf` **同 session** 互校差 ≤2%
   （0.0336 / 0.4281 / 0.4916）。首版全零输入给出的 0.92% / 4.75% / 6.35% **已作废**
   ——**退化输入让排序路径快了 2.7 倍**，根因见 `TROUBLESHOOTING.md` **#42**。
   **读法**：① greedy 下采样可忽略（~1%）；② top-k/top-p 的 15~20% 里**绝大部分是 CUB 排序**，
   而两次"绕开整段排序"的尝试（fast top-k）都更慢 → **暂无已知的优化抓手**，不必据此排期；
   ③ decode 步本身跨 session 漂 ±27%（2.458 / 2.847 / 3.365）→ 占比只在同一 session 内可比。
   **逐 kernel 分解**（attention vs MLP）仍拿不到；但**attention 的占比已由第 5 条的
   上下文扫描回答**（长上下文 ≈80%）→ §2.2 的触发条件成立；``future_iterations.md` §10.2` 仍挂着。

5. **上下文扫描（PF-9，2026-09-27 真机）：attention 在长上下文下占每步 ≈80% —— §2.2 的触发条件由此成立。**
   既然 profiler 拿不到 kernel 时间线（#41），改用"attention 随上下文增长、其余每步固定"这个
   性质做**上下文长度扫描**（开发计划 §11.4.1）。三档实测（GPT-2 原生引擎、batch=1、FP32、greedy）：

   | prompt | 平均上下文 | 每步 decode |
   |---|---|---|
   | 4 | 20.5 | **3.055 ms** |
   | 256 | 272.5 | **6.062 ms** |
   | 960 | 976.5 | **14.705 ms** |

   两段斜率 **11.93 / 12.28 ms per 1000 位置**（差 3% → **线性**，外推 1024 = +12.2 ms）→
   **长上下文下 attention 约占每步 80%**。**机制已定位**：`LaunchPagedAttention` 的
   `grid=(num_heads, batch)`（`paged_attention_plugin.cu:141`）→ 每层只有 **12 个 block**，
   而本机 **24 个 SM**（一半闲置），每块 64 线程**串行**走完 976 个位置 → **延迟受限**，
   有效带宽仅 ≈ **6 GB/s（约峰值 192 GB/s 的 3%）**。→ **要做的是把上下文维切开并行**
   （FlashDecoding 式 split-K），不是笼统 tile。**§2.2 已据此从 P3 升 P2**
   （`future_iterations.md` §0.1 / §0.2 / §0.3 第 14 项）；实现属产品代码改动，须先产计划再批准。
   
   **一个副产品**：上面那条"sampler 占 17~20%"是**短上下文**（每步 2.46 ms）下的比例；
   上下文一长分母就变大，长上下文下同样的 sampler 只占 ≈3%（0.49 ÷ 14.7）。**占比随上下文变**，
   引用时必须带上上下文长度。

**收口（2026-09-27）：`future_iterations.md` §11 关闭。** 目标（"把 decode 的时间花在哪从不知道变成知道 + 留下可复现的尺子"）
已达成；尺子还自己抓到过一次测量错误（#42）。两条记账方式变更：

- **逐 kernel 分解（P6_3-4）记为"能力边界"，不是"未完成"**——工具写好、自检过、报告能生成，
  缺的只是有 GPU 跟踪能力的机器（#41）。**别在下一轮把它当欠账去补**。
- **PF-7 移交 `future_iterations.md` §10.2**（ONNX 子图替换的前置），不再算 §11 的尾巴。

</details>

</details>

### 3.0i [DEC-FLASHDECODING] §2.2 长上下文 attention 交付（FlashDecoding 式 split-K，2026-09-27）

**计划落点**：`future_iterations.md` **§2.2**；开发计划 **§12**（12.1 计划对账 → 12.10 真机执行清单）；
测试计划 **§11**（H 组 PS-* / G 组 PG-* / P 组 PP-*）。

<details><summary>展开：3.0i [DEC-FLASHDECODING] §2.2 长上下文 attention 全文</summary>


<details><summary>展开：3.0i [DEC-FLASHDECODING] §2.2 长上下文 attention 交 全文</summary>


| 落点 | 说明 |
|---|---|
| `plugins/paged_attention_split.hpp`（新） | **分片规则 + workspace 布局的唯一实现**（`ResolveSplits` / `SplitRange` / `WorkspaceSlotOffset` / `WorkspaceBytes`），全部 `__host__ __device__` → kernel 与 host 用例共用一份 |
| `plugins/paged_attention_plugin.cu` | 新增 `PagedAttentionSplitKernel`（stage-1，按 `(head,batch,split)` 出局部 `m/l/acc`）+ `PagedAttentionMergeKernel`（stage-2，max-trick 归约，**保序、不用原子累加**）；`getWorkspaceSize()` 由 0 改为按 `.max` 报上界；`enqueue` 走 split 路径、workspace 不可用时兜底单趟；单趟/ split 共用同一份入参校验 |
| `plugins/paged_attention_kernel.hpp` | `LaunchPagedAttentionSplit`；`SetPagedAttentionNumSplitsOverride`（`>0` 强制片数 / `0` 自适应 / **`<0` 强制旧单趟路径** = 同二进制 A/B 开关，测试用） |
| `plugins/paged_attention_plugin.hpp` | `kPagedAttentionPluginVersion` **"1" → "2"**（旧引擎反序列化直接失败，见 `future_iterations_development_plan.md` §12.8 第 2 条） |
| `core/builder.cpp` | `kEngineGraphVersion` **1 → 2**（**安全必需**：workspace 由 0 变正数，复用旧引擎会"往 0 字节缓冲里写"） |
| 用例 | H 8 条（`PagedAttentionSplitPlanTest.*`，纯函数 + 分解数学自洽）+ G 7 条（`PagedAttentionSplitKernelTest.*` + `Fp16PathTest.PagedAttentionSplitMatchesFp16Reference`）+ P 2 条（`PagedAttentionSplitPerf.SlopeByContextLength`、`Gpt2DecodePerf.ContextLengthSweepSplitVsSinglePass`） |

**真机实测（2026-09-27）**：**259 条 / 1 红 / 1 跳过**（唯一红 = 按设计的 GPT-2 FP16 NaN，
§5.11；跳过 = `int8_crosscheck` 缺报告）。

| 量 | 结果 |
|---|---|
| **斜率降幅**（F1 的 5A，判据 ≥40%） | **88.85%**（PP-1，kernel 级，同二进制）／**88.20%**（PP-2，端到端同 session 同引擎；前一次 88.12% → **两次复现**） |
| 每步 decode（PP-2） | prompt 4 → **2.775 / 2.724 ms**（split / 单趟）；256 → **3.629 / 5.491**；960 → **3.993 / 13.044**（**3.27×**） |
| 每层每步（PP-1，heads=12/head_size=64） | ctx 32 → 0.025764 / 0.023600 ms（**split 慢 9.2%**）；256 → 0.094844 / 0.174143（1.84×）；1024 → 0.125568 / 0.918948（**7.32×**） |
| 语义不变 | 三档 prompt 的生成 token 与单趟**完全一致**（PP-2 打印）；`Gpt2DecodeConsistency.*` 与 FP32 8-token 冻结基线通过 |
| 新旧差异（诊断） | `max_abs=3.57628e-07`、`max_rel=1.52484e-06` —— 比算子判据（rel<1e-4）低约 65×，**没有任何阈值被放宽** |
| 两把尺子互校 | PP-1 单趟 ctx=1024 0.918948×12 = **11.03 ms/步** vs PP-2 单趟斜率 10.795×0.976 = **10.5 ms/步**（差约 5%） |
| 机制证据 | PP-1 的比值随上下文：0.916× → 1.836× → 7.318×，正是"越长的上下文切开越划算"的交叉曲线，7.3× 贴近 8 片并行度上限 |
| 缓存行为 | 图版本 bump 后真机首次 `Engine cache stale → 重建`（627/475 MB），之后复用 → 指纹机制按设计生效 |

**关键设计（详见开发计划 §12.3 D1~D5，勿按"更自然"的写法改回去）**：

- **并行维度取上下文维**（不是 head/batch 维）：改动前每层只有 `num_heads × batch = 12` 个 block、
  每块串行走完上下文，而本机 24 SM → 一半闲置、有效带宽仅约峰值 3%；
- **两阶段归约、不用原子累加**（原子加无法保序 → 结果不可复现）；
- **片数由设备端按各自 `context_lens[b]` 推导**（**落地修正**：`block_tables` 形状固定、
  `context_lens` 在设备上 → 宿主侧拿不到"当前上下文多长"；要拿到只能走被禁止的 D2H，
  或改 runner/profile 语义。代价是短上下文每层多发一次归并 kernel，**已在动手前预估并接受**）；
- **workspace 按 `DynamicPluginTensorDesc.max` 报上界**（`desc.dims` 在动态轴上是 -1）；
- **分片策略与 workspace 布局只有一份实现**（`paged_attention_split.hpp`），`getWorkspaceSize`
  与 kernel 都调它；`AssertFixtureConsistent` 之类的护栏同理；
- **旧单趟 kernel 保留为永久 A/B 入口 + 兜底路径**（F3=A；同时消掉了"跨 session 比性能"这个坑）。

**验收判据（开发计划 §12.6）**：7 条里 **6 条达成**；第 5 条拆成
**5A（斜率降幅，✅）** 与 **5B（短上下文退化）——经作者决定改为"观测项、不设判据"**：
实测退化 **+1.877%**，机制 = 每层多一次归并发射；原判据"不超过同 session 漂移"因
三重理由作废（"漂移"有 `max`/`min`/配对差三义且前两种结论相反、曾议的"≤2%"唯一数值输入无出处、
PP-1 与 PP-2 对同一笔代价差 1.97× 未解释）→ 见 `TROUBLESHOOTING.md` **#45 / #45.1**。
**这不是放宽阈值换绿，而是判据被证伪后不假装它成立**（`AGENTS.md` §7）。

**本轮过程问题（全部已结案，产品代码零改动）**：**#43**（计划里的 `ctest -R` 写成了 gtest
过滤器语法 → **一条用例都没跑且退出码为 0** 的静默空跑）、**#44**（3 条 SEGFAULT + 1 条断言失败
全是用例夹具/断言写错：块表行宽不足以放下 `ceil(ctx/block_size)` 个块 → host 参考越界；
一条断言把 GQA 比例写反）、**#45 / #45.1**（漂移锚点的"宽松/保守"写反 + 阈值来路不明）。
**开放观察（不是缺陷）**：PP-1 的 kernel 级预测（+0.95%）与 PP-2 的端到端实测（+1.877%）
差约 2×，**差因未查**。

</details>

</details>

### 3.0j [DEC-INT8-WEIGHT-SOURCE] P4-INT8-a 结案：per-channel 整网退化的根因（2026-09-27）

**计划落点**：`future_iterations.md` **§1.5**；开发计划 **§13**（13.1 计划对账 → 13.10 回填）；
测试计划 **§3**（B1-1~B1-4 / B1-H1 / B1-H2）。排查全过程：`TROUBLESHOOTING.md` **#46**。

<details><summary>展开：3.0j [DEC-INT8-WEIGHT-SOURCE] P4-INT8-a 结案：p 全文</summary>


<details><summary>展开：3.0j [DEC-INT8-WEIGHT-SOURCE] P4-INT8-a 结案：per 全文</summary>


**结论（一句话）**：根因**在产图脚本，不在 TRT**——`quantize_resnet18.py` 的**权重范围**取自
**尚未折 BN 的 torchvision 权重**，而 Q/DQ 插在**已经折过 BN 的 ONNX 权重**上。
折叠系数 `γ/√(var+ε)` 逐输出通道跨度实测 **0.05 ~ 19.9**：per-tensor 只错一个全局倍率（后果轻），
per-channel 错的是**逐通道倍率** → 系数 >1 的通道 `round(w/s)` 越过 127 被 **clamp 饱和**。

| 落点 | 说明 |
|---|---|
| `tools/validate/qdq_reference.py`（新） | **数值标尺**：用 ONNX **官方参考实现**执行 Q/DQ 图并落盘指定张量（量化前 + 量化后两份）。为什么不再"自己折 BN"：图里 BN 已折好，直接执行就**不存在折叠这一步**——上一轮 3 次真机往返里有 2 次耗在"探针自身的 BN 折叠错"上（#30.5）。含 `--self-test`（最小 Q/DQ 图**逐位**比手算 ONNX 语义）+ 进 ctest |
| `tools/convert/add_probe_outputs.py`（新） | **采样器**：给既有 Q/DQ 图**只追加图输出**（20 个 Conv 的量化前输出 + GAP 输出）。自证"探针图 = 产物图 + 探针"：node / initializer / input / opset **逐字节不变**；二次追加与非目标图都被拒。含 `--self-test` + 进 ctest |
| `tools/convert/quantize_resnet18.py` | 新增 `--weight-range-source {torchvision,onnx}`（**默认保持 `torchvision` → 正式产物逐字节不变**；`onnx` = 从被量化那张张量上取 scale，即修复）。另修好旧 `fake_quant_check`：它原来拿**未折 BN** 的权重做 fake-quant，与图里被量化的张量不是同一个（这就是 #28"预检与实测差 5 倍"的根因），现改为"与图逐位等价的模型"（权重取自图 + BN 置成精确恒等）。再加**两道常规自检**（只报不拦，默认产物不受影响）：**来源自检**（算 scale 的张量 vs 被量化张量，不一致就 `[WARN]`＋写进 meta；默认源实测 20 层不一致）与**尺度自检**（打印被 clamp 到 ±127 的权重比例，实测 3.919% vs 改源 0.044%） |
| `tests/test_resnet18_int8_probe.cpp`（新） | B1-1/2/3/4 三条 GPU 用例：探针自证（`d_pre ≤ d_post`）、同引擎同输入逐位相同、与参考逐层对拍出曲线、探针图下复现对照（硬门） |
| `tests/CMakeLists.txt` | 注册两条 host ctest 项（`qdq_reference_selftest` / `add_probe_outputs_selftest`） |
| `models/resnet18/`（新产物，不入库） | `resnet18_qdq_per_channel.onnx`（per-channel 臂）+ 两份探针图 `..._probe_{per_tensor,per_channel}.onnx` |

**离线实测（本机 CPU，ONNX 官方参考实现，64 张真实图，`margin≥5` 余量子集 n=11）**：

| 臂 | 整体一致率 | 余量子集 | max_abs(vs FP32) |
|---|---|---|---|
| per-tensor（正式产物） | 60.9% | **100.0%** | 21.736 |
| per-channel（**错源**） | 25.0% | **54.5%** | 22.911 |
| per-channel（**改源**） | 57.8% | **100.0%** | 21.527 |

**四条关键证据**：① 参考实现给出的四个数（60.9 / 25.0 / 54.5 / 100）与 #29.2 / #29.5 记的
**真机 TRT 引擎**数字**逐位相同** → 参考实现与 TRT 一致，出问题的是当时那个"模拟"；
② 只改 scale 的来源（其它一律不动），参考实现下退化**消失**；③ 逐层曲线显示分叉**从第 0 层
（`conv1`）就开始**，在 `layer4.1.conv2`（折叠跨度最大）放大到 **16.4×**；
④ 默认参数重新生成产物 → node / initializer / output **逐字节相同**（**默认行为零改动**）。

**第五条（文件级，与后端无关）**：per-channel 的 int8 权重常量是在 Python 侧算好写进 ONNX 的，
所以"量化错没错"可以**完全不碰 TRT** 地读文件判断——数被 clamp 到 ±127 的权重比例
（对称量化下每通道约 1 个才对）：**PT 3.919% / PC(错源) 16.188% / PC(改源) 0.044%**。
**坏值已经烘进文件**，任何忠实后端都会复现同样的退化。

**真机首跑（2026-09-27，用户执行）**：`Int8Probe*` 3 条**全红，但红在用例自身的绑定**——
`RunProbeEngine` 用 `ICudaEngine::getTensorShape` 取形状，而**引擎上动态维是 -1**，
`size_t` 一转成天文数字 → `显存分配失败：input`。已改用 `IExecutionContext::getTensorShape`
并加"任何维 ≤ 0 即报错"的校验（`TROUBLESHOOTING.md` #47.1），**待复跑**。
同一次运行还量到：**探针图确实改了 tactic**（产物图 44 层/38 Int8/4 个 `i8i8` → 探针图
78 层/74 Int8/**0 个** `i8i8`，#47.2）——这正是 `future_iterations_development_plan.md` §13.3 D6 要防的事，所以 B1-4（复现对照）
从"硬门"升级成"**唯一的裁判**"，并要求它在同一次运行里通过。

**被推翻的旧结论两条**（#46.3）：#30.3 ①"模拟参照忠实"的否证**作废**（那次否证用的模拟
**同样**不忠实）；`future_iterations.md` §1.5 原表"整网模拟：两臂余量子集都 100%"一行**作废**。

**验收状态**：`future_iterations.md` §1.5 的"二选一"走**分支一**（第 0 层、机制明确），且机制被"scale 与被量化张量
一致则三级都不差"的最小复现支持。**沙箱 `ctest` 264 条 / 0 失败**（原 259 + 本轮 5）；
**真机整轮全量 264 条 / 1 红 / 0 跳过 / 310 s**（2026-09-27 复跑；唯一红按设计，见"接手必读"第 4 条）。

**真机 B1 四条：全绿（2026-09-27，首跑因用例绑定 bug 红过一次，见 #47.1）**：

| 用例 | 实测 |
|---|---|
| B1-2 确定性 | 22 个张量两次运行**逐位相同**（704 ms） |
| B1-1 探针自证 | `conv1`：PT `d_pre=8.34e-07` / `d_post=0.0398`，PC `1.43e-06` / `0.0398` → 探到的是**量化前**张量 |
| B1-3 引擎 vs 自己的图 | 逐层最大 `max_abs` PT **0.2714**（`layer4.1.conv2`，相对 2.3%）、PC **0.168**；`conv1` 仅 **8.3e-07**；两臂都未越过 `100×地板` → **两个引擎都忠实执行了各自的图** |
| B1-4 复现对照（硬门） | 256 张：整体 PT 99 / PC 26；**余量子集（n=12）PT 12/12 = 100%、PC 6/12 = 50%** → 与正式产物口径一致，**探针图是现象的有效模型**（尽管它把 `i8i8` 从 4 变成 0，#47.2） |

**所以结论不依赖任何未验证的假设**：引擎忠实（B1-3）+ 探针探对对象（B1-1）+ 现象复现（B1-4）
+ 文件级饱和统计（#47.3）+ 改源后 100%（#46.2）。
**产物身份已钉死（2026-09-27，作者指示）**：无外部依赖、零功能收益的两条"身份"决定都**暂不做**，
只在文档与产图脚本的告警里写明，避免下个会话误用：

- `resnet18_qdq.onnx` = **正式产物**（per_tensor + 默认 `torchvision` 源）；
- `resnet18_qdq_per_channel.onnx`（及其探针图）= **#46 的复现样本，不是候选基线**。

**两件"暂不做"**（作者 2026-09-27 指示：都不是默认行为、都不现在做；展开见开发计划 §13.11）：

| 事项 | 改什么 | 实测成本 / 收益 | 为什么先不做 |
|---|---|---|---|
| ① 把 `--weight-range-source` 默认切到 `onnx` | 产图规则 | **数值**：per-tensor 饱和权重 **3.919% → 0.000%（20 个）**，但 64 张上判据与一致率**完全不变**（60.9% / 100%） | 换了默认 = 换了正式产物 → `phase4_int8_plan` §4 / `PROGRESS` §3.0d / R2.6 / C 批交叉校验的数全要真机重测回填；而"更准"的证据不足（余量子集只有 11 张） |
| ② 重生成 per-channel 产物 | 一份非默认产物 | **零功能收益**（默认路径无人读它） | 它是 B1-4 的承重件；重生成会让 B1-4 变红（现象消失），必须与"退役/改写 B1-4"打包做 |

</details>

</details>

### 3.1 目录与构建

- `mini_trt_llm/CMakeLists.txt`：C++17 + CUDA C++17、`sm_75`、static library、第三方依赖接入。
- `mini_trt_llm/tests/CMakeLists.txt`：GoogleTest 集成。
- 根 `CMakeLists.txt`：加入 `mini_trt_llm`，旧模块已注释掉。

### 3.2 Utils 基础设施

| 文件 | 说明 |
|---|---|
| `include/mini_trt_llm/utils/cuda_check.hpp` | `CUDA_CHECK`、`NVINFER_CHECK` 宏 |
| `include/mini_trt_llm/utils/logger.hpp` | `MINI_TRT_LOG_*` 业务日志宏 |
| `include/mini_trt_llm/utils/timer.hpp` + `src/utils/timer.cpp` | CUDA Event 计时器（已补注释） |
| `include/mini_trt_llm/utils/memory_pool.hpp` + `src/utils/memory_pool.cpp` | `DeviceBuffer` / `PinnedBuffer` RAII 封装（已补注释） |
| `include/mini_trt_llm/utils/io.hpp` + `src/utils/io.cpp` | 文件读写 + JSON 加载 |
| `include/mini_trt_llm/utils/json.hpp` | 自研极简 JSON 解析器（已补注释） |
| `include/mini_trt_llm/utils/safetensors_loader.hpp` + `src/utils/safetensors_loader.cpp` | Safetensors 加载 + BF16→FP32/FP16 转换（已补注释） |

### 3.3 Core 通用化骨架


<details><summary>展开：3.3 Core 通用化骨架 全文</summary>

| 文件 | 说明 |
|---|---|


<details><summary>展开：3.3 Core 通用化骨架 全文</summary>

| `include/mini_trt_llm/core/precision.hpp` + `src/core/precision.cpp` | 精度枚举与 TRT 映射（已补注释） |
| `include/mini_trt_llm/core/builder.hpp` + `src/core/builder.cpp` | 统一 EngineBuilder（已补默认值注释）；入口按**构建指纹**决定复用还是重建 |
| `include/mini_trt_llm/core/engine_cache.hpp` + `src/core/engine_cache.cpp` | 引擎缓存指纹（纯逻辑、可 host 测）：`ComputeEngineFingerprint` / `Read\|WriteEngineFingerprint` / `EngineCacheIsFresh`；**缺指纹一律视为不可信**（§3.0f、`TROUBLESHOOTING.md` #34.10） |
| `include/mini_trt_llm/core/engine.hpp` + `src/core/engine.cpp` | Engine 封装 + Benchmark（已补统计注释） |
| `include/mini_trt_llm/core/imodel_builder.hpp` | 模型构建器抽象接口 |
| `include/mini_trt_llm/core/model_config.hpp` + `src/core/model_config.cpp` | JSON 配置加载 |
| `include/mini_trt_llm/core/model_registry.hpp` + `src/core/model_registry.cpp` | 模型注册表 |
| `include/mini_trt_llm/core/weight_loader.hpp` + `src/core/weight_loader.cpp` | Safetensors 权重加载 + weight_map 映射 |
| `include/mini_trt_llm/core/llm_runner.hpp/.cpp` | Phase 0 仅接口声明 |
| `include/mini_trt_llm/core/cv_runner.hpp/.cpp` | Phase 0 仅接口声明 |

</details>

</details>

### 3.4 其他模块占位

- `kv_cache/`、`plugins/`、`sampler/`、`tokenizer/` 头文件与空实现已就位，供 Phase 1/2/4 填充。

### 3.5 [DEC-TEST-INVENTORY] 测试

- `mini_trt_llm/tests/test_*.cpp`：Utils / Core 骨架（cuda_check、logger、timer、memory_pool、io、
  model_config、model_registry、safetensors_loader、engine）+ Phase 1 算子 + Phase 1.5 端到端。

<details><summary>展开：3.5 [DEC-TEST-INVENTORY] 测试 全文</summary>


<details><summary>展开：3.5 [DEC-TEST-INVENTORY] 测试 全文</summary>

- 当前状态（2026-09-27 实测，含 Phase 4 + 批次 A/B/C + `future_iterations.md` §9.2 采样器迭代 + §3.0h 性能画像基建
  + §3.0i 的 §2.2 split-K + **§3.0j 的 `future_iterations.md` §1.5 仪器**）：
  沙箱内 `ctest` **265 个用例，0 失败**（GPU / P 层用例在沙箱显式跳过）。
  **演进**：242（§3.0h 的 8 项）→ **259**（§3.0i 的 17 项：H 8 + G 7 + P 2）→ **264**
  （§3.0j 的 5 项：2 条 host 自检 + 3 条 GPU 用例）→ **265**
  （`EngineCacheTest.SourceFileIdentityIgnoresPathSpelling`，见 `TS-048`）。
  host 侧含 `onnx_graph_probe`、`GpuEnvProbe`、`int8_eval_selftest`、`tokenizer_golden_check`、
  `int8_crosscheck_selftest`、`ArgmaxCriterion*`（6 条）、`EngineCacheTest*`（5 条）、批次 A 的 16 条、
  P9_2-5/5b 的 `NucleusCutoffTest.*` 与 `SamplerReferenceTest.*`、§3.0h 的 5 条 `PerfStatsTest.*`
  与 `profile_summary_selftest`、**§3.0j 的 `qdq_reference_selftest` / `add_probe_outputs_selftest`**。
  真机（`MINI_TRT_REQUIRE_GPU=1`）**整轮全量：2026-09-27 复跑，264 条 / 1 红 / 0 跳过 / 310 s**
  （该次复跑的数；**真机总数随新增的那 1 条变为 265，待下一次真机整轮确认**——未跑过的不写"通过"。
  **唯一的红 = GPT-2 FP16 NaN 复现器，按设计**，§5.11；`int8_crosscheck` 报告齐备 → Passed）。
  **沙箱为 265 条 / 0 失败**（原 264 + 上面那条 TS-048 用例）——两边总数相同，差别只在 GPU 用例是跑还是跳过。
  §3.0i 新增的 17 项**全部真机通过**（含 split-K 的数值/性能用例）；§3.0j 新增的 3 条 GPU 用例
  也已随整轮通过（此前按 filter 跑过两遍，见 §3.0j）。
  上一轮跳过的只有 `int8_crosscheck`——那是按"先全量、后跑 C"的顺序做、缺报告 → 77；**跳过 ≠ 通过**。
  先跑 C-1/C-2 产出报告后，它执行并通过（2026-09-27 复跑）。
  **历史快照（勿当现状）**：更早的 235 条 / 1 红是 **§3.0h 新增 8 项之前、且 P9_2-5b 之前**的数，
  当时笔记写"复跑应为 243"——**那个外推也不准**（ctest 总数沙箱与真机是同一个数，
  差异只在"GPU 用例跑还是跳过"；见测试计划 §11.1 的更正）。
  更早：2026-09-26 是 215 条 / 1 红，再早是 204 条 / 2 红；2026-09-27 首批新增用例时是
  228 条 / 1 红，第二次（首次跑 S-12/S-14）是 **233 条 / 2 红**。
  **本轮（P9_2-5~7）新增的 6 条 GPU 用例（S-12 / S-13 / S-14 / S-15 / S-16 / S-21）已全部真机通过**；
  性能数字见 §3.0g。第二次真机那条多出来的红 S-14 已定位为"参考在并列行上不良定义"，
  **只改测试参考**（产品代码一行未改，见 `TROUBLESHOOTING.md` #36），**复跑后绿**。
  B（文本端到端）与 C（INT8 交叉校验）两批真机用例**都通过**；
  `int8_crosscheck` 若在"先全量、后跑 C"的顺序下会**跳过（77，设计如此）**；先跑 C-1/C-2 产出报告则执行并通过。跳过 ≠ 通过。
  实测命令：`cmake --build build -j$(nproc) && ctest --test-dir build`（build 目录已配 `BUILD_TESTS=ON`）。
  分层与覆盖度详见 §3.9 / §3.10 与 `docs/phase1_test_plan.md`。
- 待补（不阻塞 Phase 2）：`docs/phase0_model_loading_test_plan.md` 里 T2（ONNX→Engine）仍未实施；
  T1 / T3 的能力已由 Phase 1.5 的 E1/E2 以更强的形式覆盖。

</details>

</details>

### 3.6 工具与文档

- `mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`：HF checkpoint → `config.json + model.safetensors` 转换脚本（真实实现的唯一落点）。
- `scripts/ref_rope.py` / `scripts/ref_sampler.py`：参考语义自检脚本（RoPE 与 HuggingFace 交叉验证、采样器截断语义与理论概率）。

<details><summary>展开：3.6 工具与文档 全文</summary>


<details><summary>展开：3.6 工具与文档 全文</summary>

- `requirements.txt`：转换工具依赖（已移到项目根目录）。
- `docs/mini_trt_llm_design.md`：v1.0 设计文档。
- `docs/phase0_development_plan.md`：Phase 0 开发计划。
- `docs/phase0_code_review_plan.md`：Phase 0 代码 review 方案（review 由用户本人执行，尚未完成）。
- `docs/phase0_model_loading_test_plan.md`：Phase 0 模型加载测试方案。状态：T1 / T3 的能力已由
  Phase 1.5 的 E1 / E2 以更强的形式覆盖；**T2（ONNX → Engine）仍未实施**。
- `docs/phase1_development_plan.md`：Phase 1 开发方案 + 关键决策确认清单（含合并后的 15 项决策）。
- `docs/phase1_test_plan.md`：Phase 1 全流程测试计划（模型加载 → builder 分发 → Plugin 挂载 → engine 构建/反序列化 → 推理 → 采样），含前置改造清单（G1/G2/G3）与实施顺序。
  E1–E4 的**设计**与链路图在这里（唯一来源）；执行结果与验收见 `phase1_5_development_plan.md` §0/§5。
- `docs/phase1_5_test_plan.md`：Phase 1.5 测试计划——分层（**S 支撑层** host 契约 / **E1** 单算子闭环 /
  **E2** 多算子链路 / **E3** 动态 shape / **E4** 错误路径）、
  **用例清单（用例 → 判据 → 出处 → 环境 → 状态）**、执行方式（含 `MINI_TRT_REQUIRE_GPU=1` 的真机口径）、
  覆盖缺口、结果快照。层名刻意不用 `L1`/`L2`——那套编号在 `phase1_test_plan.md` 里指"算子单测/集成"。
  **该阶段原先有意不写独立测试计划**（理由是"只重复 E1–E4 的设计"）；2026-09-25 补写时把定位限定为
  "索引 + 执行口径"，**设计仍只认 `phase1_test_plan.md` §4**，避免重开两处来源的坑。
- `docs/phase2_test_plan.md`：Phase 2 测试计划（补记）——分层（L0 host 契约 / L1 建网 / L2 数值 / L3 端到端）、用例清单、判据出处与缺口（G2-1 ~ G2-4）。
- `docs/phase3_test_plan.md`：Phase 3 测试计划——分层（L0 图结构 / L1a 参数校验 / L1b 图契约 / L2 数值 / L3 性能）、用例清单、判据出处与缺口（G1c 已关闭，余 G5/G6）。
- `docs/phase1_5_development_plan.md`：Phase 1.5 开发计划（P1.5-0 ~ P1.5-7 的任务、依赖、验收）。
- `docs/TROUBLESHOOTING.md`：问题排查记录（现象 / 定位路径 / 根因 / 修复 / 回归防护）。
- `mini_trt_llm/tools/validate/`：**INT8 判据的验收集规格与评估脚本**（`README.md` 定义 meta 必需字段 /
  重叠排除规则 / 率必带 n；`int8_eval.py` 出分层报告，含 `--self-test` 并已注册为 ctest 项
  `int8_eval_selftest`）。开发期被抓到的两个问题见 `TROUBLESHOOTING.md` #32。
- `mini_trt_llm/tools/validate/qdq_reference.py`（§3.0j 新增）：**`future_iterations.md` §1.5 的数值标尺**——用 ONNX 官方
  参考实现执行 Q/DQ 图并把指定张量落盘（量化前 + 量化后两份），含 `--self-test`（最小 Q/DQ 图
  **逐位**比手算 ONNX 语义）并注册为 ctest 项 `qdq_reference_selftest`。
  **为什么标尺是"直接执行这张图"**：图里 BN 已折好 → 根本不存在"折叠"这一步（#30.5 那两轮预算
  就是耗在自己折 BN 上）。opset 17 → 21 的提升由自检逐位兜底。
- `mini_trt_llm/tools/convert/add_probe_outputs.py`（§3.0j 新增）：给既有 Q/DQ 图**只追加图输出**
  （20 个 Conv 的**量化前**输出 + GAP 输出）产出探针图；自证"探针图 = 产物图 + 探针"
  （node / initializer / input / opset **逐字节不变**），含 `--self-test` 并注册为 ctest 项
  `add_probe_outputs_selftest`。
- `docs/future_iterations.md`：后续迭代计划（**§0 = 优先级与排序规则的唯一来源**；
  章节顺序是主题分类、不是优先级；`future_iterations.md` §1.5 / `future_iterations.md` §1.6 是两条已立项条目）。
- `docs/future_iterations_development_plan.md`：**执行层**——分批（A 可立即开工 / B 触发即做 /
  C 需外部前置 / D 冻结）、文件级改动面、步序、破坏性动作预告、待拍板决策 F1~F4。
  条目内部的做法与验收**不复制**到本文件，仍以 `future_iterations.md` 为准。
- `docs/future_iterations_test_plan.md`：**测试层**——分层 H（host/CI）/ G（真机）/ P（真机性能）、
  用例 → 判据 → 出处 → 环境 → 状态；含 BPE Tokenizer 与 INT8 判据规格的具体用例。
- `mini_trt_llm/third_party/sentencepiece/README.md`：记录禁用功能。
- `docs/PROGRESS.md`：本交接文档（已按 `progress-summary` skill 更新）。

> 历史文档：`docs/phase1_pending_confirmations.md` 的内容已全部合并进 `docs/phase1_development_plan.md` §10，原文件已删除。

</details>

</details>

### 3.7 第三方依赖

- `mini_trt_llm/third_party/sentencepiece`：v0.2.0，源码嵌入。
- `mini_trt_llm/third_party/safetensors-cpp`：main 分支，源码嵌入。
- `mini_trt_llm/third_party/googletest`：release 版本，源码嵌入。

### 3.8 注释规范修复

- 已按 `cpp-comment-style` skill 检查并修复 Phase 0 代码注释：
  - 公共 API（`precision.hpp`、`timer.hpp`、`json.hpp`）补齐类级/函数级注释。
  - 纯英文注释改为中文（`builder.hpp`、`memory_pool.cpp`）。
  - 魔数与边界逻辑补充"Why"注释（`builder.hpp` 默认值、`engine.cpp` 统计、`safetensors_loader.cpp` BF16 位运算）。

### 3.9 [DEC-PHASE1-DELIVERY] Phase 1 插件与采样器


<details><summary>展开：3.9 [DEC-PHASE1-DELIVERY] Phase 1 插件与采样器 全文</summary>

| 文件 | 说明 |
|---|---|


<details><summary>展开：3.9 [DEC-PHASE1-DELIVERY] Phase 1 插件与采样器 全文</summary>

| `include/mini_trt_llm/plugins/rmsnorm_{kernel,plugin}.hpp` + `src/plugins/rmsnorm_plugin.cu` | RMSNorm Plugin（一行一 block，FP32 `float4` / FP16 8×half 向量化，不能整除时回退标量） |
| `include/mini_trt_llm/plugins/rope_{kernel,plugin}.hpp` + `src/plugins/rope_plugin.cu` | RoPE Plugin（half-split 约定，双输入双输出，`position_ids` 作为输入） |
| `include/mini_trt_llm/plugins/paged_attention_{kernel,plugin}.hpp` + `src/plugins/paged_attention_plugin.cu` | PagedAttention Plugin（仅 Decoding，GQA/MHA，online softmax 单趟扫描） |
| `include/mini_trt_llm/sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu` | Greedy / Top-K / Top-P 采样器（设备侧 API，内联 Philox 随机源）。P9_2-5 后 Top-P 的生产路径 = CUB 排序 + **行内并行**采样 kernel；旧 kernel 由 `LaunchTopPSamplerLegacy` 保留作对照（见 §3.0g） |
| `include/mini_trt_llm/sampler/nucleus_cutoff.hpp`（P9_2-5 新增） | Top-P 的交叉点定位 `FindFirstPrefixCrossing`：`__host__ __device__`，同一份实现给 kernel 与 host 用例共用（沙箱内可裁决，见 `tests/test_sampler.cpp` 的 `NucleusCutoffTest.*`） |
| `include/mini_trt_llm/utils/cuda_dtype.cuh` / `cuda_reduce.cuh` | 多 kernel 共用的 dtype 转换与 block 归约 |
| `tests/test_{rmsnorm_plugin,rmsnorm_integration,rope_plugin,paged_attention_plugin,sampler}.cpp` | 算子单测 + L2 集成测试 |
| `tests/test_gpu_guard.hpp` / `test_reference.hpp` | 共享的 GPU 门控与 CPU 参考实现 |

- 已确认决策的落地：RMSNorm 不带 bias、weight 作第二输入；RoPE `rotary_dim` 默认 `head_size` 且可配置、`position_ids` 作输入；PagedAttention `block_size` 强制显式配置、`scale` 为属性（默认 `1/sqrt(head_size)`）；采样器 k/p 为 per-batch tensor、随机源为 host seed + device Philox。
- 统一的接口纪律：所有 Plugin 的 `getWorkspaceSize()` 返回 0、`enqueue` 内零分配、失败以错误码返回（`enqueue` 是 `noexcept`）；`supportsFormatCombination` 一律只读 `inOut[0..pos]`。
- 注册表接入：`PluginRegistry::RegisterAllPlugins()` 登记三个 Plugin 的 creator，与 `REGISTER_TENSORRT_PLUGIN` 的 TRT 全局注册并存（前者给本框架按名查找，后者给 engine 反序列化）。
- 构建改动：`mini_trt_llm/CMakeLists.txt` 的源文件 glob 增加 `src/*.cu`，否则 nvcc 产物不会进静态库。
- 验证状态：沙箱内 `ctest` 67 个用例 **0 失败**（26 个 GPU 用例自动跳过）；
  用户在 WSL2 真机上跑 `--gtest_filter='RoPE*:PagedAttention*:Sampler*'` **全部通过**，
  加上此前已通过的 `RmsNorm*`，**Phase 1 全部算子（kernel 数值 + engine 集成）均已在真机验证**。
  过程中修复了一个只在部分旋转下暴露的 RoPE 缺陷（见 `docs/TROUBLESHOOTING.md` #4）。
  ⚠️ 该数字是 Phase 1 收尾时的快照，**当前总数只认 §3.5**（§3.10 与本节都只是历史快照，勿叠加）。
- 参考数据脚本：`scripts/ref_rope.py`（与 HuggingFace `apply_rotary_pos_emb` 交叉验证，最大差异 0.0）、
  `scripts/ref_sampler.py`（Top-K/Top-P 截断语义与理论概率）。

</details>

</details>

### 3.10 [DEC-PHASE15-DELIVERY] Phase 1.5：全流程测试基建与收尾


<details><summary>展开：3.10 [DEC-PHASE15-DELIVERY] Phase 1.5：全流程测试基 全文</summary>

| 文件 | 说明 |
|---|---|


<details><summary>展开：3.10 [DEC-PHASE15-DELIVERY] Phase 1.5：全流程测试基建与 全文</summary>

| `tests/e2e_safetensors_writer.{hpp,cpp}` | 测试用 Safetensors 写入 helper（B/F16/BF16），让端到端夹具自包含、不依赖 Python |
| `tests/e2e_fixture.{hpp,cpp}` | 临时模型目录（`mkdtemp` + 析构清理），组装 `config.json` + `model.safetensors` |
| `tests/test_e2e_error_paths.cpp` | E4：5 条错误路径用例（3 条 host 侧可进 CI，2 条需 GPU） |
| `tests/test_e2e_single_op.cpp` | E1：4 条单算子闭环（RMSNorm / RoPE / PagedAttention / Sampler），重量经真实 `WeightLoader` 取用 |
| `tests/test_e2e_mini_decoder.cpp` | E2：**完整链路** `RMSNorm → QKV → Slice/Reshape → RoPE → PagedAttention → RMSNorm → LM Head → Top-K Sampler`，四权重均以 BF16 存储，与独立 CPU 参考对比 |
| `tests/test_fp16_paths.cpp` | FP16 覆盖缺口：RoPE（含 GQA + batch>1 + 部分旋转）、PagedAttention、Greedy Sampler |
| `tests/test_reference_helpers.cpp` | 参考实现自身的 meta-test（7 条 host 用例），含 `RopeIsBatchAware` 回归 |
| `tests/test_e2e_dynamic_shape.cpp` | E3：Prefill 变长序列 / Decode 变 batch / 超范围拒绝 |
| `mini_trt_llm/tests/test_safetensors_loader.cpp` | P1.5-0 的 8 条回归用例（dtype 组合矩阵、指针不别名、零拷贝） |

- **产品代码修复**（详见 `docs/TROUBLESHOOTING.md` #5 / #6 / #7）：
  - `SafetensorsLoader` 转换路径的 3 个缺陷：BF16 被误判为同类型而返回原始数据、往设备内存从主机侧写、
    BF16→FP16 位截断在数值上错误；转换缓存改为按 key 隔离。
  - `IModelBuilder::Build(const WeightLoader&)` 原先取不到权重（`GetWeight` 非 const），
    读取路径改为 const + `mutable` 缓存。
  - `EngineBuilder::BuildFromConfig` 调整校验顺序：纯数据校验前置于 `createInferBuilder`。
- **动态 shape 打通**：`AddCvOptimizationProfile` / `AddLlmOptimizationProfiles` 由"只声明未定义"
  变为已实现并接入；新增 `Engine::SetOptimizationProfile` 支持多 profile 切换。
- **真机复验发现并修复的缺陷**：反序列化后的 Plugin 从未执行 `configurePlugin`，
  而 RoPE / PagedAttention 的 head 配置是靠构建期从形状推导的、不作为序列化属性，
  导致 `onShapeChange` 把正常形状误判为"改了 head 配置"，所有 RoPE 端到端用例在 enqueue 失败。
  已改为以运行期形状为准刷新，并补 4 条 **host 侧**回归用例（详见 `docs/TROUBLESHOOTING.md` #8）。
- **第二次真机复验发现并修复的缺陷**：参考实现 `ReferenceRoPE` 漏了 batch 维度，导致
  `Fp16PathTest` 在 batch>1 时误判为 kernel 出错。已把参考实现收敛为唯一来源、加 batch 参数
  与入口断言，并补 7 条 host 侧 meta-test（详见 `docs/TROUBLESHOOTING.md` #9）。
- 验证状态（Phase 1.5 收尾时的快照）：沙箱内 `ctest` **104 个用例 0 失败**
  （40 个 GPU 用例自动跳过，64 个 host 用例实际执行）；**当前总数见 §3.5**。
  参考实现这一层由 `test_reference_helpers.cpp` 在 CI 内自证。
- **真机验证**：E1 / E2 / E3 与 E4 的 2 条 GPU 用例、FP16 覆盖用例（共 5 条）**全部通过**。
  Phase 1.5 完成闭环——进入 Phase 2 建模前所需的组件（插件 / 采样器 / 权重加载 / 动态 shape /
  端到端骨架）均已实现并在真机验证。

---

</details>

</details>

## 4. 进行中 / 未完成的部分

**当前没有进行中的阶段，也没有进行中的迭代**：Phase 0 / 1 / 1.5 / 2 / 3 / 4 全部完成（§4.1 ~ §4.5），
Phase 5 已永久取消（§4.6）。**Phase 之后的四条工作流都已于 2026-09-27 收口**：

| 工作流 | 状态 | 结论在哪 |
|---|---|---|
| **`future_iterations.md` §9.2 采样器高性能 kernel** | 已关闭（P9_2-5b 判"无显著差异"、5c 不做） | §3.0g |
| **§6.3 / G6 decode 性能画像**（含 profile target 与可复现测量方法） | 已关闭（逐 kernel 分解记为**能力边界**；PF-7 移交 `future_iterations.md` §10.2） | §3.0h |
| **§2.2 长上下文 attention（split-K）** | **已交付并真机验证**（斜率降幅 88.85% / 88.20%） | **§3.0i**（关键设计 §2.17） |
| **`future_iterations.md` §1.5 / P4-INT8-a per-channel 整网退化根因** | **已结案**（根因在**产图脚本**、不在 TRT；离线反证 + 文件级证据 + 真机 B1 四条全绿） | **§3.0j**（证据链 `docs/TROUBLESHOOTING.md` #46 / #47；两件"暂不做"见开发计划 §13.11） |

下面各节保留的是**各阶段当时的交付与残余缺口快照**；仍然活着的开放项一律看 §6.6 的索引
（唯一事实来源 = `future_iterations.md` §11）。

### 4.1 Phase 1：Plugin 基础（已完成）

- 决策状态：15 项待确认问题已全部关闭，无遗留阻塞项（详见 `docs/phase1_development_plan.md` §10）。
- ✅ 完善 `IPluginV3` 基类，补齐 TRT 10.x 接口。

<details><summary>展开：4.1 Phase 1：Plugin 基础（已完成） 全文</summary>


<details><summary>展开：4.1 Phase 1：Plugin 基础（已完成） 全文</summary>

- ✅ 实现 `RMSNormPlugin` + 单元测试（GPU 用例已在真机验证通过）。
- ✅ `RMSNormPlugin` 接入 `PluginRegistry`，并补 L2 集成测试（真实 TRT network → engine 序列化/反序列化 → 推理）。
- ✅ 实现 `RoPEPlugin` + 单元测试。
- ✅ 实现 `PagedAttentionPlugin`（Decoding 阶段 GQA/MHA）+ 单元测试。
- ✅ 实现 Sampler CUDA Kernels（Greedy / Top-K / Top-P）+ 单元测试。
- ✅ 全部 GPU 用例已在用户 WSL2 真机验证通过。

Phase 1 明确不在本次范围内、留待后续的项：

- PagedAttention 的 **Prefill 阶段**（query 序列长度 > 1）——需要因果 mask 与分块，与 Decode 路径 kernel 结构差异大。
- 采样器分布级对比（D3）已落地为两件事：C++ 侧的统计检验（词频收敛到解析 softmax 概率），以及
  `scripts/ref_sampler.py` 打印的 HF 风格截断语义；尚未做的是把 Python 输出固化成数据文件供测试载入。
- ~~Sampler 的手写高性能 kernel~~ **已于 2026-09-27 交付**（§3.0g：Top-P 改"保留 CUB 排序 +
  行内并行"；Top-K 快速路径正确但性能不达标、已撤出生产）。剩下的是把采样器参考数据固化成数据文件
  = `future_iterations.md` §9.3（P2，纯 host）。

</details>

</details>

### 4.2 Phase 1.5：全流程测试基建与收尾（已完成）

- ✅ P1.5-0：修复 `SafetensorsLoader` 转换路径的 3 个缺陷（详见 `docs/TROUBLESHOOTING.md` #5）。
- ✅ P1.5-1 / P1.5-2：Safetensors 写入 helper；E4 错误路径用例。
- ✅ P1.5-3：E1 单算子闭环（4 条）。
- ⚠️ P1.5-4：E2 **缩减完成**——交付了多权重 BF16 路径的数值验证，
  完整 `RMSNorm → QKV → RoPE → PagedAttention → LM Head` 链路与 `ref_mini_block.py` 有意留后
  （理由见 `docs/phase1_5_development_plan.md` §0.1）。
- ✅ P1.5-5 / P1.5-6：Optimization profile 实现；E3 动态 shape 测试。
- ✅ P1.5-7：文档归位 4/4，含把 `phase0_development_plan.md` 的验收判据改写为可执行形式，
  并补上 Phase 0 遗漏的「Optimization profile 能力可用」一条。

真机复验：E1 / E2 / E3 与 E4 的 2 条用例**已通过**。

### 4.3 Phase 2：GPT-2 原生构建（已完成，见 §3.0a）

- 开工顺序与风险提示见 **§6.2**；关键事实（GPT-2 不用 RMSNorm / RoPE）见 **§6.1**。
- 先做多权重加载 spike（约 150 个张量、BF16/FP16 源），再确认 LayerNorm / GELU(tanh)
  用 TRT 原生层够用，然后实现 `GPT2ModelBuilder`（先 Prefill 单形状）。
- 之后接 `PagedAttention` + Decode 引擎 + Prefill/Decode 双 profile，实现 `LLMRunner`
  的 `Prefill → Decode` 自回归循环。
- 精度对比基准：`1_gpt2_onnx/ref_output.bin`（PyTorch FP32）。

### 4.4 Phase 3：GPT-2 ONNX + Plugin（已完成，见 §3.0b）

- 实现 `OnnxBuilder` + subgraph replacer。
- 对 `1_gpt2_onnx/gpt2.onnx` 替换 RoPE / RMSNorm / Attention 子图。
- 验证与方案 A 输出一致。

### 4.5 [DEC-PHASE4-STATUS] Phase 4：ResNet18 替换（✅ 已完成，2026-09-26）

**交付清单与实测数字见 §3.0d**。计划文档 `docs/phase4_development_plan.md`（§1 保留了
"先读 `0_resnet18_onnx` 历史工程"的四条关键发现，供后续参考）：

<details><summary>展开：4.5 [DEC-PHASE4-STATUS] Phase 4：ResNet18 替换（ 全文</summary>


<details><summary>展开：4.5 [DEC-PHASE4-STATUS] Phase 4：ResNet18 替换（✅  全文</summary>


1. 该 ONNX **已在导出时折叠 BatchNorm**（42 个 FP32 张量全是 Conv/Gemm 的 weight+bias），
   算子是 `Conv/Relu/Add/MaxPool/GlobalAveragePool/Flatten/Gemm`——**原生建图不需要任何 Plugin**；
2. 历史工程**没有留下外部基线**：它的"精度验证"是自相对（FP16/INT8 vs 它自己的 FP32），
   且推理输入是合成 ramp、benchmark 输入是常量 0.5 → Phase 4 必须先造 torchvision 基线；
3. **INT8 的三条口径互相矛盾**（本文档 §6 写"INT8 延后"、`future_iterations.md` §1.1 列为后续、
   历史工程其实已有 calibrator + 500 张真实校准图）→ 由计划的 **D2** 收敛；
   收敛结果 = 走 **Q/DQ 显式量化**，`future_iterations.md` §1.1 的 implicit 校准路线现已**冻结**
   （2026-09-26 重定，理由见该文件 §0.1）；
4. `EngineBuilder::Config` 的 CV `opt_batch` 默认是 **1**，历史工程用的是 **8** → 会影响性能结论。

任务分解 P4-0 ~ P4-8、测试要点、判据出处都在该计划里；INT8 子计划见 `docs/phase4_int8_plan.md`。
**结论**：ResNet18 的 **FP32 / FP16 都健康**；**INT8 走 Q/DQ 显式量化、判据用"FP32 余量子集一致率"**
（实测 12/12 = 100%），权重默认 per_tensor。**per-channel 的整网退化原因未知** → 开放项 §6.6。

</details>

</details>

### 4.6 [DEC-PHASE5-CANCELLED] Phase 5：清理旧模块（❌ 已永久取消，2026-09-26 由用户决定）

- **用户决定：Phase 5 永久取消**。旧模块 `0_resnet18_onnx/`、`1_gpt2_onnx/` 与根 `CMakeLists.txt`
  里的注释项**保持原样**，**由作者本人按需处理**；Agent **不要**删除或移动它们。
- **为什么不能擅自删**：它们不只是"旧代码"——`0_resnet18_onnx/` 还是 Phase 3（`gpt2.onnx` 走
  `1_gpt2_onnx/`）与 INT8（`calib_data/` 500 张真实图 + `resnet18.onnx`）的**本地产物来源**，
  删掉会让 ONNX / INT8 用例全部跳过。这也与 `AGENTS.md` §0.6 一致：**"这东西没人用"的判断权在作者**。

---

## 5. 已知问题与坑

### 5.0 [DEC-EOS-EARLY-STOP] `LLMRunner` 无法在解码循环内早停 EOS（有意为之的 workaround）

- **问题**：AGENTS.md §3.A.3 要求解码循环内不得有 H2D/D2H 拷贝，而"一见 EOS 就停"
  必须先知道刚采样出的 token 值（在设备上）。
- **影响**：EOS 之前仍会按 `max_new_tokens` 跑满，多余的计算被丢弃；
  返回结果在 host 侧截断到首个 EOS，因此**语义正确、只是多算**。
- **Workaround**：`LLMRunner::Config::eos_token_id`（-1 表示不截断）；
  截断发生在循环之后。
- **后续可选方案**：设备端维护一个 "stop flag" 并让循环条件读它（需要条件图或
  每步一次 4 字节 D2H+sync，后者违反上述约束）；或改成设备侧常驻的采样-停止判定。

### 5.1 自研 JSON 解析器能力有限

- **问题**：Phase 0 用 `include/mini_trt_llm/utils/json.hpp` 自研极简解析器，仅支持基础类型和简单嵌套。
- **影响**：复杂配置可能解析失败。
- **Workaround**：后续迭代替换为 `nlohmann/json` 单头文件。

### 5.2 `SafetensorsLoader::GetTensorNames()` 返回空

- **问题**：`syoyo/safetensors-cpp` 的 `ordered_dict` 不暴露 key 列表遍历接口。
- **影响**：无法枚举所有张量名。
- **Workaround**：当前只按名查询权重，不影响功能；后续可换库或自研解析。

### 5.3 SentencePiece 与 GPT-2 BPE 可能不对齐

- **问题**：GPT-2 原生用 BPE，SentencePiece 行为可能与之有差异。
- **影响**：Tokenizer 结果可能与 Python `transformers` 不完全一致。
- **Workaround**：后续实现 `BpeTokenizer : BaseTokenizer`，与 Python tokenizer 逐 case diff。

### 5.4 BF16 转换（已修复）

- **原问题**：`safetensors_loader.cpp` 中 BF16→FP16 直接截断尾数，非 round-nearest；
  且当时没有意识到 BF16 与 FP16 的指数位宽度不同（8 vs 5），位截断在数值上根本不成立。
- **现状态**：已由 P1.5-0 修复——统一先还原成 FP32 再降到目标精度，并补齐
  FP16→FP32 / FP32→FP16。详见 `docs/TROUBLESHOOTING.md` #5。

### 5.5 沙箱环境无法访问 GPU

- **问题**：Agent 运行环境的 `nvidia-smi` 报 `GPU access blocked by the operating system`，CUDA API 无法初始化。
- **影响**：Agent 无法本地验证 CUDA 相关测试。
- **Workaround**：CUDA 测试必须在用户真实 WSL2 环境手动运行验证。

### 5.6 `.gitmodules` 曾出现重复条目

- **问题**：根目录与 `mini_trt_llm/third_party/` 下曾同时存在 submodule 条目。
- **状态**：已清理，当前仅保留 `mini_trt_llm/third_party/sentencepiece` 与 `mini_trt_llm/third_party/safetensors-cpp`。
- **注意**：新增第三方依赖时避免在根目录再建 submodule。

### 5.7 [DEC-GPU-GATING] Phase 0 utils 测试缺少 GPU 门控（已修复）

- **问题**：`CudaCheckTest`、`DeviceBufferTest`、`PinnedBufferTest`、`CudaTimerTest` 共 6 个用例直接调用 CUDA API 且未做环境判断，在无 GPU 环境下抛 `cudaErrorInsufficientDriver` 而失败，而不是跳过。
- **影响**：无 GPU 的 CI / 沙箱里 `ctest` 永远不绿，真实回归信号被固定噪声淹没（用户真机上这 6 个用例是过的）。
- **Workaround**：已抽出共享的 `tests/test_gpu_guard.hpp`（`test_support::HasCudaDevice()`）给这批用例加门控；
  例外的 `CudaCheckTest.InvalidDeviceThrows` 刻意不门控，因为它验证的是 `CUDA_CHECK` 的失败路径。
- **排查过程**：见 `docs/TROUBLESHOOTING.md` #1。

### 5.8 `supportsFormatCombination` 越界读取导致 engine 构建失败（已修复）

- **问题**：`supportsFormatCombination` 扫描了 `inOut[pos+1..]`——TensorRT 未初始化的位置，导致所有格式组合都被判为不支持，`buildSerializedNetwork` 报 `could not find any supported formats consistent with input/output data types`。
- **影响**：Plugin 无法构建成 engine；host 侧单测完全无感，只有真机能复现。
- **Workaround**：已改为只与 `inOut[0]` 比对；新增回归用例 `RmsNormPluginTest.IgnoresInvalidDescriptorsAfterPos`。
- **排查过程**：见 `docs/TROUBLESHOOTING.md` #2。

### 5.9 [DEC-SAMPLER-OLD-API] 采样器曾存在两套 API（已清理）

- **问题（已解决）**：Phase 0 留下的 `sampler/{greedy,topk,topp}_sampler.{hpp,cpp}` 声明的是「标量 k/p + host `std::vector` 输出」的接口，与已确认的 Q7（per-batch tensor）和「Decode 全程驻留显存」冲突，函数体仍是 `throw not implemented`。
- **解决**：经用户授权删除 6 个旧桩文件，采样器 API 统一收敛到 `sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu`。

### 5.10 [DEC-SANDBOX-NO-GPU] GPU 用例在沙箱内无法执行（已确认为环境限制，非缺陷）

- **问题**：所有 kernel 数值与 engine 集成用例都需要 GPU，沙箱内只会 `GTEST_SKIP`。
- **影响**：Agent 侧的结论上限是「编译通过 + 契约自洽 + host 侧逻辑正确」。
- **现状**：Phase 1 全部 GPU 用例已由用户在真机验证通过；这是**流程约束而非遗留缺陷**，
  后续 Phase 每完成一个算子，都需要同样走一遍真机验证。

---

### 5.11 [DEC-GPT2-FP16-LIMIT] GPT-2 的 FP16 端到端不可用（已知限制，按政策不修）

- **问题**：真实 GPT-2 在本项目的**弱类型 FP16** 引擎下端到端产生 NaN（贪心输出恒为 0）。
  出现 NaN 的层随构建变化（实测 0/1/2），而激活幅值远未触及 FP16 上限 65504。

<details><summary>展开：5.11 [DEC-GPT2-FP16-LIMIT] GPT-2 的 FP16 端到端不 全文</summary>


<details><summary>展开：5.11 [DEC-GPT2-FP16-LIMIT] GPT-2 的 FP16 端到端不可用 全文</summary>

- **影响**：**GPT-2 的推荐精度是 FP32**。FP16 只能用于算子/网络层验证（Phase 1.5 已覆盖），
  不能用于 GPT-2 的端到端推理。
- **已排除**：LayerNorm 计算精度（显式设 FP32 后仍 NaN）、`c_fc`/`gelu_new`
  （两处切点均干净）、"残差膨胀到范围溢出"（幅值全在几十以内）。
- **Workaround**：用 FP32（已端到端验证：8/8 贪心 token 命中、logits 相对偏差 `1e-6`）。
- **后续路径**：`docs/future_iterations.md` §1.4（关键算子保 FP32 → 逐算子二分 → 激活缩放）。
- **完整定位过程（5 轮真机往返）**：`docs/TROUBLESHOOTING.md` §18.1。
- **残留仪器变成的真缺陷（已修复，2026-09-25）**：定位时在图上留的 4 个中途输出
  （`mlp_fc_0` / `mlp_gelu_0` / `attn_res_0` / `mlp_res_0`）曾经**无条件挂在图上**，
  而消费方按名绑定、没人绑它们 → 真机 6 条用例 red（建网断言 `9 ≠ 5`、decode-consistency 与
  runner 全部 enqueue 失败，TRT 直接点名 `mlp_fc_0`）。
  **现方案（F2）**：诊断输出由 `BuildOptions::export_diagnostics` 控制、**默认关**，
  只有 `Fp16PrefillOutputsDiagnostic` 打开（并用独立引擎路径）。真机复验：9 条目标用例全绿，
  诊断仪器读回的中途张量数值与当初记录逐位一致。
  完整证据与教训见 **`docs/TROUBLESHOOTING.md` #19**，任务与验收见
  **`docs/phase2_supplement_plan.md`**。

</details>

</details>

### 5.12 Phase 2 修掉的缺陷（结论索引）

Phase 2 的 5 个真缺陷（粘性 CUDA 错误 / KV 写入路径 / 多层共用 cache / 按层推进长度 /
**FP16 缓冲按假定精度分配导致越界写**）全部已修复并有回归用例，
经过与推导见 `docs/TROUBLESHOOTING.md` #13 ~ #16 与 #18（前半）。
其中两条**影响接口设计**（均已落到 §2.15 的表里，勿回改）：

- **#16**：`PagedKVCache` 的追加接口拆成 `AppendDecodeKV`（只写）+
  `AppendDecodeStep`（一次写全部层、只推进一次长度）——不要按"每层调用一次并各自推进"的直觉改回去。
- **#18（前半）**：凡"按配置推断别人的宽度"的地方都要改成"向对方查询"，即边界精度查询、
  `source_is_half` 与 decode cache 输入精度校验这三条。

### 5.13b 性能"改动前后差百分之几"在本平台的判别下限（2026-09-27 沉淀，**含 workaround**）

- **问题**：要裁决"换个实现快了百分之几"时，本平台（WSL2 + 单卡 + 无 Tensor Core）上单次/分段测量的
  噪声与固定开销**和信号同量级**：实测一个平凡的 `greedy` 内核单发就要 34~156 µs；分段测出来的
  `top-p − top-k` 会出现负值；跨 session 的同一个实现能差 ±23%。
- **影响**：把"未获支持"误写成"已否证"、或凭 cycle 估算直接改 kernel，都发生过一次（各花一轮真机）。
- **Workaround（perf harness 已内置）**：① **同二进制 A/B**（把改动前/后的两版都编进去，
  `LaunchTopPSamplerTwoLevel` 即此用途）；② **同轮交替测**（ABBA，抵消轮内顺序偏置）；
  ③ **斜率口径** `(T4−T1)/3`（同一窗口发射 1 次与 4 次求差，扣掉每窗口固定开销）；
  ④ 报**中位数 + p25/p75**（极差会被环境脉冲污染到 ±800 µs）。
- **结论性数字**：这台机器对这类问题的**判别下限约 ±400 µs**；小于该量级的差异**不要改代码**，
  先确认尺子够不够用。协议级教训见 `TROUBLESHOOTING.md` **#37 / #38**，实现见
  `tests/test_sampler.cpp` 的 `SamplerPerf.ThroughputByShape`。

### 5.13 [DEC-ARGMAX-CASE] 真机新红：`Gpt2OnnxTest.MatchesAcrossProfileShapes` 在 seq=512 上 argmax 不等（**已按方案 B 结案**）

- **问题**：2026-09-26 真机全量里，`batch=1 seq=512` 的**逐行 argmax 相等**断言失败；
  同一次运行中 `cosine`（`> 0.999999`）与相对界（`< 1e-5`）**都通过**

<details><summary>展开：5.13 [DEC-ARGMAX-CASE] 真机新红：`Gpt2OnnxTest.Ma 全文</summary>


<details><summary>展开：5.13 [DEC-ARGMAX-CASE] 真机新红：`Gpt2OnnxTest.Matc 全文</summary>

  （实测 `max_abs 0.000274658 / 相对 1.97989e-06 / cosine 1`），其余形状（`(1,1)`、`(1,64)`、`(2,4)`、`(2,64)`）也都通过。
- **机制（2026-09-26 定量，证据见 `TROUBLESHOOTING.md` #34.6）**：翻转只有 **1 行 / 512**（行 118，类别 79 vs 325），
  两次**独立重建**后位置与类别完全相同、native 余量逐位相同（`1.53e-05`）→ **不是构建噪声，是稳定并列**。
  该行上"两引擎差异"（`0.84~1.14e-04`）比并列间距大 5~7 倍；HF 第三方参考在同一行选 **79**（与原生同侧）。
  全 512 行中**只有这 1 行**的余量低于两引擎差异（第二小的余量 ≥ `1.14e-04`）。
- **为什么这构成"期望值本身有问题"的依据**：项目**已冻结**的数值口径是相对 `< 1e-5`，在 `|logit|≈87` 处
  等价于允许 `~8.7e-04` 的绝对差——比该并列间距大 ~57 倍。**准确说法是"不能保证"而不是"不可能"**：
  在并列行上全等与否**取决于两侧误差的符号**（同侧绿、异侧红），因此这条判据在并列行上
  **不携带"实现是否正确"的信息**，只反映构建态。Phase 3 那次"真机通过"与今天的红可以同时成立——
  证据与三个可核对事实见 `TROUBLESHOOTING.md` **§34.8**（输入自引入未变、native 图事后被改过两次、
  用例按文件存在性复用不随代码失效的引擎缓存）。
- **性质**：**不是**本次交付引入的（改动面不触及 GPT-2 的 ONNX / 原生建图路径，
  且两条引擎是本次新建的），但它是全量报告里除"按设计红"之外的唯一红。
- **处置（作者 2026-09-26 决定：方案 B）**：判据改为"**可判行（`m > 2d`）必须 argmax 全等；
  不可判行（`m ≤ 2d`）允许不同，但**必须打印并钉到行号**（登记表 `ExpectedUndecidableRows()`）**。`2d` 是**可证**的门槛
  （推导见 `TROUBLESHOOTING.md` §34.9），数值判据（`cosine` / 相对界）**一个字没动**。
  实现：`gpt2_test_support.hpp` 的 `CompareArgmaxByDecidability`（唯一实现）+
  `test_gpt2_onnx.cpp` 的断言与打印 + `test_argmax_criterion.cpp` 的 6 条 host 用例
  （含用实测数字复现本条的 `IncidentRow118IsClassifiedUndecidable`）。
- **结案（2026-09-26 真机复跑）**：**PASSED（4.58 s）**。五个形状的不可判行数实测为
  `(1,1)=0`、`(1,64)=0`、`(1,512)=1（行 118）`、`(2,4)=0`、`(2,64)=0` ——
  **713 行里只有那 1 行**，且与登记表逐行一致（无新增行）。数值项全过（相对 `1.05e-06 ~ 2.09e-06`，`cosine = 1`）。
- **那一行的真相（9 位有效数字）**：native 与 ONNX **各自都只看到约 `1.5e-05` 的间距，且方向相反**；
  两者对这两个 logits 本身的分歧是 `3.8e-05`/`6.9e-05`——**分歧比要分辨的间距大 2.5~4.5 倍**。
  即"不是谁算错了，而是这个问题在两边各自的精度下没有答案"。
- **已实现（同日）**：这两条引擎路径的静默复用已改为**构建指纹**驱动的显式重建（见 §3.0f / `TROUBLESHOOTING.md` #34.10）。
- **完整路径与判读规则**：`docs/TROUBLESHOOTING.md` **#34**；执行入口见
  `docs/future_iterations_development_plan.md` §8.5。

</details>

</details>

### 5.14 §2.2 过程中沉淀的三条操作类坑（2026-09-27，**结论在此，过程见 TROUBLESHOOTING**）

这三条都**不是产品缺陷**（产品代码一行未改），但每一条都能让"验证"这件事静默失效，
所以按 §6 的要求只在此留结论、把定位路径指向故障记录：

| 坑 | 结论（一句话） | 影响 / workaround | 出处 |
|---|---|---|---|
| **`ctest -R` 静默空跑** | `ctest -R` 收**正则**、`--gtest_filter` 收**过滤器**；写成 `'A.*:B.*'` 会**一条都不匹配**，ctest 打 `No tests were found!!!` 且**退出码仍是 0** | "全绿"里可能什么都没跑。**判据是"输出了几行 `Test #N`"，不是退出码**；拿不准就直接调二进制 + `--gtest_filter` | `TROUBLESHOOTING` **#43** |
| **长上下文夹具自相矛盾** | 块表行宽 < `ceil(context_len / block_size)` 时，`block_table[t/block_size]` 越界 → **host 参考 SEGFAULT**（不是 CUDA 错误） | 症状是 `Exception: SegFault` 而非断言失败。已加 `MakeLongContextFixture`（宽度由上下文反推）+ `AssertFixtureConsistent`（把夹具写错变成可读失败）。**另外：越界夹具也可能侥幸"通过"，那种通过没有信息量** | `TROUBLESHOOTING` **#44** |
| **"同 session 漂移"与阈值来路** | ① 我曾把"锚点取 max 更保守"写反（取 max 是**宽松**）；② 曾议的"退化 ≤2%"其唯一数值输入（单次发射 3~6 µs）**无出处**，且 PP-1/PP-2 对同一笔代价差 **1.97× 未解释** → 该判据**改为观测项、不设阈值** | 判据里出现"漂移"必须钉死是**哪一种**（绝对 max/min 还是配对差）；**预估里的每个数值输入都要有出处**，否则几轮后会被当成阈值引用 | `TROUBLESHOOTING` **#45 / #45.1** |

### 5.15 `future_iterations.md` §1.5 过程中沉淀的四条"仪器类"坑（2026-09-27，**结论在此，过程见 TROUBLESHOOTING**）

同 §5.14 的性质：**都不是产品缺陷**（产品代码一行未改），但每一条都能让排查本身失效。

| 坑 | 结论（一句话） | 影响 / workaround | 出处 |
|---|---|---|---|
| **量化 scale 取自"另一张张量"** | `quantize_resnet18.py` 用**未折 BN** 的 torchvision 权重算 scale、却量化**已折 BN** 的 ONNX 权重（折叠系数逐通道 0.05~19.9）→ per-channel 16.19% 的 int8 权重被 clamp 饱和 | 这就是 P4-INT8-a 的根因。修法：`--weight-range-source onnx`（默认**未改**，正式产物逐字节未变）；脚本已加**来源自检**（不一致就 `[WARN]` + 写进 meta，只报不拦） | `TROUBLESHOOTING` **#46 / #47.3** |
| **"预检"静默失真** | 旧 `fake_quant_check` 报 `max_abs ≈ 3.9`，而真机/参考是 **≈22**（差 5 倍多）——差的根因就是上一条：它量的是未折 BN 的权重 | 症状：预检"看起来很正常"，没有任何报错。修法：预检模型改成**与图逐位等价**（权重取自图 + BN 置成精确恒等）。**判据**：预检与真机/参考必须在**同一量级**，否则先查仪器 | `TROUBLESHOOTING` **#28 / #46.3** |
| **探针图会改变后端行为** | 挂 21 个图输出后，TRT 的 tactic 从 `i8i8` 变成 `volta_fp32_icudnn_int8x4_*`（层数 44 → 78）——**仪器改变了被观测对象** | 不能只看"某个计数变了"，要问"现象还复不复现"：**B1-4（复现对照）是硬门**；另加逐层 ONELINE 落盘供人核对 | `TROUBLESHOOTING` **#47.2** |
| **遍历 I/O 张量时形状问错对象** | `ICudaEngine::getTensorShape` 对动态维返回 **-1** → 转 `size_t` 成天文数字 → 报错伪装成"**显存分配失败：input**" | 形状**只能问 `IExecutionContext`**（`setInputShape` 之后），并对任何 ≤0 的维显式报错。既有用例没踩到它，是因为它把尺寸硬编码成常量、从不枚举张量 | `TROUBLESHOOTING` **#47.1 / #47.4** |

## 6. [DEC-NEXT-STEPS] 下一步计划

**没有下一阶段。** Phase 0 / 1 / 1.5 / 2 / 3 / 4 全部完成，**Phase 5（清理旧模块）已永久取消**（§4.6）。

**执行层计划（2026-09-26 产出）**：`docs/future_iterations_development_plan.md`（分批：A 可立即开工 /
B 触发即做 / C 需外部前置 / D 冻结；含文件级改动面、步序、破坏性动作预告）与
`docs/future_iterations_test_plan.md`（用例 → 判据 → 出处 → 环境 → 状态）。
**启动任何一项之前先读这两份**，再按 `AGENTS.md` §5 走一遍计划对账。

> **2026-09-27 结账：当前没有待办。** 清单里剩下的**全都是"等触发"**（需求 / 硬件 / 资产 / 测量），
> 唯二零前置的是 **`future_iterations.md` §9.3**（采样器参考数据固化）与 **§10.1**（ONNX / 原生 I/O 契约统一），
> 而它们**不服务任何现存需求**（`future_iterations.md` §10.1 的真正受益者是 §10.2，而 §10.2 要先跑 PF-7）。
> **下一步不该是"把 `future_iterations.md` 做完"**——按该文件 §0 的定位，它是**触发驱动的清单、
> 不是待办队列**；现在就动手等于为不存在的需求写代码。

**当前建议的下一步（按 `future_iterations.md` §0.3 的排序规则 = 前置可否立即满足 → 解锁广度 → 成本）**：

> **上一批（decode 性能画像 / §6.3 / G6）已于 2026-09-27 完成并关闭**，结论见 §3.0h 与开发计划
> `future_iterations_development_plan.md` §11.5.2：三个问题都有答案（sampler 占比、attention 占比、显存分配够不够贵）；逐 kernel 分解记为
> **能力边界**（本机拿不到 GPU 时间线，#41）；**PF-7 移交 `future_iterations.md` §10.2**。**原第 1 条与第 4 条都已做完**。
>
> **紧接着的 §2.2（长上下文 attention / split-K）也已同日交付并真机验证**，结论见 **§3.0i**：
> 斜率降幅 **88.85% / 88.20%**、token 与单趟一致、图版本与插件版本各 bump 一次。
> **原清单里的第 4 条到此结清**（它曾是"唯一有实测支撑、收益最大的产品代码改动"）。
> 剩下的排序**没有变化**，仍是下面 1→3；`future_iterations.md` §0.3 的第 14 项已标记为已交付。

1. **`future_iterations.md` §9.3 采样器参考数据固化**（`scripts/ref_sampler.py` 输出落成 `.bin`）—— 纯 host、无外部前置，
   是 `future_iterations.md` §9.2 的自然收尾。
2. **`future_iterations.md` §10.1 ONNX / 原生 I/O 契约统一**（ONNX 侧加 `Cast` 把 `input_ids` 降到 INT32）—— 无外部前置，
   为"ONNX 路径接进 `LLMRunner`"铺路。
3. **要碰 `future_iterations.md` §10.2（ONNX 子图替换）就先跑 PF-7**（ONNX vs 原生 prefill 的可复现对照）——它是 §10.2
   的前置，命令见开发计划 §11.9 / 测试计划 §10.2。
4. ~~**§2.2（长上下文 attention）**~~ **已交付**（2026-09-27，见 §3.0i）；若还想沿这条线继续，
   开发计划 §12.7 末行列了唯一没做的二次优化方向（block 内组织改 "warp-per-position"），
   但**必须先有测量支撑**——当前瓶颈分析与证据在 §3.0i / 开发计划 §12.3 D1。

~~`future_iterations.md` §0.1 点名 §1.5（P4-INT8-a）~~ **§1.5 已于 2026-09-27 结案**
（根因 + 离线反证 + 文件级证据 + **真机 B1 四条全绿**，见 §3.0j / 开发计划 §13）。
**它没有留下待办**：两件后续事项（把 `--weight-range-source` 默认切到 `onnx`、重生成 per-channel
产物）**按作者决定暂不做**——触发条件 = `future_iterations.md` **§1.6** 的验收集到位（当前那批
图判别力不足，得不出"谁更好"），详见开发计划 **§13.11**。
若真想再推进一件实事，仍是先补 **`future_iterations.md` §1.6 的验收集**（要联网，解锁最多），其次 §9.3 / §10.1。

**接下来做什么，取决于触发条件**（全部见 §6.6 的开放项索引与 `future_iterations.md` §11）：

- ~~需要更高 INT8 精度 → **P4-INT8-a**~~ **已结案（2026-09-27）**：根因在产图脚本的权重 scale 来源，
  改 `--weight-range-source onnx` 后 per-channel 回到 100%（§3.0j / `TROUBLESHOOTING.md` #46）；
  只剩真机 B1 四条同机确认；
- 需要 INT8 的绝对误差保证 → **P4-INT8-b**（**已立项为 `future_iterations.md` §1.6**；
  前置依赖是联网下载带真值标签的验收集，须先获批）；
- 真要迁强类型网络 → **P4-FP16-a / P4-INT8 的强类型路线**；
- 要扩展 CV（新模型/动态分辨率/图像解码）→ 见 `future_iterations.md` 对应章节，**先产出计划文档再动手**（AGENTS.md §5）。

**每轮的开工纪律**（Phase 4 全程验证过有效）：先计划对账 → 破坏性动作一次性列清单 → 真机全量回归 → 回填文档。

<details><summary>Phase 2 原始开工顺序（已完成，保留备查）</summary>

**Phase 2：GPT-2 原生构建（方案 A）**

### 6.1 [DEC-GPT2-OPS] 关键事实：GPT-2 用不上 Phase 1 的 RMSNorm / RoPE

对 `1_gpt2_onnx/gpt2.onnx` 做过算子统计：


<details><summary>展开：6.1 [DEC-GPT2-OPS] 关键事实：GPT-2 用不上 Phase 1 的  全文</summary>


<details><summary>展开：6.1 [DEC-GPT2-OPS] 关键事实：GPT-2 用不上 Phase 1 的 RM 全文</summary>

```
LayerNormalization × 25   （2/block × 12 + 最终 1 层）
Tanh × 12                 （gelu_new，tanh 近似）
MatMul × 25 / Gemm × 48 / Softmax × 12
输入 input_ids → 输出 logits
```

GPT-2 用的是 **LayerNorm + 学习式位置编码**，不含 RMSNorm、不含 RoPE。由此：

- **`RMSNormPlugin` 与 `RoPEPlugin` 在当前 Phase 2–5 的计划里没有使用者**，其价值要等到接入
  LLaMA 类模型时才兑现。这不算做错（PagedAttention 仍会用于 GPT-2 的 decode，且这三个插件是
  Phase 1 已确认的交付），但排期时需要知道。
- Phase 2 需要而 Phase 1 没提供的是 **LayerNorm**——但它在 ONNX 里是标准算子，
  TRT 10 有原生实现，**预计不需要写插件**（开工前先确认）。

> 注意：不要再用「E2 的 mini decoder 是 GPT-2 子图的缩微版」这个类比——E2 那条链
> （`RMSNorm → RoPE → PagedAttention`）是 **LLaMA 风格**的，与 GPT-2 结构不同。
> 该类比曾写进文档，已更正，见 `docs/phase1_5_development_plan.md` §0.1。

</details>

</details>

### 6.2 建议的开工顺序（按风险从高到低）

1. **多权重加载 spike（最高风险，先做）**：用 GPT-2 的真实权重（约 150 个张量、BF16/FP16 源）
   跑通 `WeightLoader → GetWeight → addConstant`。P1.5-0 修的"转换缓冲区互相覆盖"在 2 个权重时
   是 bug，在 150 个权重时是灾难——这是整个 Phase 2 最容易静默出错的地方。
2. **确认 LayerNorm / GELU(tanh) 用 TRT 原生层够用**：纯调研，成本低，避免不必要地写插件。
3. **写 `GPT2ModelBuilder`（先只做 Prefill、单形状）**，并配一条端到端用例。
4. **接 PagedAttention + Decode 引擎 + Prefill/Decode 双 profile**。

### 6.3 Phase 1.5 已扫清的前置

- dynamic shape 的 optimization profile 已打通（Phase 2 的双引擎直接依赖）；
- 多权重取用路径已有端到端验证，且修掉了会让 GPT-2 静默建错的转换缓冲区缺陷；

<details><summary>展开：6.3 Phase 1.5 已扫清的前置 全文</summary>


<details><summary>展开：6.3 Phase 1.5 已扫清的前置 全文</summary>

- 端到端骨架（模型目录 fixture / safetensors 写入 helper / 参考实现 meta-test）可直接复用。

> 开发流程提醒：Phase 1 的经验是「沙箱内 host 用例通过不代表真机没问题」——
> RoPE 的部分旋转缺陷只有真机 kernel 执行才暴露。后续每个 Phase 结束都应走一遍真机验证。

> Phase 1 / 1.5 沉淀的完整测试与验证约定见 **§2.13**（参考实现唯一性与 meta-test、
> `batch > 1` 覆盖、验证分层、失败判别方法等）。

> 本节只保留下一步入口。Phase 1 的 15 项已确认决策见 `docs/phase1_development_plan.md` §10，
> 接口约定见本文档 §2.12，测试与验证约定见 §2.13，证据与操作纪律见 §2.14，均已归档。

---

</details>
</details>

</details>

## 6.5 [DEC-WORKSPACE-STATE] 工作区与本地产物状态（新会话先看这一节）

**提交状态**：**"现在提交到哪了"一律以 `git log` / `git status` 为准，本文档不记录它**——
写过的每句"已全部提交"都会在下一次提交后立刻变成假话。

<details><summary>展开：6.5 [DEC-WORKSPACE-STATE] 工作区与本地产物状态（新会话先看这一 全文</summary>

本节只保留**提交序列的追溯**（截至 2026-09-26；新 → 旧），它不会因为新提交而变错：

1. `14844df`（"update future_iterations.md, PROGRESS.md"）——`future_iterations.md` 新增 §0 优先级重定
   （+129 行）与 `PROGRESS.md` 的对应改动。
2. `5a4017f`（"update PROGRESS.md"）——`PROGRESS.md` §6.5 提交状态段的第一次更正。
3. `4fe0b98`（"update docs"，2026-09-26）——只动文档，共 3 份：
   本节、`future_iterations.md`（新增 §1.5 / §1.6 两条立项条目 `P4-INT8-a` / `P4-INT8-b`）、
   `phase4_int8_plan.md` §7 的立项说明。**无代码改动**。
4. `61718b6`（"complate Phase 4"，2026-09-26）——Phase 4 的全部产物落盘。
5. `625939c`（"test for supplementary Phase 2"，2026-09-25）一笔记下了三件事：

   - Phase 2 的 **FP16 边界精度修复**——`src/core/llm_runner.cpp`（查询引擎声明的精度）、
     `src/kv_cache/paged_kv_cache_kernels.cu`（`WriteKVKernel` 双模板）、
     `src/core/gpt2_model_builder.cpp`（LayerNorm 显式 FP32 + 第 0 层诊断输出）；
   - **复现器与仪器**——`tests/test_gpt2_generate.cpp`、`tools/inspect_engine.cpp`；
   - 9 份文档的同步更新。

> **更正记录（2026-09-26，第二次）**：本节曾写"除下条列出的 3 份文档外全部已提交、
> 那 3 份是否提交由用户决定"——那 3 份文档已由用户提交为 `4fe0b98`，工作区随之变干净。
> **为什么必须改**：这句话与"工作区干净"直接冲突，正是本节末尾那条更正记录警告过的同一种坑
> ——照它去找未提交的 WIP，轻则白跑一趟，重则把已落盘的文档当成别人的半成品而
> `revert` / `stash`（AGENTS.md §5 第 4 条要求把这类"文档与现状矛盾"当场修掉并写明原因）。
> 教训固化（本日第二次修订，见下）：**本节只写"以 `git log` / `git status` 为准 + 追溯表"，
> 不写任何"当前提交到哪 / 是否已提交"的判断句**。
>
> **同日第三次修订**：把开头的"全部已提交、工作区干净"也改成"跑 `git status` 自己看"——
> 理由同上：本次会话又改了 `PROGRESS.md`（§6.5 / §6.6）与 `future_iterations.md`（新增 §0 优先级重定），
> 若继续在文档里写"是否已提交"，第三次修订就立刻需要第四次修订。
>
> **同日第四次修订（自检发现）**：第三次修订写了"只写不会过期的事：**提交基线** + 怎么看"——
> **这句本身就错了**：`14844df` / `5a4017f` 在会话中途落盘，被我当成"不会过期的提交基线"的
> `4fe0b98` 立刻过期。所以本节改成两个真正不过期的部分：**判据（`git log` / `git status`）
> + 提交序列追溯表**。教训：**"稳定事实"也要先问一句"它在下一次外部动作后还成立吗"**，
> 提交基线属于外部动作（用户 commit）随时会改的量，不属于稳定事实。

> **更正记录（2026-09-25）**：本节原先写"Phase 2 + Phase 3 全部产物**尚未提交**、
> `git status` 是'脏'的、这是预期状态"——那是同一笔提交落盘前的状态，提交后没有回改。
> 保留这句的害处很实在：下一个会话若照它去找"未提交的 WIP"，轻则白跑一趟，
> 重则把它当成别人的半成品而 `revert`/`stash`（AGENTS.md §5 第 4 条要的正是这种"文档与现状矛盾"的记录）。
> **判断依据以 `git log` / `git status` 为准**，不是本文档。
> 提交与 push 由用户决定（AGENTS.md §0.2）。

**不在版本控制里、但跑测试需要的产物**：

| 产物 | 位置 | 说明 |
|---|---|---|
| GPT-2 模型目录 | `models/gpt2/` | 由 `tools/convert/hf_to_mini_trt_llm.py` 生成（548 MB safetensors 被 .gitignore 忽略；`config.json` 入库） |
| 测试用引擎缓存 | `/tmp/mini_trt_llm_gpt2_*.engine` + 同名 `.fingerprint` | 首次运行自动构建（分钟级）；之后**按构建指纹复用**（日志打 `Engine cache hit`）。换代码/配置/模型会自动重建 |
| **ResNet18 本地产物（四类）** | `models/resnet18/` | ① P4-1 基线：`ref_{ramp,pixels}_b8.bin` + `inputs/*.f32.bin` + `.meta.json`（14.5 MB）；② 原生路径权重：`model.safetensors`（42 张量 / 46.7 MB）+ `config.json`（入库）；③ INT8 图：`resnet18_qdq.onnx`（**13.3 MB**，`prequant_dq` 形态）+ `.meta.json`；④ `config.json` 入库，其余 `.bin`/`.safetensors`/`.onnx` 按 `.gitignore` **不入库**。再生命令见 §7 |
| ResNet18 引擎缓存 | `/tmp/mini_trt_llm_resnet18_*.engine` + `.fingerprint` | 含 fp32（ONNX / 原生）、fp16（ONNX / 原生）、qdq_int8；**复用由构建指纹决定**（stage / 精度 / 源文件身份 / 建图参数 / 开关 / TRT·CUDA 版本 / 手工图版本） |
| 引擎 I/O 探针 | **`mini_trt_llm/tools/inspect_engine.cpp`**（已入库，手动编译） | 反序列化任意 `.engine` 并打印其 I/O 契约（名字 / 方向 / **声明精度** / 维数）——**不建 context、不推理、不写数据**。用途见 `docs/TROUBLESHOOTING.md` #18：它是把"TRT 在弱类型 FP16 网络里把 K/V 与 logits 的输出定成 FP32"这件事**读出来**的工具（在此之前只能靠推断，而推断被证伪过两次）。编译命令写在文件头（工具不进构建流程，故无 CMake 目标）。**限制**：反序列化需要 CUDA 初始化，只能在真机跑（沙箱报 `error 35`）；实测无 GPU 时它会干净报错退出而不是崩溃 |
| **性能画像报告**（**不入库**） | `/tmp/mini_trt_llm_profiles/` | `profile_*` target 的产物：`.nsys-rep`、`.app.log`、`*.kern_sum.csv`、`*.api_sum.csv`。**报告默认不入库**（作者 2026-09-27 决定）。**注意**：本机的 `.nsys-rep` **不含 GPU kernel 数据**（只有 API/OSRT/NVTX，见 #41）→ 不能用它做逐 kernel 分解 |
| **上下文扫描专用引擎** | `/tmp/mini_trt_llm_gpt2_ctxsweep_{prefill,decode}.engine` | `Gpt2DecodePerf.ContextLengthSweep` **与 `ContextLengthSweepSplitVsSinglePass`** 共用。prefill 的 `max_prefill_seq_len=992` 与主用例不同 → **必须独立路径**：引擎指纹覆盖**整个** `EngineBuilder::Config`，共用一条路径会让两个用例**交替**把对方的引擎判为过期、来回重建（分钟级）；这个坑在真机上当场暴露过（开发计划 §11.5.1） |

> **引擎缓存现状（2026-09-27，§3.0i 之后）**：`kEngineGraphVersion` 1→2 且
> `kPagedAttentionPluginVersion` 1→2，**所有旧引擎在真机上被判 `stale` 并重建过一次**
> （GPT-2 主引擎 623 / 709 MB、ctxsweep 627 / 475 MB，分钟级）——这是**预期行为**，不是故障
> （`future_iterations_development_plan.md` §12.8 第 1 条：workspace 需求由 0 变正数，复用旧引擎会往 0 字节缓冲里写）。
> 重建后指纹稳定、后续运行命中 `cache hit`。**新会话不要把这些引擎的"已重建"当成异常。**
>
> **又一次性失效（2026-09-27，`TS-048` 修复）**：`FileIdentity()` 现在先把模型路径
> `weakly_canonical` 规范化再算身份（此前"从仓库根手动跑"与"经 ctest 跑"会算出两个指纹、
> 交替重建）。**已真机验证**：手动跑一次（两个 resnet18 引擎重建，80.5 s，属预期的一次性失效）
> → 紧接着 `ctest -R ResNet18Int8AccuracyTest` **0.93 / 1.71 / 1.86 s 全命中、无 `stale`**；
> 指纹里已是绝对路径。**其它引擎（GPT-2 623/709 MB、ctxsweep 627/475 MB）会在下次被用到时各重建一次**，同属预期。
>
> **重建 ≠ 逐字节相同**：`builder.cpp` 未设 `kDETERMINISTIC`、无 timing cache → TRT 的 tactic 选择是
> timing-based。本次实测同网络重建后 `resnet18_onnx_fp32.engine` 由 **54,196,084 → 52,357,812 字节（−3.4%）**。
> **做性能对照必须用同一次构建的引擎**（与 `phase3_test_plan.md` §3.1 的"构建间噪声"同源）。

**已知会失败/跳过的测试**（避免新会话误判为回归）：

- 真机（`MINI_TRT_REQUIRE_GPU=1`）**整轮全量：2026-09-27 复跑，264 条 / 1 红 / 0 跳过 / 310 s**；
  `int8_crosscheck` 只在缺报告时按设计跳过（77）；报告齐备时（如 2026-09-27 复跑）执行并通过；
  沙箱**同为 264 条 / 0 失败**——两边**总数相同**，区别只在 GPU 用例是跑还是跳过：
  - 新增的两条 P 层用例（`Gpt2DecodePerf.StepLatencyByPhase` / `ContextLengthSweep`）在**沙箱里
    显式跳过**、只在真机跑；它们**只打印、不 assert 数值**（"凭什么通过"的答案就是它不判正确性，
    与 `Fp16PrefillOutputsDiagnostic` 同理）——别看到它们"通过"就以为验过数值。
  - `Gpt2DecodePerf.ContextLengthSweep` 首次运行会**新建两条引擎**（prefill 627 MB + decode 475 MB），
    分钟级；之后复用。
  - **§3.0i 新增的 3 条"只打印"用例同理**：`Gpt2DecodePerf.ContextLengthSweepSplitVsSinglePass`
    （PP-2，端到端 A/B，约 25 s）、`PagedAttentionSplitPerf.SlopeByContextLength`（PP-1，kernel 级
    A/B，约 0.3 s）、`PagedAttentionSplitKernelTest.MatchesSinglePassKernelDiagnostic`（PG-7，只打印
    `max_abs`/`max_rel`）。**它们"通过"不含达标信息**——判据（若作者定了）要看 stdout。
    PP-1/PP-2 都会先把 `SetPagedAttentionNumSplitsOverride` 复位，防止开关泄漏到后续用例。
  - `RealGpt2Fp16GreedyMatchesReferenceTokens`：FP16 已知限制的**按设计红**（NaN → token 不符）。
    这是**当前唯一的红**；P9_2-5~7 的 6 条新 GPU 用例（S-12 / S-13 / S-14 / S-15 / S-16 / S-21）
    与其余全部用例都通过。
  - ~~`SamplerKernelTest.TopKOnLargeVocabStaysWithinTopKSet`（S-14）第一版判据红~~
    **已修（2026-09-27）**：不是产品缺陷，是**参考在数值并列时不良定义**
    （128000 第 64 名有 4 个 token 精确并列）→ 换成并列安全的 `TopKSetByValue`，事故固化为 host 回归；
    完整推导见 `docs/TROUBLESHOOTING.md` **#36**。**复跑已绿**。
  - ~~`Gpt2OnnxTest.MatchesAcrossProfileShapes`：seq=512 逐行 argmax 不等~~ **已结案（2026-09-26）**：
    它是"两实现差异之下的并列"，判据按方案 B 修正并收紧为钉行号；单测真机复跑 PASSED（见 §5.13 / §3.0f）。
  - 146 条快照里的第二条红 `PagedKVCacheTest.AppendCrossesBlockBoundary...`（测试与
    `AppendDecodeStep` 契约不同步）已修复，见 `docs/TROUBLESHOOTING.md` #20 / 计划 P2S-6；
    （2026-09-26 那次多出的 ONNX 新红已于同日结案，2026-09-27 全量里不再出现。）
  引擎缓存现在**带构建指纹**（`*.engine.fingerprint`）：换代码/配置/模型会自动重建，不必再手工删；
  例外是"保留 mtime 地覆盖模型文件"——那时指纹看不出变化，需删引擎或 bump `kEngineGraphVersion`。
- 真机：**跑全量一定要带 `MINI_TRT_REQUIRE_GPU=1`**——否则无设备时用例会静默跳过，等于白跑。
- 真机：`Fp16PrefillOutputsDiagnostic` **应当通过**——它是**纯打印**的诊断仪器（只输出
  `max|v|` / NaN 标记，不 assert 数值），"它凭什么通过"的答案就是它不判定正确性；
  别看到"FP16 出 NaN"就以为这条也该红。
- 沙箱：全部 GPU 用例 `GTEST_SKIP`（无 GPU，见 §5.10）；`onnx_graph_probe` 在缺 `onnx` 包或
  缺 `1_gpt2_onnx/gpt2.onnx` 时返回 77 → `Skipped`（**设计如此**，缺环境 ≠ 图有问题）。

</details>

## 6.6 [DEC-OPEN-ITEMS-INDEX] 当前未解决项（开放项索引，2026-09-27）

> **事实与触发条件的唯一来源是 `docs/future_iterations.md` §11**；已**立项**的条目
> （目标 / 做法 / 验收判据 / 前置依赖）在同文件 `future_iterations.md` §1.5（P4-INT8-a）与 `future_iterations.md` §1.6（P4-INT8-b）。

<details><summary>展开：6.6 [DEC-OPEN-ITEMS-INDEX] 当前未解决项（开放项索引，2026 全文</summary>

> **优先级排序见同文件 §0**（P0~P3 与"冻结"的定义、每条的建议级别与理由）——
> 该文件的**章节顺序是主题分类、不是优先级**（2026-09-26 重定，原标签有 8 处与现状不符）。
> 本节只做**索引**，避免多处维护。**这些都不是"已知缺陷"**——已发现的缺陷一律进
> `TROUBLESHOOTING.md` 并配回归用例；开放项是"还没被盯住的地方 / 已知的限制"。

| 开放项 | 一句话 | 触发条件（什么时候做） |
|---|---|---|
| ~~**P4-INT8-a**~~ **已结案（2026-09-27）** | ~~权重 per-channel 量化在整网上比 per-tensor 差（余量子集 54.5% vs 100%），原因未知~~ → **根因 = 权重 scale 取自未折 BN 的权重、量化对象是已折 BN 的权重**（逐通道折叠系数 0.05~19.9）；改 `--weight-range-source onnx` 后 **54.5% → 100%**；真机 B1 四条全绿（含**引擎忠实**与**现象复现**两条硬证据）。见 **§3.0j** 与 `TROUBLESHOOTING.md` **#46 / #47**（含被推翻的旧结论两条） | **剩余**：待拍板 = 默认是否切到 `onnx` / 是否重生成 per-channel 产物（都不是默认行为） |
| **P4-INT8-b** | INT8 的**绝对数值界未定**（当前只用"FP32 余量子集一致率"判，阈值 ≥90%，实测 12/12；验收集无真值标签） | 需要给出 INT8 绝对误差保证时。**已立项：`future_iterations.md` §1.6**（前置 = 联网下载带标签验收集，须先获批） |
| **P4-FP16-a** | **FP16 路径仍用已废弃的 `BuilderFlag::kFP16`**（TRT 10.12 起废弃、指向 strong typing）；实测可用 | 真要迁到强类型网络时（两条 builder 的每个算子都要显式设类型） |
| **G5** | ONNX 子图识别只做计数（未做拓扑级）；~~G6 性能无可复现测量方法~~ **已于 2026-09-27 交付并随 §11 关闭**（协议 + 工具链见开发计划 §11.3 / §11.5.2） | 真要做子图替换时（`future_iterations.md` §10.2） |
| ~~**§2.2 长上下文 attention**~~ **已交付** | **长上下文下 attention 占 decode 每步 ≈80%**（PF-9 实测 3.055 / 6.062 / 14.705 ms）；机制 = 每层只有 **12 个 block / 24 个 SM**、每块串行扫完上下文（有效带宽 ≈ 峰值 3%） | **2026-09-27 交付并真机验证**：上下文维 split-K + 两阶段归约，斜率降幅 **88.85%（kernel 级）/ 88.20%（端到端）**、token 与单趟一致、`max_rel=1.5e-06`。见 **§3.0i**（含关键设计 §2.17、图版本 / 插件版本 bump、短上下文 +1.877% 观测项的决定） |
| **PF-7（= §10.2 的前置）** | ONNX vs 原生 prefill 的**跨构建对照**没跑（≥3 次构建 / ≥20 次推理）；跑之前"ONNX 更快还是更慢"仍是未定（两次测量方向相反） | 要碰 §10.2（ONNX 子图替换）时**先跑它**；命令见开发计划 §11.9 / 测试计划 §10.2。**已从 §11 移交过来**（§11 已关闭） |
| **G2-3 / G2-4** | `LLMRunner` 只支持 `batch = 1`（有意限定）／EOS 无法在循环内早停（语义正确、多算） | 需要批处理 / 需要真早停时 |
| **P1.5-b ~ P1.5-d** | E2 完整链路留后；E3 未验多 profile 切换；采样器分布数据未固化 | 见 `future_iterations.md` §11（各有触发条件）。**P1.5-a 已于 2026-09-27 关闭**（Top-K/Top-P 的 FP16 分支真机通过，见 §3.0g） |
| ~~**P9_2-5b**~~ | Top-P 收尾段并行化：**已实施**（重扫长度 197→13 / 500→32），同二进制 A/B 判定 **效果无显著差异**（中位数 +4.8/+27.3/−330.5/−81.3 µs，p25/p75 全跨 0） | **已关闭**：代码保留、`LaunchTopPSamplerTwoLevel` 留作永久对照入口（`TROUBLESHOOTING.md` #38） |
| ~~**P9_2-5c**~~ | ~~第一趟访存/MLP~~ | **不做**：greedy（本来就完全合并访存）净成本 35~155 µs，而 `top-p 净 − top-k 净` @50257×1 仅 60.1 µs → 优化空间见底（#38） |

**另有两条"已取消 / 不属于开放项"的说明**：

1. **Phase 5（清理旧模块）已永久取消**（用户 2026-09-26 决定）：`0_resnet18_onnx/`、`1_gpt2_onnx/`
   与根 `CMakeLists.txt` 的注释项**由用户自行处理**，Agent 不要删除或移动它们——
   它们还是 Phase 3/INT8 对拍与标定的**本地产物来源**（删掉会让 ONNX/INT8 用例跳过）。
2. **R0.1（`ResNet18ConfigTest.LoadsCnnConfig`）不单独落地**：config 解析断言已由
   `ResNet18WeightContractTest` 承担（见 `phase4_test_plan.md` §7），刻意不建重复用例。

</details>

## 7. [DEC-ENVIRONMENT] 重要环境信息

| 项目 | 版本 / 说明 |
|---|---|
| 操作系统 | Ubuntu on WSL2 |
| GPU | NVIDIA GeForce GTX 1660 Ti Mobile（Turing） |
| Compute Capability | `sm_75` |
| TensorRT | 10.15.1（版本宏 v101501，`libnvinfer.so.10.15.1`） |
| CUDA Toolkit | 12.6.85 |
| GCC | 13.3.0 |
| C++ 标准 | C++17 |
| CMake | 3.28.3（要求 >= 3.18） |
| SentencePiece | v0.2.0 |
| safetensors-cpp | main（commit af90b6c） |
| GoogleTest | release（源码嵌入） |
| Python 依赖 | `torch 2.5.1+cu121`（已装，`scripts/` 下的参考脚本直接可跑）；`transformers>=4.40`, `safetensors>=0.4`, `onnx>=1.16`（图结构探针用） |

**跑真机用例的前提**（不在版本控制里，需先生成）：

| 产物 | 生成方式 | 被谁需要 |
|---|---|---|
| `models/gpt2/config.json` + `model.safetensors`（548 MB） | `python3 mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py --model_name_or_path <HF gpt2 目录> --output_dir models/gpt2` | 全部 GPT-2 真机用例（config.json 入库，safetensors 被 .gitignore 忽略） |
| `1_gpt2_onnx/gpt2.onnx`（652 MB，仓库内已有） | 随仓库提供 | Phase 3 的对拍与探针 |
| `/tmp/mini_trt_llm_gpt2_*.engine` | 首次跑用例时自动构建（分钟级），之后复用 | 真机用例；**删掉它会强制重建**（测构建耗时时需要） |
| `models/resnet18/`（P4-1 基线：logits + 契约输入张量 + 元数据，共 14.5 MB） | `python3 scripts/ref_resnet18.py --input {ramp,pixels} --output models/resnet18/ref_{ramp,pixels}_b8.bin` | Phase 4 的 L2/L3 对拍；**缺了就 skip**（与 GPT-2 缺 `models/gpt2` 同口径） |
| `models/resnet18/model.safetensors`（42 张量，46.7 MB；`config.json` 入库） | `python3 mini_trt_llm/tools/convert/onnx_to_mini_trt_llm.py --onnx 0_resnet18_onnx/resnet18.onnx --output_dir models/resnet18` | 原生 builder（P4-5）的权重来源；**缺了 host 用例会 skip** |
| `models/resnet18/resnet18_qdq.onnx`（13.3 MB）+ `.meta.json` | `python3 mini_trt_llm/tools/convert/quantize_resnet18.py --onnx 0_resnet18_onnx/resnet18.onnx --calib-dir 0_resnet18_onnx/calib_data --output models/resnet18/resnet18_qdq.onnx --calib-images 500 --calib-percentile 99.9 --weight-form prequant_dq` | INT8 用例（`ResNet18Int8*`）；**缺了会 skip**。**身份 = 正式产物**（per_tensor + 默认 `torchvision` 源；默认路径逐字节可复现） |
| `models/resnet18/resnet18_qdq_per_channel.onnx` + 两份探针图 | `quantize_resnet18.py --weight-scope per_channel ...` → `add_probe_outputs.py`（见开发计划 §13.9） | **只被 `Int8ProbeTest` 的 PC 臂使用**。**身份 = `TROUBLESHOOTING.md` #46 的复现样本，不是候选基线**——它按"错源"生成（权重 scale 取自未折 BN 的权重），16.19% 的 int8 权重被 clamp 饱和。**别拿它做粒度对比、也别把 B1-4 的红当成故障**（B1-4 要的就是"PC 更差"，见开发计划 §13.11） |
| `0_resnet18_onnx/calib_data/`（500 张真实图，300 MB） | `python3 0_resnet18_onnx/prepare_calib_data.py`（需 datasets/PIL，联网下载 tiny-imagenet） | 生成 pixels 基线、INT8 校准 |
| torchvision 权重缓存 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`（46 MB） | torchvision `ResNet18_Weights.DEFAULT` 首次使用时下载 | 生成 Phase 4 基线；**已在本地缓存** |

> **注意**：上述 `models/resnet18/*.bin`、`calib_data/`、`*.onnx`、`*.safetensors` **都不入库**
> （`.gitignore` 规则：`*.bin` / `*.onnx` / `*.safetensors` / `**/calib_data/`）。
> 仓库只跟踪源码与文档；换机器要按上表重建本地产物。入库的 `.meta.json` 里，**基线（`ref_*`）含
> SHA256**（权重缓存 / 契约输入 / 归一化输入 / logits），可回答"基线有没有被改过"；
> 但 **INT8 的 `resnet18_qdq.meta.json` 目前不含任何 SHA256** → "正式 ONNX 有没有被改过"当前回答不了
> （已知缺口；补它要改转换脚本，需另行批准）。

---

*本文档用于新会话快速接手项目，不记录对话过程，只保留决策与状态。*
