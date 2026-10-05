# mini_trt_llm 项目进度交接文档

> **最后核对：2026-10-05** ｜ 本文只写**结论与指针**，不写过程：
> 条目状态的真值 = `docs/dev/<REQ>/STATE.md`（唯一快照 = `docs/dev/INDEX.md` §1）；
> 排查过程 = `docs/TROUBLESHOOTING.md`（TS-NNN）；没做的事 = `docs/future_iterations.md`；
> 文档规约 = `docs/README.md`；开工与证据纪律 = `AGENTS.md`（不复述）。
> 本文**不使用 `<details>`**：写完就压缩，折叠不是藏细节的地方。

## 0. 现状卡

- **项目一句话**：面向 Turing / `sm_75` 的极简多模态 TensorRT 推理框架，用 `mini_trt_llm` 统一承载 CV（ResNet18）与 LLM（GPT-2），替换两个历史 ONNX 示例工程。
- **现在在哪**：Phase 0 / 1 / 1.5 / 2 / 3 / 4 / 5 全部交付；**活着的条目 4 条** → §4。
  **这 4 条当前全部停在"等真机验证"**（作者 2026-10-06 决定；**本机可做的已做完**）——
  真机窗口第一动作是**一次编译全量**（不是"编一条改一条"），顺序
  `REQ-016 → REQ-018 → REQ-017 → REQ-019 的 S0/S2`；顺序与理由的唯一详述见
  `docs/dev/REQ-019-onnx-subgraph/design.md` 的「跨条目串行顺序」。
- **当前基线（唯一现状口径，2026-09-28）**：沙箱 **268 条 / 0 失败**；真机整轮 **267 条 / 1 红 / 0 跳过 / 301.72 s**。
  **待复跑**：2026-10-05 起新增 3 条用例（`docs_index_check` ×1 + REQ-018 的图 A / 图 C ×2，见
  `mini_trt_llm/tests/CMakeLists.txt` 与 `test_gpt2_generate.cpp`）→ 按 `gtest_discover_tests`
  "一条 case = 一条 ctest 条目"的口径，沙箱**应为 271**；在复跑确认前，268 仍是唯一的**实测**值。
  **再加 1 条（2026-10-06）**：`onnx_topology_selftest`（REQ-019 的 S1 反例自检，只依赖 `onnx` 包）→
  沙箱**应为 272**；同理，未复跑前不写成实测值。
  唯一红 = `RealGpt2Fp16GreedyMatchesReferenceTokens`（GPT-2 FP16 NaN，**按设计**，见 §5.11）；真机必须带 `MINI_TRT_REQUIRE_GPU=1`（否则 GPU 用例静默跳过 = 白跑），证据 = `build/Testing/Temporary/LastTest.log`。
- **精度口径**：GPT-2 用 **FP32**；ResNet18 的 FP32 / FP16 都健康；INT8 走 **Q/DQ 显式量化**，判据 = "FP32 余量子集一致率"（实测 12/12），权重默认 per_tensor。见 §5.11 / §2.15 / §3.0j。
- **环境**：TensorRT 10.15.1 ｜ CUDA 12.6.85 ｜ C++17 ｜ sm_75（无 Tensor Core）→ §7。
- **最近交付（2026-10-06）**：REQ-018 的**离线部分全部落码**（设计 v2 + 评审 BLOCK + 策略撤销与口径同步 + 探针全层 7 切点与指纹开关项 + 诊断用例自证 + 图 A / 图 C 两臂 + 逐层精度护栏 + P2 共享头），**全部未编译验证**，真机部分搁置；**前一交付（2026-10-05）**：REQ-017 路线 C、REQ-016 S1–S5 落码，同前未编译验证。
- **引擎缓存**：升级后首次使用会重建，属预期 → §6.5。
- **开放项**：REQ-016 / REQ-017 / REQ-018 / REQ-019 + SP-1 等 → §6.6；事实与触发条件在 `future_iterations.md`。
- **接手顺序**：本文 §0 → §4 → §6.6 → 命中条目读它的 `STATE.md`。

## 1. 项目目标

构建一个面向 NVIDIA Turing / `sm_75` 的极简多模态 TensorRT 推理框架，以 `mini_trt_llm` 统一替换现有 `0_resnet18_onnx` 与 `1_gpt2_onnx` 两个独立 ONNX 加载模块。

## 2. 当前架构与关键决策

> §2.12 ~ §2.18 是**契约与纪律**（"勿回改"类）：这里只留**规则句**，事故经过一律在 `TROUBLESHOOTING.md`。

### 2.1 定位：多模态推理框架，而非单一 LLM 引擎

一个框架同时承载 CV + LLM（只换 LLM 会让 ResNet18 orphaned；各做各的会重复代码）；接口预留 Encoder-Decoder / ViT / 多模态。

### 2.2 模型构建：配置驱动 + 模型注册表

`config.json` 描述结构 → `ModelRegistry` 分发到具体 `IModelBuilder`（GPT2 / ResNet18）。加模型只加 builder + 注册，不动公共代码。

### 2.3 模型来源：双模式支持

方案 A 原生构建（Safetensors + config，走 TRT Network API）；方案 B ONNX + Plugin（子图替换）。两条路的 profile 规则必须是同一套。

### 2.4 权重格式：Safetensors

不用 PyTorch `.bin`：无 pickle 反序列化风险，且支持零拷贝映射。

### 2.5 Tokenizer：SentencePiece 源码嵌入 + 多模态抽象

`BaseTokenizer` + 两个实现（`SentencePieceTokenizer` / `BpeTokenizer`）；源码嵌入避免系统包版本漂移。**注意**：SentencePiece 至今是未验证件（SP-1，见 §6.6）。

### 2.6 Plugin API：IPluginV3（TRT 10.x）

按 `IPluginV3OneCore` / `OneBuild` / `OneRuntime` 拆分能力，不用 legacy `IPluginV2DynamicExt`；当前 TensorRT 10.15.1 支持 V3。

### 2.7 日志：独立业务日志宏

`MINI_TRT_LOG_*` 不复用 TRT `ILogger`：业务日志不应绑定 TRT 回调接口。

### 2.8 测试框架：GoogleTest 源码嵌入

系统无 `libgtest-dev` 且网络受限 → 源码嵌入最稳。

### 2.9 构建选项

`BUILD_TESTS` / `BUILD_EXAMPLES` 默认 OFF（日常编译快），CI 显式打开。

### 2.10 INT8 与性能优化延后

Phase 0 只做 `cudaMalloc` / `cudaFree` 的 RAII 封装；内存池延后（触发条件见 `future_iterations.md` §2.1，实测触发不成立）。

### 2.11 代码注释规范

启用 `cpp-comment-style` skill：注释写 Why 不写 What，必须覆盖魔数 / workaround / 非直观逻辑 / 公共 API。

### 2.12 [DEC-IFACE-PHASE1] Phase 1 算子层的接口约定（实现时确立，勿回改）

前三条与 `docs/dev/REQ-002-plugins/phase1_development_plan.md` 的早期表述**不一致**，是经作者确认的有意偏离，不要按 plan 原文"修正"回去：

1. **形状优先、属性兜底**：`configurePlugin` 以输入形状为权威；只有动态轴（`<= 0`）才回退属性 → 校验属性的用例必须构造动态轴。
2. **RoPE 用 half-split**（`out[j] = x[j]cos − x[j+h]sin`），不是"相邻两维配对"；已与 HuggingFace `apply_rotary_pos_emb` 逐位对齐（`scripts/ref_rope.py`）。
3. **RoPE 不序列化 head 配置**：只序列化 `rotary_dim` / `base`，`num_heads` / `num_kv_heads` / `head_size` 全从形状推导（避免第二份可能与形状失配的状态）。
4. **PagedAttention 没有 `max_context_len` 输入**：online softmax 按 per-batch `context_lens[b]` 循环，全局上界用不到 → 保留就是死参数。
5. **采样器只有设备侧 API**：`Launch*Sampler` 直接写回 device，无 host `std::vector` 出口；实现在 `src/sampler/sampler_kernels.cu`（采样器不是 Plugin）。

另有一条 kernel 纪律（`TROUBLESHOOTING.md` #4）：**输出与输入分离的 kernel，只要存在"部分写入"路径，就必须显式处理未覆盖区间**。

### 2.13 [DEC-TEST-CONVENTIONS] 测试与验证约定（Phase 1 / 1.5 沉淀，后续沿用）

- **参考必须自带断言 + host meta-test**，且参考实现全局唯一（标尺错了会给出错误裁决，见 #9）。
- **带 batch 维的算子必须覆盖 `batch > 1`**：Phase 1.5 的两个缺陷在 `batch = 1` 时都表现为通过。
- 验证分层：host（含参考 meta-test）进沙箱 / CI；GPU 用例在真机跑。**沙箱 host 全绿 ≠ 真机没问题**。
- 每个 Phase 结束**必须走一遍真机验证**，不留给下个阶段。
- 失败时的判别法：先看"错误从哪个维度边界开始"，再用"某个配置能过"反推排除哪些路径——都比逐行读代码快。
- `pipeline` 组件要区分构建期 / 运行期：凡"从形状推导"的状态，`onShapeChange` 必须能自行推导（#8）。
- GPU 用例的跳过必须显式：统一走 `ProbeCudaDevice()`、跳过信息里打印探测结果、`MINI_TRT_REQUIRE_GPU=1` 时**跳过即失败**；"无设备"才允许跳过，"有设备但建 builder 失败"是故障。实现必须用 `MINI_TRT_SKIP_IF_NO_CUDA` 宏（`GTEST_SKIP` 封装成函数只会 return）。
- 精度比较的用例**必须显式写出并打印目标精度**：`Config::precision` 默认 FP16，而弱类型引擎的 I/O 常被 TRT 定成 FP32；第一版 ResNet18 对拍就因此量到 0.033（真值 9.5e-6，见 #21）。
- 借别的模型的产物当负例夹具 = 把两边契约耦合：必须在注释里点明依赖，并保证真机全量跑得到它（#22.1）。
- **改完接口 / 契约必须跑真机全量**：一次抓到"旧夹具失效 + 单跑通过、全量 SEGFAULT"两条（#22）。
- **护栏必须有用例证明它会拦人**：每条拒绝逻辑都要有一份故意改坏的输入（G1c 的教训）。
- **新增源文件后必须重新 configure**（`file(GLOB)` 只在 configure 时求值；第二次踩，见 #24.2）。
- **TRT 的 element-wise 要求两侧 rank 相同**（不是 NumPy 广播）：偏置常量要写 `[1, N]`；写错时 `Build()` 仍返回 true，错误拖到引擎构建期（#24.1）。
- 测试里的路径 helper 必须返回它名字声称的东西（返回文件却叫 `FindXxxDir()` → 用例静默跳过；**静默跳过比失败更贵**，见 #25）。
- 文档只记用例名、不记 ctest 序号（序号会随用例增加漂移，#34）。
- 跨实现比较的"精确判据"必须带**可判性**前提：argmax 判等只作用在可判行（`m > 2d`），不可判行必须钉到行号（`ExpectedUndecidableRows()`），**新增行即红**（#34）。

### 2.14 [DEC-EVIDENCE-DISCIPLINE] 证据纪律与操作纪律（Phase 2 沉淀，后续沿用）

**A. 禁止用"放宽期望值"换取通过** —— 实测与期望不符时只允许三种动作：继续查（写明"原因未知"）／证明**期望值本身**错（必须给独立依据并写进文档）／标成"已知失败 + 原因未知"保持红。禁止调阈值、删断言、把断言降级成打印、跳过用例（#15 就是放宽阈值掩盖问题的实例）。
配套：**阈值旁边必须写出处**，且阈值不跨精度复用（FP16 的 `1e-3` 套到 FP32 上等于把尺子放宽千倍）；放宽前先量"与正确性无关的差异"，观测值高出几个数量级就只能去查。

**B. 操作纪律**：未经作者点名不得改产品代码 / 文档结论；"批准目标 ≠ 批准手段"（细则见 `AGENTS.md` §0）。

**C. 诊断代码也必须自证**：D2H 读回的范围必须落在同一段分配内，且写清比较对象（比了哪两个张量、各自的布局与形状）——否则会输出"看起来像真故障"的数字（#15 / `AGENTS.md` §7）。

**C-补（2026-10-06，来自 REQ-018 的复评）**：同族两条仪器教训（详情 `TROUBLESHOOTING.md` + 18.2，此处只留指针）：① **仪器换一次，读数就不可比**——探针输出会改变 TRT 的融合与 tactic（ResNet18 探针上实测过）；② **"位置不变"不等于否证**。推论：每轮改图的排查等于每轮换仪器，要的是"同一次构建内读数可比"。

### 2.15 [DEC-IFACE-PHASE23] Phase 2 / 3 确立的接口约定（实现时确立，勿回改）

与 §2.12 同性质：都是"与直觉写法相反"的约定，不要按"更自然"的写法改回去。

| 约定 | 为什么（写错会怎样） |
|---|---|
| `IModelBuilder::Build` 带 `BuildOptions{stage, weight_dtype}` | builder 必须显式知道精度与"建哪个切面"（`kSingle` = 单引擎双 profile；`kPrefill` / `kDecode` 各一组） |
| decode 的 cache 输入是**每层一对** `key_cache_<layer>` / `value_cache_<layer>` | 共用一张张量 → 每层都读第 0 段 cache（2 层差 `1.2e-3`，12 层完全失真；#15） |
| `AppendDecodeKV`（只写、不推进长度）+ `AppendDecodeStep`（一次写全层、只推进一次长度） | 长度是 per-token 量，挂在"每层一次"的接口上会被推进 `n_layer` 倍（第 3 个新 token 起发散；#16） |
| `BuildFromOnnx(model_dir, onnx_path, engine_path, subgraph_names)` 的 profile 规则与方案 A **同一套**；`subgraph_names` 写错即失败；图必须有 `input_ids` / `logits` | 两条路要能"对齐"才有意义 |
| ONNX 的 I/O 与原生**不同**（`input_ids` 是 INT64、无 `position_ids`） | 调用方必须按**引擎声明的**契约准备输入（统一见 `future_iterations.md` §10.1） |
| `LLMRunner::Generate` 返回**新生成**的 token（不含 prompt）；失败返回空 vector；`temperature != 1.0` 显式失败；循环内零 H2D/D2H | 失败与"生成 0 个"必须可区分 |
| 缓冲按**引擎声明的**边界精度分配（`prefill_kv_half_` 等），**不按 `Config::is_half`** | 弱类型网络下 K/V 与 logits 的类型由 TRT 决定（FP16 引擎实测为 FP32）；按 config 假定宽度 = 越界写 → 非法访存，且 FP32 下永不暴露（#18） |
| `PagedKVCache::Config::source_is_half`（源 = 引擎导出的 K/V）与 `is_half`（目标 cache 布局）是**两个独立**字段 | `WriteKVKernel` 因此是双模板 `<SrcT, DstT>`、按源×目标四种组合分发；假定一致则 FP32 源写进 FP16 cache 时宽度与数值全错 |
| 启动时校验 decode `key_cache_0` 的声明精度 == `Config::is_half`，不一致**拒绝构造** | 把"内存越界"变成一条可读启动错误；无论将来走强类型还是消费方适配都要保留 |
| 诊断输出（`mlp_fc_0` 等中途张量）必须由 `BuildOptions::export_diagnostics` **显式打开、默认关** | `markOutput` = 改 I/O 契约；默认带上曾在真机打挂 6 条用例（#19）。新增诊断输出前先 grep 全部绑定方与输出计数断言 |
| 引擎缓存路径不得在"两种 I/O 契约"之间共用（如诊断开 / 关） | 两套契约是两张图；共用会让每次切换都触发重建（分钟级）→ 诊断仪器单用 `..._diag.engine` |
| ONNX 的 I/O 契约**按 `architecture` 选**（`cnn` → `input` / `output`；其余 → `input_ids` / `logits`），且该映射必须能被 host 用例直接测到 | 契约映射内联在 `BuildFromOnnx` 里 → 只在真机才验证得到 |
| CV 的 profile `opt_batch = 8`（历史工程口径），**不要改回 1** | TRT 针对 kOPT 形状挑最快 kernel；改回会让 batch ≥ 2 走非最优 kernel、性能结论失真 |
| `CVRunner` 输入契约 = **float32 / NCHW / `[0,255]` 像素质**，归一化由 Runner 自己做；失败返回空 vector / 零值统计并打日志 | 与 P4-1 的契约输入一致（可直接对拍）；HWC→CHW 留给调用方 |
| `CVRunner` 的维度与 batch 范围**必须向引擎查询**（输入 `getProfileShape`、输出 `getTensorShape`） | 写死 224/1000/16 = 把"模型是什么"焊进 Runner；两个 API 分工不同（对输出用 `getProfileShape` 会返回 `Dims{-1,{}}`，#23.2） |
| `CVRunner` 的前处理入参是 **`pixels_per_channel`（= H*W）显式传入**，不从总长度反推 | 用"总元素数 / C"当分母在 `batch = 1` 恰好等价、batch > 1 才错（#23.1）→ 凡按 NCHW 拆下标都要覆盖 `batch > 1` |
| **INT8 走 Q/DQ 显式量化**：`SetupBuilder` 不设任何 INT8 flag；Q/DQ 必须**对称**（zero_point 恒为 0） | `kINT8` 自 TRT 10.12 废弃；非对称图报 `Non-zero zero point is not supported`（#27）；QDQ 图不受我们传的 precision 影响（层信息才是证据） |
| **`BpeTokenizer::Load` 的入参是"目录"**，不是单个文件 | byte-level BPE 要 `vocab.json` + `merges.txt` **两个**文件，而基类只有一个路径参数 |
| **预切分里的 ` ?` 只吃字面空格 U+0020**（tab / 换行 / 不换行空格都不算） | 写成"任何空白都能当前导空格"会把 `"a\tb"` 切成 `["a","\tb"]`（HF 是 `["a","\t","b"]`）；同类错误还踩过一次（#33.2 / #33.3） |
| **引擎缓存必须带构建指纹**（`<engine>.fingerprint`），缺指纹一律重建；**图代码变了必须手工 bump `kEngineGraphVersion`** | 缓存过去只按路径名复用、不随代码 / 配置失效（#34 的牵连因素）；测试里的 `exists()` 存在性门全部取消——复用与否只能由 builder 决定（#34.10） |
| **INT8 判据 = "FP32 有余量子集的一致率"（≥90%）**，整体一致率只作"没崩坏"下界；用 `IEngineInspector`（需 `Config::detailed_profiling = true`）判 `Format/Datatype: Int8`（不是 `[I8]` 标签） | 这批图 FP32 自身摇摆（55% 样本 margin < 2）；不设 kDETAILED 读不出逐层精度（#29.4 / #30.5） |

### 2.16 协作与权限规则（2026-09-27 收紧）

流程权威 = `AGENTS.md` §5 + 技能 `trt-inference-engineering`；条目状态落点 = `docs/dev/<REQ>/STATE.md`。改 `AGENTS.md` 要口令"授权修改AGENTS.md一次"；**其他任何工作**（写文档 / 改代码 / 跑实验 / 真机）必须由作者**点名到具体条目**——笼统的"开始吧 / 继续 / 看着办"不算批准，批准范围不外扩。技能内的基准与测量阶段（P4 / P7）已预授权，其余单测外测试仍须先确认。

### 2.17 §2.2 确立的接口约定（同 §2.12 / §2.15 的性质：**勿按"更自然"的写法改回去**）

| 约定 | 为什么（写错会怎样） |
|---|---|
| 片数由设备端按各自 `context_lens[b]` 推导；stage-1 的 grid z 恒为上限、超出的 block 立即返回 | 宿主侧拿不到当前上下文长度 → 写成"宿主判定"会让 split 永远走分支另一侧 |
| `getWorkspaceSize()` 按 `desc.max` 报上界，不是 `desc.dims` | 动态轴在 `desc.dims` 里是 −1 → 算小了会越界写（只在真机暴露） |
| 分片切法与 workspace 布局只有一份实现（`paged_attention_split.hpp`） | 两边各写一份 → 改片数 / head_size 后静默错位 |
| `SetPagedAttentionNumSplitsOverride`：`>0` 强制片数、`0` 自适应、`<0` 强制旧单趟 | `<0` 是同二进制 A/B 开关；把负数钳成 0 会让 A/B 静默失效 |
| 旧单趟 kernel 永久保留 | A/B 只能靠 git；workspace 拿不到时没有"正确但慢"的退路 |
| `kPagedAttentionPluginVersion` 与 `kEngineGraphVersion` 同批 bump | workspace 需求由 0 变正数，复用旧引擎 = 往 0 字节缓冲里写 |
| 短上下文多一次归并是已知代价（实测 +1.877%，观测项） | 它来自"宿主判定不可行"，不是顺手能优化的东西 |

### 2.18 `future_iterations.md` §1.5 确立的仪器 / 产物纪律（2026-09-27；**下个会话按这个来，不要回退**）

| 纪律 | 为什么（违反时会怎样） |
|---|---|
| 量化类转换：**scale 必须取自"被量化那张张量"本身** | "尺子量 A、裁剪 B" 是本项目最贵的教训：per-channel 16.19% 权重被 clamp、整网余量子集 54.5%（#46） |
| 标尺必须独立于被测实现，且自己先被校准 | 标尺 = ONNX 官方参考实现（`tools/validate/qdq_reference.py`，带 `--self-test`），不是"自己折 BN 的 torch 模型" |
| 探针图必须可证"= 产物图 + 探针" | `add_probe_outputs.py` 只追加 `graph.output`，并断言 node / initializer / input / opset 逐字节不变 |
| 探针要探"量化前"的 float 张量，不探量化后 | 量化台阶会把正常差异放大成 ±1 格噪声；自证用 `d_pre ≤ d_post`（真机 8.34e-07 vs 0.0398） |
| 产物身份必须钉死：正式产物 vs 复现样本 | `resnet18_qdq.onnx` = 正式产物；`resnet18_qdq_per_channel.onnx` 及其探针图 = #46 的**复现样本、不是候选基线** |
| 能离线验的别上真机 | §1.5 整条在 CPU 上 4 分钟跑完；上一轮同类排查花了 3 次真机往返（#30.5） |
| 形状只能问 `IExecutionContext`，不能问 `ICudaEngine` | 引擎上动态维是 −1，转 `size_t` 就成天文数字 → 报错伪装成"显存分配失败"（#47.1） |

## 3. 已完成的部分

> §3.0a ~ §3.0k 是**历史交付的索引**：每条只留结论与指针，明细在各条目的 `summary.md`（REQ-001~015 全有）。

### 3.0a [DEC-PHASE2-DELIVERY] Phase 2 交付（GPT-2 原生构建，2026-09-25）

GPT-2 原生路径打通：双引擎（prefill / decode）+ 自回归循环 + 逐层 K/V。→ `docs/dev/REQ-004-gpt2-native/summary.md`

### 3.0b [DEC-PHASE3-DELIVERY] Phase 3 交付（GPT-2 ONNX 路径，2026-09-25）

ONNX + Plugin 路径打通（prefill 推理与对拍；I/O 契约与原生不同）。→ `docs/dev/REQ-006-gpt2-onnx/summary.md`

### 3.0c [DEC-PHASE2-PATCH] Phase 2 补丁（诊断输出开关 + CUDA 环境判定，2026-09-25）

诊断输出改为"默认关的显式开关"；无 GPU 环境显式判定。→ `docs/dev/REQ-005-diagnostics-fix/summary.md`

### 3.0d [DEC-PHASE4-DELIVERY] Phase 4 交付（ResNet18 / CV 路径，2026-09-26）

CV 路径打通：ONNX + 原生两条 builder、`CVRunner`、FP16、INT8（Q/DQ）。→ `docs/dev/REQ-007-resnet18/summary.md`、`REQ-008-int8-qdq/summary.md`

### 3.0e [DEC-BATCH-A-DELIVERY] future_iterations 批次 A 交付（2026-09-26）

BPE Tokenizer（文本 ↔ token）+ INT8 判据的离线口径（`int8_eval.py`：3 项分层数学 + 7 道护栏）。→ `REQ-010-bpe-tokenizer/summary.md`、`REQ-011-int8-criteria/summary.md`

### 3.0f [DEC-ENGINE-FINGERPRINT] 判据修正 + 引擎缓存指纹（2026-09-26，承接真机新红 #34）

argmax 判据改为"可判行必须全等 + 不可判行钉行号"；引擎缓存加构建指纹（`<engine>.fingerprint`）。→ `TROUBLESHOOTING.md` #34

### 3.0g [DEC-SAMPLER-KERNEL] `future_iterations.md` §9.2 采样器高性能 kernel（P9_2-0 ~ P9_2-5b，2026-09-26 ~ 27，**已关闭**）

Top-P 改"保留 CUB 排序 + 行内并行"（配对净收益 12.6 / 22.4 / 15.7 / 17.6×）；Top-K 快速路径性能不达标、已撤出生产；P9_2-5b 判"无显著差异"、5c 不做。→ `REQ-012-sampler-kernel/summary.md`

### 3.0h [DEC-PERF-PROFILE] decode 性能画像基建（2026-09-27，**已出首份数据；kernel 时间线待宿主机**）

profile target ×4 + 一键脚本 + 分桶脚本；PF-8 采样器占比（greedy 1.25% / top-k 17.7% / top-p 20.1%）、PF-9 attention 占比（长上下文 ≈ 每步 80%）、PF-5 显存分配（触发不成立）。**能力边界**：本机 WSL2 拿不到 GPU kernel 时间线（#41）。→ `REQ-013-perf-profile/summary.md`

### 3.0i [DEC-FLASHDECODING] §2.2 长上下文 attention 交付（FlashDecoding 式 split-K，2026-09-27）

斜率降幅 88.85%（PP-1 kernel 级）/ 88.20%（PP-2 端到端，两次复现 88.12 / 88.20）；三档 prompt 的 token 与单趟完全一致（`max_rel 1.5e-06`）。图版本与 `kPagedAttentionPluginVersion` 各 bump 一次（旧引擎失效一次，属预期）。→ `REQ-014-attention-splitk/summary.md`

### 3.0j [DEC-INT8-WEIGHT-SOURCE] P4-INT8-a 结案：per-channel 整网退化的根因（2026-09-27）

根因在**产图脚本**：权重 scale 取自未折 BN 的 torchvision 权重，而 Q/DQ 插在已折 BN 的 ONNX 权重上（折叠系数 0.05~19.9）→ 16.19% 权重被 clamp。改 `--weight-range-source onnx` 后 per-channel 余量子集 **54.5% → 100%**；真机 B1 四条全绿（引擎忠实 `max_abs 0.2714`；探针复现 PT 12/12 vs PC 6/12）。**默认行为与正式产物逐字节未变**。→ `REQ-015-int8-perchannel/summary.md`、`TROUBLESHOOTING.md` #46 / #47

### 3.0k [DEC-REQ-NUMBERING] 需求归档与统一编号（2026-10-01）

历史文档的需求抽成 `docs/dev/REQ-NNN-<slug>/`，共 **19 条**（9 个阶段 + 6 个已关闭批次 + 4 个进行中）；16 份阶段文档迁入（文件名不变、PH 锚点继续有效，全仓 240 处引用同步改写）。索引 = `docs/dev/INDEX.md`；编号只承载身份、不承载状态。→ `docs/README.md` §4

### 3.1 目录与构建

`mini_trt_llm/` 下分 `include/` `src/` `tests/` `tools/` `third_party/`；CMake：C++17 + CUDA C++17、`sm_75`、static library。

### 3.2 Utils 基础设施

`cuda_check` / `cuda_dtype` / `cuda_reduce` / `io` / `json` / `logger` / `memory_pool` / `safetensors_loader` / `timer` 全部就位并有单测。

### 3.3 Core 通用化骨架

`Engine` / `EngineBuilder` / `EngineCache` / `IModelBuilder` / `ModelRegistry` / `ModelConfig` / `WeightLoader` / `Precision`，以及两个具体 builder（GPT2 / ResNet18）。

### 3.4 其他模块占位

`kv_cache/` / `plugins/` / `sampler/` / `tokenizer/` 都已填充（不再是占位）。

### 3.5 [DEC-TEST-INVENTORY] 测试

单一目标 `mini_trt_llm_tests`（GoogleTest 源码嵌入）；host 用例沙箱全绿、GPU 用例真机跑；另有一批 Python 自检项（`*_selftest`）注册进 ctest。当前条数基线见 §0。

### 3.6 工具与文档

转换：`tools/convert/{hf_to_mini_trt_llm, onnx_to_mini_trt_llm, quantize_resnet18, quantize_gpt2, add_probe_outputs}.py`；校验：`tools/validate/{int8_eval, qdq_reference, crosscheck_reports}.py`；工具：`tools/inspect_engine.cpp`、`inspect_onnx.py`、`make_tiny_onnx.py`、`profile/*`、`check_skips.py`、`check_docs_index.py`。

### 3.7 第三方依赖

`sentencepiece` v0.2.0、`safetensors-cpp`（main，commit af90b6c）、GoogleTest——全部**源码嵌入**，无系统包依赖。

### 3.8 注释规范修复

Phase 0 代码按 `cpp-comment-style` 过了一遍：去掉复述型注释，公共 API 补齐类级 / 函数级说明。

### 3.9 [DEC-PHASE1-DELIVERY] Phase 1 插件与采样器

RMSNorm / RoPE / PagedAttention 三个 IPluginV3 插件 + 采样器（greedy / top-k / top-p）交付，全部带单测与 L2 集成用例。→ `REQ-002-plugins/summary.md`

### 3.10 [DEC-PHASE15-DELIVERY] Phase 1.5：全流程测试基建与收尾

端到端测试基建（E1~E4）、动态 shape、错误路径、参考实现与 meta-test。→ `REQ-003-test-infra/summary.md`

## 4. 进行中 / 未完成的部分

> 本节只写**人话 + 链接**。阶段值的真值 = `docs/dev/<REQ>/STATE.md`，唯一快照 = `docs/dev/INDEX.md` §1
> （一致性由 `mini_trt_llm/tools/check_docs_index.py` 校验）。历史阶段（§4.1~§4.6）只留一行区间。

### 4.1 Phase 1：Plugin 基础（已完成）

→ `REQ-002-plugins/summary.md`

### 4.2 Phase 1.5：全流程测试基建与收尾（已完成）

→ `REQ-003-test-infra/summary.md`

### 4.3 Phase 2：GPT-2 原生构建（已完成，见 §3.0a）

→ `REQ-004-gpt2-native/summary.md`

### 4.4 Phase 3：GPT-2 ONNX + Plugin（已完成，见 §3.0b）

→ `REQ-006-gpt2-onnx/summary.md`

### 4.5 [DEC-PHASE4-STATUS] Phase 4：ResNet18 替换（✅ 已完成，2026-09-26）

→ `REQ-007-resnet18/summary.md`、`REQ-008-int8-qdq/summary.md`

### 4.6 [DEC-PHASE5-REOPENED] Phase 5：清理旧模块（🔄 2026-09-26 取消 → 2026-09-27 重新立项）

**已完成（2026-09-28）**：两个历史目录删除、资产迁到 `assets/legacy/`、真机验收与文档回填完成。执行记录 → `REQ-009-retire-legacy/phase5_development_plan.md`（"永久取消"的旧决定已加日期批注修正）。

### 4.7 [DEC-REQ017-P5-STATUS] REQ-017（LLM weight-only INT8）的当前状态与未完成项（2026-10-05 更新）

**现在在哪**：已经进到"测试"这一步（作者已点名测试阶段）——用例与判据已落（17 条 + 追溯 9 行），**结果一律记"未验证"**。
**"实现"这步的出口还没收**：编译通过 / 无新增 warning、`addDequantize` 的签名与广播约束、D7（DQ 是否被吸收 → **引擎体积必须下降**）、`wte` 新路径在 FP32 / FP16 下的表现——全部等真机。
**能在这里做完的都已做完**：Python 自检（6 道护栏）+ 三项机械核对；其余一律"未验证"，不得当成通过（`AGENTS.md` §7）。
阶段值与逐条裁决 → `docs/dev/REQ-017-llm-int8-quant/STATE.md`。

**未完成项表（行数 = 5；`docs/dev/REQ-017-llm-int8-quant/test_plan.md` §5 的"覆盖对照"以本表为唯一来源）**：

| # | 未完成项 | 原因 | 解除条件 |
|---|---|---|---|
| 1 | "编译通过 / 无新增 warning"（P5 Exit Gate） | 本机没有编译器（无 nvcc / cmake / TensorRT） | 真机窗口第一步 |
| 2 | `addDequantize` 的签名与 scale / zeroPoint 广播约束核对 | 本机没有 `NvInfer.h`，代码按作者"假设有头文件"的指令写 | 真机（读头文件 + 建最小图） |
| 3 | **D7 判据**：DQ 是否被吸收（**引擎体积必须下降**） | 需要真机建引擎 | 真机最小图实验 |
| 4 | `wte` 新路径在 FP32 / FP16 下的确认 | 同上 | 真机 |
| 5 | P6 的用例与 `test_plan.md` | **已落档**（2026-10-05 作者点名 P6；四类 17 条 + 追溯 9 行） | 真机按清单逐条打勾后回填 |

### 4.8 [DEC-REQ018-STATUS] REQ-018（GPT-2 FP16 端到端 NaN，bugfix）

**现状**：**离线部分已全部落码、未编译验证；真机部分搁置**（以它的 `STATE.md` 为准）。作者 2026-10-06 裁决两件事：① **撤销"按政策不修"**——依据是**正确性**（默认精度不可用即产品缺陷），不是 §5.11 / `future_iterations.md` §1.4 的"低精度**性能**"触发（旧的收益结论仍成立）；② **当前设备无真机条件 → 真机部分整体搁置**，本轮不产出任何实测读数。
方案与评审：`analysis.md` 的 `## Candidate Fixes（B2）` = 重设计方案 v2（一次真机三图对照 + 定点一处显式精度；原 `design.md` 已删）；`review.md` 的 **Decision = BLOCK**——4 条 P0、3 条同轮关闭，P0-3（Softmax 是否有精度接口）需真机读 `NvInfer.h`（离线四条路 + 联网取 wheel 已全部试过并排除，留痕在 `STATE.md`）。已落码均**未编译验证**：探针全层 7 切点 + 指纹开关项、诊断用例自证（按声明 dtype 填、未识别名字即失败、RMS / 首个非有限下标、逐层落盘）、图 A / 图 C 两臂、逐层精度启动护栏、P2 共享头。
它是真机上**唯一按设计的红**，别当成新回归（见 §5.11）。

### 4.9 [DEC-REQ019-STATUS] REQ-019（外部图子图替换）

**现状（2026-10-06 重设计后）**：按**乙框架**重做 P2——**性能门只挡"子图替换"，不挡契约统一 / 拓扑识别 / 自定义算子进图**。`design.md` 已重写、`analysis.md` 补了 `## Terminology`，状态回到 P3-Review（`status = in-progress`，以它的 `STATE.md` 为准）。
**为什么改**：`requirement.md` 的 Goal / Excluded 只约束"替换"，上一版把 PF-7 提升成整条 feature 的开门条件属放大；当前设备非目标真机（GTX 960），后果是三项与替换收益无因果关系的工作无限期停摆。
**S3 待定**：作者 2026-10-06 接受"S3 长期标待定（PF-7 待真机）"；PF-7 口径见 §6.6 与 `future_iterations_development_plan.md` §11.9。**PF-7 需要一段新代码**——`OnnxVsNative.PerfPerBuildMedian` 在代码里不存在，现有测量是 5 次取平均、不出极差（测试计划里"无新代码"的表述已于 2026-10-06 更正）；该薄用例尚未获授权。
**P3 复评结论（2026-10-06）= BLOCK**：`review.md` 逐条回答四份 checklists 的 35 条 P0 + 17 条 P1，问题集中在 S2——① `createParser` 带 `IPluginRegistry` 的重载**未核实**（本机无 TRT 头）；② 四个 creator 的 `getPluginNamespace()` 返回**空串**，而 ONNX 自定义域必须非空（已核实的代码事实，修法有连带重建的影响面待裁决）。**S0 / S1 的设计不需返工**，只欠两处落码前置探针（真实图的形态）。
**同日作者裁决**：① **暂时不真机** → ① 那条保持未闭环（`review.md` 的 P0-1），S2 不落码；② **接受**把 creator namespace 设为 `mini_trt_llm` 及可能的既有引擎连带重建（P0-2 闭合）；③ S0 的识别基线归属取 **B**（`--check` 只对源图生效）。当前唯一放行障碍 = P0-1。
**同日落码（第二批放行后）**：**S1 已完成并在沙箱跑绿**——`inspect_onnx.py --check-topology`（T1 块边界不共享 / T2 块内 Q-K-V 同源 / T3 输出经投影回主线 / T4 位置编码为学习式查表）+ 夹具 `make_topology_fixture.py`（`good` / `swap-softmax-inputs` / `share-score`，三者**算子计数逐个相等**）+ ctest 条目 `onnx_topology_selftest`（**沙箱条目数 +1**）。**PF-7 用例 `OnnxVsNative.PerfPerBuildMedian` 已落码、未编译**（每边 3 次独立构建 × 20 次推理，纯打印）。S1 唯一没闭的是**真实图的串联形态**（需 652MB 资产，T7）。
**同日 T12 裁决（跨条目串行顺序）**：**REQ-016 → REQ-018 → REQ-017 → REQ-019 的 S0/S2**；"收口"定义为**编译通过 + 沙箱用例全绿**（真机部分继续记搁置）；四条共用的文件在编译验证前**写冻结**。详述（唯一）见 `docs/dev/REQ-019-onnx-subgraph/design.md` 的「跨条目串行顺序」，REQ-016 / 017 / 018 各自的 `STATE.md` 留指针。
**本条当前处置（作者 2026-10-06）**：**整体搁置，等真机环境就绪**——本机可做的已做完（S1 落码 + 沙箱自检通过；PF-7 用例落码未编译），S0 / S2 受 T12 写冻结。真机窗口按 `STATE.md` 的「真机窗口执行清单」走，**第一件是读 `NvOnnxParser.h` 清 P0-1**。

## 5. 已知问题与坑

> 每条只写：问题 → 影响 → workaround。过程、命令与日志一律在 `docs/TROUBLESHOOTING.md` 的对应 TS 条目。

### 5.0 [DEC-EOS-EARLY-STOP] `LLMRunner` 无法在解码循环内早停 EOS（有意为之的 workaround）

问题：`AGENTS.md` §3.A.3 禁止循环内 H2D/D2H，而"见 EOS 就停"必须先知道设备上的 token 值。影响：多算几步，语义仍正确。workaround：不早停；要真早停需设备侧 stop flag + 条件图（G2-4）。

### 5.1 自研 JSON 解析器能力有限

问题：`utils/json.hpp` 只支持基础类型与简单嵌套。影响：复杂配置可能解析失败。workaround：配置控制在支持范围内；要更复杂的解析就换库（需批准）。→ `TROUBLESHOOTING.md` #33.1（`vocab.json` 读不进来就是这类）。

### 5.2 `SafetensorsLoader::GetTensorNames()` 返回空

问题：上游 `safetensors-cpp` 的 `ordered_dict` 不暴露 key 遍历接口。影响：无法枚举张量名。workaround：按已知名单逐个 `GetTensor`。→ #5。

### 5.3 SentencePiece 与 GPT-2 BPE 可能不对齐

问题：GPT-2 原生是 byte-level BPE，SentencePiece 行为可能不同。影响：tokenizer 结果可能与 `transformers` 不完全一致。workaround：GPT-2 用 `BpeTokenizer`；SentencePiece 至今未验证（SP-1，见 §6.6）。→ #33（BPE 实现的三个坑）。

### 5.4 BF16 转换（已修复）

原问题：BF16→FP16 直接截断尾数（忽略了 BF16 与 FP16 指数位宽 8 vs 5 的差异），位截断在数值上不成立。现状：已按 round-nearest 修复。→ #5。

### 5.5 沙箱环境无法访问 GPU

问题：沙箱 `nvidia-smi` 报 `GPU access blocked by the operating system`，CUDA 无法初始化。影响：Agent 侧结论上限 = 编译通过 + 契约自洽 + host 逻辑正确。workaround：GPU 用例统一 `MINI_TRT_SKIP_IF_NO_CUDA` 跳过；真机带 `MINI_TRT_REQUIRE_GPU=1` 时跳过即失败。

### 5.6 `.gitmodules` 曾出现重复条目

问题：根目录与 `mini_trt_llm/third_party/` 下曾同时存在 submodule 条目。现状：已清理，只保留 `sentencepiece` 与 `safetensors-cpp`。

### 5.7 [DEC-GPU-GATING] Phase 0 utils 测试缺少 GPU 门控（已修复）

原问题：`CudaCheckTest` / `DeviceBufferTest` / `PinnedBufferTest` / `CudaTimerTest` 共 6 个用例直接调 CUDA 且没做环境判断。影响：无 GPU 的 CI / 沙箱永远不绿，真实回归信号被固定噪声淹没。现状：已统一门控。→ #1。

### 5.8 `supportsFormatCombination` 越界读取导致 engine 构建失败（已修复）

原问题：扫描了 TensorRT 未初始化的 `inOut[pos+1..]` → 所有格式组合都被判不支持。影响：Plugin 建不成 engine，host 单测完全无感。现状：只读 `inOut[0..pos]`。→ #2。

### 5.9 [DEC-SAMPLER-OLD-API] 采样器曾存在两套 API（已清理）

原问题：Phase 0 遗留的"标量 k/p + host `std::vector` 输出"与 Q7（per-batch tensor）及"Decode 全程驻留显存"冲突。现状：经作者授权删除 6 个旧桩文件，统一到 `sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu`。

### 5.10 [DEC-SANDBOX-NO-GPU] GPU 用例在沙箱内无法执行（已确认为环境限制，非缺陷）

见 §5.5。**判定要点**："无设备"才是环境限制（允许跳过）；"有设备但 `createInferBuilder` 失败"是故障（必须红）。

### 5.11 [DEC-GPT2-FP16-LIMIT] GPT-2 的 FP16 端到端不可用（已知限制 → **2026-10-06 撤销"按政策不修"**）

问题：真实 GPT-2 在本项目的**弱类型 FP16** 引擎下端到端产生 NaN（贪心输出恒为 0）；出 NaN 的层随构建变化（实测 0/1/2），而激活幅值远未触及 65504。已排除：LayerNorm 计算精度、`c_fc` / GELU、残差溢出。
workaround：**GPT-2 用 FP32**（8/8 贪心 token 命中、logits 相对偏差 `1e-6`）。后续路径 → `future_iterations.md` §1.4；完整定位（5 轮真机）→ `TROUBLESHOOTING.md` #18。

**2026-10-06 口径变更**：本条不再作为策略前提——作者裁决撤销"按政策不修"，转修复（`REQ-018`）。撤销依据是**正确性**（默认构建精度不可用），**不是**"性能收益变好"；上面的 workaround 与收益判断本身仍然成立。后续以 `docs/dev/REQ-018-gpt2-fp16-nan/STATE.md` 为准，追加留痕见 `TROUBLESHOOTING.md` + 18.2。

### 5.12 Phase 2 修掉的缺陷（结论索引）

5 个真缺陷（粘性 CUDA 错误 / KV 写入路径 / 多层共用 cache / 按层推进长度 / FP16 缓冲按假定精度分配导致越界写）全部已修复并有回归用例 → `TROUBLESHOOTING.md` #14 ~ #18。

### 5.13 [DEC-ARGMAX-CASE] 真机新红：`Gpt2OnnxTest.MatchesAcrossProfileShapes` 在 seq=512 上 argmax 不等（**已按方案 B 结案**）

问题：`batch=1 seq=512` 的逐行 argmax 判等失败，而同一次的 cosine 与相对界都通过。
性质：**不是缺陷**——两条独立实现在该行的分歧（`3.8e-05` / `6.9e-05`）大于要分辨的间距（约 `1.5e-05`），判据在这行**不携带"实现对错"的信息**。
处置：精确判据只在可判行（`m > 2d`）生效；不可判行必须钉到行号（713 行里只有 1 行 = 行 118）。真机复跑 **PASSED**，数值判据一个字没改。→ `TROUBLESHOOTING.md` #34

### 5.13b 性能"改动前后差百分之几"在本平台的判别下限（2026-09-27 沉淀，**含 workaround**）

问题：本平台的噪声与固定开销和信号同量级（一个平凡的 `greedy` kernel 单发就要 34~156 µs）。
影响：小于 **±400~600 µs** 的差异判不出来，只能写"无显著差异"，**不得**按百分比自动接受。
workaround：同 session、同二进制、逐轮交替（ABBA）、报中位数与四分位；**先声明判别下限再下结论**。→ #42（量法本身翻过车）。

### 5.14 §2.2 过程中沉淀的三条操作类坑（2026-09-27，**结论在此，过程见 TROUBLESHOOTING**）

① "漂移"要写清是哪个量的漂移（`max` 锚点曾被首档未稳态污染）；② 代码里"保守 / 宽松"的措辞必须与判据口径一致（写反过一次）；③ 观测项与判据必须分开记（F1 的 B 半句已按作者决定改为观测项）。都不是产品缺陷。→ `TROUBLESHOOTING.md` #45

### 5.15 `future_iterations.md` §1.5 过程中沉淀的四条"仪器类"坑（2026-09-27，**结论在此，过程见 TROUBLESHOOTING**）

同 §5.14 的性质（都不是产品缺陷，但每条都能让排查本身失效）：① 动态维在 `ICudaEngine` 上是 −1（形状只能问 `IExecutionContext`）；② 探针图会改变 tactic 选择（`i8i8` 从 4 变 0）；③ 用例绑定写错会把"仪器问题"伪装成"实现错了"；④ 首跑 3 条红全部来自用例自身的绑定。→ `TROUBLESHOOTING.md` #47

### 5.16 [DEC-PERF-UNVERIFIED] REQ-016 连续批的性能未验证（Dependency Missing，2026-10-05）

问题：REQ-016（S3 调度 / S4 packed 混合批 / S5 分块 prefill）的**全部性能结论未验证**。
影响：AC6（性能可复现）与 AC7（不浪费）无从结；`max_batch` 与 packed 引擎 profile 的 `opt` 仍是保守值 + "待实测"；2026-10-05 的 `chunk_limit` 修订让分块真的启用，默认路径的行为画像随之改变——更不允许声称收益。
workaround：环境恢复后按 `REQ-016/benchmark_before.md` 量**五项**（含 S5 的 chunk 维度对照）；`summary.md` 的 Performance 行在 P8 清。

### 5.17 [DEC-REQ017-PERF-UNVERIFIED] REQ-017 的 INT8 收益未验证（Dependency Missing，2026-10-05）

问题：weight-only INT8 的性能收益**没有任何实测**（P4 记 `N/A`，P7 随之不可执行 → `Gate-B: N/A`）。
影响：AC4 与"访存降到约 1/4"只是推导，不得当结论引用；**D7 未验证**——若 DQ 被构建期常量折叠，引擎体积与带宽都不会降、收益归零**且不报错**。
workaround：真机第一步做 D7 最小图实验（**判据 = 引擎体积必须下降**），再做 FP32 / INT8 四项基线与 P7 的 A/B；详见 `REQ-017/benchmark_before.md`。

## 6. [DEC-NEXT-STEPS] 下一步计划

> **当前没有"自动往下走"的阶段**：下一步一律等作者点名（`AGENTS.md` §0.7）。活着的条目看 §4 与 §6.6。

### 6.1 [DEC-GPT2-OPS] 关键事实：GPT-2 用不上 Phase 1 的 RMSNorm / RoPE

GPT-2 用 LayerNorm + 学习式位置编码，**不含 RMSNorm、不含 RoPE**（对 `1_gpt2_onnx/gpt2.onnx` 做过算子统计）。所以 Phase 1 的这两个插件在 GPT-2 上不被调用，它们服务于 LLaMA 类模型（后置）。

### 6.2 建议的开工顺序（按风险从高到低）

① 多权重加载 spike（最高风险：GPT-2 真实权重约 150 个张量、BF16/FP16 源）；② 建图 + 双引擎；③ 端到端对齐。**这是 Phase 2 之前的历史建议，Phase 2~5 已全部完成**——当前顺序只由作者点名决定。

### 6.3 Phase 1.5 已扫清的前置

dynamic shape 的 optimization profile 已打通（Phase 2 的双引擎直接依赖）；多权重取用路径已有端到端验证，且修掉了会让 GPT-2 静默建错的转换缓冲区缺陷。

## 6.5 [DEC-WORKSPACE-STATE] 工作区与本地产物状态（新会话先看这一节）

**提交状态一律以 `git log` / `git status` 为准**——本文不写"已提交 / 工作区干净"这类会立刻过期的判断句。

**本地产物与再生命令**（`models/`、`assets/legacy/`、`/tmp` 引擎，都不入版本控制）→ §7 的表。

**引擎缓存"一次性失效"的已知事件**（都是预期，别当异常）：① 指纹机制上线（2026-09-27）；② 模型路径规范化（`FileIdentity` 用绝对路径，#48）；③ `n_positions` 进指纹（REQ-016，2026-10-05）；④ **`kEngineGraphVersion` 3→4、`kPackedPrefillGraphVersion` 6→7，且量化清单与 int8 权重进指纹（REQ-017，2026-10-05）**。

**重建 ≠ 逐字节相同**：`builder.cpp` 未设 `kDETERMINISTIC`、无 timing cache → 同一网络重建后引擎体积会变（实测 `54,196,084 → 52,357,812`）。**性能对照必须用同一次构建的引擎**。

**已知会失败/跳过的测试**：真机唯一红 = §5.11 的 FP16 复现器（按设计）；`int8_crosscheck` 只在缺报告时跳过（77）；`Fp16PrefillOutputsDiagnostic` 与性能类是"只打印"用例——它们通过 ≠ 验过数值。

## 6.6 [DEC-OPEN-ITEMS-INDEX] 当前未解决项（开放项索引，2026-10-05 刷新）

> 事实与触发条件的唯一来源 = `docs/future_iterations.md`（§11 覆盖缺口 + §1.5 / §1.6 / §10.2 等立项条目）；
> 已立项条目自 2026-10-01 起落在 `docs/dev/REQ-NNN-*/`，状态以各 `STATE.md` 为准。本节只做索引，不复制状态值。

| 开放项 | 一句话 | 详见 |
|---|---|---|
| **REQ-016** | 连续批 / packed 混合批 / 分块 prefill：代码已落（S1–S5）、**未编译验证**；性能未验证（§5.16） | `docs/dev/REQ-016-continuous-batching/STATE.md` |
| **REQ-017** | LLM 权重 INT8（路线 C）：P6 用例与判据已落、**未编译未运行**；D7 与收益待真机（§5.17） | `docs/dev/REQ-017-llm-int8-quant/STATE.md` |
| **REQ-018** | GPT-2 FP16 端到端 NaN（bugfix）：**离线部分已全部落码（未编译验证）**；方案已评审（BLOCK：P0-3 待真机）、**真机搁置**（§4.8） | `docs/dev/REQ-018-gpt2-fp16-nan/STATE.md` |
| **REQ-019** | 外部图子图替换（乙框架）：**S1 已落码 + 沙箱自检通过**（连接级识别与反例夹具）；S0 设计就绪但受 T12 写冻结；S2 **阻塞**（P0-1 待真机核实），S3 待定（PF-7 待真机）（§4.9） | `docs/dev/REQ-019-onnx-subgraph/STATE.md` |
| **SP-1** | `SentencePieceTokenizer` 无用例、无资产、无调用方：正确性从未被任何参考裁决过 | `future_iterations.md` §0.1 / §11 |
| **P4-INT8-b** | INT8 的绝对数值界未定（当前只用"FP32 余量子集一致率"判，阈值 ≥90%，实测 12/12；验收集无真值标签） | `future_iterations.md` §1.6 |
| **P4-FP16-a** | FP16 路径仍用已废弃的 `BuilderFlag::kFP16`（TRT 10.12 起指向 strong typing）；实测可用 | `future_iterations.md` §11 |
| **G5** | ONNX 子图识别只做计数（未做拓扑级）——**已由 REQ-019 的 S1 立项并放行**（2026-10-06），不再挂"真做子图替换"这个条件 | `future_iterations.md` §11 |
| **PF-7** | ONNX vs 原生 prefill 的跨构建对照未跑（两次测量方向相反）→ 它只决定 REQ-019 的 **S3（子图替换）**走向，不再决定整条（2026-10-06 乙框架） | `future_iterations.md` §10.2 |
| **G2-3 / G2-4** | `LLMRunner` 曾有意只支持 `batch = 1`（batch 已由 REQ-016 扩）／EOS 无法循环内早停（§5.0） | `future_iterations.md` §11 |
| **P1.5-b ~ P1.5-d** | E2 完整链路留后；E3 未验多 profile 切换；采样器分布数据未固化 | `future_iterations.md` §11 |
| ~~P4-INT8-a~~ | **已结案（2026-09-27）**：根因在产图脚本的 scale 来源，见 §3.0j | `TROUBLESHOOTING.md` #46 / #47 |
| ~~G6 / §2.2 attention / P9_2-5b / P9_2-5c~~ | 均已交付或已关闭，见 §3.0g ~ §3.0i | — |
| **Phase 5（清理旧模块）** | **已完成（2026-09-28）**：两个历史目录已删、资产在 `assets/legacy/`、`MINI_TRT_REQUIRE_ASSETS` 闸门守着"缺资产不许静默跳过" | `REQ-009-retire-legacy/phase5_development_plan.md` |

## 7. [DEC-ENVIRONMENT] 重要环境信息

> **设备规则（2026-10-05 作者定）**：**检测到当前设备不是目标真机时，必须立即确认它能否真机测试**。
> 条文落在 `docs/dev/REQ-017-llm-int8-quant/requirement.md` 的 `## Constraints`（本处只作指针）；
> 若要它自动作用于每个会话，仍需进 `AGENTS.md` §1（受口令保护）。

| 项目 | 版本 / 说明 |
|---|---|
| 操作系统 | Ubuntu on WSL2 |
| GPU | NVIDIA GeForce GTX 1660 Ti Mobile（Turing，无 Tensor Core） |
| Compute Capability | `sm_75` |
| TensorRT | 10.15.1（`libnvinfer.so.10.15.1`） |
| CUDA Toolkit | 12.6.85 |
| GCC / C++ | 13.3.0 / C++17 |
| CMake | 3.28.3（要求 >= 3.18） |
| 第三方 | sentencepiece v0.2.0、safetensors-cpp（main，af90b6c）、GoogleTest（源码嵌入） |
| Python | `torch 2.5.1+cu121`；`transformers >= 4.40`、`safetensors >= 0.4`、`onnx >= 1.16` |

**跑真机用例的前提**（都不入库，换机器要按表重建；`.gitignore` 忽略 `*.bin` / `*.onnx` / `*.safetensors` / `**/calib_data/`）：

| 产物 | 生成方式 | 谁需要 |
|---|---|---|
| `models/gpt2/{config.json, model.safetensors}`（548 MB） | `tools/convert/hf_to_mini_trt_llm.py --model_name_or_path <HF gpt2 目录> --output_dir models/gpt2` | 全部 GPT-2 真机用例 |
| `assets/legacy/gpt2_onnx/gpt2.onnx`（652 MB） | `assets/legacy/scripts/gpt2_load_model.py` | Phase 3 的对拍与探针 |
| `models/resnet18/ref_*_b8.bin` + 元数据（14.5 MB） | `scripts/ref_resnet18.py --input {ramp,pixels} --output ...` | Phase 4 的 L2 / L3 对拍（缺则 skip） |
| `models/resnet18/model.safetensors`（42 张量，46.7 MB） | `tools/convert/onnx_to_mini_trt_llm.py --onnx assets/legacy/resnet18_onnx/resnet18.onnx --output_dir models/resnet18` | 原生 builder 的权重来源 |
| `models/resnet18/resnet18_qdq.onnx`（13.3 MB，**正式产物**）+ `.meta.json` | `tools/convert/quantize_resnet18.py ... --weight-form prequant_dq` | INT8 用例（缺则 skip）；默认路径逐字节可复现 |
| `models/resnet18/resnet18_qdq_per_channel.onnx` + 两份探针图 | 同上加 `--weight-scope per_channel` → `add_probe_outputs.py` | 只被 `Int8ProbeTest` 的 PC 臂使用；身份 = **#46 的复现样本，不是候选基线** |
| `assets/legacy/resnet18_onnx/calib_data/`（500 张，300 MB） | `assets/legacy/scripts/resnet18_prepare_calib_data.py`（联网下载） | 生成 pixels 基线、INT8 校准 |
| `/tmp/mini_trt_llm_*.engine` + `.fingerprint` | 首次跑用例时自动构建（分钟级），之后按指纹复用 | 真机用例；删掉即强制重建 |

> **入库的 `.meta.json`**：基线（`ref_*`）含 SHA256（可回答"基线有没有被改过"）；INT8 的
> `resnet18_qdq.meta.json` **目前不含任何 SHA256** → "正式 ONNX 有没有被改过"当前回答不了
> （已知缺口；补它要改转换脚本，需另行批准）。

---

*本文档用于新会话快速接手项目，不记录对话过程，只保留决策与状态。*
