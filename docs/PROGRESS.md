# mini_trt_llm 项目进度交接文档

> 最后更新：2026-09-26（**Phase 4 收口**：CV 路径打通 + INT8 落地）  
> 当前阶段：**Phase 4 已完成**（ResNet18：ONNX 路径 / 原生路径 / `CVRunner` / 转换工具 / FP16 / INT8）；
> **没有下一阶段**——**Phase 5（清理旧模块）已永久取消**，旧模块由作者自行处理（见 §6.6）。
>
> **接手必读五件事**：
> 1. **GPT-2 的推荐精度是 FP32** —— FP16 端到端数值不稳定（NaN，层数随构建变化），
>    按政策不修，见 §5.11 与 `docs/TROUBLESHOOTING.md` §18.1；
> 2. Phase 2 的残余缺口（含已定位的已知限制）见 §5.12 与 `docs/phase2_test_plan.md` §5；
> 3. Phase 3 仅剩 G5/G6 两项按触发条件处理的缺口，见 `docs/future_iterations.md` §11。
> 4. **真机全量当前是 182 条 / 1 条红 / 0 跳过**（2026-09-26 实测，`MINI_TRT_REQUIRE_GPU=1`）：
>    唯一的红是 GPT-2 的 FP16 NaN **按设计红**复现器（`RealGpt2Fp16GreedyMatchesReferenceTokens`，
>    按 AGENTS.md §7 保持红色）。**ResNet18 侧没有红**。
>    跑真机时请带 `MINI_TRT_REQUIRE_GPU=1`：否则 GPU 用例会静默跳过，等于白跑。
> 5. **Phase 4 的精度现状**：ResNet18 的 **FP32 / FP16 都健康**（argmax 全一致）；
>    **INT8 走 Q/DQ 显式量化**，判据是"**FP32 有余量子集的一致率**"（实测 12/12 = 100%），
>    权重默认 **per_tensor**；**per-channel 的整网退化原因未知**（开放项 P4-INT8-a，见 §6.6）。
>    细节见 §3.0d 与 `docs/phase4_int8_plan.md`。
>    **两条开放项已于 2026-09-26 立项**：P4-INT8-a → `docs/future_iterations.md` **§1.5**
>    （per-channel 整网退化根因；前 3 步**不需要联网**）、P4-INT8-b → 同文件 **§1.6**
>    （INT8 绝对误差判据 + 带真值标签的验收集；**前置依赖 = 联网下载，须先获批**）。
>    §6.6 与 `future_iterations.md` §11 只保留索引，目标 / 做法 / 验收判据在 §1.5 / §1.6。

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

### 2.12 Phase 1 算子层的接口约定（实现时确立，勿回改）

以下五条是实现 Phase 1 Plugin 时定下的约定，其中前三条与 `phase1_development_plan.md` 的早期表述**不一致**，
是经用户确认后的有意偏离，后续会话不要按 plan 原文"修正"回去：

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

### 2.13 测试与验证约定（Phase 1 / 1.5 沉淀，后续沿用）

- **参考实现必须唯一、必须自带断言、必须有 host 侧 meta-test**。
  **为什么**：参考实现是裁决对错的标尺，标尺错了会给出错误裁决——Phase 1.5 就吃过一次
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

### 2.15 Phase 2 / 3 确立的接口约定（实现时确立，勿回改）

与 §2.12 同性质：这些是实现时定下、且**与直觉写法相反**的约定。后续会话不要按"更自然"的写法改回去。

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
| 引擎缓存路径不得在"两种 I/O 契约"之间共用（如诊断开 / 诊断关） | 缓存只按路径名区分、**不随代码或开关失效**：共用一条路径时先跑的那次会决定后续用例拿到哪个引擎，测出假结果。诊断仪器因此单独用 `..._diag.engine` |
| ONNX 的 I/O 契约**按 `architecture` 选**（`OnnxIoContractFor`：`cnn` → `input`/`output`；其余 → `input_ids`/`logits`），且该映射必须能被 host 用例直接测到 | 契约映射内联在 `BuildFromOnnx` 里时，它只在"解析 ONNX + `createInferBuilder`"之后才执行——那两步都要 CUDA，等于护栏只在真机才验证得到。抽成函数后沙箱即可覆盖（`OnnxIoContractTest.*`） |
| CV 的 profile `opt_batch = 8`（历史工程口径），**不要改回 1** | TRT 针对 kOPT 形状挑最快 kernel；沿用 1 会让 batch ≥ 2 的推理走非最优 kernel、性能结论失真 |
| `CVRunner` 的输入契约 = **float32、NCHW、`[0,255]` 像素质**，归一化由 Runner 自己做；失败一律返回**空 vector / 零值统计**并打日志 | 该形态与 P4-1 的契约输入一致（可直接对拍）；HWC→CHW 留给调用方——形参名 `image_nchw` 就是这么定的。失败约定与 `LLMRunner::Generate` 保持一致（异常只用于构造期与底层库） |
| `CVRunner` 的维度与 batch 范围**必须向引擎查询**：输入用 `getProfileShape`、输出用 `getTensorShape` | 写死 224/1000/16 等于把"模型是什么"焊进 Runner；而两个查询 API 分工不同——`getProfileShape` **只对输入有效**（对输出返回 `Dims{-1,{}}`），用错的表现很像"契约不合法"（#23.2） |
| `CVRunner` 的前处理入参是 **`pixels_per_channel`（=H*W）显式传入**，不从总长度反推 | NCHW 的通道下标是 `(i/(H*W))%C`；用"总元素数/C"当分母在 **batch=1 时恰好等价**、batch>1 才错（#23.1）。凡是按 NCHW 拆下标的代码都要覆盖 `batch>1` |
| **建图侧 `markOutput` = 改 I/O 契约**：新增诊断输出必须由 `BuildOptions::export_diagnostics` 显式打开（默认关），且给独立引擎路径 | 默认带上诊断输出曾在真机打挂 6 条用例（#19）；引擎缓存只按路径名区分、不随开关失效 |
| **凡是做精度比较的用例，必须显式写出目标精度并打印**（`Config::precision` 默认 FP16，弱类型引擎的 I/O 却常被 TRT 定成 FP32） | 第一版 ResNet18 对拍把 FP16 引擎当 FP32 用，量到 0.033 差点被当成"TRT 差异大"（#21） |
| **INT8 走 Q/DQ 显式量化**：`SetupBuilder` 里 **不设任何 INT8 flag**，精度由图中的 Q/DQ 决定；Q/DQ 必须**对称**（zero_point 恒为 0），否则 TRT 解析期直接拒 | `kINT8` 自 TRT 10.12 废弃（由 strong typing / Q/DQ 取代）；非对称图报 `Non-zero zero point is not supported`（#27）。另外 Q/DQ **不受我们传的 precision 影响**——同一张 QDQ 图在 FP32/INT8 配置下都跑 INT8（层信息才是证据） |
| **INT8 的判据是"FP32 有余量子集的一致率"**（阈值 ≥90%），整体一致率只作"没崩坏"下界；**能用 `IEngineInspector` 自证在跑 INT8**（需 `Config::detailed_profiling = true`，且判 `Format/Datatype: Int8`，**不是** `[I8]` 标签） | 这批图 FP32 自身摇摆（55% 样本 margin<2），整体一致率主要在测测试集噪声；不设 `kDETAILED` 则读不出逐层精度，会误判成"没跑 INT8"（#29.4 / #30.5 / `phase4_int8_plan` §4） |

### 2.14 证据纪律与操作纪律（Phase 2 沉淀，后续沿用）

本节记录两条**流程级**教训，源自 Phase 2 的两次实际事故：
阈值放宽（技术事故，详见 `docs/TROUBLESHOOTING.md` #15）与擅自改写 Phase 0 文件
（操作事故，记录即本节 B 条）。它们不是技术缺陷，但代价比技术缺陷更高：
一次是让用户承担了本可以避免的决策负担，一次差点让一个真 bug 以"全绿"的形态留下来。

#### A. 禁止用"放宽期望值"换取通过

当实测与期望不一致时，**允许的动作只有三种**：

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

---

## 3. 已完成的部分

### 3.0a Phase 2 交付（GPT-2 原生构建，2026-09-25）

| 文件 / 模块 | 说明 |
|---|---|
| `core/gpt2_model_builder.{hpp,cpp}` | GPT-2 原生建图；`kSingle` / `kPrefill` / `kDecode` 三种切面共用同一份代码，只有注意力分支不同 |
| `core/llm_runner.{hpp,cpp}` + `core/llm_runner_kernel.{hpp,cu}` | Prefill→Decode→Sampler 自回归循环；循环内零 H2D/D2H（`position_ids` 由设备端 `context_lens` 填） |
| `kv_cache/paged_kv_cache.{hpp,cpp}` + `paged_kv_cache_kernels.{hpp,cu}` | 分页 cache：块池、序列预留、prefill 覆盖写、decode 追加（`AppendDecodeStep`） |
| `plugins/paged_attention_plugin.*` | 扩展为 5 / 7 输入两形态（第 6/7 个输入是当前 token 的 K/V） |
| `tools/convert/hf_to_mini_trt_llm.py` | 产出 mini_trt_llm 原生 `config.json`（`weight_map` / `skipped_tensors` / 布局声明 / `block_size`） |
| `models/gpt2/` | 转换产物（`model.safetensors` 被 .gitignore 忽略，`config.json` 入库） |
| 用例 | `test_gpt2_config` / `test_gpt2_network_build` / `test_gpt2_decode_consistency` / `test_gpt2_generate` / `test_gpt2_prefill_accuracy` / `test_paged_kv_cache` / `tests/gpt2_test_support.hpp` |
| `tools/inspect_engine.cpp` | 引擎 I/O 探针：反序列化任意 `.engine` 并打印各 I/O 的名字 / 方向 / **声明精度** / 维数（不建 context、不推理）。**不进构建流程**，编译命令写在文件头；用途与限制见 §6.5 |
| `tests/test_gpt2_generate.cpp` 的 FP16 补测 | 复现器 `RealGpt2Fp16GreedyMatchesReferenceTokens`（**真机预期失败**，见 §6.5）+ 纯打印诊断仪器 `Fp16PrefillOutputsDiagnostic`（逐输出给 `max|v|` 与 NaN 标记） |

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

### 3.0b Phase 3 交付（GPT-2 ONNX 路径，2026-09-25）

| 文件 / 模块 | 说明 |
|---|---|
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

### 3.0c Phase 2 补丁（诊断输出开关 + CUDA 环境判定，2026-09-25）

| 文件 / 模块 | 说明 |
|---|---|
| `core/imodel_builder.hpp` + `core/builder.{hpp,cpp}` | `BuildOptions::export_diagnostics` / `EngineBuilder::Config::export_diagnostics`，**默认关** |
| `core/gpt2_model_builder.cpp` | 4 处诊断 `markOutput` 改由开关控制 → 默认构建的输出数回到 `2*n_layer + 1` |
| `tests/test_gpu_guard.hpp` | `ProbeCudaDevice()`（主判定 `cudaGetDeviceCount`）+ `MINI_TRT_SKIP_IF_NO_CUDA` 宏 + `MINI_TRT_REQUIRE_GPU` 闸门 |
| `tests/test_cuda_check.cpp` | `GpuEnvProbe.ReportsCudaAvailability`：**永不跳过**，每次运行都打印环境事实 |
| `tests/test_gpt2_network_build.cpp` | SetUp 三分支："无设备"跳过、"有设备但建不出 builder"**判失败** |
| `tests/*.cpp`（19 个文件、59 处） | GPU 跳过统一走显式探测宏，跳过信息里带 `cudaGetDeviceCount` 原始结果 |

| `tests/test_paged_kv_cache.cpp` | P2S-6：追加用例改为**每层一对**并补两层读回断言；新增负例 `AppendDecodeStepRejectsLayerCountMismatch`（见 `TROUBLESHOOTING.md` #20） |

计划与验收见 `docs/phase2_supplement_plan.md`；缺陷与实测见 `docs/TROUBLESHOOTING.md` #19 / #20。
**真机全量结果**：146 条，0 跳过（`MINI_TRT_REQUIRE_GPU=1`），**1 条红 = FP16 NaN 复现器（按设计红）**。

### 3.0d Phase 4 交付（ResNet18 / CV 路径，2026-09-26）

计划与测试计划：`docs/phase4_development_plan.md`、`docs/phase4_test_plan.md`；
INT8 子计划：`docs/phase4_int8_plan.md`；缺陷与排查：#21 ~ #31。

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
| **真机全量** | **182 条 / 1 红 / 0 跳过**（唯一红 = GPT-2 的 FP16 已知限制） |

**过程中的真缺陷/真问题（全部留痕）**：#21（把引擎建成 FP16 却比 FP32）、#22（旧夹具失效 + 越界切片）、
#23（前处理通道下标在 batch>1 时算错、`getProfileShape` 用错 API）、#24（element-wise 的 rank 匹配、GLOB 需重跑 configure）、
#25（路径 helper 返回文件 → 用例静默跳过）、#26（FP16 阈值不能用 FP32 的尺子）、
#27 ~ #31（INT8：非对称 Q/DQ 被拒、per-channel 整网退化及其 11 条被否证的假设）。

**开放项**：见 §6.6（P4-INT8-a / P4-INT8-b / P4-FP16-a 等）。

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

| 文件 | 说明 |
|---|---|
| `include/mini_trt_llm/core/precision.hpp` + `src/core/precision.cpp` | 精度枚举与 TRT 映射（已补注释） |
| `include/mini_trt_llm/core/builder.hpp` + `src/core/builder.cpp` | 统一 EngineBuilder（已补默认值注释） |
| `include/mini_trt_llm/core/engine.hpp` + `src/core/engine.cpp` | Engine 封装 + Benchmark（已补统计注释） |
| `include/mini_trt_llm/core/imodel_builder.hpp` | 模型构建器抽象接口 |
| `include/mini_trt_llm/core/model_config.hpp` + `src/core/model_config.cpp` | JSON 配置加载 |
| `include/mini_trt_llm/core/model_registry.hpp` + `src/core/model_registry.cpp` | 模型注册表 |
| `include/mini_trt_llm/core/weight_loader.hpp` + `src/core/weight_loader.cpp` | Safetensors 权重加载 + weight_map 映射 |
| `include/mini_trt_llm/core/llm_runner.hpp/.cpp` | Phase 0 仅接口声明 |
| `include/mini_trt_llm/core/cv_runner.hpp/.cpp` | Phase 0 仅接口声明 |

### 3.4 其他模块占位

- `kv_cache/`、`plugins/`、`sampler/`、`tokenizer/` 头文件与空实现已就位，供 Phase 1/2/4 填充。

### 3.5 测试

- `mini_trt_llm/tests/test_*.cpp`：Utils / Core 骨架（cuda_check、logger、timer、memory_pool、io、
  model_config、model_registry、safetensors_loader、engine）+ Phase 1 算子 + Phase 1.5 端到端。
- 当前状态（2026-09-26 实测，含 Phase 4）：沙箱内 `ctest` **182 个用例，0 失败**
  （**87 个 GPU 用例自动跳过、95 个 host 用例实际执行**，含 `onnx_graph_probe` 与 `GpuEnvProbe`）。
  真机全量（`MINI_TRT_REQUIRE_GPU=1`）**182 条 / 0 跳过 / 1 条红**（GPT-2 的 FP16 NaN 复现器）。
  实测命令：`cmake --build build -j$(nproc) && ctest --test-dir build`（build 目录已配 `BUILD_TESTS=ON`）。
  分层与覆盖度详见 §3.9 / §3.10 与 `docs/phase1_test_plan.md`。
- 待补（不阻塞 Phase 2）：`docs/phase0_model_loading_test_plan.md` 里 T2（ONNX→Engine）仍未实施；
  T1 / T3 的能力已由 Phase 1.5 的 E1/E2 以更强的形式覆盖。

### 3.6 工具与文档

- `mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`：HF checkpoint → `config.json + model.safetensors` 转换脚本（真实实现的唯一落点）。
- `scripts/ref_rope.py` / `scripts/ref_sampler.py`：参考语义自检脚本（RoPE 与 HuggingFace 交叉验证、采样器截断语义与理论概率）。
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
- `docs/future_iterations.md`：后续迭代计划。
- `mini_trt_llm/third_party/sentencepiece/README.md`：记录禁用功能。
- `docs/PROGRESS.md`：本交接文档（已按 `progress-summary` skill 更新）。

> 历史文档：`docs/phase1_pending_confirmations.md` 的内容已全部合并进 `docs/phase1_development_plan.md` §10，原文件已删除。

### 3.7 第三方依赖

- `mini_trt_llm/third_party/sentencepiece`：v0.2.0，源码嵌入。
- `mini_trt_llm/third_party/safetensors-cpp`：main 分支，源码嵌入。
- `mini_trt_llm/third_party/googletest`：release 版本，源码嵌入。

### 3.8 注释规范修复

- 已按 `cpp-comment-style` skill 检查并修复 Phase 0 代码注释：
  - 公共 API（`precision.hpp`、`timer.hpp`、`json.hpp`）补齐类级/函数级注释。
  - 纯英文注释改为中文（`builder.hpp`、`memory_pool.cpp`）。
  - 魔数与边界逻辑补充"Why"注释（`builder.hpp` 默认值、`engine.cpp` 统计、`safetensors_loader.cpp` BF16 位运算）。

### 3.9 Phase 1 插件与采样器

| 文件 | 说明 |
|---|---|
| `include/mini_trt_llm/plugins/rmsnorm_{kernel,plugin}.hpp` + `src/plugins/rmsnorm_plugin.cu` | RMSNorm Plugin（一行一 block，FP32 `float4` / FP16 8×half 向量化，不能整除时回退标量） |
| `include/mini_trt_llm/plugins/rope_{kernel,plugin}.hpp` + `src/plugins/rope_plugin.cu` | RoPE Plugin（half-split 约定，双输入双输出，`position_ids` 作为输入） |
| `include/mini_trt_llm/plugins/paged_attention_{kernel,plugin}.hpp` + `src/plugins/paged_attention_plugin.cu` | PagedAttention Plugin（仅 Decoding，GQA/MHA，online softmax 单趟扫描） |
| `include/mini_trt_llm/sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu` | Greedy / Top-K / Top-P 采样器（设备侧 API，内联 Philox 随机源） |
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

### 3.10 Phase 1.5：全流程测试基建与收尾

| 文件 | 说明 |
|---|---|
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

## 4. 进行中 / 未完成的部分

### 4.1 Phase 1：Plugin 基础（已完成）

- 决策状态：15 项待确认问题已全部关闭，无遗留阻塞项（详见 `docs/phase1_development_plan.md` §10）。
- ✅ 完善 `IPluginV3` 基类，补齐 TRT 10.x 接口。
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
- Sampler 的手写高性能 kernel（Phase 1 用 CUB 分段排序保证正确性）。

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

### 4.3 Phase 2：GPT-2 原生构建（已完成，见 §3.0）

- 开工顺序与风险提示见 **§6.2**；关键事实（GPT-2 不用 RMSNorm / RoPE）见 **§6.1**。
- 先做多权重加载 spike（约 150 个张量、BF16/FP16 源），再确认 LayerNorm / GELU(tanh)
  用 TRT 原生层够用，然后实现 `GPT2ModelBuilder`（先 Prefill 单形状）。
- 之后接 `PagedAttention` + Decode 引擎 + Prefill/Decode 双 profile，实现 `LLMRunner`
  的 `Prefill → Decode` 自回归循环。
- 精度对比基准：`1_gpt2_onnx/ref_output.bin`（PyTorch FP32）。

### 4.4 Phase 3：GPT-2 ONNX + Plugin（已完成，见 §3.0.5）

- 实现 `OnnxBuilder` + subgraph replacer。
- 对 `1_gpt2_onnx/gpt2.onnx` 替换 RoPE / RMSNorm / Attention 子图。
- 验证与方案 A 输出一致。

### 4.5 Phase 4：ResNet18 替换（✅ 已完成，2026-09-26）

**交付清单与实测数字见 §3.0d**。计划文档 `docs/phase4_development_plan.md`（§1 保留了
"先读 `0_resnet18_onnx` 历史工程"的四条关键发现，供后续参考）：

1. 该 ONNX **已在导出时折叠 BatchNorm**（42 个 FP32 张量全是 Conv/Gemm 的 weight+bias），
   算子是 `Conv/Relu/Add/MaxPool/GlobalAveragePool/Flatten/Gemm`——**原生建图不需要任何 Plugin**；
2. 历史工程**没有留下外部基线**：它的"精度验证"是自相对（FP16/INT8 vs 它自己的 FP32），
   且推理输入是合成 ramp、benchmark 输入是常量 0.5 → Phase 4 必须先造 torchvision 基线；
3. **INT8 的三条口径互相矛盾**（本文档 §6 写"INT8 延后"、`future_iterations.md` §1.1 列为后续、
   历史工程其实已有 calibrator + 500 张真实校准图）→ 由计划的 **D2** 收敛；
4. `EngineBuilder::Config` 的 CV `opt_batch` 默认是 **1**，历史工程用的是 **8** → 会影响性能结论。

任务分解 P4-0 ~ P4-8、测试要点、判据出处都在该计划里；INT8 子计划见 `docs/phase4_int8_plan.md`。
**结论**：ResNet18 的 **FP32 / FP16 都健康**；**INT8 走 Q/DQ 显式量化、判据用"FP32 余量子集一致率"**
（实测 12/12 = 100%），权重默认 per_tensor。**per-channel 的整网退化原因未知** → 开放项 §6.6。

### 4.6 Phase 5：清理旧模块（❌ 已永久取消，2026-09-26 由用户决定）

- **用户决定：Phase 5 永久取消**。旧模块 `0_resnet18_onnx/`、`1_gpt2_onnx/` 与根 `CMakeLists.txt`
  里的注释项**保持原样**，**由作者本人按需处理**；Agent **不要**删除或移动它们。
- **为什么不能擅自删**：它们不只是"旧代码"——`0_resnet18_onnx/` 还是 Phase 3（`gpt2.onnx` 走
  `1_gpt2_onnx/`）与 INT8（`calib_data/` 500 张真实图 + `resnet18.onnx`）的**本地产物来源**，
  删掉会让 ONNX / INT8 用例全部跳过。这也与 AGENTS.md §0.6 一致：**"这东西没人用"的判断权在作者**。

---

## 5. 已知问题与坑

### 5.0 `LLMRunner` 无法在解码循环内早停 EOS（有意为之的 workaround）

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

### 5.7 Phase 0 utils 测试缺少 GPU 门控（已修复）

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

### 5.9 采样器曾存在两套 API（已清理）

- **问题（已解决）**：Phase 0 留下的 `sampler/{greedy,topk,topp}_sampler.{hpp,cpp}` 声明的是「标量 k/p + host `std::vector` 输出」的接口，与已确认的 Q7（per-batch tensor）和「Decode 全程驻留显存」冲突，函数体仍是 `throw not implemented`。
- **解决**：经用户授权删除 6 个旧桩文件，采样器 API 统一收敛到 `sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu`。

### 5.10 GPU 用例在沙箱内无法执行（已确认为环境限制，非缺陷）

- **问题**：所有 kernel 数值与 engine 集成用例都需要 GPU，沙箱内只会 `GTEST_SKIP`。
- **影响**：Agent 侧的结论上限是「编译通过 + 契约自洽 + host 侧逻辑正确」。
- **现状**：Phase 1 全部 GPU 用例已由用户在真机验证通过；这是**流程约束而非遗留缺陷**，
  后续 Phase 每完成一个算子，都需要同样走一遍真机验证。

---

### 5.11 GPT-2 的 FP16 端到端不可用（已知限制，按政策不修）

- **问题**：真实 GPT-2 在本项目的**弱类型 FP16** 引擎下端到端产生 NaN（贪心输出恒为 0）。
  出现 NaN 的层随构建变化（实测 0/1/2），而激活幅值远未触及 FP16 上限 65504。
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

### 5.12 Phase 2 修掉的缺陷（结论索引）

Phase 2 的 5 个真缺陷（粘性 CUDA 错误 / KV 写入路径 / 多层共用 cache / 按层推进长度 /
**FP16 缓冲按假定精度分配导致越界写**）全部已修复并有回归用例，
经过与推导见 `docs/TROUBLESHOOTING.md` #13 ~ #16 与 #18（前半）。
其中两条**影响接口设计**（均已落到 §2.15 的表里，勿回改）：

- **#16**：`PagedKVCache` 的追加接口拆成 `AppendDecodeKV`（只写）+
  `AppendDecodeStep`（一次写全部层、只推进一次长度）——不要按"每层调用一次并各自推进"的直觉改回去。
- **#18（前半）**：凡"按配置推断别人的宽度"的地方都要改成"向对方查询"，即边界精度查询、
  `source_is_half` 与 decode cache 输入精度校验这三条。

## 6. 下一步计划

**没有下一阶段。** Phase 0 / 1 / 1.5 / 2 / 3 / 4 全部完成，**Phase 5（清理旧模块）已永久取消**（§4.6）。

**接下来做什么，取决于触发条件**（全部见 §6.6 的开放项索引与 `future_iterations.md` §11）：

- 需要更高 INT8 精度 → **P4-INT8-a**（per-channel 整网退化；**已立项为 `future_iterations.md` §1.5**，
  含做法与验收判据，不依赖联网）；
- 需要 INT8 的绝对误差保证 → **P4-INT8-b**（**已立项为 `future_iterations.md` §1.6**；
  前置依赖是联网下载带真值标签的验收集，须先获批）；
- 真要迁强类型网络 → **P4-FP16-a / P4-INT8 的强类型路线**；
- 要扩展 CV（新模型/动态分辨率/图像解码）→ 见 `future_iterations.md` 对应章节，**先产出计划文档再动手**（AGENTS.md §5）。

**每轮的开工纪律**（Phase 4 全程验证过有效）：先计划对账 → 破坏性动作一次性列清单 → 真机全量回归 → 回填文档。

<details><summary>Phase 2 原始开工顺序（已完成，保留备查）</summary>

**Phase 2：GPT-2 原生构建（方案 A）**

### 6.1 关键事实：GPT-2 用不上 Phase 1 的 RMSNorm / RoPE

对 `1_gpt2_onnx/gpt2.onnx` 做过算子统计：

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
- 端到端骨架（模型目录 fixture / safetensors 写入 helper / 参考实现 meta-test）可直接复用。

> 开发流程提醒：Phase 1 的经验是「沙箱内 host 用例通过不代表真机没问题」——
> RoPE 的部分旋转缺陷只有真机 kernel 执行才暴露。后续每个 Phase 结束都应走一遍真机验证。

> Phase 1 / 1.5 沉淀的完整测试与验证约定见 **§2.13**（参考实现唯一性与 meta-test、
> `batch > 1` 覆盖、验证分层、失败判别方法等）。

> 本节只保留下一步入口。Phase 1 的 15 项已确认决策见 `docs/phase1_development_plan.md` §10，
> 接口约定见本文档 §2.12，测试与验证约定见 §2.13，证据与操作纪律见 §2.14，均已归档。
>
> </details>

---

## 6.5 工作区与本地产物状态（新会话先看这一节）

**代码与文档的提交状态**：**除下一条列出的 3 份文档外，全部已提交**（提交基线 `61718b6`；
不要去找"未提交的 WIP"——那批产物已经落盘，见下）。
最近一次提交 `61718b6`（"complate Phase 4"，2026-09-26）——Phase 4 的全部产物落盘。
再往前 `625939c`（"test for supplementary Phase 2"，2026-09-25）一笔记下了三件事：

1. Phase 2 的 **FP16 边界精度修复**——`src/core/llm_runner.cpp`（查询引擎声明的精度）、
   `src/kv_cache/paged_kv_cache_kernels.cu`（`WriteKVKernel` 双模板）、
   `src/core/gpt2_model_builder.cpp`（LayerNorm 显式 FP32 + 第 0 层诊断输出）；
2. **复现器与仪器**——`tests/test_gpt2_generate.cpp`、`tools/inspect_engine.cpp`；
3. 9 份文档的同步更新。

> **本次会话的未提交改动（2026-09-26）**：只有 3 份文档——本节、`future_iterations.md`
> （新增 §1.5 / §1.6 两条立项条目，`P4-INT8-a` / `P4-INT8-b`）、`phase4_int8_plan.md` §7
> 的立项说明。**没有代码改动**。是否提交由用户决定（AGENTS.md §0.2）。
> 上一版这里写的是"最近一次提交 `625939c`、工作区干净"——那在 `61718b6` 落盘后就已经过期，
> 2026-09-26 一并更正（同 §5 第 4 条：文档与现状矛盾要当场修，并把原因写进去）。

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
| 测试用引擎缓存 | `/tmp/mini_trt_llm_gpt2_*.engine` | 首次运行自动构建（分钟级），之后复用；**删掉会强制重建**（测构建耗时时需要） |
| **ResNet18 本地产物（四类）** | `models/resnet18/` | ① P4-1 基线：`ref_{ramp,pixels}_b8.bin` + `inputs/*.f32.bin` + `.meta.json`（14.5 MB）；② 原生路径权重：`model.safetensors`（42 张量 / 46.7 MB）+ `config.json`（入库）；③ INT8 图：`resnet18_qdq.onnx`（**13.3 MB**，`prequant_dq` 形态）+ `.meta.json`；④ `config.json` 入库，其余 `.bin`/`.safetensors`/`.onnx` 按 `.gitignore` **不入库**。再生命令见 §7 |
| ResNet18 引擎缓存 | `/tmp/mini_trt_llm_resnet18_*.engine` | 含 fp32（ONNX / 原生）、fp16（ONNX / 原生）、qdq_int8；缓存**只按路径名区分**，改了产物或建图后必须先删再跑（见 §2.15） |
| 引擎 I/O 探针 | **`mini_trt_llm/tools/inspect_engine.cpp`**（已入库，手动编译） | 反序列化任意 `.engine` 并打印其 I/O 契约（名字 / 方向 / **声明精度** / 维数）——**不建 context、不推理、不写数据**。用途见 `docs/TROUBLESHOOTING.md` #18：它是把"TRT 在弱类型 FP16 网络里把 K/V 与 logits 的输出定成 FP32"这件事**读出来**的工具（在此之前只能靠推断，而推断被证伪过两次）。编译命令写在文件头（工具不进构建流程，故无 CMake 目标）。**限制**：反序列化需要 CUDA 初始化，只能在真机跑（沙箱报 `error 35`）；实测无 GPU 时它会干净报错退出而不是崩溃 |

**已知会失败/跳过的测试**（避免新会话误判为回归）：

- 真机（`MINI_TRT_REQUIRE_GPU=1`，2026-09-25 全量实测 146 条 / 0 跳过 / **1 红**）：
  - `RealGpt2Fp16GreedyMatchesReferenceTokens`：FP16 已知限制的**按设计红**（NaN → token 不符）。
  - 原第二条红 `PagedKVCacheTest.AppendCrossesBlockBoundary...`（测试与 `AppendDecodeStep`
    契约不同步）已修复，见 `docs/TROUBLESHOOTING.md` #20 / 计划 P2S-6。
  改动过建图或开关时，**务必先删 `/tmp` 里的引擎缓存**（缓存路径只按名字区分，不随代码失效）。
- 真机：**跑全量一定要带 `MINI_TRT_REQUIRE_GPU=1`**——否则无设备时用例会静默跳过，等于白跑。
- 真机：`Fp16PrefillOutputsDiagnostic` **应当通过**——它是**纯打印**的诊断仪器（只输出
  `max|v|` / NaN 标记，不 assert 数值），"它凭什么通过"的答案就是它不判定正确性；
  别看到"FP16 出 NaN"就以为这条也该红。
- 沙箱：全部 GPU 用例 `GTEST_SKIP`（无 GPU，见 §5.10）；`onnx_graph_probe` 在缺 `onnx` 包或
  缺 `1_gpt2_onnx/gpt2.onnx` 时返回 77 → `Skipped`（**设计如此**，缺环境 ≠ 图有问题）。

## 6.6 当前未解决项（开放项索引，2026-09-26）

> **事实与触发条件的唯一来源是 `docs/future_iterations.md` §11**；已**立项**的条目
> （目标 / 做法 / 验收判据 / 前置依赖）在同文件 §1.5（P4-INT8-a）与 §1.6（P4-INT8-b）。
> 本节只做**索引**，避免多处维护。**这些都不是"已知缺陷"**——已发现的缺陷一律进
> `TROUBLESHOOTING.md` 并配回归用例；开放项是"还没被盯住的地方 / 已知的限制"。

| 开放项 | 一句话 | 触发条件（什么时候做） |
|---|---|---|
| **P4-INT8-a** | **权重 per-channel 量化在整网上比 per-tensor 差**（余量子集 54.5% vs 100%），而单卷积与最小残差 block 上它**不差**——原因未知；已排除 11 条假设（`TROUBLESHOOTING.md` #28/#29/#30/#31） | 需要更高 INT8 精度时。**已立项：`future_iterations.md` §1.5**（做法 = 探**量化前**的 float 张量，量化后的会被 bin 边界 ±1 格噪声淹没） |
| **P4-INT8-b** | INT8 的**绝对数值界未定**（当前只用"FP32 余量子集一致率"判，阈值 ≥90%，实测 12/12；验收集无真值标签） | 需要给出 INT8 绝对误差保证时。**已立项：`future_iterations.md` §1.6**（前置 = 联网下载带标签验收集，须先获批） |
| **P4-FP16-a** | **FP16 路径仍用已废弃的 `BuilderFlag::kFP16`**（TRT 10.12 起废弃、指向 strong typing）；实测可用 | 真要迁到强类型网络时（两条 builder 的每个算子都要显式设类型） |
| **G5 / G6** | ONNX 子图识别只做计数（未做拓扑级）／ONNX 性能无可复现测量方法 | 真要做子图替换 / 真要优化 ONNX 路径性能时（`future_iterations.md` §10.2） |
| **G2-3 / G2-4** | `LLMRunner` 只支持 `batch = 1`（有意限定）／EOS 无法在循环内早停（语义正确、多算） | 需要批处理 / 需要真早停时 |
| **P1.5-a ~ P1.5-d** | Top-K/Top-P 的 FP16 未覆盖；E2 完整链路留后；E3 未验多 profile 切换；采样器分布数据未固化 | 见 `future_iterations.md` §11（各有触发条件） |

**另有两条"已取消 / 不属于开放项"的说明**：

1. **Phase 5（清理旧模块）已永久取消**（用户 2026-09-26 决定）：`0_resnet18_onnx/`、`1_gpt2_onnx/`
   与根 `CMakeLists.txt` 的注释项**由用户自行处理**，Agent 不要删除或移动它们——
   它们还是 Phase 3/INT8 对拍与标定的**本地产物来源**（删掉会让 ONNX/INT8 用例跳过）。
2. **R0.1（`ResNet18ConfigTest.LoadsCnnConfig`）不单独落地**：config 解析断言已由
   `ResNet18WeightContractTest` 承担（见 `phase4_test_plan.md` §7），刻意不建重复用例。

## 7. 重要环境信息

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
| `models/resnet18/resnet18_qdq.onnx`（13.3 MB）+ `.meta.json` | `python3 mini_trt_llm/tools/convert/quantize_resnet18.py --onnx 0_resnet18_onnx/resnet18.onnx --calib-dir 0_resnet18_onnx/calib_data --output models/resnet18/resnet18_qdq.onnx --calib-images 500 --calib-percentile 99.9 --weight-form prequant_dq`（默认权重粒度 per_tensor；`--weight-scope per_channel` 可复现那个开放项） | INT8 用例（`ResNet18Int8*`）；**缺了会 skip** |
| `0_resnet18_onnx/calib_data/`（500 张真实图，300 MB） | `python3 0_resnet18_onnx/prepare_calib_data.py`（需 datasets/PIL，联网下载 tiny-imagenet） | 生成 pixels 基线、INT8 校准 |
| torchvision 权重缓存 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`（46 MB） | torchvision `ResNet18_Weights.DEFAULT` 首次使用时下载 | 生成 Phase 4 基线；**已在本地缓存** |

> **注意**：上述 `models/resnet18/*.bin`、`calib_data/`、`*.onnx`、`*.safetensors` **都不入库**
> （`.gitignore` 规则：`*.bin` / `*.onnx` / `*.safetensors` / `**/calib_data/`）。
> 仓库只跟踪源码与文档；换机器要按上表重建本地产物。入库的只有 `.meta.json`（含各产物 SHA256，
> 用于回答"基线有没有被改过"）。

---

*本文档用于新会话快速接手项目，不记录对话过程，只保留决策与状态。*
