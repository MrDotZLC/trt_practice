# mini_trt_llm 面试材料

> **用途与定位**：面向面试的个人项目材料，**不是项目状态文档**，也不参与 `docs/` 的 SSOT。
> 本文**不新增任何技术结论**：所有数字、判据与结论的出处仍是 `PROGRESS.md`（现状与基线）、
> `TROUBLESHOOTING.md`（排查过程，TS-NNN 编号）、`future_iterations.md`（未做与触发条件）。
> 面试口径与那三份冲突时，以那三份为准。
>
> **组织方式**：按面试官实际的提问顺序排成五步——
> Part 1 介绍项目 → Part 2 亮点深挖 → Part 3 系统问题 → Part 4 最难的问题 → Part 5 用 Codex 怎么开发。
> 正文可以直接照着讲；被追问时翻附录（完整功能清单 / 全量数字表 / 剩余问答）。
>
> 更新规则：只在措辞、组织与"怎么讲"上改动；引用实测值时连同出处一起更新，
> 不在此处复制未经实测的数字。

---

## Part 0｜答题纪律（贯穿全程，面试前扫一眼）

### 0.1 红线（这些话说出口会被当场问穿）

- 别说"实现了 TensorRT-LLM"、"支持 continuous batching"、"支持 batch > 1"——runner **有意限定** `batch = 1`。
- 别说"支持 LLaMA / Qwen / 任意 HuggingFace 模型"——注册表里只有 `gpt2` 与 `resnet18`；
  **算子件齐 ≠ 能跑**（LLaMA 缺 builder、SentencePiece 未验证）。也说不出"支持任意
  decoder-only Transformer"：每个新架构都要新 builder + 一轮对拍。
- 别说"TRT 没有 attention / RoPE / KV cache 设施"——TRT 10.15 **有** `IAttention`
  （`addAttention(q,k,v,normOp,causal)`）、`IRotaryEmbeddingLayer`、`IKVCacheUpdateLayer`
  （KV cache **仅 `kLINEAR`**）。我们自研的真实理由是**分页布局**与**并行控制面**，
  以及 prefill 手搭那条明确取舍（`docs/dev/REQ-004-gpt2-native/phase2_development_plan.md` D1）。
  把 TRT 说小会被当场问穿。
- 别说"支持 INT4 / FP8"、"KV cache 支持量化"——都没有；PagedAttention 只接受 FP32 / FP16。
- 别说"FP16 端到端可用"；提"1 红"时要顺带说明那是**按设计**的 FP16 复现器，
  否则听起来像留了个未修的 bug。
- 别说"做了 ONNX 子图替换"——那两条路只做了**子图识别计数**，替换是未做的开放项。
- 温度 ≠ 1 是**直接拒绝**而非静默忽略；被问到时解释成有意设计（静默忽略会让"调参无效"
  看起来像"模型就是这样"）。
- 不要引用未经自己实测的数字，也不要引用别的项目的性能数字来给自己背书。

### 0.2 只背这 8 组数字（其余别背，答不出细节反而受伤）

| # | 数字 | 出处 |
|---|---|---|
| 1 | GPT-2 FP32 贪心 8 token 与 HF **逐 token 一致** | `PROGRESS.md` §3、§5 |
| 2 | GPT-2 prefill logits：`max_abs 9.92e-05`、**相对 `9.19e-07`**、cosine 1.0 | 同上 |
| 3 | ResNet18 FP32 对 torchvision `9.5e-06`；原生 vs ONNX `1.07e-06` | 同上 |
| 4 | ResNet18 INT8：FP32 余量子集一致率 **12/12**（整体 38.3%） | 同上、`REQ-011` |
| 5 | split-K：斜率降 **88.85%**（kernel）/ **88.20%**（端到端）；`max_rel 1.5e-06` | 同上、`REQ-014` |
| 6 | ctx = 1024：每层每步 **0.919 → 0.126 ms** | 同上 |
| 7 | Top-P 采样：配对净收益 **12.6 / 22.4 / 15.7 / 17.6×** | 同上、`REQ-012` |
| 8 | 规模：`src` + `include` 约 **9.9k 行**、**265** 条注册用例、**50** 条排查记录 | `PROGRESS.md` §3、`TROUBLESHOOTING.md` 条目数 |

环境固定项：TensorRT 10.15.1、CUDA 12.6.85、`sm_75`、C++17。

### 0.3 复用地图（同一个说法会在哪几个问题里用到）

一个素材会被问两次，第二次必须换角度讲，否则听起来像复读。

| 说法 | 会出现在 | 换角度 |
|---|---|---|
| 弱类型网络下精度只能查、不能假定 | 2.2 主线 B / 4.2 最隐蔽的 bug / 4.4 数值正确 | 讲架构时讲"契约"，讲 bug 时讲"为什么 FP32 下永远不暴露" |
| 每层一对独立 cache 张量 | 2.1 追问 / 3.1 为什么 70B 不现实 / 4.2 | 讲设计时讲正确性，讲规模时讲 I/O 数量 |
| INT8"量 A 裁 B" | 2.3 主线 C / 4.2 最贵的坑 | 2.3 讲判据设计，4.2 讲排查路径与代价 |
| 分页 KV + split-K | 2.1 / 3.3 集群适配 / 4.1 最难技术点 | 讲实现 / 讲扩展 / 讲约束 |
| 阈值必须写出处 | 2.3 / 4.4 / 5.3 | 讲量化判据时讲"为什么不用绝对误差" |
| 引擎构建指纹 | 1.4 流程 / 2.2 主线 B / 3.2 快速部署 | 讲缓存机制 / 讲手工 bump 图版本 |
| `batch = 1` 是有意限定 | 3.3 / 3.4 / 红线 | 讲取舍，不讲缺陷 |

---

## Part 1｜第一步：介绍项目（做了什么 / 目的 / 流程 / 效果）

### 1.1 30 秒版

> 我用 C++ / CUDA 从零写了一个面向 Turing `sm_75` 的极简 TensorRT 推理框架，把 TensorRT-LLM 的
> 核心机制实现了一遍：原生建图、Paged KV Cache、自研 PagedAttention 插件、设备侧采样器、
> prefill/decode 双引擎。在长上下文下我做了 FlashDecoding 式的上下文维切分，每步 decode 的
> 延迟增长斜率降了 88%，生成结果与单趟实现完全一致。

一句话定位：一个面向 NVIDIA Turing / `sm_75` 的极简 TensorRT 推理框架，把 TensorRT-LLM 的核心机制
（原生建图、Paged KV Cache、自研 PagedAttention 插件、算子融合、设备侧采样、prefill/decode 双引擎）
从零实现了一遍，并用一套带出处的数值与性能判据在真机验证过。

### 1.2 2 分钟版

1. **背景**：想搞清 TRT-LLM 为什么快，光调库学不到东西，于是给自己定约束——只依赖
   TensorRT + CUDA，关键 kernel 自己写。
2. **做法**：配置驱动 + 模型注册表建图；权重走 Safetensors；KV Cache 分页；decode 注意力用
   `IPluginV3` 手写；采样器（greedy / top-k / top-p）全部设备侧、解码循环内零 H2D/D2H；
   另留一条 ONNX 路径做交叉验证。
3. **验证**：与 HuggingFace / PyTorch 参考对拍——GPT-2 FP32 贪心 8 token 全中、
   logits 相对偏差 `9.19e-07`；ResNet18 FP32 对 torchvision `9.5e-06`；
   INT8 用"FP32 余量子集一致率"判，实测 12/12。
4. **性能**：先建可复现的测量方法，再定位瓶颈——长上下文下 attention 占每步约 80%，
   机制是每层只有 12 个 block、24 个 SM 一半闲置；改成上下文维 split-K 后斜率降 88%。
5. **收尾**：引擎构建指纹缓存、插件与图版本的手工代次契约、50 条带编号的排查记录。

### 1.3 目的与约束（为什么做 / 为什么不用现成框架）

- **起点**：仓库里原本是两个独立的 TensorRT 示例（ResNet18 / GPT-2，各写各的 ONNX 加载），
  我把它们合并成一个框架，并给自己定了一条约束：**只依赖 TensorRT + CUDA，关键 kernel 自己写**。
- **为什么不用 TRT-LLM / vLLM**：① 目标是搞清楚机制，调库学不到；② 这台机器**没有 Tensor Core**，
  靠调库吃低精度吞吐走不通，力气只能放在访存与并行上；③ 生产替代本来就不成立。
- **和 vLLM / TRT-LLM 比的优势**：体量小、全链路可读可改、kernel 全自有、数值与性能判据都带出处；
  **劣势**：生态、模型覆盖、绝对性能。
- **追问链**（问到动机时按这个顺序答）：动机（合并 + 不调库）→ 为什么不用 TRT-LLM（机制 + 硬件吃不到收益）
  → 收获（工程纪律）→ 不足（清楚边界）。

### 1.4 流程：从权重到 token，每一步我们做了什么

> 面试官问"流程是什么"时，**不要念通用流水线**，念我们在每一步的具体实现与取舍。

| 阶段 | 本项目做了什么（含取舍） |
|---|---|
| 离线产图 | 自研 `hf_to_mini_trt_llm.py` / `onnx_to_mini_trt_llm.py`；核心设计是 `config.json` 里的 **`weight_map`**（"TRT 侧名字 → 权重源 key"），把权重改名与建图代码解耦；ResNet18 的 BN 在导出时就折进卷积 |
| 读配置 | 自研递归下降 JSON 解析器（支持 `\uXXXX`——GPT-2 词表 key 全是 `"\u0120the"` 这种转义）；`ModelRegistry` 按 `model_type` 选 `IModelBuilder`，加新模型只需注册一个子类 |
| 加载权重 | 解析复用第三方 `safetensors-cpp`；**dtype 转换自己写**：FP16 / BF16 / FP64 统一先还原 FP32 再降到目标精度（BF16 不能靠位截断）；转换结果按张量名缓存，保证 TRT 拿到的裸指针在整次建图期间稳定 |
| 建图 | 用 TRT Network API 手搭：GPT-2 逐层拼 LayerNorm（**显式 `setComputePrecision(FP32)`**，不赌默认值）、QKV（`Shuffle`/`Slice` 拆分）、FFN、GELU-tanh、残差、LM Head；**prefill 的注意力是自己拼的 `MatMul → 缩放 → 因果 mask → Softmax → MatMul` 子图，decode 换成自己的 PagedAttention 插件**；`tie_word_embeddings` 时在图上转置 `wte`，不物化冗余拷贝 |
| Profile 与缓存 | LLM 按 stage 挂 prefill / decode 两组 optimization profile（ONNX 路径只挂 prefill 一组）；引擎旁写**构建指纹**（源文件 size+mtime、全部建图数值参数、开关、TRT·CUDA 版本、手工图版本 → FNV-1a），指纹不一致或缺失就重建 |
| 装引擎 | 反序列化 → 建 `ExecutionContext`；这一段是 TRT API 的薄包装，真正的 tactic 选择与显存规划都在 TRT 里。**唯一自己加的护栏：缓冲精度一律向引擎查询声明值**，不按配置假定 |
| Prefill | 绑 `input_ids` / `position_ids` 与每层 K/V 输出 → `enqueueV3` → **手写的分页写入 kernel** 按 block table 把 `[B, H, S, D]` 散写进 cache（源精度 × 目标精度四种组合显式分发）→ **手写采样 kernel** 出第 1 个 token |
| Decode 循环 | **手写 kernel / 插件，循环内零 H2D/D2H**：用设备端 `context_lens` 填 `position_ids`（不为一个整数回主机）→ PagedAttention 插件（online softmax + 上下文维 split-K + 两阶段归约，当前 token 的 K/V 一起参与）→ 手写追加 kernel（写当前 token 的每层 K/V，**整步只推进一次长度**）→ 再采样，token 直接留在显存里，下一步绑成 `input_ids` |
| 出结果 | 循环外一次性 D2H + 同步 → 主机侧 EOS 截断（**循环内不早停是有意取舍**：早停必须每步同步）→ `int32 → int64`；文本进出时用手写的 byte-level BPE。CV 路径同构：手写归一化 + 维度与 batch 范围一律向引擎查询 |

### 1.5 达到的效果

- **LLM 侧**"从权重到 token"的全链路是通的：原生 + ONNX 两条路、prefill/decode 双引擎、
  分页 KV、自研注意力与采样；GPT-2 FP32 贪心 8/8 token 与 HF 一致。
- **CV 侧**从 FP32 到 INT8 是通的：ResNet18 三种精度都验过。
- **判据带出处**：每条阈值旁边写清是哪次实测 / 哪个标准 / 哪份文档。
- **长上下文性能**：每步 decode 延迟增长斜率降 88%（kernel 级两套量法互相印证）。
- **工程基建**比功能面更突出：265 条注册用例、资产闸门、50 条排查记录、三层文档体系。

### 1.6 规模与技术栈

- C++17 / CUDA 12.6（`CUDART_VERSION 12060`）/ TensorRT 10.15.1（`IPluginV3`）/ CUB /
  SentencePiece / GoogleTest；Python 侧 torch + onnx + transformers 做参考实现与产图。
- `src` + `include` 约 **9.9k 行**（不含测试与第三方）；**265** 条注册用例；**50** 条排查记录。
- `include/` 39 个头文件、`src/` 27 个实现文件、`tests/` 56 个文件（含 42 个 `test_*.cpp`）。
- 开发过程按 `docs/dev/REQ-NNN-*` 分 19 个条目推进，每个条目一套 artifact（见 Part 5）。

---

## Part 2｜第二步：亮点深挖（面试官说"挑一个你觉得最难的讲"）

三条主线都可以讲。每条按**30 秒 → 2 分钟 → 深挖**三档准备，末尾挂"会被追问"。选哪条见 2.4。

### 2.1 主线 A：PagedAttention 全链路 + 长上下文 split-K

**对应 TRT-LLM**：paged KV cache / block manager；generation phase 的 decoder attention
over paged KV cache（XQA 那一类 kernel），需支持 MHA / GQA / MQA；
以及多 block 的 split-KV + reduce 变体（FlashDecoding 思路）。

**30 秒**：分页 KV cache 管显存 → 自研 `IPluginV3` decode 注意力插件 → 长上下文下按上下文维切分，
每步 decode 的延迟增长斜率降 88%，生成结果与单趟实现完全一致。

**2 分钟**

- `BlockAllocator` + `PagedKVCache`：定长块池 + free list，序列按需分配，`block_tables` 定形；
  decode 网络对**每一层**都接收一对 `key_cache_<layer>` / `value_cache_<layer>`。
- `PagedAttentionPlugin`：单层注意力，online softmax **单趟扫描**，通过块表跨物理块寻址；
  两种输入形态（5 输入 = 只用 cache；7 输入 = 额外接当前 token 的 K/V）。
- 长上下文 split-K：stage-1 按 `(head, batch, split)` 出局部 `m/l/acc`；
  stage-2 用 max-trick 做**保序**归约；`num_splits == 1` 退化回单趟路径。

**深挖（难在约束，不难在算法）**

1. **宿主侧拿不到"当前上下文多长"**：`block_tables` 形状固定、`context_lens` 在设备上，又不允许 D2H。
   因此 stage-1 的 grid 恒为上限，超出有效片数的 block 必须立即返回、不读不写，
   并给空片写哨兵（`m = -inf, l = 0`）；否则归并会读到未初始化显存。
2. **归约不能用原子累加**：原子加不保序 → 结果不可复现。保序归约是硬要求。
3. **workspace 必须按 `DynamicPluginTensorDesc.max` 报上界**：动态轴在 `desc.dims` 里是 `-1`，
   按它算出的 workspace 偏小、kernel 照写就是越界写。
4. **分片规则与 workspace 布局只有一份实现**（`__host__ __device__` 头文件），kernel 与 host 用例共用；
   各写一份就会在"改片数 / 改 head_size"时静默错位。
5. **插件反序列化后不会再执行 `configurePlugin`**，而 head 配置是从输入形状推导的 →
   运行期入口必须以当前形状为准重新自洽，否则正常形状会被误判成"改了 head 配置"。

**会被追问**

- *为什么切上下文维而不是 head / batch 维*：改动前 grid 是 `(num_heads, batch)` = 12 个 block，
  本机 24 个 SM 一半闲置，且每块串行扫完 976 个位置，有效带宽只有峰值的约 3%。
- *片数怎么定*：每片目标 128 个位置、上限 8 片；再加会被每位置的 block 归约与归并发射吃掉收益；
  短上下文自动退成 1 片。
- *怎么保证可复现*：两阶段归约，stage-2 用 max-trick 保序，不用原子累加；分片边界确定性
  （均分、余数摊给前几片）。
- *`supportsFormatCombination` 只能读 `inOut[0..pos]`*：越界读未初始化描述符会让所有格式组合被判不支持，
  报 `could not find any supported formats...`，且只有真机暴露。
- *每层必须独立 cache 张量*：共用一张会让每层都去读第 0 段，2 层差 `1.2e-3`、12 层完全失真。
- *写 cache 与推进长度必须拆开*：按"每层一次"的接口推进长度会推 `n_layer` 倍，第 3 个新 token 起发散。
- *包含 FlashAttention 吗*：**包含 FlashDecoding，不包含经典 FlashAttention**。decode 侧是
  online softmax 单趟扫描 + 上下文维 split-K + 两阶段归约（= FlashDecoding）；
  **prefill 侧没做** FlashAttention/FMHA，那里是显式 `MatMul → mask → Softmax → MatMul` 子图。
  口径：借用了它的数值技巧（online softmax），没借用它的 IO 分块技巧——query 长度 1 时 Q 维没有可分的块。
- *TRT 自己提供 FlashDecoding 吗*：**stock TRT 没有这个名字的 API 或开关**。它给的是 `IAttention`
  （头文件原话"尽量用单个融合 kernel"，选哪个 tactic 不透明）；`IKVCacheUpdateLayer` 的
  `KVCacheMode` **只有 `kLINEAR`** → 分页/块表布局没有。显式控制多 block 是 TensorRT-LLM 那套库的事。
  我们自研的真实理由是"分页布局 + 并行控制面"，不是"TRT 没有 attention"。

**证据**：MHA / GQA / MQA / `batch > 1` / 空片 / `context_len == 0` 对 double 参考全绿；
斜率降 **88.85%**（kernel 级、同二进制 A/B）/ **88.20%**（端到端、同 session ABBA，两次复现）；
ctx = 1024 时每层每步 **0.919 → 0.126 ms**；与单趟差异 `max_abs 3.58e-07` / `max_rel 1.52e-06`；
三档 prompt 的生成 token 与单趟**完全一致**；GPT-2 端到端 8/8 贪心 token 与 HF 一致。

### 2.2 主线 B：精度契约与双引擎

**对应 TRT-LLM**：context phase 与 generation phase 分离（双引擎或单引擎两组 optimization profile）；
executor 侧的 buffer / workspace / engine 管理。

**30 秒**：同一个 builder 出三种切面；prefill 走显式子图、decode 走分页插件；
但真正难的是**弱类型网络下精度不能假定、只能查询**，以及缓存与图版本的代次契约。

**2 分钟**

- 同一个 builder 支持 `kSingle` / `kPrefill` / `kDecode` 三种切面，共用全部权重与算子代码，
  只有注意力分支不同：prefill 吃动态 `seq_len`，decode 吃动态 `batch`。
- 引擎构建指纹：stage / 精度 / 源文件身份（size+mtime）/ 全部建图参数 / 开关 /
  TRT·CUDA 版本 / 手工图版本；缺指纹一律重建，命中打 `Engine cache hit`。
- 执行侧：缓冲按请求扩容，**解码循环内零分配**；RAII 管 stream 与显存；
  所有 CUDA / TRT 调用走错误检查宏。

**深挖**

1. **弱类型网络下 K/V 与 logits 的输出精度由 TRT 决定**，不由传入精度决定（FP16 引擎实测声明成 FP32）。
   按"配置精度"分配缓冲 → 4 字节写进 2 字节 → 越界写；而 FP32 下 2 与 4 恰好一致，
   **永远不会暴露**。修法：一律**向引擎查询声明精度**。
2. **所有权边界就是契约**：decode 网络对每层都要一对独立 cache 张量；
   cache 宽度是引擎与 KV cache 管理器之间的契约 → 加启动闸：decode `key_cache_0` 的声明精度
   与配置不一致就**拒绝构造**，把内存越界变成一条可读的启动错误。
3. **写入 kernel 按"源精度 × 目标精度"四种组合显式分发**（FP32 源写进 FP16 cache 是合法组合，
   假定二者一致则宽度与数值全错）。
4. **TRT 的 tactic 是 timing-based**：同一网络重建后引擎大小实测从 54.2 MB 变为 52.4 MB →
   "改动前后"的性能对比必须用**同一次构建**的引擎。
5. **指纹看不见插件源码的变化**（只看模型/配置的 size+mtime），所以"workspace 需求从 0 变正数"
   这类改动必须**手工 bump 图版本**，否则复用旧引擎就是往 0 字节缓冲里写。
6. **新增诊断输出等于改 I/O 契约**：每个消费方都要多分配并绑定，TRT 对未绑定输出直接拒绝 enqueue
   → 诊断输出默认关闭且走独立引擎路径。

**证据**：prefill logits 对 `ref_output.bin` 相对 `9.19e-07`、cosine 1.0；
decode 单步与 prefill 对应位置在 `1e-5` 下一致；引擎指纹上线后第二遍运行 `Engine cache hit`。

### 2.3 主线 C：量化与判据纪律（"量 A 裁 B"那次最贵的坑）

**对应 TRT-LLM**：quantization 一整套（W8A8 / W4A16、FP8、KV cache 量化、SmoothQuant）。
我们只做了本平台可用的那个子集。

**30 秒**：INT8 走显式 Q/DQ，判据设计成"FP32 余量子集一致率"；
最贵的教训是 scale 取自未折 BN 的权重、量化对象是已折 BN 的权重——"尺子量 A、裁剪 B"。

**2 分钟**

- **实现**：INT8 走**显式 Q/DQ**（对称、zero_point 恒 0；pre-quantized DQ 形态让 ONNX 从
  44.7 MB 降到 13.3 MB），绕开 TRT 10.12 起废弃的 `kINT8` / `IInt8Calibrator` 路线。
- **判据**：这批图 FP32 自身不稳（多数样本 margin 很小），绝对差被少数样本放大（实测 21.6），
  所以**故意不用绝对误差**；主判据是"FP32 余量子集上的 top-1 一致率 ≥ 90%"，实测 **12/12**。
  绝对误差界需要带真值标签的验收集，那是独立立项的事。
- **怎么自证在跑 INT8**：Q/DQ 不受传入精度影响，也不会体现在"看 I/O 精度"上 →
  必须用 `IEngineInspector` 读逐层的 Format/Datatype。

**深挖（最贵的一次）**

- 现象：per-channel 整网退化到 21.9%，最小图却证明写法没错。
- 根因：脚本用**未折 BN** 的权重算 scale、却量化**已折 BN** 的权重 → 逐通道错配
  （折叠系数 0.05~19.9）→ **16.19% 的 int8 权重被 clamp 饱和**。
- 破局：ONNX 官方参考实现 + **文件级证据**（数被 clamp 到 ±127 的比例：3.9% / 16.2% / 0.04%）；
  改 scale 来源后余量子集一致率 **54.5% → 100%**。
- 代价：3 次真机往返 + 11 条假设被否证；结案后的同类排查在本机 CPU 上 4 分钟跑完。
  教训：**先找不依赖后端行为的证据**。

**诚实边界**：LLM 侧的 INT4 / FP8 与 KV cache 量化**未做**，这是与 TRT-LLM 最大的技术差距之一；
本机是 GTX 1660 Ti（TU116，无 Tensor Core），那条收益曲线本来也拿不到。

### 2.4 怎么选主线（按面试官背景切换）

- 对方做**推理框架 / kernel**：讲 A（并行度、访存、约束）。
- 对方做**量化 / 精度**：讲 C（判据设计 + 最贵的坑）。
- 对方做**运行时 / 系统工程**：讲 B（精度契约、缓存代次、I/O 约束）。
- 时间只够讲一条：讲 A——证据最全，而且能自然带出"先量、再优化、再验证"的方法论。
- 三条之间的桥：A 解决"怎么快"→ B 解决"快了以后精度与缓存怎么不出错"→
  C 解决"精度判据本身怎么才算数"。

### 2.5 次要点速答（被问到再展开）

**RMSNorm / RoPE 融合插件**

- RMSNorm：一行一 block，FP32 用 `float4`、FP16 用 8×half 向量化，不能整除时回退标量；
  权重作第二输入（走常量折叠路径，无动态显存开销）。
- RoPE：half-split 约定（与 HuggingFace `apply_rotary_pos_emb` 对齐，独立脚本交叉验证最大差异 **0**），
  `rotary_dim` 可小于 `head_size`，`position_ids` 是网络**输入**而非常量（支持非连续位置与 KV cache 场景）。
- 两个坑：**部分旋转时必须显式写出尾部**，否则输出与输入分离的 kernel 会留一段未初始化区间；
  **旋转配对方式必须与参考统一**——plan 原文写"相邻两维配对"，而 HF 是 half-split，选错会得到
  "看着差不多但就是不对"的结果。head 配置只从形状推导，只序列化 `rotary_dim` / `base` 两个属性。

**采样内核**

- greedy / top-k / top-p 三个设备侧 kernel，token 直接写回显存；k/p 是 per-batch 张量；
  随机源为 host seed + device Philox（确定性可复现）；Top-P 生产路径 = 保留 CUB 排序 + 行内并行。
- **参数要按 per-batch 设计**，否则做连续批处理就得改接口；当前 runner 仍是 `batch = 1`。
- **绕开整段排序的尝试失败了**：手写 fast top-k 正确性与旧实现逐 token 相同，但慢 **6~9 倍**。
  结论：看起来更聪明的算法在真实访存模式下可能更差，必须实测才能下结论。已撤出生产，代码保留作 A/B 入口。
- 证据：Top-P 对旧实现的配对净收益 **12.6 / 22.4 / 15.7 / 17.6×**；
  采样占 decode 每步 greedy **1.25%** / top-k **17.7%** / top-p **20.1%**（短上下文），
  据此判定 greedy 下不值得继续优化。
- *是投机解码吗*：**不是**。采样器是解码的"最后一米"——从一行 logits 里选下一个 token，
  因为 logits 与 token_ids 都是**设备指针**，结果直接写回显存，解码循环内零 D2D/D2H 同步。
  投机解码是另一件事（草稿模型 + 一次前向验证），我们**没做**；它最硬的前置是 **KV 回退**，
  而分页 cache 目前是"只追加"（`AppendDecodeStep` 只推进长度、无截断）。
- *选几个 logit / 是不是归一化后取最大值*：**只有 greedy 取最大值，而且它连归一化都不用**
  （softmax 单调，`argmax(logits)` 等价）。top-k 是**恰好 k 个**，在前 k 个内算软概率后随机抽一个；
  top-p 是**个数可变**，取"累计概率首次 ≥ p 的最短前缀"再随机抽。归一化是**隐式**的：
  不逐项除 `total`，而是把阈值乘上 `total`（`u · Σ`）。

**Tokenizer 与工具链**

- BPE Tokenizer：byte-level BPE（GPT-2），`Load` 入参是目录（`vocab.json` + `merges.txt`）；
  与 HF 逐 token 全等，含 21 个样本的 golden 与 SHA256 自证。
- SentencePieceTokenizer：已实现并完成源码嵌入，但**零用例、无资产、无调用方，正确性从未裁决**——
  面试时只说"接口预留"，别当成已验证能力。
- 转换 / 探针脚本：`hf_to_mini_trt_llm.py`、`onnx_to_mini_trt_llm.py`、`quantize_resnet18.py`、
  `add_probe_outputs.py`、`inspect_onnx.py`、`inspect_engine.cpp`、`make_tiny_onnx.py`、
  `make_tokenizer_golden.py`；`tools/validate/` 做验收集规格与两侧报告交叉比对；
  四个 `profile_*` target + `run_profile.sh` + `summarize_nsys.py`，都带 `--self-test`。

### 2.6 归属追问："哪些是你自己写的"

**一句话口径**：**算子（layer）是 TensorRT 提供的，怎么拼、精度怎么定、缓存与调度怎么管全部是自己写的**；
第三方只有 safetensors 解析、CUB 排序、nvonnxparser、SentencePiece 与 TRT 运行时本身。

| 环节 | 我们手写 | 复用第三方 / TRT |
|---|---|---|
| 权重导出 | 转换 / 产图脚本、`weight_map` 设计 | safetensors 文件格式 |
| JSON / 配置 | 递归下降解析器、`ModelConfig` | — |
| 权重加载 | dtype 转换、转换缓存、名字映射 | `safetensors-cpp` 解析 + mmap |
| 建图 | 算子组合、命名、精度决策、全部建图护栏 | TRT 的层算子（MatMul / Slice / Softmax / Gather / Conv…） |
| Plugin | 三个 `IPluginV3` 封装 + 全部 CUDA kernel | TRT 的插件框架 |
| Profile / 缓存 | profile 规则、构建指纹 sidecar | TRT 的构建与 tactic 选择 |
| KV Cache | 分页布局、块分配器、写入 kernel、长度推进 | — |
| 采样 | greedy / top-k / top-p kernel、Philox PRNG | CUB 分段排序 |
| 执行 | 绑定、形状设置、符号查询、循环编排 | TRT Runtime / `ExecutionContext` |
| Tokenizer | byte-level BPE | SentencePiece（已嵌入但**未验证**） |

**TRT 有 `IAttention`，prefill 为什么还手搭**：这是
`docs/dev/REQ-004-gpt2-native/phase2_development_plan.md` **D1 的明确取舍**——备选就是
`IAttention(causal=true)`，当时选手搭的三条理由是：可按算子对拍定位差异、把同一份 K/V 张量复用给
KV Cache、以及 Turing 上融合 kernel 的可用性需真机确认。代价是两条路各自建引擎。
要不要换成 `IAttention` 属**待测量决定**（Phase 3 D1，前置 PF-7）。

### 2.7 与 TRT-LLM 的能力对照表

| TRT-LLM 技术 | 状态 |
|---|---|
| Paged KV cache / block manager | 已实现 |
| generation phase attention（MHA / GQA / MQA + 多 block 归约） | 已实现（自研插件 + split-K） |
| fused RMSNorm / RoPE | 已实现 |
| prefill / decode 分离 + 双 optimization profile | 已实现（双引擎） |
| 采样内核（greedy / top-k / top-p） | 已实现 |
| 构建指纹 / workspace / 缓冲复用 | 已实现 |
| 量化 | 仅 CV 侧 INT8 Q/DQ；LLM 量化未做 |
| INT4 / FP8 / KV cache 量化 | 未做（硬件与 API 代次都不支持） |
| in-flight / continuous batching | 未做（runner 有意限定 `batch = 1`） |
| TP / PP / EP 多卡 | 未做（单卡环境） |
| CUDA Graph / chunked prefill / 投机解码 / LoRA / beam search | 未做 |

---

## Part 3｜第三步：系统问题

### 3.1 支持什么模型

**先给事实**：`EngineBuilder` 注册表里只有两个名字——`gpt2` 与 `resnet18`
（`src/core/builder.cpp:127-128`）；插件接受的 dtype 只有 FP32 / FP16
（`paged_attention_plugin.cu:528`）。

| 档 | 模型 | 状态 |
|---|---|---|
| **已交付 + 真机验证** | GPT-2（原生路径 FP32；ONNX 路径只能做 prefill 推理与对拍）、ResNet18（ONNX / 原生 / `CVRunner`；FP32 / FP16 / INT8） | 有数值证据 |
| **同族可直接用，但未实测** | GPT-2 家族的其它 checkpoint（`distilgpt2` / `medium` / `large` 等） | 结构上同算子集，转换脚本按 HF 的 `transformer.` 前缀映射；**从未实测，不能当结论** |
| **写 builder 后理论可接** | ① LLaMA / Qwen2 / Phi 类（RMSNorm + RoPE + GQA/MQA + SwiGLU）——算子件全在；② BERT 类 encoder、ViT（当**纯前向**用） | 缺的是 builder / forward-only runner；LLaMA 系还差 SentencePiece 路径验证 |
| **需要补 kernel 才能接** | Mistral 的 sliding window；T5 / BART / Whisper 的 cross-attention；MoE 动态路由；量化 KV cache | 不是"改 builder"能解决的 |
| **接不了** | 依赖 FP8 / INT4 Tensor Core 的路线（本机 TU116 无 Tensor Core）；70B 级 | 与 builder 无关的硬约束 |

**会被追问**

- *现在能部署哪些*：两个——GPT-2（推荐 FP32）与 ResNet18（FP32 / FP16 / INT8）。
  其余都是"框架有件、没有模型"。
- *不考虑 builder，理论上能接哪些*：判据是"**算子集是否落在现成件里**"。能接的是标准
  **decoder-only Transformer** 整族（LayerNorm 或 RMSNorm + 绝对位置或 RoPE + MHA/GQA/MQA + MLP），
  以及把 Transformer 当**纯前向**用的 encoder / ViT。接不了的是要靠新 kernel 的东西。
- *LLaMA 为什么现在跑不了*：缺 **builder**，不是缺 kernel——RMSNorm 插件、RoPE 插件、
  支持 GQA 的分页注意力都在，SiLU/SwiGLU 用 `sigmoid × mul` 就能表达；但没有
  `model_type: llama` 的 builder，GPT-2 builder 走的是 LayerNorm + 学习式位置编码。
  另外 SentencePiece 路径未验证。这题的关键是**把"算子件齐"和"能跑"分开**。
- *MoE 为什么难*：先说准——TRT 有 `IGatherLayer`（`kELEMENT` / `kND`）与
  `IScatterLayer`（`kELEMENT` / `kND`），索引可以是张量，按 token 路由**能表达**，
  不是"不支持"。难的是工程代价：① 每个专家收到多少 token 是**动态**的 → 要么固定容量 + padding，
  要么接受丢弃策略；② 每次前向要从大块专家权重里 gather，访存代价高（TRT 的 gather 不会像手写
  kernel 那样分块复用）；③ token 重排 / 去重与容量溢出策略在建图标量图里很难表达。
  所以准确说法是"**能表达，但要模型级设计**"，退路是"全部专家都算 + 掩码"（费算力）。
- *Mistral 的 sliding window*：decode 侧的分页注意力按 `context_len` 扫全量，插件里**没有窗口参数**。
  窗口不只是 mask 问题（decode 每步只来 1 个 token，窗口决定"看多少历史"），
  所以长上下文下与全注意力**不等价**——要改 kernel。
- *70B 为什么"不现实"*：两条叠加——① decode 网络**每层一对** KV cache 输入（这是真 bug 修出来的设计），
  80 层 = 160 个 I/O，引擎体积与构建时间都成规模瓶颈；② 单卡显存。不是"慢"，是装不下。
- *和 TRT-LLM 在模型覆盖上的差距*：它覆盖几十种架构 + TP/PP + 量化 + in-flight batching；
  我们只有 2 个模型、单卡、`batch = 1`。诚实说法是"**同一套机制的极小复刻**"。
- *给你三个月先接哪个*：**LLaMA 类**——算子件最齐（RMSNorm / RoPE / GQA 分页注意力都在），
  收益最大（能顺带把 SentencePiece 与 GQA 的生产路径验通）。前提是**先造参考再动实现**，
  不要拿模型去试 tokenizer。

### 3.2 怎么快速部署一个新模型

**现状的"快"来自三处设计**：`config.json` 里的 `weight_map` 把权重改名与建图解耦；
`ModelRegistry` 让新增模型只注册一个 builder、不改构建入口；引擎构建指纹让重复部署直接命中缓存。

**接一个新模型要动的六步**

1. 转换脚本产 `config.json` + `model.safetensors`（含 `weight_map`）；
2. 写一个 `IModelBuilder` 并按名注册（三种切面：single / prefill / decode）；
3. 若架构与 GPT-2 不同，prefill 的注意力子图要自己拼；
4. runner 层适配（`LLMRunner` 只收 decoder-only + token id；encoder 类要 forward-only runner）；
5. tokenizer 侧验证（BPE 已验，SP 未验）；
6. 与参考实现（HF / PyTorch）做分层对拍，阈值写出处。

**部署侧还没做的**：引擎文件与 TRT 版本强绑定，端侧要么预构建分发、要么在设备上重建；
没有批量转换工具链，也没有服务化 API。

### 3.3 端侧 vs 集群：两个方向各要做什么

> **先给定位**：本机是 WSL2 + GTX 1660 Ti 移动版，**既不是集群也不是端侧的代表硬件**
> （端侧主力是 Jetson / 手机 NPU，集群是 A100/H100 多卡）。所以这题不声称"适配了哪个场景"，
> 只讲两件事：现有设计踩在哪一边，往两个方向各要补什么。

**现有设计的取向**：单卡、`batch = 1`、分页 KV 按需分配不预留、无 Tensor Core 只能靠访存与并行优化——
这些更接近**端侧单请求低延迟**的形态，但硬件不代表端侧。

**上端侧要做什么**

- **量化**：LLM 侧现在只有 FP32 / FP16（且 FP16 端到端 NaN 未解决），INT8 / INT4 完全没有，
  而这正是端侧最吃的一环；CV 侧的 INT8 Q/DQ 路径可以当模板复用。
- **显存预算**：分页块池本身就是端侧方向的设计（按需分配、不按最大长度预留），
  要补的是块池大小按设备显存配置与回退策略。
- **启动与分发**：引擎构建很贵，指纹缓存保证不重建，但**引擎与 TRT 版本强绑定**——
  端侧需要预构建分发或 OTA 时重建，这块没做过。
- **后端单一**：全栈依赖 CUDA + TensorRT，只覆盖 NVIDIA 路线；NPU / OpenCL 后端为零。
- **硬件现实**：端侧主力（如 Jetson Orin）反而**有 Tensor Core**，低精度收益曲线与本机完全不同，
  本机的性能结论不能直接外推。

**上集群要做什么**

- **调度**：continuous / in-flight batching 现在是空的，`LLMRunner` 是单序列 →
  要加请求队列、每请求独立 block table、完成序列的块回收与抢占（preemption）。
- **好消息**：分页注意力插件本身**已经支持 `batch > 1`**（真机用例验过），
  槽位留在 **runner 与调度层**，不用推倒 kernel。
- **并行**：TP / PP 需要按 head 维切 MatMul 与 attention + NCCL 通信；
  而且"每层一对 KV cache 输入"的设计在 80 层模型上会变成 160 个 I/O，上多卡前要先重做这个 I/O 契约。
- **服务化**：HTTP / gRPC 前端、请求级采样参数（k/p 已经是 per-batch 张量，接口留好了）、
  流式返回（早停需要 device 侧标志，而不是每步同步）。
- **低精度**：只有到 `sm_80+` 才谈得上 FP8 / INT4 的收益，而这条路线本机根本跑不了。

**要支持 `batch = 8` 和 continuous batching 具体改什么**：`LLMRunner` 从 `batch = 1` 扩开——
多序列 block 分配与回收、每序列各自的 `context_lens`、per-batch 采样参数、请求级调度。
KV cache 的追加接口已按"每层一次、只推进一次长度"设计，扩 batch 时不用推倒。

### 3.4 生产化缺口

| 能力 | 状态 |
|---|---|
| in-flight / continuous batching、runner 的 `batch > 1` | 未做（有意限定） |
| LLM 侧量化（INT8 / INT4 / FP8）、KV cache 量化 | 未做 |
| ONNX 路径接进 `LLMRunner` | 未做（两条路 I/O 契约不同：INT64 vs INT32、无 `position_ids`） |
| prefill 阶段的 paged attention kernel / 单引擎含 prefill | 未做（prefill 走原生子图 + 独立引擎） |
| TP / PP 多卡、服务化 API、CUDA Graph、投机解码、LoRA、beam search | 未做 |
| CV 动态分辨率 | 未做 |
| SentencePiece 路径的正确性 | 未验证（零用例） |
| 显存池 | 实测触发不成立，不排期 |

**"这个项目最大的不足"**：没有生产级调度与 LLM 量化；FP16 端到端未解决；
单卡 `sm_75` 拿不到 Tensor Core 收益，性能天花板不高。**主动承认边界，不要硬撑。**

---

## Part 4｜第四步：最难的问题与怎么解决

### 4.1 最难的技术点：长上下文 split-K

分页注意力是第一个难点，split-K 是它的进阶。**难的不是算法，是五条约束**：

1. 宿主侧拿不到上下文长度 → grid 恒为上限；
2. 空片必须写哨兵（`m = -inf, l = 0`），否则归并读到未初始化显存；
3. 归约不能用原子加（不保序 → 不可复现）；
4. workspace 必须按 `.max` 报（动态轴在 `desc.dims` 里是 `-1`）；
5. 分片规则与布局只能有一份实现。

顺带能带出方法论：先量出"attention 占每步约 80% 且 grid 只有 12 个 block"，
再决定切上下文维，最后用同二进制 A/B 与 ABBA 复现验证。

### 4.2 最难查的三个 bug（类型不同，任选一个讲）

- **最贵**：INT8 per-channel 整网退化——scale 取自**未折 BN** 的权重、量化对象是**已折 BN** 的权重
  （"量 A 裁 B"），3 次真机往返 + 11 条假设被否证，最后靠 ONNX 官方参考实现 +
  **文件级证据**（数被 clamp 到 ±127 的比例：3.9% / 16.2% / 0.04%）破局；
  改 scale 来源后 54.5% → 100%。详见 2.3。
- **最隐蔽**：FP16 端到端非法访存——缓冲按"配置精度"分配，而弱类型网络下 TRT 把 K/V 与 logits
  声明成 FP32；**FP32 下 2 与 4 恰好一致，所以永远不暴露**。修法 = 一律向引擎查询声明精度。详见 2.2。
- **最难下结论**：并列行上的 argmax——判据本身在那行不良定义（分歧比要分辨的间距大 5~7 倍），
  翻绿翻红只取决于误差符号。修法是加"可判性"前提（只在 `margin > 2×` 两侧最大差的行上要求全等），
  **数值判据一个字没动**。

### 4.3 怎么保证性能结论可靠

1. 先量本机的**判别下限**：单发采样器 kernel 34~156 µs、跨 session 同一实现漂 ±23%；
   `PROGRESS.md` §5.13b 给出约 **±400~600 µs** 的判别下限。
2. 四条协议：同二进制 A/B（改动前后两版都编进去）、同轮 ABBA 交替、
   斜率口径 `(T4−T1)/3`、报中位数与 p25/p75。
3. 拿不到 kernel 时间线时用**上下文长度扫描**反推占比：prompt 4 / 256 / 960 对应每步
   3.055 / 6.062 / 14.705 ms，两段斜率近似线性（11.93 vs 12.28 ms per 1000 位置）。
4. **尺子本身也要验**——它抓到过一次测量错误：全等输入让排序变快 2.7 倍，
   把 sampler 占比从 17.7% 假降到 4.75%。
5. 结论：**小于判别下限的差异不改代码，只能写"无显著差异"**，不得按百分比自动接受。

### 4.4 怎么保证数值正确

- **参考实现唯一、自带断言、有 host 侧 meta-test**（踩过"参考漏 batch 维把正确 kernel 判错"）。
- 带 batch 的算子强制覆盖 `batch > 1`。
- **阈值写出处、不跨精度复用、禁止放宽阈值换绿**。例：ResNet18 的 FP32 阈值 `1e-4` 是
  "最大无关差异（BN 折叠 `1.9e-5`、CPU/GPU `7.6e-6`）的约 5 倍"；FP16 另立两档（0.1 / 0.05），
  因为它有 0.018~0.027 的纯舍入噪声底，套 FP32 的尺子没有意义。
- **精确判据只在可判行生效**（见 4.2 第三条）。
- 与期望不符时**只允许三种动作**：继续查（写"原因未知"）/ 证明期望值本身错并给独立依据 /
  标"已知失败 + 原因未知"保持红色。
- 诊断输出要自证：说明比较对象（哪两个张量、布局与形状）、D2H 范围落在同一段分配内、
  绝对差与相对差同时给。

---

## Part 5｜第五步：用 Codex 怎么开发这个项目

> 这条线在项目里有一手材料：根目录 `AGENTS.md`（硬约束）、
> `.agents/skills/trt-inference-engineering/`（阶段链与模板）、`docs/dev/REQ-*/`（19 个条目的 artifact）、
> `docs/TROUBLESHOOTING.md`（50 条排查记录）。

### 5.1 怎么用 Codex 开发

- **契约先行**：把硬件限制（`sm_75`、无 Tensor Core、TRT 10.15.1）、设计规范、权限边界、
  证据纪律全部写进 `AGENTS.md`，每个会话自动加载——**约束写在文档里，不靠每次口头重复**。
- **阶段链**：用技能 `trt-inference-engineering` 把工作切成
  P0 需求 → P1 分析 → P2 设计 → P3 评审 → P4 基线 → P5 实现 → P6 测试 → P7 基准 → P8 文档 → P9 面试，
  每阶段有 checklist、Gate 判据和 artifact 模板；Gate 决定"设计是否被接受"。
- **条目化**：每个功能一个 `docs/dev/REQ-NNN-*/` 目录，按需产出
  `requirement` / `analysis` / `design` / `review` / `benchmark_before` / `benchmark` /
  `test_plan` / `summary` / `interview_notes` + `STATE.md`（当前状态与 blocker 的唯一落点）。
  已推进 19 个条目：15 个完整件套，4 个进行中（`REQ-016`~`REQ-019`，只有四件套）。
- **开工前"计划对账"四问**：有没有计划文档 / 本次任务与文档是否一致 / 不一致先改文档再动代码 /
  反向也要查（文档与代码矛盾属于必须当场修的问题，并写明原因）。

### 5.2 难点在哪

- **权限与"自动推进"的冲突**：技能里写"P0 目标明确即自动进 P1"，但项目规则要求
  **没被点名批准就不许开工**——两者并行，Gate 管设计，批准管开工。工具很能干，
  所以更要划清批准范围：批准写计划 ≠ 批准改代码，批准改代码 ≠ 批准跑真机。
- **长篇上下文会丢**：会话被压缩后，agent 会**照过期文档改回去**。项目里真实发生过两次：
  计划文档写"6 个输入"、代码已是"每层一对 cache 输入"（照文档改回去就恢复一个真 bug）；
  以及口头确认过的接口改动没有回到文档里。解药就是计划对账 + "一个事实只有一个落点"。
- **文档与代码漂移**：文档一旦落后于代码，下一个会话就会照文档把修正改回去。
- **最大的诱惑是"放宽阈值换绿"**：调大阈值、删断言、把断言降级成打印、跳过用例——
  本项目最难的几个 bug 全是数值正确性问题，放宽阈值恰好只对它们生效，所以被明令禁止。
- **性能数字容易自我欺骗**：本机判别下限约 ±400~600 µs，低于它的观测只能说"无显著差异"。

### 5.3 怎么保证正确性与可靠性

- **测试分层与门控**：265 条注册用例按 host / GPU / 性能分层；host 用例缺资产返回 77 → Skipped。
  GPU 用例统一走 `MINI_TRT_SKIP_IF_NO_CUDA`；真机带 `MINI_TRT_REQUIRE_GPU=1` 时**跳过即失败**。
  刻意不把"无 GPU 跳过"整体改成失败，否则 CI 永远不绿。
- **资产闸门**：`tools/check_skips.py` + `tests/data/expected_skips.txt`（基线为空），
  防止用例"静默少跑"。
- **判据纪律**：阈值旁边写出处、不跨精度复用；精确判据只在可判行生效；
  与期望不符时只允许三种动作（继续查 / 证明期望值错 / 保持红色）。
- **留痕**：`TROUBLESHOOTING.md` 只增不改、每条带编号（现象 / 用过的命令 / 关键证据 / 根因 /
  回归防护）；`PROGRESS.md` 只留结论并指过去，避免交接文档膨胀成流水账。
- **提交粒度**：每个 commit 只完成一个 Phase 子任务，禁止跨多个无关功能；
  无 Phase 归属的文档改动单独成笔。
- **参考实现自证**：参考实现唯一 + 自带断言 + host 侧 meta-test，防止"参考写错把正确实现判错"。

### 5.4 这套流程抓到的真实案例（最有力的举证）

- **TS-043**：计划里的真机命令把 `ctest -R` 写成 gtest 过滤器语法 → **一条用例都不跑，退出码仍是 0**。
  此后判据改成"输出了几行 `Test #N`"，不是退出码。
- **TS-040**：`PROGRESS.md` 写"16 处存在性门全部去掉"，实际还剩 3 处——文档结论与代码不一致，
  当场补齐并写明原因。
- **TS-009**：参考实现漏了 batch 维，把正确的 kernel 判成错的——所以参考实现自己也需要 meta-test。
- **TS-025**：路径 helper 返回了文件而不是目录 → 新用例**静默跳过**——这是资产闸门立项的直接原因。
- **TS-046**：INT8"量 A 裁 B"整案，3 次真机往返 + 11 条假设被否证，最后靠文件级证据破局（详见 2.3）。
- **TS-036**：真机第一红，定位为"判据在并列行上不是良定义"——改的是测试参考，kernel 一个字节没动。
- **TS-042**：同一个 kernel、同一组参数，两套量法差 2.5 倍——定位到"全等输入让排序路径变快"，
  属于尺子本身的问题。

---

## 附录 A｜完整功能清单（现状盘点）

> 事实出处：`PROGRESS.md` §3「已完成的部分」。本节只做归纳，不新增结论。
> 模型支持面见正文 3.1。

**框架与构建层**

- 配置驱动 + 模型注册表：JSON `config.json` → `ModelRegistry` → 具体 `IModelBuilder`，新增模型只需注册一个 builder。
- 两条构建路径：原生（Safetensors + TRT Network API 手搭）与 ONNX（解析 + 插件）。
- 三种构建切面：`kSingle` / `kPrefill` / `kDecode`，共用同一份建图代码，只有注意力分支不同。
- 动态 shape：LLM 分 Prefill / Decode 两组 optimization profile；CV 支持动态 batch。
- Engine 封装：set input shape / bind tensor address / enqueue / benchmark / 多 profile 切换。
- 引擎构建指纹缓存：stage、精度、源文件身份、全部建图参数、开关、TRT·CUDA 版本、手工图版本都进指纹；
  缺指纹一律重建，命中打 `Engine cache hit`。
- 边界精度契约：缓冲按引擎**声明**的精度分配；decode cache 的声明精度与配置不一致时拒绝构造。

**权重与配置**

- Safetensors 加载，接受 F32 / F16 / BF16（BF16 先还原 FP32 再降到目标精度）。
- `weight_map` JSON 映射 + C++ 硬编码兜底；`const` 读取路径 + 按 key 隔离的转换缓存。
- 自研极简 JSON 解析器，含 `\uXXXX` 与 UTF-16 代理对（GPT-2 词表全是这种转义）。

**LLM 路径（GPT-2）**

- 原生建图：embedding + N 层 Transformer（LayerNorm、QKV、注意力、FFN、GELU-tanh、残差）+ LM Head。
- prefill 注意力：`MatMul → causal mask → Softmax → MatMul` 显式子图，吃动态 `seq_len`。
- decode 注意力：自研 PagedAttention 插件，吃动态 `batch`。
- `LLMRunner` 自回归循环：Prefill → Decode → 采样，全程显存驻留，循环内零 H2D/D2H。
- 生成选项：`max_new_tokens`、greedy / top-k / top-p、固定 seed、EOS 后截断；
  `temperature != 1.0` 显式报错而不是静默忽略。
- 已验证：FP32 贪心 8 token 与 HF 逐 token 一致，prefill logits 相对偏差 `9.19e-07`。
- 限制：`batch = 1`（有意限定）；EOS 只能在循环后截断。

**自定义插件（`IPluginV3`）**

- PagedAttention：decode 阶段，支持 MHA / GQA / MQA，两种输入形态（5 输入 / 7 输入含当前 token 的 K/V），
  online softmax 单趟扫描，长上下文 split-K + 两阶段保序归约。
- RMSNorm：FP32 `float4` / FP16 8×half 向量化，不能整除时回退标量；权重作第二输入。
- RoPE：half-split 约定，`rotary_dim` 可小于 `head_size`，`position_ids` 作网络输入。
- 全部支持动态 shape、`serialize` / `deserialize`；`PluginRegistry` + `REGISTER_TENSORRT_PLUGIN` 双注册。

**KV Cache**

- `BlockAllocator`：定长块池 + free list，`Allocate` / `Free` / `NumFree`。
- `PagedKVCache`：序列预留、prefill 覆盖写、decode 追加；写与推进长度**拆成两个接口**
  （`AppendDecodeKV` 只写 / `AppendDecodeStep` 一次写全部层并只推进一次长度）。
- 写入 kernel 按"源精度 × 目标精度"四种组合显式分发。

**采样**

- greedy / top-k / top-p 三个设备侧采样 kernel，token 直接写回显存。
- k/p 是 per-batch 张量；随机源为 host seed + device Philox（确定性可复现）。
- Top-P 生产路径 = 保留 CUB 排序 + 行内并行；旧单趟实现与两级实现保留作同二进制 A/B 入口。
- Top-K 手写快速路径正确但性能不达标，已撤出生产（代码保留）。
- `nucleus_cutoff.hpp` 是交叉点定位的唯一实现，host / device 共用。

**Tokenizer**

- `BaseTokenizer` 抽象 + 多模态扩展位。
- BPE Tokenizer：byte-level BPE（GPT-2），`Load` 入参是目录（`vocab.json` + `merges.txt`）；
  与 HF 逐 token 全等，含 21 个样本的 golden 与 SHA256 自证。
- SentencePieceTokenizer：已实现并完成源码嵌入，但零用例、无资产、无调用方，正确性**从未被裁决**。

**CV 路径（ResNet18）**

- 原生建图：零插件（conv / relu / add / maxpool / GAP / flatten / gemm，BN 已在导出时折叠）。
- ONNX 路径：复用同一套 profile / 精度语义与 I/O 契约校验。
- `CVRunner`：输入契约 = NCHW float `[0,255]`，归一化在 Runner 内完成；维度与 batch 范围
  **向引擎查询**；失败返回空 vector / 零统计。
- FP32 / FP16 均已验证（argmax 全一致、无 NaN）。
- INT8：Q/DQ 显式量化（对称、`prequant_dq` 形态，ONNX 44.7 MB → 13.3 MB）；
  判据 = FP32 余量子集一致率（实测 12/12）。
- 限制：只支持动态 batch，不支持动态分辨率。

**工具链与工程基建**

- 转换：`hf_to_mini_trt_llm.py`、`onnx_to_mini_trt_llm.py`、`quantize_resnet18.py`、`add_probe_outputs.py`。
- 探针 / 夹具：`inspect_onnx.py`、`inspect_engine.cpp`、`make_tiny_onnx.py`、`make_tokenizer_golden.py`。
- 校验：`tools/validate/`（验收集规格、分层统计、ONNX 官方参考实现、两侧报告交叉比对）。
- 性能：四个 `profile_*` target + `run_profile.sh`（一键 nsys / ncu、记录温度与时钟）+
  `summarize_nsys.py`（三层分桶、`--api` 分配统计），都带 `--self-test`。
- 265 条注册用例（GoogleTest 源码嵌入），按 host / GPU / 性能分层；host 用例缺资产返回 77 → Skipped。
- 沙箱 268 条 / 0 失败；真机整轮 267 条 / 1 红 / 0 跳过 / 301.72 s（唯一红 = 按设计的 FP16 复现器）。

## 附录 B｜全量数字表

| 项 | 数值 |
|---|---|
| GPT-2 FP32 贪心 8 token | 与 HF 逐 token 一致 |
| GPT-2 prefill logits | `max_abs 9.92e-05`、相对 `9.19e-07`、cosine 1.0 |
| ONNX vs 原生 | 相对 `5.66e-07` |
| ResNet18 FP32 | ONNX vs torchvision `9.5e-06`；原生 vs ONNX `1.07e-06` |
| ResNet18 INT8 | 余量子集一致率 12/12；整体 38.3% |
| split-K | 斜率降 88.85%（kernel）/ 88.20%（端到端）；`max_rel 1.5e-06` |
| split-K 单点 | ctx = 1024 每层每步 0.919 → 0.126 ms |
| Top-P sampler | 配对净收益 12.6 / 22.4 / 15.7 / 17.6× |
| sampler 占比（短上下文） | greedy 1.25% / top-k 17.7% / top-p 20.1% |
| 测试 | 沙箱 268 条 / 0 失败；真机整轮 267 条 / 1 红 / 0 跳过 / 301.72 s |
| 规模 | `src` + `include` 约 9.9k 行；265 条用例；50 条排查记录 |
| 环境 | TRT 10.15.1、CUDA 12.6.85、sm_75、C++17 |

## 附录 C｜剩余问答（未进主线的条目）

1. **为什么用 `IPluginV3`？**
   TRT 10.x 的推荐接口，能力按 Core / Build / Runtime 拆分；V2 属 legacy，长期维护成本高。

2. **短上下文下 split-K 有退化吗？**
   有，实测 **+1.877%**，机制是每层多一次归并发射。**没有**把它写成硬判据：
   原措辞里的"漂移"有三种读法且前两种结论相反，曾议的"≤2%"也没有出处。
   判据被证伪就不假装它成立，只作观测项。

3. **FP16 为什么 NaN？为什么不修？**
   弱类型 FP16 引擎下端到端 NaN，出现 NaN 的层随构建变化（0/1/2），而激活幅值远未到 65504；
   已排除 LayerNorm 精度、激活函数、残差膨胀。`sm_75` 无 Tensor Core，低精度收益有限，
   因此按政策登记为**已知限制**并保留红色复现器，另立三条解决路线（`REQ-018`）。

4. **给你一周，你优化什么？**
   先量。长上下文 attention 已切过、sampler 只占约 3%，凭直觉改 kernel 大概率白干。
   真要动，先跑 ONNX vs 原生的可复现对照，再决定要不要做 prefill 侧 attention。

5. **（判断类）有人说 `mini_trt_llm/` 这一层多余，代码应该挪到根目录，你怎么答？**
   **架构上同意**：这一层原本是为了与两个示例工程并列，而那两个目录已随 Phase 5 下线，前提没了。
   **工程上不动**：代价是全局 592 行路径引用；其中三类特别贵——① `AGENTS.md` 把目录树与
   `./build/mini_trt_llm/tests/mini_trt_llm_tests` 写死（改它需要授权）；② 16 份冻结文档会集体失真；
   ③ 有一处**会静默失效**：测试按路径找 `mini_trt_llm/tools/make_tiny_onnx.py`，搬完候选全不匹配
   → 走 `GTEST_SKIP`，而那类跳过**故意没进资产闸门**。
   结论口径：**"该做，但收益是美观、代价是全局路径与权威文档失真——属于排序靠后的事"**；
   真要做就先把闸门扩到那类跳过，让损失可见，再搬。

6. **怎么防止"跑了但没跑"？**
   见 5.4 的 TS-043：`ctest -R` 收的是正则，写成 gtest 过滤器语法会一条都不匹配**且退出码仍为 0**。
   判据是"输出了几行 `Test #N`"，不是退出码。

7. **排查怎么留痕？**
   `TROUBLESHOOTING.md` 只增不改、每条带编号（现象 / 用过的命令 / 关键证据 / 根因 / 回归防护）；
   `PROGRESS.md` 只留结论并指过去。

8. **你踩过最贵的坑？**
   见 2.3 / 4.2：INT8 的"量 A 裁 B"。整条排查在本机 CPU 上 4 分钟跑完，
   而此前同类排查花了 3 次真机往返。教训：**先找不依赖后端行为的证据**。

9. **文档和代码不一致怎么办？**
   一个事实只有一个落点（决策 / 过程 / 开放项分家）；每次开工先做"计划对账"四问；
   发现矛盾当场修，并写明原因。
