# mini_trt_llm 面试材料（项目总结 / 亮点 / 问答）

> **用途与定位**：面向面试的个人项目材料，**不是项目状态文档**，也不参与 `docs/` 的 SSOT。
> 本文**不新增任何技术结论**：所有数字、判据与结论的出处仍是
> `PROGRESS.md`（现状与基线）、`TROUBLESHOOTING.md`（排查过程）、
> `future_iterations.md`（未做与触发条件）。面试口径与文档口径冲突时，以那三份为准。
>
> 更新规则：只在措辞、组织与"怎么讲"上改动；引用实测值时连同出处一起更新，
> 不在此处复制未经实测的数字。

---

## 1. 一句话定位

一个面向 NVIDIA Turing / `sm_75` 的极简 TensorRT 推理框架：把 TensorRT-LLM 的核心机制
（原生建图、Paged KV Cache、自研 PagedAttention 插件、算子融合、设备侧采样、prefill/decode 双引擎）
从零实现了一遍，并用一套带出处的数值与性能判据在真机验证过。

### 30 秒口述版

> 我用 C++ / CUDA 从零写了一个面向 Turing sm_75 的极简 TensorRT 推理框架，把 TensorRT-LLM 的
> 核心机制实现了一遍：原生建图、Paged KV Cache、自研 PagedAttention 插件、设备侧采样器、
> prefill/decode 双引擎。在长上下文下我做了 FlashDecoding 式的上下文维切分，每步 decode 的
> 延迟增长斜率降了 88%，生成结果与单趟实现完全一致。

### 2 分钟口述版

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
5. **收尾**：引擎构建指纹缓存、插件与图版本的手工代次契约、48 条带编号的排查记录。

### 技术栈与规模

C++17 / CUDA 12.6 / TensorRT 10.15.1（`IPluginV3`）/ CUB / SentencePiece / GoogleTest；
Python 侧 torch + onnx + transformers 做参考实现与产图。
`src` + `include` 约 **9.9k 行**（不含测试与第三方），**265** 条注册用例，48 条排查记录。

---

## 2. 功能清单（现状盘点）

> 事实出处：`PROGRESS.md` §3「已完成的部分」（各阶段的交付清单与实测数字都记在那里）；
> 本节只做"面试口径的归纳"，不新增结论。
> 代码现状：`include/` 39 个头文件、`src/` 27 个实现文件、`tests/` 56 个文件
> （含 42 个 `test_*.cpp`）。

### 2.1 框架与构建层

- **配置驱动 + 模型注册表**：JSON `config.json` → `ModelRegistry` → 具体 `IModelBuilder`，
  新增模型只需注册一个 builder。
- **两条构建路径**：原生（Safetensors + TRT Network API 手搭）与 ONNX（解析 + 插件）。
- **三种构建切面**：`kSingle` / `kPrefill` / `kDecode`，共用同一份建图代码，只有注意力分支不同。
- **动态 shape**：LLM 分 Prefill / Decode 两组 optimization profile；CV 支持动态 batch。
- **Engine 封装**：set input shape / bind tensor address / enqueue / benchmark / 多 profile 切换。
- **引擎构建指纹缓存**：stage、精度、源文件身份、全部建图参数、开关、TRT·CUDA 版本、
  手工图版本都进指纹；缺指纹一律重建，命中打 `Engine cache hit`。
- **边界精度契约**：缓冲按引擎**声明**的精度分配（不是按配置假定）；decode cache 的声明精度
  与配置不一致时拒绝构造。

### 2.2 权重与配置

- Safetensors 加载，接受 F32 / F16 / BF16（BF16 先还原 FP32 再降到目标精度）。
- `weight_map` JSON 映射 + C++ 硬编码兜底；`const` 读取路径 + 按 key 隔离的转换缓存。
- 自研极简 JSON 解析器，含 `\uXXXX` 与 UTF-16 代理对（GPT-2 词表全是这种转义）。

### 2.3 LLM 路径（GPT-2）

- **原生建图**：embedding + N 层 Transformer（LayerNorm、QKV、注意力、FFN、GELU-tanh、残差）
  + LM Head。
- **prefill 注意力**：`MatMul → causal mask → Softmax → MatMul` 显式子图，吃动态 `seq_len`。
- **decode 注意力**：自研 PagedAttention 插件，吃动态 `batch`。
- **`LLMRunner` 自回归循环**：Prefill → Decode → 采样，全程显存驻留，循环内零 H2D/D2H。
- **生成选项**：`max_new_tokens`、greedy / top-k / top-p、固定 seed、EOS 后截断；
  `temperature != 1.0` 显式报错而不是静默忽略。
- **已验证**：FP32 贪心 8 token 与 HF 逐 token 一致，prefill logits 相对偏差 `9.19e-07`。
- **限制**：`batch = 1`（有意限定）；EOS 只能在循环后截断。

### 2.4 自定义插件（`IPluginV3`）

- **PagedAttention**：decode 阶段，支持 MHA / GQA / MQA，两种输入形态（5 输入 / 7 输入含当前
  token 的 K/V），online softmax 单趟扫描，长上下文 split-K + 两阶段保序归约。
- **RMSNorm**：FP32 `float4` / FP16 8×half 向量化，不能整除时回退标量；权重作第二输入。
- **RoPE**：half-split 约定，`rotary_dim` 可小于 `head_size`，`position_ids` 作网络输入。
- 全部支持动态 shape、`serialize` / `deserialize`；`PluginRegistry` +
  `REGISTER_TENSORRT_PLUGIN` 双注册。

### 2.5 KV Cache

- `BlockAllocator`：定长块池 + free list，`Allocate` / `Free` / `NumFree`。
- `PagedKVCache`：序列预留、prefill 覆盖写、decode 追加；写与推进长度**拆成两个接口**
  （`AppendDecodeKV` 只写 / `AppendDecodeStep` 一次写全部层并只推进一次长度）。
- 写入 kernel 按"源精度 × 目标精度"四种组合显式分发。

### 2.6 采样

- **greedy / top-k / top-p** 三个设备侧采样 kernel，token 直接写回显存。
- k/p 是 per-batch 张量；随机源为 host seed + device Philox（确定性可复现）。
- Top-P 生产路径 = 保留 CUB 排序 + 行内并行；旧单趟实现与两级实现保留作同二进制 A/B 入口。
- Top-K 手写快速路径正确但性能不达标，已撤出生产（代码保留）。
- `nucleus_cutoff.hpp` 是交叉点定位的唯一实现，host / device 共用。

### 2.7 Tokenizer

- `BaseTokenizer` 抽象 + 多模态扩展位。
- **BPE Tokenizer**：byte-level BPE（GPT-2），`Load` 入参是目录（`vocab.json` + `merges.txt`）；
  与 HF 逐 token 全等，含 21 个样本的 golden 与 SHA256 自证。
- **SentencePieceTokenizer**：已实现并完成源码嵌入，但零用例、无资产、无调用方，
  正确性**从未被裁决**——面试时只说"接口预留"，别当成已验证能力。

### 2.8 CV 路径（ResNet18）

- **原生建图**：零插件（conv / relu / add / maxpool / GAP / flatten / gemm，BN 已在导出时折叠）。
- **ONNX 路径**：复用同一套 profile / 精度语义与 I/O 契约校验。
- **`CVRunner`**：输入契约 = NCHW float `[0,255]`，归一化在 Runner 内完成；维度与 batch 范围
  **向引擎查询**；失败返回空 vector / 零统计。
- **FP32 / FP16 均已验证**（argmax 全一致、无 NaN）。
- **INT8**：Q/DQ 显式量化（对称、`prequant_dq` 形态，ONNX 44.7 MB → 13.3 MB）；
  判据 = FP32 余量子集一致率（实测 12/12）。
- **限制**：只支持动态 batch，不支持动态分辨率。

### 2.9 工具链

- **转换**：`hf_to_mini_trt_llm.py`（GPT-2）、`onnx_to_mini_trt_llm.py`（ResNet18）、
  `quantize_resnet18.py`、`add_probe_outputs.py`。
- **探针 / 夹具**：`inspect_onnx.py`（图结构 + 子图识别计数）、`inspect_engine.cpp`
  （引擎 I/O 契约）、`make_tiny_onnx.py`、`make_tokenizer_golden.py`。
- **校验**：`tools/validate/`（验收集规格、分层统计、ONNX 官方参考实现、两侧报告交叉比对）。
- **性能**：四个 `profile_*` target + `run_profile.sh`（一键 nsys / ncu、记录温度与时钟）
  + `summarize_nsys.py`（三层分桶、`--api` 分配统计），都带 `--self-test`。
- **参考脚本**：`scripts/ref_resnet18.py`、`ref_rope.py`、`ref_sampler.py`。

### 2.10 测试与工程基建

- **265 条注册用例**（GoogleTest 源码嵌入），按 host / GPU / 性能分层；host 用例缺资产返回 77 → Skipped。
- GPU 门控统一走 `MINI_TRT_SKIP_IF_NO_CUDA`；真机带 `MINI_TRT_REQUIRE_GPU=1` 时**跳过即失败**。
- 参考实现唯一 + 自带断言 + host 侧 meta-test；阈值旁边写出处。
- 48 条排查记录（append-only）与三层文档体系（现状 / 过程 / 开放项）。
- 沙箱 265 条 / 0 失败；真机整轮 264 条 / 1 红（按设计的 FP16 复现器）/ 310 s。

### 2.11 明确未实现的部分

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

**一句话概括**：LLM 侧"从权重到 token"的全链路是通的（原生 + ONNX 两条路、双引擎、分页 KV、
自研注意力与采样），CV 侧从 FP32 到 INT8 也是通的；工程基建比功能面更突出，
缺口集中在 batching / 调度、LLM 量化、服务化这三块。

---

## 3. 面试亮点（对齐 TensorRT-LLM 的技术与难点）

> 组织方式：**TRT-LLM 里的哪个技术 → 我实现了什么 → 难点在哪 → 证据**。
> 前四项是 kernel / 运行时相关的技术件，第五项是量化；工程流程类的内容不放在这里，
> 而是下沉到 §4 的问答（面试官通常以提问的方式考它）。

### 亮点 1｜PagedAttention 全链路：paged KV Cache → generation-phase attention → 长上下文 split-K

**对应 TRT-LLM**：paged KV cache / block manager；generation phase 的 decoder attention
over paged KV cache（即 XQA 那一类 kernel），需支持 MHA / GQA / MQA；
以及多 block 的 split-KV + reduce 变体（FlashDecoding 思路）。

**实现了什么**

- `BlockAllocator` + `PagedKVCache`：块池按需分配，`block_tables` 定形；decode 网络对
  **每一层**都接收一对 `key_cache_<layer>` / `value_cache_<layer>`。
- `PagedAttentionPlugin`（`IPluginV3`）：单层注意力，online softmax 单趟扫描，
  通过块表跨物理块寻址；两种输入形态（5 输入=只用 cache；7 输入=额外接当前 token 的 K/V）。
- 长上下文 split-K：stage-1 按 `(head, batch, split)` 出局部 `m/l/acc`，
  stage-2 用 max-trick 做**保序**归约；`num_splits == 1` 退化回单趟路径。

**难点（这几条能讲深）**

- **宿主侧拿不到"当前上下文多长"**：`block_tables` 形状固定、`context_lens` 在设备上，
  又不允许 D2H。因此 stage-1 的 grid 恒为上限，超出有效片数的 block 必须立即返回、
  不读不写，并给空片写哨兵（`m = -inf, l = 0`）；否则归并会读到未初始化显存。
- **归约不能用原子累加**：原子加不保序 → 结果不可复现。保序归约是硬要求。
- **workspace 必须按 `DynamicPluginTensorDesc.max` 报上界**：动态轴在 `desc.dims` 里是 `-1`，
  按它算出的 workspace 偏小、kernel 照写就是越界写。
- **分片规则与 workspace 布局只有一份实现**（`__host__ __device__` 头文件），
  kernel 与 host 用例共用；各写一份就会在"改片数 / 改 head_size"时静默错位。
- **插件反序列化后不会再执行 `configurePlugin`**，而 head 配置是从输入形状推导的 →
  运行期入口必须以当前形状为准重新自洽，否则正常形状会被误判成"改了 head 配置"。
- `supportsFormatCombination` **只能读 `inOut[0..pos]`**：越界读未初始化描述符会让所有格式
  组合被判不支持，报 `could not find any supported formats...`，且只有真机暴露。
- **每层必须独立 cache 张量**：共用一张会让每层都去读第 0 段，2 层差 `1.2e-3`、12 层完全失真。
- **写 cache 与推进长度必须拆开**：按"每层一次"的接口推进长度会推 `n_layer` 倍，
  第 3 个新 token 起发散。

**证据**

- MHA / GQA / MQA / `batch > 1` / 空片 / `context_len == 0` 对 double 参考全绿。
- 长上下文每步 decode 斜率降 **88.85%**（kernel 级、同二进制 A/B）/
  **88.20%**（端到端、同 session ABBA，两次复现）；ctx = 1024 时每层每步
  **0.919 → 0.126 ms**；与单趟差异 `max_abs 3.58e-07` / `max_rel 1.52e-06`。
- 三档 prompt 的生成 token 与单趟**完全一致**；GPT-2 端到端 8/8 贪心 token 与 HF 一致。

### 亮点 2｜算子融合插件：fused RMSNorm 与 fused RoPE

**对应 TRT-LLM**：把多算子折成一个 kernel 的融合件（fused RMSNorm、fused rotary embedding）。

**实现了什么**

- RMSNorm：一行一 block，FP32 用 `float4`、FP16 用 8×half 向量化，不能整除时回退标量；
  权重作为第二输入（走常量折叠路径，无动态显存开销）。
- RoPE：half-split 约定，`rotary_dim` 可小于 `head_size`，`position_ids` 是网络**输入**而非常量
  （支持非连续位置与 KV cache 场景）。

**难点**

- **部分旋转时必须显式写出尾部**：`rotary_dim < head_size` 时尾部不参与旋转，
  漏写会让"输出与输入分离"的 kernel 留一段未初始化区间——只在部分旋转配置下暴露。
- **旋转配对方式必须与参考统一**：plan 原文写"相邻两维配对"，而 HuggingFace 的
  `apply_rotary_pos_emb` 是 half-split；选错会得到"看着差不多但就是不对"的结果，
  因此先用独立脚本交叉验证（最大差异 `0.000e+00`）。
- **head 配置只从形状推导**，只把 `rotary_dim` / `base` 序列化成属性，避免多出一份
  可能与形状失配的状态。

**证据**：与 HF 参考差异 0；FP16 分支（含 GQA、`batch > 1`、部分旋转）真机通过。

### 亮点 3｜prefill / decode 双阶段，以及弱类型网络下的精度与缓存契约

**对应 TRT-LLM**：context phase 与 generation phase 分离（双引擎或单引擎两组
optimization profile）；executor 侧的 buffer / workspace / engine 管理。

**实现了什么**

- 同一个 builder 支持 `kSingle` / `kPrefill` / `kDecode` 三种切面，共用全部权重与算子代码，
  只有注意力分支不同：prefill 是 `MatMul → causal mask → Softmax → MatMul` 显式子图
  （吃动态 `seq_len`），decode 是 paged 插件（吃动态 `batch`）。
- 引擎构建指纹：stage / 精度 / 源文件身份（size+mtime）/ 全部建图参数 / 开关 /
  TRT·CUDA 版本 / 手工图版本；缺指纹一律重建。
- 缓冲按请求扩容，**解码循环内零分配**；RAII 管 stream 与显存；所有 CUDA / TRT 调用走错误检查宏。

**难点**

- **弱类型网络下 K/V 与 logits 的输出精度由 TRT 决定**，不由传入精度决定（FP16 引擎实测声明成
  FP32）。按"配置精度"分配缓冲 → 4 字节写进 2 字节 → 越界写；而 FP32 下 2 与 4 恰好一致，
  **永远不会暴露**。修法：一律**向引擎查询声明精度**。
- cache 宽度是引擎与 KV cache 管理器之间的契约 → 加启动闸：decode `key_cache_0` 的声明精度
  与配置不一致就**拒绝构造**，把内存越界变成一条可读的启动错误。
- cache 写入 kernel 按"源精度 × 目标精度"四种组合显式分发（FP32 源写进 FP16 cache 是合法组合，
  假定二者一致则宽度与数值全错）。
- **TRT 的 tactic 是 timing-based**：同一网络重建后引擎大小实测从 54.2 MB 变为 52.4 MB →
  "改动前后"的性能对比必须用**同一次构建**的引擎。
- 指纹看不见插件源码的变化（只看模型/配置的 size+mtime），所以"workspace 需求从 0 变正数"
  这类改动必须**手工 bump 图版本**，否则复用旧引擎就是往 0 字节缓冲里写。
- 新增诊断输出等于**改 I/O 契约**：每个消费方都要多分配并绑定，TRT 对未绑定输出直接拒绝 enqueue
  → 诊断输出默认关闭且走独立引擎路径。

**证据**：prefill logits 对 `ref_output.bin` 相对 `9.19e-07`、cosine 1.0；decode 单步与 prefill
对应位置在 `1e-5` 下一致；引擎指纹上线后第二遍运行 `Engine cache hit`（不再重建）。

### 亮点 4｜采样内核

**对应 TRT-LLM**：sampling kernels（greedy / top-k / top-p）与"logits 全程驻留显存"的约束。

**实现了什么**：设备侧 API，token 结果直接写回显存；k/p 是 per-batch 张量；
随机源为 host seed + device Philox（确定性可复现）；Top-P 保留 CUB 排序、只把行内数学块内并行。

**难点**

- **参数要按 per-batch 设计**：否则一旦做连续批处理就得改接口（当前 runner 仍是 batch = 1）。
- **绕开整段排序的尝试失败了**：手写 fast top-k 正确性与旧实现逐 token 相同，但慢 6~9 倍 →
  说明"看起来更聪明的算法"在真实访存模式下可能更差，必须实测才能下结论。
- **性能对比必须控制变量**：同二进制 A/B（改动前后两版都编进去）+ 同轮 ABBA 交替 +
  斜率口径扣掉每窗口固定开销，否则几个百分点的差异根本分不出来。

**证据**：Top-P 对旧实现的配对净收益 **12.6 / 22.4 / 15.7 / 17.6×**；
采样占 decode 每步 greedy 1.25% / top-k 17.7% / top-p 20.1%（短上下文），
据此判定 greedy 下不值得继续优化。

### 亮点 5｜量化（本平台可用的那个子集）

**对应 TRT-LLM**：quantization 一整套（W8A8 / W4A16、FP8、KV cache 量化、SmoothQuant）。

**实现了什么**：INT8 走**显式 Q/DQ**（对称、zero_point 恒 0；pre-quantized DQ 形态让 ONNX
从 44.7 MB 降到 13.3 MB），绕开 TRT 10.12 起废弃的 `kINT8` / `IInt8Calibrator` 路线；
判据设计为"FP32 有余量子集上的 top-1 一致率"。

**难点**

- Q/DQ 不受传入精度影响，也不会体现在"看 I/O 精度"上 → 必须用 `IEngineInspector` 读逐层的
  Format/Datatype 才能自证"确实在跑 INT8"。
- **绝对误差界不能拿来当判据**：这批图 FP32 自身都不稳（多数样本 margin 很小），
  绝对差被少数样本放大到 21.6 → 主判据改用"余量子集一致率"。
- **scale 必须取自被量化的那张张量**：用未折 BN 的权重算 scale、却量化已折 BN 的权重，
  per-channel 会逐通道错配（折叠系数 0.05~19.9）→ 16.19% 的 int8 权重被 clamp 饱和。
  改成"从被量化张量上取"后，余量子集一致率 **54.5% → 100%**。这是整个项目最贵的一次教训。

**诚实边界**：LLM 侧的 INT4 / FP8 与 KV cache 量化**未做**，这是与 TRT-LLM 最大的技术差距之一；
本机是 GTX 1660 Ti（TU116，无 Tensor Core），那条收益曲线本来也拿不到。

### 与 TRT-LLM 的能力对照（把边界说清楚，避免被问倒）

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
| in-flight / continuous batching | 未做（runner 有意限定 batch = 1） |
| TP / PP / EP 多卡 | 未做（单卡环境） |
| CUDA Graph / chunked prefill / 投机解码 / LoRA / beam search | 未做 |

---

## 4. 可能被问的问题（分类 + 答题要点）

### A. 架构设计

1. **为什么用 Paged KV Cache 而不是连续 buffer？**
   块池按需分配、长度增长不搬迁；`block_tables` 是定形张量，宿主侧不需要知道当前长度，
   正好满足"解码循环内零 D2H"。
2. **为什么 decode 用插件、prefill 用原生子图？**
   prefill 是整段算子，TRT 原生融合更好；decode 每步只有 1 个 token，需要跨 block 寻址 +
   online softmax，TRT 没有现成算子。代价是两条路各自建引擎。
3. **为什么每层一对 cache 张量？**
   插件是"单层注意力"实现，只接收一个 4-D cache；共用一张会让每层都读第 0 段——
   这是我踩过的真 bug（2 层差 1.2e-3、12 层完全失真）。
4. **为什么用 `IPluginV3`？**
   TRT 10.x 的推荐接口，能力按 Core / Build / Runtime 拆分；V2 属 legacy，长期维护成本高。
5. **RoPE 为什么是 half-split？**
   与 HuggingFace `apply_rotary_pos_emb` 对齐，独立脚本交叉验证最大差异为 0；
   GPT-NeoX 的相邻配对风格留作属性开关。
6. **采样参数为什么是 per-batch tensor？**
   为将来每个请求独立配 k/p，且避免 host 标量进 kernel；当前 batch = 1，但接口不是为 1 设计的。
7. **引擎为什么要缓存指纹？**
   过去只按路径名复用，代码/配置变了也照用旧引擎，只能靠人记得删 `/tmp`；
   现在指纹不一致或缺失就重建。
8. **指纹覆盖源码，为什么还要手工 bump 图版本？**
   指纹只看模型/配置文件的 size+mtime，看不见插件源码变化。插件版本与图版本是**手工声明的代次**，
   workspace 需求从 0 变正数这类改动必须 bump。

### B. 性能

1. **怎么知道瓶颈在 attention？**
   本机 nsys/ncu 拿不到 GPU kernel 时间线，改用上下文长度扫描：prompt 4 / 256 / 960 对应
   每步 3.055 / 6.062 / 14.705 ms，两段斜率近似线性（11.93 vs 12.28 ms per 1000 位置）。
2. **为什么切上下文维而不是 head 或 batch 维？**
   改动前 grid 是 `(num_heads, batch)` = 12 个 block，本机 24 个 SM 一半闲置，
   且每块串行扫完 976 个位置，有效带宽只有峰值的约 3%。head / batch 维没有可切的空间。
3. **split-K 怎么保证结果可复现？**
   两阶段归约，stage-2 用 max-trick 保序归约、不用原子累加；分片边界确定性（均分、余数摊给前几片）。
4. **片数怎么定？**
   每片目标 128 个位置、上限 8 片；短上下文自动退成 1 片。上限取 8 是因为再加会被
   每位置的 block 归约与归并发射吃掉收益。
5. **为什么 workspace 要按 max 形状报？**
   动态轴在 `desc.dims` 里是 -1，按它算会偏小、kernel 照写就是越界——与 FP16 那次同类错误。
6. **短上下文有退化吗？**
   有，实测 **+1.877%**，机制是每层多一次归并发射。**没有**把它写成硬判据：原措辞里的"漂移"
   有三种读法且前两种结论相反，曾议的"≤2%"也没有出处。判据被证伪就不假装它成立，只作观测项。
7. **Top-P 为什么不干脆绕开排序？**
   试过：fast top-k 正确性逐 token 相同但慢 6~9 倍，已撤出生产。最终"保留 CUB 排序 +
   行内并行"，配对净收益 12.6~22.4×；采样内核净开销已只剩裸读一遍行的约 1.7 倍。

### C. 数值与精度

1. **你怎么证明引擎是对的？**
   参考实现（HF / PyTorch / ONNX 官方实现）+ 分层判据（cosine / 相对界 / 逐位置 argmax /
   逐 token）+ 每条阈值标注出处；参考实现自己还有 meta-test 自证。
2. **GPT-2 FP32 对齐到什么程度？**
   8/8 贪心 token 与 HF 基线一致；prefill logits `max_abs 9.92e-05`、相对 `9.19e-07`、
   cosine 1.0；ONNX 与原生两条路相对偏差 `5.66e-07`。
3. **FP16 为什么 NaN？为什么不修？**
   弱类型 FP16 引擎下端到端 NaN，出现 NaN 的层随构建变化（0/1/2），而激活幅值远未到 65504；
   已排除 LayerNorm 精度、激活函数、残差膨胀。sm_75 无 Tensor Core，低精度收益有限，
   因此按政策登记为**已知限制**并保留红色复现器，另立三条解决路线。
4. **INT8 为什么用"余量子集一致率"而不是绝对误差？**
   这批图 FP32 自身不稳（多数样本 margin 小），整体率主要在测测试集噪声；
   绝对差被少数样本放大（实测 21.6），故意不作判据。主判据是余量子集 top-1 一致率 ≥90%，
   实测 12/12。绝对误差界需要带真值标签的验收集，那是独立立项的事。
5. **阈值凭什么这么定？**
   每条旁边写出处。例：ResNet18 的 FP32 阈值 `1e-4` 是"最大无关差异（BN 折叠 1.9e-5、
   CPU/GPU 7.6e-6）的约 5 倍"；FP16 另立两档（0.1 / 0.05），因为它有 0.018~0.027 的
   纯舍入噪声底，套 FP32 的尺子没有意义。
6. **argmax 这种精确判据遇到并列怎么办？**
   只在"可判行"（margin > 2×两侧最大差）上要求全等；不可判行允许不同，但必须打印并钉到行号登记，
   **新增行即红**。

### D. 工程与流程

1. **你怎么保证性能结论可靠？**
   ① 先量本机的**判别下限**（单发采样器 kernel 34~156 µs、跨 session 同一实现漂 ±23%）；
   ② 四条协议：同二进制 A/B、同轮 ABBA、斜率口径 `(T4−T1)/3`、报中位数与 p25/p75；
   ③ 拿不到 kernel 时间线时用上下文扫描反推占比；④ 尺子本身也要验——它抓到过一次测量错误：
   全等输入让排序"变快" 2.7 倍，把 sampler 占比从 17.7% 假降到 4.75%。
   结论：**小于 ±400 µs 的差异不改代码，先确认尺子够不够用**。
2. **你怎么保证数值是对的？**
   ① 参考实现唯一、自带断言、有 host 侧 meta-test（踩过"参考漏 batch 维把正确 kernel 判错"）；
   ② 带 batch 的算子强制覆盖 `batch > 1`；③ 阈值写出处、不跨精度复用、禁止放宽阈值换绿；
   ④ 精确判据只在可判行生效；⑤ 与期望不符时只允许三种动作：继续查 / 证明期望值本身错
   并给独立依据 / 标"已知失败 + 原因未知"保持红色。
3. **没有 GPU 的 CI 怎么处理 GPU 用例？**
   显式跳过并打印 `cudaGetDeviceCount` 的探测结果；真机带 `MINI_TRT_REQUIRE_GPU=1`，
   此时**跳过即失败**。刻意不把"无 GPU 跳过"整体改成失败，否则 CI 永远不绿。
4. **怎么防止"跑了但没跑"？**
   踩过：`ctest -R` 收的是正则，写成 gtest 过滤器语法会一条都不匹配**且退出码仍为 0**。
   所以判据是"输出了几行 `Test #N`"，不是退出码。
5. **排查怎么留痕？**
   `TROUBLESHOOTING.md` 只增不改、每条带编号（现象 / 用过的命令 / 关键证据 / 根因 / 回归防护）；
   `PROGRESS.md` 只留结论并指过去。
6. **你踩过最贵的坑？**
   INT8 那次：脚本用未折 BN 的权重算 scale、却量化已折 BN 的权重——"尺子量 A、裁剪 B"。
   整条排查在本机 CPU 上 4 分钟跑完，而此前同类排查花了 3 次真机往返。
   教训：**先找不依赖后端行为的证据**。
7. **文档和代码不一致怎么办？**
   一个事实只有一个落点（决策 / 过程 / 开放项分家）；每次开工先做"计划对账"四问；
   发现矛盾当场修，并写明原因。

### E. 反问 / 压力题

1. **这算 TensorRT-LLM 的简化版吗？**
   机制上是核心机制的最小可运行复刻；生产替代不成立（差 batching / 调度、LLM 量化、服务化、多卡）。
   这题要主动承认边界，不要硬撑。
2. **和 vLLM / TRT-LLM 比你的优势？**
   体量小、全链路可读可改、kernel 全自有、数值与性能判据都带出处；劣势是生态、模型覆盖、绝对性能。
3. **要支持 batch = 8 和 continuous batching，你改什么？**
   `LLMRunner` 从 batch = 1 扩开：多序列 block 分配与回收、每序列各自的 `context_lens`、
   per-batch 采样参数、请求级调度。KV cache 的追加接口已按"每层一次、只推进一次长度"设计，
   扩 batch 时不用推倒。
4. **给你一周，你优化什么？**
   先量。长上下文 attention 已切过、sampler 只占约 3%，凭直觉改 kernel 大概率白干。
   真要动，先跑 ONNX vs 原生的可复现对照，再决定要不要做 prefill 侧 attention。
5. **这个项目最大的不足？**
   没有生产级调度与 LLM 量化；FP16 端到端未解决；单卡 sm_75 拿不到 Tensor Core 收益，性能天花板不高。

---

## 5. 速查数字表

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
| 测试 | 沙箱 265 条 / 0 失败；真机整轮 264 条 / 1 红（按设计的 FP16 复现器）/ 310 s |
| 规模 | `src` + `include` 约 9.9k 行；265 条用例；48 条排查记录 |
| 环境 | TRT 10.15.1、CUDA 12.6.85、sm_75、C++17 |

---

## 6. 红线（不要说）

- 别说"实现了 TensorRT-LLM"、"支持 continuous batching"、"支持 batch > 1"——runner 有意限定 batch = 1。
- 别说"支持 INT4 / FP8"、"KV cache 支持量化"——都没有；PagedAttention 只接受 FP32 / FP16。
- 别说"FP16 端到端可用"；提"1 红"时要顺带说明那是**按设计**的 FP16 复现器，
  否则听起来像留了个未修的 bug。
- 别说"做了 ONNX 子图替换"——那两条路只做了**子图识别计数**，替换是未做的开放项。
- 温度 ≠ 1 是**直接拒绝**而非静默忽略；被问到时解释成有意设计（静默忽略会让"调参无效"
  看起来像"模型就是这样"）。
- 不要引用未经自己实测的数字，也不要引用别的项目的性能数字来给自己背书。
