# Analysis

<!--
只描述当前系统，不写设计方案（技能 P1 规则）。
本版 2026-10-03 重做：逐条对照代码核实现状，不沿用上一版的结论。
核实范围：core（runner / 建图 / 引擎缓存 / 缓冲）、kv_cache、plugins、sampler 与相关测试。
-->

## Current Architecture

自回归推理被拆成**两个引擎**：一个吃整段 prompt（prefill），一个吃单个 token（decode）。
两者由同一个运行时对象持有，按 `Prompt → 写 K/V → 取首 token → 循环 decode` 的顺序驱动。

- **prefill 引擎**：输入 `input_ids` / `position_ids`（`[B, S]` INT32，批维与序列维都声明为动态），
  除 `logits`（`[B, S, V]`）外**每层再导出一对 K/V**（`[B, kv_heads, S, head_size]`）。
  注意力是显式拼的子图：`MatMul → 加因果 mask → Softmax → MatMul`。
  因果 mask 是**常量** `[1, 1, n_positions, n_positions]`，运行期按当前 `S×S` 切片使用；
  图里**没有 mask 输入**，因此无法表达"这一行是填充、不要参与注意力"。
- **decode 引擎**：输入 `input_ids` / `position_ids`（`[B, 1]`）、`block_tables`
  （`[B, blocks_per_seq]` INT32，第二维静态）、`context_lens`（`[B]` INT32），外加每层一对
  `key_cache_N` / `value_cache_N`（4-D `[num_blocks, block_size, kv_heads, head_size]`）；
  每层再导出一对当前 token 的 K/V。注意力走自研 PagedAttention 插件（含 split-K 变体）。
- **块表宽度与 cache 第 0 维是同一个数**：建图侧用 `cfg.num_blocks()`（即
  `ceil(n_positions / block_size)`）同时声明 `block_tables` 的第二维与每层 cache 的第 0 维
  （`src/core/gpt2_model_builder.cpp` 的 `blocks_per_seq` / `cache_dims`）。
  运行时期的**物理块池可以更大**，见 §Existing Limitation 6。
- **profile 与批维约定**：构建侧把动态张量的第 0 维解释为 batch。默认范围：
  prefill `batch = 1 / 1 / 4`、`seq = 1 / 64 / 512`；decode `batch = 1 / 1 / 4`，
  非 batch 动态维固定为 1。
- **引擎缓存指纹**：stage / 精度 / 来源 / 源文件（`size + mtime`）/ 建图数值参数 / 建图开关 /
  TRT 与 CUDA 版本 / 手工维护的 `graph_version`。**建图代码本身不进指纹**——任何改变图行为的
  改动都必须手工把 `graph_version` +1，否则会静默复用旧引擎。

## Module Structure

| 模块 | 现状职责 | 是否已具备批量能力 |
|---|---|---|
| 运行时（`LLMRunner`） | 驱动 prefill/decode 循环、管理缓冲、采样、结果收集 | **否**（整条路径按批量 = 1 写死） |
| 分页 K/V 缓存（`PagedKVCache`） | 块池、序列登记、块表与长度镜像、写 K/V | **接口上已具备**（按 seq_id 分配/回收，有批内顺序表） |
| 块分配器（`BlockAllocator`） | 块级 free-list | 是（与批量无关；失败**抛异常**） |
| PagedAttention 插件 | 单层 decode 注意力 | **是**（`grid.y = batch`，batch 取自 query 的 dim 0） |
| 位置编码填充 kernel | `context_lens[b] → position_ids[b]` | **是**（按 batch 索引） |
| 采样器 | Greedy / Top-K / Top-P | **接口上已具备**（参数是 per-batch 指针，greedy 的 grid 就是 `batch_size`） |
| 模型构建器 | 建 prefill/decode 两张图 | **是**（batch 维声明为动态；只有"批内 prompt 不等长"才需要改图） |

## Data Flow

当前一次生成请求的数据流：

```text
input_ids[1,S0]
   ↓ 绑定 + 设形状
prefill 引擎 ──→ logits[1,S0,V]（取最后一行）
              └→ k_layerN/v_layerN[1,kv_heads,S0,D]
   ↓ 常驻诊断：同步 D2H 拷回末行 logits + 逐层扫 K/V 找 NaN
   ↓ 逐层写入分页 cache（同时把语境长度推到 S0）
采样 ──→ token_0（留在显存）
   ↓ 循环 max_new_tokens-1 次
decode 引擎（读 cache + context_lens/block_tables）─→ logits[1,1,V]
   ↓ 逐层追加当前 token 的 K/V（整步只推进一次长度）
采样 ──→ token_i
   ↓ 循环结束后一次性 D2H
结果（EOS 截断在主机侧完成）
```

## Runtime Flow

- **序列登记**：每次调用先释放上一次的序列，再为**同一个固定序列号**登记
  `prompt_len + max_new_tokens` 个 token 的块，然后整块上传块表与长度镜像。
- **元数据缓冲的生命周期**：登记序列时按**当前批大小**重建块表 / 长度缓冲的 host 镜像并调用设备分配；
  设备分配在"当前容量已够"时**直接复用，不重新 `cudaMalloc`**。因此批大小不变时跨请求的设备指针
  稳定；只有批大小增长才会让指针变化。而 decode 每一步都会重新绑定块表 / 长度 / 每层 cache 的地址，
  所以"指针失效导致读旧地址"在当前代码里不会发生——它依赖的是"每步重绑"这个事实。
- **元数据上传**：块表 / 长度以 host 镜像整块拷到设备；此后 decode 的推进**只在设备端**发生，
  host 镜像不再同步（避免循环内 H2D）。
- **结束**：循环后一次性把 token 拷回主机并同步，再做 EOS 截断。

## Relevant Code Path

| 位置 | 与本次相关的行为 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp` | 运行时配置（层数 / 头数 / 块大小 / 块池 / 块表宽度）、缓冲成员、单一序列号常量 |
| `mini_trt_llm/src/core/llm_runner.cpp` | 形状设置、缓冲分配、prefill 绑定与写 cache、常驻诊断、decode 循环、采样、结果收集 |
| `mini_trt_llm/src/kv_cache/paged_kv_cache.cpp` | 序列登记、块表与长度镜像、元数据上传、写 prefill K/V、追加 decode K/V |
| `mini_trt_llm/src/kv_cache/block_allocator.cpp` | 块分配：失败路径是 `throw std::runtime_error`（无空闲块 / 非法 id / 重复释放） |
| `mini_trt_llm/src/core/llm_runner_kernel.cu` | 位置编码填充（已按 batch 索引） |
| `mini_trt_llm/src/plugins/paged_attention_plugin.cu` | decode 注意力（`grid.y = batch`；cache 块维必须等于配置的块大小） |
| `mini_trt_llm/src/core/gpt2_model_builder.cpp` | 声明 `block_tables` / `context_lens` / 每层 cache 输入；prefill 的固定因果 mask |
| `mini_trt_llm/include/mini_trt_llm/core/builder.hpp` | profile 的批量 / 序列默认范围 |
| `mini_trt_llm/include/mini_trt_llm/core/engine_cache.hpp` | 构建指纹字段与 `graph_version` 语义 |
| `mini_trt_llm/include/mini_trt_llm/sampler/sampler_common.hpp` | 采样参数结构（`batch_size` + per-batch 的 k / p 指针） |
| `mini_trt_llm/src/utils/memory_pool.cpp` | 设备缓冲语义：容量够则复用，不重复分配 |
| `mini_trt_llm/tests/test_paged_attention_plugin.cpp` | `PagedAttentionSplitKernelTest.MultiBatchWithDifferentContextLens` 覆盖 batch > 1、长度不同（**插件级**，非整链） |
| `mini_trt_llm/tests/test_e2e_dynamic_shape.cpp` | `DecodeAcceptsVaryingBatch` 覆盖 batch = 2 的 decode profile 被接受并成功执行（**合成网络**，非 GPT-2） |
| `mini_trt_llm/tests/test_paged_kv_cache.cpp` | 覆盖跨块边界追加、长度推进、越界拒绝 |

## Existing Limitation

1. **运行时全链路按批量 = 1 写死**：序列号是常量 `kSequenceId = 0`；每次调用先释放再登记同一条序列；
   缓冲形状是 `[1, S]` / `[1, 1]`；logits 取行只认唯一一行且行步长按词表宽算；
   结果缓冲只有一条序列；采样参数是单元素缓冲且调用时 `batch_size = 1`；位置填充也传 `batch_size = 1`。
2. **批内 prompt 形状必须一致**：prefill 的因果 mask 是固定常量，**没有 padding mask 输入**，
   所以一批里各序列 prompt 长度不同时不能靠右填充糊过去（填充位置会参与注意力）。
3. **采样参数不可按序列区分**：设备端只有一份单元素参数缓冲；接口本身是 per-batch 的。
4. **元数据缓冲随批大小重建**：见 §Runtime Flow。批大小增长会触发设备缓冲重分配，但当前被
   "每步重绑"掩盖；它不是一个已观测到的缺陷，而是一条**尚未被任何用例固定下来的隐含依赖**。
5. **无调度**：没有请求队列、没有"运行中增补 / 退出"的路径，也没有"结束后归还块"的独立保证
   （当前只是"下一次调用开头顺手释放上一条"）。
6. **块池大小与 cache 张量第 0 维之间没有校验，而两者本就是同一个数推出的**：建图侧
   `blocks_per_seq = cfg.num_blocks() = ceil(n_positions / block_size)` 同时当块表宽度与 cache 第 0 维。
   物理块池**可以**比它更大（现有小模型用例就是：宽度 4 / 池 8）。它能跑，是因为插件用运行期算出的
   偏移在更长的连续缓冲里寻址，且 TRT 不校验输入缓冲的实际容量。这条"池 ≥ 宽度即可"的关系
   **没有被写进任何契约，也没有断言**（运行时只校验 `max_blocks_per_seq <= num_blocks`）。
7. **每请求一次的常驻诊断**：prefill 之后无条件做同步 D2H——末行 logits 的少量统计，
   以及**逐层扫描 K/V 找 NaN 并打印前 3 层的 max|v|**。后者在真机 GPT-2、`S = 512` 时约 18 MiB/请求，
   且随批量线性增长。它不在 decode 循环里（不违反硬约束），但会进入吞吐 / 延迟读数。
   注意它与 build 期的 `export_diagnostics` **不是一回事**：后者只在建图时挂第 0 层的中间输出、
   会改 I/O 契约，且默认关闭；运行期这段诊断没有任何开关。
8. **块分配失败是异常而非返回值**：无空闲块 / 非法块号 / 重复释放都 `throw`，调用方没有捕获。
   在"反复接纳请求"的调度循环里，这条路径会变得更常走。
9. **运行时对两个引擎都用 profile 0**：这只在"prefill 与 decode 由两次独立构建产出、各自只有一个
   profile"的前提下成立（单引擎构建出来的引擎会同时挂两组 profile，此时 decode 并不是 0 号）。
   当前没有任何校验拦这件事。
10. **prefill 的 logits 缓冲是显存大头之一**：真机 GPT-2、`S = 512`、FP32 时单条约 98 MiB
    （`512 × 50257 × 4`），且随批量线性增长；相比之下 KV 池在 64 块时总共约 72 MiB。

## Extension Point

- **运行时**：批量入口、每序列状态表、按最大批预分配缓冲 → 改动集中在这里。
- **分页 K/V**：按 `seq_id` 的分配 / 回收、批内顺序表、块表导出都已就位，
  缺的是"元数据缓冲按最大批预分配"与"按批导出 / 回填"。
- **插件与 kernel**：批量维已经打通（decode 注意力、split-K、位置填充），本轮不需要改 kernel。
  **前提**：cache 寻址依赖"池 ≥ 块表宽度且缓冲连续"这条关系（§Existing Limitation 6），
  本轮若要放大池，必须把它显式化。
- **采样器**：per-batch 接口已就位，只差把缓冲从 1 元素扩成 B 元素、并把 `batch_size` 传对。
- **模型构建器**：批维已是动态；只有"批内 prompt 长度不等"才需要加 mask 输入——那属于改图，
  必须同时 bump `graph_version`，否则引擎缓存会判定"仍是最新"而拒绝重建。
- **测试**：kernel 级多批已覆盖；缺的是 runner 级"批量 == 逐条单跑"的对拍。

## Terminology

| 模糊名词（出自 requirement.md） | 本项目的可验证定义 | 怎样算没做到 |
|---|---|---|
| 批量 > 1 / 批 | 同一次 prefill 或 decode 调用里，引擎输入的 batch 维 > 1，且这些行各自属于一个独立请求 | 仍是"一次调用只服务一条序列"，多条请求只能串行 |
| 静态批 | 一批请求**同进同出**：批次成员在前向开始前固定，直到整批结束都不变 | 批次成员在中途变化 |
| 最小连续批 | 批次成员可在**两步之间**变化：某序列结束即归还其块并从批内移除，新请求可在此时进入；但**一步之内**不变化（不做抢占、不做换出） | 运行期没有任何"进入 / 退出"路径；或一步之内还允许成员变化 |
| 每序列独立的采样参数 | 同一批内不同行可以使用不同的 k / p / seed，且逐行生效 | 整批共用一个 k / p，或只有第 0 行的参数生效 |
| 跨请求重复调用时不泄漏 | 连续 N 次调用（含中途失败返回）后，空闲块数回到调用前的水位 | 空闲块数单调下降；或失败路径把已分配的块留在池里 |
| 完全一致（数值一致性） | 同一组请求、同一份权重与采样参数下，批量运行的每条序列的每个 token id 与"逐条单独运行"**逐位相同** | 出现任何一位 token id 不同；"top-1 相同但概率不同"不算通过 |
| 逐序列正确（位置编码与语境长度） | 第 b 行用的位置索引等于该序列当前已占用的 token 数，与该行在批内的位置无关 | 任一行用了别的行的长度，或用了批内下标当位置 |
| 空闲块数回到初始值 | 分配器报告的可用块数等于该轮开始前的值（引用相等，不是"接近"） | 存在差值 |
| 无显著差异 | 观测差小于本次口径的判别下限（本机这类测量约 ±400~600 µs，`docs/PROGRESS.md` §5.13b） | 把小于下限的差值写成"更快 / 更慢 X%" |
| 可复现且不自证 | 同 session、同二进制 A/B、逐轮交替测量，报中位数与四分位；且事先声明判别下限 | 只有单次读数；或用实现自己产出对比实现自己；或事后才补判别下限 |
| 分块 / chunk（Included 8、AC9） | **指针式登记**（作者 2026-10-04 定）：定义以 `p5_s5_interface_spec.md` §2 为**唯一来源** —— `chunk_len = min(prompt_len - prompt_done, chunk_limit)`；非末块对齐（恒为 `chunk_limit`）、末块按剩余实际长度。本表只登记指针，不复制正文，避免两处定义漂移 | 把末块变短当成"回退"；或同一输入切出不同的 chunk 组（自适应切法） |
| 末块 / 非末块 | 同上（`p5_s5_interface_spec.md` §2）：同一条规则的两半 | 把末块当异常，或把非末块切成短块 |
| chunk 的绝对位置 | 同上（`p5_s5_interface_spec.md` §2）：chunk 内第 i 个 token 的 `position_ids` = `prompt_done + i`；首 chunk 退化为 `0..L_c-1` | 用段内下标 `i` 当位置（第二块起查错位置表，且不报错） |
| `max_prefill_seq_len`（设计层，2026-10-05） | **指针式登记**：定义以 `p5_s5_interface_spec.md` §2 为唯一来源（含"只在 packed 模式有语义、非 packed 必须留 0"与五条交叉校验） | 在非 packed 路径上给它填值并期待生效；或未声明就让 packed 模式继续跑 |
| `max_positions` / `n_positions`（设计层，2026-10-04） | 同上（`p5_s5_interface_spec.md` §2 末）：位置表长度，与引擎交叉校验（`prompt_len + max_new - 1 <= max_positions`） | 让 `prompt_len + max_new - 1` 越过 `n_positions`（wpe 查表越界读，且不报错） |
| 交叉校验（设计层，2026-10-05） | 同上（spec §2）：**意图由调用方声明、上界由引擎裁决** —— 声明值必须过 profile 上界的检查 | 只信声明值、不查引擎（等于"按配置假定"）；或只用引擎反推、不声明意图 |
| 下游单一 / 声明侧单一（设计层，2026-10-05） | 同上（spec §2 + `design.md` D16 的 Trade-off）：单一是"chunk 策略与 profile 校验共用同一声明值"；声明侧仍是 builder 与 runner **两处** | 把"单一事实来源"读成"只需给一处"，从而不做两侧一致性检查 |
| `policy < cap`（设计层，2026-10-05） | 同上（`design.md` D16 的 Trade-off）：本轮不做第二个旋钮；**触发条件** = "想用更小的 chunk 换调度灵活性、但不想动建图上界" | 现在无数据就引入第二个旋钮；或把 cap 当 policy 用而不写明两者区别 |
