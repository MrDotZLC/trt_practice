# Analysis

<!--
只描述当前系统，不写设计方案。
本版（2026-10-02）逐条对照代码核实过上一版的结论；被核实推翻的三处列在 §Existing Limitation 的
注释里，并已改写。核实范围：core / kv_cache / plugins / sampler 与相关测试。
-->

## Current Architecture

自回归推理被拆成**两个引擎**：一个吃整段 prompt（prefill），一个吃单个 token（decode）。
两者由同一个运行时对象持有，按 `Prompt → 写 K/V → 取首 token → 循环 decode` 的顺序驱动。

- **prefill 引擎**：输入 `input_ids` / `position_ids`（`[B, S]` INT32，**批维与序列维都是动态的**），
  除 `logits`（`[B, S, V]`）外**每层再导出一对 K/V**（`[B, kv_heads, S, head_size]`），
  供运行时代码写进分页 cache。注意力是显式拼的子图：`MatMul → 加一个因果 mask 常量 → Softmax → MatMul`。
  **mask 是常量 `[1, 1, n_positions, n_positions]`，按运行期的 `S×S` 动态切片**——
  图里没有 padding mask 输入。
- **decode 引擎**：输入 `input_ids` / `position_ids`（`[B, 1]`）、`block_tables`
  （`[B, blocks_per_seq]` INT32，第二维静态）、`context_lens`（`[B]` INT32），外加每层一对
  `key_cache_N` / `value_cache_N`（4-D `[num_blocks, block_size, kv_heads, head_size]`，
  **第 0 维由模型超参推出，见 §Existing Limitation 7**）；每层再导出一对当前 token 的 K/V。
  注意力走自研 PagedAttention 插件（含 split-K 两阶段变体）。
- **profile 与批维约定**：构建侧把所有动态张量的**第 0 维**解释为 batch，其余动态维解释为序列维。
  默认范围：prefill `batch = 1/1/4`、`seq = 1/64/512`；decode `batch = 1/1/4`、
  非 batch 动态维**固定为 1**（decode 的 `input_ids` 也直接声明成 `[B, 1]`）。
- **引擎缓存指纹**：stage / 精度 / 来源 / 源文件（`size + mtime`）/ 建图数值参数 / 建图开关 /
  TRT 与 CUDA 版本 / 一个**手工维护的 `graph_version`**。最后一项是唯一的"代码代次"表达——
  **建图代码本身不进指纹**，所以任何改动图行为的改动都要手工把它 +1，否则会静默复用旧引擎。

## Module Structure

| 模块 | 现状职责 | 是否已具备批量能力 |
|---|---|---|
| 运行时（`LLMRunner`） | 驱动 prefill/decode 循环、管理缓冲、采样、结果收集 | **否**（全部按批量 = 1 写死） |
| 分页 K/V 缓存（`PagedKVCache`） | 块池、序列登记、块表与长度镜像、写 K/V | **接口上已具备**（按 `seq_id` 分配/回收，有批内顺序表） |
| 块分配器（`BlockAllocator`） | 块级 free-list | 是（与批量无关；失败**抛异常**） |
| PagedAttention 插件 | 单层 decode 注意力 | **是**（`grid.y = batch`，batch 取自 query 的 dim 0；split-K 的 workspace 偏移也按 batch 算） |
| 位置编码填充 kernel | `context_lens[b] → position_ids[b]` | **是**（按 batch 索引） |
| 采样器 | Greedy / Top-K / Top-P | **接口上已具备**（参数是 per-batch 指针，greedy 的 grid 就是 `batch_size`），但调用方只传 1 行 |
| 模型构建器 | 建 prefill/decode 两张图 | **是**（batch 维声明为 `-1`；只有"批内 prompt 不等长"才需要改图） |

## Data Flow

一次生成请求当前的数据流：

```text
input_ids[1,S0]
   ↓ 绑定 + 设形状
prefill 引擎 ──→ logits[1,S0,V]（取最后一行）
              └→ k_layerN/v_layerN[1,kv_heads,S0,D]
   ↓ 常驻诊断：同步 D2H 拷贝末行 logits + 逐层扫 K/V 找 NaN（见 §Existing Limitation 8）
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

- **序列登记**：每次调用先释放上一次的序列、再为一个固定序列号登记
  `prompt_len + max_new_tokens` 个 token 的块，然后整块上传块表与长度镜像。
- **元数据缓冲的生命周期（本版已核实并改写）**：登记序列时会按**当前批大小**重建块表 / 长度缓冲的
  host 镜像并调用设备分配；但设备分配在"当前容量已够"时**直接复用，不重新 cudaMalloc**。
  因此：**批大小不变时，跨请求的设备指针是稳定的；只有批大小增长才会 `cudaFree + cudaMalloc`
  并让指针变化**。而 decode 的每一步都会重新绑定块表 / 长度 / 每层 cache 的地址，
  所以"指针失效导致静默读旧地址"在当前代码里**不会发生**——它依赖的是"每步重绑"这个事实，
  而不是"指针不变"这个假设。
- **元数据上传**：块表 / 长度以 host 镜像整块拷到设备；此后 decode 的推进**只在设备端**发生，
  host 镜像不再同步（避免循环内 H2D）。
- **结束**：循环后一次性把 token 拷回主机并同步，再做 EOS 截断。

## Relevant Code Path

| 位置 | 与本次相关的行为 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp` | 运行时配置（层数 / 头数 / 块大小 / 块池 / 块表宽度）、缓冲成员、单一序列号常量 |
| `mini_trt_llm/src/core/llm_runner.cpp` | 形状设置、缓冲分配、prefill 绑定与写 cache、常驻诊断、decode 循环、采样、结果收集 |
| `mini_trt_llm/src/kv_cache/paged_kv_cache.cpp` | 序列登记、块表与长度镜像、元数据上传、写 prefill K/V、追加 decode K/V |
| `mini_trt_llm/include/mini_trt_llm/kv_cache/block_allocator.hpp` | 块分配：失败路径是 `throw std::runtime_error`（无空闲块 / 非法 id / 重复释放） |
| `mini_trt_llm/src/core/llm_runner_kernel.cu` | 位置编码填充（已按 batch 索引） |
| `mini_trt_llm/src/plugins/paged_attention_plugin.cu` | decode 注意力（`grid.y = batch`；cache 块维必须等于配置的块大小） |
| `mini_trt_llm/src/core/gpt2_model_builder.cpp` | 声明 `block_tables` / `context_lens` / 每层 cache 输入；prefill 的固定因果 mask |
| `mini_trt_llm/include/mini_trt_llm/core/builder.hpp` | profile 的批量 / 序列默认范围 |
| `mini_trt_llm/include/mini_trt_llm/core/engine_cache.hpp` | 构建指纹的字段与 `graph_version` 语义 |
| `mini_trt_llm/include/mini_trt_llm/sampler/sampler_common.hpp` | 采样参数的结构（`batch_size` + per-batch 的 k / p 指针） |
| `mini_trt_llm/include/mini_trt_llm/utils/memory_pool.hpp` | 设备缓冲语义：容量够则复用，不重复分配 |
| `mini_trt_llm/tests/test_paged_attention_plugin.cpp` | `PagedAttentionSplitKernelTest.MultiBatchWithDifferentContextLens` 覆盖 batch > 1、长度不同（**插件级**，非整链） |
| `mini_trt_llm/tests/test_e2e_dynamic_shape.cpp` | `DecodeAcceptsVaryingBatch` 覆盖 batch = 2 的 decode profile 被接受并成功执行（**合成网络**，非 GPT-2） |
| `mini_trt_llm/tests/test_paged_kv_cache.cpp` | 覆盖跨块边界追加、长度推进、越界拒绝 |

## Existing Limitation

1. **运行时全链路按批量 = 1 写死**：序列号是常量；每次调用先释放再登记同一条序列；
   缓冲形状是 `[1, S]` / `[1, 1]`；logits 取行只认"唯一一行"且行步长按词表宽算；
   结果缓冲只有一条序列；采样参数是单元素缓冲且采样时 `batch_size` 传 1；
   位置填充也传 `batch_size = 1`。
2. **批内 prompt 形状必须一致**：prefill 图的因果 mask 是固定常量，**没有 padding mask 输入**。
   因此一批里各序列的 prompt 长度不同时，不能靠右填充糊过去（填充位置会参与注意力）。
3. **采样参数不可按序列区分**：设备端只有一份单元素参数缓冲（接口本身是 per-batch 的）。
4. **元数据缓冲随批大小重建**：见 §Runtime Flow。批大小增长会触发设备缓冲重分配，
   但这在当前流程里被"每步重绑"掩盖；它不是一个已经被观测到的缺陷，而是一条**尚未被任何用例
   固定下来的隐含依赖**。
5. **无调度**：没有请求队列、没有"运行中增补 / 退出"的路径，也没有"结束后归还块"的独立保证
   （当前只是"下一次调用开头顺手释放上一条"）。
6. **批量上限目前只是声明**：默认 profile 允许 batch ≤ 4；合成网络上已证明 decode profile 能接受
   batch = 2 并成功执行。但 **GPT-2 + PagedAttention + 运行时**这条整链从未以 batch > 1 跑过，
   所以"引擎能否按该批量执行端到端"仍未被任何用例固定。
7. **块池大小与引擎 cache 张量第 0 维之间没有校验，而两者本就是同一个数推出的**：
   引擎把 cache 输入声明为 `[ceil(n_positions / block_size), block_size, kv_heads, head_size]`，
   而 `ceil(n_positions / block_size)` 同时也被用作块表宽度。运行时的物理块池**可以**比它更大
   ——现有用例就是这个状态（小模型宽度 4 / 池 8；真机 GPT-2 两者都是 64）。
   它之所以能跑，是因为插件用运行期算出的偏移在**更长的连续缓冲**里寻址，且 TRT 不校验输入缓冲的
   实际容量。这条"池 ≥ 宽度即可"的关系**没有被写进任何契约，也没有任何断言**。
8. **每请求一次的常驻诊断**：prefill 之后会做两次同步 D2H —— 末行 logits 的少量统计，以及
   **逐层扫描 K/V 找 NaN**。后者在真机 GPT-2、`S = 512` 时约 18 MiB/请求，且随批量线性增长。
   它不在 decode 循环里（不违反硬约束），但会进入吞吐 / 延迟读数。
9. **块分配失败是异常而非返回值**：无空闲块 / 非法块号 / 重复释放都 `throw`，调用方没有捕获。
   在"反复接纳请求"的调度循环里，这条路径会变得更常走。
10. **运行时对两个引擎都用 profile 0**：这只在"prefill 与 decode 由两次独立构建产出、各自只有一个
    profile"的前提下成立（`kSingle` 构建出来的引擎同时挂两组 profile，此时 decode 并不是 0 号）。
    当前没有任何校验拦这件事。
11. **prefill 的 logits 缓冲是显存大头之一**：真机 GPT-2、`S = 512`、FP32 时单条约 98 MiB
    （`512 × 50257 × 4`），且随批量线性增长；相比之下 KV 池在 64 块时总共约 72 MiB。

## Extension Point

- **运行时**：批量入口、每序列状态表、按最大批预分配缓冲 → 改动集中在这里。
- **分页 K/V**：`AllocateSequence` / `FreeSequence` / 批内顺序表 / 块表导出已经就位，
  缺的是"元数据缓冲按最大批预分配"与"按批导出 / 回填"。
- **插件与 kernel**：批量维已经打通（decode 注意力、split-K、位置填充），**本轮不需要改 kernel**。
  **前提**：cache 寻址依赖"池 ≥ 块表宽度且缓冲连续"这条关系（§Existing Limitation 7），
  本轮若要放大池，必须把它显式化。
- **采样器**：per-batch 接口已就位，只差把缓冲从 1 元素扩成 B 元素、并把 `batch_size` 传对。
- **模型构建器**：批量为 `-1`，形状上已支持；只有"批内 prompt 长度不等"才需要加 mask 输入
  ——那属于改图，必须同时 bump 建图版本号（`engine_cache.hpp` 的 `graph_version`），
  否则引擎缓存会判定"仍是最新"而拒绝重建。
- **测试**：kernel 级多批已覆盖；缺的是 runner 级"批量 == 逐条单跑"的对拍。
