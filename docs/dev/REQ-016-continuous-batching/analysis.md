# Analysis

<!--
只描述当前系统，不写设计方案。
-->

## Current Architecture

自回归推理被拆成**两个引擎**：一个吃整段 prompt（prefill），一个吃单个 token（decode）。
两者由同一个运行时对象持有，按 `Prompt → 写 K/V → 取首 token → 循环 decode` 的顺序驱动。

- **prefill 引擎**：输入 `input_ids` / `position_ids`（`[B, S]` INT32），除 `logits` 外
  **每层再导出一对 K/V**（`[B, kv_heads, S, head_size]`），供运行时代码写进分页 cache。
  注意力是显式拼的子图：`MatMul → 加一个固定因果 mask 常量 → Softmax → MatMul`。
- **decode 引擎**：输入 `input_ids` / `position_ids`（`[B, 1]`）、`block_tables`
  （`[B, blocks_per_seq]` INT32）、`context_lens`（`[B]` INT32），外加每层一对
  `key_cache_N` / `value_cache_N`（4-D `[num_blocks, block_size, kv_heads, head_size]`）；
  每层再导出一对当前 token 的 K/V。注意力走自研 PagedAttention 插件。
- **profile**：构建配置里已经有 `min/opt/max_prefill_batch = 1/1/4` 与
  `min/opt/max_decode_batch = 1/1/4` 的默认值，decode 的序列维固定为 1。
  但运行时只设 `[1, ...]` 并只用 profile 0。

## Module Structure

| 模块 | 现状职责 | 是否已具备批量能力 |
|---|---|---|
| 运行时（`LLMRunner`） | 驱动 prefill/decode 循环、管理缓冲、采样、结果收集 | **否**（全部按批量 = 1 写死） |
| 分页 K/V 缓存（`PagedKVCache`） | 块池、序列登记、块表与长度镜像、写 K/V | **接口上已具备**（按 `seq_id` 分配/回收，有批内顺序表） |
| 块分配器（`BlockAllocator`） | 块级 free-list | 是（与批量无关） |
| PagedAttention 插件 | 单层 decode 注意力 | **是**（`grid.y = batch`，取 batch 自 query 的 dim 0） |
| PagedAttention split-K kernel | 上下文维切分 + 两阶段归约 | **是**（workspace 偏移按 `batch_size` 计算） |
| 位置编码填充 kernel | `context_lens[b] → position_ids[b]` | **是**（按 batch 索引） |
| 采样器 | Greedy / Top-K / Top-P | **接口上已具备**（参数是 per-batch 指针），但调用方只传 1 行 |
| 模型构建器 | 建 prefill/decode 两张图 | **是**（batch 维声明为 `-1`） |

## Data Flow

一次生成请求当前的数据流：

```text
input_ids[1,S0]
   ↓ 绑定 + 设形状
prefill 引擎 ──→ logits[1,S0,V]（取最后一行）
              └→ k_layerN/v_layerN[1,kv_heads,S0,D]
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
- **元数据上传**：块表 / 长度以 host 镜像整块拷到设备；此后 decode 的推进**只在设备端**发生，
  host 镜像不再同步（避免循环内 H2D）。
- **长度记账**：由"追加整步 K/V"的入口负责推进一次，而不是每层各推一次
  （按层推进会把它算成层数倍——本项目真机上踩过，见 `docs/TROUBLESHOOTING.md` + TS-016）。
- **缓冲**：prompt 缓冲、位置缓冲、每层 K/V 缓冲、logits 缓冲都按"批量 = 1 的形状"一次性
  备好，循环内不再分配；容量不足时按需扩容。
- **采样**：采样参数（k / p / seed）是调用级标量，上传一份单元素设备缓冲；采样器自己
  按 logits 的**实际声明精度**读（弱类型网络下由 TRT 决定，见 TS-018）。
- **结束**：循环后一次性把 token 拷回主机并同步，再做 EOS 截断。

## Relevant Code Path

| 位置 | 与本次相关的行为 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp` | 运行时配置（层数 / 头数 / 块大小 / 块池 / 块表宽度）、缓冲成员、单一序列号常量 |
| `mini_trt_llm/src/core/llm_runner.cpp` | 形状设置、缓冲分配、prefill 绑定与写 cache、decode 循环、采样、结果收集 |
| `mini_trt_llm/src/kv_cache/paged_kv_cache.cpp` | 序列登记、块表与长度镜像、元数据上传、写 prefill K/V、追加 decode K/V |
| `mini_trt_llm/src/core/llm_runner_kernel.cu` | 位置编码填充（已按 batch 索引） |
| `mini_trt_llm/src/plugins/paged_attention_plugin.cu` | decode 注意力（`grid.y = batch`；cache 块维必须等于配置的块大小） |
| `mini_trt_llm/src/core/gpt2_model_builder.cpp` | 声明 `block_tables` / `context_lens` / 每层 cache 输入；prefill 的固定因果 mask |
| `mini_trt_llm/include/mini_trt_llm/core/builder.hpp` | profile 的批量默认范围（prefill/decode 各自的 min/opt/max） |
| `mini_trt_llm/tests/test_paged_attention_plugin.cpp` | `...SplitKernelTest.MultiBatchWithDifferentContextLens` 已覆盖 batch = 2、长度不同 |
| `mini_trt_llm/tests/test_paged_kv_cache.cpp` | 覆盖跨块边界追加、长度推进、越界拒绝 |

## Existing Limitation

1. **运行时全链路按批量 = 1 写死**：序列号是常量；每次调用先释放再登记同一条序列；
   缓冲形状是 `[1, S]` / `[1, 1]`；logits 取行只认"唯一一行"；结果缓冲只有一条序列；
   采样参数是标量成员且采样时 `batch_size` 传 1。
2. **元数据缓冲随序列数重分配**：每次登记序列时按当前批大小重新分配块表 / 长度缓冲。
   在"反复接纳请求"的调度循环里，这既是每次 `cudaMalloc`/`cudaFree`，也会让**此前绑定的
   设备指针失效**——必须改成按最大批预分配。
3. **prefill 的批内形状必须一致**：prefill 图的因果 mask 是固定常量
   `[1, 1, n_positions, n_positions]`，**没有 padding mask 输入**。因此一批里各序列的
   prompt 长度不同时，不能靠右填充糊过去（填充位置会参与注意力）。
4. **采样参数不可按序列区分**：设备端只有一份单元素参数缓冲。
5. **无调度**：没有请求队列、没有"运行中增补 / 退出"的路径，也没有"结束后归还块"的保证。
6. **批量上限与配置的耦合未验证**：profile 声明了 max batch = 4，但从未以 > 1 的批量
   跑过端到端，因此"引擎是否真能按该批量执行"目前只是声明。

## Extension Point

- **运行时**：批量入口、每序列状态表、按最大批预分配缓冲 → 改动集中在这里。
- **分页 K/V**：`AllocateSequence` / `FreeSequence` / 批内顺序表 / 块表导出已经就位，
  缺的是"元数据缓冲预分配"与"按批导出"。
- **插件与 kernel**：批量维已经打通（decode 注意力、split-K、位置填充），**本轮不需要改 kernel**。
- **模型构建器**：批量为 `-1`，形状上已支持；只有"批内 prompt 长度不等"才需要加 mask 输入。
- **测试**：kernel 级多批已覆盖；缺的是 runner 级"批量 == 逐条单跑"的对拍。
