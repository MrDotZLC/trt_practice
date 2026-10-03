# P5-S4 接口细化（混合批：packed 输入 + 分段 attention 分派）

<!--
本文件只写"接口形态与设计裁决"，不含产品代码。2026-10-04 新建：作者给出 S4 的口径
（"选择性批处理" = 一个 packed 张量装两相、attention 按段分派两条 kernel），
本文件是 P2 级设计草案，**待作者确认后再进入 P3 复评 / Gate**。
-->

## 0. 路线声明（以及刻意不做的部分）

**采用**：每一步装**一个** packed 张量 —— context 段的全部 token 在前、generation 段的 1 token/行在后，
中间无填充；attention 在一个调用内**按段分派**两条 kernel（context 走 varlen 自注意力，
generation 走分页注意力 + 当前 token 自包含）。**每步一次引擎调用**（不再是 S3 的两段式）。

**为什么顺序是硬约束**：只有"一个张量里两段相邻"才需要"context 在前、generation 在后"，
kernel 才能用一个运行时边界标量把两条路径切开。顺序一乱，分派就得逐 token 查表。

**与 S3 的关系**：S3 的 padding + 两段式保留为**对照 / 回退路径**（AC8 要求可回退、切换不改调用方接口）；
两条路径共用下游契约（写回 / 采样 / 调度策略），数值不保证跨路径逐位相同。

**刻意不做**：

| 不做的事 | 理由 |
|---|---|
| chunked prefill（把一个 prompt 切成多步） | requirement Excluded；S4 只做"两相同批"，不做"同一条序列跨步" |
| 抢占与换出 | requirement Excluded（块不足仍按 D9 留在队列） |
| 服务层形态（流式 Submit/Step） | 项目定位不做服务层 |

## 1. 已核实的既有能力（决定改动面）

| 事实 | 证据 | 对 S4 的意义 |
|---|---|---|
| generation 的注意力**自包含当前 token** | `paged_attention_plugin.cu`：`total_len = context_len + (has_current_token ? 1 : 0)`，`key_new/value_new` 扫完 cache 后作为第 `context_lens[b]` 个位置参与 softmax | generation 段可以直接用 packed 里它自己那个 token 的 K/V，**不需要**先写缓存再算注意力（与 S1/S3 一致） |
| 该插件**只支持 decode** | 同文件：`query` 的 seq_len 必须为 1 | context 段需要**新的 varlen 插件**，不能复用 |
| 缓存追加在图外、按行推进 | `PagedKVCache::AppendDecodeStep(row_count)`（S3 落码） | generation 段只需把"packed 里的第 i 行那个 token"追加进它的缓存行 |
| 写回已带行映射与逐行真长度 | `WritePrefillKV(rows, row_count, row_lengths)`（S3 落码） | context 段复用同一语义，只需把**源寻址**从 `row*stride + t` 换成 packed 的 `cu_seqlens[row] + t` |
| 采样参数、随机步号、finish flag 已是 per-row | `SamplerArgs` 的 `seeds/offsets/eos_hit` + `SchedulerStats`（S3 落码） | 采样侧只需把"行"换成"packed 里的行序"，不需要新机制 |
| 改图必须手工 bump `graph_version` | S3 已留下这条规则（padding_bias 那笔就 bump 过） | S4 的新图必须 bump，且与 S3 的两套图共存 |

## 2. 混合批模型

```text
一步的输入（一个张量，无填充）：
  [ context 段: 本步新入批的 B_ctx 条序列的全部 token ] ++ [ generation 段: 本步在跑的 B_gen 行的 1 个 token ]
  B_total = B_ctx + B_gen，T = Σ(context 段各序列长度) + B_gen
段边界：context_seq_count = B_ctx（标量输入）⇒ kernel 取 num_context_tokens = cu_seqlens[B_ctx]
```

一步的流程（**一次引擎调用**）：

```text
① retire / ② admit：与 S3 完全相同（活跃表、压实、D9 预算）——本设计不重写调度策略
③ 打包：context 行在前（顺序 = 准入顺序）、generation 行在后（顺序 = 活跃表前缀）
        逐行重建 input_ids[T] / position_ids[T] / cu_seqlens[B_total+1]
④ 一次调用：attention 插件按段分派（context → varlen；generation → 分页 + 自包含当前 token）
⑤ 采样：按"每序列末位 = cu_seqlens[i+1]-1"聚集 logits → [B_total, V] → 采样（per-row offset/seed/eos）
⑥ 图外两块 K/V 工作：context 段写回缓存（按行映射 + 逐行真长度）；generation 段追加 1 token/行
```

## 3. 输入契约（图的边界）

图上输入是 **token id**（不是 hidden），所以 packed 化的落点是：

| 输入 | 形状 / dtype | 含义 |
|---|---|---|
| `input_ids` | `[T]` INT32 | context 段按序列相接；随后 generation 段每行 1 个 token |
| `position_ids` | `[T]` INT32 | context 段第 i 条 = `0..L_i-1`；generation 段第 j 行 = 该序列当前的 `context_lens[j]`（**推进前**的值，与 S3 的 `LaunchFillPositionIds` 同源） |
| `cu_seqlens` | `[B_total+1]` INT32 | 第 i 条序列的 token 区间 = `[cu_seqlens[i], cu_seqlens[i+1])`；generation 段的区间长度恒为 1 |
| `context_seq_count` | `[1]` INT32 | 段边界（context 段的序列数）；`num_context_tokens = cu_seqlens[context_seq_count]` 由 kernel 自取，不需要第二个标量 |
| `block_tables` | `[B_total, W]` INT32 | 只有 generation 段会读；context 段不读（自注意力用 packed 自己的 K/V） |
| `context_lens` | `[B_total]` INT32 | 同上，只有 generation 段读（**推进前**的值） |
| `key_cache_{i}` / `value_cache_{i}` | 每层一段 | generation 段读分页缓存；context 段不读 |

**去掉 `padding_bias`**：packed 无填充，掩码由 `cu_seqlens` 分段表达。两套图的输入契约不同 →
S3 / S4 各自一套图（见 §7）。

## 4. 段边界与映射（**两个必须先想清的点**）

1. **packed 顺序 ≠ 缓存行序**。作者要求 context 段在前；而缓存行序由 S2 的"新序列追加在尾部"决定
   （S3 的写回与压实都建立在它上面）。因此必须有显式映射：
   `packed_row → cache_row`（写回 / 追加用）与 `packed_row → result_slot`（采样与结果用）。
   **不变量 4 的口径随之改写**：引擎（packed）行、缓存行、采样参数、结果段之间靠**显式映射数组**同源，
   不再依靠"下标天然相同"。
   *为什么不改缓存行序*：那会动 S2 的压实契约与 `AllocateSequence` 的尾部追加，代价远大于一个映射数组。
2. **末位定位统一**：context 段与 generation 段"第 i 条序列的末位"都是 `cu_seqlens[i+1]-1` ——
   采样前的聚集只需一条规则覆盖两段（S3 里是 `(row, L-1)` 与 decode 的逐行两种）。

## 5. K/V 与写回

- 图按层导出 packed 的 `k_layer{i}` / `v_layer{i}`：`[T, H_kv, D]`（源）→ 目标仍是分页缓存。
- **context 段**：第 i 条序列把 `[cu_seqlens[i], cu_seqlens[i+1])` 的 token 写进它的缓存行第 `0..L_i-1` 位，
  并把这些行的 `context_lens` 设为 `L_i`（逐行真长度，S3 的 `row_lengths` 语义不变）。
  写回 kernel 的源寻址从 `row * tokens + t` 改成 `cu_seqlens[row] + t`（**契约改动，要进本文件**）。
- **generation 段**：第 i 行把它那 1 个 token 追加进自己的缓存行（源索引 = `cu_seqlens[i]`），
  写完推进该行长度 1（S3 的"只推进前 row_count 行"语义，行集换成 generation 行集合）。
- **顺序**：写回/追加都必须在**同一次引擎调用的输出**之后、且在下一步调用之前完成（同一 stream 串行）。

## 6. 采样与 logits

- 图输出 `logits`：默认 `[T, V]`（不做图内 gather，先求简单可验证）。
- 采样前按 `cu_seqlens[i+1]-1` 聚集出 `[B_total, V]`，行序 = packed 行序 = **context 行在前、generation 行在后**。
- 采样参数（top_k/top_p/seed）、随机步号（`offsets` = 该行已生成计数）、`eos_hit` 全部按 packed 行序上传；
  结果写进"按请求槽位聚集"的结果缓冲（S3 的机制不变），映射用 §4 的 `packed_row → result_slot`。
- **可选优化（P4 量过再定）**：图内 `Gather(cu_seqlens[i+1]-1)` 直接出 `[B_total, V]`，
  显存从 `T·V` 降到 `B·V`。

**显存账（AC7 的实现层解释）**：padding 路径的 prefill logits 是 `B · S_max · V`（短序列被最长序列拖着），
packed 路径是 `T · V ≈ Σ L_i · V` —— 这正是"不等长批按真实长度计费"在显存上的体现。

## 7. 引擎 / 图 / profile（**本设计最大的一个裁决**）

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 一个混合引擎（采用）** | 每步一次调用：context-only / generation-only / 混合三种步共用同一张图；`context_seq_count == 0` 即纯 generation（context 段为空，kernel 天然跳过） | 省掉 D10 说的"每步第二次调用"；纯 generation 步与今天的 decode 用法一致；代价是插件复杂（一个插件内两份 attention 实现）与缓存输入在纯 context 步也绑定 |
| B. 两个引擎（各自 packed） | context 引擎不带 cache 输入、generation 引擎带 —— 但两段不在同一次调用里，"选择性批处理"退化成"各自 packed" | 保留 S3 的两段式开销；**与作者口径不符**（作者明确两者参与同一个 packed 张量） |

**决策：A。** 段边界是运行时标量，图里没有数据相关的分支；把两段放进一次调用、由 attention 插件按
`context_seq_count` 内部分派，是唯一既能满足作者口径、又不引入"空槽 / 空行"的形态。

**插件形态（子决策，建议 A1）**：

- **A1（建议）**：**一个新插件**，内部按段分派 —— context 段走 varlen 自注意力，generation 段走分页 + 自包含当前 token。
  理由：段边界在设备端；两插件方案要在图里按运行时标量切片再合并（Select/Concat），多一层显存与出错面。
  **它不替代现有 `PagedAttentionPlugin`**：S3 回退路径仍用它。
- A2：两个插件 + 图内按 `context_seq_count` 切片/合并。留作对照（若 A1 的 kernel 复杂度失控）。

**profile**：packed 引擎的"token 维" `T ∈ [1, max_batch × max_prefill_seq_len]`（opt 取典型批 × 典型长），
`B_total ≤ max_batch`，单序列长度仍受 `n_positions` 约束；`graph_version` 必须 bump 并与 S3 的两套图区分。

## 8. 路径开关与回退（AC8）

- `LLMRunner::Config` 加 `prefill_mode`（`kPackedMixed` 默认 / `kPaddedTwoPhase`），**不改调用方接口**。
- 两条路径共用：准入/退出策略、结果收集、采样参数口径、D9 预算。
- 分叉点只有三处：输入打包、注意力（插件/图）、写回/追加的源寻址。
- 引擎按需构建（默认建 packed；回退路径用时再建），两者指纹与 `graph_version` 独立。

## 9. 判据与用例（待实现后补进 test_plan.md）

| 用例 | 判据 |
|---|---|
| `PackedEqualsSequential` | AC1 在 packed 路径内部成立：同一请求同 seed，packed 批跑 == 逐条单跑（逐位） |
| `MixedStepContextAndGeneration` | 同一步里既有新入批的 context 行、又有在跑的 generation 行 → 两者结果都对（这是 S4 的核心场景） |
| `ContextTokensPrecedeGeneration` | 打包顺序违反"context 在前"时**必须报错**（入口校验），而不是静默算错 |
| `CuSeqlensBoundaryCases` | `context_seq_count == 0`（纯 generation）与 `== B_total`（纯 context）两种极端都要正确 |
| `PackedWriteBackMapsCorrectly` | packed 源 + 行映射的写回落到各序列自己的块，未参与的行不动（对应 S3 的守门用例） |
| `PackedShortSequenceNotPenalized` | AC7：同批长度差很大时，短序列的结果不受最长序列长度影响（与 padding 路径的对照） |
| `FallbackSwitchKeepsResults` | AC8：切到 `kPaddedTwoPhase` 后仍全部正确；两条路径各自满足 AC1（不要求跨路径逐位相同） |

## 10. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 打包顺序写错（generation 行混进 context 段） | 静默算错（会被当成"短序列结果不对"） | 入口显式校验顺序 + 用例 `ContextTokensPrecedeGeneration` |
| 一张图里两种 attention 的实现分叉 | 维护成本、回归面翻倍 | 只加**新插件**，不动 `PagedAttentionPlugin`；S3 路径作为对照 |
| packed 的 `T` 上限与 profile 冲突 | 形状越界、静默用错 profile | 构造期校验 `T_max = max_batch × max_prefill_seq_len` 与 profile 一致（沿用 D8 的入口） |
| 首版就上"图内 Gather 末位" | 图复杂、错了难查 | 默认 `logits[T,V]`，Gather 作为 P4 之后再定的优化 |
| 两条路径都要过 AC1 | 测试面翻倍 | 每条路径内部的用例分开跑；跨路径只比"各自成立"，不比逐位 |

## 11. 待作者确认（进入 P3 之前）

1. **requirement.md Excluded 的口径要收窄**：现在写着排除"prefill/decode 混批的调度策略"，
   而 S4 正是"同一次前向内 mix"，需要改成——"排除 **chunked prefill**（同一条序列跨步切块）；
   同一次前向内按段分派 kernel 不属于混批"。**这条属需求变更，由作者改或授权我改。**
2. **design.md 的"每次引擎调用只装一种相"要改成分路径前提**：S3 padding 路径仍要求它；
   S4 packed 路径一次装两相、由 kernel 分段（D12 的覆盖隐患在 packed 下不成立，因为 context 段只含新入批的 token）。
3. **`p5_s3_interface_spec.md` §10 的"不重写调度"要限定**：调度**策略**不重写，
   每步的**调用形态**从两段式变一次调用。
4. **插件形态选 A1（单插件内部分派，建议）还是 A2（两插件 + 图内切片）**。
5. **§4 的映射与不变量 4 口径改写**：包的映射数组成为行号同源的唯一依据 —— 确认可接受。
