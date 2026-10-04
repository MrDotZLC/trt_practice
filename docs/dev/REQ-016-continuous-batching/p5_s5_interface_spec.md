# P5-S5 接口细化（chunked prefill：长 prompt 跨步分批）

<!--
本文件只写"接口形态与设计裁决"，不含产品代码。2026-10-04 新建：S5 由作者改判立项
（requirement Included 第 8 条、AC9），本文件是 P2 级设计草案，**待作者确认后再进 P3 复评**。
依赖：S4 的 packed 契约与按段分派的 attention 插件（见 p5_s4_interface_spec.md）。
-->

## 0. 路线声明（以及刻意不做的部分）

**采用**：一条序列的 prompt 允许**跨多步**推进（每步只送一段 chunk），用来压住"长 prompt 独占一步"
造成的 decode 尾延迟抖动。chunk 与 S4 的 packed 契约是**加法**：S4 要求"每序列 token 数 ≥ 1"，
S5 只允许它**小于** prompt 长度。

**为什么它是独立里程碑**：S4 的 context 段是**自包含**自注意力（K/V 全来自本次调用）；
chunk 的第二段之后必须**读缓存**里前面已经写好的 prompt K/V —— 这是**第三种计算模式**，
加上每序列的 prompt 进度状态与判据面，工作量与风险都不小。

**刻意不做**：

| 不做的事 | 理由 |
|---|---|
| 调度层的分块优先级策略（例如按 chunk 重排准入顺序） | requirement 的 Excluded 保留项；S5 只做执行层的分块 |
| 抢占 / 换出（分块中途把序列换出去再换回来） | requirement Excluded；分块的进度只在前向内部推进 |
| 跨步 partial-KV 的"回滚" | 块与前向都是一次性追加，没有回滚语义 |

## 1. 已核实的既有能力（决定改动面）

| 事实 | 证据 | 对 S5 的意义 |
|---|---|---|
| packed 契约支持"每序列 token 数 ≥ 1" | S4 落码的 `PackedAttentionPlugin` + `cu_seqlens_ctx`（段内下标） | chunk 只是"某条序列本步的区间**短于**它的 prompt" |
| 段内前缀和与逐行长度已是显式输入 | `cu_seqlens_ctx` / `row_lengths`（S3+S4） | chunk 段的 token 区间不需要新机制 |
| context 段目前**不读缓存** | `PackedAttentionPlugin` 的 context kernel（varlen 自注意力） | 这是 S5 要改的唯一一处计算逻辑 |
| generation 段已经会读缓存（分页 + 当前 token 自包含） | `PagedAttentionSplitKernel`（含 S4 加的行/token 基址） | chunk 的注意力语义与它**同族**：query 数 > 1 而已 |
| 语境长度（`context_lens`）是"已写入的位置数" | `PagedKVCache` 的 host 镜像 + `AppendDecodeStep` 的逐行推进 | chunk 的写回位置 = 该序列**已写入的长度**起（首次为 0） |

## 2. 分块模型

```text
每条序列的状态（活跃表新增两个字段）：
  prompt_done —— 已 prefill 的 prompt token 数（首次准入为 0）
  （prompt_len / generated / max_new 等保持不变）

一步里每条序列的 chunk：
  chunk_len = min(prompt_len - prompt_done, chunk_limit)
            → 非末块恒为 chunk_limit（**对齐**），末块 = 剩余的实际长度
  context 段 = 所有"prompt_done < prompt_len"的序列各自的 chunk（仍按 packed 顺序相接）
  写完 chunk 后：prompt_done += chunk_len
  prompt_done == prompt_len ⇒ 该序列进入 generation 相（从下一步起参与生成段）
```

**`chunk_limit` 的取值（2026-10-04 作者确认：不暴露给调用方，由 profile 上限推导）**：

- 取 `chunk_limit = profile 的单序列上限（`max_prefill_seq_len`）` —— 这是 packed 图里"每行 token 数"
  的天然上界（超过它行就会越出形状），所以"由上限推导"是唯一不需要新配置的口径。
- **它必须落在 fused kernel 支持的常量集合里**；不在集合内时**构造期直接拒绝**（见 §3 的兜底纪律），
  不静默换慢路径。
- "非末块对齐、末块按实际长度"是**同一条规则的两半**：非末块一律 `chunk_limit`（对齐 → 形状与 kernel
  假设稳定），末块是该序列剩余的实际长度（允许更短，按 `cu_seqlens` 分段处理，**不是回退**）。

**关键语义（需作者确认，见 §8）**：

- **chunk 期间不出 token**：序列只有在 prompt 全部 prefill 完成后才采第 0 个 token，
  `max_new` 的计时也从那时开始（否则"分块"会改变可见的生成语义）。
- **确定性**：`chunk_limit` 是常量（不做"按队列长度自适应"那类调度策略），
  同一输入必然切出同一组 chunk —— AC1/AC9 的逐位对拍依赖这条。
- **与 D9 的联动**：预留量仍按 `prompt_len + max_new`（S4 已改成**按真实长度**），
  chunk 只是把 K/V 分多步写进去，不改变预算口径。

## 3. 注意力：统一成"分页因果注意力"（**S5 的唯一计算改动**）

现状（S4）：context 段走 varlen 自注意力（K/V 全来自本次调用）；generation 段走分页 + 自包含当前 token。

**S5 的裁决：把 context 段也统一到"分页因果注意力"**：

```text
query = 本步该序列的 chunk（L_c 个 token）
K/V   = 分页缓存里 [0, prompt_done) 的已有 K/V  ++  本 chunk 自身的 K/V（自包含，与 generation 同款）
mask  = 因果（chunk 内）+ 按 cu_seqlens_ctx 分段（不同序列不互相看见）
写回  = 本 chunk 追加到该序列缓存的 [prompt_done, prompt_done + L_c) 位置
```

**为什么统一而不是加第三种 kernel**：

- 首 chunk（`prompt_done == 0`）时"缓存部分为空"，公式**自动退化**成 S4 的 varlen 自注意力；
- 后续 chunk 与 generation 段共用同一族分页寻址（块表 + `context_lens`），只是 query 数 > 1；
- 一条实现 + 一个判据（AC9 要的就是"分块 == 不分块"，两条路径走同一段代码最容易对齐）。

**代价（写清楚）**：query 数 > 1 的分页注意力比"纯自包含 varlen"更重（要按块表扫缓存），
且首 chunk 也要走一遍分页路径 —— 这是拿"少一种模式"换"首块略慢"。若 P4/P7 实测首块代价显著，
再考虑"首块走 varlen、后续走分页"的双模式（届时判据不变）。

**兜底纪律（2026-10-04 作者定：显式拒绝，不静默回退）**：这条统一路径依赖 fused kernel
（`chunk_limit` 是受支持的常量、query 数可被内核一次消化）。凡是**无法走 fused 路径的配置**——
例如推导出的 `chunk_limit` 不在支持集合内、或所需的 workspace / 形状假设不满足——
一律**在构造期或入口报错并说清原因**；**禁止**悄悄退到一条更慢的通用路径。
理由与项目既有纪律一致（"不许把未验证/慢路径当默认"）：静默回退会让性能结论与现象都无法解释。
末块（长度 < `chunk_limit`）**不算**这种情形：它由同一个 kernel 按 `cu_seqlens` 处理。

## 4. 与 S4 的接线（改动面）

| 位置 | 改动 |
|---|---|
| `PackedAttentionPlugin` | context 段改成"分页因果 + 自包含当前 chunk"（§3）；段边界仍用 `context_seq_count` |
| 活跃表 | 新增 `prompt_done`；`RunPackedMixedStep` 按 §2 切 chunk、按 `prompt_done` 决定写回位置 |
| 写回 | 位置从 `prompt_done` 起（不再假定"从 0 覆盖写"）；`row_lengths` 用 **chunk 长度** |
| 采样 | 只有 `prompt_done == prompt_len` 的行参与采样；chunk 中的行不产生 token（§2） |
| 退出判据 | `max_new` 计时从 prefill 完成起（§2） |
| 判据/用例 | 见 §6（新增一组，不改 S4/S3 的既有用例） |

**不改的东西**：打包顺序（context 在前）、段边界与下标纪律、split-K 的复用、块映射与不变量 4 的口径、
`prefill_mode` 开关（S5 是 packed 路径内部的能力，不新增开关 —— 见 §8 待确认）。

## 5. 形状 / profile

- packed 的 token 维 `T` 仍然是"本步所有参与行的 token 总数"：chunk 只会让它**更小**，
  不改变 S4 已定的范围（`[1, max_batch × max_prefill_seq_len]`）。
- `chunk_limit` 是引擎的**输入约束**不是形状：它只决定"每步送多少"，不需要进 profile。
- 引擎与图**不需要重建**（同一张 packed 图）：这是 S4 把"每序列 token 数 ≥ 1"写进契约
  换来的好处 —— S5 不动图、不动 `graph_version`。

## 6. 判据与用例（待实现后补进 test_plan.md）

| 用例 | 判据 |
|---|---|
| `ChunkedEqualsWholePrompt` | AC9：同一条 prompt 无论 `chunk_limit` 取多少（1 / 中间值 / ≥ prompt_len），token **逐位相同** |
| `ChunkBoundaryDoesNotDisturbOthers` | 分块不影响同批其它序列（含正在 generation 的行） |
| `ChunkedShortPromptsUnchanged` | `chunk_limit ≥ prompt_len` 时行为与不分块逐位相同（S4 的路径不受影响） |
| `ChunkProgressStateIsCorrect` | `prompt_done` 推进正确：chunk 期间不出 token、完成后才采第 0 个、`max_new` 从那时计时 |
| `ChunkedRetireAndBlocks` | 分块跨步时的块记账与退出归还正确（AC3 在分块下的形态） |

## 7. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| `prompt_done` 与压实/退出的交互 | 行号每步变 + 进度状态 → 记错就静默算错 | 状态挂在活跃表行上（与 seq_id 同源），用例 `ChunkProgressStateIsCorrect` |
| 写回位置写错（从 0 覆盖而不是从 `prompt_done` 续） | 覆盖自己的 prompt K/V | 位置由 `prompt_done` 决定并在用例中断言（`ChunkedEqualsWholePrompt` 会直接变红） |
| 首块走分页路径变慢 | 短 prompt 也吃分页开销 | §3 已写明是"少一种模式"的代价；P4/P7 实测后可退回双模式 |
| chunk 切法不确定（自适应） | 破坏 AC1/AC9 的逐位对拍 | §2 写死"常量 `chunk_limit` + 确定性切法" |

## 8. 待作者确认（进入 P3 之前）

**作者 2026-10-04 已全部确认**：

| # | 事项 | 结论 |
|---|---|---|
| 1 | chunk 的语义（期间不出 token、`max_new` 从 prefill 完成起计时） | **同意** |
| 2 | `chunk_limit` 的取值 | **不暴露给调用方，由 profile 上限推导**（并必须落在 fused kernel 支持的常量集合内） |
| 3 | §3 的"统一成分页因果" | **接受首块略慢换少一种模式**；同时定下兜底纪律：**无法使用 fused kernel 的配置要显式拒绝，不许静默回退** |
| 4 | 开关 | **同意**：S5 是 packed 内部能力，不新增开关；**非末块对齐、末块按实际长度** |

**下一步**：进 P3 增量复评（见 `review.md` 的 S5 复评节），复评过了才动 S5 的代码。
