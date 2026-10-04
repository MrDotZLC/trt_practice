# P5-S5 接口细化（chunked prefill：长 prompt 跨步分批）

<!--
本文件只写"接口形态与设计裁决"，不含产品代码。2026-10-04 新建：S5 由作者改判立项
（requirement Included 第 8 条、AC9），本文件是 P2 级设计草案，**已过 P3 两轮复评**
（见 `review.md` 的 S5 复评与第二遍复评；第二轮后按作者确认修订，见 §8/§9）。
依赖：S4 的 packed 契约与按段分派的 attention 插件（见 p5_s4_interface_spec.md）。

**2026-10-04 第二遍复评后修订**：作者要求对 S5 的设计再评一遍，复评发现原稿有三处会导致
**静默算错**的落点缺失（chunk 的绝对位置、chunked context kernel 的真实形态、采样行集的表达），
已按下述各节改写；改判依据记在 review.md 的「S5 设计的第二遍复评」一节。
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
| generation 段已经会读缓存（分页 + 当前 token 自包含） | `PagedAttentionSplitKernel`（含 S4 加的行/token 基址） | chunk 的注意力**寻址方式**与它同族（块表 + `context_lens`），但那条 kernel 是 **decode 专用**（每行 1 个 query）；query 数 > 1 要**新写** kernel（见 §3） |
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

- 取 **`chunk_limit` = profile 的单序列上限**（prefill 的 `input_ids` / `position_ids` 第 1 维
  `.max`，即建图时的 `max_prefill_seq_len`）—— 这是 packed 图里"每行 token 数"的天然上界
  （超过它行就会越出形状），所以"由上限推导"是唯一不需要新配置的口径。
- **推导的来源（2026-10-04 作者授权折入）**：`LLMRunner::Config` 里**没有** `max_prefill_seq_len`，
  `Engine` 类也不暴露 profile 查询（只有 `SetInputShape` / `SetTensorAddress` / `Enqueue` 等），
  所以"由 profile 推导"目前无从下手。本设计采用**给 `Engine` 加一个 profile 查询接口**
  （查 `ICudaEngine::getProfileShape(tensor, profile, kOPT|kMAX)`），由 runner 在构造期推出
  `chunk_limit`，而不是往调用方再要一个可能与实际建图漂移的配置值 —— 依据是项目既有的
  "**按对方查询、不按配置假定**"（workspace 版见 `paged_attention_split.hpp` 的注释与
  `PROGRESS.md` §2.15）。**未编译验证**：接口名与 TRT 10.15 的实际签名要在真机窗口核对。
- **推导结果必须先过三条真实约束**（替换原稿"fused kernel 支持的常量集合"）：
  ① `chunk_limit >= 1`；② `chunk_limit <= profile 的行上界`；③ `prompt_done + pos + 1 <= prompt_len
  <= n_positions <= kPackedAttentionMaxContextSeqLen = 1024`。任何一条不满足 →
  **构造期直接拒绝并打印实际值**（见 §3 的兜底纪律），不静默换慢路径。
- 若 profile 查询拿不到（接口不可用 / 张量名不符），同样在**构造期**报错，**不允许**退回一个
  猜测的默认值 —— 那等于把"建图和运行时的口径不一致"变成静默错。
- **`n_positions` 的来源 —— "不新增 Config 字段"的唯一例外（2026-10-04 作者点名"一并修掉"后补，
  `TS-051` 第 4 条）**：与 `chunk_limit` **相反**，它**查不到** —— 引擎里没有任何张量带着
  `n_positions`（只留下 `ceil(n_positions / block_size)`：cache 第 0 维 / `block_tables` 第 1 维），
  而 runner 也看不到 `config.json`。所以只能由调用方声明：`LLMRunner::Config::max_positions`
  （与 `block_size` / `num_blocks` 属于同一类"runner 推不出来"的几何参数）。
  packed 模式下**必填**（`<= 0` → 构造期拒绝），并做两条一致性检查（`<=` cache/块表容量、
  `<=` 插件上限 `kPackedAttentionMaxContextSeqLen`）；入口按
  `prompt_len + max_new - 1 <= max_positions` 拒绝（prompt 与随后 `max_new` 个生成 token 用到的
  最大位置是 `prompt_len + max_new - 2`）。**为什么不能省**：S5 的分块让 `prompt_len` 超过单步形状
  上界成为正常路径，而 `n_positions` 是位置编码查表（`addGather(wpe, position_ids)`）的**真实**
  上界 —— 越过它 wpe 的 gather 越界读，且**不报错**。两条来源的分工是"查得到 / 查不到"，不是口径反复。
- "非末块对齐、末块按实际长度"是**同一条规则的两半**：非末块一律 `chunk_limit`（对齐 → 形状与 kernel
  假设稳定），末块是该序列剩余的实际长度（允许更短，按 `cu_seqlens` 分段处理，**不是回退**）。

**关键语义（作者 2026-10-04 已确认，见 §8）**：

- **chunk 期间不出 token**：序列只有在 prompt 全部 prefill 完成后才采第 0 个 token，
  `max_new` 的计时也从那时开始（否则"分块"会改变可见的生成语义）。
- **确定性**：`chunk_limit` 是常量（不做"按队列长度自适应"那类调度策略），
  同一输入必然切出同一组 chunk —— AC1/AC9 的逐位对拍依赖这条。
- **与 D9 的联动**：预留量仍按 `prompt_len + max_new`（S4 已改成**按真实长度**），
  chunk 只是把 K/V 分多步写进去，不改变预算口径。
- **位置用绝对位置（2026-10-04 第二遍复评补）**：context 行的第 `i` 个 chunk token 的
  `position_ids` = `prompt_done + i`；首 chunk 退化成 `0..L_c-1`（与 S4 现状一致）。
  依据：模型是绝对位置查表（`gpt2_model_builder.cpp` 的 `addGather(wpe, position_ids)`）；
  若沿用现状的"段内位置从 0 起"，第二块起会查错位置表且**不报错**（AC9 必然变红）。
- **采样行集用显式行列表（2026-10-04 第二遍复评补）**：本步参与采样的行 = 生成段前缀 ∪
  本步完成 prefill 的 chunk 行，而完成的行在活跃表里**不保证连续**（长 prompt 分块中 + 新准入的
  短 prompt 本步完成时会出现"洞"）。采样器只吃连续 `[count]`，所以 runner 用显式行列表把
  （末位 logits / per-row 采样参数 / EOS 标记 / 输出 token）在紧凑槽位上聚集与散开 ——
  **不动 `SampleBatch` 签名，也不动不变量 4 的行号纪律**。
- **`prompt_done` 与 cache 长度同源（2026-10-04 第二遍复评补）**：`prompt_done` 就是
  `PagedKVCache::SequenceLength(seq_id)`（S4 的 `host_context_lens` 已经这么取）。本文件 §2 的
  "活跃表新增字段"只表示**语义**，实现上优先**直接复用这一个来源**，避免第二份拷贝漂移；
  若确实要缓存到活跃表，必须写明刷新点与"谁是真源"。

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
- 后续 chunk 与 generation 段共用同一套分页寻址（块表 + `context_lens`），只是 query 数 > 1；
- 一条实现 + 一个判据（AC9 要的就是"分块 == 不分块"，两条路径走同一段代码最容易对齐）。

**kernel 形态（2026-10-04 第二遍复评后定稿，替换原"依赖 fused kernel"的笼统表述）**：

- **新写 chunked context kernel**，网格与 S4 的 `PackedContextAttentionKernel` 同形：
  `(num_heads, B_ctx, max_seq_len)`，一个 block 负责 `(head, seq, 段内 query 位置 pos)`，
  `pos >= L_c` 立即返回（`L_c` 由 `cu_seqlens_ctx` 给出，是设备值）。
- **K/V 两段**：① 分页缓存 `[0, prompt_done)`——注意力时刻的 `context_lens[seq]` 就是 `prompt_done`
  （runner 是"先上传元数据 → 再跑引擎 → 最后写回"，见 `llm_runner.cpp` 的 ②③④）；
  ② 本 chunk 自包含 `[0, pos]`。因果边界 = `prompt_done + pos`；输出只写这 `L_c` 个 query 位置。
- **score 仍在 shared，`getWorkspaceSize` 不变**：per-query 的 key 数 = `prompt_done + pos + 1 <=
  prompt_len <= n_positions`，而 `configurePlugin` 已要求 `n_positions <=
  kPackedAttentionMaxContextSeqLen = 1024`（`gpt2_model_builder.cpp` 把 `cfg.n_positions` 当插件的
  `max_seq_len` 传入）。**不新增 workspace 需求**，这是"可以复用已构建引擎"的条件。
- **不许静默跳过**：现有 kernel 在 `len > 1024` 时直接 `return`（不写输出）——新 kernel 不得沿用
  这种形态，越界一律在构造期 / 入口拒绝。

**代价（写清楚）**：query 数 > 1 的分页注意力比"纯自包含 varlen"更重（要按块表扫缓存），
且首 chunk 也要走一遍分页路径 —— 这是拿"少一种模式"换"首块略慢"。若 P4/P7 实测首块代价显著，
再考虑"首块走 varlen、后续走分页"的双模式（届时判据不变）。

**兜底纪律（2026-10-04 作者定；第二遍复评把检查对象落到真实常量上）**：
凡是**无法走这条路径的配置**——`n_positions > 1024`、`prompt_len > n_positions`、
或所需的形状假设不满足——一律**在构造期或入口报错并说清原因**；**禁止**悄悄退到一条更慢的通用路径。
理由与项目既有纪律一致（"不许把未验证/慢路径当默认"）：静默回退会让性能结论与现象都无法解释。
**注**：原稿写的"`chunk_limit` 必须落在 fused kernel 支持的常量集合内"在代码里**没有对应物**——
`chunk_limit` 是数据不是模板常量；真实约束就是上面两条。末块（长度 < `chunk_limit`）**不算**这种情形：
它由同一个 kernel 按 `cu_seqlens` 处理。

**兜底纪律的适用范围（2026-10-04 作者授权折入）**：分界是"**能否在构造期 / 入口判定**"。

| 类别 | 例子 | 要求 |
|---|---|---|
| **配置 / 形状类**（可判定） | `n_positions > 1024`、`prompt_len > n_positions`（由 `Config::max_positions` 守，见 §2 末）、推导不出 `chunk_limit`、profile 查询失败 | **显式拒绝**，错误信息里带上实际值与上界；禁止静默换路 |
| **运行期资源类**（不可判定） | TRT 没给 workspace（现有 paged / packed 插件会退单趟 + WARN，`paged_attention_plugin.cu` 的兜底注释） | 保留既有降级，但必须满足：① 只影响速度、不影响正确性；② 打一次 WARN 说明原因；③ **S5 新增的路径不得引入新的这类静默降级** |

写这条分界的理由：两条既有降级（paged / packed 的 split-K → 单趟）是 REQ-014 交付时定的，
它们解决的是"运行期拿不到显存缓冲"，不是"配置不可用"。把纪律写成"一律拒绝"会与既有结论冲突，
写成"都可以静默退"又会放走 S5 要拦的配置类问题 —— 所以按可判定性切开。

## 4. 与 S4 的接线（改动面）

| 位置 | 改动 |
|---|---|
| `PackedAttentionPlugin` | context 段换成 chunked 分页因果 kernel（§3）；段边界仍用 `context_seq_count`，**输入契约与 `getWorkspaceSize` 都不变** |
| position_ids | context 行的第 `i` 个 chunk token 用 `prompt_done + i`（绝对位置）；首 chunk 退化为 `0..L_c-1`（§2） |
| 活跃表 | 记录 `prompt_done`（优先直接取 `PagedKVCache::SequenceLength`，见 §2）；`RunPackedMixedStep` 按 §2 切 chunk |
| 写回 | 位置从 `prompt_done` 起：kernel 用**写回时刻**的 `context_lens[row] + t`（无需新输入）；`row_lengths` 用 **chunk 长度** |
| generation 段的行映射 | **加显式行映射**（`rows`，`nullptr` = 恒等，S1/S2/S3 行为与开销不变）：`AppendDecodeKV` / `AppendDecodeStep` / 推进 `context_lens` 的 kernel 都接受一张"每步重建的 cache 行号列表"。理由：本步完成 prefill 的行可能被未完成的行隔开，而 generation 段现在按"cache 行 = 前缀"恒等寻址（`paged_kv_cache.cpp` 的四处）。**不改行序、不置换 `order_`** |
| cache 记账 | host 侧从"赋值"改成"**累加**"（`context_lens_host_[row] += row_lengths[i]`、`Sequence::length` 同）；预留量校验改用**累计长度** —— 只改 kernel 不改这两处 = 第二块起静默错 |
| 采样 | 只有 `prompt_done == prompt_len` 的行参与采样；行集用**显式行列表 + 紧凑暂存**，不动 `SampleBatch` 签名与行号纪律（§2） |
| 退出判据 | `max_new` 计时从 prefill 完成起（§2） |
| `Engine`（新增只读接口） | 加一个 profile 查询（`getProfileShape`）供 runner 在构造期推导 `chunk_limit` 与做入口拒绝；**`chunk_limit` 不新增 `Config` 字段**（§2；`n_positions` 是唯一例外，见下一行） |
| `LLMRunner::Config`（新增 `max_positions`） | 位置表长度 `n_positions`：引擎侧查不到（只留 `ceil(n_positions/block_size)`），只能由调用方声明；packed 模式下必填 + 两条上界检查，入口按 `prompt_len + max_new - 1` 拒绝（§2 末、§3 的适用范围表） |
| 图 / profile（`builder.cpp`） | `cu_seqlens_ctx` 的行维上界从 `max_prefill_batch` 改成 `max_prefill_batch + 1`（它的长度是 `B_ctx + 1`；用行维范围会让"整批都是 context 行"的首步 `setInputShape` 失败，`TS-051` 第 6 条） |
| 测试专用钩子 | `SetChunkLimitOverride` / `ChunkLimitOverride`（声明在 `llm_runner.hpp`，生产路径恒不设置）：给 S5-3 的三种切法用例用；先例 `SetPagedAttentionNumSplitsOverride`（§6） |
| 图版本 | **已定（作者 2026-10-04）：bump 4 → 5**，依据 `engine_cache.hpp` 的"任何改动插件行为的代码变更都要 +1"与 1 → 2 的先例；**同日收口再 5 → 6**：`cu_seqlens_ctx` 的 profile 区间变了（上一行），而 profile 区间同样进不了指纹。代价：真机需要重建 packed 引擎（版本 5 的缓存作废） |
| 判据/用例 | 见 §6（新增一组，不改 S4/S3 的既有用例） |

**不改的东西**：打包顺序（context 在前）、段边界与下标纪律、split-K 的复用（generation 段照旧）、
块映射与不变量 4 的口径、`prefill_mode` 开关（S5 是 packed 路径内部的能力，不新增开关 —— 见 §8）；
packed 插件的输入个数与顺序、**其余** profile 区间（`cu_seqlens_ctx` 的行维范围那一处例外见 §5）。

**"下标纪律"的精确口径（2026-10-04 按方案 A 修正）**：不变量 4 要求的是"**引擎行 ↔ cache 行同源**"，
不是"必须恒等映射"。S3/S4 的 generation 段之所以用恒等行号，是因为那时"能生成的行恰好是活跃前缀"
（= cache 前缀）。S5 分块后这条前提不再成立，所以 generation 段改成**显式行映射**（`rows` 数组，
`nullptr` = 恒等）；原稿"不动下标纪律"的说法由此修正为"**不变量 4 不变，映射方式显式化**"。
行序（`order_`）与段边界（`context_seq_count`）、打包顺序都不动。

**改判的东西（2026-10-04 第二遍复评 + 作者确认）**：原稿把 `graph_version` 也列进"不改"，但那是
**没有依据的结论**（`engine_cache.hpp` 要求"任何改动插件行为的代码变更都要 +1"；`builder.cpp` 记着
1 → 2 正是因为 `PagedAttentionPlugin::getWorkspaceSize` 从 0 变正数）。作者 2026-10-04 复核后
**确定 bump 4 → 5**（不再保留"沿用 4 + 写豁免条件"的分支）。

## 5. 形状 / profile

- packed 的 token 维 `T` 仍然是"本步所有参与行的 token 总数"：chunk 只会让它**更小**，
  不改变 S4 已定的范围（`[1, max_batch × max_prefill_seq_len]`）。
- `chunk_limit` = 从引擎 profile **查出来**的单序列上限（`input_ids` / `position_ids` 第 1 维 `.max`，
  不是 `config_` 里的字段 —— `LLMRunner::Config` 没有这个字段，见 §2）。它是**数据**不是形状，
  只决定"每步送多少"，不需要进 profile。
- **`cu_seqlens_ctx` 的行维范围 = `[1, max_prefill_batch + 1]`**（2026-10-04 收口修正）：它的长度是
  `B_ctx + 1`（最后一项是段内 token 总数，插件也靠 `dim[0] - 1` 推 `B_ctx`），比 `block_tables` /
  `context_lens` 的行维**多一格**。沿用行维范围时，`B_ctx = max_prefill_batch` 的首步（整批都是
  context 行）会直接 `setInputShape` 失败（`TS-051` 第 6 条）。
- **真实的形状约束只有三条**（第二遍复评替换了原稿"支持集合"的笼统说法）：
  ① `chunk_limit >= 1`（否则切不动）；
  ② `chunk_limit <= profile 的行上界`（切出来的 chunk 不能越出形状）；
  ③ 每 query 的 key 数 = `prompt_done + pos + 1 <= prompt_len <= n_positions <=
  kPackedAttentionMaxContextSeqLen = 1024` —— 最后一步不等式是建图期已有的校验
  （`configurePlugin` 按插件属性 `max_seq_len = cfg.n_positions` 拦），前两步是数据依赖的运行时事实。
  注意 **`prompt_len` 允许大于 `chunk_limit`**（长 prompt 分多步），约束落在"每 query 的 key 数"上。
  任何一条不满足 → 构造期/入口**显式拒绝**（§3 的适用范围表）。
- 引擎与图：packed 的 I/O 契约与 `getWorkspaceSize` 都**不变**（方案 A：score 仍在 shared），
  所以"S5 不动图"这句话在**拓扑层面**成立；`graph_version` 仍 **bump 4 → 5**（作者 2026-10-04 确认，
  见 §4），因为 `engine_cache` 看不见"插件对同一绑定的计算语义变了"。

## 6. 判据与用例（待实现后补进 test_plan.md）

> **`chunk_limit` 在用例里怎么变（2026-10-04 定，作者可否决）**：它不暴露给调用方，但 AC9 要跑
> "1 / 中间值 / ≥ prompt_len"三种切法 → 取**测试专用覆盖钩子**（`SetChunkLimitOverride` /
> `ChunkLimitOverride`，生产路径恒不设置），照 `SetPagedAttentionNumSplitsOverride` +
> `tests/paged_attention_test_support.hpp` 的先例；**不**走"为三种切法建三个引擎"那条路。

| 用例 | 判据 |
|---|---|
| `ChunkedEqualsWholePrompt` | AC9：同一条 prompt 无论 `chunk_limit` 取多少（1 / 中间值 / ≥ prompt_len），token **逐位相同**；同时覆盖"全对齐步"与"含末块步"两种形状 |
| `ChunkBoundaryDoesNotDisturbOthers` | 分块不影响同批其它序列（含正在 generation 的行） |
| `ChunkedShortPromptsUnchanged` | `chunk_limit ≥ prompt_len` 时行为与不分块逐位相同（S4 的路径不受影响） |
| `ChunkProgressStateIsCorrect` | `prompt_done` 推进正确：chunk 期间不出 token、完成后才采第 0 个、`max_new` 从那时计时 |
| `ChunkedRetireAndBlocks` | 分块跨步时的块记账与退出归还正确（AC3 在分块下的形态） |
| `ChunkedPositionsAreAbsolute` | 第二块起的 `position_ids` 是 `prompt_done + i`（不是段内 `i`）；这条单独锁住，因为它错了也只会表现为 token 逐位不同 |
| `ChunkedSamplingRowSetIsCompacted` | 同一批里"分块中的长 prompt"排在"本步完成的短 prompt"**之前**时，完成的那行仍被正确采样、未完成的行不出 token（显式行列表的紧凑暂存） |
| `ChunkLimitRejectedConfigs` | 配置 / 形状类不可用被**显式拒绝**，错误信息里带上实际值与上界；反向断言"没有静默换路"：① 非法 `Config`（沙箱可判）；② `chunk_limit` 越界；③ `max_positions` 未声明 / 超过引擎侧上界；④ 请求需要的位置超过 `max_positions`（入口拒绝，用 `max_positions = 8 < 池容量 16` 把"池装不下"那条检查排除掉，再加正向对照） |

## 7. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| `prompt_done` 与压实/退出的交互 | 行号每步变 + 进度状态 → 记错就静默算错 | 状态挂在活跃表行上（与 seq_id 同源），用例 `ChunkProgressStateIsCorrect` |
| 写回位置写错（从 0 覆盖而不是从 `prompt_done` 续） | 覆盖自己的 prompt K/V | 位置由 `prompt_done` 决定并在用例中断言（`ChunkedEqualsWholePrompt` 会直接变红） |
| 首块走分页路径变慢 | 短 prompt 也吃分页开销 | §3 已写明是"少一种模式"的代价；P4/P7 实测后可退回双模式 |
| chunk 切法不确定（自适应） | 破坏 AC1/AC9 的逐位对拍 | §2 写死"常量 `chunk_limit` + 确定性切法" |
| **位置用了段内 `i` 而不是 `prompt_done + i`** | 第二块起查错位置表，token 静默逐位不同 | §2 的绝对位置规则 + 用例 `ChunkedPositionsAreAbsolute`（2026-10-04 第二遍复评补） |
| **只改 kernel 忘了改 cache 侧的累加记账** | 第二块起 `context_lens` 被写回旧值 → 覆盖自己的 K/V / 后续 decode 位置错 | §4 的"cache 记账"行 + 预留量按累计长度校验（2026-10-04 第二遍复评补） |
| **采样行集有"洞"却不做紧凑暂存** | 未完成的行被当成完成行采样（多出 token）或完成的行被跳过（少出 token） | §2 的显式行列表 + 用例 `ChunkedSamplingRowSetIsCompacted`（2026-10-04 第二遍复评补） |
| **复用旧引擎却只改了插件语义** | 旧引擎 + 新代码 = 现象与结论无法解释（`engine_cache` 看不见插件行为变化） | §4/§5：`kPackedPrefillGraphVersion` **bump 4 → 5**（作者 2026-10-04 确认） |
| **把 `chunk_limit` 的来源写成"config 里的字段"** | `LLMRunner::Config` 里**没有** `max_prefill_seq_len`，实现时无从取值，最容易退回"猜一个默认值"（= 运行时口径与建图口径不一致的静默错） | §2 定"Engine profile 查询 + 查询失败即构造期报错"；用例 `ChunkLimitRejectedConfigs` |
| **兜底纪律被写成一刀切** | 要么与 REQ-014 的既有降级结论冲突，要么放走 S5 要拦的配置类问题 | §3 的适用范围表（按"能否在构造期/入口判定"切开） |

## 8. 确认记录与待定项

**作者 2026-10-04 已全部确认**：

| # | 事项 | 结论 |
|---|---|---|
| 1 | chunk 的语义（期间不出 token、`max_new` 从 prefill 完成起计时） | **同意** |
| 2 | `chunk_limit` 的取值 | **不暴露给调用方，由 profile 上限推导**（原写的"必须落在 fused kernel 支持的常量集合内"在代码里无对应物，已按 §3/§5 改写成 `n_positions <= 1024` ＋ `prompt_len <= n_positions` 两条，口径不变） |
| 3 | §3 的"统一成分页因果" | **接受首块略慢换少一种模式**；同时定下兜底纪律：**无法走这条路径的配置要显式拒绝，不许静默回退**（检查对象见 §3 的改写） |
| 4 | 开关 | **同意**：S5 是 packed 内部能力，不新增开关；**非末块对齐、末块按实际长度** |

**第二轮确认（作者 2026-10-04，第二遍复评之后）**：

| # | 事项 | 结论 |
|---|---|---|
| 5 | `graph_version` | **确定 bump 4 → 5**（不再保留"沿用 4 + 写豁免条件"的分支） |
| 6 | "分块"的术语口径 | **指针式登记**：`analysis.md` 的 Terminology 表里登记条目，但定义**不复制正文**，指向本节 §2（`p5_s5_interface_spec.md` §2 是唯一来源） |
| 7 | 第二遍复评查出的三条缺口 | **全部折进设计**：① `chunk_limit` 的来源 = `Engine` 的 profile 查询（不新增 `Config` 字段）；② 入口拒绝要带上实际值与上界；③ 兜底纪律按"能否在构造期 / 入口判定"分适用范围（§2/§3） |

**第三轮确认（作者 2026-10-04 点名"一并修掉"，收口 `TS-051`）**：

| # | 事项 | 结论 |
|---|---|---|
| 8 | `n_positions` 的来源 | 引擎侧**查不到** → 新增 `LLMRunner::Config::max_positions`（"不新增 `Config` 字段"的唯一例外）；packed 模式必填 + 两条上界检查 + 入口拒绝（§2 末） |
| 9 | `cu_seqlens_ctx` 的 profile 上界 | 从行维范围拆出，改成 `[1, max_prefill_batch + 1]`；`kPackedPrefillGraphVersion` **5 → 6**（真机需重建一次 packed 引擎） |

**下一步**：设计修订已出（本文件 + `design.md` D15/D16 + `analysis.md` 的术语登记），
Gate-A 重开待作者确认；确认后按 `review.md` 的 S5 用例清单开始 S5-1 的实现。

## 9. 修订记录

**2026-10-04（第一版）**：建立 S5 的接口草案；作者同日确认 §8 的四条。
依据：作者立项（requirement Included 8 / AC9）。

**2026-10-04（第二遍复评后）**：依据 `review.md` 的「S5 设计的第二遍复评」（该轮的 P0 三条、
P1 三条在此收口），逐节修订：

| # | 节 | 改了什么 |
|---|---|---|
| ① | §2 | 补"绝对位置"与"采样行集用显式行列表"两条语义；把 `prompt_done` 收敛到 `PagedKVCache::SequenceLength` 这一个来源 |
| ② | §3 | 换掉"依赖 fused kernel"的笼统表述，写清 chunked context kernel 的形态（新写 kernel、score 仍在 shared、`getWorkspaceSize` 不变、禁止静默跳过）；"支持集合"落成两条真实约束 |
| ③ | §4 | 补 position_ids / cache 累加记账 / 采样行集三条改动；`graph_version` 从"不改"改为"保守口径 bump 4 → 5（**待作者复核** → 当轮确认，见下段）" |
| ④ | §5 | 写清形状约束的真实来源（`n_positions <= 1024` 与 `prompt_len <= n_positions`） |
| ⑤ | §6 | 增用例 `ChunkedPositionsAreAbsolute` / `ChunkedSamplingRowSetIsCompacted` |
| ⑥ | §7 | 增四条风险（位置、累加记账、采样行集、复用旧引擎） |

**2026-10-04（作者第二轮确认后）**：把作者确认的三条落进本文件 ——
① §2 把 `chunk_limit` 的来源定成"`Engine` 的 profile 查询"（并写明 `LLMRunner::Config` 里没有该字段）；
② §3 补兜底纪律的适用范围表（按"能否在构造期 / 入口判定"切开，与 REQ-014 的既有降级并存）；
③ §4/§5 把 `graph_version` 从"待复核"改成"确定 bump 4 → 5"；另在 §6/§7 增
`ChunkLimitRejectedConfigs` 用例与两条风险。依据：作者 2026-10-04 的第二轮确认（§8 表 2）。

**2026-10-04（作者点名"一并修掉"后）**：收口 `TS-051` 的六条（1 处编译错误 + 5 处缺陷）。本文件
改动：§2 末新增"`n_positions` 的来源"（唯一新增 `Config` 字段 `max_positions`）；§3 的适用范围表
注明它由谁守；§4 的改动面表新增两行（`Config::max_positions`、`builder.cpp` 的 `cu_seqlens_ctx`
行维范围）并把图版本改成 4 → 5 → 6；§5 写清 `cu_seqlens_ctx` 的行维范围是 `[1, B + 1]`；
§6 的 `ChunkLimitRejectedConfigs` 判据扩成四组。依据：作者 2026-10-04 的"一并修掉"指令
（`TS-051` 的四条 + 收口时新查出的两条）。
