# P5-S3 接口细化（活跃批 + padding mask）

<!--
本文件只写"接口形态与改动点"，不含产品代码。2026-10-04 整篇重写：
路线由"槽位 + 固定全批形状"改为"**活跃批 + 压实**"（design.md D12 已同步推翻重写），
并补齐"写回行映射"这一关键改动。S4（打包路径）另见 design.md D13 与第 10 节。
-->

## 0. 路线声明（以及刻意不做的部分）

**采用**：批 = **本步活跃的序列**（不保留空槽；退出即压实行号）+ 上下文/生成两段式 +
右填充 `padding_bias` + 设备侧 finish flag。

**本 feature 内的两条路径**（见 D13）：S3 交付"填充 + 掩码"路径；S4 交付"打包"路径并作为**默认**。

**刻意不做**：

| 不做的事 | 谁在做 | 本 feature 的理由 |
|---|---|---|
| prefill 与 decode **同批计算**（chunked prefill 混批） | vLLM 的 chunked prefill | requirement 明确 Excluded；本 feature 走两段式（与 TRT-LLM 的 context / generation 分相对齐） |
| 输入打包（`remove_input_padding`） | TRT-LLM 完整版 | **属 S4**（本 feature 的里程碑，默认路径） |
| 抢占与换出 | 大显存服务 | requirement Excluded；块不足时请求留在队列（D9） |

## 1. 已核实的既有能力（决定改动面）

| 事实 | 证据 | 对 S3 的意义 |
|---|---|---|
| 已有**两个独立引擎** | `llm_runner.hpp` 持 `prefill_engine_` / `decode_engine_` | 与"上下文/生成分相"天然对齐 |
| prefill 图**没有 cache 输入** | `gpt2_model_builder.cpp` 的 cache 输入只在 `if (is_decode)` 里声明 | 它按"图内算出的 K/V"工作；S3 不改 |
| decode 的 profile 批范围 1/1/4 | `core/builder.hpp` | 两段的批大小都落在动态范围内 |
| PagedAttention 的 batch 取自 query 的 dim 0 | `paged_attention_plugin.cu` 的 `grid.y = batch` | 活跃批下每行都是真实序列，不存在"空行"问题 |
| 采样参数是 per-batch 指针（k / p / seed） | `sampler_common.hpp` | 逐行缓冲已具备；S3 每步重填 |
| 元数据缓冲已按 `max_batch` 预分配、指针恒定 | S2（已提交） | 每步重建元数据**不会触发重新分配** |
| 采样器已支持可选 `eos_hit` 输出 + `eos_token_id` | S3 已提交的一部分 | 停止判据的设备侧来源 |

## 2. 活跃批模型

```text
活跃表（行号 = 本步活跃序列的序号，**每步可能变**）
  每条: seq_id / state{context, generation} / prompt_len / generated / max_new
        / top_k / top_p / seed / arrival_step

一步 = ① retire → ② admit → ③ context → ④ generation → ⑤ sample&flag

① retire: 读上一步异步回读的 finish flag（或 generated 达上限）
           → 释放其块、从活跃表移除 → **压实行号**（S2 的 FreeSequence 已具备该能力）
② admit:  从等待队列取 arrival_step ≤ 当前步 的请求 → 先检查空闲块（D9）→ 追加到活跃表尾部
③ context: **只装本步新入批的序列**，形状 [B_new, S_step]，padding_bias 按每行真实长度填
④ generation: **只装本步处于 generation 的序列**，形状 [B_active, 1]
⑤ 采样：写 token，并让设备侧计算并写 finish flag
```

**核心：每次引擎调用只装一种相**。decode 行不进 context 那张图；context 段落进的序列也不会被
generation 段再算一遍（它从下一步起才参与）。

**行号每步可能变**（退出即压实），所以逐行缓冲必须**每步重建**；不变量 4（块表第 i 行 /
`context_lens[i]` / 采样参数第 i 项 / 结果第 i 段同源）由"每步重建 + 显式校验"来保证。

## 3. 形状与写回映射（**两个必须先验证的点**）

**形状**：

- context 段：`[B_new, S_step]`，`B_new` = 本步新入批的序列数，`S_step` = 其中最大的 prompt 长度（≤ profile 的 `max_prefill_seq_len = 512`）；
- generation 段：`[B_active, 1]`，`B_active` = 本步在跑的序列数；
- 两者都落在引擎 profile 的动态范围内；**全程不引入"填充行"**。

**写回必须带行映射**（关键）：

`WritePrefillKV` 现在的行数取自"缓存已登记序列数"（`args.batch_size = order_.size()`）。S1/S2 里它恒等于引擎的 B，所以一直没暴露；活跃批下 `B_new` **小于**活跃序列数，于是：

```cpp
// rows[i] = 引擎第 i 行 → 缓存批内第 rows[i] 行
// row_lengths[i] = 引擎第 i 行的**真实** token 数（<= tokens）
cudaError_t WritePrefillKV(int32_t layer, const void* key, const void* value, int32_t tokens,
                           const int32_t* rows, int32_t row_count,
                           const int32_t* row_lengths, cudaStream_t stream);
```

kernel 里把寻址从 `block_tables[b * W + …]` 换成 `block_tables[rows[b] * W + …]`。

**`tokens` 与 `row_lengths` 必须分开（2026-10-04 补，落码时发现）**：`tokens` 是源张量的 token 轴长度，
也就是**每行写入的位置数**（padding 路径下 = 本步的 `S_step`）；而 `context_lens` 必须停在各行的
**真实** prompt 长度上，否则 decode 会从填充位置起算、并把填充位置纳入注意力（AC2 直接不成立）。
S1/S2 的批内等长让两者恒等，所以过去一个 `tokens` 就够。

- 写完后 `context_lens[rows[i]] = row_lengths[i]`（**不是** `tokens`）。
- 契约：`row_lengths[i] ∈ (0, tokens]`；全等于 `tokens` 就是静态批的原行为。
- 填充位置的 K/V **仍会**被写进 cache（写入按 stride 走，才能保持一次 launch 与合并访存）；
  它们不在 `context_lens` 内 → 从不参与注意力，且会被后续 decode 逐步覆盖。
  代价是**块预留要按 `tokens`（stride）算**，不是按真实长度：`ceil((S_step + max_new) / block_size)`。
  若实测出池压力（D9 的准入被预算卡住），再考虑"按 `row_lengths[i]` 逐行截断写入"
  （那需要把 lengths 也搬上设备）——**本步不做**。

**不改它的后果**：写回会按 `order_.size()` 逐行写，而源缓冲里只有前 `B_new` 行是本次算出来的、其余是**上一轮的残留** —— 会覆盖别的序列自己的 prompt K/V（静默算错）。

**映射从哪来**：给 `PagedKVCache` 加 `int32_t RowOf(int32_t seq_id)`（显式查询某序列当前在批内第几行）。

**追加也要按行数收口（2026-10-04 补，落码时发现）**：`AppendDecodeStep` / `AppendDecodeKV` 原来按
`batch_size()`（全部已登记行）追加并推进长度。生成段只有**活跃表前缀**有本步的 K/V——本步刚入批的
context 行还没走 generation，给它们也追加就会写进**它们自己的块**、并把它们的语境长度多推一格
（和"写回覆盖"是同一类静默错）。因此两个入口都加 `row_count`：只写、只推进前 `row_count` 行。
静态批传 `batch_size()`，行为逐位不变。
备选是靠"`AllocateSequence` 一定追加在尾部"推出 `rows[i] = order_.size() - B_new + i` —— 那是**隐式约定**，按 D6 的教训不再引入。

**必须先验证的两个点**：

1. 行映射的边界——`WritePrefillKV` 需要映射；`AppendDecodeStep` **不需要**（它的行序就等于 `order_`，即活跃表顺序）；
2. `padding_bias` 的每行真实长度——S1/S2 全 0 即对；S3 起必须按 `prompt_len` 填 0 / -1e4，否则填充位置参与注意力。

## 4. 停止判据（设备侧 finish flag + 异步回读）

```text
采样 kernel（可选输出）: eos_hit[b] = (token == eos_token_id)
主机每步: cudaMemcpyAsync(B 字节 → pinned) + cudaStreamQuery 轮询
          未落地 → 把退出判定延后一步；连续 N 步未落地则强制同步一次（兜底）
退出条件: eos_hit[b] == 1  或  generated + 1 >= max_new（后者主机侧就能判，不需要回读）
```

- **与硬约束的关系**：用的是**异步拷贝 + 步边界检查**，不是"循环内同步等待" —— 不违反 requirement 里"解码循环内禁止 H2D/D2H 同步拷贝"。
- **兜底**：回读长期不落地时退化为"按 `max_new` 退出"——功能不受影响，只是少赚一点。

## 5. 确定性

调度必须可复现（AC1 的逐位对拍要求）：请求由调用方给 `arrival_step`，准入严格按 `(arrival_step, 请求下标)` 的字典序，**追加到活跃表尾部**。同一输入 → 同一结果。

**随机步号必须逐行（2026-10-04 补，落码时发现）**：`SamplerArgs::offset` 是标量，但调度下同一次引擎
调用里各行的"已生成计数"不同（有的在采第 3 个 token、有的才第 0 个）。AC1 要求同一请求无论
`arrival_step` 怎么排都逐位相同 → 随机流只能由 `(请求 seed, 该请求自己的步号)` 决定，
因此加 per-row `offsets`（为空退回标量 `offset`，只服务 S1 / 兼容路径）。标量 offset 下
`BatchEqualsSequentialUnderScheduling` 直接不成立。

## 6. 接口形态

```cpp
struct SchedulerRequest {
    GenerateRequest request;      // 复用 S1 的结构（input_ids / options / seq_id）
    int32_t arrival_step = 0;     // 该请求在第几步进入等待队列
};

std::vector<GenerateResult> RunScheduler(const std::vector<SchedulerRequest>& requests);

// 只读观测口（2026-10-04 补，落码时发现两条判据在公开接口上不可观测）：
//   steps         —— 本次调用走了多少轮循环：判"EOS 是否在下一步退出"
//   context_rows  —— Σ B_new：判"context 段是否只装本步新入批的行"
// 其余字段（max_active / prefill_calls / decode_calls）供用例与排查使用。
// 注意：它反映"调度怎么走的"，不是性能指标。
struct SchedulerStats { int32_t steps, max_active, context_rows, prefill_calls, decode_calls; };
const SchedulerStats& scheduler_stats() const;
```

不进流式 `Submit/Step`：那是服务层形态，项目定位不做服务层；将来要接，在外面包一层即可。

## 7. 改动点（文件级）

| 文件 | 改动 |
|---|---|
| `include/.../kv_cache/paged_kv_cache.hpp` | `WritePrefillKV` 加 `rows` / `row_count` / `row_lengths`（逐行真实长度，见 §3）；新增 `RowOf(seq_id)`；契约注释更新 |
| `include/.../kv_cache/paged_kv_cache.hpp`（追加路径） | `AppendDecodeKV` / `AppendDecodeStep` 加 `row_count`：只追加、只推进前 `row_count` 行（见 §3） |
| `include/.../sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu` | `SamplerArgs` 加 per-row `offsets`（见 §5）；6 个 kernel / launch 同步 |
| `src/kv_cache/paged_kv_cache.cpp` + `paged_kv_cache_kernels.cu` | 写回 kernel 用 `rows[b]` 寻址；`RowOf` 实现 |
| `include/.../core/llm_runner.hpp` | 活跃表与 finish flag 的 pinned 暂存随实现落地（`RunScheduler` 接口已在） |
| `src/core/llm_runner.cpp` | 五步调度循环；**每步重建逐行缓冲**；`padding_bias` 按真实长度填；finish flag 回读与兜底 |
| `tests/test_llm_runner_scheduler.cpp`（新增） | §8 的用例 |

规模：6~7 个文件，是三个里程碑里最大的一笔（提交说明要写明）。

## 8. 判据与测试

| 用例 | 判据 |
|---|---|
| `SequenceRetiresAndRowCompacts` | 一条跑完 → 释放块、活跃表移除、**其余序列行号前移**（S2 的压实路径） |
| `UnequalPromptLengthsInFlight` | 批内 prompt 长度不同（AC2），逐行位置与语境长度正确 |
| `EosRetiresImmediately` | 采到 EOS 的序列在**下一步**就退出（不是等 `max_new`） |
| **`ContextPassDoesNotTouchInactiveSequences`** | 只对新入批的序列跑 context 段时，**其它序列的 K/V 逐位不变**——直接锁住"覆盖"那个静默错。**落点**：cache 层（runner 不暴露 cache），逐字节比对 |
| `ContextSegmentOnlyCoversNewRows`（2026-10-04 补） | 同一条判据的 **runner 层同伴**：晚到的请求入批那一步 `context_rows == 2 && prefill_calls == 2`（整批跑会变成 3）——靠只读观测口判 |
| `WriteBackRowsMapCorrectly` | 行映射 `rows[i]` 与 `RowOf()` 一致；映射故意错位时结果会不同（证明它真的起作用） |
| `DeterminismWithArrivalSteps` | 同一 `arrival_step` 序列重复跑，结果逐位相同 |
| `BlocksReturnAtEnd` | 全部结束后空闲块回到初始水位（AC3） |
| `BatchEqualsSequentialUnderScheduling` | **AC1 在动态批下仍成立**：同一请求同 seed，无论 `arrival_step` 怎么排，token 逐位相同 |

最后一条是总闸；`ContextPassDoesNotTouchInactiveSequences` 是本次路线修正的守门用例，必须在实现前想清期望值。

**两条判据要靠观测口才成立（2026-10-04 落码后补）**：`EosRetiresImmediately` 的"下一步退出"与
`ContextSegmentOnlyCoversNewRows` 的"只装新入批的行"，单看 token 结果都判不出来
（EOS 之后的 token 反正被截掉；runner 读不回 cache）。用例用 `SchedulerStats.steps` /
`context_rows` 固定，其中 EOS 那条用"同组请求跑两遍（设 EOS / 不设 EOS）"的**相对判据**自校准，
避免写死步数阈值被异步回读的正常延迟判成假红。

## 9. 风险

| 风险 | 缓解 |
|---|---|
| **写回映射漏传 / 传错** | 契约里写死"必须带 `rows`"；`RowOf()` + 不变量 4 的校验；守门用例覆盖 |
| 行号每步变导致逐行缓冲错位 | 每步重建 + 显式校验（不变量 4，落码时校的是 `RowOf(seq) == generation_rows + j`）；用例 `WriteBackRowsMapCorrectly` |
| **token 若按行号存放，退出压实后就会喂错 token** | token 按**序列**存进结果缓冲（槽位 = 请求下标），生成段每步按行聚集一次输入、采样后按行散射回序列槽位 |
| finish flag 回读与循环重叠 | 最多一个回读在飞（未消费就**不发新的**，两次 D2H 写同一 pinned 缓冲是数据竞争）；连续 4 步没落地强制同步一次 |
| finish flag 回读长期不落地 | 连续 N 步后强制同步一次（退化为按 `max_new` 退出） |
| `padding_bias` 仍填 0（忘了按真实长度） | S3 的用例里必须有"长度不齐"这条（`UnequalPromptLengthsInFlight`） |
| 两段的批大小超出 profile | 构造期已有 D8 的 profile 校验；`max_batch` 必须 ≤ profile 上限 |

## 10. 与 S4（打包路径）的关系

- **共用契约**：prefill 的产出 =「每序列的 prompt K/V」+「每序列末位 logits」；下游（写回 / 采样 / 调度）不分叉 → S4 只替换"注意力 + 输入布局"，**调度策略不重写**（准入 / 退出 / 压实 / D9 预算不分叉）。
  **限定（2026-10-04 作者确认）**：不重写的是**策略**；每步的**调用形态**会从 S3 的两段式变成"一次调用装两相"（见 `p5_s4_interface_spec.md` §7 与 design.md D14）。
- **数值**：两条路径**不保证逐位相同**（kernel 不同、浮点累加顺序不同）。AC1 在**每条路径内部**成立；跨路径差异按 `AGENTS.md` §7 写清来源与容差出处。
- **默认**：作者指定 **S4 为默认路径**（2026-10-04）；S4 落地后 S3 的填充路径转为对照 / 回退。这是**设计决定**，不是实测结论——S4 落地后仍应做 P4/P7 的 A/B 给默认值一个带判别下限的依据。

## 11. 待作者确认

**无阻塞项**——路线（活跃批）、写回接口（行映射数组）、两条路径的定位都已定。
真机相关的两项（P4 baseline / P7 对照）按 `AGENTS.md` §5 属已授权范围，但需要环境（当前不在 GTX 1660 Ti 上）。
