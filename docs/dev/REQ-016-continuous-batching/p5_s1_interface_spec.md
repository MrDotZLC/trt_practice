# P5-S1 接口细化（可评审后再落代码）

<!--
本文件只写"接口形态与改动点"，不含产品代码。作用：让 S1 的签名、行号绑定与缓冲形状
在写代码之前先被作者过一遍——签名一旦定错，三个文件都要返工。
落代码时本文件不删除；P5 完成后由 `summary.md` 收口。
-->

## 0. 已核实的既有能力（决定 S1 的改动面）

| 事实 | 证据 | 对 S1 的意义 |
|---|---|---|
| 写 K/V 的 kernel **已是 batch 感知** | `src/kv_cache/paged_kv_cache_kernels.cu` 的 `WriteKVKernel` 从线性下标反解 `(b, t, h, d)`，按 `block_tables[b * max_blocks_per_seq + position / block_size]` 寻址；`AdvanceContextLensKernel` 按 `batch_size` 推进 | **S1 不改 kv_cache、不改 kernel** |
| 块表镜像本来就是按批布局 | `block_tables_host_[b * max_blocks_per_seq + i]`，`b` 是 `order_` 里的序号 | 只要**按请求顺序**登记序列，引擎第 i 行就对上第 i 个请求 |
| 位置填充 kernel 已是 batch 感知 | `src/core/llm_runner_kernel.cu` 的 `FillPositionIdsKernel` 按 `b < batch_size` 写 | 只需把 `batch_size` 传对，缓冲能容纳 B 个元素 |
| 采样器参数是 per-batch 指针 | `sampler_common.hpp` 的 `TopKSamplerArgs::top_k` / `TopPSamplerArgs::top_p` 都是 `[batch_size]` | D3 只需把缓冲从 1 元素扩成 B 元素 |
| **采样器的 logits 契约是连续的 `[batch, vocab]`** | `SamplerArgs::logits` 注释：`[batch_size, vocab_size]` | **prefill 的 logits 是 `[B,S,V]`，第 i 行末行在内存里跨步**（行距 `S*V`）→ 必须先收集成连续 `[B,V]` 才能采样 |

最后一行是本次检查**新发现**的约束，也是本方案里唯一新增的拷贝。

## 1. 接口形态

```cpp
struct GenerateRequest {
    std::vector<int64_t> input_ids;   // prompt；S1 要求批内等长（D2=A）
    GenerateOptions     options;      // 每序列独立采样参数（D3）
    int32_t             seq_id = -1;  // <0 → 由 runner 分配为"批内下标"；>=0 → 调用方指定，须批内唯一
};

struct GenerateResult {
    int32_t              seq_id = -1; // 回填，便于调用方对齐
    std::vector<int64_t> tokens;      // 新生成的 token（不含 prompt）
    bool                 ok = false;
};

// 批量入口；成功时返回 size == requests.size()，失败返回空 vector。
std::vector<GenerateResult> GenerateBatch(const std::vector<GenerateRequest>& requests);
```

**为什么 `seq_id` 现在就加**：S3 的请求要跨步存活，必须有一个稳定 id；现在加上，S3 就不用改签名（D11 的"把差异收成一个点"）。

**`Generate` 保持原签名**，内部转成单元素批：

```text
Generate(ids, opt) == GenerateBatch({{ids, opt, -1}})[0].tokens，失败返回 {}
```

这从结构上保证 AC5（批量为 1 时语义与现状逐 token 相同），而不是靠"小心别改坏"。

## 2. 失败语义（整批拒绝）

以下情形整批拒绝（返回空 vector + 日志写明原因），**不做部分成功**：

| 情形 | 日志要求 |
|---|---|
| `requests` 为空，或某条 `max_new_tokens <= 0` | 指出是哪一条 |
| **批内 prompt 长度不等**（D2=A） | 打印各条长度与第一个不等的位置 |
| 批大小 > 引擎 profile 上限 | 打印 B 与上限 |
| 总块需求 > 空闲块（D9 的"先检查后分配"在 S1 的形态） | 打印需要多少块、空闲多少块 |
| 任一行的 `top_k > kTopKFastMaxK` | **必须显式拦截**：采样器对该行写哨兵 `-1`，不拦就表现为"生成了 token = -1"这种难查现象 |
| 绑定 / 入队 / 采样失败 | 沿用现有日志 |

块需求按 `Σ ceil((prompt_len_i + max_new_i) / block_size)` 一次算清，**检查与分配之间不做任何会改块池的操作**。

## 3. 缓冲形状（`llm_runner.hpp`）

`B_max` 与 `S_max` 在构造期确定；`B_max` 由 `Config` 新增字段 `max_batch` 提供（默认 1）。

| 缓冲 | 现状 | S1 | 说明 |
|---|---|---|---|
| `d_prompt_` | `[1, S]` | `[B_max, S_max]` | prefill `input_ids` |
| `d_position_` | `[S]` | `max(B_max*S_max, B_max)` | prefill 是 `[B,S]`、decode 是 `[B,1]`，共用一个足够大的缓冲 |
| `d_prefill_logits_` | `[S, V]` | `[B_max, S_max, V]` | 行距 `S*V`（跨步的来源） |
| `d_prefill_last_logits_`（**新增**） | — | `[B_max, V]` | 收集缓冲：把 `[B,S,V]` 每行的第 `S-1` 行收成连续 `[B,V]`，供采样器使用 |
| `d_decode_logits_` | `[V]` | `[B_max, V]` | `[B,1,V]` 天然连续 |
| `d_prefill_kv_[2L]` | `kv_elems*S` | `kv_elems*S*B_max` | 引擎输出 `[B, kv_heads, S, D]` |
| `d_decode_kv_[2L]` | `kv_elems` | `kv_elems*B_max` | 引擎输出 `[B, kv_heads, 1, D]` |
| `d_tokens_` | `[max_new]` | **`[max_new, B_max]`（步优先）** | 采样器写的是连续 `[batch]`，步优先布局让它**直接写**、不需要逐步散播；结果是主机侧按 `raw[i * B_max + b]` 切分 |
| `d_top_k_` / `d_top_p_` | `[1]` | `[B_max]` | D3 |
| `d_sampler_workspace_` | `batch=1` | `batch=B_max` | `*WorkspaceBytes(B, V)` |

**`B_max` 的取值**：受引擎 profile 上限与 prefill logits 显存共同约束，**P4 没跑之前不能给实测值**。
建议先写 `max_batch = 2` 并在代码注释里标注"**暂定，待 P4 实测**"；
`benchmark_before.md` 已记下环境恢复后必须补的测量项。

## 4. 改动点（`llm_runner.cpp`）

| 函数 | 现状 | S1 改动 |
|---|---|---|
| `ReserveBuffers(prompt_len, max_new)` | 按 1 条算尺寸 | 增加 `batch` 参数，按上表尺寸 |
| `BindPrefill(tokens)` | 设 `[1,S]`、绑 `[1,S,V]` | 设 `[B,S]`；K/V 输出按 `[B, kv, S, D]` 绑 |
| `BindDecode(input_token)` | 设 `[1,1]`、block_tables `[1,W]`、context_lens `[1]` | 设 `[B,1]`、`[B,W]`、`[B]`；`input_token` 变成"每行当前 token"的首地址（行步长 `max_new`） |
| `LogitsRow(row)` | 只支持"唯一一行" | 拆成 `PrefillLogitsRow(b)`（`(b*S + S-1)*V`）与 `DecodeLogitsRow(b)`（`b*V`） |
| `SampleInto(...)` | `batch_size = 1` | `batch_size = B`；prefill 采样前先把末行收集到 `d_prefill_last_logits_`（一次 D2D，**每请求一次，不在循环内**） |
| `Generate` | 主体 | 变成单元素批的包装 |
| `GenerateBatch`（新增） | — | 依次：校验 → 预算块 → 按序登记 `seq_id` → `UploadMetadata` → prefill → 写 cache → 采样 → decode 循环 → 一次 D2H → 逐行 EOS 截断 → 归还块 |

**decode 循环内的形状**：每步 `[B,1]`；`FillPositionIds(context_lens, d_position_, /*batch_size=*/B)`；
`AppendDecodeStep` 传 B 组指针 → **整步只推进一次长度**（沿用现有入口）。

## 5. 与设计 D4 的一处澄清

D4 选的是"每序列独立 token 缓冲"。实现上取 **`[max_new, B_max]` 单次分配、步优先布局**：
每行仍是一段独立的结果，只是共用一次分配与一次 D2H。为什么不是"B 个独立缓冲"：采样器写的
是**连续 `[batch]`**，步优先让每步直接写、省掉"写暂存再逐步散播"。S1 是静态批、循环固定
跑满 `max_new` 步，**不存在"先结束的序列要填充"**（截断在循环后逐行做），所以 D4 里担心的
那个语义风险不成立。D4 的说明已按此改写（design.md，2026-10-03）。

## 6. 测试方式

落码后实际写入 `tests/test_llm_runner_batch.cpp` 的 8 条用例（2026-10-03 与代码核对过）：

| 用例 | 判据 | 环境 |
|---|---|---|
| `BatchEqualsSequential` | 同一组请求，批量结果与逐条单跑**逐 token 逐位相同**（AC1，贪心） | 真机 |
| `BatchEqualsSequentialWithTopP` | 同上但走 **Top-P 随机路径** —— 锁死"随机流与批位置无关"（AC1，随机采样） | 真机 |
| `BatchSingleRowMatchesGenerate` | B=1 时 `GenerateBatch` 与 `Generate` 逐 token 相同（AC5） | 真机 |
| `RejectsUnequalPromptLengths` | 不等长批被整批拒绝且日志指出第一条（D2=A） | 真机 |
| `RejectsBatchOverMaxBatch` | 超 `max_batch` 被入口拦下（不靠引擎报错兜底） | 真机 |
| `RejectsMixedSamplingStrategy` | 批内混策略被拒绝（S1 收窄，见 §8） | 真机 |
| `RejectsTopKOverFastMax` | 某行 `top_k` 超限被拒，**不出现 token = -1** | 真机 |
| `RejectsDuplicateSeqId` | 调用方指定的 `seq_id` 批内重复被拒 | 真机 |
| ~~`BlocksReturnedAfterBatch`~~ | **移到 S2**（见 §8 第 3 条） | 真机 |
| 全量回归 | 既有用例不新增红（AC4） | 真机 |

沙箱（当前环境）：**无编译器**，只能做与编译无关的静态检查（括号平衡、未使用符号、外部符号签名、
include 完整性）——真机编译前所有代码一律按"未验证"对待。

## 7. 待作者确认的三点

1. **接口形态**：`GenerateRequest{input_ids, options, seq_id}` + `GenerateResult{seq_id, tokens, ok}` + `GenerateBatch(...)`；`Generate` 变包装。是否认可？
2. **`max_batch` 暂定 2**：P4 未跑，只能用保守值并标注"待实测"。是否接受？
3. **`seq_id` 现在就引入**（为 S3 铺路，避免二次改签名）。是否认可？

## 8. 实现时新发现的约束（2026-10-03，落码时记录）

以下三条是写代码时才暴露的，**改动方案阅读者必须知道**：

1. **采样策略必须批内统一**（S1 收窄）。三种采样 kernel 各自是"整批一个分支"
   （`LaunchGreedySampler` / `LaunchTopKSampler` / `LaunchTopPSampler` 各接受一个 `batch_size`）。
   因此 S1 的批内约束是：**策略统一；`top_k` / `top_p` / `seed` 均可逐行独立**。
   这是入口校验层面的收窄，放宽校验即可扩回；"同批混策略"才需要改采样器（分组调用或 kernel 内分支）。

   **同日已解决（随机流与批位置无关）**：原实现是 `Uniform01(seed, offset, 行号)`，行号进了哈希，
   于是同一个请求换到别的批位置就换一串 token（固定 seed 的评估不可复现、线上问题无法复现）。
   现改为 **per-batch `seeds` + `RowUniform01`**（行号不进随机流）→ **AC1 对随机采样也成立**。
   代价：`sampler_kernels.cu` 4 个 kernel 签名 + 4 个调用点 + 6 个 launch；`seeds` 为空时保留旧行为，
   既有用例逐位不变。回归用例：`BatchEqualsSequentialWithTopP`。

2. **结果缓冲改成步优先 `[max_new, B_max]`**（见 §3 表）。原因：采样器写的是连续 `[batch]`，
   步优先布局让它每步直接写，省掉"写暂存 + 逐步散播"。

3. **AC3（块回收）的断言移到 S2**：S1 沿用既有"下次调用开头释放"的形态，跑到第 N 轮时
   第 N 轮的块仍被持有。把块生命周期收敛到调用内属于 S2。

另：批入口的"先检查后分配"（D9）目前靠 `PagedKVCache::AllocateSequence` 自身的**先查后分配**
实现（它在动分配器之前先比 `NumFree()`，因此不会走进 `BlockAllocator` 的异常路径）。
代价是日志里报不出"空闲多少块"——要报得给 `PagedKVCache` 加 `NumFreeBlocks()`，
那属于 S2 的资源可见性工作。