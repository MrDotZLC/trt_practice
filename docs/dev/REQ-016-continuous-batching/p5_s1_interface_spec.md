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

D4 选的是"每序列独立 token 缓冲"。实现上取 **`[B_max, max_new]` 单次分配 + 行步长切分**：
每行仍是一段独立的结果，只是共用一次分配与一次 D2H。S1 是静态批、循环固定跑 `max_new` 步，
**不存在"先结束的序列要填充"**（截断在循环后逐行做），所以 D4 里担心的那个语义风险不成立。
这一点在落代码时要同步改写 D4 的说明，避免下一个人照 D4 字面做出 B 次分配。

## 6. 测试方式

| 用例 | 判据 | 环境 |
|---|---|---|
| `BatchEqualsSequential` | 同一组请求，批量结果与逐条单跑**逐 token 逐位相同**（AC1） | 真机 |
| `BatchSingleRowMatchesGenerate` | B=1 时 `GenerateBatch` 与 `Generate` 逐 token 相同（AC5） | 真机 |
| `RejectsUnequalPromptLengths` | 不等长批被整批拒绝且日志指出第一条（D2=A） | 真机 |
| `RejectsBatchOverProfileLimit` | 超 profile 上限被拒绝（不靠引擎报错兜底） | 真机 |
| `RejectsTopKOverFastMax` | 某行 `top_k` 超限被拒，**不出现 token = -1** | 真机 |
| ~~`BlocksReturnedAfterBatch`~~ | **移到 S2**：S1 沿用"下次调用开头释放"的既有形态，跑到第 N 轮时第 N 轮的块仍被持有，不满足"回到初始值"。AC3 的断言属于 S2 的块生命周期工作 | 真机 |
| 全量回归 | 既有用例不新增红（AC4） | 真机 |

沙箱（当前环境）：**无编译器，只能做静态检查**——本轮不写产品代码，就是为了避免
"写进去一堆没人编得过的代码"。

## 7. 待作者确认的三点

1. **接口形态**：`GenerateRequest{input_ids, options, seq_id}` + `GenerateResult{seq_id, tokens, ok}` + `GenerateBatch(...)`；`Generate` 变包装。是否认可？
2. **`max_batch` 暂定 2**：P4 未跑，只能用保守值并标注"待实测"。是否接受？
3. **`seq_id` 现在就引入**（为 S3 铺路，避免二次改签名）。是否认可？11## 8. 实现时新发现的约束（2026-10-03，落码时记录）11以下三条是写代码时才暴露的，**改动方案阅读者必须知道**：111. **采样策略与 seed 必须批内统一**（S1 收窄）。三种采样 kernel 各自是"整批一个分支"1   （`LaunchGreedySampler` / `LaunchTopKSampler` / `LaunchTopPSampler` 各接受一个 `batch_size`），1   而 `seed` 是整批参数（随机流由 `seed + offset + 行号` 混出）。因此 S1 的批内约束是：1   **策略与 seed 统一，`top_k` / `top_p` 逐行独立** —— D3 的"逐行参数"只落到了 k / p。1   实现方式是把校验放在入口（不满足即整批拒绝），**放回接口即可扩宽，不需要改签名**；1   要实现"同批混合策略 / 逐行 seed"必须改采样器（分组调用或 kernel 内分支）。12. **结果缓冲改成步优先 `[max_new, B_max]`**（见 §3 表）。原因：采样器写的是连续 `[batch]`，1   步优先布局让它每步直接写 8 字节起的一段，省掉"写暂存 + 逐步散播"。13. **AC3（块回收）的断言移到 S2**：S1 沿用既有"下次调用开头释放"的形态，跑到第 N 轮时1   第 N 轮的块仍被持有。把块生命周期收敛到调用内属于 S2。11另：批入口的"先检查后分配"（D9）目前靠 `PagedKVCache::AllocateSequence` 自身的**先查后分配**1实现（它在动分配器之前先比 `NumFree()`，因此不会走进 `BlockAllocator` 的异常路径）。1代价是日志里报不出"空闲多少块"——要报得给 `PagedKVCache` 加 `NumFreeBlocks()`，1那属于 S2 的资源可见性工作。1