# P5-S3 接口细化（简化版 TensorRT-LLM 路线）

<!--
本文件只写"接口形态与改动点"，不含产品代码。三条决策由作者 2026-10-03 定：
① 路线 = 简化版 TensorRT-LLM（槽位 + 固定形状 + 设备侧 finish flag + 两段式）；
② 停止判据 = EOS 驱动退出（异步回读，不违反"循环内禁止同步拷贝"）；
③ 与完整 TRT-LLM 的差距 = 不做 remove_input_padding（打包）。
要求：写代码前先过一遍 §10 的待确认项。
-->

## 0. 路线声明（以及刻意不做的部分）

**采用**：槽位表（slot）+ decode 段固定形状 + 上下文/生成两段式 + 设备侧 finish flag + 右填充 + padding mask。

**刻意不做**（逐条写清，免得下一个人以为漏了）：

| 不做的事 | 完整 TRT-LLM 的做法 | 本项目的理由 |
|---|---|---|
| `remove_input_padding`（打包成一维 + 累积长度） | 招牌优化，注意力走 varlen/packing kernel | sm_75 上没有现成的 packing attention kernel；本项目的 prefill 注意力是显式子图（`MatMul → mask → Softmax → MatMul`），要做 packing 得自己写 varlen prefill 插件。**这是与完整实现最大的差距** |
| prefill/decode 混批（chunked prefill） | 高负载下把两者混在一批以填满算力 | requirement 明确 Excluded；本项目按 TRT-LLM 的**两段式**（context 段 + generation 段）走 |
| 抢占与换出 | 显存不足时换出到主存 / 丢弃重算 | requirement Excluded；块不足时请求留在队列（D9） |
| 输入打包带来的"一个请求分多步 prefill" | chunked context | 同上，属 chunked prefill |

## 1. 已核实的既有能力（决定改动面）

| 事实 | 证据 | 对 S3 的意义 |
|---|---|---|
| 已有**两个独立引擎** | `llm_runner` 持 prefill / decode 两个 `Engine` | 与 TRT-LLM 的 context / generation 两段式天然对齐 —— D10=A **不是**降级实现 |
| decode 的 profile 批范围是 1/1/4 | `core/builder.hpp` | 固定形状取 `max_batch`（≤4）落在 profile 内，不需要重建引擎的 profile |
| PagedAttention 的 batch 取自 query 的 dim 0 | `paged_attention_plugin.cu` 的 `grid.y = batch` | **空槽行也会被算到** → 必须定义空槽的安全输入（§3） |
| prefill 的因果 mask 是常量、**没有 mask 输入** | `gpt2_model_builder.cpp`（按 `S×S` 切片） | 不等长批必须**改图**加 padding mask（D2=B） |
| 采样参数是 per-batch 指针（k / p / seed） | `sampler_common.hpp` | 逐行缓冲已具备；S3 只需每步重填 |
| 元数据缓冲已按 `max_batch` 预分配、指针恒定 | S2（已提交） | 槽位方案的形状恒定依赖它 |
| 采样 kernel 目前只输出 token | `sampler_kernels.cu` | 要加**可选的 finish flag 输出** + `eos_token_id` 入参 |

## 2. 槽位模型

```text
SlotTable（固定 max_batch 行，行号 = 槽位号，**终身不变**）
  每槽: seq_id / state{empty, context, generation, finished} / prompt_len / generated / max_new
        / top_k / top_p / seed / arrival_step / 结果缓冲段

一步 = ① retire → ② admit → ③ context → ④ generation → ⑤ sample&flag

① retire: 读上一步异步回读的 finish flag；命中者标 finished
           → 释放其块 → 槽位改 empty（**行原地不动**，不压实）
② admit:  从等待队列取 arrival_step ≤ 当前步 的请求，找空槽放入；
           无空槽或块不够 → 留在队列（先检查后分配，D9）
③ context: 对本步**新放入槽**的请求跑一次 prefill（形状 [max_batch, S_step]，padding mask）
④ generation: 对**全部 max_batch 行**跑一次 decode（形状恒定 [max_batch, 1]）
              空槽 / finished 行也参与计算，输出丢弃
⑤ 采样：写 token，并让设备侧计算并写 finish flag
```

**行号终身不变**是这条路线的核心收益：不变量 4（块表第 i 行 / `context_lens[i]` / 采样参数第 i 项 / 结果第 i 段同源）在结构上天然成立，不需要靠"每步小心重排"维持。

## 3. 固定形状与空槽约定（**三个必须先验证的点**）

**形状**：

- decode 段：**恒定** `[max_batch, 1]` —— 热路径，这是固定形状收益最大的地方。
- context 段：`[max_batch, S_step]`，`S_step` = 本步新入槽请求的最大 prompt 长度（≤ profile 的 `max_prefill_seq_len = 512`）。context 段每个请求只跑一次，且只有新请求入槽时才发生，因此不钉死 S 也不会污染热路径。

**空槽的安全输入**（这一节的三点都要先用例固定，见 §8）：

| 项 | 约定 | 为什么 |
|---|---|---|
| padding mask（prefill） | 空槽行与"填充位置"一起被 mask 掉，注意力不产生输出 | 图里原本没有 mask 输入，这正是 D2=B 要加的 |
| `context_lens = 0`（decode） | 空槽行以 0 长度参与 | **必须先确认 PagedAttention 在 `len = 0` 时的行为**：它若是"按长度循环"则天然无输出；若假设 `len ≥ 1` 则要短路 |
| 空槽的块表 | **保留该槽上一次用过的块表**（不改成全 0） | 全 0 会指向物理块 0——那是别的序列的数据；保留旧块表 + `len = 0` 才安全 |
| 空槽的采样参数 | 填合法值（`top_k ≥ 1`、`top_p ∈ (0,1]`、任意 seed） | 采样器会对整批计算，参数非法会越界 / 写哨兵 `-1` |

## 4. 停止判据（设备侧 finish flag + 异步回读）

```text
采样 kernel（可选输出）: finish_flags[b] = (token == eos_token_id) || (generated + 1 >= max_new)
主机每步: cudaMemcpyAsync(B 字节 → pinned) + cudaStreamQuery 轮询
          未落地 → 把退出判定延后一步；连续 N 步未落地则强制同步一次（兜底）
```

- **与硬约束的关系**：用的是**异步拷贝 + 步边界检查**，不是"循环内同步等待"——不违反 requirement 里"解码循环内禁止 H2D/D2H 同步拷贝"。
- **代价**：每步 B 字节的拷贝 + 一次轮询；收益是真正的 EOS 驱动退出（短请求立刻让位）。
- **兜底**：若回读长期不落地，退化为"按 `max_new` 退出"——即 S1 的行为，功能不受影响，只是少赚一点。

## 5. 确定性

调度结果必须**可复现**（AC1 的逐位对拍要求）：所有新请求由调用方给 `arrival_step`，调度严格按 (arrival_step, 请求下标) 的字典序取用，槽位分配取最小空槽号。同一输入 → 同一结果。

## 6. 接口形态

```cpp
struct SchedulerRequest {
    GenerateRequest request;      // 复用 S1 的结构（input_ids / options / seq_id）
    int32_t arrival_step = 0;     // 该请求在第几步进入等待队列
};

// 跑完整个请求集合（内部逐步调度），返回按 requests 顺序回填的结果。
// 失败语义与 GenerateBatch 一致：任一不可恢复错误 → 整批拒绝（返回空 vector）。
std::vector<GenerateResult> RunScheduler(const std::vector<SchedulerRequest>& requests);
```

不进流式 `Submit/Step`：那是服务层形态，而项目定位不做服务层；将来真要接，在外面包一层即可，内核不用改。

## 7. 改动点（文件级）

| 文件 | 改动 |
|---|---|
| `include/.../core/llm_runner.hpp` | `SlotTable` / `SchedulerRequest` / `RunScheduler`；逐步重填逐行缓冲；pinned finish 缓冲 |
| `src/core/llm_runner.cpp` | 调度循环（§2 五步）；空槽行的输入构造；finish flag 回读与兜底 |
| `src/core/gpt2_model_builder.cpp` | **prefill 加 padding mask 输入**（改 I/O 契约） |
| `include/.../core/builder.hpp` + `engine_cache.hpp` | **`graph_version` +1**（改图必须 bump，否则静默复用旧图） |
| `include/.../sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu` | 采样器加可选 `finish_flags` 输出与 `eos_token_id` 入参 |
| `tests/test_llm_runner_scheduler.cpp`（新增） | §8 的用例 |

规模 6~8 个文件，是三个里程碑里最大的一笔（远超技能软约束，提交说明要写明）。

## 8. 判据与测试

| 用例 | 判据 |
|---|---|
| `SlotRetiresAndReuses` | 一条跑完 → 槽位释放 → 新请求复用同一槽；其余序列**行号不变** |
| `UnequalPromptLengthsInFlight` | 批内 prompt 长度不同（AC2），逐行位置与语境长度正确 |
| `EosRetiresImmediately` | 采到 EOS 的序列在**下一步**就退出（不是等 `max_new`） |
| `EmptySlotsAreHarmless` | 空槽行的 `context_lens = 0` / padding mask 生效：结果与非空槽方案逐位相同 |
| `DeterminismWithArrivalSteps` | 同一 `arrival_step` 序列重复跑，结果逐位相同 |
| `BlocksReturnAtEnd` | 全部结束后空闲块回到初始水位（AC3） |
| `BatchEqualsSequentialUnderScheduling` | **AC1 在动态批下仍成立**：同一请求同 seed，无论 `arrival_step` 怎么排，token 逐位相同 |

其中 `EmptySlotsAreHarmless` 是**空槽安全性**（§3 那三个点）的守门用例，必须在实现前先想清它的期望值。

## 9. 风险

| 风险 | 缓解 |
|---|---|
| **空槽算力**（`B_max − B_active` 行的 decode 白算） | P4 的两种负载对照裁决；真亏就退回动态 B + 压实（S2 的压实逻辑正是那条路） |
| 改图未 bump `graph_version` | 同一个提交里 bump；引擎缓存指纹不含建图代码 |
| PagedAttention 对 `len = 0` / 空槽行的行为未验证 | `EmptySlotsAreHarmless` 先固定；必要时给空槽专用占位块 |
| finish flag 回读长期不落地 | 连续 N 步后强制同步一次（退化到按 `max_new` 退出，功能不受影响） |
| 固定形状与 profile 上限不符 | `max_batch` 必须 ≤ 引擎 profile 的 batch 上限（构造期已有 D8 的 profile 校验） |

## 10. 待作者确认

1. **requirement 的措辞改动**（这是范围变更，要你单独批）：
   Excluded 里那条 `EOS 在循环内早停（缺口 G2-4，语义正确、只是多算）`
   → 改成 **`循环内同步等待 EOS`**（异步拷贝 + 步边界检查属于本轮范围）。
   同时 design.md 的 Excluded 与覆盖表要跟着改。
2. **形状粒度**：decode 恒定 `[max_batch, 1]`；prefill 取 `[max_batch, S_step]`（`S_step` = 本步新请求最大长度），不把 S 钉成 512。是否认可？
3. **空槽块表约定**：保留该槽上一次的块表 + `context_lens = 0`（而不是全 0 或专用占位块）。是否认可？
4. **`FreeSequence` 的压实语义去留**（S2 已交付、已测）：保留为动态 B 备选路径，还是随槽位方案改造为槽位语义？按 §0.6，这归你判。