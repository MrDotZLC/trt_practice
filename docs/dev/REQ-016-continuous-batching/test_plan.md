# Test Plan

<!--
P6 产物。本版 2026-10-03 建立（S1 代码已落、未编译）。
`Actual Result` / `Status` 全部待真机填写——**未跑过的项不得写成通过**（AGENTS §7）。
-->

## Requirement Traceability

| 需求条目 | 判据 | 用例 | 结果 |
|---|---|---|---|
| Included 1：批量 > 1 的 prefill/decode | 批量前向能跑通且数值正确 | `LlmRunnerBatchTest.BatchEqualsSequential` | 待真机 |
| Included 2：每序列长度记账 / 位置编码 / 结果收集 | 每行用自己的长度与位置 | 同上 + `BatchEqualsSequentialWithTopP` | 待真机 |
| Included 3：每序列独立采样参数 | `top_k` / `top_p` / `seed` 逐行生效 | `BatchEqualsSequentialWithTopP`（逐行不同 seed） | 待真机 |
| Included 4：每序列 K/V 分配与回收 | 跑 N 轮后空闲块回初始值 | **属于 S2**（S1 沿用"下次调用开头释放"） | 待 S2 |
| Included 5：请求级调度 | 静态批先跑通；连续批（活跃批 + 压实）在 S3 | `BatchEqualsSequential`（静态批）+ `BatchEqualsSequentialUnderScheduling`（连续批总闸） | 待真机 |
| Included 6：批量 == 逐条单跑 | 逐 token 逐位相同 | `BatchEqualsSequential` + `BatchEqualsSequentialWithTopP` | 待真机 |
| Included 7：按真实长度计费 + 两条路径（打包为默认） | 两条路径各自成立；packed 只算 T 个 token | `PackedEqualsSequential` + `PackedShortSequenceNotPenalized`（S4）+ `FallbackSwitchKeepsResults`（两路径可回退） | 待真机 |
| AC1 数值一致性 | 同上 | 同上两条（贪心 + 随机各一条） | 待真机 |
| AC2 长度不齐 | 批内长度不同时逐行正确 | `UnequalPromptLengthsInFlight`（S3）+ `RejectsUnequalPromptLengths`（S1 静态批拒绝） | 待真机 |
| AC3 资源回收 | 空闲块回到初始值 | `BlocksReturnAtEnd`（S3，含失败路径）+ `FreeBlocksReturnAfterBatch` / `FreeBlocksUnchangedAfterFailure`（S2） | 待真机 |
| AC4 不回归 | 既有用例不新增红 | 全量 `mini_trt_llm_tests` | 待真机 |
| AC5 单序列语义不变 | B=1 与旧路径逐位相同 | `BatchSingleRowMatchesGenerate` | 待真机 |
| AC6 性能可复现 | 先声明判别下限，再 A/B | **属于 P4/P7**（环境搁置） | 待环境 |
| AC7 不浪费 | 短序列不受最长序列影响 | `PackedShortSequenceNotPenalized`（S4；**代价**那条属 P4/P7） | 待真机 |
| AC8 两条路径各自成立且可回退 | 各自满足 AC1；切换不改调用方接口 | `FallbackSwitchKeepsResults`（S4）+ `PackedEqualsSequential` | 待真机 |
| Included 8：长 prompt 的分块推进 | 跨多步推进，且与不分块等价 | `ChunkedEqualsWholePrompt`（S5） | 待真机 |
| AC9 分块与不分块等价 | 三种切法逐位相同；不影响同批其它序列 | `ChunkedEqualsWholePrompt` + `ChunkBoundaryDoesNotDisturbOthers`（S5） | 待真机 |

## Unit Test

S1 没有新增单元级用例（改动集中在 runner 与采样器的接口层）。
但**采样器本身有既有单测**，必须在真机确认仍然全绿——见 Regression Test。

**S5 的 B1+A1 修订新增了一组纯 host 单测**（2026-10-05，落 `tests/test_engine_cache.cpp`，
断言 `ReadEngineSidecarField`；**不需要 GPU / TRT**，有编译器就能跑 —— 比下面三条要真机窗口的
runner 侧用例更早能验）。用例清单以该文件为准，这里只登记 5 条的**判据**：

| 用例 | 判据 |
|---|---|
| `EngineCacheTest.SidecarFieldReadsNumericParamByFullLineKey` | 完整行键 `num.model.n_positions` 读到 `16`；**半个键**（`model.n_positions`，漏 `num.` 前缀）读到空串（在 `LLMRunner` 里 = 拒绝启动，不是静默用默认值） |
| `EngineCacheTest.SidecarFieldDoesNotMatchByPrefix` | `num.prefill.max_seq` 与 `num.prefill.max_seq_extra` 共存时各读到自己的值（读到 1024 说明做成了前缀匹配） |
| `EngineCacheTest.SidecarFieldMissingKeyOrFileIsEmpty` | 缺键 / 空 key / 侧车文件不存在 → 一律空串（不可信） |
| `EngineCacheTest.SidecarFieldReadsBodyOnly` | 用 `fingerprint` 当 key 读不到第一行（正文只从 `---` 之后开始）；只有第一行、没有 `---` → 空串 |
| `EngineCacheTest.SidecarFieldToleratesCrlfLineEndings` | 整份 sidecar 用 CRLF（含 `---` 行）时仍能读到值（Windows 上文本模式写出的形态） |

## Integration Test

`tests/test_llm_runner_batch.cpp`（新建，8 条）：

| 用例 | 判据 |
|---|---|
| `BatchEqualsSequential` | 批量与逐条单跑**逐 token 逐位相同**（贪心，AC1） |
| `BatchEqualsSequentialWithTopP` | 同上但走 Top-P 随机路径——锁死"随机流与批位置无关"（AC1） |
| `BatchSingleRowMatchesGenerate` | B=1 时 `GenerateBatch` == `Generate`（AC5） |
| `RejectsUnequalPromptLengths` | 不等长整批拒绝（D2=A） |
| `RejectsBatchOverMaxBatch` | 超 `max_batch` 由入口拦下 |
| `RejectsMixedSamplingStrategy` | 批内混策略被拒（S1 收窄） |
| `RejectsTopKOverFastMax` | `top_k > kTopKFastMaxK` 被拒，**不出现 token = -1** |
| `RejectsDuplicateSeqId` | `seq_id` 批内重复被拒 |
| `FreeBlocksReturnAfterBatch`（S2） | 一次批量调用后空闲块数**回到调用前水位**（AC3） |
| `FreeBlocksUnchangedAfterFailure`（S2） | 块不足整批拒绝后水位不变（失败路径也要归还） |

### S3 调度（`LlmRunnerSchedulerTest.*`，2026-10-04 落码，未编译验证）

> 两条判据在公开接口上**本来不可观测**，靠 `LLMRunner::SchedulerStats`（只读观测口）固定：
> `steps` 判"EOS 是否提前退出"，`context_rows` 判"context 段是否只装新入批的行"。

| 用例 | 判据 |
|---|---|
| `ContextPassDoesNotTouchInactiveSequences` | 只映射到第 1 行的写回，第 0 行**逐字节不变**（cache 层）；反向自证"确实写了第 1 行" |
| `WriteBackRowsMapCorrectly` | `RowOf()` 与行映射一致；`rows = {2}` 时 K/V 落到 seq 11 自己的块，未映射行的长度不动 |
| `ContextSegmentOnlyCoversNewRows` | `context_rows == 2 && prefill_calls == 2`（整批跑会变成 3）—— 守门用例的 runner 层同伴 |
| `SequenceRetiresAndRowCompacts` | 3 条 / `max_batch = 2` → 必须"退出→准入"；结果逐位等于单跑；块全归还 |
| `UnequalPromptLengthsInFlight` | AC2：长度 4 与 6 同批（右填充），逐条与单跑逐位相同 |
| `EosRetiresImmediately` | `max_batch = 1`，同一组请求跑两遍自校准：`steps(设 EOS) + 4 ≤ steps(不设 EOS)`（差值下界给异步回读留余量），且两遍的第二条 token 逐位相同；EOS 不进结果 |
| `DeterminismWithArrivalSteps` | 换一组 `arrival_step`（Top-P 随机流）→ 逐条逐位相同（锁 per-row 随机步号） |
| `BlocksReturnAtEnd` | AC3：正常路径与"重复 seq_id 整批拒绝"路径都全归还 |
| `BatchEqualsSequentialUnderScheduling` | **总闸**：4 条 > `max_batch`、长度不齐、Top-P，全部与逐条单跑逐位相同；顺带锁 `context_rows == 请求数` / `prefill_calls == 2` / `max_active ≤ max_batch` |

### S4 packed 混合批（`LlmRunnerPackedTest.*`，2026-10-04 落码，未编译验证）

> **packed 路径的"逐条单跑"参考实现 = 单请求的 `RunScheduler`**：packed 模式下
> `GenerateBatch` / `Generate` 会明确拒绝（它们是 padding 路径的入口，拿去跑 packed 引擎只会绑错张量）。
> 另：两条用例的判据是"**由构造保证 + 由结果兜住**"（runner 不暴露内部缓冲，顺序/行序没有外部入口），
> 已在用例注释里写明，避免后人高估。

| 用例 | 判据 |
|---|---|
| `PackedEqualsSequential` | AC1 在 packed 路径内部成立：packed 批跑 == 单请求跑（Top-P 随机流也逐位相同） |
| `MixedStepContextAndGeneration` | **核心场景**：同一步里既有新入批的 context 行、又有在跑的 generation 行，两者结果都对 |
| `ContextTokensPrecedeGeneration` | 打包顺序（context token 在前）；由构造保证，用对顺序敏感的场景兜住 |
| `CuSeqlensBoundaryCases` | `context_seq_count == 0`（纯 generation）与首步纯 context 两种极端都发生过且结果对 |
| `PackedWriteBackMapsCorrectly` | cache 层：packed 源（**行长不等**）+ 行映射的写回落到各自序列自己的块，逐行长度正确 |
| `PackedMetadataFollowsPackedOrder` | `block_tables` / `context_lens` 按 packed 行序重建（照 S3 直传镜像会让 generation 行读到别人的块） |
| `PackedShortSequenceNotPenalized` | AC7 的可观测部分：同批长度差很大时短序列结果不受影响（代价那条属 P4/P7） |
| `FallbackSwitchKeepsResults` | AC8：回退到 `kPaddedTwoPhase` 后仍各自成立（不要求跨路径逐位相同） |

### S5 chunked prefill（`LlmRunnerChunkedTest.*`，2026-10-04 落码，**未编译验证**）

> **落点**：`tests/test_llm_runner_chunked.cpp`（`file(GLOB)` 自动收，CMake 未改）。单独成文件的理由：
> 本组要"一个引擎文件 + **每个切法一个 runner**"（`chunk_limit` 只在 `LLMRunner` 构造期读一次），
> 与 packed 用例文件的单 runner 夹具不同。
>
> **两条已知局限（写下来免得后人高估这一组）**：
> ① **结果侧断言本身不能证明"真的分了块"** —— 切法被静默忽略时结果也会与不分块相同，所以每条都
> 配了形状侧断言（`SchedulerStats::context_rows == ceil(prompt_len/chunk_limit)`）；
> ② 所以这组依赖 `SchedulerStats` 的 `context_rows`。packed 路径的 stats 曾把最后一步的计数
> **重复累加一遍**（`RunPackedMixedStep` 早退不清零，`TS-051` 第 2 条）——**2026-10-04 已修**，
> `ChunkedEqualsWholePrompt` / `ChunkedShortPromptsUnchanged` 因此补上了 `generation_rows` 断言
> （= `max_new - 1` 的形态），作为该缺陷的回归守卫。
> ③ **"逐位相同"是一个待真机确认的假设**：不同切法会让每一步的 token 数 T 不同（T=1 vs T=prompt_len），
> 若 TRT 为不同 T 选了不同的 tactic，per-token 的投影结果可能有末位差异，进而让某个 token 变红。
> 真机上若真出现这种红：按 AGENTS §7 **先诊断**（比 logits / 找第一个分叉位置），**不要**直接把
> "逐位相同"降级成"前缀相同"或加容差——那正是掩盖静默算错的做法。
> ④ **真机前提**：S5 收口把 `cu_seqlens_ctx` 的 profile 行维范围改成 `[1, max_prefill_batch + 1]`
> 并把 `kPackedPrefillGraphVersion` 提到 **6**，所以首次跑这组用例前 packed 引擎会重建一次。

> **`chunk_limit` 在用例里怎么变（2026-10-05 修订，取代"测试专用覆盖钩子"那版）**：AC9 要跑
> "1 / 中间值 / ≥ prompt_len"三种切法 → **直接在构造期给 `Config::max_prefill_seq_len`**（每个切法一个
> runner，共用同一份引擎文件、各自反序列化 —— 见 `test_llm_runner_chunked.cpp` 的夹具）；
> 原先的 `SetChunkLimitOverride` 钩子随 `TS-052` 发现 2 的修订**退役**（一个机制，而不是"字段 +
> 进程级覆盖"两套）。**不走**"为三种切法建三个引擎"那条路（成本明显更高）。
> 新增的构造期拒绝判据：`max_prefill_seq_len` 未声明 / 越界 / 违反交叉校验（`p5_s5_interface_spec.md`
> §2 的 ①～⑤）—— 见 `ChunkLimitRejectedConfigs`。
> **packed 路径的"逐条单跑"参考**同样是单请求 `RunScheduler`（同 S4）。

| 用例 | 判据 |
|---|---|
| `ChunkedEqualsWholePrompt` | AC9：同一条 prompt 在 `chunk_limit` 取 1 / 中间值 / ≥ prompt_len 三种切法下 token **逐位相同**；覆盖"全对齐步"与"含末块步"两种形状 |
| `ChunkedPositionsAreAbsolute` | 第二块起的 `position_ids` = `prompt_done + i`（用段内 `i` 会查错位置表，且**不报错**） |
| `ChunkBoundaryDoesNotDisturbOthers` | 分块不影响同批其它序列（含正在 generation 的行） |
| `ChunkedShortPromptsUnchanged` | `chunk_limit ≥ prompt_len` 时与不分块逐位相同（S4 路径的红利） |
| `ChunkProgressStateIsCorrect` | `prompt_done` 推进正确：chunk 期间不出 token、完成后才采第 0 个、`max_new` 从那时计时 |
| `ChunkedSamplingRowSetIsCompacted` | "分块中的长 prompt"排在"本步完成的短 prompt"**之前**时，完成的那行仍被正确采样、未完成的行不出 token |
| `ChunkedRetireAndBlocks` | 分块跨步时的块记账与退出归还正确（AC3 在分块下的形态） |
| `ChunkLimitRejectedConfigs` | 配置 / 形状类不可用被**显式拒绝**，错误信息带实际值与上界；反向断言"没有静默换路"。用例内部分组用 **(A)/(B)**（`①②③④⑤` 在本表只用于 spec §2 的交叉校验编号）：**(A)** 非法 `Config`（沙箱可判）；**(B)** `max_prefill_seq_len` **未声明** / 越界（含交叉校验 ③）。**注（2026-10-05 B1+A1 修订后）**：`Config::max_positions` 已删除，原先的"声明值超 `max_positions`"与"入口位置拒绝"两组**并入池容量检查** —— 在 B1 的整除约束下"池容量 == `n_positions`"，位置越界与池装不下是**同一个条件**（合并后不再单列，避免制造"两条独立判据"的错觉） |
| `ChunkLimitCrossCheckRejectsOverStepBudget`（2026-10-05 补登，**沙箱已写 / 真机跑**） | **spec §2 交叉校验 ③ 的独立触发**：建图侧 per-row 上界 = 8、模型 `n_positions` 仍 16（S5 的真实形态）⇒ `rows_max = 4`、`T_max = 32`、交叉校验 ③ 的合法上界 = 8，而交叉校验 ② 的上界仍是 16；断言 `L = 8` **接受**、`L = 9` **被交叉校验 ③ 拒绝**（② 放它过去）。**为什么单列一条**：其它用例里 `L_build = n_positions = 16` 使交叉校验 `③ ⟺ L ≤ 16`、被 ② 先拦，拿不到"③ 自己拦人"的证据 |
| `MissingFingerprintSidecarRejected`（2026-10-05 B1+A1 新增，**沙箱已写 / 真机跑**） | `n_positions` 的来源是引擎侧车：把 `.engine` 拷到别处**不带 `.fingerprint`** → runner 构造期**拒绝启动**（不猜默认值；与 `EngineCacheIsFresh` 的"缺 sidecar 一律不可信"同一纪律） |
| `SidecarPositionsMismatchRejected`（同上） | 侧车里 `num.model.n_positions`（完整行键）与引擎自洽性不符（测试里改写 sidecar 正文一行，使 `ceil(N / block_size) != block_tables` 的 dim1）→ 构造期**拒绝启动**（**不取 min、不静默**）；用例带**正对照**：引擎与侧车一起拷贝、内容未改时必须照常接受 |
| `NonDivisibleBlockSizeRejected`（同上，**建图期**） | `n_positions % block_size != 0`（如 `n_positions = 16`、`block_size = 3`）→ **packed 建图硬失败**（信息带实际值与建议因数）；**padding 路径不受影响**（反向对照：同一配置建 padding 引擎应当成功） |

## Regression Test

**必须重点看的是采样器**：本轮改了随机流的来源（`Uniform01(seed, offset, 行号)` →
`Uniform01(seeds[row], offset, 0)`）。静态核对的结论是"不会打破既有黄金数据"，理由是：

| 既有用例类型 | 为什么不受影响 |
|---|---|
| fast-vs-legacy 互比（如 `TopKFastMatchesLegacyTokens`） | 两边用同一套 `RowUniform01`，一起变 → 仍然一致 |
| FP16 分布判据（S-12 / S-13，按 3σ 对解析概率） | 判的是分布，不绑定具体 token |
| 固定 seed 的精确 token 期望 | 全库检索**未发现**把随机采样 token 与硬编码黄金值绑定的用例 |

**这条结论必须在真机用一次全量运行确认**（静态核对不能替代运行结果）。

## Failure Test

S1 的失败路径（对应 `GenerateBatch` 的整批拒绝）：

1. 空批 / 某条 `max_new_tokens <= 0`
2. 批内 prompt 不等长
3. 批大小 > `max_batch`
4. 批内混采样策略
5. 某行 `top_k > kTopKFastMaxK`
6. `seq_id` 批内重复
7. 块不足（`AllocateSequence` 先查后分配失败）

**未覆盖的失败路径**（登记，供 S2/S3 处理）：

- ~~失败发生在已分配部分块之后~~ **已覆盖（S2）**：`FreeBlocksUnchangedAfterFailure`；正常路径另由 `FreeBlocksReturnAfterBatch` 锁住。
- 引擎 `Enqueue` / 采样失败时的中途退出：同样只做了归还，没有用例覆盖。

## 不变量 ↔ 代码对账

design.md 的 5 条不变量，逐条对到当前代码（2026-10-03 静态核对；其中 1 / 2 / 4 已当场补齐）：

| 不变量 | 代码现状 | 待补 |
|---|---|---|
| 1 池 ≥ 块表宽度 | **已落地（2026-10-03）**：构造期校验 + `paged_kv_cache.hpp` 的 `Config` 里写明"为什么池可以更大"的依据（D6） | — |
| 2 profile 归属（两引擎各只有一个 profile） | **已落地（2026-10-03）**：构造期 `check_profile_count()` 要求两个引擎各恰好 1 个 profile，否则拒绝启动（D8）。**`getNbOptimizationProfiles()` 的 API 可用性待真机编译确认** | — |
| 3 循环内零 H2D/D2H | 静态核对通过：解码循环内只有设备侧动作（`SetInputAddress` / 填位置 kernel / enqueue / 追加 K/V / 采样），无 `cudaMemcpy*` | 用例化：P6 里加一条"解码循环内不发生同步拷贝"的检查（可用 `cudaMemcpy` 计数或代码审查 + 注释） |
| 4 批内行号同源 | **已落地（2026-10-03）**：五处行号的清单写成注释 + 登记后显式校验 `active_seqs_[b] == seq_ids[b]`；输出侧由 AC1 的逐位对拍覆盖 | — |
| 5 元数据缓冲指针恒定 | **已落地（S2，2026-10-03）**：构造期按 `max_batch` 预分配；`PagedKVCacheTest.MetadataPointersStableAcrossAllocFree` 锁住指针恒定 | — |

## Expected Result

1. 沙箱：静态检查全过（已完成）——**不等于能编译**。
2. 真机：`cmake --build build -j` 通过、无新增 warning；`mini_trt_llm_tests` 全绿（含 8 条新用例）。
3. 不变量 1 / 2 / 4 的待补项落地后，对应断言在真机通过。

## Actual Result

（待真机填写）

## Status

**未开始（环境不可用）**。沙箱无 nvcc / cmake / TensorRT，代码为**未编译验证**状态；
P4 baseline 亦按 Dependency Missing 记 N/A。
