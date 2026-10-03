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
| AC1 数值一致性 | 同上 | 同上两条（贪心 + 随机各一条） | 待真机 |
| AC2 长度不齐 | 批内长度不同时逐行正确 | `UnequalPromptLengthsInFlight`（S3）+ `RejectsUnequalPromptLengths`（S1 静态批拒绝） | 待真机 |
| AC3 资源回收 | 空闲块回到初始值 | `BlocksReturnAtEnd`（S3，含失败路径）+ `FreeBlocksReturnAfterBatch` / `FreeBlocksUnchangedAfterFailure`（S2） | 待真机 |
| AC4 不回归 | 既有用例不新增红 | 全量 `mini_trt_llm_tests` | 待真机 |
| AC5 单序列语义不变 | B=1 与旧路径逐位相同 | `BatchSingleRowMatchesGenerate` | 待真机 |
| AC6 性能可复现 | 先声明判别下限，再 A/B | **属于 P4/P7**（环境搁置） | 待环境 |

## Unit Test

S1 没有新增单元级用例（改动集中在 runner 与采样器的接口层）。
但**采样器本身有既有单测**，必须在真机确认仍然全绿——见 Regression Test。

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
| `EosRetiresImmediately` | `max_batch = 1` + `steps ≤ 8`（不提前退出则 ≈ 11）；EOS 不进结果；同批另一条不受影响 |
| `DeterminismWithArrivalSteps` | 换一组 `arrival_step`（Top-P 随机流）→ 逐条逐位相同（锁 per-row 随机步号） |
| `BlocksReturnAtEnd` | AC3：正常路径与"重复 seq_id 整批拒绝"路径都全归还 |
| `BatchEqualsSequentialUnderScheduling` | **总闸**：4 条 > `max_batch`、长度不齐、Top-P，全部与逐条单跑逐位相同；顺带锁 `context_rows == 请求数` / `prefill_calls == 2` / `max_active ≤ max_batch` |

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
