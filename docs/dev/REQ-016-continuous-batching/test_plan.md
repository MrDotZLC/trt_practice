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
| Included 5：请求级调度 | 静态批先跑通；连续批在 S3 | `BatchEqualsSequential`（静态批） | 待真机 |
| Included 6：批量 == 逐条单跑 | 逐 token 逐位相同 | `BatchEqualsSequential` + `BatchEqualsSequentialWithTopP` | 待真机 |
| AC1 数值一致性 | 同上 | 同上两条（贪心 + 随机各一条） | 待真机 |
| AC2 长度不齐 | 批内长度不同时逐行正确 | **属于 S3**（S1 限制等长） | 待 S3 |
| AC3 资源回收 | 空闲块回到初始值 | **属于 S2** | 待 S2 |
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

- 失败发生在**已分配部分块之后**：当前代码逐个 `FreeSequence` 归还，但**没有用例**验证"失败后空闲块数回到调用前水位"——属于 S2 的 AC3。
- 引擎 `Enqueue` / 采样失败时的中途退出：同样只做了归还，没有用例覆盖。

## 不变量 ↔ 代码对账

design.md 的 5 条不变量，逐条对到当前代码（2026-10-03 静态核对；其中 1 / 2 / 4 已当场补齐）：

| 不变量 | 代码现状 | 待补 |
|---|---|---|
| 1 池 ≥ 块表宽度 | **已落地（2026-10-03）**：构造期校验 + `paged_kv_cache.hpp` 的 `Config` 里写明"为什么池可以更大"的依据（D6） | — |
| 2 profile 归属（两引擎各只有一个 profile） | **已落地（2026-10-03）**：构造期 `check_profile_count()` 要求两个引擎各恰好 1 个 profile，否则拒绝启动（D8）。**`getNbOptimizationProfiles()` 的 API 可用性待真机编译确认** | — |
| 3 循环内零 H2D/D2H | 静态核对通过：解码循环内只有设备侧动作（`SetInputAddress` / 填位置 kernel / enqueue / 追加 K/V / 采样），无 `cudaMemcpy*` | 用例化：P6 里加一条"解码循环内不发生同步拷贝"的检查（可用 `cudaMemcpy` 计数或代码审查 + 注释） |
| 4 批内行号同源 | **已落地（2026-10-03）**：五处行号的清单写成注释 + 登记后显式校验 `active_seqs_[b] == seq_ids[b]`；输出侧由 AC1 的逐位对拍覆盖 | — |
| 5 元数据缓冲指针恒定 | 属于 S2（S1 仍是"容量够则复用"） | S2 落地后加"多次调用后指针不变"的断言 |

## Expected Result

1. 沙箱：静态检查全过（已完成）——**不等于能编译**。
2. 真机：`cmake --build build -j` 通过、无新增 warning；`mini_trt_llm_tests` 全绿（含 8 条新用例）。
3. 不变量 1 / 2 / 4 的待补项落地后，对应断言在真机通过。

## Actual Result

（待真机填写）

## Status

**未开始（环境不可用）**。沙箱无 nvcc / cmake / TensorRT，代码为**未编译验证**状态；
P4 baseline 亦按 Dependency Missing 记 N/A。