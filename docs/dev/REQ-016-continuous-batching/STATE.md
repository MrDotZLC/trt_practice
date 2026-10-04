# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-016-continuous-batching |
| phase | P5-Implementation |
| phase_index | 5 |
| status | in-progress |
| updated | 2026-10-04 |
| owner | Codex |

---

## Completed Artifacts

- `requirement.md`（P0-Requirement，2026-10-03 重做）
- `analysis.md`（P1-Analysis，2026-10-03 重做，**新增 Terminology**）
- `design.md`（P2-Design，2026-10-03 重做，**新增 Requirement Coverage、验证策略、D10/D11**）
- `review.md`（P3-Review，2026-10-03 新建，结论 PASS）
- `benchmark_before.md`（P4-Baseline，2026-10-03，**N/A：环境不可用，已按 Dependency Missing 记账**）
- `p5_s1_interface_spec.md`（P5 补充：S1 接口细化，2026-10-03，经作者确认）

旧版产物可用 `git show f9f8502:docs/dev/REQ-016-continuous-batching/<file>` 取回。

---

## Current Blockers

- **S4 的 generation 段已改为复用 split-K（2026-10-04 作者指出 → 当日修完，待编译验证）**：
  原先我在新插件里自写了一份**单趟** generation kernel，而 `paged_attention_plugin.cu` 的
  **生产路径早就是 split-K**（REQ-014 交付；单趟只是 A/B 参考与 workspace 缺失时的兜底）——
  那等于在 S4 这条默认路径上丢掉 REQ-014 的收益，是**回退**而不是取舍（最初的提交说明写错了）。
  **已落地**：给 `PagedAttentionKernelArgs` 与三条 kernel（单趟 / split 第一阶段 / 归并）加
  "行 / token 基址的设备端读取"参数（`cu_seqlens_ctx` + `context_seq_count`，默认 null/0 = 加参数前的
  逐位行为），索引走 `row_base + batch` / `token_base + batch`，**workspace 槽位仍按段内 batch**
  → 归并的 `batch_size` 口径不变；S4 插件的 generation 段改为直接调 `LaunchPagedAttentionSplit`
  （工作区拿不到时才退单趟，与 paged 插件同一条兜底），`getWorkspaceSize` 改报 split-K 的构建期上界。
  **待办**：真机编译窗口要重跑既有 decode 用例（`PagedAttentionPlugin` / `PagedAttentionSplitKernelTest`），
  确认这次机械改动没有动坏 S1/S2/S3 的 decode 路径。
- **S1 批量用例的 profile 配置不足（2026-10-04 发现，已修 `ed52098`）**：`tests/test_llm_runner_batch.cpp`
  用 `SmallGpt2BuilderConfig()`（`max_prefill_batch = max_decode_batch = 1`）却声明 `max_batch = 2` ——
  真机首次跑 P6 时这批用例会因 profile 形状越界而红。已改成 `SmallGpt2BuilderConfig(max_batch)`
  （默认仍是 1/1/1，单序列用例不受影响）。
- **S4/S5 已过 P3 增量复评 + 第二遍复评（2026-10-04）**：S4 设计经作者确认 4 条（分路径前提、
  `p5_s3` §10 限定、插件 **A1**、映射与不变量 4 口径）+ 段内下标纪律；**chunked prefill 立项为 S5**
  （requirement 的 Included/AC 与 Excluded 已同步改口径，design.md 新增 D15）；第二遍复评又补了
  5 处实现级缺口，并把 `SchedulerStats` 口径与 profile 的 `opt` 两条按推荐定案。
  复评结论：P0 无、P1 一条（性能类判据绑真机）+ S5 的 P2 待补 —— 见 review.md。
  **S4 可在真机窗口进入实现；S5 在它自己的 P2 补齐前不开工。**
- **P4 / P7 搁置（2026-10-03）**：当前不在 GTX 1660 Ti 环境，无法取基线。按
  `phases/p4_baseline.md` 的 Dependency Missing 记 N/A；`benchmark_before.md` 写明环境恢复后
  必须补的四项测量。**批上限（`max_batch`）暂时只能取保守值并标注"待实测"**，不得写成实测结论。
- **本沙箱无编译能力**：没有 nvcc / cmake / 任何 C++ 编译器，也没有 TensorRT 与 build 目录，
  因此 P5 的 Exit Gate（"编译通过、无新增 warning"）在本环境**无法执行**。
  **S1 的代码改动（含采样器）全部未编译验证。**

---

## Next Action

0. **S4（下一步主线）**：作者 2026-10-04 给出"选择性批处理"的口径（两相共享一个 packed 张量、
   attention 按段分派、context token 必须在前），据此已出 **P2 级设计草案**：新建
   `p5_s4_interface_spec.md`（§11 列了 5 条待确认）+ design.md 的 D13 扩充 / S4 小节 / D14。
   **5 条已全部定**（4 条确认 + chunked prefill 立项为 S5），并已过 P3 增量复评（review.md）。
   **下一步**：真机窗口内实现 S4（新图 + 新插件 + 打包/写回/采样适配 + 开关），
   然后按 §9 的 7 条用例补 `test_plan.md`。
1. **S5（chunked prefill，作者 2026-10-04 立项）**：范围/代价/依赖已进 design.md D15 与
   requirement Included 8 / AC9。**开工前必须先补它自己的 P2（接口细化）**；排在 S4 之后。
3. **P5-S1（代码已落，待编译）**：改了 `llm_runner.hpp` / `llm_runner.cpp` /
   `sampler_common.hpp` / `sampler_kernels.cu`，新增 `tests/test_llm_runner_batch.cpp`。
   真机下一步：`cmake --build build -j` → 全量 `mini_trt_llm_tests` → 新增的
   `LlmRunnerBatchTest.*`（8 条）。编译错误与用例结果都要回填本文与 `test_plan.md`（P6）。
4. **P5-S2 / S3**：**S3 已全部落码（写回行映射 → 逐行真长度 → 调度循环 → 9 条用例 + 只读观测口），
   全部未编译验证**。S3 真机收口按序做：① 编译（P5 Exit Gate）；② 跑 `mini_trt_llm_tests` 全量 +
   `LlmRunnerSchedulerTest.*` 9 条；③ 结果回填 `test_plan.md`（P6）。
5. 环境恢复后补 P4，再按 D10 的两种负载跑 P7。

---

## Implementation Plan

（技能 P5 要求：当前修改模块 / 预计文件 / 测试方式）

### P5-S4（**进行中，未编译验证**，2026-10-04 开工）

**当前修改模块**：S4 packed 混合批——插件与分析/设计（已落）、图/构建（已落）、runner（待做）、用例（待做）。

| 步骤 | 内容 | 状态 |
|---|---|---|
| 插件 | `packed_attention_plugin.{hpp,cu}` + `BlockReduceMax` + 注册；generation 段**复用 paged 的 split-K**（给 `PagedAttentionKernelArgs` 与三条 kernel 加行/token 基址） | **已落**（`257aedd` / `487bc9d`），未编译 |
| ① 图/构建 | 接口（`BuildOptions::packed_mixed` / `Config::packed_mixed_prefill`，默认 **false**）+ `kPackedPrefillGraphVersion = 4` + 指纹按开关选代次；**图本体**：`input_ids/position_ids` 用 `Dims2(1,-1)`、去掉 `padding_bias`、加 `block_tables`/`context_lens`/每层 cache/`cu_seqlens_ctx`/`context_seq_count`、注意力换 `PackedAttentionPlugin`（Q/K/V 转 token-major `[T,NH,D]`，输出再转回 heads-major）、**K/V 输出导 token-major**；`ApplyProfile` 改成**按输入名**给范围（packed 的两条动态轴） | **已落**，未编译 |
| ② runner + cache | `Config::prefill_mode`（默认 padding；packed 模式下只用 `prefill_engine_`）、`RunPackedMixedStep`（打包 → 一次调用 → 两段 K/V → 采样）、`UploadRowParamsByOrder`、**按 packed 行序重建** `block_tables`/`context_lens`、⑤ 的 packed↔活跃行号换算与 eos 顺序；cache 侧：`WritePrefillKV`/`AppendDecodeStep` 加 `cu_seqlens_ctx`（**默认 null/0 = 既有行为**）+ 新增 packed-prefill 写回 kernel（一个 block 一行，行长不等） | **已落**，未编译 |
| ③ 用例 | §9 的 8 条（`PackedEqualsSequential` / `MixedStepContextAndGeneration` / `ContextTokensPrecedeGeneration` / `CuSeqlensBoundaryCases` / `PackedWriteBackMapsCorrectly` / `PackedMetadataFollowsPackedOrder` / `PackedShortSequenceNotPenalized` / `FallbackSwitchKeepsResults`）+ test_plan 行 | 待做 |

**测试方式**：本沙箱无编译器 → 只做静态自检（逐行括号深度、符号成对、最长行、CRLF/无 BOM）。
真机：编译 → 跑两条路径的用例 → 重跑既有 decode 用例（这次动过 paged kernel 的索引基址）。

**当前修改模块**：S1 批量执行 —— 运行时（`LLMRunner`）的批量入口与批量缓冲；
外加一处**采样器随机流口径修正**（见 Recovery Notes）。**状态：已落码，未编译。**

| 文件 | 实际改动 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp` | `Config` 加 `max_batch`（暂定 2）与 `enable_diagnostics`（D7，默认关）；新增 `GenerateRequest` / `GenerateResult` / `GenerateBatch`；私有函数与成员改成批量形态；新增 `d_seeds_` |
| `mini_trt_llm/src/core/llm_runner.cpp` | 批量入口与缓冲、按行绑定、批量采样、解码循环、结果切分；`Generate` 变单元素批包装；新增 prefill 末行收集 |
| `mini_trt_llm/include/mini_trt_llm/sampler/sampler_common.hpp` | `SamplerArgs` 新增 per-batch `seeds` |
| `mini_trt_llm/src/sampler/sampler_kernels.cu` | 新增 `RowUniform01`；4 个 kernel 签名 + 4 个调用点 + 6 个 launch 改为带 `seeds` |
| `mini_trt_llm/tests/test_llm_runner_batch.cpp`（新增） | 8 条用例（含随机采样的 AC1 对拍） |

**测试方式**：

1. 沙箱：**只能做与编译无关的静态检查**（括号平衡、未使用符号、外部符号签名、include 完整性）——
   本轮已做，抓到并修掉 3 处（漏掉的命名空间收尾大括号、未使用的 `BlocksForTokens`/`DecodeLogitsRow`、
   测试里的 `kPromptLen`）。
2. 真机必跑：`mini_trt_llm_tests` 全量（回归）+ `LlmRunnerBatchTest.*` 8 条。
   **块回收（AC3）的用例属于 S2**——S1 沿用"下次调用开头释放"的形态，跑到第 N 轮时最后一轮的块仍被持有。

### P5-S3 第 3 步：写回行映射（**状态：已落码，未编译验证**，2026-10-04）

**当前修改模块**：分页 cache 的 prefill 写回路径（`PagedKVCache` + 写回 kernel）与它的调用点。

| 文件 | 实际改动 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/kv_cache/paged_kv_cache.hpp` | `WritePrefillKV` 加 `rows` / `row_count`（契约注释）；新增 `RowOf(seq_id)`；新增 `rows_device_` / `identity_rows_device_` |
| `mini_trt_llm/src/kv_cache/paged_kv_cache.cpp` | 行映射校验（行号范围 + 被映射行的预留量）、映射 H2D 到常驻缓冲、只更新**被映射行**的长度；`RowOf` 实现；构造期推恒等表；`AppendDecodeKV` 复用恒等表 |
| `mini_trt_llm/include/mini_trt_llm/kv_cache/paged_kv_cache_kernels.hpp` | `PagedKVWriteArgs`：去掉 `batch_size`，加 `rows` / `row_count` |
| `mini_trt_llm/src/kv_cache/paged_kv_cache_kernels.cu` | 寻址改 `block_tables[rows[b] * W + position / block_size]` 与 `context_lens[rows[b]]`；元素总数按 `row_count`；launch 同步传参 |
| `mini_trt_llm/src/core/llm_runner.cpp` | `GenerateBatch` 的 prefill 写回显式传**恒等映射**（静态批行为逐位不变，AC5 不受影响） |
| `mini_trt_llm/tests/test_paged_kv_cache.cpp`、`tests/test_gpt2_decode_consistency.cpp` | 6 处按旧签名调用的点改成恒等映射（签名变更的机械后果，断言未动） |
| `docs/dev/REQ-016-continuous-batching/p5_s3_interface_spec.md` | §3 的写回签名补 `row_lengths` 与理由；§7 的文件级改动同步 |

**为什么必须带映射**：S3 的活跃批下上下文段只装本步新入批的序列（B_new 行），而缓存批里还有正在
generation 的行。按"缓存已登记序列数"整批写会拿源缓冲里上一轮的残留行去覆盖**别的序列自己的**
prompt K/V（静默算错）。依据见 `p5_s3_interface_spec.md` §3。`AppendDecodeStep` 的行序就等于
`order_`，不需要映射，本步未动它。

**测试方式**：本沙箱无编译器 → 只做静态自检（锚点唯一、花括号/圆括号/方括号平衡、最长行 < 200、
关键符号成对、CRLF 无 BOM）。真机待跑：`mini_trt_llm_tests` 全量。

**两处发现的收口（2026-10-04，作者授权后）**：

1. `tests/test_paged_kv_cache.cpp` 的 `RejectsPrefillBeyondReservedTokens` 的收尾 `}` 被 P5-S2 提交
   `182fff0` 挪到了文件末尾（新用例插在旧用例的收尾大括号之前）→ 其后两个 TEST 被嵌进它的函数体
   （逐行深度 3）。**已修**（`45ac102`）：大括号搬回原位、删掉末尾多出来的那个；改后 8 个 TEST 深度
   全为 2、`final_depth = 0`。**注意**：那一笔的总量仍然平衡（末尾那个大括号顶了缺），所以只看总量的
   括号检查抓不到这类错，必须看**逐行深度**。
2. padding 路径下每行真实 prompt 长度不同，而 `WritePrefillKV` 只有全局 `tokens`，会把被映射行的
   `context_lens` 统一推成 stride（decode 会从填充位置起算、并把填充位置纳入注意力）。**已修**（本笔）：
   加 `row_lengths`（逐行真实长度，契约 `(0, tokens]`），写完后 `context_lens[rows[i]] = row_lengths[i]`；
   `row_lengths[i] == tokens` 就是 S1/S2 的原行为。签名与理由已回填 `p5_s3_interface_spec.md` §3/§7。
   **残留约束（调度器步要照做）**：填充位置的 K/V 仍按 stride 写进 cache，所以块预留要按
   `ceil((S_step + max_new) / block_size)` 算，不是按真实长度。

### P5-S3 第 4 步：调度循环（**状态：已落码，未编译验证**，2026-10-04）

**当前修改模块**：运行时（`LLMRunner`）的请求级调度 + 两处被它逼出来的接口（采样器随机流、按行追加 K/V）。

| 文件 | 计划改动 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/sampler/sampler_common.hpp` | `SamplerArgs` 加 per-row `offsets`（为空 → 退回标量 `offset`） |
| `mini_trt_llm/src/sampler/sampler_kernels.cu` | `RowUniform01` 与 6 个 kernel / launch 支持 per-row offset |
| `mini_trt_llm/include/.../kv_cache/paged_kv_cache.hpp`、`src/.../paged_kv_cache.cpp` | `AppendDecodeStep` / `AppendDecodeKV` 加 `row_count`：只追加并推进**前 row_count 行** |
| `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp` | 活跃表结构、每步重建用的成员（步 token / 结果 token / eos flag / per-row 参数）、辅助函数签名 |
| `mini_trt_llm/src/core/llm_runner.cpp` | `RunScheduler` 五步循环；`BindPrefill` 按行长度填 `padding_bias`；`PrefillLogitsRow` 按行取真实末位；`SampleBatch` 传 per-row offset 与 `eos_hit`；finish flag 异步回读 + 兜底 |
| `docs/dev/REQ-016-continuous-batching/p5_s3_interface_spec.md` | §3/§4 补 per-row offset 与按行追加；§7 文件表同步 |

**为什么 offset 也要逐行**：AC1（批量 == 逐条单跑）要求随机流只由 `(请求 seed, 该请求自己的步号)` 决定。
调度下同一引擎步里各行的已生成计数不同，标量 offset 会让"到得晚"的请求拿到不同的随机流 ——
`BatchEqualsSequentialUnderScheduling` 直接不成立。

**测试方式**：本沙箱无编译器 → 静态自检（逐行括号深度、符号成对、最长行、CRLF/无 BOM）。
真机待跑：`mini_trt_llm_tests` 全量 + S3 的 8 条（下一步补）。

**本步的取舍（写下来供作者复核）**：context 段之后**不做每步同步**（沿用 decode 段"每步重绑、同流有序"的既有形态）；
若真机出现"重绑踩到未执行完的 enqueue"，退路是加 event 依赖或只在"还有待准入请求"时同步一次。

**落码时的两条自我修正（记下来，都是"不写下来下一个人会踩"的坑）**：

1. **token 不能按行号存放**：退出即压实行号，按行号存的"当前 token"会在压实后喂错行。
   最终形态：结果缓冲按**序列**（槽位 = 请求下标）聚集，生成段每步按行聚集一次输入、
   采样后按行散射回序列槽位（每步 ≤ 2B 次 4 字节 D2D）。
2. **finish flag 最多一个回读在飞**：未消费就发新的会让两次 D2H 同时写同一块 pinned 缓冲（数据竞争）。
   推迟期间退出判定晚一步，连续 4 步没落地强制同步一次（兜底）。

### P5-S3 第 5 步：8 条用例（**状态：已落码，未编译验证**，2026-10-04）

**新增**：`mini_trt_llm/tests/test_llm_runner_scheduler.cpp`（`file(GLOB)` 收，CMake 无需改）。

| 用例 | 层次 | 锁的东西 |
|---|---|---|
| `ContextPassDoesNotTouchInactiveSequences` | cache | 只映射到第 1 行的写回，第 0 行**逐字节不变**；反向自证"确实写了第 1 行" |
| `WriteBackRowsMapCorrectly` | cache | `RowOf()` 与行映射一致；`rows={2}` 时 K/V 落到 seq 11 自己的块、行 0/1 长度不动 |
| `SequenceRetiresAndRowCompacts` | runner | 3 条请求 / `max_batch=2` → 必须"退出→准入"；结果与单跑逐位相同 + 块全归还 |
| `UnequalPromptLengthsInFlight` | runner | AC2：长度 4 与 6 同批（右填充），逐条与单跑逐位相同 |
| `EosRetiresImmediately` | runner | EOS 截断口径与 S1 一致；同批另一条不受影响 |
| `DeterminismWithArrivalSteps` | runner | 换一组 `arrival_step`（含 Top-P 随机流）→ 逐条逐位相同 |
| `BlocksReturnAtEnd` | runner | AC3：正常路径与"重复 seq_id 整批拒绝"路径都全归还 |
| `BatchEqualsSequentialUnderScheduling` | runner | **总闸**：4 条 > `max_batch`、长度不齐、Top-P，全部与逐条单跑逐位相同 |

**两条必须写下来的局限（免得后人高估这两条用例）**：

1. `EosRetiresImmediately` **判不了"退出的时刻"**：公开接口看不到第几步退出，而"下一步退出"与
   "跑满 max_new 再截断"在结果上等价（EOS 之后的 token 反正被截掉）。要判时刻，需要可观测的步数
   计数器或在池压力下做差分 —— 记入下一步待办。
2. `ContextPassDoesNotTouchInactiveSequences` 在 **cache 层**验（runner 没有读回 cache 的观测口），
   锁的是写回机制；"调度器确实只装新入批的行"由总闸从结果侧兜住。

**本步的 builder config 自带**：用例用 `SchedulerBuilderConfig()`（批上限抬到 4），
因为 `SmallGpt2BuilderConfig()` 是 `max_prefill_batch = max_decode_batch = 1`。

**由此发现的既有问题（未改，Independent 缺陷）**：S1 的批量用例
（`tests/test_llm_runner_batch.cpp`）直接复用 `SmallGpt2BuilderConfig()` 却声明 `max_batch = 2`，
**批 2 的 prefill/decode 落在 profile 之外**（`SetInputShape` 会失败）——真机首次跑 P6 时这批用例
会红，且红的不是批量逻辑而是 profile 配置。修法是把 S1 的 fixture 也换成抬批上限的配置
（或把该配置提到 `gpt2_test_support.hpp` 里）。**待作者定夺**。

---

## Phase History

- 2026-10-01: P0 -> P1（第一版）
- 2026-10-01: P1 -> P2（第一版）
- 2026-10-01: P2 -> Gate-A（第一版，未通过）
- 2026-10-02: 作者批准重做 P0–P2；三件产物按代码核实结果重写（第二版）
- 2026-10-02: P2 -> Gate-A（第二版，未通过）
- 2026-10-03: 作者批准"重新开始所有内容"；P0 -> P1 -> P2 -> P3 全量重做
- 2026-10-03: **Gate-A 通过**（D5 / D7 / D10 已拍板）
- 2026-10-03: P4 -> N/A（无 GPU 环境，搁置）→ P5-Implementation
- 2026-10-03: P5-S1 落码（5 个文件），**未编译验证**
- 2026-10-04: **S3 路线修正**：D12 由"固定槽位 + 全批定长"改为"**活跃批 + 压实**"——前者被证明会让上下文段把正在 generation 的行也算一遍、写回时覆盖其 prompt K/V；新增 D13（S3 padding 为生产路径、S4 packed 为默认路径）
- 2026-10-04: **S4 并入本 feature**（打包路径，默认；`REQ-020` 曾分配后同日撤销，编号作废不复用）
- 2026-10-04: **P5-S3 第 3 步落码**（写回行映射：`WritePrefillKV` 带 `rows`/`row_count` + `RowOf` +
  kernel 按映射寻址 + 恒等表复用；7 个文件），**未编译验证**
- 2026-10-04: 作者授权收口两处发现：修复 `tests/test_paged_kv_cache.cpp` 被错位的大括号（`45ac102`）；
  `WritePrefillKV` 加逐行真实长度 `row_lengths`（padding 路径的 `context_lens`）；删掉交接用的
  `next_session_prompt.md`
- 2026-10-04: **作者授权按序收口四项**：① S3 复评（review.md 增量复评 + 两条 P0 由"待确认"改判）；
  ② `test_plan.md` 补 S3 行；③ 加只读观测口 `SchedulerStats`（`steps` / `context_rows` 把两条
  原本不可观测的判据固定进用例）；④ 公共 fixture `SmallGpt2BuilderConfig(max_batch)` 修掉
  S1 批量用例的 profile 越界。**全部未编译验证**
- 2026-10-04: **S4 出 P2 级设计草案**：作者给出"选择性批处理"的口径（两相共享 packed 张量、
  attention 按段分派、context token 在前）→ 新建 `p5_s4_interface_spec.md`，design.md 扩写
  D13 / 新增 S4 小节与 D14（引擎形态）；§11 的 5 条待作者确认后才进 P3 复评
- 2026-10-04: **作者确认 S4 设计的 4/5 条**（分路径前提、§10 限定为"调度策略不重写"、插件 A1 单插件内部分派、
  映射 + 不变量 4 口径改写），并采纳"段边界一个标量 + 每段一份 `cu_seqlens`、段内下标禁止跨段混用"
  （作者指出原 §4 那句"两段第 i 条末位都是 `cu_seqlens[i+1]-1`"会诱导把段内下标当全局用，已改）。
  唯一待定：chunked prefill 是否纳入范围（原 Excluded 第 3 条）
- 2026-10-04: **chunked prefill 立项为 S5**（作者改判）：requirement 的 Excluded 口径收窄（只排除
  调度层的混批/分块优先级策略）、Included 增第 8 条、AC 增第 9 条（分块与不分块逐位相同）；
  design.md 新增 **D15**（范围 / 依赖 S4 / 仍然排除项 / S5 自己设计要解的四件事）。
  **S4/S5 的 P3 增量复评已出**（review.md）：P0 无、P1 两条（性能类判据绑真机；S5 的 P2 待补）
- 2026-10-04: **S4 文档第二遍复评**（作者要求）：修 6 处陈旧措辞（§2/§5/§6 仍用单数组时代的
  `cu_seqlens[...]` 与本应只在 §4 出现的"两段共用一个 i"表述）、补 5 处实现级缺口
  （S3 元数据镜像直传在 S4 失效 → 每步按 packed 行序重建；`position_ids` 走 host 长度镜像；
  写回/追加的源基址与设备端 `cu_seqlens`；**块预留不再需要 stride**（S4 的收益）；
  采样聚集复用 S3 的逐行 async D2D），新增用例 `PackedMetadataFollowsPackedOrder` 与两条风险。
  另新增 2 条待作者定：`SchedulerStats` 口径在 S4 下失效、profile 的 `opt` 值（建议 P4 后定）
- 2026-10-04: **第二遍复评的两条由作者按推荐确认**：① `SchedulerStats` **按路径分别定义** ——
  跨路径口径 = `steps` / `max_active` / `context_rows` / `generation_rows`，`prefill_calls` /
  `decode_calls` **仅 S3**；S3 代码已补 `generation_rows` 并在头文件写明该口径（**未编译验证**）。
  ② packed 引擎 profile 的 `opt` **P4 实测后定**（实现时先取显式标注"待实测"的保守值）
- 2026-10-03: 不变量 1 / 2 / 4 落地：D6 依据注释、D8 构造期 profile 校验、行号同源显式校验
- 2026-10-03: **P5-S2 落码**（6 个文件）：元数据缓冲按 max_batch 预分配、`NumFreeBlocks()`、
  `FreeSequence` 补"压实行 + 重建镜像"（补掉一个被掩盖的洞）、调用内归还（RAII 守卫）、
  D9 预算检查；新增 4 条用例
- 2026-10-03: 随机流改为 per-batch `seeds`（行号不进随机流）→ AC1 对**所有采样策略**成立；
  同时把静态审查反查出的 4 处文档↔代码不一致改齐

---

## Recovery Notes

- **Gate-A 决定（2026-10-03）**：D5 本 feature 先做；D7 诊断开关默认关；**D10 做 S3**；
  AC2 不下修；D11 补齐与后续 feature 的接口面。
- **P4 为什么是 N/A**：作者当前不在 GTX 1660 Ti 环境，先开发代码、GPU 测试搁置。
  `benchmark_before.md` 写明恢复后必须补的四项与口径（关诊断）；**不要**当成"性能已验证"。
- **代码在本环境无法编译**：写代码可以，但编译 / 单测 / 真机验证都要等环境恢复；
  在此之前所有 P5 改动都应视为未验证，**不能在文档里写"已通过"**。
- **随机流口径（2026-10-03，不要回退）**：采样不再用行号做随机输入。`SamplerArgs::seeds`（per-batch）
  非空时走 `RowUniform01` → `Uniform01(seeds[row], offset, 0)`；为空时保留旧行为（单行 / 兼容路径）。
  这条决定 AC1 能不能对所有采样策略成立（原来行号进哈希 → 同一请求换批位置就换输出）。
- **S3 的两条硬前提（2026-10-04）**：
  ① **每次引擎调用只装一种相**——上下文段只装本步新入批的序列、生成段只装本步在跑的序列。
     若让上下文段按整批跑，正在 generation 的行会被算出无意义的 K/V，写回按行号落进**它们自己的块**的
     `0..S-1`，把真实 prompt K/V 覆盖掉（静默错）。padding mask 拦不住（它只作用在图内 attention scores，
     而 K/V 是投影输出）。
  ② **写回必须带行映射**——`WritePrefillKV` 的行数原本取自"缓存已登记序列数"，S1/S2 里恒等于引擎的 B；
     活跃批下上下文段的 `B_new` 小于活跃序列数，必须显式给"引擎第 i 行 → 缓存第 rows[i] 行"。
     不改这一步会拿源缓冲里上一轮的残留行去覆盖别的序列。
- **S2（2026-10-03）**：元数据缓冲**构造期按 max_batch 预分配**（`PagedKVCache::Config::max_batch`，默认 4），
  跨请求不再重分配 → 指针恒定；块生命周期收敛到**调用内**（RAII 守卫），失败路径也归还；
  `FreeSequence` 内部压实行并重建镜像（此前只删 order_，靠"下次登记会整体重建"掩盖）。
- **S1 的批内约束（入口校验）**：prompt 等长（D2=A）；**同一种采样策略**；
  `top_k` / `top_p` / `seed` 均可逐行独立。
- **写冲突**：本 feature 与 `REQ-017` / `REQ-019` 共用运行时入口。`REQ-019` 早已登记；
  `REQ-017` 已于 2026-10-03 补登记。三者不并行改同一文件，本 feature 先做。
- **复核过、仍成立的既有事实**：
  1. 元数据缓冲设备分配容量够就复用；正确性依赖"decode 每步重绑"，不是"指针不变"。
  2. 引擎把 cache 第 0 维声明成 `ceil(n_positions / block_size)`，同一个数又当块表宽度用；
     运行时期物理池可以更大，此关系此前未进任何契约。
  3. 随批增长的显存大头是 **prefill logits**（`S=512` 约 98 MiB/条），不是 K/V 池（64 块约 72 MiB）。
  4. 建图代码不进引擎指纹，**改图必须手工 bump `graph_version`**。
  5. 运行期常驻诊断（同步 D2H + 逐层扫 K/V）**没有任何开关**——D7 要解决的是它。
