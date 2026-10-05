# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-016-continuous-batching |
| phase | P5-Implementation |
| phase_index | 5 |
| status | in-progress |
| updated | 2026-10-05 |
| owner | Codex |

---

## Completed Artifacts

> **口径（2026-10-05 盘点，作者点名清单第 6 项）**：清单以技能
> `.agents/skills/trt-inference-engineering/workflows/feature.md` 的 `## Required Artifacts`
> 为**唯一来源**（10 件，与 `SKILL.md` 的 Artifact Directory 一致）—— 本表**行数 = 10**，
> 每行只写"现状 + 证据指针"，**不转述**技能对每件内容的定义（Mandatory #8）。

| # | 产物 | 产出阶段 | 现状 | 证据 / 缺口 |
|---|---|---|---|---|
| 1 | `STATE.md` | 全程 | 在（维护中） | 本文件 |
| 2 | `requirement.md` | P0-Requirement | 在 | 2026-10-03 重做；Included 8 / AC9 |
| 3 | `analysis.md` | P1-Analysis | 在 | 2026-10-03 重做；含 Terminology（指针式登记） |
| 4 | `design.md` | P2-Design | 在 | D1–D16 + Requirement Coverage + 验证策略 |
| 5 | `review.md` | P3-Review | 在 | 四遍复评（Gate-A PASS + S5 的三个增量） |
| 6 | `benchmark_before.md` | P4-Baseline | 在（记 `N/A`） | `## Result` = `N/A` + 恢复清单**五项**（P4 Dependency Missing 三处留痕之一） |
| 7 | `benchmark.md` | P7-Benchmark | **缺** | 卡外部依赖：`phases/p7_benchmark.md` 要求先有 before 数据（无 before 不可执行）+ 独占 GPU；P4 环境未恢复 |
| 8 | `test_plan.md` | P6-Test | 在（未填结果） | 四节清单（S1/S3/S4/S5）+ `TS-055` 已对齐名字/条数/顺序；`## Actual Result` 等真机 |
| 9 | `summary.md` | P8-Documentation | **缺** | P8 产物，依赖 `benchmark.md` 与 P6 结论；其 Performance 一节的 `N/A` 义务已登记（见 `## Next Action` 第 5 条，**P8 必清**） |
| 10 | `interview_notes.md` | P9-Interview | **缺** | P9 产物，依赖 `summary.md` |

**另外 5 件**（不在那 10 件里，属本 feature 的 P5 接口细化，同样落在 Artifact Directory）：
`p5_s1_interface_spec.md` … `p5_s5_interface_spec.md` —— 是 S1–S5 的接口/口径来源，被
`design.md` / `review.md` / `test_plan.md` 引用。

**盘点结论（2026-10-05）**：10 件里 **7 件在、3 件缺**（`benchmark.md` / `summary.md` /
`interview_notes.md`）；三件**都卡在阶段顺序与外部依赖上**（P7 需 before 数据与独占 GPU、P8 需 P7、
P9 需 P8），**不是漏产**。所以本 feature 停在 P5/P6 是符合流程的位置：技能的"未满足（阻塞）"
已按 `## 判据对照` 记账（P4 的三处留痕 + P8 的义务锚点）。**本盘点只读**：不新建那三件产物 ——
`summary.md` / `interview_notes.md` 的**起草**要等 P8/P9 入口，且需作者单独点名。

旧版产物可用 `git show f9f8502:docs/dev/REQ-016-continuous-batching/<file>` 取回。

---

## Current Blockers

- **【静态自检：4/4 已对齐】`TS-055`：四节用例清单 ↔ 测试文件的"同名同序"核对**（2026-10-05，
  作者点名清单第 5 项；完整表见 `docs/TROUBLESHOOTING.md` 的 `TS-055`）：`test_plan.md` 的
  S1 / S3 / S4 / S5 四节 ↔ 四个测试文件的 `TEST` 名，按**条数 / 名字集合 / 顺序**逐位比：
  名字与条数**本来就是一致的**（S1 10 / S3 9 / S4 8 / S5 12），差异只在**清单顺序**（S1 有 1 条位置、
  S3 有 2 条互换、S4 有 1 条位置；S5 本来就一致）。已把**顺序统一到文件**并写成口径
  （"清单顺序 = 文件里 `TEST` 的书写顺序；真机按清单逐条打勾"），同时改掉 `Expected Result` 里陈旧的
  "含 8 条新用例"。**纯文档改动**，不碰代码 / `graph_version` / 指纹。
- **【静态自检：0 处不满足；1 处潜在陷阱已按作者指令修复】`TS-054`：`rows` 的"默认恒等"全量对账**
  （2026-10-05，作者点名清单第 4 项；完整表见 `docs/TROUBLESHOOTING.md` 的 `TS-054`）：
  四处消费者（`WritePrefillKV` / `AppendDecodeKV` / `AppendDecodeStep` / `AdvanceContextLensKernel`）
  的 **6 个生产调用点 + 15 个 tests 调用点**逐个核对"漏传就默认恒等、而语义已变"：
  **0 处不满足** —— 两处用 `nullptr` 的地方（S1 静态批、S3 padding 生成段）都成立，且 S3 那条有
  **代码级依据**：`generation_rows` 在"退出压实之后、admit 之前"取（所以生成行确实是前 `generation_rows`
  行），admit 又用行号同源断言把新行钉在尾部。
  **发现 1 处潜在陷阱（已修）**：`row_starts` 只被 packed prefill kernel 消费，通用 kernel 没有这个
  形参 → "传了 `row_starts` 但没传 `cu_seqlens_ctx`"会被**静默忽略**（K/V 从 0 写，而 host 记账按
  `row_starts + row_lengths` 累加 ⇒ 设备长度与实际写入不一致，静默算错）。当时无调用点触发、tests 零覆盖。
  **已修（作者点名"先修登记的潜在陷阱"）**：`WritePrefillKV` 入口响亮拒绝 + `LaunchWriteKV` 同义兜底
  + 两条用例（`WritePrefillKVRowStartsContinuesInsteadOfOverwriting` 正向覆盖 `row_starts` 的**首条**测试、
  `WritePrefillKVRejectsRowStartsWithoutPackedSource` 反向）。不动图 / profile / 指纹。
- **【静态自检发现，0 处 P0；3 处缺口已按作者指令处理（纯注释）】`TS-053`：host 指针进设备侧的
  全量对账**
  （2026-10-05，作者点名"系统性扫查"；完整清单见 `docs/TROUBLESHOOTING.md` 的 `TS-053`）：
  逐个核对"kernel launch 实参 / `SetTensorAddress` / 采样器 launch / 建图期权重指针"的**可读侧**，
  **没有任何一处实际传错**（`TS-052` 发现 1 的修法在册）。三处**注释 / 契约级**缺口**已改**：
  ① `llm_runner.cpp` 的 `RunPackedMixedStep` 里"目标行集 = 活跃表前缀（走恒等映射）"是 **S4 残留的
  注释**，与同一段上文（以及代码传的 `generation_rows_host`）矛盾 —— 代码对、注释旧 → **已改成
  "行号可能带洞、不是恒等映射"**；
  ② `paged_kv_cache.hpp` 的公开入口里 `rows` / `row_starts` / `row_lengths` 是 **host 数组**
  （内部 H2D），而 `cu_seqlens_ctx` 是**设备数组**，与 kernel 层（`PagedKVWriteArgs`）的同名参数
  语义**相反** → **已把可读侧整段写明**（复核时收窄：`WritePrefillKV` 的 `rows` 原本已有这句，
  真正缺的是 `row_starts` / `row_lengths` / `AppendDecodeKV` 与 `AppendDecodeStep` 的 `rows`）；
  ③ 采样器的 `seeds` / `offsets` / `eos_hit` / `top_k` / `top_p` 被 kernel 直接解引用却没写设备侧
  → **已升成"本结构体所有指针字段都必须是设备可读"的总则 + 就地标注**。
  **三处改动都是纯注释、无行为变化**，不影响 `graph_version` 与指纹。
- **【静态自检发现，**均已修**（未编译验证）】`TS-052`：1 处 P0 + 1 处 P1**（2026-10-05，REQ-016 静态自检包；
  本机无编译器 / GPU，纯读代码 + 机械配对；完整路径见 `docs/TROUBLESHOOTING.md` 的 `TS-052`）：
  1. **P0 —— `AppendDecodeStep` 把 host 的 `rows` 直接交给设备端 kernel**：`paged_kv_cache.cpp:440`
     把调用方的 host 数组传给 `LaunchAdvanceContextLens`，而 `AdvanceContextLensKernel` 在**设备上**
     解引用 `rows[i]`（契约见 `paged_kv_cache.hpp:113`"rows 是 host 数组"）。**S5 的 packed 路径每一步
     都走**（generation 段传 `generation_rows_host.data()`），S3 路径传 `nullptr` 所以既有用例抓不到。
     **已修（2026-10-05，作者点名"执行 1、3"；未编译验证）**：改用已由 `AppendDecodeKV` 拷好的
     `rows_device_.data()`（一行 + 注释，不动图/profile，**不需要 bump `graph_version`**）；
     回归守卫 = `PagedKVCacheTest.AppendDecodeStepAdvancesMappedRowsOnly`（乱序 + 带洞映射，
     断言设备端 `context_lens` 逐行 +1、未映射行不动、K/V 落点正确）。
  2. **P1 —— `chunk_limit` 用了"总 token 上界"**（`max_prefill_batch × max_prefill_seq_len`，从
     `input_ids` dim1 的 kMAX 查得）而不是 spec §2 / design D16 写的"单序列上限" → ① 真实配置下
     `prompt_len ≤ n_positions ≈ max_prefill_seq_len`，切块条件几乎不可能成立（S5 变死代码）；
     ② 真触发时多行同批的 Σ chunk 会超过 profile 的 T 上界 → `setInputShape` 响亮失败。
     **2026-10-05 作者采纳方向**：不取 A/B 的任一"反推"写法，而是把 **`max_prefill_seq_len` 提成显式
     配置**（`Config` 新字段，与建图侧同名同值）+ **五条交叉校验**（`p5_s5_interface_spec.md` §2 ①～⑤）；
     **P2 文档与代码都已落**（见 `## Implementation Plan` 的"P5-S5 修订一"节，提交 `5a43b6c`）。
  **发现 1 已修**（见上条）；**发现 2 已收口**（P2 文档 + P3 复评（设计层 PASS）+ 代码；`review.md`
  的 P1-1/2/3 三条 P1 均已收口：AC8 限定、P7 口径登记、**钩子退役**）。
- **`TS-051` 的 1 处编译错误 + 5 处缺陷：已修（2026-10-04，作者点名"一并修掉"，提交 `a4dee90`；
  未编译验证）** —— 发现路径见 `docs/TROUBLESHOOTING.md` 的 `TS-051`。逐条（括号里是修法）：
  1. **编译错误（P0）**：`llm_runner.cpp` 的 packed 分支引用 `new_rows`，而声明在 `else` 分支内；
     根因是 `RunPackedMixedStep` 的两个形参（`generation_rows` / `new_rows`）属于 S4 的"generation 行 =
     活跃表前缀"前提，S5 下既无用也不成立（**删掉两个形参**，调用点同步）。
  2. **空活跃表的越界写 + stats 重复累加（P0，每次 packed 调用都走到）**：最后一条序列在**轮首**
     retire 之后，同一轮仍会跑到 ⑤；而早退路径（`b_total <= 0`）不复位
     `sample_active_indices_` / `packed_context_rows_` / `packed_generation_rows_` → ⑤ 拿上一步的行号
     索引**空的 `active`**，并把 token 写到 `d_result_tokens_` 界外（`TS-051` 第 2 条）
     （**函数开头先复位**，`record_token` 再加一道显式越界失败：UB → 报错）。
  3. **`cu_seqlens_ctx` 的声明形状偏小（P1）**：用 S4 的"新入批行数 + 1"，而 S5 里"仍在分块中的行"
     也是 context 行（**改用 `packed_context_rows_ + 1`**）。
  4. **缺 `prompt_len <= n_positions` 的入口拒绝（P1）**：spec §3 与插件注释都假定有，代码里没有
     （**新增 `Config::max_positions`** + packed 模式必填 + 两条上界检查 + 入口按
     `prompt_len + max_new - 1` 拒绝；见下面"偏离设计一处"）。
  5. **收口时新查出**：generation token 的逐行搬运按"活跃表前缀"取行（S4 写法），分块下取错行 ——
     `generated == 0` 时 `src_index = -1`（读结果缓冲之外）并把别人的 token 喂错行
     （**改按 `generation_active` 取行**）。
  6. **收口时新查出**：`cu_seqlens_ctx` 的 profile 行维上界是 `max_prefill_batch`，但它的长度是
     `B_ctx + 1` —— "整批都是 context 行"的首步 `setInputShape` 直接失败
     （**拆出独立范围 `[1, max_prefill_batch + 1]`**；profile 区间进不了指纹 →
     `kPackedPrefillGraphVersion` **5 → 6**，真机首次跑 packed 用例要重建一次引擎）。
  **偏离设计一处**：spec §2 原写"`chunk_limit` 不新增 `Config` 字段"，第 4 条新增了
  `Config::max_positions` —— 理由是 `n_positions` 引擎侧**查不到**（只剩 `ceil(n_positions/block_size)`），
  已在 spec §2 末 / §4 表 / §8 表 3 与 `design.md` D16 记为"唯一例外"。
  **2026-10-05 更新**：这处"唯一例外"**已撤销** —— 作者裁决 **B1+A1**（真值改由引擎侧车给出、
  packed 建图期强制整除），`Config::max_positions` **已删除**（见"P5-S5 修订二"节与 spec §8 表 5）。
  六条**全部未编译验证**；真机窗口第一步是编译 + 重建 packed 引擎。
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
- **S4 已过 P3 增量复评 + 第二遍复评（2026-10-04）**：S4 设计经作者确认 4 条（分路径前提、
  `p5_s3` §10 限定、插件 **A1**、映射与不变量 4 口径）+ 段内下标纪律；第二遍复评又补了
  5 处实现级缺口，并把 `SchedulerStats` 口径与 profile 的 `opt` 两条按推荐定案。
  结论：P0 无、P1 一条（性能类判据绑真机）—— 见 review.md。**S4 可在真机窗口进入实现。**
- **S5 已完成两轮 P3 复评 + 设计修订，Gate-A 已重开（2026-10-04）**：第二遍复评把上一节的
  "P0 无 / PASS"**改判为 BLOCK** —— 查出 3 条会静默算错的落点缺失（chunk 的绝对位置、
  chunked context kernel 的真实形态、采样行集的表达）与 3 条 P1。修订已落 `design.md` D15/D16
  与 `p5_s5_interface_spec.md`，作者同日确认：① `kPackedPrefillGraphVersion` **bump 4 → 5**；
  ② 分块术语**指针式登记**（`analysis.md`，定义以 spec §2 为唯一来源）；③ 三条缺口全部折入
  （`chunk_limit` 改由 `Engine` 的 profile 查询推导、入口拒绝带实际值与上界、兜底纪律按
  "能否在构造期/入口判定"分适用范围）。**结论：S5 设计层可开工；代码一行未动**（见 `## Implementation Plan`）。
- **P4 / P7 搁置（2026-10-03）**：当前不在 GTX 1660 Ti 环境，无法取基线。按
  `phases/p4_baseline.md` 的 Dependency Missing 记 N/A；`benchmark_before.md` 写明环境恢复后
  必须补的四项测量。**批上限（`max_batch`）暂时只能取保守值并标注"待实测"**，不得写成实测结论。
  **`Gate-B: N/A（缺依赖：本机无 GPU / 无 nvcc·cmake·TRT，无法取基线）`** —— 技能要求的
  **三处留痕**（引用 `phases/p4_baseline.md` 的 Dependency Missing，逐项打勾，不转述）：
  ① `benchmark_before.md` 的 `## Result` = `N/A: <原因>` ✅ 已写（含恢复清单）；
  ② `STATE.md` 的 Next Action / Current Blockers 写 `Gate-B: N/A（缺依赖：<原因>）` ✅ 本条；
  ③ `summary.md` 的 Performance 一节写 N/A + 原因 ⏳ **未满足（阻塞）**：该文件是 P8 产物，P4 时点
     不可能产出 —— 已按 `AGENTS.md` §5 第 5 条登记+上报（作者 2026-10-05 指令"全落"，含本条），
     义务落在 Next Action 第 5 条，P8 清。
  （`docs/PROGRESS.md` §5.16 是**附加**留痕，不在技能点名的三处里；2026-10-05 补。）
- **本沙箱无编译能力**：没有 nvcc / cmake / 任何 C++ 编译器，也没有 TensorRT 与 build 目录，
  因此 P5 的 Exit Gate（"编译通过、无新增 warning"）在本环境**无法执行**。
  **S1 的代码改动（含采样器）全部未编译验证。**

---

## Next Action

> **下一步（唯一）**：真机窗口 → `cmake --build build -j` → 跑 `mini_trt_llm_tests` 全量 +
> `LlmRunnerPackedTest.*`(8) / `LlmRunnerSchedulerTest.*`(9) / `LlmRunnerChunkedTest.*`(12) +
> 本轮新增的 7 条（`EngineCacheTest.*` 5 条 host、`PagedKVCacheTest.WritePrefillKV*` 2 条）→
> 按 `test_plan.md` 的 S1/S3/S4/S5 四节清单逐条打勾并回填 `## Actual Result`（P6）。
> **首跑会看到 GPT-2 的引擎各判 `stale` 并重建一次**（`docs/PROGRESS.md` §6.5 已预告，属预期）。
> 下面 1–6 条是**按 S1→S5 排列的历史清单**（每条的细节与证据都在），当前只有上面这一件是真的"下一步"。

1. **S4（代码已落；真机验证待环境）**：作者 2026-10-04 给出"选择性批处理"的口径（两相共享一个 packed 张量、
   attention 按段分派、context token 必须在前），据此已出 **P2 级设计草案**：新建
   `p5_s4_interface_spec.md`（§11 列了 5 条待确认）+ design.md 的 D13 扩充 / S4 小节 / D14。
   **5 条已全部定**（4 条确认 + chunked prefill 立项为 S5），并已过 P3 增量复评（review.md）。
  **下一步**：真机窗口跑 S4 的用例（`LlmRunnerPackedTest.*` 8 条，代码已落、未编译验证）
  并按 §9 的 7 条回填 `test_plan.md`。
2. **S5（chunked prefill，作者 2026-10-04 立项）**：设计已定稿并过**两轮** P3 复评 ——
   `requirement` Included 8 / AC9；`design.md` D15/D16；`p5_s5_interface_spec.md`；`review.md` 的
   两节复评。第二遍复评把上一节的"P0 无"**改判为 BLOCK**（3 条 P0 + 3 条 P1），修订后作者同日确认：
   `graph_version` **bump 4 → 5**、术语**指针式登记**、三条缺口全部折入设计（见 `## Current Blockers`）。
   **子步进度（作者 2026-10-04 分三次点名）**：**S5-1 / S5-2 / S5-3 与 `step_limit` 修正全部已落码，
   全部未编译验证** —— 见 `## Implementation Plan` 的 P5-S5 小节。下一步是**真机窗口**：
  编译（P5 Exit Gate）→ 跑 S4/S5 的用例 → 结果回填 `test_plan.md`（P6）。**进场前先看
  `## Current Blockers` 与 `## 判据对照`**：`TS-051` 六条 / `TS-052` 两条**均已修（未编译验证）**；
  `TS-052` 发现 2 的修订（`chunk_limit` 取显式声明值）已落码（`5a43b6c`）——真机第一步就是编译它。
   **实现顺序（作者 2026-10-04 改判）**：S4 的真机测试先搁置、S5 的代码先做（共用同一张 packed 图）；
   真机窗口恢复后按 S4 → S5 一起验证。**交接用的 `next_session_prompt.md` 已按作者指令删除**，
   本节 + `review.md` 的两节复评即交接入口。
3. **P5-S1（代码已落，待编译）**：改了 `llm_runner.hpp` / `llm_runner.cpp` /
   `sampler_common.hpp` / `sampler_kernels.cu`，新增 `tests/test_llm_runner_batch.cpp`。
   真机下一步：`cmake --build build -j` → 全量 `mini_trt_llm_tests` → 新增的
   `LlmRunnerBatchTest.*`（8 条）。编译错误与用例结果都要回填本文与 `test_plan.md`（P6）。
4. **P5-S2 / S3**：**S3 已全部落码（写回行映射 → 逐行真长度 → 调度循环 → 9 条用例 + 只读观测口），
   全部未编译验证**。S3 真机收口按序做：① 编译（P5 Exit Gate）；② 跑 `mini_trt_llm_tests` 全量 +
   `LlmRunnerSchedulerTest.*` 9 条；③ 结果回填 `test_plan.md`（P6）。
5. 环境恢复后补 P4，再按 D10 的两种负载跑 P7。**注意（2026-10-05）**：`TS-052` 发现 2 的修订
   （`chunk_limit` 改由显式声明派生）会让**分块真的启用**（此前几乎不触发）→ 默认路径的行为画像变了，
   P7 结果必须按"**性能未验证 / 不声称收益**"的口径呈现，AC6 不得凭空结。
   **P8 义务（2026-10-05 登记）**：`summary.md` 的 Performance 一节必须写 `N/A` + 原因 —— 它是
   `phases/p4_baseline.md` 的 Dependency Missing **三处留痕里的第 3 处**（P4 时点不可能产出，
   故记"未满足（阻塞）"并在此锚定义务）。恢复后要量的**五项**见 `benchmark_before.md` 的 `## Result`
   （第 5 项 = S5 的 chunk 维度对照）。
6. **`n_positions` 的来源：已裁决（作者 2026-10-05 选 **B1+A1**）** —— **A1**：真值（解析自
   `config.json`）作为 `numeric_params` 的新项写进 `<engine>.fingerprint`，runner 构造期读回 +
    一致性自检（`ceil(N / block_size)` 与 `block_tables` 的 dim1 相符、`N <= 1024`、
   `max_prefill_seq_len <= N`），**sidecar 缺失 / 损坏即拒绝启动**；**B1**：packed 建图强制
   `n_positions % block_size == 0`（硬失败 + spec §5 记一条，**只限 packed**）；`Config::max_positions`
   **删除**。评估与排除记录（V1–V5）见本文件末尾的核查清单 —— 真机核查动作**已取消**。
   **代码已落**（2026-10-05，未编译验证）—— 见 `## Implementation Plan` 的
   「P5-S5 修订二：`n_positions` 走 B1+A1」；真机只剩"编译 + 跑 3 条新用例"。

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
| ③ 用例 | `tests/test_llm_runner_packed.cpp`（8 条：`PackedEqualsSequential` / `MixedStepContextAndGeneration` / `ContextTokensPrecedeGeneration` / `CuSeqlensBoundaryCases` / `PackedWriteBackMapsCorrectly` / `PackedMetadataFollowsPackedOrder` / `PackedShortSequenceNotPenalized` / `FallbackSwitchKeepsResults`）+ `test_plan.md` 的 S4 行 | **已落**，未编译 |

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

### P5-S5（**进行中**：S5-1 已落码、未编译验证，2026-10-04）

**Implementation Plan（技能 P5 要求：当前模块 / 预计文件 / 测试方式）**

- **S5-1（本次，插件侧）—— 已落码（未编译验证）**
  - **实改**：`packed_attention_plugin.cu` 的 `PackedContextAttentionKernel` 换成 chunked 分页因果版
    （新增 `key_cache` / `value_cache` / `block_tables` / `context_lens` / `block_size` /
    `max_blocks_per_seq` 六个参数；`key_count = cache_len + pos + 1`；K/V 按 `t < cache_len`
    分流到分页缓存或本 chunk；**同一个循环里分支**而不是两段循环 —— `cache_len == 0` 时线程认领的
    t 序列与 S4 完全一致，因此与不分块基线**逐位相同**）；`enqueue` 的两处 launch 同步补齐实参；
    `.hpp` 的输入契约注释改为 S4+S5 双语义（并写明 `context_lens` 两段都取"推进前"的值）；
    `builder.cpp` 的 `kPackedPrefillGraphVersion` 4 → 5（插件语义变更同提交 bump）。
  - **未做（属 S5-2）**：`chunk_limit` 的推导与入口拒绝（需要 `Engine` 的 profile 查询）、
    runner 侧的切 chunk / 绝对位置 / 采样行集、cache 侧的写回起点与累加记账。
  - **当前模块**：TensorRT 插件（context 段的注意力语义）+ 图代次；**3 个文件**，改动量 ~90 行
    （技能 P5 的 soft 约束是 <=3 文件 / <=300 行）。
  - **测试方式**：本环境无 nvcc / cmake / TRT，改动**全部标未编译验证**；用例按 spec §6 在 **S5-3**
    落地（届时二选一：新建 `tests/test_llm_runner_chunked.cpp` 或并入 packed 用例文件，并说明理由），
    真机窗口先编译再跑。
  - **边界**：**不动**输入个数与顺序、不动 `getWorkspaceSize`、不动 profile；`chunk_limit` 的推导与
    入口拒绝属 S5-2（runner 侧），插件只保留建图期的 `n_positions <= 1024` 校验。
- **S5-2（**已落码，未编译验证**；已提交 `120bbe0` + `fc2a973`）**
  - **S5-2a**：`Engine` 加只读 profile 查询（`GetProfileDims` / `GetProfileDim`）；`LLMRunner` 构造期从
    `input_ids` 第 1 维 `kMAX` 推导 `chunk_limit_`（**不新增 Config 字段**）；加**测试专用**
    `SetChunkLimitOverride`（默认 0 = 不覆盖）；推导失败 / 值 <= 0 / 覆盖值超 profile 上界 →
    **构造期直接拒绝**。
  - **S5-2b**：`RunPackedMixedStep` 按 **cache 已写入长度**分类（`written >= prompt_len` ⇒ generation 段，
    不新增 `prompt_done` 字段）→ `chunk_len = min(prompt_len - written, chunk_limit_)` → 写**绝对位置**
    （`written + i`）→ 写回带 `row_starts = written`（第 2 块起从 `prompt_done` 续写；cache 层同步把
    host 记账改成累加、预留量按累计末端校验）→ **采样行集紧凑暂存** `sample_active_indices_`
    （槽位 = 数组下标，升序活跃行号）→ 结果落位 / EOS 回读 / `SchedulerStats` 三处同口径。
  - **原设计缺口（"完成的行被未完成的行隔开时怎么进 generation 段"）按作者选定的方案 A 收口**：
    `AppendDecodeKV` / `AppendDecodeStep` / 推进 `context_lens` 的 kernel 都接受显式 `rows`
    （`nullptr` = 恒等 → S1/S2/S3 行为与开销不变），**不改行序**；文档同步在 `d232d05`。
- **S5 剩余（P6 之前）—— 作者 2026-10-04 点名的两项都已落码（未编译验证）**：
  ① **S5-3 用例**：新建 `tests/test_llm_runner_chunked.cpp`（`LlmRunnerChunkedTest.*`，8 条，
  与 `test_plan.md` 的 S5 一节同名同序：`ChunkedEqualsWholePrompt` / `ChunkedPositionsAreAbsolute` /
  `ChunkBoundaryDoesNotDisturbOthers` / `ChunkedShortPromptsUnchanged` / `ChunkProgressStateIsCorrect` /
  `ChunkedSamplingRowSetIsCompacted` / `ChunkedRetireAndBlocks` / `ChunkLimitRejectedConfigs`）。
  **为什么新建文件而不是并进 packed 用例文件**：本组要"**一个引擎文件 + 每个切法一个 runner**"
  （`chunk_limit` 只在构造期读一次），packed 文件的夹具只有单 runner；且 test_plan 已按
  `LlmRunnerChunkedTest.*` 登记、`tests/CMakeLists.txt` 用 `file(GLOB *.cpp)` 自动收源文件（不改构建脚本）。
  ② **`step_limit` 计入分块步数**：packed 路径下加
  `chunk_step_slack = max_prompt × request_count`（= Σ `ceil(prompt_len/chunk_limit)` 的松上界，
  与 `chunk_limit` 取值无关、永远够用）；**只在 packed 下放宽**（S3 两段式一步送完 prompt，
  放宽它只会白白损失对死循环的敏感度）。守门用例是 `ChunkedEqualsWholePrompt` 的 `limit=1` 分支
  （12 个分块步 vs 老上界 9 步 —— 老上界下这条用例会以 `scheduler step limit exceeded` 收场）。

### P5-S5 收口（**作者点名"一并修掉"，提交 `a4dee90`；未编译验证**）

**当前修改模块**：packed 路径的边界（逐行状态复位 / 结果缓冲不越界）与 profile 区间
（`cu_seqlens_ctx` 的行维范围），外加位置表上界的入口拒绝。

| 文件 | 实际改动 |
|---|---|
| `include/.../core/llm_runner.hpp` | `Config::max_positions`（位置表长度；packed 模式必填）；`RunPackedMixedStep` 去掉 `generation_rows` / `new_rows` 两个形参 |
| `src/core/llm_runner.cpp` | 构造期 `max_positions` 校验（必填 + ≤ 池/块表容量 + ≤ 插件上限）；入口按 `prompt_len + max_new - 1` 拒绝；`RunPackedMixedStep` 开头复位逐行状态；`record_token` 越界显式失败；generation token 按 `generation_active` 取行；`cu_seqlens_ctx` 形状用 `packed_context_rows_ + 1` |
| `src/core/builder.cpp` | `cu_seqlens_ctx` 的 profile 范围拆出 `[1, max_prefill_batch + 1]`；`kPackedPrefillGraphVersion` **5 → 6** |
| `tests/test_llm_runner_packed.cpp` | 夹具补 `max_positions = kPositions`（接口变更的机械后果） |
| `tests/test_llm_runner_chunked.cpp` | 夹具带 `max_positions`；`ChunkLimitRejectedConfigs` 扩成四组（含"未声明 / 越界"与入口拒绝的正反对照）；`ChunkedEqualsWholePrompt`、`ChunkedShortPromptsUnchanged` 补 `generation_rows` 断言（`TS-051` 第 2 条的回归守卫） |

**测试方式**：同 S5-3 —— 本环境只做静态自检（括号逐行深度、符号成对、最长行、CRLF 无 BOM）；
真机窗口 `cmake --build` + 跑 S4/S5 用例。**新增真机前提**：packed 引擎按 `graph_version` **重建一次**——
本节写此条时该值是 6，**2026-10-05 已被 REQ-017 推到 7**（`builder.cpp` 顶部注释的 `6 → 7`，属预期），
真机按**当前值 7** 预期即可（重建仍只发生一次）。

### P5-S5 修订一：`max_prefill_seq_len` 显式配置（**文档与代码均已落**，2026-10-05）

**背景**：`TS-052` 发现 2 —— `chunk_limit` 原本从 profile 的**总 token 上界**反推（`input_ids` dim1
kMAX = `max_prefill_batch × max_prefill_seq_len`），既让分块在真实配置下几乎不触发，又会在真触发时
让多行同批越出 profile。作者采纳"显式配置 + 交叉校验"方向并提出四点修订（字段的非 packed 语义、
1024 的来源、"单一来源"的准确口径、"policy < cap"的触发条件）。

**当前修改模块**：packed 的 chunk 策略与构造期校验（`LLMRunner`）+ 两个 packed 测试夹具。

| 文件 | 计划改动 |
|---|---|
| `include/.../core/llm_runner.hpp` | 新增 `Config::max_prefill_seq_len`（契约：**只在 packed 模式有语义、非 packed 必须留 0**；`chunk_limit` 的唯一来源）；**退役** `SetChunkLimitOverride` / `ChunkLimitOverride` |
| `src/core/llm_runner.cpp` | 构造期五条交叉校验（§2 ①～⑤，拒绝时打印实际值与上界）；`chunk_limit_ = config_.max_prefill_seq_len`（删掉 override 全局量与"从 profile dim1 推导"的旧写法） |
| `tests/test_llm_runner_chunked.cpp` | 夹具改按字段传三种切法（删 `ScopedChunkLimitOverride`）；`ChunkLimitRejectedConfigs` 扩为"未声明 / 超上界 / 违反交叉校验 ③ / 超 1024"四组 |
| `tests/test_llm_runner_packed.cpp` | 夹具补 `max_prefill_seq_len = kPositions`（与建图侧同源） |

**后续修订（同日更晚）**：本节引入的 `Config::max_positions` 已被 **B1+A1** 取代并**删除**
（见下一节）—— 交叉校验 ② 的上界改为**侧车真值**，`ChunkLimitRejectedConfigs` 的两组"声明值
越界 / 入口位置拒绝"随之并入池容量检查。本表保留为那一轮的**计划记录**，不代表当前接口。

**测试方式**：同 S5-3（静态自检 + 真机窗口）。**不动建图 / profile 区间 → 保持 `graph_version = 6`，
不需要再 bump**。**前置**：本节的 P2 文档修订（spec §2/§4/§5/§6/§8/§9、design D16、test_plan 的 S5
注记）与 **P3 式增量复评**（`review.md` 的第三遍复评，**设计层 PASS**）**都已落**；`requirement.md`
的 AC8 按作者选的 (a) 加了限定；`analysis.md` 补了 5 条术语指针；
`review.md` 的 P1-3 已由作者确认为**退役**（删 `SetChunkLimitOverride` / `ChunkLimitOverride`）。
**代码已落（2026-10-05，未编译验证）**：`llm_runner.hpp/.cpp`（新字段 + 五条构造期校验 + 钩子退役）、
`test_llm_runner_chunked.cpp`（夹具改构造期传参 + `ChunkLimitRejectedConfigs` 重写）、
`test_llm_runner_packed.cpp`（夹具补字段）。
**当时登记的遗留（已收口）**：交叉校验 ③ 在小夹具里无法独立触发（上界检查先拦）→ 已由独立用例
`ChunkLimitCrossCheckRejectsOverStepBudget`（`da54400`）解决；编译与真机用例仍待环境。

### P5-S5 修订二：`n_positions` 走 **B1+A1**（**文档与代码均已落，未编译验证**，2026-10-05）

**背景**：`TS-051` 第 4 条的位置上界原由调用方声明（`Config::max_positions`）。作者 2026-10-05
裁决 **B1+A1**：真值改由**引擎侧车**给出（A1），并在 packed 建图期加**整除硬失败**（B1），
同时**删除** `Config::max_positions`。完整口径见 `p5_s5_interface_spec.md` §2/§5、`design.md` D16、
`review.md` 的第四遍复评。

**当前修改模块**：引擎侧车的读写（`engine_cache` / `Engine` / `builder`）+ packed 建图期约束
（`gpt2_model_builder`）+ runner 构造期读回与自检（`LLMRunner`）+ 3 条新用例。

| 文件 | 实际改动 |
|---|---|
| `src/core/engine_cache.cpp` | 新增 `ReadEngineSidecarField(path, key)`：逐行比对**完整行键**（数值项的字面量是 `num.<名字>`），只读 `---` 之后的正文；缺文件 / 缺键 → 空串 |
| `include/.../core/engine.hpp` / `src/core/engine.cpp` | 新增 `const std::string& Path()`（构造期记下引擎文件路径）—— runner 靠它定位 `<engine>.fingerprint` |
| `src/core/builder.cpp` | `numeric_params` 新增 `{"model.n_positions", <解析自 config.json 的真值>}`（`TryGetModelPositions` 静默失败时**不写这一项**，该引擎的 runner 会拒绝启动）；图与 profile **不动** |
| `src/core/gpt2_model_builder.cpp` | packed 分支在进入层循环前硬失败 `n_positions % block_size != 0`，错误信息带实际值 / 上界 / 建议因数；**padding 路径不受影响** |
| `include/.../core/llm_runner.hpp` | **删除** `Config::max_positions`；新增成员 `n_positions_`；`max_prefill_seq_len` 的契约注释同步（② 的上界改为侧车真值） |
| `src/core/llm_runner.cpp` | 构造期读侧车 + 两项自检（`ceil(N/block_size)` 对 `block_tables` dim1、`N <= 1024`）；读不到 / 不自洽 → 拒绝启动；交叉校验 ② 改用 `n_positions_`；入口位置上界同样用它 |
| `tests/test_llm_runner_packed.cpp` | 夹具去掉 `max_positions`（接口变更的机械后果） |
| `tests/test_llm_runner_chunked.cpp` | 夹具去掉 `max_positions`；`ChunkLimitRejectedConfigs` 收敛为 (A)(B) 两组；**新增 3 条**：`MissingFingerprintSidecarRejected` / `SidecarPositionsMismatchRejected` / `NonDivisibleBlockSizeRejected`（第三条含 padding 反向对照） |

**测试方式**：同 S5-3（沙箱只做静态自检：逐行括号深度、符号成对、最长行、CRLF 无 BOM）。
**不动建图 / profile 区间 → `graph_version` 保持 6**。**指纹加项的失效范围（2026-10-05 复核后写清）**：
`MakeFingerprintInputs` 是**所有 stage 共用**的 → 凡是 `config.json` 带 `hyper_params.n_positions` 的
模型（GPT-2 的 config 与 ONNX 两条路径）所建的**全部引擎各自失效、各自重建一次**（`single` 一份、
或 `prefill` + `decode` 两份，分钟级/份）——**不是**"只重建 packed 那一份"；`hyper_params` 里没有
这一项的模型（CV/resnet18、若干最小测试模型）指纹不变、不重建。它**不 bump `graph_version`**：
变的是缓存键内容，图与 profile 一个字没动。真机窗口：`cmake --build` → 跑 S4/S5 全部用例 + 3 条新用例。

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
- 2026-10-04: **S5 第二遍复评（作者要求）**：核对对象从"设计自洽"改为"设计与代码现状对得上"，
  改判上一节的"P0 无 / PASS"为 **BLOCK** —— 3 条 P0（chunk 的**绝对位置**、chunked context kernel
  的**真实形态**、**采样行集**的表达）+ 3 条 P1（`graph_version` 裁决、常量集合口径、写回记账口径）。
  同轮修订 `design.md`（D16 重写、D15 补第 ⑤ 件、验证策略补 AC7/8/9、Requirement Coverage 补
  Included 7/8、不变量补第 6 条）与 `p5_s5_interface_spec.md`（§2～§9）
- 2026-10-04: **作者确认 S5 第二轮**：① `kPackedPrefillGraphVersion` **bump 4 → 5**；
  ② "分块 / 末块 / chunk 的绝对位置" **指针式登记**进 `analysis.md`（定义以 spec §2 为唯一来源）；
  ③ 第二遍复评查出的三条缺口全部折入设计（`chunk_limit` 改由 `Engine` 的 profile 查询推导、
  入口拒绝带实际值与上界、兜底纪律按"能否在构造期 / 入口判定"分适用范围）
- 2026-10-04: 作者指令**删除** `next_session_prompt.md`（未跟踪文件，不可从 git 恢复；其中"三条
  git 提交坑"未另处留档，其余内容已并入 design / spec / review）；同作者指令**回滚**单笔合并提交
  `313e2b4`，改按 Phase 拆三笔：`c87af79`(P3) / `90e64c4`(P2) / `272e245`(P1)
- 2026-10-04: **作者点名"做 S5-1"** → 写 Implementation Plan（本节 P5-S5）后落码：
  `packed_attention_plugin.cu` 的 context kernel 换成 **chunked 分页因果**（K/V = 分页缓存
  `[0, context_lens[seq])` ++ 本 chunk 自包含，因果边界 `cache_len + pos`；`cache_len == 0` 时
  与 S4 逐位相同）、`enqueue` 补六个实参、`.hpp` 契约注释更新、`builder.cpp` 的
  `kPackedPrefillGraphVersion` **4 → 5**。**全部未编译验证**（本环境无 nvcc / cmake / TRT）；
  用例留在 S5-3。自检：花括号 / 圆括号 / 方括号平衡，最长行 101 < 200，CRLF 无 BOM，
  `PackedContextAttentionKernel` 定义与两处调用成对
- 2026-10-04: **作者按序点名"完成 1、2"**（① test_plan 补 S5 用例 + 定 `chunk_limit` 注入口径；
  ② 做 S5-2）→ 全部落码并**分三笔提交**：`d232d05`（文档：test_plan S5 一节 + 下标纪律精确化）、
  `120bbe0`（runner 侧：Engine profile 查询、`chunk_limit_` 推导与测试钩子、切 chunk、绝对位置、
  采样行集紧凑暂存）、`fc2a973`（cache 层：`row_starts` 写回起点 + generation 段显式行映射）。
  **原设计缺口由作者选定方案 A 收口**（显式行映射，不改行序）。**全部未编译验证**；
  剩余：S5-3 用例、`step_limit` 计入分块步数（见 P5-S5 节的"S5 剩余"）
- 2026-10-04: **作者点名"S5 剩余（P6 之前）两项"** → ① `step_limit` 把分块步数计进防呆上界
  （packed 路径加 `max_prompt × request_count`，只在 packed 下放宽）；② 新建
  `tests/test_llm_runner_chunked.cpp`（`LlmRunnerChunkedTest.*`，8 条，与 test_plan 的 S5 清单
  同名同序）。**全部未编译验证**。同一轮逐行读 S5-2 的代码，发现 **1 处编译错误（`new_rows` 跨分支引用）
  + 3 处缺陷（空活跃表越界写 + stats 重复累加、`cu_seqlens_ctx` 声明形状偏小、缺
  `prompt_len <= n_positions` 入口拒绝）**，按 §0.7 **未改产品代码**，登记进 `## Current Blockers`
  与 `docs/TROUBLESHOOTING.md` 的 `TS-051`，等作者定夺
- 2026-10-04: **作者点名"一并修掉"** → 收口 `TS-051`：删 `RunPackedMixedStep` 的两个 S4 形参
  （消掉编译错误 + 修掉"按活跃表前缀取 generation token"）、逐行状态每步先复位（消掉空活跃表
  越界写 + stats 重复累加）、`cu_seqlens_ctx` 形状与 profile 行维范围（`[1, max_prefill_batch + 1]`）、
  新增 `Config::max_positions` + 入口位置上界拒绝。`kPackedPrefillGraphVersion` **5 → 6**；
  S5-3 用例补 `generation_rows` 回归断言与 `max_positions` 的拒绝对照。**全部未编译验证**
- 2026-10-05: `chunk_limit` 改由**显式声明** `Config::max_prefill_seq_len` 派生（`TS-052` 发现 2 的
  收口）：`5a43b6c`（代码 + 五条交叉校验、`SetChunkLimitOverride` 退役）、`ccc3992`/`0ce2862`
  （P4 的三处留痕 + 判据对照）、`da54400`（交叉校验 ③ 的独立触发用例）、`1408bf0`（编号与时效修正）。
  **全部未编译验证**
- 2026-10-05: **作者裁决 `n_positions` 的 → B1+A1**（`Config::max_positions` 删除）→ 文档落盘
  （spec §2/§4/§5/§8 表 5、`design.md` D16、`review.md` 第四遍复评、`test_plan.md` 的 3 条新用例）+
  代码落盘（`engine_cache` 读字段 helper、`Engine::Path()`、`builder.cpp` 的 `model.n_positions`、
  `gpt2_model_builder.cpp` 的整除硬失败、runner 读回与两项自检、两个夹具与 3 条新用例）。
  **不动建图 / profile → `graph_version` 保持 6**；指纹加项 → 带 `hyper_params.n_positions` 的模型的
  **全部 stage** 引擎各自重建一次（范围与理由见 `## Implementation Plan` 的"P5-S5 修订二"）
- 2026-10-05: **`TS-053`（静态自检：host 指针进设备侧的全量对账）** —— 逐个核对 kernel 实参 /
  `SetTensorAddress` / 采样器 launch / 建图期权重指针的可读侧：**0 处 P0**（`TS-052` 发现 1 的修法
  在册）；登记三处**注释 / 契约级**缺口（S4 残留注释、公开入口里 `rows`/`row_starts` 是 host 而
  `cu_seqlens_ctx` 是设备、采样器指针字段未标"设备可读"）→ **作者点名"先处理缺口"后三处均已改**
  （纯注释，无行为变化；按 `cpp-comment-style` 复核）
- 2026-10-05: **`TS-054`（静态自检：`rows` 的"默认恒等"全量对账）** —— 四处消费者 ×（6 生产 + 15 tests）
  调用点逐个核对：**0 处不满足**（两处 `nullptr` 都有代码级依据）；发现 1 处**潜在陷阱**
  （`row_starts` 在非 packed 路径被静默忽略）→ **作者点名"先修潜在陷阱"后已修**：两道闸
  （`WritePrefillKV` 入口拒绝 + `LaunchWriteKV` 兜底）+ 两条用例（正向首次覆盖 `row_starts`、反向拒绝），
  **未编译验证**
- 2026-10-05: **`TS-055`（静态自检：四节用例清单的"同名同序"核对）** —— S1/S3/S4/S5 四节 ↔ 四个测试
  文件的 `TEST` 名逐位比对：名字与条数本来就一致（10/9/8/12），**顺序**在 S1/S3/S4 三节不同 →
  把清单顺序统一到文件，并在 `test_plan.md` 的 Integration Test 写明这条口径（另修掉 `Expected Result`
  里陈旧的"8 条"）；**纯文档改动**
- 2026-10-05: **产物完备性盘点（作者点名清单第 6 项）** —— 按技能 `workflows/feature.md` 的
  `## Required Artifacts`（10 件，唯一来源）逐件对现状：**7 件在、3 件缺**（`benchmark.md` /
  `summary.md` / `interview_notes.md`），三件都卡在阶段顺序与外部依赖（P7 需 before + 独占 GPU、
  P8 需 P7、P9 需 P8），**不是漏产**；另 5 件 `p5_s*_interface_spec.md` 是 P5 接口细化（不在那 10 件里）。
  本盘点**只读**，未新建产物；`## Completed Artifacts` 已从旧版 6 行清单改写成这张 10 行表，
  `updated` 字段同步到 2026-10-05

---

## Recovery Notes

- **流程纪律：批准的是意图，执行的是它的闭包（2026-10-05 复盘，作者要求落档）** —— 本 feature 连着
  两轮踩同一个坑：把**需要改契约的改动**和**贴合既有设计的改动**捆在一批里执行，于是"该不该先改
  文档"在批内没有唯一答案，事后整批看起来都像回填。注意"补文档"这个词本身是误导的：对"贴合既有
  设计"那一类，先代码后文档就是正常的事后记账（与回填 STATE 同性质）；真正出错的是**混批**。
  五条可检验的做法：
  1. **报 finding / 报方案时必须展开"兑现它的闭包"**：输入从哪来、是否新增对外字段 / 接口、是否改
     profile 或图版本、是否需要新权限。缺输入的写成"**待定：缺 X，选项 A/B**"，**不要**写成
     "加一条检查" —— 本次 `TS-051` 第 4 条正是被压缩成一行检查之后才被批准的，你批准的文字与
     我执行的闭包之间的差集，就是偏离的全部来源。
  2. **动手前分类，不同类不混批**：判据 = 是否修改 spec / design 的"已定"条目、是否新增对外字段或
     接口、是否 bump 图版本、是否改 profile。命中任一 → **先确认、先改文档、再改代码**；其余可以
     代码先行、文档事后记账。同批改动必须**同质**（这一条不是"少做事"，而是保证批内对"文档先行"
     只有一个答案）。
  3. **探索性缺陷只报不做**：改 A 时发现 B（尤其动接口 / 图版本的），停下报"新发现 + 是否仍在原授权
     范围内"，不"顺手修"。本次第 6 条（`cu_seqlens_ctx` 的 profile 上界）把 `graph_version` 从
     5 推到 6，而 5 是作者当天拍板的 —— 这类必须先问。
  4. **评审式阅读的停止条件** —— **总则**见 `AGENTS.md` §5「计划对账」第 5 条 + 技能
     `phases/p3_review.md` 的 Entry（找不到来源记 P0，且**停止推演实现方案**）；本 feature 实例：
     `TS-051` 第 4 条（`prompt_len > n_positions` 被压缩成"入口加一条检查"才被批准）。
  5. **自我披露必须落载体** —— **总则**见 `SKILL.md` 的「会话收尾（Handoff）」+
     `templates/STATE.md` 的 Recovery Notes 说明；本 feature 实例：上一轮的"计划对账补记"没有载体，
     下一轮就升级成"改了作者拍过板的条目"。

  （④⑤ 的措辞权威在总则，这里只做**指针式登记** —— 见技能 `SKILL.md` 的 Mandatory Rules 第 7 条。）
- **设计欠债（不是流程问题，2026-10-05 登记）**：`p5_s5_interface_spec.md` §3 的适用范围表把
  `prompt_len > n_positions` 列为"可判定 / 必须显式拒绝"，`packed_attention_plugin.cu` 的注释也写
  "runner 入口已有这条"，但**没有任何一处写 `n_positions` 从哪来**。也就是说：即使当时按第 4 条
  停下回报，spec 里也没有现成答案可抄 —— 这条债最后以"实现时才暴露"的形式到期。**遇到这类条目要
  按"设计欠债"登记（并被当成设计缺口去补），不要只记成流程失误。**
- **已裁决（作者 2026-10-05 选 **B1+A1**，取代同日的"待决策"）**：`n_positions` 的来源定案为
  **A1**（真值来自**引擎侧车**：建图期把解析出的 `n_positions` 写进 `<engine>.fingerprint` 的
  `num.model.n_positions`，runner 构造期读回 + 一致性自检）**+ B1**（packed 建图强制
  `n_positions % block_size == 0`，硬失败），并**删除 `Config::max_positions`**。两点都写死在
  spec §2/§5 与 `design.md` D16；评估与排除记录（V1–V5）留在本文件末尾的核查清单里。
  **裁决推翻了同日的初判**：初判是"V1/V2/V3 都不通过 → 保持 A"，而裁决走的是**不依赖 TRT 回读
  插件属性的注入路径**（写进我们自己的侧车）—— 这条初判当时没被列入 V1–V5，复盘结论是
  "**枚举候选时要连'我们自己可控的载体'一起列**"，否则会把"没有已知可行方案"误当成"只能保持现状"。
  **真机核查动作（打印 I/O shape / 导出 Inspector JSON）已取消**；真机只剩"编译 + 跑 3 条新用例"。
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
  **2026-10-06 作者裁决（取代上句）**：串行顺序 = **本 feature → REQ-018 → REQ-017 → REQ-019 的 S0/S2**；
  "收口" = 编译通过 + 沙箱用例全绿（真机部分继续记搁置）；冻结面与理由的**唯一详述**见
  `docs/dev/REQ-019-onnx-subgraph/design.md` 的「跨条目串行顺序」。
- **复核过、仍成立的既有事实**：
  1. 元数据缓冲设备分配容量够就复用；正确性依赖"decode 每步重绑"，不是"指针不变"。
  2. 引擎把 cache 第 0 维声明成 `ceil(n_positions / block_size)`，同一个数又当块表宽度用；
     运行时期物理池可以更大，此关系此前未进任何契约。
  3. 随批增长的显存大头是 **prefill logits**（`S=512` 约 98 MiB/条），不是 K/V 池（64 块约 72 MiB）。
  4. 建图代码不进引擎指纹，**改图必须手工 bump `graph_version`**。
  5. 运行期常驻诊断（同步 D2H + 逐层扫 K/V）**没有任何开关**——D7 要解决的是它。

---

## 判据对照

> 口径（技能 `SKILL.md` 的 Mandatory #8 / #10）：只在**异常路径**（出现"不适用 / 未满足 / 放行"）时写；
> 行数 = 该阶段判据数；清单**只引用出处、逐项打勾**，不转述。本 feature 目前**只有 P4 有异常项**，
> 其余阶段全绿（按轻量口径不写）。

| 判据（出处：`phases/p4_baseline.md` 的 `### Dependency Missing`） | 状态 | 证据 / 放行 |
|---|---|---|
| `benchmark_before.md` 记 `N/A: <原因>`（缺依赖时视作通过本自检） | 满足 | `benchmark_before.md` 的 `## Result`（原因 + 恢复清单五项） |
| 留痕 ①：`STATE.md` 的 Next Action / Current Blockers 写 `Gate-B: N/A（缺依赖：<原因>）` | 满足 | `## Current Blockers` 的"P4 / P7 搁置（2026-10-03）"那条（字面行 2026-10-05 补） |
| 留痕 ②：`summary.md` 的 Performance 一节写 N/A + 原因 | **未满足（阻塞）** | P8 产物，P4 时点不可能产出。**放行记录**：作者 2026-10-05 指令"全落"（含本条）→ 放行范围 = **允许 P4 以"阻塞"状态继续**，义务锚定在 `## Next Action` 第 5 条，**P8 必清** |
| 留痕 ③：`docs/PROGRESS.md` 的"已知问题与坑"留一条（性能未验证及原因） | 满足 | `docs/PROGRESS.md` §5.16（2026-10-05 补） |

---

## 待决策核查清单（④ `n_positions` 的来源）

> **状态：已裁决（作者 2026-10-05 选 **B1+A1**；裁决记录见 spec §8 表 5、评审见 `review.md` 的
> 第四遍复评）。本节**转为决策记录，真机核查动作取消**（不再需要打印 I/O shape / 导出 Inspector JSON）**
> —— 下面 V1–V5 保留为"评估与排除"的留档。
>
> **原用途（留档）**：这是 `## Recovery Notes` 里"待决策"那条的**未展开项**（"建图侧注入需要 runner 能把插件 /
> 图属性读回来，而 TRT 10.15 是否有这条公开路径**未验证**"）的展开。**真机窗口里在"编译之后、
> 重建 packed 引擎之前"执行**（顺序见 `## Next Action` 第 5/6 条）——若结论会改接口，先改再建引擎，
> 避免白建一次。编号用 **V1–V5**（`①②③④⑤` 只归 spec §2 的交叉校验，不与之撞号）。
>
> **判定口径（先写死，免得结论含糊）**：
> - **"精确"** = 读回来的值等于建图时的 `cfg.n_positions`；`ceil(n_positions / block_size)` 这类
>   **取整上界不算通过**（那正是当前池检查用的界）。
> - **"可复现"** = 核查必须留下**引擎文件 + 导出产物 + 命令**（指针写进 `TROUBLESHOOTING.md` 或本节），
>   不接受"我看过 API 文档"这种无据结论。
> - **取舍判据**（作者 2026-10-05 已定）：语义正确性 —— `n_positions` 是**图属性**；
>   "是否撞 spec §2 的字段禁令"**不作为**判据。

**已知前提（省得重复查）**：(a) 引擎侧唯一带 `n_positions` 痕迹的现成 I/O 是
`ceil(n_positions / block_size)`（cache 第 0 维 / `block_tables` 第 1 维）→ **上界、非精确**；
(b) 我们**自己**的插件把 `max_seq_len` 作为属性序列化进了引擎，问题只在"TRT 是否给了读回路径"；
(c) runner 构造期能拿到的只有 `Engine`（`ICudaEngine` + `IExecutionContext`）与引擎文件路径，**没有**
model dir / `config.json`。

| # | 核什么 | 怎么核；判定标准 |
|---|---|---|
| **V1** | `ICudaEngine` 的**插件层枚举 + 属性读回**：是否有 `getNbLayers()` / `getLayerInformation(idx, kJSON / kONELINE)`；该 JSON 对插件层是否含 `PluginName` / `PluginNamespace` / **插件字段值（如 `max_seq_len`）** | **预期不通过（作者 2026-10-05 依官方文档判定）**：Inspector 的 JSON 里 **"layer parameters" 指标准层参数**（conv 的 kernel/padding/stride、激活类型…），**不含自定义插件的序列化属性** —— PluginField 只用于"反序列化时恢复插件状态"，不是配置查询接口；遍历所有层大概率只看到插件层的 name / 输入输出形状 / 精度。**仍要留证据**：`NvInfer.h` 里 `class ICudaEngine` 的**行号** + 一次导出 JSON 里插件层的**实际字段列表** |
| **V2** | `IEngineInspector` 在 **kDETAILED** 下输出的**权重 / 常量形状**（`runtime->createEngineInspector(*engine)`）：JSON 里有形如 `"Weights": {"Type": …, "Count": N}` 的字段；我们要的是 wpe 的 `[n_positions, hidden]` | **技术上存在，但收益与风险不成正比（作者 2026-10-05）**：① **wpe 未必是可辨认的独立层**（可能被融合或识别成 Constant，取决于建图方式与 TRT 版本）；② **动态维在未绑定 context 时显示为 `-1`** —— 要拿精确值必须先建 context、设好输入 shape、再 `inspector->setExecutionContext(context)`，实现复杂度上升；③ **JSON schema 跨版本会变**（TRT 11.x 已破坏性变更：Bindings → 结构化 I/O Tensors、Format/DataType 拆分）→ 若走这条路，design **必须写明"升级 TRT 时 JSON 解析逻辑要回归"**。**验证成本硬约束：若现有引擎的 `profilingVerbosity` 不是 `kDETAILED`，不许为了验 V2 重建引擎** —— 直接标"需重建才能验；决策阶段不做" |
| **V3** | 现有 I/O 张量里有没有**精确**载体 | 列全部 I/O 的 `getTensorShape` + profile **min/max/opt** dims（`input_ids` / `position_ids` / `block_tables` / `context_lens` / `cu_seqlens_ctx` / `context_seq_count` / `key_cache_i` / `value_cache_i` / `k_layeri` / `v_layeri` / `logits`）。**预期不通过**：`input_ids` / `position_ids` 是 `[batch, 批内实际长度]` 或扁平的 `[total_tokens]`（**不是**模型支持的最大位置数）；`block_tables` 第 1 维与 cache 第 0 维都是 `ceil(n_positions / block_size)`（= 前提 (a) 的**上界**）；其余也不承载它。**仍必须打印每个张量的维与数值当证据** |
| **V4** | 我们自己可控的"注入"手段：**V4a** 建图时加一个**常量输出**（如单元素 `positions_len`，形状里带 n_positions）；**V4b** 把 cache 第 0 维改成精确载体 | **V4a = 唯一能得到"精确、稳定、跨版本可靠"值的手段**（作者 2026-10-05）。代价：I/O 契约变更 → **必须 bump `graph_version`**；`mini_trt_llm/tools/inspect_engine.cpp` 等工具**必须同步**（否则 I/O 数量不匹配会报错或静默忽略）；引擎多一个无计算作用的输出（性能影响可忽略，内存布局微变）。**硬约束：不要为了"验 V4a"或"为 V4a"单独建一次引擎** —— 在下一次**本来就需要的**重建里一并改（bump + 工具同步同轮完成）。**V4b 仅记录、不建议实施**（改 cache 的内存布局与语义，影响面比 V4a 大得多） |
| **V5** | 若 V1–V3 都不通过（**预期就是这个结果**） | **保持 A**（调用方声明 + 五条交叉校验）：把 V1/V2/V3 的证据（`ICudaEngine` 头文件行号、导出 JSON 里插件层的字段列表、全部 I/O 的形状打印）写进 `design.md` 的决策记录 —— 下次不再重问。**例外（业务触发条件）**：若**无法接受** A 的唯一残留缺口 —— "runner 声明 < 建图声明"这条**安全方向**的不一致（结果仍对，只是分块更细）—— 且 runner 构造期**必须**精确知道 `n_positions` → 选 **V4a**，并**并入下一次重建** |

**方案成本对照（供裁决）**：

| 方案 | 若通过需要改什么 | 影响面 |
|---|---|---|
| **A（现状）** | 无 | 调用方必须声明；"两侧声明不一致"的**安全方向**（声明 < 建图）发现不了 |
| **B1 = V1** | `Engine` 加"读插件属性"helper；runner 构造期改用它；`Config::max_positions` 删除或降级为可选覆盖 | 接口变更（删/改字段）→ 走 P2/P3 |
| **B2 = V2** | 同 B1，但依赖 JSON 解析（随 TRT 版本脆弱） | 同 B1 + 脆弱性 |
| **B3 = V4a** | 改建图（加哑输出）→ bump `graph_version`；工具同步 | 最大（图 + 引擎重建 + 工具） |

**执行顺序（作者 2026-10-05 定，真机窗口"编译之后、重建引擎之前"）**：

1. **先跑 V3**：打印全部 I/O 的 `getTensorShape` 与 profile min/max/opt dims —— 预期确认"无一精确等于 `n_positions`"，**留作证据**。
2. **同轮做 V1 / V2 的静态检查（不重建引擎）**：`trtexec --dumpLayerInfo --profilingVerbosity=detailed --exportLayerInfo=<json>` 导**现有引擎**的 JSON，人工看插件层有没有字段值（V1）、wpe 形状可不可见（V2）。**若现有引擎的 `profilingVerbosity` 不是 `kDETAILED` → 直接标"需重建才能验；决策阶段不允许重建"**，不为此建引擎。
3. **若 V1 / V2 / V3 都不通过（大概率）**：**保持 A**，把证据写进 `design.md` 的决策记录（含"V1 在结构上不可能：Inspector 不含插件属性"这条）。
4. **若不能接受 A 的残留缺口、且必须精确**：选 **V4a**，把"改图 + bump `graph_version` + 工具同步"**并入下一次本来就需要的重建**，避免单独为它建一次引擎。
