# Phase 2 补丁计划：诊断输出开关（F2）+ CUDA 环境显式判定

> **状态**：**P2S-1 ~ P2S-4 已完成并真机复验**（2026-09-25，用户确认"不做 3c / 采纳闸门 / 跑真机全量"）；
> P2S-5 文档收口已完成；§8 有完整结果。**新增发现**见 §8 的"计划外"一栏。
>
> **定位**：`625939c` 为定位 FP16 NaN 在 GPT-2 图上留了 4 个诊断输出，但没有显式关闭手段，
> 导致 `kPrefill` / `kDecode` 引擎的 I/O 契约被改变、真机 6 条用例变红（实测见
> `docs/TROUBLESHOOTING.md` #19）。本计划把诊断输出改成**默认关闭的显式开关**（F2），
> 并补一道"CUDA 环境不具备"与"环境具备但初始化失败"的判别闸门。
>
> **与既有文档的分工（唯一来源原则，见 `PROGRESS.md` §2.13）**：
> - **缺陷的事实、证据、实测输出** → `docs/TROUBLESHOOTING.md` #19（本文件不复述细节）；
> - **要做什么、顺序、验收、破坏性清单** → 本文件；
> - **结果与当前状态** → 执行完后回填本文档 §8，并更新 `docs/PROGRESS.md`。

---

## 0. 计划对账（AGENTS.md §5 第 0 步：动手前必须回答的四件事）

### 0.1 有没有计划文档

**没有。** 现有文档都不覆盖本次改动：

| 文档 | 覆盖了什么 | 是否覆盖本次任务 |
|---|---|---|
| `docs/TROUBLESHOOTING.md` #19 | 缺陷的现象 / 根因 / 三重证据 / 6 条实测红 | ❌ 只记事实，**不定义修复方案与验收** |
| `docs/phase2_test_plan.md` §5（G2-1） | FP16 端到端的缺口与结论 | ❌ 不涉及诊断输出开关 |
| `docs/phase2_development_plan.md` | Phase 2 的建模与 runner 任务 | ❌ 本缺陷发生在该阶段**收口之后** |
| `docs/future_iterations.md` §1.4 | FP16 NaN 的后续解决路径 | ❌ 不涉及开关 |

→ 因此**先产出本文件并等确认**，不直接开工。

### 0.2 逐条对照：任务 / 接口 / 验收 vs 现状

| 本次要做的事 | 与现状对照 | 结论 |
|---|---|---|
| 给诊断输出加默认关闭的开关 | 现状：4 处 `markOutput` 硬编码在 `gpt2_model_builder.cpp` 的 `export_kv && layer == 0` 分支里，无开关 | **偏差**：需新增构建期参数（`BuildOptions`）并从 `EngineBuilder::Config` 穿下来 |
| `getNbOutputs()` 恢复为 5 | 现状实测 9（2026-09-25 真机，`--gtest_filter='Gpt2NetworkBuildTest.*'`） | 一致：目标就是回到 5，**断言本身不用改**（这正是选 F2 的理由：期望值本来就对） |
| 回归 9 条用例全绿 | 现状：6 红（2 建网 + 3 decode-consistency + 1 runner）、3 对照绿，见 #19 | 一致 |
| CUDA 判定用 `cudaGetDeviceCount` | 现状：`test_gpu_guard.hpp` 用 `cudaGetDevice()`；`test_gpt2_network_build.cpp` 用 `createInferBuilder()==nullptr → GTEST_SKIP` | **偏差**：判定口径要改，且"有设备但建不出 builder"必须从 skip 改成 fail |

### 0.3 偏差怎么处理

**先改文档、再改代码**：本文件（新增）→ `docs/phase2_test_plan.md` 的 L1 行修正 → 再动代码。
同时把新约定写进 `PROGRESS.md` §2.15 的"勿回改"表（见 §2 的 P2S-5）。

### 0.4 反向查：文档里与代码现状矛盾之处（本次当场修）

| 矛盾 | 证据 | 处理 |
|---|---|---|
| `tests/test_gpt2_network_build.cpp:73-74` 注释写"建网络不需要 CUDA 设备，可以在沙箱 / CI 里跑" | 本轮实测：沙箱内 `createInferBuilder` 报 `Error Code 6 ... CUDA initialization failure with error: 35` → 返回 null → `GTEST_SKIP` | **以实测为准改注释**；这条错误假设正是缺陷藏身处（它把"本该红"的断言伪装成"环境不具备"） |
| `docs/phase2_test_plan.md` §3 的 L1 行写"真机（`createInferBuilder` 需要 CUDA，实测确认）" | 与上一条直接矛盾 | 保留这条（它是对的），并在测试注释里指向它 |
| `PROGRESS.md` 曾写"真机只有 1 条预期失败" | 已实测 6 条红 | 已在 §5.11 / §6.5 更正；本计划执行完再按结果复述一次 |

---

## 1. 判据来源（期望值凭什么是这个数）

本补丁的判据全部是**计数与布尔值**，不涉及精度阈值，因此不存在"放宽阈值"的空间：

| 判据 | 出处 |
|---|---|
| 默认构建下 `network->getNbOutputs() == 2 * n_layer + 1`（= 5） | 测试既有断言（`test_gpt2_network_build.cpp:147` / `:274`），来源 `docs/phase2_test_plan.md` §4.6 的 I/O 契约 |
| 打开开关后多出 4 个输出：`mlp_fc_0` / `mlp_gelu_0` / `attn_res_0` / `mlp_res_0` | `docs/TROUBLESHOOTING.md` #18.1 轮次 3/4（当时的人工读数） |
| `kSingle` 切面永远是 1 个输出 | `test_gpt2_network_build.cpp:184`（既有对照） |
| 沙箱内 GPU 用例**允许**跳过 | `PROGRESS.md` §5.7 / §5.10：无 GPU 的 CI 必须能绿，否则真实回归信号被固定噪声淹没 |

---

## 2. 任务分解

### P2S-1 诊断输出开关（F2 的代码面）

**做什么**：

1. `include/mini_trt_llm/core/imodel_builder.hpp`：`BuildOptions` 增
   `bool export_diagnostics = false;`
2. `include/mini_trt_llm/core/builder.hpp` + `src/core/builder.cpp`：
   `EngineBuilder::Config` 增同名开关，`BuildFromConfig` 里透传给 `BuildOptions`。
3. `src/core/gpt2_model_builder.cpp`：4 处 `markOutput` 的条件由
   `export_kv && layer == 0` 改为 `options.export_diagnostics && export_kv && layer == 0`。

**为什么把开关放在这两处**：本项目所有构建期参数（precision / workspace / profile 范围）
都已统一走 `Config → BuildOptions` 这条路径；不给 `BuildFromConfig` 再加默认参数，是为了
不继续膨胀 §2.15 里已冻结的签名。默认值必须是 `false`——它是**契约变更**，默认打开等于
把"要不要多 4 个输出"这个决定推给每一个调用方。

**验收**：默认构建下 `getNbOutputs() == 5`；打开开关后为 9 且 4 个名字可读。

### P2S-2 诊断仪器改为显式 opt-in，并换独立引擎路径

**做什么**：`tests/test_gpt2_generate.cpp::Fp16PrefillOutputsDiagnostic` 里
`builder_config.export_diagnostics = true`，引擎路径由
`/tmp/mini_trt_llm_gpt2_real_prefill_fp16.engine` 改为
`/tmp/mini_trt_llm_gpt2_real_prefill_fp16_diag.engine`。

**为什么要换路径（不是洁癖）**：引擎缓存**只按路径名区分、不随代码或开关失效**。原有的
`_fp16` 路径同时被"要诊断输出"的仪器与"不要诊断输出"的 NaN 复现器使用——两者需求相反，
先跑谁就会把后跑的那条测成假结果。这与 #19 里"跑之前必须删缓存"是同一个坑，换路径是根治。

**验收**：诊断用例仍能列出 4 个诊断输出；NaN 复现器走默认（关闭）路径。

### P2S-3 CUDA 环境显式判定（消除"假绿滑梯"）

**做什么**：

1. `tests/test_gpu_guard.hpp` 重写为显式探测：
   - `ProbeCudaDevice()`：以 **`cudaGetDeviceCount`** 为主要判定（`err != cudaSuccess ||
     count <= 0` 即不可用），同时取 `cudaDriverGetVersion` / `cudaRuntimeGetVersion` 与
     `cudaGetErrorString` 的文本，打包成可打印的一行；
   - `HasCudaDevice()` 语义不变（兼容既有约 30 处调用）；
   - 新增 `SkipIfNoCuda()`：默认 `GTEST_SKIP()` 并**打印探测结果**；当环境变量
     `MINI_TRT_REQUIRE_GPU=1` 时改为 `GTEST_FAIL()` —— 这是"真机 / 带 GPU 的 CI 上任何一次
     跳过都算失败"的闸门，直击"静默跳过 → 假绿"。

   本轮已实测两种环境下的探测值，作为该机制的依据：

   | 环境 | `cudaGetDeviceCount` | `cudaDriverGetVersion` |
   |---|---|---|
   | 沙箱 | `err = 35 (cudaErrorInsufficientDriver)`，`count = -1` | `0`（无驱动可见） |
   | 真机 | `err = 0`，`count = 1` | `12060` |

2. `tests/test_gpt2_network_build.cpp::SetUp()` 改三分支：
   - 无设备 → `SkipIfNoCuda()`（带探测结果）；
   - **有设备但 `createInferBuilder` 返回 null → `ASSERT_NE(builder_, nullptr)` 直接 FAIL**
     （这是环境/驱动故障，不是"环境不具备"；原先把两者合成一个 skip 正是缺陷的藏身处）；
   - 有设备且 builder 可用 → 正常建网。
3. 新增一条**永不跳过**的用例 `GpuEnvProbe.ReportsCudaAvailability`：打印设备数 / 驱动版本 /
   是否设了 `MINI_TRT_REQUIRE_GPU`，让每次 ctest 输出里都有一行明确的环境事实。
4. 语义边界（写进注释与验收）：**沙箱里 GPU 用例仍然跳过、ctest 仍然全绿**（§5.7 的要求）。
   本任务解决的是"跳过要显式、且不能把真故障伪装成跳过"，**不是**"把跳过变红"。

**可选（P2S-3c，待你定）**：给 GPU 套件打 CTest label，便于 `ctest -L gpu`。代价是
`gtest_discover_tests` 单次注册下按套件打 label 需按前缀拆成多次注册，收益与"真机直接跑
测试二进制"重叠——**我倾向不做，除非你希望 CI 里用 label 选测**。

### P2S-4 真机回归验证

**做什么**：重新编译后，在真机跑下列 9 条（6 条先前红 + 3 条对照）：

```
Gpt2NetworkBuildTest.PrefillNetworkBuildsWithExpectedIo
Gpt2NetworkBuildTest.DecodeNetworkBuildsWithPagedAttentionInputs
Gpt2NetworkBuildTest.SingleStageOmitsKvOutputs                 (对照)
Gpt2NetworkBuildTest.MissingWeightFailsTheBuild                (对照)
Gpt2DecodeConsistencyTest.DecodeStepMatchesPrefillAtSamePosition
Gpt2DecodeConsistencyTest.DecodeWithEmptyCacheMatchesSingleTokenPrefill
Gpt2DecodeConsistencyTest.TwoStepDecodeMatchesPrefillAfterAppend
Gpt2GenerateTest.RunnerMatchesFullRecomputeWithoutCache
Gpt2GenerateTest.RejectsUnsupportedTemperature                 (对照)
```

**前置**：确认 `/tmp/mini_trt_llm_gpt2_*.engine` 无遗留（本轮已查：真机上**没有**这些缓存，
修复后首跑会全量重建，分钟级）。

**附加判据（很关键、别漏）**：修复后 `Gpt2GenerateTest.RealGpt2Fp16GreedyMatchesReferenceTokens`
若仍以 **enqueue 失败**告终，说明开关没穿到底；它应当回到"NaN → 8 个 token 全 0"这一
**按设计红**的形态。这条同时验证"开关默认关闭"与"LLMRunner 路径恢复"。

**可选**：再跑一次真机全量（含 2 条 `RealGpt2*`，分钟级），确认除上述 1 条按设计红外无新增红。

### P2S-5 文档收口

1. `docs/TROUBLESHOOTING.md` #19：状态改"已修复"，补"回归防护"（哪条用例/断言现在能拦住
   同类改动：`getNbOutputs()` 计数断言 + `MINI_TRT_REQUIRE_GPU` 闸门）。
2. `docs/PROGRESS.md`：§2.15 增两行"勿回改"约定——**诊断输出必须显式 opt-in、默认关**、
   **引擎缓存路径不得在两种契约间共用**；同步 §5.11 / §6.5 / 表头第 4 条的失败计数。
3. `docs/phase2_test_plan.md`：G2-1 行补"诊断仪器已改为 opt-in"；修正 L1 行与测试注释的矛盾。
4. 本文件 §8 回填执行结果。

### P2S-6 修复 #20：`PagedKVCacheTest` 的追加用例与 `AppendDecodeStep` 契约对齐

**状态**：⏳ **待确认**（#20 由真机全量跑出来，不在原计划清单内，按 §0.5 重新列清单）。

**问题回顾**（细节见 `TROUBLESHOOTING.md` #20）：用例 `AppendCrossesBlockBoundaryAndAdvancesContextLens`
的 cache 配置是 2 层（`MakeConfig()`），却只给 `AppendDecodeStep` 传了 1 对 K/V，与该 API
"每层一对、一次写全部层"的契约（#16 定下）冲突 → 自 d6af2eb 起就不可能通过。

**做什么（方案 A：只改测试，不动产品代码）**：

1. **按契约传每层一对**，并让每层数据不同：
   - layer 0 的 step 值沿用现在的 `100 + step + idx`，layer 1 用 `200 + step + idx`；
   - 每层各分配一份 `DeviceBuffer`，调用改成
     `AppendDecodeStep({d_key[0], d_key[1]}, {d_value[0], d_value[1]}, nullptr)`。
2. **把断言从"只看 layer 0"补成两层都验**：现有跨块断言（位置 3 / 4）对 layer 0 保持不变；
   新增对 `cache.key_cache(1)` 的同位置断言，期望值换成 layer 1 的基址。
   **为什么值得补**：只绑 layer 0 时，这条用例其实漏掉了 `AppendDecodeStep` 的核心语义
   （"每层写自己的那一段"）；#15（各层共用同一 cache 张量）正是这个形态的 bug。
3. **新增一条负例用例** `AppendDecodeStepRejectsLayerCountMismatch`：
   传 1 对而配置 2 层 → 断言返回 `cudaErrorInvalidValue`，**并且** host 侧
   `SequenceLength(0)` 与设备端 `context_lens` 都必须保持不变。
   **为什么要这条**：拒绝路径不能留下半推进的长度，否则一次非法调用会污染后续推理；
   当前实现是先校验后写，这条断言把这个顺序钉住。

**明确不做**：

- **不改** `MakeConfig()` 的 `num_layers = 2`——另一条用例依赖"第 1 层保持 0"这一事实；
- **不**给产品代码加"单对自动广播到所有层"的分支：真实调用方（`llm_runner.cpp:524`、
  `test_gpt2_decode_consistency.cpp:637`）都传全层，加广播只会让"漏了某层"这种 bug 静默通过；
- 不动 `AppendDecodeKV` 的单层语义（它是 runner 之外的细粒度入口，契约本身没问题）。

**验收判据**：

| # | 判据 | 出处 |
|---|---|---|
| B1 | 真机 `PagedKVCacheTest.*` 全绿（3 条既有 + 1 条新负例） | 该用例文件既有判据 + #16 契约 |
| B2 | 真机全量从 **2 红 → 1 红**，仅剩 `RealGpt2Fp16Greedy...`（按设计红） | `PROGRESS.md` §6.5 的口径 |
| B3 | 新负例能抓住"拒绝后长度仍被推进"——即把实现改成先写后校验时它会红 | 反向验证（可选，手工确认一次） |
| B4 | 沙箱仍 145 条 0 失败（GPU 用法例照常跳过） | §5.7 |

**破坏性动作**：覆盖修改 `mini_trt_llm/tests/test_paged_kv_cache.cpp`（1 处调用 + 1 条新用例）；
文档 3 处（`TROUBLESHOOTING.md` #20 状态、`phase2_test_plan.md` 的 `PagedKVCacheTest.*` 状态行、
`PROGRESS.md` 的失败计数）；本文件 §8 追加一行。**不删文件、不动产品代码、不碰 git。**

---

## 3. 改动清单（文件级）

**新建**

| 文件 | 内容 |
|---|---|
| `docs/phase2_supplement_plan.md` | 本文件 |

**修改（覆盖）**

| # | 文件 | 改动 |
|---|---|---|
| 1 | `mini_trt_llm/include/mini_trt_llm/core/imodel_builder.hpp` | `BuildOptions::export_diagnostics = false` |
| 2 | `mini_trt_llm/include/mini_trt_llm/core/builder.hpp` | `EngineBuilder::Config::export_diagnostics = false` |
| 3 | `mini_trt_llm/src/core/builder.cpp` | `BuildFromConfig` 透传开关 |
| 4 | `mini_trt_llm/src/core/gpt2_model_builder.cpp` | 4 处 `markOutput` 加开关条件（注释指向 #19） |
| 5 | `mini_trt_llm/tests/test_gpu_guard.hpp` | `ProbeCudaDevice()` / `SkipIfNoCuda()`，主判定改 `cudaGetDeviceCount` |
| 6 | `mini_trt_llm/tests/test_gpt2_network_build.cpp` | SetUp 三分支（含 builder 失败→FAIL）；修掉错误注释 |
| 7 | `mini_trt_llm/tests/test_gpt2_generate.cpp` | 诊断用例 opt-in + 独立引擎路径 |
| 8 | 约 10 个测试文件的 `GTEST_SKIP` → `SkipIfNoCuda()`（机械替换，保留各自附加条件） | `test_cuda_check` / `test_e2e_*` / `test_fp16_paths` / `test_gpt2_*` |
| 9 | `mini_trt_llm/tests/CMakeLists.txt` | **仅当采纳 P2S-3c** 时改（label / 注册 `GpuEnvProbe`） |
| 10 | `docs/TROUBLESHOOTING.md` / `docs/PROGRESS.md` / `docs/phase2_test_plan.md` | 见 P2S-5 |

**删除**：无文件删除。**不执行** `git commit` / `git push`（按 AGENTS.md §0.2 归你）。

---

## 4. 验收判据

| # | 判据 | 出处 |
|---|---|---|
| A1 | 默认构建 `getNbOutputs() == 5`；`kSingle` 仍为 1 | 既有断言（§1） |
| A2 | 打开开关的仪器路径能列出 4 个诊断输出 | #18.1 轮次 3/4 |
| A3 | 真机 9 条用例**全绿**（6 条先前红 + 3 条对照） | 用户本轮要求 |
| A4 | 真机 `RealGpt2Fp16Greedy...` 回到"token 全 0"的按设计红，**不再是 enqueue 失败** | #19 / §18.1 |
| A5 | 沙箱 `ctest` 仍 **0 失败**，且 GPU 跳过条目打印显式探测原因（`cudaGetDeviceCount` 结果） | §5.7 与用户本轮要求 |
| A6 | 沙箱内 `createInferBuilder` 失败不再表现为 skip，而是 FAIL | 本计划 P2S-3 |

---

## 5. 破坏性动作清单（一次性确认，AGENTS.md §2.14 B）

1. **新建** `docs/phase2_supplement_plan.md`（本文件，已落盘等你确认）。
2. **覆盖修改** §3 表里的 1–8 共 8 个源文件 / 测试文件（其中 #8 是批量机械替换）。
3. **覆盖修改** 3 份文档（`TROUBLESHOOTING.md`、`PROGRESS.md`、`phase2_test_plan.md`）。
4. **可选**：`mini_trt_llm/tests/CMakeLists.txt`（仅当采纳 P2S-3c）。
5. **不删除任何文件**；**不删 `/tmp` 引擎缓存**（已确认真机无遗留）；**不碰 git 历史**。
6. 真机回归会**构建新引擎**（写入 `/tmp`，分钟级），并在 `build/` 下重编译。

> 与 §0.5 的关系：批准本计划 = 批准上述**这批具体动作**；执行中若需要新增动作（例如发现
> 还得改别的文件），我会**停下来重新列清单**，不自行扩大范围。

---

## 6. 范围外（明确不做）

- **不修 FP16 NaN 本身**（`future_iterations.md` §1.4 的路径，不在本次）。
- **不改 `kSingle` 的语义**，不动 `PagedAttention` / 采样器 / KV Cache。
- **不引入引擎缓存的失效机制**（代码版本/开关参与缓存 key）——本计划只用"换路径"绕开，
  真要根治属于测试基建改造，另开任务。
- **不把 GPU 用例的跳过整体改成失败**（会破坏 §5.7 的 CI 可绿要求）。

---

## 7. 风险与回滚

| 风险 | 判据/缓解 |
|---|---|
| 开关穿透不完整（某条路径没传） | A3 + A4 同时看：9 条全绿**且** FP16 复现器回到 NaN 形态 |
| 诊断用例与 NaN 复现器抢同一个引擎缓存 | P2S-2 的独立路径；A2 验仪器仍可用 |
| GPU 判定改动让沙箱误红 | A5：沙箱必须仍 0 失败 |
| 机械替换 `GTEST_SKIP` 时改坏条件（如丢掉"模型目录不存在"分支） | 替换后逐文件 diff 复核；沙箱用例数与跳过数应保持不变（当前 144 / 63 跳过 / 81 执行） |
| 回滚 | 改动全部是本轮新增的开关与判定逻辑，`git checkout` 上述文件即可回到当前状态 |

---

## 8. 执行结果（已回填，2026-09-25）

| ID | 状态 | 实际产出 |
|---|---|---|
| P2S-1 | ✅ 完成 | `BuildOptions::export_diagnostics` + `EngineBuilder::Config::export_diagnostics` 透传；`gpt2_model_builder.cpp` 4 处 `markOutput` 由开关控制。默认构建输出数回到 5（真机实测） |
| P2S-2 | ✅ 完成 | `Fp16PrefillOutputsDiagnostic` 打开开关并改用 `/tmp/mini_trt_llm_gpt2_real_prefill_fp16_diag.engine` |
| P2S-3 | ✅ 完成 | `test_gpu_guard.hpp` 重写（`ProbeCudaDevice` / `MINI_TRT_SKIP_IF_NO_CUDA` 宏 / `MINI_TRT_REQUIRE_GPU`）；`Gpt2NetworkBuildTest::SetUp` 三分支；新增 `GpuEnvProbe.ReportsCudaAvailability`；19 个测试文件、59 处跳过点统一（比计划的"约 10 个文件"多——凡是 `HasCudaDevice()` 的跳过点都在内，否则闸门会半生效） |
| P2S-3c | ⛔ 按用户决定不做 | 未引入 CTest label |
| P2S-4 | ✅ 完成 | 真机全量 `MINI_TRT_REQUIRE_GPU=1 ctest`：145 条、**0 跳过**、538 s、2 条红（见下表） |
| P2S-5 | ✅ 完成 | #19 转"已修复 + 回归防护"、新增 #20、`PROGRESS.md`（§2.13/§2.15/§3.0c/§3.5/§5.11/§6.5/表头）、`phase2_test_plan.md`（L1 更正 + G2-1） |
| P2S-6 | ✅ 完成 | `test_paged_kv_cache.cpp`：追加调用改每层一对 + 两层读回断言 + 新负例（层数不匹配必须被拒且不留副作用）。真机 `PagedKVCacheTest.*` 4/4 通过；真机全量 **146 条 / 1 红**（仅 FP16 按设计红） |

### 8.1 验收判据逐条结果

| 判据 | 结果 |
|---|---|
| A1 默认 `getNbOutputs() == 5`；`kSingle` 为 1 | ✅ 真机 `Passed`（prefill 与 decode 两条建网用例 + `SingleStageOmitsKvOutputs`） |
| A2 打开开关的仪器仍能列出 4 个诊断输出 | ✅ `Fp16PrefillOutputsDiagnostic` Passed，读回 `mlp_fc_0=11.5`、`mlp_gelu_0=11.5`、`attn_res_0=13.86`、`mlp_res_0=95.44`（与 #18.1 记录逐位一致） |
| A3 真机 9 条全绿 | ✅ 6 条先前红全部转绿，3 条对照仍绿 |
| A4 FP16 复现器回到"按设计红" | ✅ 失败原因是 NaN（`首个含 NaN/Inf 的层 = 1`、"第 7 个 token 不符"），**不再是 enqueue 失败** |
| A5 沙箱仍 0 失败且跳过带显式原因 | ✅ 145 条 0 失败 / 63 跳过 / 82 执行；每条跳过都带 `cudaGetDeviceCount` 结果 |
| A6 有设备但建不出 builder 判失败 | ✅ 代码路径改为 `ASSERT_NE(builder_, nullptr)`（沙箱路径由 A5 覆盖；真机上 builder 正常，未触发该分支） |

### 8.2 计划外的发现（已修：P2S-6）

真机全量跑出一条与本补丁无关的红：
`PagedKVCacheTest.AppendCrossesBlockBoundaryAndAdvancesContextLens` ——
用例传 1 对 K/V，而 `AppendDecodeStep` 的契约（TROUBLESHOOTING #16 定下）要求每层一对、
该配置是 2 层。`git log -S` 显示用例与 `AppendDecodeStep` **同源于 d6af2eb**，
也就是说它自诞生起就不可能通过，只是沙箱一直跳过、文档又写着"✅ 真机"。
已登记为 `docs/TROUBLESHOOTING.md` **#20**（含修法与影响面）。**修复动作不在本计划清单内，等你确认。**

> **后续（2026-09-25）**：该修复已获用户确认并按 **P2S-6**（见 §2）完成并复验——
> 真机 `PagedKVCacheTest.*` 4/4 通过、全量 146 条只剩 1 条按设计红。
> 唯一未做的是可选项 B3（临时改产品代码验证负例有效），原因记在 `TROUBLESHOOTING.md` #20。

**待用户确认的点**：

1. ~~是否采纳 P2S-3c~~ → **不做**（用户已定）。
2. ~~闸门是否可接受~~ → **可接受**（用户已定），已实测生效。
3. ~~回归范围~~ → **真机全量**（用户已定），已完成；由此发现 #20。
4. ~~#20 是否现在修~~ → **已确认并完成**（P2S-6）。
