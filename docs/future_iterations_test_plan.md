# future_iterations 测试计划

> **状态**：2026-09-26 产出，**未执行**（表内状态列一律为"未开始"）。
>
> **定位**：与 `docs/future_iterations_development_plan.md` 配套。**条目内部的目标 / 做法 / 验收**
> 以 `docs/future_iterations.md` 为唯一来源（§1.5 / §1.6 / §0.1），本文件只负责把它落成
> "用例 → 判据 → 出处 → 环境 → 状态"。
>
> **判定口径**（沿用本项目惯例）：**未跑过的不写"通过"**；"跳过"必须显式且打印探测结果；
> 阈值必须能回答"凭什么这么定"（`AGENTS.md` §7）。

---

## 1. 分层与环境

**层名刻意新起**（不与 phase 计划的 `L0~L3`、`S/E1~E4`、`R0.x~R3.x` 混用）——本项目已经因为
层名冲突吃过一次"同一编号指两件事"的亏（`docs/phase1_5_test_plan.md` 开头的说明）。

| 层 | 含义 | 能在沙箱跑吗 | 归属 |
|---|---|---|---|
| **H** | host 契约 / 纯算法（无 GPU、无 TRT 运行时） | 能，进 CI / ctest | 本计划的**主体**（tokenizer、规格护栏、参考自证） |
| **G** | 真机（GPU + TRT）：kernel / engine / 端到端 | 不能（无设备 → 显式跳过） | 用户在 WSL2 真机执行 |
| **P** | 真机性能（需可复现口径） | 不能 | 仅 §4 的 C 组涉及，且必须先按缺口 **G6** 建测量方法 |

**运行口径**：

```bash
# 沙箱 / CI：host 用例实际执行，GPU 用例显式跳过
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc) && ctest --test-dir build

# 真机：GPU 用例跳过即失败（不带这个变量等于白跑）
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure
```

**当前基线**：沙箱 **215 条 / 0 失败**（2026-09-26 实测；**89 条跳过、126 条实际执行**，
含本计划新增的 `int8_eval_selftest`、`tokenizer_golden_check`、`int8_crosscheck_selftest`、
批次 A 的 16 条 host 用例，以及 B / C 两条**真机用例在沙箱显式跳过**）；
真机口径（`MINI_TRT_REQUIRE_GPU=1`）**2026-09-26 全量重跑：215 条 / 1 红**（唯一红 = FP16 按设计红）；
更早那次的 204 条 / 2 红 是修正判据之前的快照，其中 ONNX argmax 那条已按方案 B 结案——
红 1 = `RealGpt2Fp16GreedyMatchesReferenceTokens`（GPT-2 FP16 已知限制的按设计红，见 `docs/PROGRESS.md` §5.11）；
**红 2 = `Gpt2OnnxTest.MatchesAcrossProfileShapes`（seq=512 逐行 argmax；数值项都通过）——
新红、原因未知、保持红**，见 `docs/TROUBLESHOOTING.md` #34 与 `PROGRESS.md` §5.13。
新增用例后基线总数会变，**改动时必须同步回填 `PROGRESS.md` §3.5**。

---

## 2. 批次 A：现在就能开工的用例

### 2.1 A1 = BPE Tokenizer（对应 `future_iterations.md` §5.1）

**参考实现与出处**：`transformers 4.44.0` 的 `GPT2TokenizerFast`，**离线**从本地 HF 快照加载
（`local_files_only=True`，快照 `models--gpt2/snapshots/607a30d...`）。2026-09-26 实测：
`vocab_size = 50257`，`"The quick brown fox"` → `[464, 2068, 7586, 21831]`。
参考数据由 `tools/make_tokenizer_golden.py` 导出为 `tests/data/gpt2_tokenizer_golden.json`（含环境与 SHA256）。
**另有实现期追加的一项前置**（见 A1-11）：`utils/json.hpp` 原本不支持 `\uXXXX`，而 `vocab.json`
的 key 全是这种转义——补它需要自己的一份覆盖。

| 用例 ID | 用例名（拟） | 层 | 判据 | 出处 | 前置 | 状态 |
|---|---|---|---|---|---|---|
| A1-1 | `BpeTokenizerTest.LoadsFromDirectory` | H | 传入含 `vocab.json` + `merges.txt` 的目录 → `Load` 返回 true，`VocabSize() == 50257` | HF 快照实测 | tokenizer 目录 | ✅ 通过（2026-09-26） |
| A1-2 | `BpeTokenizerTest.RejectsMissingOrMalformed` | H | 目录不存在 / 缺 `merges.txt` / 坏 JSON / merges 行无空格 → 返回 false（**不抛异常、不静默降级**） | 契约要求 | 临时目录 | ✅ 通过（4 条负例） |
| A1-3 | `BpeTokenizerTest.EncodeMatchesGoldenBasic` | H | 与 golden **逐 token 全等**（不是率） | golden（HF 参考） | golden 文件 | ✅ 通过（3 个样本） |
| A1-4 | `BpeTokenizerTest.EncodeMatchesGoldenWhitespace` | H | 同上；样本含**前导空格 / 连续空格 / 换行 / 制表符** | 同上；这四类是 GPT-2 `Ġ` 语义最易错处（实现时**两次**踩坑，见 `TROUBLESHOOTING.md` #33.2 / #33.3） | golden | ✅ 通过（5 个样本） |
| A1-5 | `BpeTokenizerTest.EncodeMatchesGoldenUtf8` | H | 同上；样本含中文、emoji、中英混排、重音拉丁、日文、西里尔、全角字母数字、CJK 标点、ZWJ 序列、货币符号（byte-level 回退路径 + "非 ASCII 算不算字母"这个近似点） | 同上 | golden | ✅ 通过（10 个样本，一次全过） |
| A1-6 | `BpeTokenizerTest.EncodeMatchesGoldenLongText` | H | 同上；样本 ≥256 token，覆盖多次 merge 的深路径（实测 306 token） | 同上 | golden | ✅ 通过 |
| A1-7 | `BpeTokenizerTest.DecodeMatchesGolden` | H | 对 golden 给出的 ids 解码 → 文本与参考**全等**；并确认参考自身 `decoded == text` | 同上；**只对参考 ids 判**（任意 ids 不可逆是 BPE 固有性质） | golden | ✅ 通过（15 个样本） |
| A1-8 | `BpeTokenizerTest.EmptyStringAndEdgeCases` | H | 空串 → 空 ids；`Decode({}) == ""`；仅标点样本与 golden 全等 | 同上 | golden | ✅ 通过 |
| A1-9 | `BpeTokenizerReferenceTest.GoldenIsSelfConsistent` | H | **参考自证**：来源字段齐全、`local_files_only == true`、`vocab_size == 50257`、每个 id 落在词表内、样本结构完整 | `PROGRESS.md` §2.13（参考实现必须自带断言） | golden | ✅ 通过。**来源可复现性**由 ctest 项 `tokenizer_golden_check` 承担（重新用 HF 算一遍再比对，3.21 s） |
| A1-10 | `BpeTokenizerWithRunnerTest.TextPromptMatchesReferenceTokens` | **H**（原定 G，见下） | `Encode("The quick brown fox")` **逐 token 等于** `test_gpt2_generate.cpp` 里的 `kExpectedPrompt = {464, 2068, 7586, 21831}` | 既有常量（`tests/test_gpt2_generate.cpp`，注释即 `kPromptText`）；runner 侧数值路径由既有 `Gpt2GenerateTest.RealGpt2GreedyMatchesReferenceTokens`（真机）覆盖 | tokenizer 目录 | ✅ 通过 |
| A1-11 | `JsonParserTest.*`（6 条） | H | 公共 JSON 解析器的 `\uXXXX` 支持：单转义 → 正确码点；代理对 → **单个**增补平面码点（UTF-8 必须 4 字节）；孤立代理 / 非法十六进制 → 抛错；ASCII 转义不回归 | `TROUBLESHOOTING.md` #33.1（A1 的前置障碍）；改公共件必须有覆盖 | 无 | ✅ 通过（6 条，2026-09-26 **实现期追加**） |

**A1-10 的层级从 G 改成 H（2026-09-26，动手时确定的偏差）**：原计划要它在真机上跑一遍
"文本 prompt → 8 个贪心 token"，但那条链路的数值部分**已经**由 `RealGpt2GreedyMatchesReferenceTokens`
覆盖，再搭一次双引擎不会带来新信息，只会消耗真机往返预算。**本用例的变量只有一个——分词**，
所以放在 host 上判"`Encode(文本) == 既有基线常量`"，既证明了"文本入口"与"token 入口"是同一条路，
又把这条判据变成了 CI 每次都会跑的东西。真机总量因此不变（详见 §1 的基线说明）。

**计划与实现的一致性核对（2026-09-26）**：用 `--gtest_list_tests` 列出实际用例名，
与上表逐条比对——**A1-1~A1-10 的用例名 10/10 一致**；**A1-11 是实现期追加的**（计划里没写
`utils/json.hpp` 那处前置改动，属于"先看数据再定改动面"这条教训的又一例，见开发计划 §2.1.1）。
文件面的偏差同样记在开发计划 §2.1.1，不在两处重复。

### 2.3 B / C 两批的用例（2026-09-26 追加，代码就绪、待真机）

| 用例 ID | 用例名 | 层 | 判据 | 出处 | 前置 | 状态 |
|---|---|---|---|---|---|---|
| B-1 | `Gpt2GenerateTest.RealGpt2TextPromptEndToEnd` | G | ① `Encode(kPromptText) == kExpectedPrompt`；② `Generate(prompt, 8 token, top_k=1) == kExpectedTokens`；③ `Decode(全部) == "The quick brown foxes are a great way to get a"` | ①②既有 HF 基线常量（Phase 2 真机验证过）；③本机 HF `decode` 离线实测（`transformers 4.44.0`） | `models/gpt2` + tokenizer 目录 + GPU | ✅ **真机通过**（2026-09-26，作者执行"3. pass"） |
| C-1 | `ResNet18Int8AccuracyTest.DumpsLogitsAndCppReportForCrossCheck` | G | 见 §2.2 的 A2-7 | —— | 同上 + `calib_data` | ✅ 真机执行通过（作者执行"4. pass"，C-1/C-2/C-3 三步全过） |
| C-2 | `int8_crosscheck` | H | 见 §2.2 的 A2-8 | —— | 真机产出的两份报告 | ✅ **真机通过**：两侧实现对同一批 logits 给出同一组 n 与分子（缺报告时仍为 77 跳过） |
| C-3 | `int8_crosscheck_selftest` | H | 见 §2.2 的 A2-9 | —— | 无 | ✅ 通过 |

**B 为什么值得单独一条**：A1-10（host）只证明"分词 == 基线常量"，
本用例才把 `文本 → token → 引擎 → token → 文本` 四段接成一条链；
它与 A1-10 不重复——前者是"桥"，后者是"桥两端的路都走一遍"。

**为什么 A1-10 必须有**：前九条只证明"分词与 HF 一致"，只有 A1-10 证明"框架端到端能吃文本"——
否则 §5.1 的"能力门槛"这一说法没有被验证过。它几乎不新增成本：真机引擎与该基线常量都已存在，
新增的只是"把文本 encode 成这 4 个 id"这一步。

> golden 若由脚本产出，建议按 `resnet18_convert_selftest` 的先例注册成 ctest 项，
> 缺 Python / 缺资产时返回 **77 → Skipped**（缺环境 ≠ 有问题）。

### 2.2 A2 = INT8 判据的离线口径（对应 `future_iterations.md` §1.6 的离线子项）

**被测对象**：`tools/validate/int8_eval.py`（新）与它的验收规格。**本轮不下载任何数据，也不定绝对误差阈值。**

| 用例 ID | 用例名（拟） | 层 | 判据 | 出处 | 前置 | 状态 |
|---|---|---|---|---|---|---|
| A2-1 | `Int8EvalSpecTest.MetaRequiresProvenance` | H | meta 缺 来源 / 版本 / SHA256 / 样本量任一 → **拒绝**并指出缺哪个字段 | §1.6 做法第 1 条 | 合成 meta | ✅ 已实现（在 `--self-test` 的"缺 manifest_sha256"/"num_samples 是字符串"两项里） |
| A2-2 | `Int8EvalSpecTest.RejectsCalibOverlap` | H | 验收集与 `calib_data` 有重叠（按文件名或 SHA256 命中）→ **拒绝** | §1.6 做法第 1 条（标定集污染验证集会让一致率系统性高估） | 合成重叠清单 | ✅ 已实现（`--self-test`）；另用真实 `calib_data`（500 文件）跑过 smoke |
| A2-3 | `Int8EvalSpecTest.ReportRequiresSampleCount` | H | 只报率、不报 n 的报告 → **拒绝**（"只报率不报 n 的结论不可复核"） | §1.6 做法第 2 条 | 合成报告 | ✅ 已实现（`--self-test` 的"只报率不报 n"/"率与分子分母不自洽"两项） |
| A2-4 | `Int8EvalSpecTest.StratificationMatchesExpected` | H | 合成 logits（已知 margin 分布）→ 分层统计的分子 / 分母**逐桶可预测** | 与 C++ 现役实现同口径（`tests/test_resnet18_int8.cpp`） | 合成张量 | ✅ 已实现（`--self-test` 的 3 项分层数学断言） |
| A2-5 | `Int8EvalCrossCheckTest.MatchesCppStratifiedStats` | G | 同一批真实 logits：脚本报出的"余量子集率与样本量"与 C++ 用例**完全一致**（当前应为 12/12 = 100%） | `phase4_int8_plan.md` §4 + `PROGRESS.md` §3.0d | `models/resnet18/` 全产物 | 未开始 |
| A2-6 | `int8_eval_selftest`（ctest 脚本项） | H | 脚本对 **7 类**"故意改坏"的输入逐个拒绝：① 缺 `manifest_sha256`；② 验收集与标定集重叠（文件名 / `sha256` 命中）；③ `num_samples` 是字符串；④ 清单被改过（`sha256` 不符）；⑤ logits 大小与 `shape` 不符；⑥ 只报率不报 n；⑦ 率与分子分母不自洽 | `AGENTS.md` §2.13「护栏必须有用例证明它会拦人」 | 无 | ✅ **已注册进 ctest 并通过**（2026-09-26，`ctest -R int8_eval_selftest` → Passed 0.06 s；无 `SKIP_RETURN_CODE`——它不依赖任何外部资产，没跑起来就是真问题。**序号会漂移，不记序号**） |
| A2-7 | `ResNet18Int8AccuracyTest.DumpsLogitsAndCppReportForCrossCheck` | G | 用两个引擎跑同一批 256 张图 → 落 `fp32.f32.bin` / `int8.f32.bin` / `val_manifest.json` / `meta.json` / `cpp_report.json`，并**断言五个文件都写出来且非空**（静默失败不许伪装成"跑过了"）。正确性判据不在这里（由上一条精度用例负责） | 本计划 A2-5 的设计；口径常量与精度用例共用 `kConfidentMargin` / `kBucketEdges` | `models/resnet18` + `calib_data` + GPU | 🟡 已实现，**沙箱显式跳过**；执行见开发计划 §8.3 |
| A2-8 | `int8_crosscheck`（ctest 脚本项） | H | 比对 `cpp_report.json` 与 `py_report.json`：整体 / 余量子集的 `n` 与分子、逐桶 `n` 与分子、以及两侧阈值（`confident_margin` / `bucket_edges`）必须一致 | A2-5："两条独立实现对同一批数据必须给出同一组数字；不一致说明口径漂移，**不许改阈值**" | 两份报告（真机产出） | 🟡 已注册；缺报告 → **77 跳过**（跳过不是通过）。执行见开发计划 §8.3 |
| A2-9 | `int8_crosscheck_selftest`（ctest 脚本项） | H | 比对器自证 5 项：口径一致 → 0；余量子集分母漂移 / 阈值漂移 / 逐桶分子漂移 → 1；缺报告 + `--skip-if-missing` → 77 | `AGENTS.md` §2.13（护栏必须有用例证明它会拦人） | 无 | ✅ 通过（2026-09-26） |

**A2-5 的意义**：它是两条实现（Python 报告与 C++ 现役统计）的**互证**。
数值不同就说明口径漂移，此时**以实测核对为准**，不许改任一侧的阈值。

---

## 3. 批次 B：P4-INT8-a 的用例（对应 `future_iterations.md` §1.5）

**注意：本批的验收不是"绿 / 红"**，而是 §1.5 写的二选一结论（定位到第 N 层与机制，或证明查空）。
因此下面只有 B1-1 允许带阈值，且该阈值**必须先量再定**。

| 用例 ID | 用例名（拟） | 层 | 判据 | 出处 | 前置 | 状态 |
|---|---|---|---|---|---|---|
| B1-1 | `Int8ProbeTest.PrequantProbeMatchesTorchFoldedBn` | G | **仪器自证**：探针读到的"量化前"张量 vs torch **已折叠 BN** 的模型同点对拍，差异落在"FP32 kernel 正常差异"量级内 | `TROUBLESHOOTING.md` #30.5（上次 2/3 轮预算耗在探针自身的 BN 折叠错） | 探针图 | 未开始 |
| B1-2 | `Int8ProbeTest.SameEngineSameInputBitIdentical` | G | 同一引擎、同一输入重复跑 → 逐位相同（先证明确定性，才谈误差曲线） | §1.5 做法第 2 条 | 引擎缓存 | 未开始 |
| B1-3 | `Int8ProbeTest.LayerwiseErrorGrowthReported` | G | 输出**逐层误差增长曲线**并标出首个越过噪声带的层号；**不预设曲线单调** | §1.5 做法第 2 条 | B1-1 通过 | 未开始 |
| B1-4 | `Int8ProbeTest.PerChannelNotWorseOnIsolatedConv` | G | 回归：单卷积上 per-channel 与模拟的差异仍 ≈`1.9e-6` 量级（结论不因仪器改动而漂移） | §1.5 现状表 | 引擎 | 未开始 |
| B1-5 | `Int8ProbeTest.PerChannelNotWorseOnMinimalBlock` | G | 回归：最小残差 block 上 per-channel 仍**不差于** per-tensor（实测 0.0134 vs 0.0457） | §1.5 现状表 | 引擎 | 未开始 |
| B1-6 | `Int8ProbeTest.CoversDownsampleAndGapFc` | G | 覆盖上次最小复现**没覆盖**的两段：3 个下采样卷积（1×1/s2）与 `GAP + fc` | §1.5 做法第 3 条（#30.3 第 2 条） | 引擎 | 未开始 |

**禁止**：把 B1 的任一判据写成"per-channel 必须优于 per-tensor"——那是预设结论，
而本条目的问题恰恰是"为什么整网上更差"（`AGENTS.md` §7 第 1 / 3 种动作之外都不允许）。

---

## 4. 批次 C：骨架（触发后再细化）

**为什么现在只写骨架**：这些条目的前提（数据 / 模型 / 测量方法）都还没到位，
此刻写出的判据与阈值必然是拍脑袋的，等触发时会变成"来路不明的阈值"。
触发后先产独立 phase 计划，再回填本表。

| 条目 | 触发后**必须先建**的测试 | 层 | 需要什么资产 |
|---|---|---|---|
| §1.2 LLM INT8 / INT4 | `PagedAttentionPlugin` 的 INT8 KV cache dtype 契约 + 数值对拍（先扩插件） | H + G | 扩展后的插件；INT8 参考实现 |
| §2.1 显存池 | 分配开销基准（按 G6：≥3 次构建 / ≥20 次推理，报中位数与极差）+ 接口不变回归 | H + P | 无新资产 |
| §2.3 Continuous Batching | 多序列调度下的 block 分配 / `context_lens` 正确性 + 与 `batch = 1` 的数值一致性 | G | 扩 batch 后的 runner |
| §2.4 CV 动态分辨率 | profile 接受 / 拒绝的 host 契约 + 真机非方形输入数值 | H + G | 变分辨率输入 |
| §3.1 / §3.2 / §3.3 / §3.4 | 复用既有分层（host 建图契约 + 真机数值 + 端到端） | H + G | 目标模型权重与基线 |
| §4.1 / §4.2 / §4.3 | 每个新 Plugin 的算子单测 + L2 集成（沿用 Phase 1 的分层） | G | 目标模型 |
| §6.x | 工具自检（护栏自证）+ 测量协议 | H + P | 无 |
| §9.2 / §9.3 | decode 性能 profile（先确认 sampler 占比）+ 采样截断语义数据固化 | G + P | 参考数据 `.bin` |
| §10.1 | 两路径 I/O 契约统一后的 host 契约用例 + 逐 token 一致 | H + G | 无需新资产 |

---

## 5. 判据与出处（每条都要能回答"凭什么"）

| 判据 | 当前值 | 出处 | 备注 |
|---|---|---|---|
| Tokenizer 与 HF 一致 | **逐 token 全等**（不是率） | 2026-09-26 实测快照 + `transformers 4.44.0` | 全等是可能的，因为它俩是同一套确定性算法；给不出全等就说明实现有偏差 |
| INT8 余量子集一致率 | `>= 90%`（实测 12/12 = 100%） | `phase4_int8_plan.md` §4 | **不许套到 FP16 / FP32 上**（跨精度复用） |
| INT8 整体一致率 | `>= 30%`（实测 37.9% / 38.3%） | 同上，"没崩坏"下界 | 主要在测测试集噪声，不是质量指标 |
| INT8 绝对误差界 | **未定**（实测 `max_abs ≈ 21.6`，故意不作判据） | 同上 + §1.6 | 有真值标签的验收集到位后按 p95 / p99 分布定 |
| 探针仪器自证 | **待实测后定**（先量"FP32 kernel 正常差异"的量级） | §1.5 做法第 1 条 | 未量之前**不写数字**——写了就是来路不明的阈值 |
| 基线（沙箱 / 真机） | 沙箱 204 条 / 0 失败；真机上次 182 条 / 1 红（应为 204，待复验） | `PROGRESS.md` §3.5 / §3.0d | 每批收口同步更新 |

---

## 6. 执行方式

1. **新增源文件后先重新 configure**：`mini_trt_llm/CMakeLists.txt` 与 `tests/CMakeLists.txt` 用
   `file(GLOB ...)`，漏跑的症状是链接期 `undefined reference to vtable`（`PROGRESS.md` §2.13）。
2. **产物或建图变了先删引擎缓存**：`/tmp/mini_trt_llm_*.engine` 只按路径名区分。
3. **跳过语义**：GPU 用例用 `MINI_TRT_SKIP_IF_NO_CUDA`；脚本类 ctest 项返回 **77**；
   `MINI_TRT_REQUIRE_GPU=1` 时跳过即失败。缺 `models/gpt2` / `models/resnet18` 等资产一律**跳过**。
4. **真机全量与单跑都要**：改过接口 / 契约的批次，单跑通过不算数（越界类缺陷只在完整套件里暴露，
   `PROGRESS.md` §2.13）。
5. **参考实现唯一 + meta 自证**：A1-9 是这条纪律在本计划的落点；同一算子的两份参考必须合并。
6. **带 batch 维的算子必须覆盖 `batch > 1`**（本计划目前只有 CV / Runner 侧涉及）。

---

## 7. 覆盖缺口（本计划明确不覆盖什么）

- **不覆盖 C 组细则**：见 §4 的说明（触发后由各自的 phase 测试计划细化）。
- **不覆盖服务化压测**（吞吐 / QPS / 并发）：属 §7.1 / §7.2，需要服务实现与服务级口径。
- **不覆盖多语言 tokenizer 的通用正确性**：A1 只对 **GPT-2 英文 + 少量 UTF-8** 与 HF 对拍；
  中文分词是否"合理"不在判据内（判据是"与参考一致"，不是"分得好"）。
- **不覆盖 EOS 早停**（`G2-4`，语义已正确、只是多算）。
- **不覆盖 INT8 绝对数值保证**：§1.6 到位前，本计划只能说"与 FP32 一致率"，不构成对任意输入的数值承诺。

---

## 8. 结果回填

> 回填要求：只写"状态 + 实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。

| 批次 | 状态 | 实测值 | 出处 | 回填日期 |
|---|---|---|---|---|
| A1 BPE Tokenizer | ✅ 完成（A1-1~A1-10 全部通过；A1-10 层级由 G 改 H，理由见 §2.1） | `Encode` 与 HF 逐 token 全等（21 样本）、`Decode` 一致、4 条 `Load` 负例拒绝、参考自证通过 | `./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='Bpe*'`（10 条全过）；`ctest -R tokenizer_golden_check`（Passed 3.21 s）；沙箱全量 `ctest` **204 条 / 0 失败** | 2026-09-26 |
| A2 INT8 判据离线规格 | ✅ 完成（A2-1~A2-4 / A2-6 / **A2-9**；**A2-5 已由 C 批在真机完成**，见 §2.3） | `--self-test` 全绿（3 项分层数学 + 7 道护栏 + 1 项 legacy 标注）；真实 `calib_data` 500 文件 smoke 报告正确；C 批：C++ 与 Python 两侧统计给出同一组 n 与分子 | `python3 mini_trt_llm/tools/validate/int8_eval.py --self-test`；`ctest -R int8_eval_selftest`；`ctest -R int8_crosscheck`；沙箱 `ctest` **215/0** | 2026-09-26 |
| B1 P4-INT8-a | 未触发 | —— | —— | —— |
| C 组 | 未触发 | —— | —— | —— |

---

*本计划不含对话过程；判据的唯一来源是 `docs/future_iterations.md` 与各阶段计划 / 测试计划。*

---

## 9. §9.2 Sampler 高性能 kernel（测试计划）

> 配套开发计划见 `future_iterations_development_plan.md` §10。条目事实来源：`docs/future_iterations.md` §9.2。

### 9.1 分层与环境

层名沿用本计划的 **H / G / P** 三档（刻意不复用 phase 计划的 `L0~L3` / `R0.x`，避免同一编号指两回事）：

| 层 | 含义 | 沙箱可跑 | 归属 |
|---|---|---|---|
| **H** | host 契约（workspace 尺寸、参数校验、覆盖率/cutoff 的纯逻辑） | 能 | CI / ctest |
| **G** | 真机数值（kernel 正确性、分布、FP16、大 vocab） | 不能（无设备 → 显式跳过） | 作者真机 |
| **P** | 真机性能（按 G6 口径：≥3 次构建 / ≥20 次推理，报中位数与极差） | 不能 | 作者真机 |

执行口径：沙箱 `ctest --test-dir build`；真机 `MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure`
（不带该变量时 GPU 用例会静默跳过，等于白跑）。**当前真机基线：215 条 / 1 红**（唯一红 = FP16 NaN 复现器，按设计）。

---

### 9.2 用例清单

#### 9.2.1 既有回归（**主判据：语义不变、判据不改、全部通过**）

出处：`mini_trt_llm/tests/test_sampler.cpp`（11 条）；它们的判据与依据写在各自注释里，本次**不许改**。

| 编号 | 用例 | 层 | 这条在锁什么 |
|---|---|---|---|
| S-1 | `SamplerTest.WorkspaceSizingIsPositiveAndRejectsInvalidDims` | H/G | workspace 尺寸契约 + 非法维度返回 0 |
| S-2 | `SamplerTest.RejectsNullPointersAndWorkspace` | G | 空指针/空 workspace → 非 `cudaSuccess` |
| S-3 | `SamplerKernelTest.GreedyMatchesArgmax` / `GreedyPrefersLowestIndexOnTie` | G | greedy 语义（并列取小下标）——本次不改 greedy，作对照 |
| S-4 | `SamplerKernelTest.TopKWithKEqualsOneBehavesLikeGreedy` | G | **并列语义**的间接锁（k=1 必等于 argmax） |
| S-5 | `SamplerKernelTest.TopKSamplingIsDeterministicForFixedSeed` / `TopPIsDeterministicForFixedSeed` | G | 同 seed **同实现**可复现（随机数消费方式不变） |
| S-6 | `SamplerKernelTest.TopKResultAlwaysWithinTopKSet` | G | 采样必落在 top-K 集合内 |
| S-7 | `SamplerKernelTest.TopPWithTinyPicksArgmax` | G | p 极小 → 退化为 argmax |
| S-8 | `SamplerKernelTest.TopKDistributionMatchesSoftmaxProbabilities` | G | **主判据**：词频收敛到解析 softmax 概率（3σ） |
| S-9 | `SamplerKernelTest.TopPWithFullProbabilityDoesNotOverTruncate` | G | p=1 不得过截断（最小概率 token 也必须能采到） |
| S-10 | `test_e2e_mini_decoder.cpp` 的 Top-K（k=1）段 | G | 端到端里采样的接入形态 |
| S-11 | `Fp16PathTest.GreedySamplerMatchesArgmaxOnFp16Logits` | G | FP16 分支（greedy 已覆盖） |

#### 9.2.2 新增用例（本次交付）

| 编号 | 用例名（拟） | 层 | 判据 | 出处 | 状态 |
|---|---|---|---|---|---|
| **S-12** | `SamplerKernelTest.TopKSamplingDistributionMatchesFp32UnderFp16` | G | FP16 logits 与 FP32 logits 的**同一组**概率的采样词频都收敛到解析值（各 3σ）；两个 dtype 的词频互相也在 3σ 内 | 沿用 S-8 的判据口径；补 `future_iterations.md` §11 的 **P1.5-a** | 未开始 |
| **S-13** | `SamplerKernelTest.TopPSamplingDistributionMatchesFp32UnderFp16` | G | 同 S-12，改成 Top-P（p=0.9） | 同上 | 未开始 |
| **S-14** | `SamplerKernelTest.TopKOnLargeVocabStaysWithinTopKSet` | G | vocab = 128000（合成）与 50257（真实形状）下，采样 token ∈ 解析 top-K 集合；k 取 {1, 8, 64} | 集合成员关系判定（**无阈值**，因此无出处问题） | 未开始 |
| **S-15** | `SamplerKernelTest.TopPNucleusCoversLargeVocabCutoff` | G | p = 0.9 / 0.99 下，采样 token ∈ 解析 nucleus；并**打印**候选覆盖率与回退行数（观测指标，不作判据） | 同上 | 未开始 |
| **S-16** | `SamplerKernelTest.TopPWithHugeNucleusFallsBackCorrectly` | G | 构造"nucleus 远超 M"的分布（如均匀 logits + p=1.0）→ 走回退；仍满足 S-9 的判据（最小概率 token 可采到） | S-9 的口径；回退是**正确性优先**的设计（开发计划 D2 第 5 条） | 未开始 |
| **S-17** | `SamplerTest.WorkspaceSizingAfterRewrite` | G | 新实现的 `WorkspaceBytes` > 0、且 `Launch*` 在该大小下成功（含 batch=1/batch=8） | 契约；调用方只查询大小（见开发计划 §0.2） | 未开始 |
| **S-18** | `SamplerPerf.ThroughputByShape`（**P 层**，已实现） | P | vocab ∈ {50257, 128000} × batch ∈ {1, 8}：warmup 3 + 采样 **21** 次，报**中位数 / 极差 / min-max**，并给 greedy 作"一趟扫描"参照；与改实现后的同一形状对比 | 协议 = `phase3_test_plan.md` §5 的 **G6**（≥3 次构建 / ≥20 次推理、报中位数与极差） | 🟡 **仪器已完成**（`tests/test_sampler.cpp`，2026-09-26）；**基线数据待真机** |

---

### 9.3 判据与出处（每条都要能回答"凭什么"）

| 判据 | 值 / 形式 | 出处 |
|---|---|---|
| 分布一致性 | 与解析 softmax 概率比较，容差 `3σ + 1e-3` | `tests/test_sampler.cpp` 的 `TopKDistributionMatchesSoftmaxProbabilities` 注释（本次沿用，不改） |
| FP16 分布一致性 | 同 3σ 口径（**不另立更松的尺子**） | 同上；`AGENTS.md` §7「阈值不跨精度复用」在这里的意思相反——**同一算子同一 dtype 语义，就该用同一把尺子** |
| 集合成员关系 | 布尔判定（token ∈ top-K / ∈ nucleus） | 无阈值 → 无出处问题；构造上保证集合可解析 |
| 性能加速比 | **待 P9_2-0 实测后填写**（连同"哪次实测"） | `future_iterations.md` §9.2 的触发时机 + `phase3_test_plan.md` §5 的 G6 |
| 回退行数 / 覆盖率 | **观测指标，不作判据** | 避免用"调 M"把回退率刷成 0 从而掩盖覆盖率问题 |
| 全量基线 | 真机 215 条 / 1 红 | `PROGRESS.md` §3.5 / §3.0f |

---

### 9.4 语义等价的范围（写清楚，免得把正常差异当回归）

- **允许不同**：同一 `(seed, offset)` 下逐 token 输出与**旧实现**可以不同——候选归并顺序可能改变
  逆变换的采样点。**这不是回归**。
- **必须相同**：① 同 seed 同实现两次运行逐 token 一致（S-5）；② 分布（S-8/S-12/S-13）；
  ③ `k = 1` 等价于 argmax（S-4）；④ p 极小 → argmax（S-7）、p = 1 不过截断（S-9）；
  ⑤ 并列 → 小下标优先（S-3/S-4 的间接锁 + 实现内显式保证，见开发计划 D3）。

---

### 9.5 执行方式

1. **改了 `.cu` 之后必须重跑 configure**（`file(GLOB_RECURSE ...)` 只在 configure 时求值）；
2. 新增测试文件同样依赖 GLOB（重跑 configure 即可，无需手改 CMakeLists；若新增 ctest 脚本项才需追加）；
3. 性能测量按 G6：同一 session、≥20 次、报中位数与极差；**先跑基线 P9_2-0**再跑新实现；
4. 真机跑全量时带 `MINI_TRT_REQUIRE_GPU=1`；新增用例若在沙箱"跳过"，要在真机确认它**真的跑了**。

---

### 9.6 覆盖缺口（本计划明确不覆盖什么）

- **不覆盖** continuous batching 下的多请求调度（依赖 §2.3 / G2-3）；
- **不覆盖**温度、重复惩罚等新采样语义（本轮只做实现替换）；
- **不覆盖** 128K 真实模型端到端（本地没有该规模的模型；S-14/S-15 用合成分布覆盖 kernel 形状）；
- **不覆盖** 采样器在 CUDA Graph 捕获下的行为（当前 runner 未用 graph capture）。

---

### 9.7 结果回填（待回填）

> 回填要求：只写"状态 + 实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。

| 用例 | 状态 | 实测值 / 出处 |
|---|---|---|
| S-1 ~ S-11（既有回归） | 未开始（等实现改动后跑） | —— |
| S-12 ~ S-17（新增） | 未开始 | —— |
| S-18（性能基线） | 🟡 仪器已完成，待真机取数 | 沙箱显式跳过；真机命令：`MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='*SamplerPerf*'` |

---

*本计划只含执行安排与判据；条目目标与触发条件的唯一来源仍是 `docs/future_iterations.md` §9.2。*
