# future_iterations 测试计划

> **状态**：2026-09-26 产出；**批次 A/B/C 与 §9 的 P9_2-0~5 已执行到"沙箱 / 真机"两级，
> 各自的状态写在对应表格的状态列里**（原先那句"未执行、状态列一律未开始"已被实际执行推翻）。
> 未跑过的一律写"未开始"，不写"通过"。
>
> **定位**：与 `docs/future_iterations_development_plan.md` 配套。**条目内部的目标 / 做法 / 验收**
> 以 `docs/future_iterations.md` 为唯一来源（§1.5 / §1.6 / §0.1），本文件只负责把它落成
> "用例 → 判据 → 出处 → 环境 → 状态"。
>
> **判定口径**（沿用本项目惯例）：**未跑过的不写"通过"**；"跳过"必须显式且打印探测结果；
> 阈值必须能回答"凭什么这么定"（`AGENTS.md` §7）。
>
> **开发状态落点（2026-10-01 起）**：各 feature 的**状态与设计**在 `docs/dev/<feature>/`
> （入口 = `STATE.md`；阶段链与 Gate 见 `AGENTS.md` §5 与技能 `trt-inference-engineering`）。
> 本文件保留"怎么验"（用例 → 判据 → 出处 → 环境 → 状态）与结果回填。

---

## 1. 分层与环境

**层名刻意新起**（不与 phase 计划的 `L0~L3`、`S/E1~E4`、`R0.x~R3.x` 混用）——本项目已经因为
层名冲突吃过一次"同一编号指两件事"的亏（`docs/dev/REQ-003-test-infra/phase1_5_test_plan.md` 开头的说明）。

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

**当前基线**：**沙箱 268 条 / 0 失败**；真机整轮 267 条 / 1 红 / 0 跳过 / 301.72 s（2026-09-28，
该轮尚无 P5-0-2 的 `check_skips_selftest` → 总数随之为 268、待复跑）。
唯一出处见 `PROGRESS.md` 的「当前基线」（真机跳过集合由 `LastTest.log` 逐条解析确认，见那里的说明）。
（红 = 按设计的 `RealGpt2Fp16GreedyMatchesReferenceTokens`，见 `docs/PROGRESS.md` §5.11；
`int8_crosscheck` 报告齐备 → Passed，只在缺报告时按设计跳过 77）。
两边**总数相同**，差别只在 GPU 用例是跑还是跳过。
**新增用例后基线总数会变，改动时必须同步回填本行、§10.1 / §11.1 的"当前基线"与 `PROGRESS.md`。**

**历史快照（勿当现状）**：2026-09-26 是沙箱 215 / 真机 215 条（1 红）；更早的 204 条 / 2 红
是修正判据之前的快照，其中 `Gpt2OnnxTest.MatchesAcrossProfileShapes` 那条已按方案 B 结案
（`docs/TROUBLESHOOTING.md` #34、`PROGRESS.md` §5.13）。当时还有一条"红 2"——
**那条已结案，现在真机只剩 1 红**。

---

## 2. 批次 A：现在就能开工的用例

### 2.1 [OI-BPE-TOKENIZER-TESTS] A1 = BPE Tokenizer（对应 `future_iterations.md` §5.1）

**参考实现与出处**：`transformers 4.44.0` 的 `GPT2TokenizerFast`，**离线**从本地 HF 快照加载
（`local_files_only=True`，快照 `models--gpt2/snapshots/607a30d...`）。2026-09-26 实测：

<details><summary>展开：2.1 [OI-BPE-TOKENIZER-TESTS] A1 = BPE Tokenize 全文</summary>

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

</details>

### 2.3 B / C 两批的用例（2026-09-26 追加，**同日真机执行完毕**）


<details><summary>展开：2.3 B / C 两批的用例（2026-09-26 追加，**同日真机执行完毕**） 全文</summary>

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
否则 `future_iterations.md` §5.1 的"能力门槛"这一说法没有被验证过。它几乎不新增成本：真机引擎与该基线常量都已存在，
新增的只是"把文本 encode 成这 4 个 id"这一步。

> golden 若由脚本产出，建议按 `resnet18_convert_selftest` 的先例注册成 ctest 项，
> 缺 Python / 缺资产时返回 **77 → Skipped**（缺环境 ≠ 有问题）。

</details>

### 2.2 A2 = INT8 判据的离线口径（对应 `future_iterations.md` §1.6 的离线子项）

**被测对象**：`tools/validate/int8_eval.py`（新）与它的验收规格。**本轮不下载任何数据，也不定绝对误差阈值。**


<details><summary>展开：2.2 A2 = INT8 判据的离线口径（对应 `future_iterations.md 全文</summary>

| 用例 ID | 用例名（拟） | 层 | 判据 | 出处 | 前置 | 状态 |
|---|---|---|---|---|---|---|
| A2-1 | `Int8EvalSpecTest.MetaRequiresProvenance` | H | meta 缺 来源 / 版本 / SHA256 / 样本量任一 → **拒绝**并指出缺哪个字段 | `future_iterations.md` §1.6 做法第 1 条 | 合成 meta | ✅ 已实现（在 `--self-test` 的"缺 manifest_sha256"/"num_samples 是字符串"两项里） |
| A2-2 | `Int8EvalSpecTest.RejectsCalibOverlap` | H | 验收集与 `calib_data` 有重叠（按文件名或 SHA256 命中）→ **拒绝** | `future_iterations.md` §1.6 做法第 1 条（标定集污染验证集会让一致率系统性高估） | 合成重叠清单 | ✅ 已实现（`--self-test`）；另用真实 `calib_data`（500 文件）跑过 smoke |
| A2-3 | `Int8EvalSpecTest.ReportRequiresSampleCount` | H | 只报率、不报 n 的报告 → **拒绝**（"只报率不报 n 的结论不可复核"） | `future_iterations.md` §1.6 做法第 2 条 | 合成报告 | ✅ 已实现（`--self-test` 的"只报率不报 n"/"率与分子分母不自洽"两项） |
| A2-4 | `Int8EvalSpecTest.StratificationMatchesExpected` | H | 合成 logits（已知 margin 分布）→ 分层统计的分子 / 分母**逐桶可预测** | 与 C++ 现役实现同口径（`tests/test_resnet18_int8.cpp`） | 合成张量 | ✅ 已实现（`--self-test` 的 3 项分层数学断言） |
| A2-5 | `Int8EvalCrossCheckTest.MatchesCppStratifiedStats`（**计划名**） | G | 同一批真实 logits：脚本报出的"余量子集率与样本量"与 C++ 用例**完全一致**（当前应为 12/12 = 100%） | `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §4 + `PROGRESS.md` §3.0d | `models/resnet18/` 全产物 | ✅ **已由 A2-7 + A2-8 落地，并真机通过**（2026-09-26，C-1/C-2/C-3 三步全过）——**更正**：实现时没有建这条同名 gtest，而是拆成"落 artefact"（A2-7）与"两份报告比对"（A2-8）两步；留原名只为追溯 |
| A2-6 | `int8_eval_selftest`（ctest 脚本项） | H | 脚本对 **7 类**"故意改坏"的输入逐个拒绝：① 缺 `manifest_sha256`；② 验收集与标定集重叠（文件名 / `sha256` 命中）；③ `num_samples` 是字符串；④ 清单被改过（`sha256` 不符）；⑤ logits 大小与 `shape` 不符；⑥ 只报率不报 n；⑦ 率与分子分母不自洽 | `PROGRESS.md` §2.13「护栏必须有用例证明它会拦人」 | 无 | ✅ **已注册进 ctest 并通过**（2026-09-26，`ctest -R int8_eval_selftest` → Passed 0.06 s；无 `SKIP_RETURN_CODE`——它不依赖任何外部资产，没跑起来就是真问题。**序号会漂移，不记序号**） |
| A2-7 | `ResNet18Int8AccuracyTest.DumpsLogitsAndCppReportForCrossCheck` | G | 用两个引擎跑同一批 256 张图 → 落 `fp32.f32.bin` / `int8.f32.bin` / `val_manifest.json` / `meta.json` / `cpp_report.json`，并**断言五个文件都写出来且非空**（静默失败不许伪装成"跑过了"）。正确性判据不在这里（由上一条精度用例负责） | 本计划 A2-5 的设计；口径常量与精度用例共用 `kConfidentMargin` / `kBucketEdges` | `models/resnet18` + `calib_data` + GPU | 🟡 已实现，**沙箱显式跳过**；执行见开发计划 §8.3 |
| A2-8 | `int8_crosscheck`（ctest 脚本项） | H | 比对 `cpp_report.json` 与 `py_report.json`：整体 / 余量子集的 `n` 与分子、逐桶 `n` 与分子、以及两侧阈值（`confident_margin` / `bucket_edges`）必须一致 | A2-5："两条独立实现对同一批数据必须给出同一组数字；不一致说明口径漂移，**不许改阈值**" | 两份报告（真机产出） | 🟡 已注册；缺报告 → **77 跳过**（跳过不是通过）。执行见开发计划 §8.3 |
| A2-9 | `int8_crosscheck_selftest`（ctest 脚本项） | H | 比对器自证 5 项：口径一致 → 0；余量子集分母漂移 / 阈值漂移 / 逐桶分子漂移 → 1；缺报告 + `--skip-if-missing` → 77 | `PROGRESS.md` §2.13（护栏必须有用例证明它会拦人） | 无 | ✅ 通过（2026-09-26） |

**A2-5 的意义**：它是两条实现（Python 报告与 C++ 现役统计）的**互证**。
数值不同就说明口径漂移，此时**以实测核对为准**，不许改任一侧的阈值。

---

</details>

## 3. 批次 B：P4-INT8-a 的用例（对应 `future_iterations.md` §1.5）

> **2026-09-27 细化**：执行设计（D1~D6）、测量口径、真机清单在
> `future_iterations_development_plan.md` **§13**。原表把 B1-1 的判据写成"差异落在 FP32 kernel 正常
> 差异量级内"——**没说这个量级从哪来**（AGENTS.md §7 不许来路不明的阈值）。现改成：
> **噪声地板由 PT 臂当场量出**（`future_iterations_development_plan.md` §13.3 D4），下面是落地后的用例清单。

**注意：本批的验收不是"绿 / 红"**，而是 `future_iterations.md` §1.5 写的二选一结论（定位到第 N 层与机制，或证明查空）。
因此只有"位相同"与"首层落在噪声地带内"两条**结构性**判据，其余一律"打印 + 落盘报告"。

| 用例 ID | 用例名 | 层 | 判据 | 出处 | 前置 | 状态 |
|---|---|---|---|---|---|---|
| B1-1 | `Int8ProbeTest.LayerwiseErrorGrowthVsOnnxReference`（自证部分） | G | **仪器自证（比较型）**：两臂 conv1 的 `d_pre = \|引擎探针 − 参考量化前\| ≤ d_post = \|引擎探针 − 参考量化后\|`，且首层落在噪声地带内。**落地方案在实施时改了**：原写"数格点占比"，实做改为**同时落量化前/量化后两份参考、比 d_pre 与 d_post**——不用先估 scale，少一个可能出错的环节（见 `future_iterations_development_plan.md` §13.3 D3 的落地说明） | `future_iterations_development_plan.md` §13.3 D3 | 探针图 | ✅ **真机通过**（2026-09-27）：`conv1` PT `d_pre=8.34e-07`/`d_post=0.0398`、PC `1.43e-06`/`0.0398`（差 4.7 个数量级） |
| B1-2 | `Int8ProbeTest.SameEngineSameInputBitIdentical` | G | 同一引擎、同一输入重复跑 → 逐位相同（先证明确定性，才谈误差曲线） | `future_iterations.md` §1.5 做法第 2 条 | 引擎缓存 | ✅ **真机通过**（2026-09-27）：22 个张量逐位相同（首跑红在用例绑定，见 `TROUBLESHOOTING.md` #47.1，已修） |
| B1-3 | `Int8ProbeTest.LayerwiseErrorGrowthVsOnnxReference`（曲线部分） | G | 与 **ONNX 官方参考实现**逐层对拍：PC / PT 各出一条 `max_abs` 曲线并落盘；标出 PC 首个越过 `100 × noise_floor`（`noise_floor` 由 PT 臂当场量出）的层号；**不预设曲线单调** | `future_iterations.md` §1.5 做法第 1~3 条；`future_iterations_development_plan.md` §13.3 D1/D2/D4 | B1-1 / B1-2 通过 | ✅ **真机通过**（2026-09-27）：引擎 vs **自己的图**逐层最大 `max_abs` PT 0.2714（`layer4.1.conv2`，相对 2.3%）、PC 0.168、`conv1` 8.3e-07；两臂都未越界 → **引擎忠实**。另加**逐层 ONELINE 落盘**（`<ref-dir>/layers_{pt,pc}.txt`），因为探针图改了 tactic（#47.2）。**判读措辞已修正**：这条判据问的是"引擎有没有跑偏自己的图"，不是"两臂谁更准"（#47.4） |
| B1-4 | `Int8ProbeTest.PerChannelDegradationReproducesUnderProbe` | G | **硬门**：探针图下 256 张仍复现退化（余量子集一致率 PC 明显 < PT；正式产物口径 PC 54.5% / PT 100%）。不复现 → 本轮作废 | `future_iterations_development_plan.md` §13.3 D6（挂图输出可能改变融合） | 引擎 | ✅ **真机通过（硬门）**（2026-09-27）：整体 PT 99/256、PC 26/256；**余量子集（n=12）PT 12/12 = 100%、PC 6/12 = 50%** → 与正式产物口径一致，探针图是现象的有效模型（尽管它把 `i8i8` 从 4 变成 0） |

> **B1-4 的前提是"PC 臂用那份**错源**的 per-channel 图"**（`models/resnet18/resnet18_qdq_probe_per_channel.onnx`）。
> 那份产物的身份已钉为 **#46 的复现样本、不是候选基线**（开发计划 §13.11）。**若哪天把它重生成成
> 改源版，B1-4 会立刻变红——那不是故障，是它的前提消失了**：届时必须把"重生成 + 退役/改写 B1-4"
> **打包**做（改成"两臂都对 FP32 全一致"之类），并按 `AGENTS.md` §7 把"为什么可以改这条期望"写进文档。
> **不要**为了让 B1-4 变绿而去动阈值或删断言。

**覆盖说明（原 B1-6 并进 B1-3）**：探针清单 = 20 个 Conv 的**量化前**输出（含 3 个
`1×1/s2` 下采样卷积）+ `GlobalAveragePool` 输出 + 契约输出 `output`，覆盖 `future_iterations.md` §1.5 做法第 3 条点名的
"下采样卷积"与"GAP+fc 段"。残差 `Add` / `Relu` 的输出可由已探张量逐元素推出，不另挂输出
（理由见 `future_iterations_development_plan.md` §13.3 D5）。**原 B1-4 / B1-5**（单卷积 / 最小残留 block 上 per-channel 不更差）是
历史结论，本轮**不重做**——它们的可复现性是 Python 侧，见 B1-H2。

| 用例 ID | 用例名 | 层 | 判据 | 出处 | 前置 | 状态 |
|---|---|---|---|---|---|---|
| B1-H1 | `qdq_reference_selftest`（ctest） | H | 参考工具自证：最小 Q/DQ 图**逐位**比手算值（0 差）、per-channel 与 per-tensor 在构造样本上可分辨、index/meta 格式正确、坏输入被拒 | `future_iterations_development_plan.md` §13.3 D1；`PROGRESS.md` §2.13（参考必须唯一且独立） | 无（沙箱可跑） | ✅ **通过**（`ctest -R qdq_reference_selftest`；自检当场抓出"把 producer 当 consumer"的配对 bug） |
| B1-H2 | `add_probe_outputs_selftest`（ctest） | H | 探针图变换自证：节点 / initializer / 输入 / opset **逐字节不变**，只有 `graph.output` 变多；探针数与形状自洽 | `future_iterations_development_plan.md` §13.3 D5；"探针图 = 产物图 + 探针"是整条结论的地基 | 无（沙箱可跑） | ✅ **通过**（`ctest -R add_probe_outputs_selftest`） |

**禁止**：把 B1 的任一判据写成"per-channel 必须优于 per-tensor"——那是预设结论，
而本条目的问题恰恰是"为什么整网上更差"（`AGENTS.md` §7 第 1 / 3 种动作之外都不允许）。
**也不许**用 B1-3 的曲线去解释原来的退化，除非 B1-4 先通过（`future_iterations_development_plan.md` §13.6）。

---

## 4. 批次 C：骨架（触发后再细化）

**为什么现在只写骨架**：这些条目的前提（数据 / 模型 / 测量方法）都还没到位，
此刻写出的判据与阈值必然是拍脑袋的，等触发时会变成"来路不明的阈值"。
触发后先产独立 phase 计划，再回填本表。

| 条目 | 触发后**必须先建**的测试 | 层 | 需要什么资产 |
|---|---|---|---|
| `future_iterations.md` §1.2 LLM INT8 / INT4 | `PagedAttentionPlugin` 的 INT8 KV cache dtype 契约 + 数值对拍（先扩插件） | H + G | 扩展后的插件；INT8 参考实现 |
| §2.1 显存池 | 分配开销基准（按 G6：≥3 次构建 / ≥20 次推理，报中位数与极差）+ 接口不变回归 | H + P | 无新资产 |
| §2.3 Continuous Batching | 多序列调度下的 block 分配 / `context_lens` 正确性 + 与 `batch = 1` 的数值一致性 | G | 扩 batch 后的 runner |
| `future_iterations.md` §2.4 CV 动态分辨率 | profile 接受 / 拒绝的 host 契约 + 真机非方形输入数值 | H + G | 变分辨率输入 |
| `future_iterations.md` §3.1 / §3.2 / §3.3 / §3.4 | 复用既有分层（host 建图契约 + 真机数值 + 端到端） | H + G | 目标模型权重与基线 |
| `future_iterations.md` §4.1 / §4.2 / §4.3 | 每个新 Plugin 的算子单测 + L2 集成（沿用 Phase 1 的分层） | G | 目标模型 |
| §6.x | 工具自检（护栏自证）+ 测量协议 | H + P | 无 |
| §9.2 / §9.3 | decode 性能 profile（先确认 sampler 占比）+ 采样截断语义数据固化 | G + P | 参考数据 `.bin` |
| §10.1 | 两路径 I/O 契约统一后的 host 契约用例 + 逐 token 一致 | H + G | 无需新资产 |

---

## 5. 判据与出处（每条都要能回答"凭什么"）

| 判据 | 当前值 | 出处 | 备注 |
|---|---|---|---|
| Tokenizer 与 HF 一致 | **逐 token 全等**（不是率） | 2026-09-26 实测快照 + `transformers 4.44.0` | 全等是可能的，因为它俩是同一套确定性算法；给不出全等就说明实现有偏差 |
| INT8 余量子集一致率 | `>= 90%`（实测 12/12 = 100%） | `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §4 | **不许套到 FP16 / FP32 上**（跨精度复用） |
| INT8 整体一致率 | `>= 30%`（实测 37.9% / 38.3%） | 同上，"没崩坏"下界 | 主要在测测试集噪声，不是质量指标 |
| INT8 绝对误差界 | **未定**（实测 `max_abs ≈ 21.6`，故意不作判据） | 同上 + `future_iterations.md` §1.6 | 有真值标签的验收集到位后按 p95 / p99 分布定 |
| 探针仪器自证（`future_iterations.md` §1.5） | **`d_pre ≤ d_post` + 首层落在噪声地带内**（真机：`conv1` 8.34e-07 vs 0.0398；噪声地板 0.2714） | `future_iterations_development_plan.md` §13.3 D3/D4；`TROUBLESHOOTING.md` #46.4 | 落地时由"数格点"改成"比 d_pre / d_post"——不用先估 scale，少一个可能出错的环节 |
| 基线（沙箱 / 真机） | **沙箱 268 条 / 0 失败**；真机整轮全量 **267 条 / 1 红 / 0 跳过 / 301.72 s**（2026-09-28 复跑；该轮在 `check_skips_selftest` 注册之前 → 下次真机应为 268） | `PROGRESS.md` 当前基线 / §3.0j | 每批收口同步更新（演进：182 → 204 → 215 → 242 → 259 → 264 → 265 → 267 → **268**） |

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
- **不覆盖服务化压测**（吞吐 / QPS / 并发）：属 `future_iterations.md` §7.1 / §7.2，需要服务实现与服务级口径。
- **不覆盖多语言 tokenizer 的通用正确性**：A1 只对 **GPT-2 英文 + 少量 UTF-8** 与 HF 对拍；
  中文分词是否"合理"不在判据内（判据是"与参考一致"，不是"分得好"）。
- **不覆盖 EOS 早停**（`G2-4`，语义已正确、只是多算）。
- **不覆盖 INT8 绝对数值保证**：`future_iterations.md` §1.6 到位前，本计划只能说"与 FP32 一致率"，不构成对任意输入的数值承诺。

---

## 8. 结果回填

> 回填要求：只写"状态 + 实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。

| 批次 | 状态 | 实测值 | 出处 | 回填日期 |
|---|---|---|---|---|
| A1 BPE Tokenizer | ✅ 完成（A1-1~A1-10 全部通过；A1-10 层级由 G 改 H，理由见 §2.1） | `Encode` 与 HF 逐 token 全等（21 样本）、`Decode` 一致、4 条 `Load` 负例拒绝、参考自证通过 | `./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='Bpe*'`（10 条全过）；`ctest -R tokenizer_golden_check`（Passed 3.21 s）；沙箱全量 `ctest` **204 条 / 0 失败** | 2026-09-26 |
| A2 INT8 判据离线规格 | ✅ 完成（A2-1~A2-4 / A2-6 / **A2-9**；**A2-5 已由 C 批在真机完成**，见 §2.3） | `--self-test` 全绿（3 项分层数学 + 7 道护栏 + 1 项 legacy 标注）；真实 `calib_data` 500 文件 smoke 报告正确；C 批：C++ 与 Python 两侧统计给出同一组 n 与分子 | `python3 mini_trt_llm/tools/validate/int8_eval.py --self-test`；`ctest -R int8_eval_selftest`；`ctest -R int8_crosscheck`；沙箱 `ctest` **215/0** | 2026-09-26 |
| B1 P4-INT8-a | ✅ **结案**（根因定位 + 离线反证 + **真机 B1-1~B1-4 全绿**） | 根因 = 权重 scale 取自**未折 BN** 的权重、量化对象是**已折 BN** 的权重（逐通道折叠系数 0.05~19.9）。**离线**（ONNX 官方参考、64 张）：PT 60.9%/100%（n=11）、PC(错源) 25.0%/54.5%、**PC(改源) 57.8%/100%**；PC(错源) 与 #29.2/#29.5 的**真机 TRT 数字逐位相同**。**文件级**：饱和(±127)权重 PT 3.919% / PC(错源) **16.188%** / PC(改源) **0.044%**。**真机 B1**：确定性 22 张量逐位相同；探针自证 8.34e-07 vs 0.0398；引擎 vs 自己的图逐层最大 0.2714（未越界 → 引擎忠实）；探针图复现对照 **PT 12/12=100% / PC 6/12=50%** | `ctest -R 'qdq_reference_selftest\|add_probe_outputs_selftest'`（2/2 Passed）；沙箱 `ctest` **264 条 / 0 失败**；真机 `--gtest_filter='Int8Probe*'` **3/3 Passed**；完整记录 `TROUBLESHOOTING.md` #46 / #47 | 2026-09-27 |
| C 组 | 未触发 | —— | —— | —— |

---

*本计划不含对话过程；判据的唯一来源是 `docs/future_iterations.md` 与各阶段计划 / 测试计划。*

---

## 9. [OI-SAMPLER-KERNEL-TESTS] §9.2 Sampler 高性能 kernel（测试计划）

> 配套开发计划见 `future_iterations_development_plan.md` §10。条目事实来源：`docs/future_iterations.md` §9.2。

### 9.1 分层与环境

层名沿用本计划的 **H / G / P** 三档（刻意不复用 phase 计划的 `L0~L3` / `R0.x`，避免同一编号指两回事）：


<details><summary>展开：9.1 分层与环境 全文</summary>

| 层 | 含义 | 沙箱可跑 | 归属 |
|---|---|---|---|
| **H** | host 契约（workspace 尺寸、参数校验、覆盖率/cutoff 的纯逻辑） | 能 | CI / ctest |
| **G** | 真机数值（kernel 正确性、分布、FP16、大 vocab） | 不能（无设备 → 显式跳过） | 作者真机 |
| **P** | 真机性能（按 G6 口径：≥3 次构建 / ≥20 次推理，报中位数与极差） | 不能 | 作者真机 |

执行口径：沙箱 `ctest --test-dir build`；真机 `MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure`
（不带该变量时 GPU 用例会静默跳过，等于白跑）。**当前真机基线（2026-09-27 复跑，含 S-12/S-14）：
**235 条 / 1 红**——唯一红 = FP16 NaN 复现器（按设计），`int8_crosscheck` 仍按设计跳过；
**沙箱基线（2026-09-27，§10 新增 8 项之后）：242 条 / 0 失败**（§10 之前是 234）。
（**P9_2-5b 改的是 kernel 内部，未新增 GPU 用例 → 真机命令与判据不变**，只是同一批用例要复跑。）

---

</details>

### 9.2 用例清单

#### 9.2.1 既有回归（**主判据：语义不变、判据不改、全部通过**）


<details><summary>展开：9.2 用例清单 全文</summary>

出处：`mini_trt_llm/tests/test_sampler.cpp`（11 条）；它们的判据与依据写在各自注释里，本次**不许改**。

| 编号 | 用例 | 层 | 这条在锁什么 |
|---|---|---|---|
| S-1 | `SamplerTest.WorkspaceSizingIsPositiveAndRejectsInvalidDims` | H/G | workspace 尺寸契约 + 非法维度返回 0 |
| S-2 | `SamplerTest.RejectsNullPointersAndWorkspace` | G | 空指针/空 workspace → 非 `cudaSuccess` |
| S-3 | `SamplerKernelTest.GreedyMatchesArgmax` / `GreedyPrefersLowestIndexOnTie` | G | greedy 语义（并列取小下标）——本次不改 greedy，作对照 |
| S-4 | `SamplerKernelTest.TopKWithKEqualsOneBehavesLikeGreedy` | G | **并列语义**的间接锁（k=1 必等于 argmax） |
| S-5 | `SamplerKernelTest.TopKSamplingIsDeterministicForFixedSeed` / `TopPIsDeterministicForFixedSeed` | G | 同 seed **同实现**可复现（随机数消费方式不变） |
| S-6 | `SamplerKernelTest.TopKResultAlwaysWithinTopKSet` | G | 采样必落在 top-K 集合内。**2026-09-27 澄清口径**：集合改成**并列安全**的 `TopKSetByValue`（`value ≥ 第 k 大值`）——原来的"排序后前 k 个"在数值并列时不良定义（依据见 `TROUBLESHOOTING.md` #36）。**这不是放宽**：无并列行上两者逐元素等价 |
| S-7 | `SamplerKernelTest.TopPWithTinyPicksArgmax` | G | p 极小 → 退化为 argmax |
| S-8 | `SamplerKernelTest.TopKDistributionMatchesSoftmaxProbabilities` | G | **主判据**：词频收敛到解析 softmax 概率（3σ） |
| S-9 | `SamplerKernelTest.TopPWithFullProbabilityDoesNotOverTruncate` | G | p=1 不得过截断（最小概率 token 也必须能采到） |
| S-10 | `test_e2e_mini_decoder.cpp` 的 Top-K（k=1）段 | G | 端到端里采样的接入形态 |
| S-11 | `Fp16PathTest.GreedySamplerMatchesArgmaxOnFp16Logits` | G | FP16 分支（greedy 已覆盖） |

#### 9.2.2 新增用例（本次交付）

| 编号 | 用例名（拟） | 层 | 判据 | 出处 | 状态 |
|---|---|---|---|---|---|
| **S-12** | `Fp16PathTest.TopKSamplingDistributionMatchesAnalyticProbabilitiesInFp16` | G | k ∈ {3, 6}（vocab = 8，**真的发生截断**）：FP32 / FP16 两条路径的词频都收敛到"截断到 top-k 后重新归一化"的解析分布（各 3σ + 1e-3）；两个 dtype 的词频互相也在 3√2·σ 内 | 沿用 S-8 的 3σ 口径；补 `future_iterations.md` §11 的 **P1.5-a**。**改名理由**：FP16 的 harness 在 `test_fp16_paths.cpp` 的 `Fp16PathTest` suite 里（S-11/S-13 同处），不为它另起一个跨文件 suite | ✅ **真机通过（2026-09-27）** → **P1.5-a 据此关闭** |
| **S-13** | `Fp16PathTest.TopPSamplingDistributionMatchesAnalyticProbabilitiesInFp16` | G | FP16 / FP32 两条路径的词频都收敛到"截断 + 前缀内重新归一化"的**解析**分布（各 3σ）；两个 dtype 的词频互相也在 3σ 内 | 沿用 S-8 的 3σ 口径；补 `future_iterations.md` §11 的 **P1.5-a** | ✅ **真机通过（2026-09-27 全量）** |
| **S-14** | `SamplerKernelTest.TopKOnLargeVocabStaysWithinTopKSet` | G | vocab = 128000（合成）与 50257（真实形状）、batch ∈ {1, 2} 下，采样 token ∈ 解析 top-K 集合，k ∈ {1, 8, 64}；k=1 另行复验"采样值 = 全行最大值" | 集合成员关系判定（**无阈值**，因此无出处问题）；集合用**并列安全**的 `TopKSetByValue`。**真机第一版是红的**（2026-09-27）：判据用 `partial_sort` 取前 k 个在并列行上不良定义——128000 的第 64 名有 4 个 token 精确并列，采样到的 45721 被排掉。**修复只动测试参考**，事故已由 `SamplerReferenceTest.IncidentRow64TieIsNotASetMembershipFailure` 固化；完整推导见 `TROUBLESHOOTING.md` #36 | ✅ **真机通过（2026-09-27 复跑）** |
| **S-15** | `SamplerKernelTest.TopPOnLargeVocabStaysWithinNucleus` | G | vocab ∈ {50257（batch 1/8）, 128000}、p ∈ {0.9, 0.99} 下，新实现与 legacy 的采样 token **都** ∈ 解析 nucleus；两者的一致率只**打印**（观测指标，不作判据） | 集合成员关系判定（**无阈值**，因此无出处问题）。**改写原因**：开发计划 §10.11 保留排序、取消候选容量，"覆盖率 / 回退行数"这两个观测对象已不存在 | ✅ **真机通过（2026-09-27 全量）** |
| **S-16** | `SamplerKernelTest.TopPWithFullProbabilityCoversHugeNucleus` | G | 均匀 logits（nucleus = 整行）+ p=1.0：200 次抽样里被选下标的 `min < V/10` 且 `max > 0.9V`。若 cutoff 被截到前 m ≤ 0.9V，全落前 10% 的概率是 0.1^200 ≈ 1e-200 | S-9 的口径（p=1 不过截断）。**改写原因**：5 万词表下逐 token 计数几乎全 0，判不出问题；观测范围塌缩才是可判的信号 | ✅ **真机通过（2026-09-27 全量）** |
| ~~**S-17**~~ | ~~`SamplerTest.WorkspaceSizingAfterRewrite`~~ | — | **取消**：Top-P 新方案保留 CUB 排序、采样 kernel 不用 workspace，`TopPSamplerWorkspaceBytes` 的返回**一个字节都没变**（代码里写明了这一点），没有"改后的尺寸"可测 | 反向查的结论（开发计划 §10.4 的 P9_2-7 行） | ❌ 取消 |
| **S-18** | `SamplerPerf.ThroughputByShape`（**P 层**，已实现） | P | vocab ∈ {50257, 128000} × batch ∈ {1, 8}：warmup 3 + **15 轮**，每轮**正向 + 反向（ABBA）**；每个被测变体测"发射 1 次"与"发射 4 次"两个窗口，取 `(T4−T1)/3` = **净成本（斜率）**——把每窗口固定开销（事件+同步+首次发射，本环境可达几十 µs）减掉；逐轮算 `legacy/parallel` 与 `(top-p − top-k)` 后报**中位数 + p25/p75 + min/max**；同时保留单发口径供与历史基线对齐 | 协议 = G6（≥20 次推理、报中位数与极差）**+ 配对 + 斜率**：`TROUBLESHOOTING.md` #37（分段测没判别力）→ #38（跨协议不可直接比、**固定开销与信号同量级**）。**斜率口径是这类问题唯一够用的尺子** | ✅ 已采集四次（2026-09-26/27），含**同二进制 A/B**。**判据看配对（净）口径**：`legacy/parallel` median = **12.63 / 22.43 / 15.73 / 17.57×**（4/4 达标）。A/B（同一轮交替测 `LaunchTopPSamplerTwoLevel`）判定 **P9_2-5b 效果无显著差异**：中位数 +4.8 / +27.3 / −330.5 / −81.3 µs，p25/p75 全跨 0；同轮量到 greedy 净 = 35~155 µs、`top-p 净 − top-k 净` @50257×1 = 60.1 µs（p25=56.4 / p75=64.0，很紧）→ 采样内核净开销仅"裸读一遍行"的 ~1.7 倍。结论与上限见 `TROUBLESHOOTING.md` #38 / 开发计划 §10.12.8 |

#### 9.2.3 P9_2-5 新增（Top-P 行内并行，2026-09-26）

| 编号 | 用例名（拟） | 层 | 判据 | 出处 | 状态 |
|---|---|---|---|---|---|
| **S-19** | `SamplerKernelTest.TopKFastMatchesLegacyTokens` | G | fast 与 legacy Top-K **逐 token 相同**（4 个形状） | "语义等价"最硬的形态；开发计划 §10.9 | ✅ 真机通过（2026-09-26） |
| **S-20** | `SamplerKernelTest.TopKFastPoisonsRowsAboveContractLimit` | G | `k > kTopKFastMaxK` → 写哨兵 -1 | 契约护栏；同上 | ✅ 真机通过（2026-09-26） |
| **S-21** | `SamplerKernelTest.TopPDistributionMatchesTruncatedSoftmaxProbabilities` | G | **分布级主判据**：词频收敛到"截断 + 前缀内重新归一化"的解析分布（3σ + 1e-3），p ∈ {0.6, 0.9, 1.0} 对应 cutoff = 2 / 4 / 8；新实现与 legacy 各跑一遍 | 沿用 S-8 的口径。**为什么要它**：cutoff 多截/少截一个元素会直接改变分布形状，而集合成员关系看不出来 | ✅ **真机通过（2026-09-27 全量）** |
| **S-22** | `NucleusCutoffTest.*`（4 条，**H 层，沙箱可跑**） | H | 交叉点定位的边界语义，**每条打的是不同分支**：`>=` 含等号（阈值恰好等于某前缀和时截到该元素）、阈值 = 全行和时走到行尾（p=1 不截断）、阈值 > 全行和时走"整行都不够"分支、阈值 ≤ 0 时取首元素 | 被测对象就是产品实现（`sampler/nucleus_cutoff.hpp`，`__host__ __device__`）→ 无"参考漂移"风险。**为什么要它**：内核在沙箱跑不了，而这段最易出 off-by-one。**2026-09-27 去重**：删掉 2 条与 S-24 同断言的用例（`MatchesScalarScanOnGenericData`、二级版退化用例），见开发计划 §10.12.5 | ✅ 沙箱通过（2026-09-27），4/4 |
| **S-23** | `SamplerReferenceTest.*`（3 条，**H 层，沙箱可跑**） | H | 采样器参考实现的 host 自证：① `TopKSoftmaxProbabilities` 是良定义分布（和为 1）、支持集正好是解析 top-k，且截断后 top-1 概率严格变大；② `TruncatedSoftmaxProbabilities` 与 `AnalyticNucleus` 自洽（支持集 ⊆ nucleus、差额 ≤ 登记的那 1 个边界余量、p→0 退化到 argmax）；③ **事故回归** `IncidentRow64TieIsNotASetMembershipFailure`（真机那条红的数据原样固化：`TopKSetByValue` 必须接受并列组里的 45721） | `PROGRESS.md` §2.13「参考必须自带断言 + host meta-test」；这条纪律当场兑现——第 14 轮真机红就是参考错，而不是 kernel 错（`TROUBLESHOOTING.md` #36）。**2026-09-27 去重**：删掉"并列取小下标"（实质在测 `std::stable_sort`）与 `TopKSetByValue` 的抽象版（与事故回归重合） | ✅ 沙箱通过（2026-09-27），3/3 |
| **S-24** | `NucleusCutoffTest.ThreeLevel*`（3 条，**H 层，沙箱可跑**） | H | P9_2-5b 的三级定位（块→子块→元素）：① 通用数据上与"整行逐元素串行扫描"给出同一 cutoff（阈值远离前缀和边界）；② 子块级不命中时退化为整块重扫**且不重复计入块和**（重复计入会让阈值提前命中）；③ 第二次调用尊重 `size = cutoff`（子块越过边界时必须截断） | 被测对象 = 内核真正调用的那份实现（`FindCrossingByLevels`，`__host__ __device__`）；**它挡不住的是内核装配**（共享内存布局 / 子块和怎么算），那部分只能真机 | ✅ 沙箱通过（2026-09-27），3/3 |

---

</details>

### 9.3 判据与出处（每条都要能回答"凭什么"）


<details><summary>展开：9.3 判据与出处（每条都要能回答"凭什么"） 全文</summary>

| 判据 | 值 / 形式 | 出处 |
|---|---|---|

| 分布一致性 | 与解析 softmax 概率比较，容差 `3σ + 1e-3` | `tests/test_sampler.cpp` 的 `TopKDistributionMatchesSoftmaxProbabilities` 注释（本次沿用，不改） |
| FP16 分布一致性 | 同 3σ 口径（**不另立更松的尺子**） | 同上；`AGENTS.md` §7「阈值不跨精度复用」在这里的意思相反——**同一算子同一 dtype 语义，就该用同一把尺子** |
| 集合成员关系 | 布尔判定（token ∈ top-K / ∈ nucleus） | 无阈值 → 无出处问题；构造上保证集合可解析 |
| 性能加速比 | **目标：top-k ≥5×、top-p ≥10×**（出处 = 开发计划 §10.5 第 4 条）。实测：**top-p 对冻结基线 4/4 达标**（13.0 / 18.9 / 13.4 / 15.0 ×）；**同 session 口径 3/4**（50257×1 = 9.99×，差 0.08%）；**top-k fast 不达标**（0.106~0.163×，见 §10.10） | `future_iterations.md` §9.2 的触发时机 + `docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` §5 的 G6；两个口径的读法见开发计划 §10.9.1 |
| 回退行数 / 覆盖率 | **已取消**（Top-P 方案改定后没有候选容量，见开发计划 §10.11） | 原文是"观测指标，不作判据"；保留这句是因为 Top-K 的候选覆盖率仍待做（§10.10 的重做方向） |
| 新实现 vs legacy 的一致率 | **观测指标，不作判据** | 两者累加顺序不同，边界处允许差一格——设成判据就会把登记的允许差异变成回归 |
| 全量基线（沙箱，**当时**） | 242 条 / 0 失败（GPU 用例显式跳过；§10 新增 8 项之前是 234）。**现状见 §10.1 / §11.1 的"当前基线"** | `PROGRESS.md` §3.5 |
| 全量基线（真机，**当时**） | **235 条 / 1 红**（2026-09-27 复跑，含 S-12/S-14；唯一红 = FP16 NaN 复现器，`int8_crosscheck` 按设计跳过）。演进：228 条 / 1 红 → 233 条 / 2 红 → **235 条 / 1 红**。**现状见 §10.1 / §11.1** | `PROGRESS.md` §3.5 / §3.0f |
| 判据红（已修，**非产品缺陷**） | S-14 第一版：参考在数值并列时不良定义（128000 第 64 名有 4 个 token 精确并列）→ 换成 `TopKSetByValue` | `TROUBLESHOOTING.md` #36 |
| 参考实现自证 | 参考是标尺：每条参考都要有 host meta-test 且**能判别**（改了 k/边界会红） | `PROGRESS.md` §2.13；用例 S-23 |

---

</details>

### 9.4 语义等价的范围（写清楚，免得把正常差异当回归）

- **允许不同**：同一 `(seed, offset)` 下逐 token 输出与**旧实现**可以不同——候选归并顺序可能改变
  逆变换的采样点。**这不是回归**。

<details><summary>展开：9.4 语义等价的范围（写清楚，免得把正常差异当回归） 全文</summary>

- **允许不同（Top-P 专属，2026-09-26 补）**：Top-P 的并行实现与 legacy 的差异有两处，都在
  **浮点累加顺序**上：① 分块求和 + 块间串行合并 vs 逐元素串行；② legacy 累加 `exp/total` 再与 `p` 比，
  新版累加 `exp` 再与 `p * Σexp` 比（先除后加 vs 先加后除）。后果是**极端并列 / 恰好落在阈值边界处
  cutoff 可能差一格**。这条比 Top-K 那条宽一档（Top-K 是逐 token 相同），因此 Top-P 的判据
  必须是分布级（S-21）与集合级（S-15/S-16），**不能**写成"逐 token 相同"。
- **必须相同**：① 同 seed 同实现两次运行逐 token 一致（S-5）；② 分布（S-8/S-12/S-13）；
  ③ `k = 1` 等价于 argmax（S-4）；④ p 极小 → argmax（S-7）、p = 1 不过截断（S-9）；
  ⑤ 并列 → 小下标优先（S-3/S-4 的间接锁 + 实现内显式保证，见开发计划 D3）。

---

</details>

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

### 9.7 结果回填（已回填）

> 回填要求：只写"状态 + 实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。


<details><summary>展开：9.7 结果回填（已回填） 全文</summary>

| 用例 | 状态 | 实测值 / 出处 |
|---|---|---|
| S-1 ~ S-11（既有回归） | ✅ **真机通过（2026-09-27 全量）**；沙箱侧 S-1/S-2 通过、S-3~S-11 显式跳过 | 判据一条未改（`git diff` 里测试断言无改动，只多了 `legacy` 开关） |
| S-12 | ✅ **真机通过（2026-09-27 复跑）** → **P1.5-a 关闭** | `Fp16PathTest.TopKSamplingDistributionMatchesAnalyticProbabilitiesInFp16`（k=3/6，真的发生截断） |
| S-13 | ✅ **真机通过（2026-09-27 全量）** | `Fp16PathTest.TopPSamplingDistributionMatchesAnalyticProbabilitiesInFp16` |
| S-14 | ✅ **真机通过（2026-09-27 复跑，判据修正后）** | `SamplerKernelTest.TopKOnLargeVocabStaysWithinTopKSet`（50257/128000，batch 1/2，k ∈ {1,8,64}） |
| S-15 | ✅ **真机通过（2026-09-27 全量）** | `SamplerKernelTest.TopPOnLargeVocabStaysWithinNucleus` |
| S-16 | ✅ **真机通过（2026-09-27 全量）** | `SamplerKernelTest.TopPWithFullProbabilityCoversHugeNucleus` |
| S-17 | ❌ 取消（理由见 §9.2.2） | —— |
| S-21 | ✅ **真机通过（2026-09-27 全量）** | `SamplerKernelTest.TopPDistributionMatchesTruncatedSoftmaxProbabilities` |
| S-22（4 条 host） | ✅ **沙箱通过（2026-09-27 去重后，4/4）** | `ctest --test-dir build`：`NucleusCutoffTest.*` 全绿；它们是本次唯一能在无 GPU 环境下裁决新逻辑的用例 |
| S-23（3 条 host） | ✅ **沙箱通过（2026-09-27 去重后，3/3）** | `SamplerReferenceTest.*`：参考良定义性 / 与 nucleus 自洽 / **第 14 轮真机红的事故回归**；它们是"标尺对不对"的第一道闸，而且这次真的挡住了 |
| S-24（3 条 host，P9_2-5b） | ✅ **沙箱通过（2026-09-27，3/3）** | `NucleusCutoffTest.ThreeLevel*`：三级定位与整行串行扫描同比 / 退化时不重复计入块和 / 尊重 `size` 边界 |
| **S-19** `SamplerKernelTest.TopKFastMatchesLegacyTokens`（fast vs legacy 逐 token 相同，4 形状：k==vocab / 1024 / 50257·k64 / 128000·k64） | ✅ **真机通过**（2026-09-26） | 语义等价成立——这是"逐 token 相同"级别的锁，不是分布级 |
| **S-20** `SamplerKernelTest.TopKFastPoisonsRowsAboveContractLimit`（k > 64 → 哨兵 -1） | ✅ **真机通过**（2026-09-26） | 契约护栏有效 |
| S-18（性能基线 + 改动后复测） | ✅ 完成（基线 2026-09-26、复测 2026-09-27，均真机） | 基线见开发计划 §10.5；复测见 **§10.9.1**：对冻结基线 4/4 达标（13.0 / 18.9 / 13.4 / 15.0 ×），同 session 口径 50257×1 = 9.99×（差 0.08%，原因 = legacy 跨 session 漂移 −23.2%）。**阈值与判据一个字没动** |
| S-15 / S-16 / S-21 的"观测项"回填 | ✅ | 逐 token 一致率（观测，非判据）：50257×1 = 2/3、50257×8 = 17/24 与 12/24、128000×1 = 0/3。**词表越大越不一致**符合 §9.4 的机制（命中"下标"对累计和舍入敏感：1e-5·total 的台阶 vs ε·√N 的误差）；n=3 不足以谈差异率，故只打印 |

---

</details>

## 10. [OI-PERF-PROFILE-TESTS] decode 端到端性能画像 + G6 测量方法（测试计划）

> 配套开发计划见 `future_iterations_development_plan.md` **§11**。
> 条目事实来源：`docs/future_iterations.md` **§6.3** 与 **§11 的 G6**。
> **基线影响**：本节新增 **8 项**（5 条 `PerfStatsTest.*` + `Gpt2DecodePerf.StepLatencyByPhase`
> + `Gpt2DecodePerf.ContextLengthSweep` + `profile_summary_selftest`）→ 沙箱总数 234 → **242**。
> 本计划原写"总数不变"是**错的**——只要新增 `TEST()` 或 ctest 项，总数就会变；
> 当时那句把"H 层用例"误当成"不算用例"，已当场改回（`AGENTS.md` §5 第 4 条）。
> **事后回填（2026-09-27）**：真机复跑实际是 **259 条 / 1 红 / 1 跳过**——当时外推的"243"
> **不准**（见 §11.1 的更正：ctest 总数在沙箱与真机是**同一个数**，差异只在 GPU 用例跑还是跳过）。
> **同日整轮全量复跑：264 条 / 1 红 / 1 跳过**（449 s）→ 基线钉到 264。
> **后续（2026-09-27 复跑，provenance 改造后）：264 条 / 1 红 / 0 跳过 / 310 s**——现状见 `PROGRESS.md` 的「当前基线」。

### 10.1 分层与环境

沿用本计划的 **H / G / P** 三档（见 §1）。本轮以 **P** 为主，**G** 为辅，**H** 只用于协议里
纯统计逻辑的自证（中位数 / 分位 / 斜率 / 极差）。

<details><summary>展开：10.1 分层与环境 全文</summary>


| 层 | 含义 | 沙箱可跑 | 归属 |
|---|---|---|---|
| **H** | 统计工具自身的自证（中位数 / p25 / p75 / 斜率 / 极差） | 能 | CI / ctest |
| **G** | 真机时间线（nsys 无头导出、kernel 名与耗时） | 不能 | 作者真机 |
| **P** | 真机性能（G6 口径：同轮 ABBA / 跨构建对照） | 不能 | 作者真机 |

执行口径同 §1：沙箱 `ctest --test-dir build`；真机 `MINI_TRT_REQUIRE_GPU=1 ctest ...`。
**当前基线**：沙箱 **268 条 / 0 失败**；真机**整轮全量 = 267 条 / 1 红 / 0 跳过 / 301.72 s**
（2026-09-28 复跑的数；该轮在 `check_skips_selftest` 注册之前 → 下次真机应为 268）。
唯一出处见 `PROGRESS.md` 当前基线。
两边总数相同，差别只在 GPU 用例跑还是跳过。
（本节原写"沙箱 242、真机复跑应为 243"是**当时的历史快照**，已作废。）

</details>

### 10.2 用例清单


<details><summary>展开：10.2 用例清单 全文</summary>

| 编号 | 用例 / 项（拟） | 层 | 测什么 | 判据 | 出处 | 状态 |
|---|---|---|---|---|---|---|

| **PF-1** | `ProfileSmoke.NsysHeadlessExport`（由 `run_profile.sh` 承担，非 gtest 用例） | G | nsys 在 WSL2 能无头导出 `.nsys-rep` | 文件存在且非空 | `AGENTS.md` §1 | ⏳ **真机已试跑**：可导出（resnet18/gpt2 都生成了报告），但暴露"假 CSV"等问题，已修 → `TROUBLESHOOTING.md` #39 |
| **PF-2** | `ProfileSmoke.NcuHeadlessExport`（由 `run_profile.sh` 承担） | G | ncu 无头导出 `.ncu-rep` | 文件存在；**本机实测拿不到** | `TROUBLESHOOTING.md` #41 | ❌ **实测不可用**：`==ERROR== Unknown Error on device 0.`、无 `.ncu-rep`（用例本身 PASSED）→ 记为已知环境限制 |
| **PF-3** | `Gpt2DecodePerf.StepLatencyByPhase`（**P 层，纯打印**） | P | prefill 一步 vs decode 每步的中位数 / p25 / p75，n ≥ 15 | **只打印观测值，不设阈值**（漂移也改为只打印；见 #39） | 开发计划 §11.6"sampler 占比有结论"；无阈值 → 无出处问题 | ✅ 已实现；**真机首跑红在我加的错断言上**（0.6 ms 跨场景复用）→ 已改只打印；**出数待重跑** |
| **PF-4** | `Gpt2DecodeBreakdown.OurVsTrtVsCub`（由 `summarize_nsys.py` 承担） | G / P | 开发计划 11.4 的三层分解 + TRT 前 N 条 kernel | 三层时间之和 vs 总时间的偏差**写出** | `PROGRESS.md` §2.14 C"诊断必须自证" | ✅ 脚本 + `--self-test` 已就位；**受阻**：nsys 报告不含 GPU kernel 数据、ncu 报 `Unknown Error on device 0` → **本机两条 CLI 路径都拿不到 kernel 时间线**；绕法见 #41（推荐同 session 比值法）；**出数待跑** |
| **PF-5** | 分配开销（改由 `summarize_nsys.py --api` 承担） | P | `cudaMalloc` / `cudaFree` 的次数与总耗时 | 只打印；为 §2.1 供数 | §2.1 触发条件"先量分配开销" | ✅ **真机已有初步数据**（`cuda_api_sum` 在 WSL2 可用）：resnet18 一次运行 `cudaFree` 6 次 / 317 ms、`cudaMalloc` 3 次 / 4 ms；GPT-2 首跑 `cudaMalloc` 63 次。**注意 `cudaFree` 含隐式同步，不等于纯分配器成本**；且未区分"建引擎期"与"`Generate` 期" |
| **PF-6** | `PerfStatsTest.*`（**H 层，沙箱**） | H | 中位数 / 最近秩分位 / 斜率 / 极差的计算自证 | 已知输入 → 已知输出（含空输入哨兵、最近秩不插值、斜率扣固定开销） | `PROGRESS.md` §2.13"参考实现要自证"；`test_sampler.cpp` 的 harness 是现成参考 | ✅ **沙箱通过**（2026-09-27，5/5） |
| **PF-7** | `OnnxVsNative.PerfPerBuildMedian`（**P 层，纯打印**） | P | ONNX vs 原生 prefill：同 session ≥3 次构建 × ≥20 次推理 | 极差 < 中位数差 → 可判；否则"**未定**" | `future_iterations.md` §10.2 的**前置**；`docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` §3.1 | ⏳ **待真机**（协议已定，无新代码）。**注意：这项服务 §10.2（ONNX 子图替换），已从 §11 的收口范围移交出去**——§11 已于 2026-09-27 关闭（开发计划 §11.5.2） |
| **PF-8** | `Gpt2DecodePerf.StepLatencyByPhase` 的**同 session sampler 占比段**（**P 层，纯打印**） | P | greedy / top-k(k=64) / top-p(p=0.9) 三种采样在 vocab=50257、batch=1 下的**净成本**，及其占本次 decode 每步的比例 | **只打印比值**（无阈值）；斜率口径 `(T4−T1)/3`、正反交替、n=9 | `TROUBLESHOOTING.md` #41 绕法 1（两条 CLI profiling 路径都不可用时，用**同 session 两数相除**回答"sampler 占多少"） | ✅ **真机通过并与 `SamplerPerf` 互校一致**（2026-09-27，同 session）：greedy **0.0308**(0.0336) / top-k **0.4360**(0.4281) / top-p **0.4938**(0.4916) ms（括号= `SamplerPerf`，差 ≤2%）→ 占 2.458 ms 步的 **1.25% / 17.7% / 20.1%**。首版用**全零**输入曾差 2.5×，根因即退化输入（#42） |
| **PF-9** | `Gpt2DecodePerf.ContextLengthSweep`（**P 层，纯打印**） | P | prompt ∈ {4, 256, 960} 三档下各自的**每步 decode 耗时**（斜率 `(T32−T1)/31`），以及"每 1000 个上下文位置涨多少 ms" | **只打印，不设阈值**；斜率法（与 PF-8/§9.2 同口径）+ 正反交替 | 开发计划 **§11.4.1**（profiler 拿不到 kernel 时间线时的替代法：#41）；用来给 `future_iterations` **§2.2** 供判据 | ✅ **真机通过并出数**（2026-09-27）：ctx 20.5 → **3.055 ms/步**、ctx 272.5 → **6.062**、ctx 976.5 → **14.705**；两段斜率 **11.93 / 12.28 ms per 1000**（差 3% → **线性**，外推 1024 = +12.2 ms）→ **长上下文下 attention ≈ 每步 80%**。**据此 §2.2 的触发条件成立，已从 P3 升 P2**（`future_iterations.md` §0.1/§0.2/§0.3-14） |

**明确不做**：不新增 GPU 数值断言（那是 §9.2 的职责）；P 层用例**不许**把观测值变成阈值判据
（否则就是 `AGENTS.md` §7 禁止的"用阈值换绿"）。

</details>

### 10.3 判据与出处（每条都要能回答"凭什么"）

| 判据 | 值 / 形式 | 出处 |
|---|---|---|
| 判别下限 | ±400~600 µs（本平台实测） | `TROUBLESHOOTING.md` #37 / #38 |
| 同二进制 A/B | ABBA + 斜率 `(T4−T1)/3`，报中位数 + p25 / p75 | #37（分段没判别力）→ #38（跨协议不可比、固定开销与信号同量级） |
| 跨构建对照 | ≥3 次构建 / ≥20 次推理，报中位数与极差 | `future_iterations.md` §11 的 G6 |
| 报告完整性 | 六项：形状 / 构建态 / n + warmup / 中位数 + 分位 / 温度时钟 / 命令 | 开发计划 §11.3 D |
| 分解一致性 | 三层之和 vs 总时间的偏差写出 | `PROGRESS.md` §2.14 C |
| 不掺实现 | 本轮**零产品代码改动** | 开发计划 §11.7 末行 |

### 10.4 覆盖缺口（本计划明确不覆盖什么）

- **不覆盖** `ncu` 的深层指标（只保证导出路径打通到"能读出 kernel 名与耗时"）；
- **PF-8 给的是比值、不是逐 kernel 分解**：attention / MLP 仍包在"decode 一步"里，
  且 sampler 是在**静态 logits 缓冲**上量的（cache 状态与真实循环不同）→ 只到"占比量级"；
  要精确分解仍需 profiler（本机受阻，见 #41）；
  **互校已通过**（#42 已定位：首版全零输入是退化样本；换成与 `SamplerPerf` 同一模式后两套
  harness 在同一 session 内差 ≤2%）→ PF-8 的百分比可用，但仍受下面两条限制；
  **占比只在同一 session 内可比**：decode 步本身跨 session 已见 2.458 / 2.847 / 3.365 ms（±27%），
  跨 session 比百分比会犯 #38 的同一错误；
- **不覆盖** continuous batching / 多请求（依赖 §2.3 / G2-3）；
- **不覆盖** FP16 路径（GPT-2 FP16 产 NaN，`PROGRESS.md` §5.11）；
- **不覆盖** CUDA Graph 捕获下的时间线（当前 runner 未用 graph capture）。

### 10.5 结果回填（已回填）

> 回填要求：只写"状态 + 实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。


<details><summary>展开：10.5 结果回填（已回填） 全文</summary>

| 用例 | 状态 | 实测值 / 出处 |
|---|---|---|
| PF-6（`PerfStatsTest.*`，5 条 host） | ✅ **沙箱通过**（2026-09-27） | `ctest --test-dir build -R PerfStatsTest`：5/5 Passed；锁住最近秩不插值、空输入哨兵、斜率扣固定开销 |
| `profile_summary_selftest`（分桶护栏，含在 §10 的 7 项里） | ✅ **沙箱通过**（2026-09-27） | 合成 CSV → 已知分桶；坏输入（缺列 / 缺时间 / 空数据 / 字段数不匹配）必须报错 |
| PF-3（`Gpt2DecodePerf.StepLatencyByPhase`） | ✅ **真机通过并出数**（2026-09-27，重跑；引擎 `cache hit` ×2） | T(1) median **4.947 ms**（p25 4.730 / p75 6.298）、T(32) median **93.190 ms**（p25 90.109 / p75 103.367）；派生 **decode 2.847 ms/步**、**prefill≈2.100 ms**。同 session 漂移 **18.5% / 10.2%**（GPU 72→78 °C、44.5→65.7 W）→ 机器未进稳态。首跑曾红在我加的 0.6 ms 错断言上，见 `TROUBLESHOOTING.md` #39 |
| PF-1 ✅ / PF-2 ❌ / PF-4 ⬜能力边界 / PF-7 ➡️已移交 | PF-1 跑通（报告可导出）；**PF-2 ncu 实测不可用**；**PF-4 记为本机能力边界**（两条 CLI 路径都拿不到 kernel 时间线，工具与自检都在）；**PF-7 移交 §10.2**。**§11 已于 2026-09-27 关闭**（开发计划 §11.5.2） | 见 `TROUBLESHOOTING.md` #39 / #41；命令见开发计划 §11.9 |
| ncu 不可用这件事 | ❌ **实测确认**（2026-09-27） | `==ERROR== Unknown Error on device 0.`；被 profile 的用例本身 PASSED（`29786 ms`）。**profiler 下的耗时不可用**——判别特征：派生 `prefill≈-2.246 ms` 为负。见 #41 |
| PF-8（同 session sampler 占比） | ✅ **通过，并与 `SamplerPerf` 同 session 互校一致** | decode 步 **2.458 ms**；greedy **0.0308**(0.0336) → **1.25%**、top-k(64) **0.4360**(0.4281) → **17.7%**、top-p(0.9) **0.4938**(0.4916) → **20.1%**（括号内 = `SamplerPerf` 同 session 值，差 ≤2%）。首版全零输入给出 0.92/4.75/6.35%，**已作废**（退化样本，见 #42） |
| PF-9（上下文扫描） | ✅ **真机通过并出数**（2026-09-27） | prompt 4/256/960 → 每步 **3.055 / 6.062 / 14.705 ms**（平均上下文 20.5 / 272.5 / 976.5）；斜率 **11.93 / 12.28 ms per 1000 位置**（差 3% → 线性）→ **长上下文 attention ≈ 80%**。**引擎**：首次构建了新 prefill（627 MB）+ 新 decode（475 MB）。**修掉一处自伤**：最初的 decode 复用主用例路径，因指纹覆盖整个 Config 而与之**交替重建**；已改为独立路径 `..._ctxsweep_decode.engine` |
| 结论去向 | ✅ 已回填 | §2.2 触发成立 → `future_iterations.md` §0.1 升 **P2**、§0.2 记标签差异、§0.3 新增第 **14** 项；机制（每层 12 block / 24 SM、有效带宽 ≈ 峰值 3%）写进 §2.2 正文 |
| PF-5（分配开销） | ✅ **真机已有初步数据**（2026-09-27） | `nsys stats --report cuda_api_sum` 后处理（无需 GPU）：GPT-2 一次运行 `cudaMalloc` **63 次 / 3.335 ms**、`cudaFree` **69 次 / 249.717 ms**；resnet18 一次运行 `cudaMalloc` 3 次 / 4.010 ms、`cudaFree` 6 次 / 317.020 ms。**限制**：`cudaFree` 含隐式同步（是拆除期成本，不是分配器成本）、且统计未区分建引擎期与 `Generate` 期 → 只能当量级读；**结论：§2.1 的"频繁分配拖慢 decode"未获支持** |
| 真机首跑暴露的三个问题 | ✅ 两个已修、一个记为环境限制 | ① gpt2 target 失败 = 用例退出码被 nsys 透传（根因是我的错断言）；② `nsys stats` 消息混进 CSV → "假 CSV"；③ WSL2 无 GPU kernel 时间线。全部见 `TROUBLESHOOTING.md` #39 |
| 沙箱全量 | ✅ **242 条 / 0 失败**（2026-09-27） | `ctest --test-dir build` |

---

</details>

## 11. [OI-FLASHDECODING-TESTS] §2.2 长上下文 attention：split-K（测试计划）

> 配套开发计划见 `future_iterations_development_plan.md` **§12**。
> 条目事实来源：`docs/future_iterations.md` **§2.2**。
> **状态：已执行并回填（2026-09-27；PS-1~PS-8 / PP-1 / PP-2 结果见 §11.5）。**
> 按 `AGENTS.md` §0.7，计划文档本身不构成开工许可——当初的开工由作者点名发生。
> **基线影响（已落地）**：新增 **H 组 8 条**（比原计划多 PS-7 = workspace 契约、PS-8 = 分解数学自洽）
> + **G 组 7 条** + **P 组 2 条**（PP-1 已实现、PP-2 落成独立的 A/B 用例），
> 另有 **2 条既有端到端用例复用**（不新增编号）→ 沙箱总数 **242 → 259**
> （2026-09-27 实测）。**教训沿用**：只要新增 `TEST()`，`ctest` 总数就变，落地后必须回填
> 本节与 §1 的总数（这条更正来自 §10 开头那次误判）。
> 本节沿用 §10 的分工：不复制开发计划的目标 / 做法 / 验收，只写"用例 → 判据 → 出处 → 环境 → 状态"。
> **编号消歧**：本节启用后，本文件里的裸 `§11` 指**本节**；指向别处的一律带文件名
> （`future_iterations.md` §11 / 开发计划 §11）。§1 已记过一次"同一编号指两件事"的亏，
> 这里主动说明，免得下一个人把 §10 里的 `§11` 读成本节。

### 11.1 分层与环境

沿用本计划的 **H / G / P** 三档（见 §1）。执行口径同 §1：


<details><summary>展开：11.1 分层与环境 全文</summary>

```bash
# 沙箱 / CI：H 层实际执行，G / P 层显式跳过
ctest --test-dir build

# 真机：G / P 层；跳过即失败
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure
```

| 层 | 本轮测什么 | 沙箱可跑 | 归属 |
|---|---|---|---|
| **H** | `PlanSplits` 的分片边界与空分片语义（纯函数） | 能 | CI / ctest |
| **G** | split-K kernel 的数值正确性（vs CPU double 参考）与 FP16 分支 | 不能 | 作者真机 |
| **P** | kernel 级 / 端到端两版的**同轮交替**性能对照 | 不能 | 作者真机 |

**当前基线**：沙箱 **268 条 / 0 失败**；真机**整轮全量 = 267 条 / 1 红 / 0 跳过 / 301.72 s**
（2026-09-28 复跑的数；该轮在 `check_skips_selftest` 注册之前 → 下次真机应为 268）。
唯一出处见 `PROGRESS.md` 当前基线。
（唯一红 = 按设计的 `RealGpt2Fp16GreedyMatchesReferenceTokens`；`int8_crosscheck` 报告齐备 → Passed，
只在缺报告时按设计跳过 77）。下面这段"259"是本节**收口当天**的快照，保留作为那次复跑的记录。
**更正**：本节上一版写"真机复跑应为 260"——**错了 1 条**。ctest 的**总数**在沙箱与真机是
同一个数（差异只在"GPU 用例跑还是跳过"）；历史笔记里"沙箱 242 / 真机 243"那个 +1 口径
另有来源，需要时单独核对，**不要再拿它外推总数**。

</details>

### 11.2 用例清单

**H 组（沙箱，`paged_attention_split.hpp` 的纯函数）——建议 6 条**


<details><summary>展开：11.2 用例清单 全文</summary>

| 编号 | 用例（拟） | 测什么 | 判据 | 出处 |
|---|---|---|---|---|
| **PS-1** | `PagedAttentionSplitPlanTest.CoversRangeWithoutGapOrOverlap` | `split_idx` 递增时各段 `[begin,end)` 首尾相接；并集 == `[0,total_len)` | 逐段断言（**无缝无叠**） | `PROGRESS.md` §2.12（部分写入路径必须显式处理未覆盖区间） |
| **PS-2** | `PagedAttentionSplitPlanTest.HandlesNonDivisibleLength` | `total_len=100 / num_splits=3` 的切法确定且末段 `end == total_len` | 与写死的期望切法逐段相等 | 同上（"确定"才有可复现性） |
| **PS-3** | `PagedAttentionSplitPlanTest.EmptyChunksWhenSplitsExceedLength` | `total_len < num_splits`：前 `total_len` 段各 1 元素、其余为空段 | 空段 `begin == end` | 空分片必须被显式识别，否则 merge 读到未初始化显存 |
| **PS-4** | `PagedAttentionSplitPlanTest.SinglePositionWithEmptyCache` | `context_len=0` 且带当前 token → `total_len=1`，恰好一段含该位置 | 段数与内容 | `PagedAttentionKernelTest.CurrentTokenIsAttendedEvenWithEmptyCache`（现有语义） |
| **PS-5** | `PagedAttentionSplitPlanTest.ChoosesOneSplitForShortContext` | num_splits 选择策略：`total_len ≤ kTargetChunk → 1`（走单趟，不发 stage-2） | 边界两侧各测一次 | 开发计划 §12.3 D2（短上下文不退化成两次发射） |
| **PS-6** | `PagedAttentionSplitPlanTest.ClampsAtMaxSplits` | `total_len` 很大时 `num_splits ≤ kMaxSplits`（= workspace 契约的上界） | 上界断言 | 开发计划 §12.3 D3（`getWorkspaceSize` 的上界必须 ≥ 实际用量） |
| **PS-7** | `PagedAttentionSplitPlanTest.WorkspaceBytesCoverUsedRange` | `PagedAttentionWorkspaceBytes()` 的返回值 ≥ 各 `effective_splits` 下实际写入的区间上界 | 逐 split 索引算最大字节偏移，断言 < 返回的字节数 | 开发计划 §12.3 D3 的契约（**同一条纯函数被 `getWorkspaceSize` 与 kernel 共用**，避免两边各写一份算法） |
| **PS-8** | `PagedAttentionSplitPlanTest.SplitMergeDecompositionMatchesDirectSoftmax` | "切分 → 逐片 `(m,l,acc)` → max-trick 归并"这套**算法**是否等价于一次性 softmax | 对**直接 softmax**（独立参考）`EXPECT_NEAR(..., 1e-9)`；覆盖 `total_len ∈ {1,3,127,128,129,500,977}` × `head_size ∈ {8,64}` × override ∈ {0,1,3,8} | kernel 在沙箱跑不了，但**算法能在沙箱裁决**（开发计划 §12.3 D4 的同一精神）；挡"归并公式写错/余数摊错/空片没跳过"这类静默偏差 |

**G 组（真机，GPU + TRT）——建议 7 条**

| 编号 | 用例（拟） | 测什么 | 判据 | 出处 |
|---|---|---|---|---|
| **PG-1** | `PagedAttentionSplitKernelTest.MhaMatchesCpuReferenceAcrossMultipleBlocks` | MHA、跨多个物理块、`num_splits=4` | vs CPU **double** 参考：rel < 1e-4 / abs < 1e-5（**不放宽**） | 现 `MhaMatchesCpuReferenceAcrossMultipleBlocks` 的 split 版；阈值出处见开发计划 §12.1 反向查 |
| **PG-2** | `PagedAttentionSplitKernelTest.GqaAndMqaShareKvHeads` | GQA（4/2）与 MQA（4/1）各一次 | 同 PG-1 | 现 `GqaSharesKvHeadsAcrossQueryHeads` |
| **PG-3** | `Fp16PathTest.PagedAttentionSplitMatchesFp16Reference` | FP16 的 split-K 分支 | 同现有 FP16 参考口径 | 现 `Fp16PathTest.PagedAttentionMatchesFp16Reference` |
| **PG-4** | `PagedAttentionSplitKernelTest.MultiBatchMatchesCpuReference` | `batch=2`、两条序列各自不同 `context_len` | 同 PG-1 | `PROGRESS.md` §2.13（带 batch 的算子必须覆盖 `batch>1`） |
| **PG-5** | `PagedAttentionSplitKernelTest.ZeroContextLengthProducesZeros` | `context_len=0`（不带 / 带当前 token 两种） | 不带 → 全 0；带 → 逐元素等于 `value_new` | 现 `ZeroContextLengthProducesZeros` / `CurrentTokenIsAttendedEvenWithEmptyCache` |
| **PG-6** | `PagedAttentionSplitKernelTest.ExplicitEmptyChunksAreSkippedSafely` | 强制 `num_splits > total_len`（含 `total_len=1`） | 同 PG-1；且**不产生 NaN** | 空分片哨兵语义（H 组 PS-3 的 kernel 侧对照） |
| **PG-7** | `PagedAttentionSplitKernelTest.MatchesSinglePassKernel`（**诊断，非判据**） | 同一输入下新旧两版 kernel 的差异 | **只打印** `max_abs` / `max_rel` / 分布；先量"与正确性无关的差异"（float32 累加顺序） | `AGENTS.md` §7"放宽阈值前必须先量无关差异"；观测值若高几个数量级 → **查，不改阈值** |

**P 组（真机性能）——建议 2 条**

| 编号 | 用例（拟） | 测什么 | 判据 | 出处 |
|---|---|---|---|---|
| **PP-1** | `PagedAttentionSplitPerf.SlopeByContextLength` | kernel 级：合成 GPT-2 形状、`context_len ∈ {32,256,1024}`、两版同轮交替、斜率 `(T4−T1)/3` | **只打印**中位数 + p25/p75 + 斜率；无阈值 | 开发计划 §12.5；`TROUBLESHOOTING` #37 / #38 |
| **PP-2** | `Gpt2DecodePerf.ContextLengthSweep`（**复用既有 PF-9**，用 override 切两版） | 端到端每步 decode 三档耗时与斜率 | **只打印**；是否设"达标线"见开发计划 §12.6 第 5 条（F1 待拍板） | 测试计划 §10.2 PF-9；开发计划 §11.4.1 |

**G 组之外的复用（不新增编号）**：`Gpt2DecodeConsistency.*`（decode 一步 == prefill 对应位置）
与 `Gpt2GenerateTest.RealGpt2GreedyMatchesReferenceTokens`（8-token 逐 token 基线）——
它们是"改 kernel 没改语义"的硬判据（开发计划 §12.6 第 3 条）。

**明确不做**：不新增"放宽阈值"的用例；P 组**不许**把观测值变成阈值判据
（否则就是 `AGENTS.md` §7 禁止的"用阈值换绿"）；**不**在本轮覆盖 PagedAttention 的 prefill
（§9.1 冻结）、**不**覆盖 batch 扩展（G2-3 / §2.3）。

</details>

### 11.3 判据与出处（每条都要能回答"凭什么"）

| 判据 | 值 / 形式 | 出处 |
|---|---|---|
| 数值正确性 | rel < 1e-4 / abs < 1e-5（对 CPU double 参考） | `test_paged_attention_plugin.cpp` 的既有算子口径；**阈值出处待作者确认后补注**（开发计划 §12.1 反向查） |
| 语义不变 | GPT-2 8-token 贪心输出逐 token 一致 | `PROGRESS.md` §3.0a 的冻结基线 |
| 分片完备性 | 各段无叠无漏、空段显式哨兵（`m=-inf, l=0, acc=0`） | `PROGRESS.md` §2.12；`TROUBLESHOOTING` #4 |
| 新旧差异 | 只打印，先量"无关差异"（float32 累加顺序）；**不设阈值** | `AGENTS.md` §7 |
| 性能测量 | 同二进制 / 同 session / 同轮 ABBA + 斜率；报中位数 + p25/p75 + n + 温度 | `TROUBLESHOOTING` #37 / #38 / #39 |
| 判别下限 | 整步 decode **不能**套 ±400~600 µs（那是采样器量级的） | `TROUBLESHOOTING` #39 |
| 性能达标 | **5A**：ctx≈960 斜率至少降 40%（**F1=A 已拍板**，实测 88.85% / 88.20%）→ 达标；**5B**：ctx≈20 档退化**改为观测项、不设判据**（作者 2026-09-27 决定）——原"不超过同 session 漂移"因"漂移"三义、曾议的 2% 无出处、两把尺子差 1.97× 未解释而作废 | 开发计划 §12.6 第 5 条 / §12.4 末段；`TROUBLESHOOTING` #45 / #45.1 |
| 缓存安全 | 版本 bump 后第一次 `stale` 重建、第二次 `cache hit` | `PROGRESS.md` §3.0f |

### 11.4 覆盖缺口（本计划明确不覆盖什么）

- **不覆盖** PagedAttention 的 **prefill**（query 序列长度 > 1）——§9.1 冻结；
- **不覆盖** `LLMRunner` 的 **batch 扩展**（G2-3 / §2.3）——本条不扩 batch（开发计划 §12.1 的偏差处理）；
- **不覆盖** FP8 / INT8 KV cache（`future_iterations.md` §1.2，前置是 `PagedAttentionPlugin` 支持 INT8，未触发）；
- **不覆盖** CUDA Graph 捕获下的时间线（runner 未用 graph capture）；
- **不覆盖** 逐 kernel 的 profiler 分解（本机拿不到 GPU 时间线，`TROUBLESHOOTING` #41）；
  PP-1 / PP-2 给的是**斜率与比值**，不是 attention 的绝对耗时；
- **不覆盖** 跨 session 的性能结论（`decode` 步跨 session 已见 ±27%，占比只在同 session 内可比）。

### 11.5 结果回填（已回填）

> 回填要求：只写"状态 + 实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。
> **本节已于 2026-09-27 全部回填**（最后三行是 2026-09-27 补的——它们在收口时漏回填，见下表说明）。

<details><summary>展开：11.5 结果回填（已回填） 全文</summary>


| 用例 | 状态 | 实测值 / 出处 |
|---|---|---|
| PS-1 ~ PS-8（H 组） | ✅ **沙箱通过**（2026-09-27） | `ctest -R PagedAttentionSplitPlanTest`：**8/8 Passed**；锁住"无叠无漏 / 不整除切法确定 / 空片 / 只有当前 token / 目标片长边界 / 上限钳位 / workspace 覆盖最大偏移 / 分解数学自洽"。**护栏自证**：把"余数摊给前几片"临时改坏后 PS-1/2/3/8 当场转红，复原后全绿 |
| PG-1 ~ PG-7（G 组） | ✅ **真机通过**（2026-09-27 复跑） | 修掉夹具/断言后复跑全绿；`PG-7` 诊断用例也 Passed。首轮那 4 条红的根因与修法见 `TROUBLESHOOTING` #44（**全在测试侧**，产品代码一行未改）。**PG-7 打印的 `max_abs`/`max_rel` 数值尚未采集** |
| PG-3（FP16，单列） | ✅ **真机通过**（2026-09-27） | `Fp16PathTest.PagedAttentionSplitMatchesFp16Reference`：137 个位置上**自适应 2 片**与**强制 8 片**两种切法都对 double 参考在 1e-3 内吻合，护栏区未被改写 → **否证了"归并丢片"** |
| ~~PP-1（kernel 级 A/B）~~ | ✅ 见下方同名行 | **原写"未开始"是漏回填**（该行与下面的同名行重复）；实际结果见下面那行 |
| PP-2（端到端 A/B） | 🟡 **只有生产路径（split）的单点观测**，**不是 A/B**（2026-09-27 真机） | prompt 4 / 256 / 960 → 每步 **3.155 / 4.056 / 4.438 ms**（T(1)=14.773/32.622/124.994，T(32)=112.581/158.363/262.559）；端点斜率 **1.3415 ms / 1000 位置**（对照 PF-9 基线 3.055/6.062/14.705 与 ~12.19）→ **约 9× 下降**。**⚠️ 三条限制，缺一条这个数就不能用**：① **跨 session**，未做同 session A/B（`TROUBLESHOOTING.md` §38 禁止直接比）；② **数值正确性尚未验证**（本轮两条 GPU 数值 ctest 实际没跑，见 `TROUBLESHOOTING` #43）；③ **两段斜率不再相等**（3.575 / 0.542 ms per 1000）→ 片数随上下文变，"线性"假设已不成立，端点斜率只是粗摘要 |
| PP-1（kernel 级 A/B） | ✅ **真机通过并出数**（2026-09-27） | GPT-2 真实形状（heads=12 / head_size=64）、每轮连发 32 次只同步一次、ABBA、n=7。**每层每步耗时（ms）**：ctx=32 → 单趟 **0.023600** / split **0.025764**（0.916×，**多一次归并发射，小回归**）；ctx=256 → **0.174143 / 0.094844**（1.836×）；ctx=1024 → **0.918948 / 0.125568**（**7.318×**）。**斜率（每 1000 位置）**：单趟 **0.902569** / split **0.100609** → **降幅 88.853%** |
| PP-2 A/B（端到端） | ✅ **真机通过并两次出数**（2026-09-27） | 同 session + 同引擎 + 逐轮 ABBA、n=5。**第 2 次（带漂移锚点）**：prompt=4 → split **2.77545** / 单趟 **2.72432**（0.982×）；256 → **3.62925 / 5.4907**（1.513×）；960 → **3.9932 / 13.0443**（**3.267×**）；斜率 split **1.2738** / 单趟 **10.795** → **降幅 88.20%**（第 1 次 88.12%，**两次独立复现**）。**三档 token 全部一致=是**。GPU 58→71 °C。**两臂自身漂移**：4 → 0.97% / **12.29%**；256 → 0.71% / 0.21%；960 → 2.09% / 0.03%。**F1-B 见开发计划 §12.4 与 `TROUBLESHOOTING` #45（"漂移"有歧义、`max` 锚点被首档未稳态污染、代码里的"保守/宽松"写反已修）** |
| **PG-7 诊断数值** | ✅ **已采集**（2026-09-27） | 同一输入下新旧两版 kernel：`max_abs=3.57628e-07`、`max_rel=1.52484e-06` —— 比算子判据（rel<1e-4）低约 65 倍，**就是 float32 累加顺序不同的量级**；按 §7，没有任何放宽阈值的理由 |
| **PP-1 ↔ PP-2 互校** | ✅ 一致（差约 5%） | PP-1 单趟 ctx=1024 = 0.918948 ms/层 ×12 层 = **11.03 ms/步**；PP-2 单趟斜率 10.8048 ms/1000 × 0.976 = **10.5 ms/步** → 两把独立尺子一致（`PROGRESS.md` §2.13 的"参考要互校"） |
| 引擎缓存行为 | ✅ **真机已验证**（2026-09-27） | 图版本 bump 后**首次** `Engine cache stale → 重建`：prefill **627 MB** + decode **475 MB**（分钟级，属预期）；**第二次起 `Engine cache hit`** 且不再重建 → 指纹机制按设计生效（出处：`PROGRESS.md` §3.0i） |
| 端到端语义（复用用例） | ✅ **真机通过**（2026-09-27） | `Gpt2DecodeConsistency.*` 与 FP32 8-token 冻结基线**逐 token 一致**；PP-2 打印"三档 prompt 的生成 token 与单趟完全一致"（出处：`PROGRESS.md` §3.0i、开发计划 §12.4） |
| 沙箱全量（**当时**） | ✅ **257 条 / 0 失败**（2026-09-27） | `ctest --test-dir build`；原 242 + 本轮 15（H 8 实跑 + GPU 7 跳过）。**现状见 §1 的"当前基线"（268 / 0）** |

---

*本计划只含执行安排与判据；各条目目标与触发条件的唯一来源仍是 `docs/future_iterations.md`。*

</details>
