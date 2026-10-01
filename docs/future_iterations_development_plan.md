# future_iterations 开发计划（触发驱动的分批执行计划）

> **状态**：2026-09-26 产出。**分批计划已执行完毕**——批次 A/B/C 与 §9~§13 的执行条目均已跑完并回填；
> **现状与测试基线一律以 `PROGRESS.md` 的「当前基线」为准**。本文件是 `docs/future_iterations.md` 的**执行层**。
>
> **分工（避免两处来源漂移）**：
> - **排序 / 级别定义 / 触发条件 / 目标 / 做法 / 验收判据** → `docs/future_iterations.md`
>   （§0 优先级、`future_iterations.md` §1.5 P4-INT8-a、`future_iterations.md` §1.6 P4-INT8-b、§11 缺口索引）；
> - **本文件只回答**：轮到某条时"改哪些文件、分几步、每步怎么自检、哪一步要你批"。
>   条目内部的做法与验收**不复制过来**——复制出来的第二份必然漂移。
>
> 配套测试计划：`docs/future_iterations_test_plan.md`（用例 → 判据 → 出处 → 环境 → 状态）。
>
> **开发状态落点（2026-10-01 起）**：各 feature 的**状态与设计**在 `docs/dev/<feature>/`
> （入口 = `STATE.md`；阶段链与 Gate 见 `AGENTS.md` §5 与技能 `trt-inference-engineering`）。
> 本文件仍是"怎么做"的执行层：轮到某条时在这里看改动面 / 步序 / 破坏性清单。

---

## 0. 计划对账（AGENTS.md §5 第 0 步）

### 0.1 有没有计划文档

**有。** 能力条目与触发条件在 `docs/future_iterations.md`（2026-09-26 已重定优先级，含 §0 总表）；
本文件是它的执行层，测试侧在 `docs/future_iterations_test_plan.md`。
两条**已立项**条目（P4-INT8-a → `future_iterations.md` §1.5、P4-INT8-b → `future_iterations.md` §1.6）的目标 / 做法 / 验收已在该文件里写全。

### 0.2 逐条对照：本文件的任务 / 接口 / 验收 vs 现状

| 本次要做的 | 现状 | 是否一致 |
|---|---|---|
| 把 31 个条目按触发条件分批 | `future_iterations.md` §0.1 已给出级别与理由 | 一致，本文件只做"批次化"（同级内按依赖排序） |
| 给"可立即开工"的条目写可执行步骤 | §5.1 与 `future_iterations.md` §1.6 离线子项此前只有"工作内容"级描述 | **有偏差**：缺文件级改动面、步序、自检点 → 本文件 §2 补齐 |
| 给"触发即做"的条目定开工前提 | `future_iterations.md` §1.5 已有做法与成本校准 | 一致；本文件补"先修仪器"的硬前置与真机往返预算 |
| 给"需外部前置"的条目定第一步 | §0.1 只有一行备注 | **有意保持粗粒度**：触发后各条按 AGENTS.md §5 升格为独立 phase 计划，本文件不预写细节（预写必然过期） |
| 测试侧判据 | 各条目只写了主判据，没有用例清单与分层 | **有偏差** → `future_iterations_test_plan.md` |

### 0.3 偏差怎么处理

先补文档（本文件 + 测试计划），再动手；条目内部做法仍以 `future_iterations.md` 为准，本文件不覆盖它。
任何一次实际开工前，重新跑一次本节的对照（现状会变，尤其"前置依赖是否还在"）。

### 0.4 反向查：文档与现状矛盾之处（当场修）

- **【已修，2026-09-26】我上一轮把 §5.1 的数据前置写错了。** `future_iterations.md` §0.1 原写
  "`vocab.json` + `merges.txt` 本地没有 → 缺文件时本条降为 P2"。查证后是**我错**：

<details><summary>展开：0.4 反向查：文档与现状矛盾之处（当场修） 全文</summary>

  本地 HF 缓存里有 gpt2 的完整 tokenizer 文件——
  `~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/{vocab.json,merges.txt,tokenizer.json}`，
  且实测 `transformers 4.44.0` 能 `local_files_only=True` 离线加载并给出参考 token id
  （`"The quick brown fox"` → `[464, 2068, 7586, 21831]`，`vocab_size = 50257`）。
  **结论：`future_iterations.md` §5.1 保持 P1，不需要联网。** 我当时的判断只查了 `models/gpt2/`，把"仓库里没有"
  当成了"机器上没有"——这正是 `AGENTS.md` §7 禁止的"观测缺口当成现象"。
- **【已修，2026-09-26，同一次自检发现】** `future_iterations.md` §5.1 原先没写"byte-level BPE 需要
  **两个**文件"这个接口事实，而 `BaseTokenizer::Load` 只有一个路径参数（见 §2.1 的 F2）。
  已在该条目补"接口事实"一行（入参语义 = 目录，不改基类）。这不是矛盾，是缺口——
  但它属于"下个会话照文档做就一定会卡住"的那类缺口，所以当场补而不是留给执行时。
- **【已修，2026-09-27，§2.2 立项时】** `future_iterations.md` §2.2 的"支持 decode 阶段的
  batching 优化"与代码现状矛盾——`PagedAttentionDecodeKernel` 的 `grid.y = batch_size`，
  kernel **本来就支持任意 batch**（`PagedAttentionKernelTest.MhaMatchesCpuReferenceAcrossMultipleBlocks`
  已覆盖 `batch=2`）；把 batch 卡在 1 的是 `LLMRunner`（缺口 G2-3，属 `future_iterations.md` §2.3）。该句已作废，
  本条范围收敛为"只做上下文维 split-K"。详见 **§12.1**。
- **【已登记，未改代码，2026-09-27】** `tests/test_paged_attention_plugin.cpp` 的主判据阈值
  `1e-4f / 1e-5f` **旁边没有出处**（`AGENTS.md` §7 要求写清出处）→ §2.2 只补出处、不放宽；
  出处待作者确认。详见 **§12.1**。

</details>

### 0.5 需要你拍板的决策（F1～F4）

| 编号 | 决策 | 选项 A（推荐） | 选项 B | 影响 |
|---|---|---|---|---|
| **F1** | GPT-2 tokenizer 数据文件（`vocab.json` + `merges.txt`）放哪 | **不入库**：测试通过 CMake 变量 / 环境变量 `MINI_TRT_GPT2_TOKENIZER_DIR` 指向（默认先看 `models/gpt2/`，再看 HF 缓存路径）；缺文件 → **跳过并打印探测结果**（与 `models/gpt2` 缺 safetensors 同口径） | 把两个文件复制进 `models/gpt2/` **入库**（实测 **≈1.5 MB**：`vocab.json` 1,042,301 B + `merges.txt` 456,318 B；`.gitignore` 只忽略 `*.bin`/`*.onnx`/`*.safetensors` 等，`*.json`/`*.txt` 会被跟踪） | 选 A 与仓库既有规矩（大产物不入库、换机器按表重建）一致；选 B 换来"克隆即可跑"，代价是仓库体积与第三方文件版权/来源记录。**A 的代价**：HF 缓存不是仓库产物（且在缓存里是 `blobs/` 的软链接），被清理后用例会跳过——这正是"跳过必须打印探测结果"的用处 |
| **F2** | `BaseTokenizer::Load(const std::string& vocab_path)` 单参数怎么承载"两个文件" | **不改基类**：`BpeTokenizer::Load` 把入参解释为**目录**，取其中的 `vocab.json` / `merges.txt` | 改基类加 `Load(vocab, merges)` 重载（要让 `SentencePieceTokenizer` 也实现，改动面扩散到既有 tokenizer） | 选 A 不动已交付接口；语义写在注释里（"路径含义由子类决定"是基类原话） |
| **F3** | BPE 参考数据怎么固化 | **Python 侧导出 golden 文件**（`tests/data/gpt2_tokenizer_golden.json`，含每个样本的文本 + ids + 生成环境），C++ 侧 host 用例读它比对；另有 meta 自证用例校验 SHA256。**golden 文件入库**（体积小、它是标尺；生成脚本与 SHA256 一起提交），与本仓库"大产物不入库"的规矩不冲突 | C++ 用例内直接调 Python（引入运行时依赖，且 CI 语义变复杂） | 与 `PROGRESS.md` §2.13"参考实现唯一 + 自带断言 + host 侧 meta-test"一致 |
| **F4** | 批次 A 的开工顺序 | **A2（INT8 口径，纯离线文档 + 脚本，风险最低）先行，A1（BPE）紧随** | A1 先行 | 顺序不影响正确性；A2 更小、更快形成一次完整的"计划 → 实现 → 自检 → 回填"闭环 |

> 这四条只影响"怎么落"，不影响任何验收判据；未获答复时按选项 A 执行。

---

## 1. 分批依据

分批**只依据 `future_iterations.md` §0.1 的触发状态与外部前置**，不引入新的优先级口径：

| 批次 | 含义 | 条目 | 现在能开工吗 |
|---|---|---|---|
| **A** | 无外部前置，且触发已成立 | §5.1（BPE Tokenizer）、`future_iterations.md` §1.6 的离线子项 | **能**（见 §2） |
| **B** | 无外部前置，但触发未成立（触发即做） | `future_iterations.md` §1.5（P4-INT8-a） | 触发后能（见 §3） |
| **C** | 有外部前置（联网 / 新硬件 / 先补测量 / 需求未定） | §1.2、`future_iterations.md` §1.6 整条、§2.1、§2.3、§2.4、§3.1、§3.2、§4.1、§4.2、§6.1~`future_iterations.md` §6.4、§9.2、§9.3、§10.1 | 不能（见 §4） |
| **D** | 冻结备查 | `future_iterations.md` §1.1、§9.1 | **不做**（见 §5） |

**批次规则**：一批一次收口——每批结束时跑真机全量、回填文档、把新增的"坑"写进
`docs/TROUBLESHOOTING.md`（结论留 `PROGRESS.md`）。批次内可以并行，批次间不并行，
以便"一次真机往返只验一批"（`AGENTS.md` §7"每轮只改一个变量"的同一精神）。

---

## 2. 批次 A：现在就能开工

### 2.1 A1 = BPE Tokenizer（`future_iterations.md` §5.1，P1）

**目标（一句话）**：让框架具备"文本进 / 文本出"的 GPT-2 路径——目前 `LLMRunner` 只收发 token id，
`tokenizer_` 仅在构造时校验（`include/mini_trt_llm/core/llm_runner.hpp:16`）。

<details><summary>展开：2.1 A1 = BPE Tokenizer（`future_iterations.md`  全文</summary>


**前置（已核实，2026-09-26）**：tokenizer 数据文件在本地 HF 缓存，**不需要联网**（见 §0.4）。

**改动面（文件级）**：

| 文件 | 动作 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/tokenizer/bpe_tokenizer.hpp` | 新增：`BpeTokenizer : public BaseTokenizer`，`Load` 语义 = **目录**（F2） |
| `mini_trt_llm/src/tokenizer/bpe_tokenizer.cpp` | 新增：byte-level BPE（`vocab.json` + `merges.txt`，GPT-2 的 `Ġ`/字节回退语义） |
| `mini_trt_llm/tests/test_bpe_tokenizer.cpp` | 新增：host 用例 + 参考 meta 用例（见测试计划 §2.1） |
| `mini_trt_llm/tests/data/gpt2_tokenizer_golden.json` | 新增：Python 参考产物（F3） |
| `mini_trt_llm/tools/make_tokenizer_golden.py` | 新增：生成上面那份 golden（`transformers` 离线加载，见 §0.4 的实测命令） |
| `mini_trt_llm/tests/CMakeLists.txt` | 若 golden 用 ctest 脚本项：注册（缺文件返回 77 → Skipped） |

**下面 4 项是"实现期才发现需要"的追加项**（见 §2.1.1 的偏差说明）：

| 文件 | 动作 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/utils/json.hpp` | **改动**：补 `\uXXXX`（含 UTF-16 代理对）。GPT-2 的 `vocab.json` 5 万个 key 全是 `"\u0120the"`，原解析器遇到 `\u` 直接抛 `unknown escape sequence` → **不补这一步 A1 根本开不了工** |
| `mini_trt_llm/tests/test_json.cpp` | 新增：上面那处改动的覆盖（6 条：单转义 / 代理对 / 长度必须 4 字节 / 孤立代理报错 / 非法十六进制报错 / ASCII 转义不回归） |
| `mini_trt_llm/tests/tokenizer_test_support.hpp` | 新增：共享夹具（定位 tokenizer 目录 / 参考文件）。两个用例文件都要用，路径规则写两份必然漂移 |
| `mini_trt_llm/tests/test_gpt2_generate.cpp` | **改动**：追加 A1-10 桥接用例（断言 `Encode(kPromptText) == kExpectedPrompt`；层级由 G 改 H 的理由见测试计划 §2.1） |

#### 2.1.1 计划与实现的偏差（2026-09-26 回填）

核对方法：`--gtest_list_tests` 列出的用例名 与 本文档 + 测试计划里的用例名逐条比对；
文件面比对"计划改动面表" 与 `git status` 实际改动。

**结论：10 条计划的用例名 10/10 对得上**（含 A1-10 的层级调整，已在测试计划里写明理由）；
**文件面有 4 处计划没写、实现里动了**，即上面那张追加表。原因分两类：

- `utils/json.hpp` + `tests/test_json.cpp`：**计划时没查"词表长什么样"就写了改动面**——
  以为只需读两个文件，实际 `vocab.json` 的转义形式直接卡住公共解析器。
  教训与 §0.4 那条同源：**先看数据，再定改动面**。
- `tokenizer_test_support.hpp` + `test_gpt2_generate.cpp`：计划里 A1-10 打算自建一套真机夹具，
  实现时发现"复用既有常量 + 抽共享路径夹具"更省且判据更硬（详见测试计划 §2.1 的 A1-10 说明）。

> **不需要改 `mini_trt_llm/CMakeLists.txt`**：它用的是 `file(GLOB_RECURSE ...)`（会收录 `src/tokenizer/` 下的新
> `.cpp`），但 **GLOB 只在 configure 时求值**，所以新增文件后必须重跑一次 configure（否则链接期
> `undefined reference to vtable`）。测试侧 `tests/CMakeLists.txt` 也是 GLOB `*.cpp`，同理。

**步骤（每步都要有可观察产物）**：步骤 ID 用 **`A1-S<n>`**，与测试计划的**用例 ID `A1-<n>`** 分开——
两套 ID 若同名，"A1-3 是步骤还是用例"就要靠猜，而这个项目已经明确禁止"同一编号指两件事"。

1. **A1-S1 接口与语义定稿**：定 `Load(目录)` 的失败语义（缺文件 / 坏 JSON / 词表与 merges 不自洽分别返回什么），
   写进头文件注释（`cpp-comment-style`：注释写 Why）。
2. **A1-S2 golden 生成器**：`make_tokenizer_golden.py` 离线加载，导出样本集与 ids；**同时导出环境信息**
   （`transformers` 版本、快照目录、两个文件的 SHA256）——这一份就是后续弃用的唯一依据。
3. **A1-S3 实现 Encode**：按 byte-level BPE 逐段实现；**先让 golden 里的逐 case 全等**，再谈性能。
4. **A1-S4 实现 Decode 与 `VocabSize`**；`Decode` 只对 golden 的 ids 判（任意 ids 不可逆是 BPE 的固有性质）。
5. **A1-S5 接入与收口**：写出 `LLMRunner` 联动用例（测试计划 **A1-10**），跑真机全量，回填文档。

> **实现只用 `vocab.json` + `merges.txt`**：快照里还有一个 `tokenizer.json`（1.36 MB，HF fast tokenizer 的
> 合并产物）——**不要**读它，否则等于把"我们的 BPE 是否正确"换成"我们是否正确复刻了 HF 的 JSON 结构"。

**已知最容易错的地方（开工时按此设用例）**：前缀空格（`Ġ`）、连续空格、UTF-8 字节回退
（中文 / emoji 会被切成字节 token）、特殊 token 是否参与 BPE、数字的分词边界。

**明确不做**：`tiktoken`（`future_iterations.md` §5.2）、chat template（§5.3）、tokenizer 级批处理、性能优化。

**破坏性动作**：**无删除、无覆盖既有文件**。实际动到的既有文件有三个，全是**追加**：
`tests/CMakeLists.txt`（注册 ctest 项）、`tests/test_gpt2_generate.cpp`（加一条桥接用例）、
`include/mini_trt_llm/utils/json.hpp`（补 `\uXXXX` 转义——原先该分支直接报错，属"打开以前走不通的路径"）。
**已执行（2026-09-26）**，其余全部是新增文件。

</details>

### 2.2 A2 = INT8 判据的离线口径定义（`future_iterations.md` §1.6 的离线子项，P1）

**目标（一句话）**：把"INT8 怎么算合格"写成可执行的判据规格 + 脚本，**不下载任何数据**。


<details><summary>展开：2.2 A2 = INT8 判据的离线口径定义（`future_iterations.md` 全文</summary>

**前置**：无。现有资产：`models/resnet18/*.meta.json`（已有 `sha256` 字段体例）、
`scripts/ref_resnet18.py`（基线生成）、`tests/test_resnet18_int8.cpp`（余量分层统计的 C++ 现役实现）。

**改动面（文件级）**：

| 文件 | 动作 |
|---|---|
| `mini_trt_llm/tools/validate/int8_eval.py` | 新增：读 logits + meta → 输出**分层报告**（每层：率 + 样本量 n） |
| `mini_trt_llm/tools/validate/README.md` | 新增：验收集规格（来源 / 版本 / SHA256 / 与 `calib_data` 的重叠排除规则） |
| `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` + PH4-INT8-CRITERIA | 改：把新规格接进判据表（**先改文档**） |
| `docs/dev/REQ-007-resnet18/phase4_test_plan.md` R2.6 | 改：判据出处指向新规格 |
| `mini_trt_llm/tests/CMakeLists.txt` | 加：脚本自检项（`--self-test`，缺 Python/资产返回 77） |

**步骤**：

1. **A2-1 写规格**：验收集"必须有什么字段才能被引用"（来源 / 版本 / SHA256 / 标签 / 与标定集的重叠排除）。
2. **A2-2 写分层统计**：口径与 C++ 现役实现**必须一致**（同一 `margin` 定义、同一分桶边界来源）；
   两者不一致时以实测交叉校验为准（测试计划 A2-5）。
3. **A2-3 加护栏自检**：脚本对下面 **7 类**输入必须**拒绝**，每种都要有一份故意改坏的输入把它打出来
   （`PROGRESS.md` §2.13「护栏必须有用例证明它会拦人」）：
   ① 缺 provenance（`manifest_sha256` 缺失）；② 验收集与标定集重叠（文件名或 `sha256` 命中）；
   ③ 字段类型错（`num_samples` 是字符串）；④ 清单被改过（`sha256` 不符）；
   ⑤ logits 大小与 `shape` 不符；⑥ 报告只报率不报 n；⑦ 率与分子分母不自洽。
   **已实现并全部通过**（2026-09-26，`--self-test`，沙箱可跑、不联网）。
4. **A2-4 不下载**：规格里可以写"将来到哪里取数据"，但本轮**不执行**任何下载。

**明确不做**：下载验收集、定绝对误差阈值（那需要真数据，见 §4 的 C 组）。

**破坏性动作**：改动 `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` + PH4-INT8-CRITERIA 与 `docs/dev/REQ-007-resnet18/phase4_test_plan.md` R2.6 的**判据表文字**
（不删条目、不改判据本身）——按 `AGENTS.md` §0.5，动手前把这两处改动一次性列给你确认。
**已执行（2026-09-26 获你批准）**：两处均为**追加**指向新规格与脚本，未改动任何既有判据；
另按同一批批准把 `int8_eval_selftest` 注册进 `tests/CMakeLists.txt`。

---

</details>

## 3. 批次 B：触发即做

### 3.1 B1 = P4-INT8-a（`future_iterations.md` §1.5）

> **执行细节已细化到本文件 §13**（2026-09-27）。下面这一节保留为"开工硬前置 + 改动面"的入口，
> 与 §13 冲突时以 §13 为准（§13 是同一轮的文档，含计划对账与测量口径）。

<details><summary>展开：3.1 B1 = P4-INT8-a（`future_iterations.md` §1.5 全文</summary>


**触发条件**：需要更高 INT8 精度（当前 per-tensor 已达标，故未触发）。
**做法 / 验收**：见 `future_iterations.md` §1.5——本文件不复制。

**开工硬前置（顺序不能颠倒）**：

1. **先修仪器**：上次 3 轮真机往返里有 2 轮耗在探针自身的 **BN 折叠错**（`TROUBLESHOOTING.md` #30.5）。
   所以第一步是让探针**自证**：同一输入下，探针读到的"量化前"张量必须与
   torch 已折叠 BN 的模型在同一点对拍（用例见测试计划 B1-1）。
2. **再取数**：探"**量化前**"的 float 张量（量化后的会被 bin 边界 ±1 格噪声淹没，见 #30.6）。

**改动面（文件级，预告）**：

| 文件 | 动作 |
|---|---|
| `mini_trt_llm/tools/convert/quantize_resnet18.py` | 加开关：导出带"量化前张量输出"的探针图。**默认输出到新路径**（如 `models/resnet18/resnet18_qdq_probe.onnx`），不覆盖正式产物 |
| `mini_trt_llm/tests/test_resnet18_int8_probe.cpp` | 新增：仪器自证 + 逐层曲线报告 |
| `docs/TROUBLESHOOTING.md` | 新增节：结论（无论"定位到层"还是"查空"） |
| `docs/future_iterations.md` + OI-INT8-PERCHANNEL | 回填状态 |

**真机预算**：每轮 1 次往返；本轮自带仪器自证，目标 ≤3 轮。
**每轮只改一个变量**（`PROGRESS.md` §2.14 C）。

**破坏性动作（预告，执行前逐条确认）**：

1. 修改 `quantize_resnet18.py`（**脚本**，不是配置文件）；
2. 新增 `models/resnet18/resnet18_qdq_probe.onnx`（新文件，不覆盖）；
3. **若**需要重新生成正式 `resnet18_qdq.onnx` 或覆盖它 → **单独确认**；
4. 删除 `/tmp/mini_trt_llm_resnet18_*.engine` 以强制重建（缓存只按路径名区分）。

---

</details>

## 4. 批次 C：需要外部前置（只到"预备"深度）

> **本表是 2026-09-26 的计划快照**："前置/触发后的第一步"两列描述的是**当时**的状态，
> 其中若干条目此后已交付（如 §6.3 / G6 的 profile、§9.2 采样器、§1.6 的离线子项）——
> **哪些已交付以 `future_iterations.md` §0.1 的状态列为准**，不要照本表判断"还没做"。

**共同纪律**：触发后**先产独立 phase 计划**（`AGENTS.md` §5 第 0 步），再动手；
本表只写"前置是什么 + 触发后的第一步"，不预写做法细节（预写必然过期，且会与将来那份 phase 计划形成两份来源）。

| 条目 | 前置是什么 | 触发后的第一步 | 要联网吗 |
|---|---|---|---|
| `future_iterations.md` §1.6 整条（INT8 验收集） | 带真值标签的验收集 | **先获批**，再下载并登记 SHA256 + 重叠排除（规格来自 A2） | **要** |
| `future_iterations.md` §1.2 LLM INT8 / INT4 | `PagedAttentionPlugin` 只支持 `kFLOAT`/`kHALF`（`paged_attention_plugin.cu:265`）→ 先扩 INT8 KV cache；INT4 需先评估 sm_75 kernel | 先做"PagedAttention INT8 KV cache"的设计 + 契约用例 | 否 |
| ~~§2.1 显存池~~ **已量完（2026-09-27）** | 无前置 | ~~先量一次分配开销（按 `G6` 口径）~~ **已由 §11 的 PF-5 量完**：`cudaMalloc` 63 次 / 3.335 ms → **触发不成立、不排期**（`future_iterations.md` §0.1 / §2.1、`PROGRESS.md` §3.0h 第 3 条） | 否 |
| `future_iterations.md` §2.3 Continuous Batching | 与 `G2-3`（`batch = 1` 限定）耦合 | 先扩 `LLMRunner` 的 batch（含多序列 block 分配与 `context_lens`） | 否 |
| `future_iterations.md` §2.4 CV 动态分辨率 | 无前置，但会牵动 profile 与缓存 | 先改 `AddCvOptimizationProfile` 的接受/拒绝语义（host 可测） | 否 |
| §3.1 Encoder-Decoder / `future_iterations.md` §3.2 ViT | 需要目标模型（HF 权重 + 导出） | 先定模型与 `config.json` 契约 | 可能（取权重） |
| `future_iterations.md` §4.1 GroupNorm / InstanceNorm、§4.2 SiLU / SwiGLU | 需要用到它们的模型（CV / LLaMA 系列） | 先接模型，再写 Plugin（`LayerNorm` 一半已作废，见 §5） | 可能 |
| `future_iterations.md` §6.1 转换工具增强 | 出现新权重来源 | 先定"要支持谁"，再动脚本 | 可能 |
| `future_iterations.md` §6.2 ONNX custom op | 真做子图替换（§10.2） | 先定替换目标子图 | 否 |
| `future_iterations.md` §6.3 Nsight 一键 target | 需要按 `G6` 口径做可复现测量 | 先定测量协议（≥3 次构建 / ≥20 次推理、报中位数与极差） | 否 |
| `future_iterations.md` §6.4 CI | 无前置（本地脚本部分），GitHub Actions 属联网 | 先做本地 `ctest` 脚本，联网部分**先获批** | 部分 |
| `future_iterations.md` §9.2 Sampler 高性能 kernel | **profile 从未做过** | 先做一次 decode 性能 profile，确认 sampler 占比 | 否 |
| `future_iterations.md` §9.3 采样器参考数据固化 | 无前置 | 把 `scripts/ref_sampler.py` 输出落成 `.bin` 供 C++ 载入 | 否 |
| §10.1 ONNX / 原生 I/O 契约统一 | 要让 ONNX 路径接进 `LLMRunner` | 先加 `Cast`（把 `input_ids` 降到 INT32）并补契约用例 | 否 |

---

## 5. 批次 D：冻结备查

| 条目 | 冻结原因 | 解冻条件 |
|---|---|---|
| `future_iterations.md` §1.1 ResNet18 INT8 校准（implicit calibration / `IInt8Calibrator`） | 已被 Q/DQ 显式量化取代；该 API 路线自 TRT 10.12 起弃用 | 若将来某个模型在 Q/DQ 路线上**确实做不到**（例如算子不支持 `QuantizeLinear`），再评估 |
| `future_iterations.md` §9.1 PagedAttention 的 Prefill 阶段 | 触发条件（"Phase 2 走单引擎"）已被现状否证：`LLMRunner` 收 prefill + decode 两个引擎（`llm_runner.hpp:56`） | 真要合成"单引擎含 Prefill"时 |

**冻结 ≠ 删除**：两个条目及其历史理由都保留（`AGENTS.md` §0.5 / §0.6）。

---

## 6. 全局开工纪律（每批都适用）

1. **四步走**：计划对账 → 破坏性动作**一次性清单**确认 → 真机全量回归 → 回填文档。
2. **新增源文件后重跑 configure**：`mini_trt_llm/CMakeLists.txt` 与 `tests/CMakeLists.txt` 都用
   `file(GLOB ...)`，GLOB 只在 configure 时求值；漏跑的症状是链接期 `undefined reference to vtable`。
3. **产物或建图变了先删引擎缓存**：`/tmp/mini_trt_llm_*.engine` 只按路径名区分，不随代码失效。
4. **真机口径固定**：`MINI_TRT_REQUIRE_GPU=1`（否则 GPU 用例静默跳过，等于白跑）。
   当前基线**不在此复制**——唯一出处是 `PROGRESS.md` 的「当前基线」块。（本节曾抄一份，
   结果抄错了一次：写成"沙箱 267 / 真机 264"，而现状是**沙箱 268 / 真机 267**。）
   （沙箱含 §11 的 6 项 host 用例与脚本自检、§12 的 8 项、§13 的 2 条 host 自检，
   GPU / P 层用例在沙箱显式跳过）。红按设计（GPT-2 FP16 NaN，`PROGRESS.md` §5.11）；
   `int8_crosscheck` 报告齐备 → Passed，只在缺报告时按设计跳过。**两边总数相同**，差别只在 GPU 用例跑还是跳过。
   （`AGENTS.md` §7：未跑过的不写"通过"——这一行现在是跑过的。）
5. **跳过或失败都要显式**：`MINI_TRT_SKIP_IF_NO_CUDA` / ctest 的 77；缺资产 → 跳过并打印探测结果。
6. **不擅自删旧模块**：`0_resnet18_onnx/`、`1_gpt2_onnx/` 归作者——**未经点名批准不得删除或移动**。
   Phase 5 已于 **2026-09-28 执行完成**（`PROGRESS.md` §4.6）：两个目录连同 4 条软链接已删除、
   资产在 `assets/legacy/`，执行记录见 `docs/dev/REQ-009-retire-legacy/phase5_development_plan.md`。**本条规则本身不变**——
   今后凡"某个文件还有没有人用 / 能不能删"的判断仍归作者。
7. **阈值纪律**：每个阈值旁写出处；不跨精度复用；放宽前先量"与正确性无关的差异"。

---

## 7. 风险与回退

| 风险 | 影响 | 缓解 / 回退 |
|---|---|---|
| BPE 实现"看起来对"但边界错（前缀空格 / 字节回退） | 端到端 token 与 HF 不一致，且**只在特定文本上暴露** | golden 逐 case 全等 + 专列边界样本（测试计划 A1-3~A1-6）；参考数据版本与 SHA256 一起固化 |
| golden 文件被"顺手改绿" | 标尺失真，后续全错 | 参考 meta 用例校验 SHA256 + 样本数（`PROGRESS.md` §2.13） |
| A2 的脚本口径与 C++ 现役实现不一致 | 同一批 INT8 数据出现两个"一致率" | A2-5 交叉校验：两边必须报出同一组数字（12 张 / 100%） |
| B1 又把预算耗在仪器上 | 真机往返浪费 | 硬前置 = 仪器自证用例先绿；仪器不过不取数 |
| C 组条目被"顺手开工" | 无触发条件就动代码，判据无处可依 | 触发后先产独立 phase 计划；本文件 §4 只给第一步 |
| 文档二次维护漂移 | 下个会话照旧信息开工 | 本文件只写"怎么做"，判据一律指向 `future_iterations.md` 与测试计划 |

---

## 8. [OI-RUNBOOK] 真机执行清单（**已执行**：2026-09-26/27 多轮真机；命令保留供复跑。Agent 侧无 GPU，见 `PROGRESS.md` §5.10）

> 工作区 = 仓库根目录。先构建：`cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j$(nproc)`。
> **顺序不能反**：§8.1（全量）→ §8.2（文本端到端）→ §8.3（INT8 交叉校验）。

### 8.0 前置检查（30 秒，缺一项后面的用例会**跳过**而不是失败）

```bash
ls ~/.cache/huggingface/hub/models--gpt2/snapshots/*/vocab.json   # 缺 → export MINI_TRT_GPT2_TOKENIZER_DIR=<你的 HF gpt2 目录>
ls models/gpt2/config.json models/gpt2/model.safetensors
ls models/resnet18/resnet18_qdq.onnx models/resnet18/config.json
ls assets/legacy/resnet18_onnx/calib_data | head -3
```

### 8.1 真机全量（改动面：无）

```bash
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure
```

- **期望（2026-09-26 快照）**：**204 条**，其中（**现状基线见 `PROGRESS.md` 的「当前基线」：沙箱 267 条 / 0 失败，真机待复跑确认**）：
  - **按设计的红 1 条**：`Gpt2GenerateTest.RealGpt2Fp16GreedyMatchesReferenceTokens`（GPT-2 FP16 已知限制）；
  - ~~当前还有 1 条新红待查：`Gpt2OnnxTest.MatchesAcrossProfileShapes`~~ **已结案**（2026-09-26：
    机制 = 两实现差异之下的并列；按方案 B 修正判据后单测真机复跑 PASSED，不可判行 1/713）→ §8.5 / `TROUBLESHOOTING.md` #34.9；
  - `int8_crosscheck` 在"先跑全量、后跑 §8.3"的顺序下会是 **1 条跳过（77，设计如此——缺报告）**；
    若先跑完 §8.3 再跑全量，它就不跳过。**跳过 ≠ 通过**。
- **除上述之外，任何"跳过"都按故障处理**（说明 §8.0 的资产没到位）。

### 8.2 B：文本端到端（1 条）

```bash
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='*RealGpt2TextPromptEndToEnd*'
```

- **期望**：PASSED。三段判据：`Encode("The quick brown fox") == {464,2068,7586,21831}`、
  生成 `== {274,389,257,1049,835,284,651,257}`、`Decode(全部) == "The quick brown foxes are a great way to get a"`。
- 引擎已缓存则秒级；未缓存则先构建（分钟级）。
- 需要资产：`models/gpt2` + tokenizer 目录。

### 8.3 C：INT8 口径交叉校验（3 步，顺序不能反）

```bash
# C-1 产出 artefact（真机 + GPU）

<details><summary>展开：8.3 C：INT8 口径交叉校验（3 步，顺序不能反） 全文</summary>

MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='*DumpsLogitsAndCppReportForCrossCheck*'

# C-2 用 Python 侧实现算同一批数据（用例输出里会打印等价命令）
python3 mini_trt_llm/tools/validate/int8_eval.py \
  --fp32-logits /tmp/mini_trt_llm_int8_crosscheck/fp32.f32.bin \
  --int8-logits /tmp/mini_trt_llm_int8_crosscheck/int8.f32.bin \
  --meta /tmp/mini_trt_llm_int8_crosscheck/meta.json \
  --calib-dir assets/legacy/resnet18_onnx/calib_data --legacy-mode \
  --json-out /tmp/mini_trt_llm_int8_crosscheck/py_report.json

# C-3 比对两侧口径（不一致就是真问题）
ctest --test-dir build -R int8_crosscheck --output-on-failure
```

- **期望**：C-3 PASSED，并打印"整体 a/b == 整体 a/b、余量子集 12/12 == 12/12、逐桶一致"。
- **C 为什么用 `--legacy-mode`**：当前的测试图与标定集**同源**（都用 `calib_data`），严格模式下脚本会拒绝。
  legacy 模式不放松判据，只在报告里写明"**不满足 `future_iterations.md` §1.6 规格、一致率会被高估**，仅用于口径对齐"。
- **C-3 报差异时**：这是口径漂移（真问题）——**先查两侧口径，不许改阈值**（`AGENTS.md` §7）。

</details>

### 8.4 回填（把结果给我）

把三条命令的关键输出贴回来（§8.1 的总数 / §8.2 的判据 / §8.3 的 C-1 与 C-3 行），
我据此回填 `PROGRESS.md` §3.5 / §3.0e / §6.5 与两份计划的 §8 / §3。

### 8.5 [OI-RED-ARGMAX-CASE] 新红处置：`Gpt2OnnxTest.MatchesAcrossProfileShapes`（seq=512 逐行 argmax）——**已按方案 B 结案（真机复跑通过）**

**为什么单开一节**：它不是本次交付引入的（改动面不触及该路径，见 `TROUBLESHOOTING.md` #34.2），
但它现在是全量报告里除"按设计红"之外的唯一红，必须有人接手——否则下个会话会把它当成噪声。

<details><summary>展开：8.5 [OI-RED-ARGMAX-CASE] 新红处置：`Gpt2OnnxTest.Ma 全文</summary>


```bash
# 复跑：确认五个形状的"不可判行"与 ExpectedUndecidableRows() 登记表逐行一致（`TROUBLESHOOTING.md` §34.9）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='Gpt2OnnxTest.MatchesAcrossProfileShapes'
```

- **实测（2026-09-26 真机）**：PASSED（4.58 s）；不可判行 `(1,1)=0`、`(1,64)=0`、`(1,512)=1`（行 118）、
  `(2,4)=0`、`(2,64)=0` —— 713 行里只有那 1 行，且与登记表逐行一致。
- 判据（"可判行 `m > 2d` 必须全等；不可判行**钉到行号**"）、推导、登记表出处、与 §7 的逐条对照，
  全部写在 `TROUBLESHOOTING.md` **§34.9**；事故经过与数据在 §34.1~§34.8。
- **出现登记表之外的新增行时，动作是"先查原因"**，不是往表里加行——加行等于把真回归吸收掉。
- 沙箱侧的自证：`--gtest_filter='ArgmaxCriterion*'`（6 条 host 用例，含用实测数字复现 #34 的那条）。

---

</details>

## 9. [OI-RUNBOOK-RESULTS] 执行结果（已回填）

> 回填要求：只写"状态 + 产出 + 判据实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。

| 任务 | 状态 | 产出 | 判据实测值 / 出处 |
|---|---|---|---|
| A1 BPE Tokenizer | ✅ **完成**（2026-09-26） | `tokenizer/bpe_tokenizer.{hpp,cpp}`；`utils/json.hpp` 补 `\uXXXX`；`tools/make_tokenizer_golden.py` + `tests/data/gpt2_tokenizer_golden.json`；`tests/test_bpe_tokenizer.cpp`（8 条）+ 参考自证 1 条 + `test_json.cpp`（6 条）+ `test_gpt2_generate.cpp` 的桥接用例 1 条；ctest 项 `tokenizer_golden_check` | `Encode` 与 HF **逐 token 全等**（21 样本：basic 3 / whitespace 5 / utf8 10 / long 1 / edge 2；长文本 306 token）；`Decode` 一致；4 条 `Load` 负例全部拒绝；`ctest -R tokenizer_golden_check` Passed 3.21 s；沙箱全量 **204 条 / 0 失败**。排查过程 → `TROUBLESHOOTING.md` #33 |
| A2 INT8 判据离线规格 | ✅ **完成**（2026-09-26） | `tools/validate/README.md`（规格）、`tools/validate/int8_eval.py`（评估 + 自检 + `--legacy-mode`）、ctest 项 `int8_eval_selftest`；`docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §4 与 `docs/dev/REQ-007-resnet18/phase4_test_plan.md` R2.6 后续行已加指向 | `--self-test` 全绿（3 项分层数学 + 7 道护栏 + 1 项 legacy 标注）；真实 `calib_data`（500 文件）smoke 通过；沙箱全量 **204 条 / 0 失败** |
| A2-5 INT8 口径交叉校验 | ✅ **真机通过**（2026-09-26，作者执行） | C++ 侧 `ResNet18Int8AccuracyTest.DumpsLogitsAndCppReportForCrossCheck`；`tools/validate/int8_eval.py --legacy-mode` + `crosscheck_reports.py`；ctest 项 `int8_crosscheck` / `int8_crosscheck_selftest` | 沙箱：`--self-test` 5 项全过；**真机：三步全过（作者回报"4. pass"）**，说明两侧实现对同一批 logits 给出**同一组 n 与分子** |
| B 文本端到端 | ✅ **真机通过**（2026-09-26，作者执行） | `Gpt2GenerateTest.RealGpt2TextPromptEndToEnd` | **真机：PASSED（作者回报"3. pass"）**——分词 / 生成 / 解码文本三段判据全过，即"文本进 → 文本出"链路成立 |
| A 真机全量 | ✅ **215 条 / 1 红**（2026-09-26 全量重跑确认） | —— | 唯一红 = `RealGpt2Fp16GreedyMatchesReferenceTokens`（按设计）。更早的 204 条 / 2 红 是修正判据前的快照：ONNX argmax 那条机制见 `TROUBLESHOOTING.md` #34.6、判据见 #34.9、复跑实测见 #34.9 末尾 |
| B1 P4-INT8-a | 未触发 | —— | —— |
| C 组 | 未触发 | —— | —— |

---

*本计划不含对话过程，只含执行安排与纪律；事实与判据的唯一来源仍是 `docs/future_iterations.md`。*

---

## 10. [OI-SAMPLER-KERNEL-PLAN] `future_iterations.md` §9.2 Sampler 高性能 kernel（开发计划）

> **本文档是它当年的正式计划落点**（2026-09-26 作者决定：后续迭代中每个事项的开发/测试计划
> 都并入 `future_iterations*_plan.md` 家族）。
> **该决定已于 2026-10-01 被取代**：新 feature 的状态与设计落点改为 `docs/dev/<feature>/`
> （见 `AGENTS.md` §5 与 `docs/future_iterations.md`）；§9.2 是旧口径的**存量**，不回改。
> 配套测试计划见 `future_iterations_test_plan.md` §9。

### 10.1 计划对账（AGENTS.md §5 第 0 步）

#### 10.1.1 有没有计划文档


<details><summary>展开：10.1 计划对账（AGENTS.md §5 第 0 步） 全文</summary>

**没有。** 覆盖本次工作的只有 `future_iterations.md` §9.2 的条目（背景 / 工作内容 / 触发时机），
没有开发计划与测试计划。→ 本文件 + 本文件同级的 `future_iterations_test_plan.md` §9 即为它们。

#### 10.1.2 逐条对照：`future_iterations.md` §9.2 的描述 vs 代码现状

| `future_iterations.md` §9.2 的说法 | 代码证据 | 对上了吗 |
|---|---|---|
| "Top-K / Top-P 目前每步都要对整行 `vocab_size` 做一次降序排序" | `src/sampler/sampler_kernels.cu`：`PrepareSortInputKernel`（物化 fp32 key + 下标）→ `cub::DeviceSegmentedRadixSort::SortPairsDescending`（逐行整段排序）→ `TopKSampleKernel` / `TopPSampleKernel` | **一致** |
| "是明显的性能瓶颈（`vocab_size` 可达 128K）" | 排序 O(V log V) + 物化两份 [B,V] 中间缓冲；Top-P kernel 还要两趟全行扫描 | 一致（但**占比未测**，见 0.3 偏差 1） |
| "工作内容：warp-level Top-K 选择、bitonic sort" | 现状无任何 warp 级选择逻辑 | 一致（尚未做） |
| "与后续 continuous batching 配合" | `LLMRunner` 仍限定 `batch = 1`（缺口 G2-3） | **偏差 2**：本次只保证 kernel 层支持 batch>1（现状已支持多行），**不接**调度 |
| 接口形态 | `sampler_common.hpp` 只有设备侧 API、k/p 是 per-batch 张量、workspace 由调用方提供（`PROGRESS.md` §2.12 的约定） | **本次不改**（见 §2 D1） |
| workspace 谁分配 | `LLMRunner` 构造时按 `TopKSamplerWorkspaceBytes(1, vocab)` 一次性分配（`llm_runner.cpp:181`），测试同样查询该函数 | 说明**缩小 workspace 是兼容改动**（调用方只查大小） |

#### 10.1.3 偏差怎么处理（写在前面，动手前先认账）

- **偏差 1：触发条件（decode 性能 profile）至今没测过。** `future_iterations.md` §9.2 原文的触发时机是"先用 `nsys`/`ncu`
  定位到 sampler 占比显著"。本次由作者决定直接推进 → 处理方式：把"**先测基线**"作为本计划的第一个任务
  **P9_2-0**，这样既给性能结论一个 before 数，也保留一个合法出口：
  **若基线显示 sampler 占比可以忽略，则只落地"正确性/覆盖"改进（FP16 + 大 vocab 覆盖），不换实现**。
- **偏差 2：continuous batching 的对接不在本次范围。** 它依赖 `future_iterations.md` §2.3 + 缺口 G2-3；本次只在 kernel 层
  保证多行正确（既有能力）。

#### 10.1.4 反向查：文档与代码的矛盾（当场记）

- `future_iterations.md` §11 的 **P1.5-a** 说"Top-K / Top-P 的 FP16 分支未覆盖（Greedy 已覆盖）"——
  核对 `tests/test_fp16_paths.cpp`：**属实**（只有 `GreedySamplerMatchesArgmaxOnFp16Logits`）。
  本次顺带补齐，完成后 **P1.5-a 可关闭**并回填 §11。
- `PROGRESS.md` §2.12 的"采样器只有设备侧 API"——核对代码：**仍成立**，本次不动该形态。
- 既有用例的**并列语义**（"降序排序时相同 key 保持原下标顺序"依赖 CUB 的稳定性）——代码里没写明，
  属**隐含契约**。本次实现必须显式保持（见 §2 设计第 4 条）并在测试里锁住。

---

</details>

### 10.2 目标与范围

**目标（2026-09-26 由 §10.11 收窄，原文保留在下面的"做/不做"里以便追溯）**：
Top-K 侧把"整行降序排序"换成"每行一个 block 的部分选择（top-M）+ 只对选出的候选排序"；

<details><summary>展开：10.2 目标与范围 全文</summary>

Top-P 侧**保留 CUB 排序**、只把行内数学块内并行化。两者都在**语义不变**的前提下降低每步采样开销；
同时补齐 FP16 与大 vocab 的覆盖。

**做**：

1. 先测基线（P9_2-0），据此决定"是否换实现"以及性能判据的阈值（**阈值必须由实测给出**）；
2. 新 kernel：`TopKSelectKernel`（block-per-row，strided 扫描 + 块内归并出 top-M，M 自适应）；
3. 对 M 个候选做 `BlockRadixSort`/bitonic 降序；
4. Top-K 采样改走新路径；~~Top-P 在**语义等价**的前提下走新路径，并在"nucleus 超出 M"时**回退**到既有 CUB 路径~~
   → **已被 §10.11 取代**：Top-P 不引入 M、也不需要回退（排序保留，nucleus 一定在排好序的行里）；
5. 补齐 FP16（Top-K/Top-P）与大 vocab（≥50257）的用例；
6. 报告性能（中位数与极差）并回填文档。

**不做**：

- **不改** `sampler_common.hpp` 的 API 形态（设备侧、per-batch k/p、调用方提供 workspace）；
- 不引入 host 侧同步、不在 `Launch*` 内分配显存（`AGENTS.md` §3.B.3 / §3.A.3）；
- 不做 continuous batching 调度、不做温度/重复惩罚等新采样语义；
- 不改 greedy 实现（它已是单击比较，无排序）。

---

</details>

### 10.3 接口与设计（D1~D4）

**D1｜API 形态不变**：只允许改 `TopKSamplerWorkspaceBytes` / `TopPSamplerWorkspaceBytes` 的**返回大小**
与注释（调用方只查询大小，见 §0.2 表）。签名、参数结构、错误约定（返回 `cudaError_t`、越界/空指针返回非

<details><summary>展开：10.3 接口与设计（D1~D4） 全文</summary>

`cudaSuccess`）一律保持。

**D2｜算法（每行一个 block）**：

1. 一趟 strided 扫描求**行 max 与 sum**（Top-P 需要全词表分母；Top-K 的稳定项也用到 max）；
2. 每线程在寄存器里维护自己的 top-M 候选（阈值法：先取一个初值阈值，再按"计数 + 收敛"迭代收紧），
   块内归并出全行的 top-M，并记录**每行的实际候选覆盖率**；
3. 只对这 M 个候选做降序排序（`BlockRadixSort` / bitonic），得到"降序 logits + 原下标"；
4. **采样数学与随机数消费保持与现状一致**：`__expf(logit - max)` → 逆变换 CDF →
   同一个 `Uniform01(seed, offset, row)`（Q8），**不改 Philox 的消费方式**；
5. ~~Top-P 的 nucleus：在排好序的候选前缀上累计；若**累计未达 p**（nucleus 超出 M）→ 该行
   **回退**到既有 CUB 路径~~ → **已被 §10.11 取代**（保留排序后没有"候选容量"这个概念）。

**D3｜隐式契约显式化（并列语义）**：既有实现依赖"相同 logit 时保持原下标顺序"（等价于"并列取小下标"）。
新实现必须在块内排序时**显式**保证同 key 按小下标优先，并在测试里锁住（见测试计划 S-4）。

**D4｜数值路径不变**：softmax 用 `__expf`、减去行 max 稳定化、逆变换 CDF 比较用 `>=`、
Top-P 在保留前缀内**重新归一化**——逐条与现状对齐（这些是本阶段"语义不变"的具体含义）。

---

</details>

### 10.4 任务分解（P9_2-0 ~ P9_2-8）


<details><summary>展开：10.4 任务分解（P9_2-0 ~ P9_2-8） 全文</summary>

| 任务 | 内容 | 依赖 | 产出 |
|---|---|---|---|

| **P9_2-0** | **基线测量**：现有实现的采样耗时与占比（`nsys`/`ncu` + 单测微基准），vocab ∈ {50257, 128000}、batch ∈ {1, 8} | 真机 | ✅ **已完成（2026-09-26 真机）**：基线表见 §10.5（4 个形状的中位数）。仪器 = `tests/test_sampler.cpp` 的 `SamplerPerf.ThroughputByShape`（CudaTimer + warmup 3 + 采样 21 次，报中位数/极差/min-max，并给 greedy 作"一趟扫描"参照）。**更正**：原文写的"基线数据待真机运行"已在同日被推翻，该行当时没回改 |
| P9_2-1 | 固定"语义清单"：把现有实现的语义写成可核对条目（含并列、p 截断、重新归一化、随机数消费） | —— | 测试计划 §3 的判据表 |
| P9_2-2 | top-M 选择 kernel（含阈值迭代与覆盖率统计） | P9_2-1 | `TopKSelectKernel` + host 可测的覆盖率逻辑 |
| P9_2-3 | 候选排序（bitonic/radix，稳定同 key→小下标） | P9_2-2 | 排序 kernel |
| P9_2-4 | Top-K 接新路径 | P9_2-3 | 采样结果与语义清单逐条一致 |
| **P9_2-5** | **Top-P 行内数学并行化**（保留 CUB 排序；方案见 §10.11，不再是"接新选择路径 + 回退"） | P9_2-1 | ✅ 已实现（2026-09-26）：`TopPParallelSampleKernel` + `FindFirstPrefixCrossing`；**正确性真机通过（2026-09-27 全量）**；**性能已实测：按冻结基线 4/4 达标**（详见 §10.9.1） |
| P9_2-6 | FP16 分支覆盖（补 P1.5-a） | P9_2-5 | ✅ **完成**：S-13（Top-P FP16）+ S-12（Top-K FP16）**真机均通过（2026-09-27）** → `future_iterations.md` §11 的 **P1.5-a 已关闭** |
| P9_2-7 | 大 vocab 覆盖（≥50257、合成 128K） | P9_2-5 | ✅ **完成**：S-15（Top-P @50257/128000）+ S-14（Top-K @50257/128000、batch 1/2、k ∈ {1,8,64}）**真机均通过**（S-14 判据修正后复跑）；S-17 已取消（见测试计划 §9.2.2） |
| P9_2-8 | 性能复测 + 文档回填（含 §11 的 P1.5-a 关闭） | 全部 | ✅ **完成**：性能已复测并回填（§10.9.1、`PROGRESS.md` §3.0g）；P1.5-a 已随 S-12 真机通过而关闭；留下的唯一开放项是可选优化 **P9_2-5b**（已登记进 `future_iterations.md` §11） |

---

</details>

### 10.5 [OI-SAMPLER-KERNEL-ACCEPTANCE] 验收标准（每条都要能回答"凭什么"）
> **判据的唯一出处 = 测试计划**（`docs/future_iterations_test_plan.md` §9.3）：本节保留的是**设计侧视角**，与测试计划重复的行**以测试计划为准**。

1. **语义回归（主判据）**：`tests/test_sampler.cpp` 既有 11 条用例**不改判据、全部通过**；
   另有 `test_e2e_mini_decoder.cpp` 的 Top-K（k=1）与 `test_fp16_paths.cpp` 的 Greedy 用例通过。

<details><summary>展开：10.5 [OI-SAMPLER-KERNEL-ACCEPTANCE] 验收标准（每条都要能 全文</summary>

2. **FP16 覆盖（补 P1.5-a）**：新增 Top-K / Top-P 的 FP16 用例，判据沿用既有分布口径
   （与解析 softmax 概率的 **3σ** 比较，出处：`TopKDistributionMatchesSoftmaxProbabilities` 的注释）。
3. **大 vocab 覆盖**：vocab = 128000（合成）与 50257（真实形状）下，Top-K 结果必落在解析 top-K 集合内；
   Top-P 结果必落在解析 nucleus 内。**这两条不设阈值**，是集合成员关系判定（无阈值即无出处问题）。
**P9_2-0 实测基线（2026-09-26，真机，`SamplerPerf.ThroughputByShape`；单位 ms，n=21，报中位数）**：

| 形状 | greedy | top-k (k=64) | top-p (p=0.9) | top-k/greedy | top-p/greedy |
|---|---|---|---|---|---|
| 50257 × 1 | 0.0443 | 0.5671 | 8.1469 | 12.8× | **183.8×** |
| 50257 × 8 | 0.0873 | 0.6685 | 11.4256 | 7.7× | 130.9× |
| 128000 × 1 | 0.0776 | 1.2020 | 15.9150 | 15.5× | **205.0×** |
| 128000 × 8 | 0.1495 | 1.2377 | 20.6862 | 8.3× | 138.4× |

**读出来的三件事**：

1. **top-p 是绝对大头**（8~21 ms，是 greedy 的 131~205 倍）。原因不只是排序：`TopPSampleKernel`
   每行**四趟串行全行扫描**且每趟都要算 `__expf`（求 total / 求 cutoff / 前缀内重新归一化 /
   逆变换采样），
   而每行只由一个线程串行做（`vocab` 5 万~12.8 万）→ SFU 上的 `expf` 成为瓶颈。
   **更正（2026-09-26，见 §10.11）**：原结论"必须同时去掉全行排序 + 把行内数学并行化"把排序
   当成了同量级的成本——实际 CUB 全行排序只占约 0.57 ms（= 同形状 top-k 的耗时，见本表第 2 条），
   所以只把**行内数学**并行化就能拿到主要收益，排序不必动。
2. **top-k 也值得改**（0.57~1.24 ms，7.7~15.5× greedy）。
   **更正（同日复核代码）**：`LLMRunner` 在 `top_k ≤ 1` 时**已短路到 greedy 内核**（`llm_runner.cpp:206`），
   所以"k=1 也付 top-k 代价"的说法**是错的**。真实暴露面 = 用户显式设 `top_k ≥ 2`（0.57~1.24 ms）
   或 `top_p < 1`（8~21 ms）；**后者才是主要收益来源**。
3. **结论：不属于"占比可忽略"那一档**，按 §10.3 的出口 → **换实现**（P9_2-1 ~ P9_2-8）。
   仍未采集的是"sampler 在整步 decode 中的占比"（需要 `nsys` 那一腿）——它不影响这个决定，
   因为绝对量级（128K×8 时单次 20.7 ms）已经足够大。

4. **性能阈值（由上面的基线给出出处，**不是**先写数字）**：目标 **median 加速比 top-k ≥ 5×、top-p ≥ 10×**
   （即 top-k ≤ 0.113 ms、top-p ≤ 0.815 ms @ 50257×1，其余形状按同一比例）。
   这个目标的推导依据是**新算法的趟数**：新实现每行约 2 趟 O(V) 扫描（求 max/sum + 收集候选）+
   对 M 个候选排序，而 greedy 本身就是 1 趟 O(V)；因此"≤ 5× greedy"是可解释的目标，
   而基线 top-k 是 12.8× greedy、top-p 是 183.8× greedy。
   **达标不了时的动作是先查原因，不许直接调低这个倍数**（`AGENTS.md` §7）。
5. **语义等价范围（必须写清楚，否则会被误判为回归）**：允许"同一 `(seed, offset)` 下逐 token 输出与
   旧实现不同"——因为候选归并顺序可能改变逆变换采样点；但**确定性**（同 seed 同实现可复现）与
   **分布**（3σ）必须保持。`top_k = 1` 必须仍等价于 argmax（既有用例 `TopKWithKEqualsOneBehavesLikeGreedy`）。
6. **不回归工程约定**：`Launch*` 内零分配、无 host 同步；`WorkspaceBytes` 与新实现一致（含 0/负例）。

---

</details>

### 10.6 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 覆盖率/回退率在大 p 下偏高 | 性能收益被回退吃掉 | 回退行数**必须打印**；若回退率高，先调 M 与阈值迭代策略，再谈结论 |
| 并列语义（同 key 小下标优先）被破坏 | 与既有用例/采样语义不一致 | P9_2-1 的语义清单 + 测试计划 S-4 显式锁住 |
| 随机数消费方式被顺手改掉 | 同 seed 复现性变化 → 复现器失效 | 明确 D2 第 4 条；测试计划 S-5（同 seed 两次运行逐 token 相同） |
| workspace 需求变化影响调用方 | 运行期越界写 | 只通过 `WorkspaceBytes` 暴露；测试覆盖新尺寸与边界（S-6） |
| 先写性能阈值后测 | 违反 §7（阈值必须带出处） | 验收第 4 条：**阈值由 P9_2-0 实测给出**；先测后写 |
| 新 kernel 只在小 vocab 上"看起来对" | 大 vocab 才暴露的越界/精度问题 | P9_2-7 强制 ≥50257 覆盖 |

---

### 10.7 破坏性动作清单（**预告 → 2026-09-26 执行结果**）

清单本身不删旧条目（留作追溯），实际落点逐条对照如下——**没有任何文件被删除或改名**：


<details><summary>展开：10.7 破坏性动作清单（**预告 → 2026-09-26 执行结果**） 全文</summary>

1. **修改** `mini_trt_llm/src/sampler/sampler_kernels.cu` ✅（P9_2-5 只动了 Top-P：新增
   `TopPParallelSampleKernel`、把 `LaunchTopPSampler` 接到它、并新增 `LaunchTopPSamplerLegacy`
   保留旧 kernel 作对照；Top-K 的 CUB 排序路径未动）；
2. **修改** `mini_trt_llm/include/mini_trt_llm/sampler/sampler_common.hpp` ✅（只加声明与注释：
   `LaunchTopPSamplerLegacy` + Top-P 语义差异说明；**签名一个没动**，`WorkspaceBytes` 返回值也未变）；
3. **新增** 测试文件 ❌ → 改为**并入** `mini_trt_llm/tests/test_sampler.cpp`（S-15/S-16/S-21 + 6 条
   host 用例 `NucleusCutoffTest.*`），规模不需要独立文件；
4. **修改** `mini_trt_llm/tests/test_fp16_paths.cpp` ✅（S-13 = Top-P 的 FP16 分布用例；
   Top-K 的 FP16 用例 S-12 仍未做）；
5. `mini_trt_llm/tests/CMakeLists.txt` ✅ **未动**（没有新增 `.cpp`/`.cu`，无需重跑 configure）；
6. **不改** ✅：`llm_runner.cpp` 的调用形态（它只调 `LaunchTopPSampler`，自动走到新 kernel）、
   `sampler_common.hpp` 的 API、greedy 实现。
7. **新增（清单外，P9_2-5 实施时加）**：
   `mini_trt_llm/include/mini_trt_llm/sampler/nucleus_cutoff.hpp`（`__host__ __device__` 的交叉点
   定位，唯一实现）、`mini_trt_llm/tests/sampler_test_support.hpp`（Top-P 解析参考的唯一实现）。
   **为什么加**：内核在沙箱里跑不了，而"首个累计 >= 阈值"最易出 off-by-one——抽成 host 可测的
   同一份实现，才能让 CI 真的挡人（PROGRESS.md §2.13）。

---

</details>

### 10.8 真机执行清单（**2026-09-27：四步全部已执行**）

> 已执行结果：第 1 步与第 3 步（`ctest` 全量）**228 条 / 226 通过 / 1 红 / 1 跳过**，
> 红 = 按设计的 FP16 复现器、跳过 = `int8_crosscheck`（缺报告 → 77）。

<details><summary>展开：10.8 真机执行清单（**2026-09-27：四步全部已执行**） 全文</summary>

> 第 2 步（性能）**已单独跑完**：按 §10.5 冻结的基线 **4/4 形状达标**（13.0× ~ 18.9×）；
> 同 session 口径 `50257×1 = 9.99×`——数字、两个口径的读法与缺口归因见 **§10.9.1**。

```bash
# 0) 构建（新增 .cu 后必须重跑 configure：GLOB 只在 configure 时求值）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j$(nproc)

# 1) 语义回归 + 新增覆盖（真机）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='Sampler*:Fp16PathTest*'

# 2) 性能（协议见测试计划；P9_2-0 基线先跑一次，改完再跑一次）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='*SamplerPerf*'

# 3) 全量（确认没有连带回归）
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure
```

---

</details>

### 10.9 结果回填（已回填）


<details><summary>展开：10.9 结果回填（已回填） 全文</summary>

| 任务 | 状态 | 判据实测值 / 出处 |
|---|---|---|

| P9_2-0 基线测量 | ✅ **完成（2026-09-26 真机）** | 4 个形状的中位数见 §10.5 的基线表；结论 = **换实现**（top-p 131~205× greedy，top-k 7.7~15.5× greedy）。沙箱侧 `SamplerPerf.ThroughputByShape` 显式跳过；全量 216 条 / 0 失败 |
| P9_2-1 语义清单 | ✅ 完成 | 写在测试计划 §9.4（允许不同 / 必须相同两类） |
| P9_2-2 ~ P9_2-4 Top-K 快速路径 | ⚠️ **正确性通过、性能未达标（已回退生产接线）** | **P9_2-2 ~ P9_2-4 已实现（2026-09-26，Top-K 快速路径）**：

- **新 kernel**（`src/sampler/sampler_kernels.cu`）：`FastTopKSampleKernel<T>` —— **一行一个 warp**：
  ① 每线程在自己的 strided 切片上维护线程本地有序 top-k（`InsertCandidate`，值大优先、并列取小下标）；
  ② **k 轮 warp 归并**取全行 top-k（每轮在 32 个队首里归约最优）；
  ③ 采样数学与 `TopKSampleKernel` **逐字一致**（同稳定项、同 `Uniform01(seed, offset, row)`、同 `>=` 逆变换）。
  正确性依据：属于全局 top-k 的元素必然属于其所在切片的本地 top-k（在 k ≤ 每线程本地容量时成立）。
- **API（新增，不改旧签名）**：`LaunchTopKSamplerFast(args, stream)` + `constexpr kTopKFastMaxK = 64`。
  **契约**：调用方保证每行 `top_k ≤ 64`；越界行写哨兵 `-1`（不静默给错答案）。
  旧 `LaunchTopKSampler`（CUB 全排序）**保留不变**，作为 `k > 64` 的通用路径。
- **调用方**：`LLMRunner` 在 `options_top_k_ ≤ 64` 时走快速路径（host 侧已知，**不需要 D2H 同步**）。
  ⚠️ **这偏离了 §10.7 的"不改 `llm_runner.cpp`"**：快速路径的契约只能由"知道 k 的那一层"保证，
  而 runner 是唯一知道的地方。改动 5 行、不引入同步。
- **新用例**（`tests/test_sampler.cpp`）：`TopKFastMatchesLegacyTokens`（4 个形状，fast vs legacy **逐 token 相同**）、
  `TopKFastPoisonsRowsAboveContractLimit`（契约哨兵）。沙箱：显式跳过；全量 **218 条 / 0 失败**。
- **未做（该行是 P9_2-2~4 当时的快照）**：P9_2-5（Top-P，收益最大）、P9_2-6/7（FP16 与大 vocab 的
  **新增覆盖**用例）、P9_2-8（复测与回填）。→ **其中 P9_2-5 已于同日完成**，见下面两行与 §10.11.1。 |
| P9_2-5 Top-P 行内并行 | ✅ **正确性真机通过（2026-09-27）／性能已实测：按冻结基线 4/4 达标** | **落点**（详见 §10.11 的实施记录）：
`TopPParallelSampleKernel`（一行一个 256 线程 block）+ `FindFirstPrefixCrossing`
（`sampler/nucleus_cutoff.hpp`，host 可测的唯一实现）+ `LaunchTopPSamplerLegacy`（旧 kernel 留作对照）。
**正确性证据（沙箱）**：当时 6 条 host 用例 `NucleusCutoffTest.*` 全过（含 `>=` 边界、p=1 不截断、
阈值覆盖全行和无交叉时的退化分支）；全量 **228 条 / 0 失败**（沙箱）。
**正确性证据（真机）**：2026-09-27 全量 228 条 / 226 通过 / **1 红（= 按设计的 FP16 NaN 复现器）**，
S-13 / S-15 / S-16 / S-21 四条新 GPU 用例**全部通过**（真机上未被跳过——那次全量唯一跳过的是
`int8_crosscheck`）。
**性能（2026-09-27 真机，同 session，n=21 中位数）**：对比 §10.5 冻结的基线 **4/4 形状达标
（13.0× / 18.9× / 13.4× / 15.0×）**；同一把尺子换成"本次 session 的 legacy"则 3/4 达标，
`50257×1 = 9.99×`（差 0.08%）——**两个口径的读法与缺口归因见 §10.9.1** |
| P9_2-6 ~ P9_2-8 | ✅ **完成（2026-09-27）** | 2026-09-27 补下落着的两条判据：**S-12**（Top-K 的 FP16 分布，k=3/6 真的截断）与
**S-14**（Top-K @50257/128000、batch 1/2、k ∈ {1,8,64} 的集合成员关系）；另加 **S-23**
（当时 5 条 host：参考实现自证——分布良定义、与 nucleus 自洽、并列取小下标、`TopKSetByValue` 吸收并列组、
真机那条红的事故回归；**2026-09-27 去重后剩 3 条**）。
**第一次真机（2026-09-27）：233 条 / 2 红** —— 除按设计的 FP16 红外，S-14 **第一版判据是红的**，
机制为"参考用不稳定排序取前 k 个，而 128000 词表第 64 名有 4 个 token 精确并列"→
**kernel 无缺陷，改的是测试参考**（并列安全的 `TopKSetByValue`）；完整推导见 `TROUBLESHOOTING.md` #36。
**复跑（同日）：235 条 / 1 红**（唯一红 = 按设计的 FP16 NaN 复现器）→ **S-12 / S-14 真机通过**、
**P1.5-a 关闭**。沙箱 **235 条 / 0 失败**（S-12/S-14 显式跳过，S-23 实跑）。
S-17 取消（新方案不动 workspace，见测试计划 §9.2.2） |
| **P9_2-5b** Top-P 收尾段并行化 | ✅ **已实施、真机正确性通过；A/B 判定 = 效果无显著差异（保留代码）** | 计划见 §10.12；改动面 = `nucleus_cutoff.hpp`（两级拆成三级，`FindFirstPrefixCrossing` 变退化调用）+ `sampler_kernels.cu`（`kTopPSubChunks = 16`、第一趟顺带出子块和）+ 3 条 host 用例 S-24。串行重扫长度 197→13（50257）/ 500→32（128000）。**配对复测（§10.12.7）：验收判据 4/4 达标（12.32/19.61/13.56/15.43×），但配对 `top-p − top-k` = 58~131 µs，与改动前同量级 → 效果未判定**（跨协议不可直接比；初版"否证了旧归因"的说法已在 #38 更正为"未获支持"）。**代码保留**（真机正确性过了、无溢出、最坏串行尾 500→32；无任何一行显示它显著更慢），`LaunchTopPSamplerTwoLevel` 留作永久对照入口。**这条优化线关闭**（§10.12.8）：A/B 中位数 +4.8/+27.3/−330.5/−81.3 µs、p25/p75 全跨 0 → 效应低于本平台判别力；**P9_2-5c 也不做**（greedy 对照已否掉"访存模式"这条路） |

#### 10.9.1 P9_2-5 性能实测与两个口径的读法（2026-09-27 真机）

**原始数字**（`--gtest_filter='*SamplerPerf*'`，单位 ms，warmup 3 + 采样 21 次，报中位数）：

| 形状 | greedy | top-k(k=64) | **top-p（新）** | top-p legacy（本次 session） | 基线（§10.5，2026-09-26） |
|---|---|---|---|---|---|
| 50257 × 1 | 0.0460 | 0.5559 | **0.6264** | 6.2586 | 8.1469 |
| 50257 × 8 | 0.0528 | 0.5373 | **0.6040** | 12.4376 | 11.4256 |
| 128000 × 1 | 0.0755 | 1.1071 | **1.1897** | 17.3752 | 15.9150 |
| 128000 × 8 | 0.1513 | 1.2460 | **1.3745** | 21.2844 | 20.6862 |

**两个口径，都要说清楚**（`AGENTS.md` §7：阈值必须能回答"凭什么"）：

1. **对 §10.5 冻结的基线（8.1469 / 11.4256 / 15.9150 / 20.6862 ms）：4/4 达标**，
   加速比 **13.0× / 18.9× / 13.4× / 15.0×**——这正是 §10.5 第 4 条写的判据形式
   （"top-p ≤ 1/10 基线，即 ≤ 0.815 ms @ 50257×1，其余形状按同一比例"）。
2. **对同一次 session 里现测的 legacy：3/4 达标**，`50257×1 = 9.99193×`，**差 0.08%**。
   为什么会差：legacy 在 50257×1 上从基线的 8.1469 ms 掉到今天的 6.2586 ms（**−23.2%**），
   而另外三个形状只漂了 +2.9% ~ +9.2%。**这正是 G6 要"同一 session 内比较"的原因**——
   跨 session 的 ±25% 漂移（`docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` §5 记过同一现象）足以把 13× 变成 10×。
   **两个数字都成立、都留档**；阈值一个都没动。

**缺口归因（已定量，不是猜的）**：新路径里 `top-p − top-k` = 采样 kernel 的净成本
`70.4 / 66.7 / 82.6 / 128.5 µs`，即**排序占了新 top-p 的 89% ~ 93%**（与 §10.11 的预期一致：
保留排序是对的，它本来就便宜）。剩下的 70 µs 有一个可验证的特征：**batch 从 1 涨到 8（工作量 ×8）
时它几乎不涨**（70 → 67 µs；128000 上 83 → 129 µs，远低于线性）。**延迟受限、不是吞吐受限**，
来源就是 `TopPParallelSampleKernel` 里 thread 0 的两次"块内重扫"：每次最多 ~200 个**相互依赖**的
global load（Turing 上 L2 命中约 200~300 cycle）+ 串行 `__expf`，量级落在 60~130 µs。第一次重扫
（cutoff）与第二次（采样点）各自独立走一遍，所以是两次这样的链。

**可选优化（登记，本轮未做）**：把块内交叉也并行化（命中的那一块由整块 256 线程一次加载 +
warp 级归约/扫描定位精确下标），预期把那 67~129 µs 压到 ~5~10 µs → 50257×1 变成 ~0.57 ms
（同 session 口径 ~11×）。**但收益上限只有 11%**（其余 89% 是 CUB 排序，而 §10.11 已决定不动它），
所以除非另起一条"要不要动排序"的决定，否则不值得为它再走一轮真机。→ 记为 **P9_2-5b（可选）**。

**观测项（不作判据）**：S-15 打印的"新实现 vs legacy 逐 token 一致率"为
`50257×1: 2/3、2/3`、`50257×8: 17/24、12/24`、`128000×1: 0/3、0/3`。**词表越大越容易不一致，
这符合 `future_iterations_test_plan.md` §9.4 登记的机制**：两条实现对同一行的累计和只差舍入，但采样命中的是"第一个累计 ≥ target
的**下标**"；12.8 万词表上 nucleus 内相邻台阶只有 ~1e-5·total 的间距，而串行累加 11 万项的舍入误差
量级是 `ε·√N`（≈1e-5·total）——两者同量级，命中下标因此可能挪 1~2 位，token 标签就换了。
分布层面这只是一次 ~1e-5 的 CDF 扰动（S-21 在 3σ 下通过、S-15 在带 +1 元素边界的 nucleus 内通过）。
**n 只有 3 次抽样，不足以谈"差异率"**，所以它只打印、不判定。
**排查过程完整留痕**：`docs/TROUBLESHOOTING.md` **#35**（两把尺子的合法性论证、延迟/吞吐的
判别法、串行重扫的 cycle 估算与可选优化 P9_2-5b）。

---

</details>

### 10.12 [OI-SAMPLER-TOPP-TAIL] P9_2-5b：把 Top-P 的收尾段也并行化（2026-09-27 立项，同日实施）

#### 10.12.1 计划对账（AGENTS.md §5 第 0 步）


<details><summary>展开：10.12 [OI-SAMPLER-TOPP-TAIL] P9_2-5b：把 Top-P 的 全文</summary>

1. **有没有计划文档**：只有索引（`future_iterations.md` §11 的 P9_2-5b 行）与归因记录
   （`TROUBLESHOOTING.md` #35），**没有**做法 / 判据 / 破坏性清单 → 本节即为它的计划落点。
2. **逐条对照**：

   | §11 的说法 | 本节如何处理 |
   |---|---|
   | "thread 0 要做两次块内重扫（各最多 ~200 个相互依赖的 global load + 串行 `__expf`），实测 67~129 µs/行" | 一致（数据出处 = §10.9.1） |
   | "做法：命中的那一块由整块 256 线程一次加载 + warp 级归约/扫描定位下标" | **改成等价但更省同步的实现**：在"块和"之下再加一级"**子块和**"（每线程把自己那段切成 16 个子块），于是串行尾从 ≤200/≤500 个元素降到 **≤13/≤32 个**；搜索仍是 thread 0 做，但**全部经共享内存**（无 global 依赖链），不需要 block scan / warp 归约 |
   | "预期降到 ~5~10 µs" | **这是估算、没有出处**（假设 `__expf` + 累加链约 45 cycle/元素），**不作判据**；实测值回填 §10.9.1 |
3. **偏差怎么处理**：做法与 §11 的字面描述不同 → **先写本节再改代码**（本节即记录）。
   判据不变：正确性用例不打折、性能门槛仍是同 session **top-p ≥10×**（§10.5 第 4 条，**不许调低**）。
4. **反向查**：`future_iterations.md` §11、`PROGRESS.md` §6.6 与 §3.0g、本节 §10.9.1 都写着
   "本轮未做"——实施完成后必须逐处回填（否则下一个会话会以为它还没做）。

#### 10.12.2 做法（改动面）

1. `include/.../sampler/nucleus_cutoff.hpp`：把"定位交叉点"拆成两个可 host 测的积木
   ——`FindCrossingSegment`（在分段和上找第一个累计 ≥ 阈值的段）与 `ScanSegmentForCrossing`
   （在一段内逐元素定位）；再加 `FindCrossingByLevels`（块和 → 子块和 → 元素）。
   **`FindFirstPrefixCrossing` 改为它的退化调用**（`sub_sums == nullptr`），语义与数值行为
   与原来**逐位一致** → S-22 的 host 用例一个字都不用改（当时 6 条；2026-09-27 按"重复即删"清到 4 条）。
2. `src/sampler/sampler_kernels.cu`：`TopPParallelSampleKernel` 的第一趟额外把每线程那段的
   **16 个子块和**写进 shared（共享内存代价 256×16×4B = 16 KB）；两次交叉定位改走
   `FindCrossingByLevels`。
3. 新增 host 用例（S-24）：三级定位与"整行逐元素串行扫描"在通用数据上同比；子块级不命中时
   退化为整块（不能把整块的和算两遍）；并列/边界行为与两级版一致。
4. **不改**：`sampler_common.hpp` 的 API、`llm_runner.cpp`、`LaunchTopPSampler` 的签名与
   workspace 需求（结论：`TopPSamplerWorkspaceBytes` 不变）。

#### 10.12.3 判据（每条都要能回答"凭什么"）

1. **正确性不打折**：S-13 / S-15 / S-16 / S-21 真机通过（判据不动）；S-22/S-23/S-24 沙箱通过（去重后：4 条 + 3 条 + 3 条）。
   `future_iterations_test_plan.md` §9.4 登记的"允许差异"类别不变（仍只有浮点累加顺序；这次把"分块 + 块间串行合并"扩成
   "分块 + 子块 + 元素"三级，属同一类）。
2. **性能**：同 session 同 harness，`top-p legacy/parallel ≥ 10×`（不可调低），且
   **`top-p − top-k`（= 采样 kernel 净成本）必须明显下降**，实测值连同"哪次运行"回填 §10.9.1。
3. **不回归工程约定**：`Launch*` 内零分配；shared 用量仍 < 48 KB（17 KB）；`enqueue` 里无 host 同步。

#### 10.12.4 若仍不达标的下一步（先写下来，免得临时改口）

`top-p − top-k` 目前 67~129 µs 里，**第一趟（每线程连续读自己那段）与两次重扫各占多少，尚未单独测过**。
如果这次改完仍明显高于 ~10 µs，**下一个假设是"第一趟的访存模式"**（一个 warp 的每条 load 落在
32 条不同 cache line 上）；那时的动作是把它改成 round-strided（每轮 256 个连续元素）并配一次
块内扫描，**且改前先按 #35 的做法量一次**——不许直接猜。

#### 10.12.5 实施结果（2026-09-27，**性能待真机**）

| 项 | 内容 |
|---|---|
| `nucleus_cutoff.hpp` | 拆出 `FindCrossingSegment`（分段和上找交叉段）与 `ScanSegmentForCrossing`（段内逐元素定位）；新增 `FindCrossingByLevels`（块→子块→元素）；`FindFirstPrefixCrossing` 改成它的退化调用 → **S-22 的 host 用例一个字未改、全部仍然通过**（当时 6 条，去重后剩 4 条）——即两级行为逐位不变 |
| `sampler_kernels.cu` | `kTopPSubChunks = 16`；第一趟顺带产出子块和（shared 16 KB）；两次交叉定位改走三级 |
| `tests/test_sampler.cpp` | 新增 3 条 host 用例（S-24）：三级与整行逐元素串行扫描同比；子块级不命中时退化**且不重复计入块和**；第二次调用尊重 `size = cutoff` |

**沙箱证据**：全量 **234 条 / 0 失败**；S-22/S-23/S-24 共 10 条 host 用例实跑通过（另按"重复即删"清理了 4 条：2 条二级版与三级版同断言、2 条参考自证里的无判别力/抽象重复项）。

**独立自检**（`/tmp` 的临时 host 模拟，不入库，与内核逐句同构）：小词表下 cutoff 与解析口径
`2/4/8` 逐一相等、词频分布与"截断 + 重新归一化"解析分布吻合；均匀 128000 词表 + p=1 时
**cutoff = 128000**（不得提前截断）、被选下标铺满整行；`Σ子块和 == Σ逐元素`（无漏算/重复计）。

**真机判据（未跑，别提前写绿）**：同一 harness 同 session `top-p legacy/parallel ≥ 10×`，
且 `top-p − top-k` 明显下降；S-13 / S-15 / S-16 / S-21 正确性用例不打折。

#### 10.12.6 第一次真机复跑（2026-09-27）：**判据达标，但效果判不了 → 改测量协议**

| 形状 | top-k | top-p（新） | top-p legacy | 同 session 加速比 | top-p − top-k |
|---|---|---|---|---|---|
| 50257 × 1 | 0.565 | **0.516** | 6.670 | **12.93×** | **−49 µs** |
| 50257 × 8 | 0.558 | **0.632** | 12.515 | 19.79× | +75 µs |
| 128000 × 1 | 1.158 | **1.264** | 19.850 | 15.70× | +106 µs |
| 128000 × 8 | 1.278 | **1.421** | 24.043 | 16.92× | +144 µs |

1. **验收判据满足**：同 session 4/4 ≥10×（上次差 0.08% 的 50257×1 这次 12.93×）。
2. **改动效果无法判定**：要看的 `top-p − top-k` 出现负值，且另三个形状略升；而同一次运行里
   与实现无关的量也在漂（greedy @50257×1 0.046 → 0.094 ms，top-p 的 max/median = 2.3×）。
   信号只占总耗时的 5%~13%，**旧 harness 是分段测的，块间漂移直接进了分子分母**。
   → 既不能写"有效"，也不能写"无效"；完整推导见 `TROUBLESHOOTING.md` **#37**。
3. **动作**：`SamplerPerf.ThroughputByShape` 改成**配对测量**（每轮 4 个变体挨着各测一次，
   逐轮算比值/差值再取中位数；轮数 21 → 31），同时保留分段口径的数字供与历史基线对齐。
   沙箱 **234 条 / 0 失败**。
4. **下一步 = 配对口径复测**：判据不变（`top-p legacy/parallel ≥ 10×`，且配对
   `(top-p − top-k)` 明显低于 70~129 µs）。**若配对复测显示没有下降**，再按 §10.12.4 的
   下一个假设（第一趟访存模式）走，且**改前先量**。

#### 10.12.7 配对复测结果（2026-09-27）：**判据通过，收益未测得 → 否证 §10.12.1 的归因**

| 形状 | 配对 `legacy/parallel` | 配对 `(top-p − top-k)` | 同次分段 `legacy/parallel` | 同次分段 top-p |
|---|---|---|---|---|
| 50257 × 1 | **12.32×** | **58.3 µs** | 13.86× | 0.540 ms |
| 50257 × 8 | **19.61×** | **71.3 µs** | 19.84× | 0.618 ms |
| 128000 × 1 | **13.56×** | **88.4 µs** | 13.59× | 1.212 ms |
| 128000 × 8 | **15.43×** | **131.0 µs** | 15.48× | 1.393 ms |

1. **验收判据通过**：配对口径（更严）4/4 ≥10×。
2. **P9_2-5b 的效果：未判定**。配对差值 58~131 µs 与改动前的分段口径 70~129 µs 同量级，
   但**跨协议不能直接比**（#38），而改动前没有配对数据。**更正（同日重新评估）**：初版本节写的
   "因此主项不是重扫、而是第一趟"**是过度解读**——它依赖"跨协议 delta 可直接相减"与"改动只影响
   重扫"两个未检验的前提。正确表述：**#35 的归因"未获支持"，但也没被否证**。
3. **两条与 GPU 无关的硬证据**（`ptxas -v` + 同次 greedy 对照）：
   `TopPParallelSampleKernel` = 51 寄存器 / **0 溢出** / 17408 B shared / 1 barrier → 无明显病态；
   greedy（同一行、合并访存、几乎无计算）= 64 / 70 / 149 / 154 µs，**与整个 delta 带宽同量级**
   → delta 里很大一块是"读一遍行"的硬成本，可优化部分的上限本来就小。
4. **代码保留**（真机正确性已过、无溢出，并把"重扫"从瓶颈名单划掉），**结论写"效果未判定"**。
5. **下一项 = 先把测量做到能分辨 10~30 µs，再谈改 kernel**：① 配对轮内顺序改 **ABBA / 每轮随机**
   （现在是固定顺序，顺序偏置会进 `(top-p − top-k)`）；② 报配对差值的**中位数 + 分位数**（极差已达
   ±800 µs，被环境脉冲污染）；③ 加**只跑第一趟**的诊断入口做分段计时——**唯一能给归因定性的实验**。
   **在此之前不动第一趟**；且按 greedy 对照，P9_2-5c 的净收益很可能**远小于 11%**。
6. **判定性 A/B（已实现，待一次真机）**：斜率复测显示固定开销其实不大（greedy 单发 41 µs ≈ 净成本
   44 µs），净差值仍是 61~107 µs、与 greedy 净成本同量级。为彻底消掉"跨协议"的不确定性，
   `TopPParallelSampleKernel` 加了模板参数 `kSubChunked`：`true` = 生产版（子块级），
   `false` = P9_2-5b 之前的形态（两级、**1 KB shared / 48 寄存器**，与改动前占用一致；生产版
   51 寄存器 / 17.4 KB）。新增 `LaunchTopPSamplerTwoLevel`（**只作 A/B**），harness 同轮交替测两版、
   报 `子块版 − 两级版` 的中位数与 p25/p75。**一次真机即可定论**：
   负值且显著 → 子块级有效（保留）；≈0 → 无可测收益（**简化回两级**，撤掉这份复杂度）。

#### 10.12.8 A/B 结果与收口（2026-09-27）：**效果无显著差异 → 保留代码、关闭这条优化线**

| 形状 | A/B 中位数（子块 − 两级） | p25 / p75 | 生产版净 | 两级版净 | greedy 净 |
|---|---|---|---|---|---|
| 50257 × 1 | **+4.8 µs** | −6.4 / +9.7 | 0.4992 ms | 0.4925 ms | 0.0350 ms |
| 50257 × 8 | **+27.3 µs** | −66.2 / +128.5 | 0.6139 | 0.5954 | 0.0515 |
| 128000 × 1 | **−330.5 µs** | −558.6 / +125.3 | 1.2248 | 1.5353 | 0.0979 |
| 128000 × 8 | **−81.3 µs** | −485.6 / +390.4 | 1.4709 | 1.6927 | 0.1551 |

1. **判定 = 效果无显著差异**：方向不一致、**p25/p75 全部跨 0**；与理论预期（27 / 68 µs）只对上一半。
   **预期效应低于本轮分散度（±400~600 µs）** → 本平台对这个量级没有判别力。
2. **顺带钉死的两件事**：① `top-p 净 − top-k 净` @50257×1 = **60.1 µs（p25=56.4 / p75=64.0，很紧）**，
   而 greedy 净 = 35.0 µs → 采样内核净开销仅"裸读一遍行"的 ~1.7 倍，**优化空间见底**；
   ② **P9_2-5c 不做**（greedy 本来就完全合并访存，也要 35~155 µs）。
3. **处置**：保留子块版代码 + `LaunchTopPSamplerTwoLevel` 永久对照入口；配对（净）`legacy/parallel`
   = **12.63 / 22.43 / 15.73 / 17.57×**（验收判据 4/4 达标）。回退到两级是**独立的代码改动**，
   需单独确认。


</details>

### 10.10 P9_2-2~4 结果：正确性达标、**性能不达标**（2026-09-26 真机）

**正确性**：`TopKFastMatchesLegacyTokens`（fast vs legacy **逐 token 相同**，4 形状）与
`TopKFastPoisonsRowsAboveContractLimit` 均真机通过 → 语义等价成立。

<details><summary>展开：10.10 P9_2-2~4 结果：正确性达标、**性能不达标**（2026-09-26 真 全文</summary>


**性能（同一 harness，n=21 中位数）——未达 §10.5 第 4 条的 ≥5×：**

| 形状 | legacy top-k | fast top-k | legacy/fast |
|---|---|---|---|
| 50257 × 1 | 0.567 ms | **5.243 ms** | 0.108× |
| 50257 × 8 | 0.599 | 4.571 | 0.131× |
| 128000 × 1 | 1.209 | 7.919 | 0.153× |
| 128000 × 8 | 1.270 | 8.126 | 0.156× |

即**比旧路径慢 6~9 倍**。按 §7：**不调阈值**，先查原因。

**根因（三条，按影响排序）**：

1. **bank conflict**：每线程的候选段起点是 `lane * kTopKFastMaxK` 个 float = **256 B 步长**，
   32 个 lane 落在**同一个 bank** 上 → 每次插入的 shared 读写全部串行化。
2. **occupancy 太低**：一行只有 32 个线程（1 个 warp / block），batch=1 时整个 GPU 只有 1 个 warp 在跑，
   24 个 SM 基本闲着；而 legacy 的 CUB 分段排序是全 GPU、256 线程/块的 grid-stride。
3. **串行插入链**：每元素一次"比较 + 最多 64 次搬移"的依赖链，且每个线程要扫 vocab/32 ≈ 1570 个元素。

**已采取的止损**：`LLMRunner` **已切回旧路径**（注释里写明原因与出处）——不把 9× 慢的实现留在产品路径上；
快速路径保留在库里供后续重做与对拍使用。

**下一步（重做方向，尚未实施）**：

- 候选放**寄存器**（每线程 C=8 的有序小表），**不再用 shared 做插入** → 消除 bank conflict；
- **一行多 warp**（256 线程 = 8 warp/行），每 warp 归并出局部 top-C 后写 shared pool（16 KB，池内 2048 个候选）
  → occupancy 提升一个数量级；
- **k ≤ 8**：并集直接含全局 top-k（可证）→ 走池内 bitonic 排序 + 采样；
- **8 < k ≤ 64**：用池内第 k 名当阈值 θ（可证 θ ≤ 真 g_k），**二次扫描**收集 `≥ θ` 的元素；
  若数量超容量 → 抬 θ 重试（或该行回退 legacy）。这一步是精确 top-64 的关键，必须实测覆盖率。
- 重做后再跑同一 harness，用**同一把尺子**（≥5×）判定；仍不达标就继续查，不调阈值。


</details>

### 10.11 决策（2026-09-26）：先做 Top-P，且**改用"保留排序、并行化行内数学"的方案**

**作者决定**：跳过 Top-K 重做、直接做 **P9_2-5（Top-P）**——它是基线里的绝对大头（6.6~22 ms，
是 top-k 的 10 倍量级）。

<details><summary>展开：10.11 决策（2026-09-26）：先做 Top-P，且**改用"保留排序、并行化行内 全文</summary>


**方案改了（比 §10.10 的重做方向低一个数量级的风险）**，依据是基线分解：

- legacy Top-P 的 8~21 ms 里，**CUB 全行排序只占约 0.57 ms**（= 同形状 top-k 的耗时）；
  真正的大头是 `TopPSampleKernel` 里**三趟串行全行扫描**（求 total / 求 cutoff / 前缀内重新归一化），
  每趟每元素都算 `__expf`，而整行只由**一个线程**做。
- 所以**不必先解决"精确 top-M 选择 + 阈值回退"**：保留现有排序（它便宜且已被验证），
  把行内数学换成**块内并行**即可拿到主要收益：

| 步骤 | legacy | 新方案（每行一个 256 线程 block） |
|---|---|---|
| 排序 | CUB 分段排序 | **保留不变** |
| total = Σexp | 单线程串行全行 | 分块归约（256 线程，各扫 V/256）+ 块扫描求和 |
| cutoff（首个 cum ≥ p） | 单线程串行 | 各线程分块内求交叉点 → 块内取最小下标（一次扫描同时给出 `kept_total`） |
| 前缀内重新归一化 | **再一趟全行** | **不需要**（`kept_total` 就是交叉点的累计和） |
| 逆变换采样 | 单线程串行 | 同"交叉点"做法，对 `target` 再做一次分块交叉 |

- **语义**：cutoff 规则（首个累计 ≥ p）、稳定项取 top-1、`__expf`、`Uniform01(seed, offset, row)`、
  前缀内重新归一化全部保持；**唯一差异是浮点求和顺序**（分块 vs 串行）→ 极端并列处 cutoff 可能差一格，
  这条要写进测试计划的"语义等价范围"（比 Top-K 那条宽一档）。
- **不再需要 nucleus 回退**：因为排序还在，nucleus 一定在排好序的行里 → 没有"候选容量"这个概念了。
- **验收**：同一 harness 同形状，**top-p ≤ 1/10 基线**（即 ≥10×，见 §10.5 第 4 条）；
  正确性用现有 S-7（p→0 取 argmax）、S-9（p=1 不过截断）、分布一致性（S-13 的 FP32 部分）锁住。

**下一批任务（P9_2-5）**：① 写 `TopPParallelSampleKernel<T>`（每行一个 block，含块归约 + 两次分块交叉）；
② 接进 `LaunchTopPSampler`（**替换**采样 kernel，排序部分不动）；③ 保留 legacy 采样 kernel 作为对照
（**更正：不是"回退"**——新方案不引入候选容量，没有需要回退的行）；④ 跑同一 harness 比 top-p；
⑤ 补 S-13（FP16）与 S-15/S-16（大 vocab / p=1）。

#### 10.11.1 实施记录（2026-09-26，与上面的方案有两处有意偏离）

**已落地**：

| 落点 | 内容 |
|---|---|
| `src/sampler/sampler_kernels.cu` | 新增 `TopPParallelSampleKernel`（一行一个 256 线程 block）；`LaunchTopPSampler` 改走它；新增 `LaunchTopPSamplerLegacy` 保留旧 kernel 作对照；旧 kernel 的注释标明"已被取代、仅作对照" |
| `include/.../sampler/nucleus_cutoff.hpp`（新） | `FindFirstPrefixCrossing`：第一级在分块和上定位交叉块，第二级在块内逐元素定位精确下标；`__host__ __device__`，**host 可测** |
| `tests/test_sampler.cpp` | 6 条 host 用例 `NucleusCutoffTest.*`；S-15（大 vocab nucleus）、S-16（p=1 大 nucleus 不过截断）、S-21（分布级主判据，两条实现各跑一遍）；`RunTopP`/`CollectTokenCounts` 加 `legacy` 开关；性能 harness 增加 `top-p legacy` 一行 |
| `tests/test_fp16_paths.cpp` | S-13：Top-P 的 FP16 分布用例（FP32/FP16 各自对解析分布，且互相 3σ 内） |
| `tests/sampler_test_support.hpp`（新） | Top-P 解析参考的唯一实现（排序顺序 / nucleus / 截断后归一化分布），被上面两个测试文件共用 |

**偏离 1：两次"分块交叉"改成"分块和 → thread 0 串行合并 → 块内扫描"**。
方案原文是"各线程分块内求交叉点 → 块内取最小下标"，需要一次 block scan 求块前缀；实现改成
thread 0 在 **256 个块和**上串行定位（+ 一个块内的逐元素扫描）。**为什么**：串行尾从 O(V) 降到
O(块数 + 一个块)，量级差 100 倍以上，而省掉了一整套 block scan 的同步与代码量。
**代价**：两级累加的加法结合序不同，恰好落在 1 ulp 边界时块内可能扫不到交叉点——这条路径
**显式退化为"该块末尾"**（有注释、有 host 用例 `ThreeLevelDegradesWithoutDoubleCountingTheChunk` 锁住；
二级版的同场景用例在 2026-09-27 去重时删除），
确定性与边界安全性都不受影响。

**偏离 2：`total` 用"块和按块序串行相加"求，不用树形归约**（`BlockReduceSum` 因此没被用上）。
**为什么**：阈值 `p * total` 与所有累计和必须走同一口径，否则两者各带一份舍入，边界处会随机差一格；
块和是各线程串行累加的结果，块级合并也按块序串行做——于是阈值、cutoff、采样点三者共享
**同一套部分和**（块内逐元素、块间按块序；与逐元素全行扫描只差跨块那一层的结合方式）。

**语义差异（唯一一处，已登记进测试计划 §9.4）**：legacy 逐元素累加 `exp/total` 再与 `p` 比，
新实现累加 `exp` 再与 `p * Σexp` 比——先除后加 vs 先加后除。随机数消费、`>=` 比较、
稳定项取 top-1、前缀内重新归一化都不变。

**验证状态**：

- **沙箱**：228 条 / 0 失败（6 条 host 用例实跑通过；4 条 GPU 用例显式跳过）。
- **真机正确性：已通过（2026-09-27 全量）**——228 条 / 226 通过 / 1 红 / 1 跳过，唯一的红是
  按设计的 GPT-2 FP16 NaN 复现器、唯一的跳过是 `int8_crosscheck`；S-13 / S-15 / S-16 / S-21
  四条新 GPU 用例全过（真机上**没有**被跳过）。
- **真机性能：已实测（2026-09-27）**。按 §10.5 冻结的基线口径 **4/4 形状达标**
  （13.0× / 18.9× / 13.4× / 15.0×）；同 session 口径 `50257×1 = 9.99×`（差 0.08%，
  原因 = legacy 在 50257×1 上比基线快 23.2%，属跨 session 漂移）。**阈值未动**，
  缺口归因与可选优化见 §10.9.1。

---

</details>

## 11. [OI-PERF-PROFILE-PLAN] decode 端到端性能画像 + G6 可复现测量方法（开发计划）

> **本文档是它当年的正式计划落点**（沿用 §10 开头记的规则：后续迭代的每个事项都并入
> `future_iterations*_plan.md` 家族，不新建 per-item 文件）。
> **该规则已于 2026-10-01 被取代**（新落点 = `docs/dev/<feature>/`，见 `AGENTS.md` §5）；
> 本节属旧口径存量，不回改。
> 配套测试计划见 `future_iterations_test_plan.md` §10。
> 条目事实来源：`docs/future_iterations.md` **§6.3**（Nsight 一键 profile target）与 **§11 的 G6**
> （性能无可复现测量方法）；同时收口 `future_iterations.md` §9.2 留下的"sampler 在整步 decode 里占多少"。

### 11.1 计划对账（AGENTS.md §5 第 0 步）

#### 11.1.1 有没有计划文档


<details><summary>展开：11.1 计划对账（AGENTS.md §5 第 0 步） 全文</summary>

**有（就是本节）。** `future_iterations.md` §6.3 与 §11 的 G6 是条目的**唯一事实来源**
（是什么 / 为什么 / 触发条件）；本节只回答"改哪些文件、分几步、每步怎么自检、哪一步要你批"。

#### 11.1.2 逐条对照：任务 / 接口 / 验收 vs 现状

| 本次要做的 | 文档原文 | 现状（已核实） | 是否一致 |
|---|---|---|---|
| 加 profile target | `future_iterations.md` §6.3："CMake 增加 `profile_gpt2`、`profile_resnet18` 自定义 target；支持 `nsys profile` 与 `ncu` 导出 `.ncu-rep`" | **立项时**（2026-09-27 上午）核实：`rg "profile\|nsys\|ncu"` 在两个 CMakeLists 里**零命中**（现已落地，见 §11.5.1） | 一致；本节补文件级改动面与步序 |
| 建可复现测量方法 | §11 G6："固定机器状态 + 同一 session 内 ≥3 次构建 / ≥20 次推理，报中位数与极差" | 协议未落成文档、无脚本、无 target | **有偏差**：G6 那一行写在 `future_iterations.md` §9.2 之前，**弱于** §9.2 沉淀的协议（见 11.1.3） |
| 回答"sampler 在整步 decode 占比" | §0.1 / §0.3 第 1 项："`future_iterations.md` §9.2 的 sampler 侧测量已完成，缺的是它在整步 decode 里的占比" | `PROGRESS.md` §3.0d 的性能数字只有 `CVRunner` benchmark（`mean≈8.4 ms`），**没有任何 decode 画像** | 一致（缺口真实存在） |

#### 11.1.3 偏差怎么处理：**先改文档，再写代码**

G6 原文只要求"≥3 次构建 / ≥20 次推理 + 中位数与极差"。`future_iterations.md` §9.2 的四次真机实测证明这还不够：

- **分段 / 单点测量**对"差百分之几"没有判别力（`TROUBLESHOOTING.md` **#37**）；
- **跨协议 / 跨 session 的差值不能直接比**（#38：同一变体跨 session 漂移 −23.2%）；
- 判"改动有没有用"必须**同二进制、同轮交替（ABBA）**，并用**斜率** `(T4−T1)/3` 扣掉每窗口固定开销；
- 本平台这类测量的**判别下限约 ±400~600 µs**。

→ 本轮把 **11.3** 定为 G6 的落地口径，并把 `future_iterations.md` §11 的 G6 行**改指本节**
（只改"做法"，触发条件与判据不变）。"先改文档"的含义是：**本节写完、且那条引用改完之后才动 CMake / 脚本**。

#### 11.1.4 反向查：文档与现状矛盾之处（当场记）

- **`ncu` 在 WSL2 上的导出策略，现状比 `future_iterations.md` §6.3 的原文更明确。** §6.3 只写"支持 `ncu` 导出 `.ncu-rep`"，
  而 `AGENTS.md` §1 已定：WSL2 上 `ncu` 可能因 performance counter 权限 / 驱动拿不到，
  **无头导出（`-o`）再拷到 Windows 宿主机 GUI 看是主路径**。本节按 `AGENTS.md` §1 写，不按 §6.3 的字面。
- **`detailed_profiling` 会改变引擎指纹 → 触发重建。** `builder.hpp:83` 的 `detailed_profiling`
  控制 `IEngineInspector` 逐层信息，而它进指纹（`engine_cache.hpp:34`）。
  → profile 轮次要读逐层信息时，**必须单独一份 engine 路径**，别和生产路径共用
  （与 `export_diagnostics` 是同一类"改 I/O = 改契约"的纪律，见 `PROGRESS.md` §2.15）。

</details>

### 11.2 目标与范围

**目标（一句话）**：建立"同一 session 内可复现"的性能测量方法（G6）+ 一键 profile target（`future_iterations.md` §6.3），
并用它把 decode 阶段的时间**分解到可归因的粒度**，回答三个待决问题：

<details><summary>展开：11.2 目标与范围 全文</summary>


1. **sampler 在整步 decode 里占多少**（补 `future_iterations.md` §9.2 的遗留）；
2. **attention kernel / KV 写入 / 其余 TRT 算子**各占多少（是 §2.2"attention 分块值不值得"的前置）；
3. `LLMRunner::Generate` 一次调用里有多少 **host 侧分配**（是 §2.1"显存池值不值得动"的前置）。

**范围内**：

- `profile_gpt2` / `profile_resnet18` 两个 CMake custom target（`nsys` + `ncu` 无头导出）；
- 一份 G6 测量协议（11.3）与一个可重复执行的 decode 计时入口；
- **首次采集**：GPT-2 FP32、batch = 1 的 decode 分解；ONNX vs 原生 prefill 的可复现对照（G6 的正例）。

**范围外（明确不做）**：

- 不实现 attention / MLP 优化（§2.2）、不实现子图替换（§10.2）——本轮只建"尺子"；
- 不做 continuous batching / 多请求（`future_iterations.md` §2.3）；
- 不做 `ncu` 的深层 kernel 调优，只保证"能导出、能打开、能读出 kernel 名与耗时"；
- **不 profile FP16 路径**——GPT-2 FP16 端到端产 NaN（`PROGRESS.md` §5.11），尺子必须架在可用路径上。

</details>

### 11.3 测量协议（G6 的落地口径，**本节是判据出处**）

#### A. 机器状态固定（能固定的部分）


<details><summary>展开：11.3 测量协议（G6 的落地口径，**本节是判据出处**） 全文</summary>

- 一组对照**在同一 session 内**跑完；记录
  `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv` 的**前后**读数；
- WSL2 通常**不能**锁时钟（`nvidia-smi -lgc` 常失败）→ 不假装能锁，改用"同轮交替 + 报极差"抵消漂移；
- 关掉其它占 GPU 的进程（浏览器硬件加速、其它推理），并记进报告。

#### B. 两类问题、两种测法（不要混用）

| 问题类型 | 例子 | 测法 | 报什么 |
|---|---|---|---|
| **同二进制内的 A/B**（"改动 X 有没有用"） | `LaunchTopPSamplerTwoLevel` vs 生产版 | 两个变体**编进同一个二进制**，同一轮里正反交替（ABBA）；每个变体每轮测"发射 1 次"与"发射 4 次"两个窗口 | `(T4−T1)/3` = **净成本（斜率）** 的中位数 + p25/p75；现成对照见 `future_iterations.md` §9.2 / `TROUBLESHOOTING.md` #38 |
| **跨构建对照**（"两条路谁快"） | ONNX vs 原生 prefill | 同一 session 内**各自 ≥3 次独立构建**，每次构建 **≥20 次推理** | 每次构建的中位数 + 全部中位数的**极差**；极差 > 中位数之差 → 判"**未定**" |

#### C. 判别下限（写死，避免事后改口）

- 本平台这类"整步 decode / 采样"测量的**判别下限约 ±400~600 µs**（`TROUBLESHOOTING.md` #37 / #38 实测）；
- 观测到的效应若低于该下限 → 结论写"**无显著差异**"，**不许**写成"更快 / 更慢 X%"；
- 若"隔离计时"与"nsys 时间线"两口径方向相反 → 以**隔离计时**为准（nsys 有采样开销），并在报告里标注两口径。

#### D. 报告格式（缺一项就不算完成）

每次测完必须写出：① 引擎 / 形状 / 精度（batch、seq、vocab、FP32/FP16）；
② 构建态（`Engine cache hit` 还是 `stale` + 重建）；③ n 与 warmup；
④ 中位数 + p25/p75（或 min/max）；⑤ 温度 / 时钟前后值；⑥ 命令原文。

#### E. 引擎缓存纪律

profile 前先跑一遍让缓存热起来（确认日志是 `Engine cache hit`）；若要 `detailed_profiling`，
**新开一条 engine 路径**，不要覆盖生产引擎（它进指纹，会触发重建）。

</details>

### 11.4 [OI-PERF-PROFILE-DECOMPOSITION] 分解口径（怎么把 kernel 时间归因，避免"看起来像真故障"的数字）

TRT 内部 kernel 名（`genericNode_*` / tactic 名）**不携带语义**，硬贴"attention / MLP"标签就是
`PROGRESS.md` §2.14 C 警告过的"诊断比错对象更危险"。所以本轮只做**能自证的三层**：

<details><summary>展开：11.4 [OI-PERF-PROFILE-DECOMPOSITION] 分解口径（怎么把  全文</summary>


| 层 | 成员 | 怎么识别（自证方式） |
|---|---|---|
| **① 我们的 kernel** | `PagedAttentionDecodeKernel`（attention）、`WriteKVKernel`（KV 写入）、`AdvanceContextLensKernel`、`FillPositionIdsKernel`、采样器 kernel（`GreedyKernel` / `TopPParallelSampleKernel` / `TopKSampleKernel` / `PrepareSortInputKernel`） | 全部来自 `rg "__global__ void" mini_trt_llm/src/`，每个都对应一行我们自己的代码 |
| **② CUB** | `cub::DeviceSegmentedSortKernel*` / `DeviceSegmentedRadixSort*` 等 | 符号名带 `cub::`；这是 Top-K / Top-P 保留的排序 |
| **③ TRT 内部** | 其余全部（matmul / LN / GELU / embedding gather / softmax …） | 只按**耗时排序**列出前 N 条，**不**做语义归因；要语义就读 `IEngineInspector` 逐层信息（需 `detailed_profiling`，见 11.1.4） |

**回答三个待决问题的方法**：

- "sampler 占比" =（①里采样器 kernel 时间之和）/（decode 一步的总 GPU 时间）；
- "attention / KV 占比" = ①里对应 kernel 之和 / 总时间；
- "分配开销" = 在 `Generate` 入口 / 出口各插一个 host 侧计时点，统计 `cudaMalloc` / `cudaFree` 的
  **次数与总耗时**（**不**从 nsys 里猜）。

#### 11.4.1 逐 kernel 分解不可得时的替代：**上下文长度扫描**（2026-09-27 新增）

**背景**：① ~ ③ 的表依赖 profiler 的 kernel 时间线；而本例已证实在本机拿不到
（nsys 无 GPU 活动、ncu 报错、加 sudo 与显式 `--trace=cuda` 都无效，见 `TROUBLESHOOTING.md` #41）。
**§2.2 要的那个数（attention 占多少）换一个只靠计时的办法拿：**

- **原理**：attention 的开销随**已缓存的位置数**增长；而每步的 matmul / LayerNorm / GELU /
  KV 写入 / 位置填充 / 采样都与上下文无关。于是
  `每步耗时(长上下文) − 每步耗时(短上下文)` 的差值 ≈ **attention 的边际成本**，
  再除以上下文差就是"每 1000 个位置涨多少毫秒"。
- **为什么可信**：`LLMRunner::Generate` 每次调用开头都 `FreeSequence` + 重新 `AllocateSequence`
  （`src/core/llm_runner.cpp`），所以**每次调用的上下文都是从 prompt 长度重新开始**的，
  prompt 长度直接决定上下文区间——不需要额外改 runner。
- **实现要点**：prompt ∈ {4, 256, 960}；每个长度按 `(T(32)−T(1))/31` 取每步耗时
  （与 `future_iterations.md` §9.2 同一套斜率口径）；**只打印，不设阈值**。
  prefill 引擎需要 `max_prefill_seq_len ≥ 992` → **单独一条引擎路径**（不动主用例缓存的两个）；
  decode 引擎与上下文无关，可复用主用例那份（命中缓存）。
- **边界（必须一起读）**：
  1. 它给的是 **attention 随上下文增长的部分**，不是"attention 的绝对时间"；两者在
     短上下文下差异不大，但要写清楚；
  2. prefill 的 `opt` 形状对 kernel 选择有影响，三个 prompt 用**同一个引擎**，
     所以横向比较自洽，但绝对值不能与其它用例的引擎直接比；
  3. 仍然只在**同一 session** 内可比（decode 步跨 session 已见 ±27%）。

</details>

### 11.5 任务分解（P6_3-0 ~ P6_3-7）


<details><summary>展开：11.5 任务分解（P6_3-0 ~ P6_3-7） 全文</summary>

| 编号 | 任务 | 层 | 自检点 |
|---|---|---|---|

| **P6_3-0** | **仪器先行**：确认 WSL2 上 `nsys` / `ncu` 可用，且无头导出能产出可读报告（拿一个平凡进程试，别拿 GPT-2 试） | G | 报告文件存在、能在宿主机打开；`ncu` 若报 counter 权限则记进报告（按 `AGENTS.md` §1 走无头导出） |
| **P6_3-1** | 协议定稿：把 11.3 / 11.4 写进本节 + 测试计划 §10 | —— | 判据旁能回答"凭什么"（出处 = #37 / #38） |
| **P6_3-2** | CMake custom target：`profile_gpt2` / `profile_resnet18`（nsys + ncu 无头导出到 `/tmp/mini_trt_llm_profiles/`） | G | target 能跑、输出名带时间戳；**不进默认构建** |
| **P6_3-3** | decode 计时入口：复用 `SamplerPerf` 的 harness 形态（warmup 3 + n ≥ 15 + 中位数 / p25 / p75）做端到端 decode 计时（prompt = 文本用例的固定提示，生成 N = 32 token） | G | 与 profile target 用同一个二进制 / 过滤器；"循环内零 H2D/D2H"这条不动 |
| **P6_3-4** | 首次 decode 分解（nsys）：GPT-2 FP32 / batch = 1，输出 11.4 的三层分解 + TRT 前 N 条 kernel | G | 三层时间之和 vs 总时间的偏差写出；两口径都给 |
| **P6_3-5** | §2.1 的分配开销测量（与 P6_3-4 **同一次真机**顺带）：统计 `Generate` 的分配次数与耗时 | G | 为"要不要做显存池"供数；**本轮不改实现** |
| **P6_3-6** | G6 正例：ONNX vs 原生 prefill 的可复现对照（≥3 次构建 / ≥20 次推理，报中位数与极差），回答 §10.2 的前提 | G | 极差 > 差值 → 写"未定"（这正是 G6 存在的目的） |
| **P6_3-7** | 回填：`future_iterations.md` §6.3 / §11、`PROGRESS.md` §3、`TROUBLESHOOTING.md`（若有坑） | —— | 只写"状态 + 实测值 + 出处" |

#### 11.5.1 实施记录（2026-09-27，Agent 侧）

> **分工**：Agent 侧无 GPU，只能做"代码 / 脚本 / host 用例 + 构建验证"；
> 真机执行（P6_3-0、P6_3-4 ~ P6_3-6）由作者完成。**未跑过的一律不写"通过"**。
>
> **状态列已作废（2026-09-27）**：本表只回答"落点在哪"，**状态以 §11.5.2 为唯一来源**——
> 这里原先写的"真机未跑"等字样在真机执行后没有回改，两处并存会互相矛盾（本项目已经吃过
> "文档与现状不符"的亏）。所以下面每行只留一个 ⬜/✅ 摘要，**细节一律看 §11.5.2**。

| 任务 | 状态 | 落点 |
|---|---|---|
| P6_3-1 协议定稿 | ✅（详见 §11.5.2） | 本节 11.3 / 11.4 + 测试计划 §10（`future_iterations.md` §11 的 G6 行已改指此处） |
| P6_3-2 profile target | ✅（详见 §11.5.2） | `tools/profile/run_profile.sh`（nsys / ncu 一键：时间戳、温度 / 时钟、自动导出 kernel 与 API 摘要）；`tests/CMakeLists.txt` 的 `profile_gpt2` / `profile_resnet18` / `profile_gpt2_ncu` / `profile_resnet18_ncu`（**不进默认构建**） |
| P6_3-3 decode 计时入口 | ✅（详见 §11.5.2） | `tests/test_decode_perf.cpp` 的 `Gpt2DecodePerf.StepLatencyByPhase`（P 层：只打印；用斜率 `(T32−T1)/31` 分离 prefill 与 decode） |
| P6_3-4 kernel 分解 | ⬜ 本机不可达（详见 §11.5.2） | `tools/profile/summarize_nsys.py`：三层分桶（我们的 kernel / CUB / TRT）+ sampler / attention / KV 占比；`--self-test` 已注册 ctest 项 `profile_summary_selftest`。**本机两条 CLI 路径都拿不到 kernel 时间线** → `TROUBLESHOOTING.md` #41 |
| P6_3-5 分配开销 | ✅（详见 §11.5.2） | 同一脚本的 `--api` 模式：`cudaMalloc` / `cudaFree` 的次数与总耗时（为 §2.1 供数） |
| P6_3-6 G6 跨构建对照 | ➡️ **已移交 §10.2**（详见 §11.5.2） | 协议见 11.3 B；无新代码。这项服务的是"要不要做 ONNX 子图替换"，**不属于 §11 的收口范围** |
| P6_3-7 回填 | ✅（详见 §11.5.2） | —— |
| P6_3-0 仪器自证 | ✅（详见 §11.5.2） | 结论：**本机采不到 GPU kernel 时间线**（#41）；脚本 / target 本身可用。Agent 沙箱里 `nsys` 一跑即报 `open: Operation not permitted` |

**沙箱实测（2026-09-27）**：`ctest --test-dir build` → **242 条 / 0 失败**
（新增 5 条 `PerfStatsTest.*` 实跑通过、1 条 `Gpt2DecodePerf.StepLatencyByPhase` 无 GPU 显式跳过、
1 条 `profile_summary_selftest` 通过）。

**落地时与上文计划的两处偏差（先记，原因写清）**：

1. **分配开销改为读 nsys 的 `cuda_api_sum`**，不在 `Generate` 里插桩。**为什么**：插桩要改产品代码，
   而本轮定的是"零产品代码改动"；CUDA API 计数是 nsys 的**精确拦截**，与 11.4 禁止的
   "从 kernel 名猜语义"不是一回事。
2. **`profile_*_ncu` 默认只采 20 个 kernel**（`MINI_TRT_NCU_LAUNCH_COUNT` /
   `MINI_TRT_NCU_KERNEL_FILTER` 可覆盖）。**为什么**：整网 ncu 采集过久，本轮只要
   "导出路径打通、能读出 kernel 名与耗时"，不做深层调优。

**真机第一次执行（2026-09-27，作者；三个发现、两个当场修）**：

1. **`profile_gpt2` 失败，原因在被测用例而不在 nsys**：`nsys` 会**透传被 profile 程序的退出码**，
   ninja 的失败即等于 `Gpt2DecodePerf` 红了。用 `nsys stats --report cuda_api_sum` 后处理那份
   已生成的报告（**无需 GPU**）看到 30921 次 `cudaLaunchKernel` → 用例其实跑完了全部测量，
   红在我新加的 `EXPECT_LT(|Δmedian|, 0.6 ms)`。
   **这是我自己的错**：`0.6 ms` 是**采样器类**比较的判别下限（#37 / #38），被我套到量级大
   一到两个数量级的**整步 decode** 上——`AGENTS.md` §7 明令禁止的"阈值跨场景复用"，
   也违反本节"P 层只打印、不设阈值"。**已改为只打印**绝对 + 相对漂移。
2. **"假 CSV"**：`nsys stats` 自身的 `Generating SQLite...` / `Processing [...]` 走 **stdout**，
   和 CSV 混在一起；resnet18 那两份摘要其实只有这几行消息（`kern_sum.csv` 仅 415 B）。
   **已改为**"先写临时文件、确认有 `Total Time` 表头才落正式名"。
3. **WSL2 拿不到 GPU kernel 时间线**：`cuda_gpu_kern_sum` 直接 `SKIPPED: ... does not contain
   CUDA kernel data.`（两次运行都一样）；**CUDA API 摘要是好的**。
   → **P6_3-4（三层分解）在本平台受阻**：`profile_gpt2_ncu` 也失败
   （`==ERROR== Unknown Error on device 0.`，无 `.ncu-rep`），**且把 `.nsys-rep` 拷到 Windows
   也看不到 kernel 时间线**——报告里压根没有那份数据（#41 更正了 #39 的说法）。可行绕法见 #41
   （推荐先用"同 session 的 `SamplerPerf` + 本用例取比值"）。**P6_3-5（分配开销）不受影响**。

另外两处可用性已修：被 profile 进程的输出（gtest + TRT + nsys）一律落 `${base}.app.log`，
终端只回 ≤20 行摘要（不再"满屏看不到原因"）；去掉 `nsys profile --stats=true`。
完整定位路径与教训见 `docs/TROUBLESHOOTING.md` **#39**。

**第一次成功出数（2026-09-27，修完重跑；引擎 `cache hit` ×2、无重建）**：

| 量 | 值（n=15，warmup=3，每轮 ABBA；取自 `gpt2_decode_20260927_050355`） |
|---|---|
| T(1)（prefill 4 token + 1 个 decode 步） | median **4.947 ms**（p25 4.730 / p75 6.298，max 8.372） |
| T(32) | median **93.190 ms**（p25 90.109 / p75 103.367，max 134.839） |
| 派生 decode 每步（斜率 `(T32−T1)/31`） | **2.847 ms** |
| 派生 prefill（4 token） | ≈ **2.100 ms** |

**四个结论（含一个仍缺的）**：

1. **这台机器在一次测量内就有 10~18% 的漂移**：同 session 两组测量的 `|Δmedian|` =
   **0.917 ms (18.5%)** / T(1)、**9.478 ms (10.2%)** / T(32)；同期 GPU 从 72 °C / 44.5 W
   升到 78 °C / 65.7 W（时钟 1875→1860 MHz）。→ **机器没进稳态**；整步 decode 的比较
   **必须**同轮交替 + 报告漂移；§11.3 C 的 ±400~600 µs 下限**绝不能**套到这一量级
   （这正是 #39 的教训）。
2. **batch=1 的 decode 是"每步固定开销主导"，不是 token 数学主导**：prefill 一次算 4 个 token
   约 2.10 ms，而 decode 每 token 要 2.85 ms；本次运行 ~30921 次 `cudaLaunchKernel`（~26 次/步）。
   **假设**是 launch / 固定开销占大头，但**要等 kernel 时间线才能定论**。
3. **§2.1（显存池）的触发条件未获支持**：`cudaMalloc` 全程 63 次 / 3.335 ms、`cudaFree`
   69 次 / 249.7 ms（**含隐式同步，是拆除期成本，不是分配器成本**）→ 没有"频繁分配拖慢
   decode"的证据；要定论还需把 `api_sum` 按"建引擎期 / `Generate` 期"拆开。
4. **逐 kernel 分解仍缺，但"sampler 占多少"这一问已用绕法补上**：nsys 报告不含 GPU kernel 数据、
   `profile_gpt2_ncu` 同样失败（`Unknown Error on device 0`），且**拷到 Windows 也救不了**
   （数据没被采集）→ 逐 kernel 时间线仍拿不到（#41）。
   已按 #41 的**绕法 1 实现**：`Gpt2DecodePerf` 在**同一次运行**里再量 greedy / top-k(k=64) /
   top-p(p=0.9) 的净成本（斜率 `(T4−T1)/3`、正反交替、n=9），直接打印**占 decode 每步的比例**
   （测试计划 **PF-8**）。**边界**：这是比值、不是分解，attention / MLP 仍包在"decode 一步"里，
   且 sampler 在静态 logits 缓冲上量（cache 状态与真实循环不同）。
   **真机结果（2026-09-27，已复核）**：decode 步 **2.458 ms**；greedy **0.0308 ms（1.25%）**、
   top-k(64) **0.4360 ms（17.7%）**、top-p(0.9) **0.4938 ms（20.1%）**。
   与 `SamplerPerf` 在**同一 session** 内互校一致（0.4281 / 0.4916，差 ≤2%）。
   **首版用全零 logits 曾给出 0.92% / 4.75% / 6.35%（差 2.5×）——那是退化输入，已作废；
   根因与推理见 `TROUBLESHOOTING.md` #42。**
   **注意量级边界**：decode 步本身跨 session 已见 2.458 / 2.847 / 3.365 ms（±27%）→
   占比**只在同一 session 内可比**。

**§11.4.1 上下文扫描的执行结果（2026-09-27 真机，测试计划 PF-9）**：

| prompt | 平均上下文 | 每步 decode |
|---|---|---|
| 4 | 20.5 | **3.055 ms** |
| 256 | 272.5 | **6.062 ms** |
| 960 | 976.5 | **14.705 ms** |

- 两段斜率 **11.93 / 12.28 ms per 1000 位置**（差 3%）→ **线性**，外推 1024 = +12.2 ms
  → **长上下文下 attention ≈ 每步 80%**；**§2.2 的触发条件据此成立，已从 P3 升 P2**
  （`future_iterations.md` §0.1 / §0.2 / §0.3 第 14 项）。
- **机制**：`LaunchPagedAttention` 的 `grid=(num_heads, batch)` → 每层 12 个 block，
  本机 24 个 SM（一半闲置）；每块 64 线程串行走完上下文 → 延迟受限，有效带宽 ≈6 GB/s
  （约峰值的 3%）。方向 = **把上下文维切开并行**（FlashDecoding 式 split-K）。
- **踩到并修掉的一个自伤**：最初这条用例只给 prefill 换了引擎路径、decode 复用主用例那条，
  但**指纹覆盖整个 `EngineBuilder::Config`**（含 prefill 的 seq 参数）→ 真机日志直接出现
  `Engine cache stale: gpt2_real_decode.engine → 重建`，两个用例会**交替**判对方过期、
  每次来回都重建一次（分钟级）。已改成两条都用自己的路径
  （`..._ctxsweep_{prefill,decode}.engine`）——这正是 `PROGRESS.md` §2.15
  "缓存路径不得在不同配置间共用"那条规矩，这次是**自己踩给自己看**。
- **两条边界**（写给引用这些数的人）：① 这是 attention **随上下文增长的部分**，
  不是它的绝对时间；② 这些比值**只在同一 session 内可比**，且 sampler 那个 17~20%
  是**短上下文**下的比例——长上下文下分母变成 14.7 ms，同样的 sampler 只占 ≈3%。

#### 11.5.2 §11 收口状态（2026-09-27）

> 有人问过"§11.5 算完成了吗"——**答案是"目标达成，但有一项没跑"**。逐条列清楚，免得下个会话重推。

| 任务 | 状态 | 说明 |
|---|---|---|
| P6_3-0 仪器自证 | ✅ 完成 | 结论：**WSL2 采不到 GPU kernel 时间线**（nsys 无 GPU 活动、ncu 报错、sudo 与显式 `--trace=cuda` 都无效）→ `TROUBLESHOOTING.md` #41 |
| P6_3-1 协议定稿 | ✅ 完成 | §11.3 / §11.4 |
| P6_3-2 profile target | ✅ 完成 | 四个 target 跑通，摘要导出与"假 CSV"问题已修（#39） |
| P6_3-3 decode 计时入口 | ✅ 完成并出数 | PF-3：T(1)/T(32)/每步/漂移都打印 |
| P6_3-4 nsys 三层分解 | ❌ **本机不可达** | 但**目的已达成**：sampler 占比 → PF-8（真机出数并互校）；attention 占比 → PF-9（上下文扫描） |
| P6_3-5 分配开销 | ✅ 有数据 | `cuda_api_sum`（精确拦截）；结论：**§2.1 的触发条件未获支持** |
| P6_3-6 G6 跨构建对照 | ➡️ **已移交 §10.2** | PF-7（ONNX vs 原生 prefill，≥3 构建 / ≥20 推理）**没跑，但它回答的是"要不要做 ONNX 子图替换"**，与本节的"decode 时间花在哪"无关 → 移交到 §10.2 的前置条件（`future_iterations.md` §0.1 / §10.2） |
| P6_3-7 回填 | ✅ 完成 | 除 P6_3-6 无数据可填 |

**结论**：`future_iterations.md` §6.3（profile target）与 G6（可复现测量方法）**目标已达成**——
测量方法建立、被 PF-8/PF-9 两次实际使用并给出结论（含一次纠错 #42）。

**收口决定（2026-09-27）：§11 关闭。** 理由与边界如下——

1. **目标达成**：§11 存在的理由是"把 decode 的时间花在哪从'不知道'变成'知道'，并留下可复现的尺子"。
   两件都做到了；**尺子自己抓到过一次测量错误（#42）**，这是它可用的最硬证据。
2. **P6_3-4 记为"能力边界"而非"未完成"**：工具写好、自检过、报告能生成，缺的只是
   "有 GPU 跟踪能力的机器"（#41 已证本机做不到）。换一台能采 GPU 时间线的机器即可直接用。
3. **P6_3-6 / PF-7 移交 §10.2**：它回答的是"要不要做 ONNX 子图替换"，与"decode 时间花在哪"无关；
   挂在 §11 里会让一个已完成的章节永远收不了尾。**移交后 §10.2 的前置 = 先跑 PF-7**。
4. **引用这些数时的三条纪律**（写给下一个人）：① 占比**只在同一 session 内可比**
   （decode 步跨 session 已见 ±27%）；② sampler 的 17~20% 是**短上下文**下的比例，
   长上下文下分母变大、它只占 ≈3%；③ "attention ≈80%" 是**随上下文增长的边际部分**。

</details>

### 11.6 验收判据（每条都要能回答"凭什么"）
> **判据的唯一出处 = 测试计划**（`docs/future_iterations_test_plan.md` §10.3）：本节保留的是**设计侧视角**，与测试计划重复的行**以测试计划为准**。

| 判据 | 值 / 形式 | 凭什么 |
|---|---|---|
| nsys 无头导出可用 | `.nsys-rep` 存在且非空；`--stats=true` 能在无 GUI 下打出 kernel 摘要 | `AGENTS.md` §1 的无头导出策略 |
| ncu 无头导出可用 | `.ncu-rep` 存在；**本机实测不可用**（`Unknown Error on device 0`）→ 记为已知限制而非通过 | `TROUBLESHOOTING.md` #41 |
| 同二进制 A/B 可判 | 两次 ABBA 的中位数之差 **> ±400~600 µs** 才算"有差异" | `TROUBLESHOOTING.md` #37 / #38 的真机实测下限 |
| 跨构建对照可判 | 同一 session 内 ≥3 次构建的中位数**极差** < 两组中位数之差 → 可判；否则"未定" | G6 原文（`future_iterations.md` §11）；这是 G6 存在的理由 |
| 分解可复核 | 三层时间之和 vs 总时间的偏差写出来；命令、引擎指纹状态、形状、n 全给 | `PROGRESS.md` §2.14 C"诊断必须自证" |
| sampler 占比有结论 | 给出中位数与区间（**观测值，不设阈值**） | 这是"回答一问"，不是"达标判据"；设阈值会退化成 §7 禁止的判据 |
| 生产路径不受影响 | profile target 不进默认构建；`detailed_profiling` 用独立 engine 路径 | `PROGRESS.md` §2.15（改 I/O / 指纹 = 改契约） |

### 11.7 风险与回退

| 风险 | 影响 | 缓解 / 回退 |
|---|---|---|
| WSL2 上 `ncu` 拿不到 performance counter | **kernel 级时间线整体缺失**（nsys 同病） | 走 #41 的绕法：同 session 的 `SamplerPerf` + 本用例取比值（不改产品代码）；或在 Windows 宿主侧采集；或改产品代码插桩（须单独立项） |
| nsys 采样开销改变读数 | 与隔离计时口径不符 | 两口径都报、以隔离计时为准（11.3 C） |
| TRT kernel 名无语义 | 分解被硬贴标签 | 只做 11.4 的三层；要语义就读 `IEngineInspector`（独立 engine） |
| 首次构建分钟级 + tactic 随机器状态变 | 跨构建结论不可复现 | 先 warm cache；跨构建必须报极差，极差大就判"未定" |
| 温度 / 功耗漂移被当成改动效果 | 假结论 | 同轮 ABBA + 记录温度 / 时钟（11.3 A / D） |
| profile 轮次顺手改了实现 | 尺子与对象混在一起 | 本轮**零产品代码改动**；发现真瓶颈只登记，留给 §2.2 / §10.2 |

### 11.8 破坏性动作清单（**动手前一次性确认**）

按 `AGENTS.md` §0.5：下面每条都要单独获批，**不**因为"计划里写了"就自动生效。


<details><summary>展开：11.8 破坏性动作清单（**动手前一次性确认**） 全文</summary>

1. **改构建脚本**：修改 `mini_trt_llm/CMakeLists.txt`（和 / 或 `tests/CMakeLists.txt`）新增 profile target；
2. **新增 profile 输出目录**（默认 `/tmp/mini_trt_llm_profiles/`，**不入库**）；
3. **可能新增计时入口**：若放在 `tests/` 下（新文件或改 `test_gpt2_generate.cpp`）→ 新增源文件**必须重跑 configure**；
4. **删除 / 重建引擎缓存**：`/tmp/mini_trt_llm_gpt2_*.engine`（只在需要固定构建态时；单列获批）；
5. **真机执行 profiling / ncu**：属"单测外的测试任务"，按 `AGENTS.md` §0.3 **须先获批**；
6. **是否把报告入库**：默认**不入库**（体积大）；若要入库，`.gitignore` 也要改 → 单独确认。

> **执行结果（2026-09-27，作者批准 1~5、明确"报告不入库"）**：
> 1 / 2 / 3 已落地（见 §11.5.1 的落点表）；4 与 5 属真机动作，**尚未执行**；
> 6 按"不入库"执行——报告与摘要一律落 `/tmp/mini_trt_llm_profiles/`，`.gitignore` 未改。

</details>

### 11.9 真机执行清单（**已执行**：2026-09-27 四步全部跑完；命令保留供复跑。Agent 侧无 GPU）

```bash
# 0) 构建（不改产品代码时先跑一遍让引擎缓存热起来）

<details><summary>展开：11.9 真机执行清单（**已执行**：2026-09-27 四步全部跑完；命令保留供复跑。Agent 侧无 GPU） 全文</summary>

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# 1) 仪器自证（P6_3-0）：先拿平凡进程确认 nsys / ncu 能无头导出
nsys profile -o /tmp/mini_trt_llm_profiles/smoke_nsys true
# ncu 在本机不可用（`Unknown Error on device 0`，见 TROUBLESHOOTING #41）——
# 别指望"拷到 Windows 看"：报告根本没生成，nsys 那份也没采到 kernel 数据

# 2) decode 分解（P6_3-4）：profile 固定过滤器下的用例
nsys profile --stats=true -o /tmp/mini_trt_llm_profiles/gpt2_decode \
    ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='<P6_3-3 的过滤器>'

# 3) G6 跨构建对照（P6_3-6）：同一 session 内各构建 3 次、每次推理 ≥20 次
#    每次都记录：Engine cache hit/stale、中位数、极差、温度 / 时钟

# 4) 回填（把 11.3 D 的六项 + 11.4 的三层分解贴回来）
```

- **每轮只改一个变量**（`PROGRESS.md` §2.14 C）；
- **真机跑全量仍要带 `MINI_TRT_REQUIRE_GPU=1`**；
- 报告先落 `/tmp`，确认无敏感信息再谈是否入库。

---

</details>

## 12. [OI-FLASHDECODING-PLAN] §2.2 长上下文 attention：FlashDecoding 式 split-K（开发计划）

> 条目事实来源：`docs/future_iterations.md` **§2.2**（是什么 / 为什么 / 触发条件）。
> 本节的唯一职责：**改哪些文件、分几步、每步怎么自检、哪一步要作者批**。
> 配套测试计划：`future_iterations_test_plan.md` **§11**。
> **状态：已交付并真机验证（2026-09-27；性能数据与验收逐条对照见本节末的回填表）。**
> 按 `AGENTS.md` §0.7，计划文档本身**不构成开工许可**——当初的开工由作者点名到 P2_2-0 ~ P2_2-9 才发生。
> 沿用 §11 的分工：本节不复制 §2.2 的事实与实测值，只写执行安排。

### 12.1 计划对账（AGENTS.md §5 第 0 步）

**1) 有没有计划文档**：此前**没有**。§2.2 只有两行"工作内容"级描述，没有文件级改动面、
没有步序、没有验收判据 → 属于"没有执行计划"。本节 + 测试计划 §11 就是补上的那一份。

<details><summary>展开：12.1 计划对账（AGENTS.md §5 第 0 步） 全文</summary>


**2) 是否一致**（逐条对照 §2.2 原文）：

| §2.2 原文 | 现状核实 | 一致？ |
|---|---|---|
| "在 `PagedAttentionPlugin` 中把**上下文维**切成多个 block 并行，再做两阶段 softmax 归约" | 现状是 `grid=(num_heads, batch)`、每块 64 线程串行走完 `total_len`（`paged_attention_plugin.cu`）；改动方向与此一致 | ✅ 一致 |
| "支持 decode 阶段的 batching 优化（与 `G2-3` 的 batch 扩展一起考虑更省事）" | **有偏差**：`PagedAttentionDecodeKernel` 的 `grid.y = args.batch_size`，**kernel 本来就支持任意 batch**，`PagedAttentionKernelTest.MhaMatchesCpuReferenceAcrossMultipleBlocks` 已覆盖 `batch=2`；真正把 batch 卡在 1 的是 `LLMRunner`（缺口 **G2-3**，属 `future_iterations.md` §2.3） | ❌ **偏差** → 按 3) 处理 |
| "sm_75 无官方实现，仍需自研 kernel" | 与现状一致（无官方 FlashAttention） | ✅ |
| 触发条件"长上下文 attention 占每步 ~80%" | 出处 PF-9（`PROGRESS.md` §3.0h、测试计划 §10.2 PF-9），三档实测与斜率都在 | ✅ |

**3) 偏差怎么处理**：**先改文档再改代码**。§2.2 的"batching 优化"一句**已在
`future_iterations.md` §2.2 当场作废**（保留删除线 + 理由），本条**范围收敛为：只做上下文维
split-K，不扩 batch**。理由：把 batch 扩展混进来会一次改两个变量，读数不可归因
（`PROGRESS.md` §2.14 C"每轮只改一个变量"）。

**4) 反向查（文档与代码矛盾处，当场修；修不了的登记）**：

- **【本轮已修，文档】** §2.2 的 batching 一句与 `grid.y = batch_size` 直接冲突 → 已改；
  改动原因写进 `future_iterations.md` §2.2 与本节。
- **【本轮登记，未改代码】** `tests/test_paged_attention_plugin.cpp` 的
  `ExpectMatchesReference` 里阈值 `1e-4f / 1e-5f` **旁边没有出处**，而 `AGENTS.md` §7 要求
  "每个数值阈值旁边必须写清出处"。它是本条的**主判据** → 本轮**只补出处、不放宽**；
  出处需作者确认（拟：Phase 1 算子层对 double 参考的既有口径，自 PE1 起沿用）。
  **未获批准前不动代码。**
- **【本轮登记】** `PagedAttentionPlugin::getWorkspaceSize()` 现返回 0，注释写"online softmax
  只用到 static shared memory"。split-K 需要一块 partial 缓冲 → **这条注释会过期**，
  必须与实现同批改（否则下一个人会照注释把 workspace 又改回 0）。

</details>

### 12.2 目标与范围

**目标（一句话）**：把 decode 阶段 PagedAttention 的**上下文维**切开并行
（FlashDecoding 式 split-K + 两阶段 softmax 归约），**在不放松任何数值判据的前提下**

<details><summary>展开：12.2 目标与范围 全文</summary>

降低长上下文每步 decode 的 attention 边际成本。

**范围内**：FP32 / FP16 的 split-K kernel、workspace 契约、分片策略（host 可测）、
插件装配、引擎缓存安全（bump 图版本）、A/B 入口、kernel 级与端到端 P 层测量。

**范围外（明确不做，防止范围外扩）**：

- **不扩 `LLMRunner` 的 batch**（那是 G2-3 / `future_iterations.md` §2.3）；
- **不做 PagedAttention 的 prefill 阶段**（`future_iterations.md` §9.1 已冻结：双引擎路径下不需要）；
- **不改精度口径 / 不碰 Q/DQ**；**不做 CUDA Graph 捕获**；**不动 sampler**（`future_iterations.md` §9.2 已关闭）。

</details>

### 12.3 设计决策（D1~D5）

**D1　并行维度 = 上下文维（split-K），不是 head 维 / batch 维。**
GPT-2 decode 的并行度是 `num_heads(12) × batch(1) = 12` 块，本机 **24 SM** → 一半闲置；

<details><summary>展开：12.3 设计决策（D1~D5） 全文</summary>

head / batch 维不缺并行度，缺的是把 976 个位置串行变并行。
**不选**"再细分 head 维"——block 数翻倍但每块仍要串行走完整段上下文，延迟不减。

**D2　两阶段结构（stage-1 分片 → stage-2 归约），不用块间原子合并。**

- stage-1（`grid = (num_heads, batch, num_splits)` 或等价的展平）：每个 block 负责一段
  连续逻辑位置，算局部 `m_i / l_i / acc_i[d]`（**一律按 float 累积**），写进 workspace；
- stage-2（`grid = (num_heads, batch)`）：`M = max m_i`，再
  `l = Σ l_i·exp(m_i−M)`、`acc = Σ acc_i·exp(m_i−M)`，输出 `acc / l`。

**为什么不用原子合并**：原子累加无法保序 → 结果不可复现，与本项目的可复现判据冲突。
**`num_splits == 1` 时直接走原单趟 kernel**（连 stage-2 都不发）→ 短上下文不退化成两次发射。

> **落地修正（2026-09-27，P2_2-1 设计定稿时发现）**：上面那句"`num_splits == 1` 时直接走原单趟
> kernel"**在宿主侧判不出来**——`llm_runner.cpp:303` 把 `block_tables` 的形状钉成固定的
> `[batch, max_blocks_per_seq]`，`context_lens` 只在设备上，插件 `enqueue` 拿到的运行期形状
> 里**没有任何字段能反映"当前上下文有多长"**。要拿到它只有两条路：① 一次 4 字节 D2H + sync
> ——**被 `AGENTS.md` §3.A.3 明令禁止**（decode 循环内零 H2D/D2H）；② 改 `LLMRunner` 让
> `block_tables` 按当前需要收窄——**超出 §12.2 的范围**（那会动引擎 profile 语义，且要同步改
> ONNX 路径）。
> **改为**：stage-1 的 grid **恒为 `kMaxSplits`**，每个 block 用自己那份 `context_lens[b]`
> 算出 `effective_splits`，`split_idx >= effective_splits` 的 block **立即返回、不读不写**；
> stage-2 的每个 block 同样只读自己 `effective_splits` 范围内的 partial。
> **代价**：短上下文也多发一次（每层 1 次、12 层 → 每步多 12 次发射；按单次 ≈3~6 µs 估
> ≈36~72 µs/步，相对 ctx≈20 档的 3.055 ms 是 **1~2%**）→ 由 §12.6 第 5 条的 B 半句
> （"退化不超过同 session 漂移"，实测漂移 10~18%）吸收。
> **为什么不做"一次发射 + 原子 ticket 由最后一个 block 归约"**：它能省掉这次发射，但引入
> 跨 block 的内存序依赖，属于"看运气"的缺陷类别（本项目 #4 / #5 / #8 / #10 / #15 全是数值
> 正确性问题）→ **先落最简、可复核的两段式**；若将来实测确认这次发射是瓶颈，再单独立项。

**D3　workspace 必须走 `getWorkspaceSize()`，尺寸按动态形状的 `.max` 上界给。**

- `AGENTS.md` §3.B.3 禁止在 `enqueue` 里分配；分片缓冲
  （`kMaxSplits × batch × heads × (2 + head_size)` 个 float）在 `getWorkspaceSize` 里报**上界**；
- `DynamicPluginTensorDesc` 带 `min / opt / max` 三组形状 → **用 `.max` 求上界**，
  **不是**用构建期的 `desc.dims`（那时动态轴可能是 -1，会让 workspace 偏小 → 越界写）。
  这条要写进注释，否则下一个人很可能"顺手"用 `desc.dims`。
- **契约**：`getWorkspaceSize()` 报的上界必须 ≥ `enqueue` 实际用到的量——
  这是 workspace 版的"按对方查询，不按配置假定"（`PROGRESS.md` §2.15）。

**D4　分片策略做成 `__host__ __device__` 纯函数，host 侧可裁决。**
照 `sampler/nucleus_cutoff.hpp` 的先例：`PlanSplits(total_len, split_idx, num_splits) → {begin, end}`，
并把"**空分片 → `m=-inf, l=0, acc=0`**"定成显式语义（`PROGRESS.md` §2.12：存在部分写入路径的
kernel 必须显式处理未覆盖区间——`TROUBLESHOOTING` #4 就是这么来的）。
**为什么**：分片边界（不整除、`total_len < num_splits`、`context_len=0` + 带当前 token）
**全是纯逻辑**，能在沙箱裁掉就不该留给真机——真机往返是最贵的资源（`PROGRESS.md` §2.13）。

**D5　旧单趟 kernel 保留为永久 A/B 入口（不删）。**
照 `future_iterations.md` §9.2 的 `LaunchTopPSamplerTwoLevel` 先例：kernel 级与 P 层都能**同二进制、同 session、
同轮交替**测两版（`TROUBLESHOOTING` #37 / #38 的教训：跨协议、跨 session 的差值不可直接比）。
同时它还是 `num_splits == 1` 的生产路径，不是死代码。

</details>

### 12.4 任务分解（P2_2-0 ~ P2_2-9）


<details><summary>展开：12.4 任务分解（P2_2-0 ~ P2_2-9） 全文</summary>

| 编号 | 任务 | 层 | 自检（做完立刻能看见什么） |
|---|---|---|---|

| **P2_2-0** | **基线复测**：在**未改动的代码**上跑 PF-9（`ContextLengthSweep`），留同 session 基线 | P | 打印三档每步耗时与斜率；与 §11.5.1 的 3.055 / 6.062 / 14.705 ms 对照（**跨 session 不可比**，只作"没跑错"的粗校验） |
| **P2_2-1** | 设计定稿：workspace 契约、分片策略、空分片语义、FP16 partial 用 float（D2~D4） | —— | 本节 §12.3 落地成代码注释，无产品代码 |
| **P2_2-2** | `paged_attention_kernel.hpp`：加 `LaunchPagedAttentionSplit`（或 `args.num_splits`）+ 测试用 `SetPagedAttentionNumSplitsOverride`；`PagedAttentionKernelArgs` 增字段 | H（契约） | 头文件编译通过；公开 API 带 Why 注释 |
| **P2_2-3** | 新头 `paged_attention_split.hpp`：`PlanSplits` 纯函数 + num_splits 选择策略 | H | **host 用例**（沙箱可跑）覆盖边界，见测试计划 §11.2 的 H 组 |
| **P2_2-4** | `paged_attention_plugin.cu`：stage-1 / stage-2 kernel（FP32 / FP16 各一套实例化）、`getWorkspaceSize` 改报上界、`enqueue` 按 `num_splits` 选路径 | G | 沙箱编译通过；真机单测见 §11.2 的 G 组 |
| **P2_2-5** | 插件装配：`configurePlugin` / `onShapeChange` 刷新分片状态；**bump `kEngineGraphVersion`**（安全必需，见 §12.8 第 1 条） | G | 真机日志出现 `Engine cache stale → 重建`，第二次运行出现 `cache hit` |
| **P2_2-6** | 单测扩展 `test_paged_attention_plugin.cpp`：split-K vs CPU double 参考（MHA / GQA / MQA、`batch>1`、跨多块、`context_len=0`、带当前 token、`total_len < num_splits`）+ 新旧 kernel 同二进制对照 | G | 判据 = §12.6 第 1 / 2 / 4 条 |
| **P2_2-7** | FP16 覆盖 `test_fp16_paths.cpp`：split-K 的 FP16 分支对原有 FP16 参考 | G | 同 §11.2 的 G-3 |
| **P2_2-8** | 端到端回归：`test_gpt2_decode_consistency` + `test_gpt2_generate` 的 8-token 基线 | G | **逐 token 与既有基线一致**（"改 kernel 没改语义"的硬判据） |
| **P2_2-9** | 性能采集（kernel 级 A/B + PF-9 端到端 A/B，同轮交替）与文档回填 | P | 见 §12.5 与测试计划 §11.3 / §11.5 |

**顺序纪律**：0 → 1 →（2 / 3 可并行）→ 4 → 5 → 6 → 7 → 8 → 9。
**每轮只改一个变量**（`PROGRESS.md` §2.14 C）：先只落 split-K；测完再决定要不要动下面 §12.7 末行那类
"block 内组织"的二次优化。

**执行状态（2026-09-27，Agent 侧已做完的部分）**：

| 任务 | 状态 | 落地内容 / 证据 |
|---|---|---|
| P2_2-0 基线复测 | ⬜ **未做（属作者真机）** | 按 §12.3 D2 的落地修正，基线可在实现后**同一 session 内**用旧 kernel 的 A/B 入口一起量 |
| P2_2-1 设计定稿 | ✅ | D2 记了落地修正（宿主拿不到 `total_len`）；F1~F4 已按 A 拍板、7 条破坏性动作已批 |
| P2_2-2 头文件 | ✅ | `paged_attention_kernel.hpp` 增 `LaunchPagedAttentionSplit` + override 读写；`kPagedAttentionPluginVersion` 1→2 |
| P2_2-3 分片策略纯函数 | ✅ | 新头 `paged_attention_split.hpp`（`ResolveSplits` / `SplitRange` / `WorkspaceSlotOffset` / `WorkspaceBytes`）；`tests/test_paged_attention_split.cpp` 8 条 host |
| P2_2-4 kernel | ✅（沙箱只验编译） | `paged_attention_plugin.cu` 增 `PagedAttentionSplitKernel` / `PagedAttentionMergeKernel`（FP32+FP16）、`getWorkspaceSize` 改按 `.max` 报上界、`enqueue` 走 split 路径 + 无 workspace 时兜底单趟 |
| P2_2-5 装配 + 图版本 | ✅ | `configurePlugin` 无需新状态（片数由设备端按 `context_lens` 推导）；`kEngineGraphVersion` 1→2 |
| P2_2-6 单测 | ✅ **真机通过**（修夹具后复跑） | 6 条 `PagedAttentionSplitKernelTest.*`（含护栏区越界检查与 PG-7 诊断）。首轮 3 条 SEGFAULT + 1 条断言失败**根因全在测试侧**（块表行宽不足以放下 `ceil(ctx/block_size)` 个块 → host 参考越界；一条断言把 GQA 比例写反）→ 已加 `MakeLongContextFixture` + `AssertFixtureConsistent` 并修正断言，见 `TROUBLESHOOTING` #44。**产品代码一行未改** |
| P2_2-7 FP16 | ✅ **真机通过** | `Fp16PathTest.PagedAttentionSplitMatchesFp16Reference`（自适应 2 片 + 强制 8 片）→ **这是 split-K 多片归并的第一份真机数值证据** |
| P2_2-8 端到端回归 | ✅ **真机通过** | `Gpt2DecodeConsistency.*` 与 `Gpt2GenerateTest`（FP32 8-token 冻结基线）均在通过之列；唯一红是 `RealGpt2Fp16GreedyMatchesReferenceTokens`（**按设计**，`PROGRESS.md` §5.11 的 FP16 NaN） |
| P2_2-9 性能采集 + 回填 | ✅ **真机出数并回填** | PP-1（kernel 级）降幅 **88.85%**、PP-2（端到端）降幅 **88.12%**——两把独立尺子互校差约 5%，均远超 F1 的 40%；详见测试计划 §11.5 |

**验收判据逐条对照（§12.6；全部达成，2026-09-27 真机）**：

| # | 判据 | 结果 |
|---|---|---|
| 1 | 对 CPU double 参考 rel < 1e-4 / abs < 1e-5（**不放宽**） | ✅ PG-1/2/4/5/6 全绿 |
| 2 | 边界覆盖（MHA/GQA/MQA、batch>1、跨多块、`ctx=0`、带当前 token、空分片） | ✅ 每条都有对应用例且通过 |
| 3 | 语义不变（decode 一致性 + 8-token 冻结基线逐 token 一致） | ✅ `Gpt2DecodeConsistency.*` 与 FP32 8-token 基线通过；PP-2 三档 **token 一致=是** |
| 4 | 新旧差异只作诊断 | ✅ `max_abs=3.58e-07` / `max_rel=1.52e-06`（比判据低 ~65×）→ 无需放宽任何阈值 |
| 5A | 性能：ctx≈960 斜率至少降 40% | ✅ **88.85%（PP-1）/ 88.20%（PP-2，两次复现：88.12 / 88.20）** |
| 5B | ~~性能：ctx≈20 档退化不超过同 session 漂移~~ → **观测项，不设判据**（作者 2026-09-27 决定） | 只报数：实测 **+1.877%**（端到端），机制 = 每层多一次归并发射（D2 在拿到数据前已预估并接受）。原判据作废的三个理由：① "漂移"有 `max`/`min`/配对差三种读法、前两种结论相反；② 曾议的"退化 ≤2%"唯一数值输入（单次发射 3~6 µs）**无出处**；③ PP-1 与 PP-2 对同一笔代价差 **1.97×** 且未解释。见 `TROUBLESHOOTING` #45 / #45.1 |
| 6 | 缓存安全：bump 后首次 stale 重建、次次 cache hit | ✅ 真机日志 `Engine cache stale → 重建`（627/475 MB）后复跑复用 |
| 7 | 无回归：真机只剩按设计的 1 红 | ✅ 259 条 / 1 红（FP16 NaN）/ 1 跳过（`int8_crosscheck` 缺报告） |

**F1-B 最终定为"观测项、不设判据"（作者 2026-09-27 决定）。** 原措辞是"ctx≈20 档退化不超过
**同 session 实测漂移**"，它在真机数据面前暴露了三重问题，导致这个判据无法成立：

1. **"漂移"有歧义，且两种读法结论相反**：绝对漂移取 `max` = 12.29% → 判"通过"；
   取 `min` = 0.97% → 判"不通过"（配对差口径的更严格读法还要再测）。而那个 12.29% 来自
   **整轮第一档、机器未进稳态**（单趟臂两块之差；同档 split 臂只漂 0.97%，其余档两臂都
   在 0.03~2%）→ 宽松那个锚点本身就不可信。
2. **改写成绝对线也不行**：曾议"退化 ≤2%"，但那个 2% 的唯一数值输入（单次发射 3~6 µs）
   是**拍的、无出处**，而 `AGENTS.md` §7 要求每个阈值都能回答"凭什么"。
3. **两把尺子对同一笔代价差 1.97× 且未解释**：PP-1 kernel 级推出 **+0.95%**
   （(0.025764−0.023600)×12/2.724），PP-2 端到端实测 **+1.877%**。在没被解释的量上画阈值
   等于把问题埋起来。

**故降级为观测项**（照 PF-8 / PF-9 的先例只报数），保留的记录是：短上下文（ctx≈20）退化
**+1.877%**，机制 = 每层多一次归并 kernel 发射（**D2 在拿到数据之前就预估并接受了这笔代价**）。
代码相应地打印 `max`/`min` 绝对锚点 + 配对差，并**不做任何自动判定**。
完整账见 `TROUBLESHOOTING` #45 / **#45.1**；若将来要恢复成硬判据，路径是**先量后定**
（把单次发射开销单独校出来，使 `12×launch/步长` 成为推导），**不要回头捡那个 2%**。

**沙箱实测（2026-09-27）**：`ctest --test-dir build` → **257 条 / 0 失败**
（原 242 + 新增 15：H 组 8 条实跑通过、GPU 组 7 条显式跳过）。
**注意**：新用例的 ctest 序号会随用例数继续漂移——文档只记用例名。

**护栏自证（`PROGRESS.md` §2.13"护栏必须有用例证明它会拦人"）**：把
`PagedAttentionSplitRange` 的"余数摊给前几片"临时改成"不摊"，4 条 host 用例
（PS-1 / PS-2 / PS-3 / PS-8）**当场转红**，复原后 8/8 绿——证明这批 host 用例不是
"注释里的祈使句"。**这一步只证明 host 侧护栏有效**；kernel 的 CUDA 取址仍必须靠真机的
PG-1~PG-7（沙箱无法执行）。

</details>

### 12.5 测量口径（沿用 §11.3，不新起协议）

- **kernel 级（主）**：在 `tests/` 里用合成数据（GPT-2 形状：`heads=12, head_size=64,
  block_size=16, batch=1`，`context_len ∈ {32, 256, 1024}`）直接调单趟与 split 两个入口，
  **同二进制、同轮交替（ABBA）**，每档取"发射 1 次 / 发射 N 次"的**斜率**扣掉每窗口固定开销；
- **端到端（从）**：复用 PF-9 的 `Gpt2DecodePerf.ContextLengthSweep`，用 override 切两版，
  **同一 session 内交替**；**沿用同一条** `..._ctxsweep_{prefill,decode}.engine` 路径，
  **不新增路径**（否则重踩 §11.4.1 那个"两用例交替判对方过期、来回重建"的坑）；
- 报数一律：**中位数 + p25/p75**、n、温度 / 时钟、引擎 `cache hit/stale`（§11.3 D）；
- **判别下限**：整步 decode 量级**不能**套采样器的 ±400~600 µs（`TROUBLESHOOTING` #39 的教训）
  ——同 session 漂移已见 10~18%，跨 session 见过 ±27%。

### 12.6 验收判据（每条都要能回答"凭什么"）
> **判据的唯一出处 = 测试计划**（`docs/future_iterations_test_plan.md` §11.3）：本节保留的是**设计侧视角**，与测试计划重复的行**以测试计划为准**。

| # | 判据 | 值 / 形式 | 凭什么 |
|---|---|---|---|
| 1 | 正确性（主） | 生产路径对 CPU **double** 参考仍满足 **rel < 1e-4 / abs < 1e-5** | 沿用 `test_paged_attention_plugin.cpp` 的既有算子口径（**出处待作者确认后补注**，见 §12.1 反向查）。**本轮不放宽** |
| 2 | 边界覆盖 | MHA / GQA / MQA、`batch>1`、跨多块、`context_len=0`、带当前 token、`total_len < num_splits`（空分片）**每项有用例** | `PROGRESS.md` §2.12（部分写入路径要显式处理）+ §2.13（带 batch 的算子必须覆盖 `batch>1`） |
| 3 | 语义不变（硬） | GPT-2 decode 一致性用例通过 + 8-token 贪心输出与既有基线**逐 token 一致** | `PROGRESS.md` §3.0a 的冻结基线 |
| 4 | 新旧差异（诊断，非判据） | 打印 `max_abs` / `max_rel`；**先量"与正确性无关的差异"**（float32 累加顺序不同）再谈阈值 | `AGENTS.md` §7"放宽阈值前必须先量无关差异"；观测值若高几个数量级 → **唯一的动作是查** |
| 5 | 性能（**F1=A，已拍板**） | ctx≈960 档**每步 decode 斜率**相对基线**至少降 40%**（≈1.67×）；ctx≈20 档退化不超过**同 session 实测漂移**。**前提**：PP-2 必须**同时打印它自己的重复测量漂移**（否则"不超过漂移"没有数值锚点） | 阈值出处 = 作者 2026-09-27 拍板（§12.9 F1=A）；40% 相对"并行度 12 块 → 最多 96 块"的理论天花板取值，属**工程取定**而非推导（已如实标注）；漂移参考 = PF-3 同 session 实测 10~18% |
| 6 | 缓存安全 | bump 图版本后第一次全部 `stale` 重建、第二次 `cache hit` | `PROGRESS.md` §3.0f 的既有观察法 |
| 7 | 无回归 | 沙箱全量 0 失败；真机全量仍**只有 1 红**（GPT-2 FP16 NaN，按设计） | `PROGRESS.md` §3.5 的既有口径 |

### 12.7 风险与回退


<details><summary>展开：12.7 风险与回退 全文</summary>

| 风险 | 影响 | 缓解 / 回退 |
|---|---|---|

| **旧 `.engine` 记录的 workspace 是 0，新 kernel 却要往 workspace 写** | **越界写 → 非法访存**（与 `TROUBLESHOOTING` #18 同类，只在真机暴露） | **必须 bump `kEngineGraphVersion`**（§12.8 第 1 条）；保险起见同批 bump plugin version |
| 分片打破"顺序累加" | 与单趟版有微小数值差异 | 主判据仍是 vs double 参考（#1），不是 vs 旧 kernel；新旧差异按 #4 只作诊断 |
| 空分片 / 边界算错 | 某些 `total_len` 下结果错或出 NaN | 分片策略 host 可测（D4）+ 空分片显式哨兵（`m=-inf, l=0, acc=0`） |
| 多一次发射 | 短上下文小回归 | `num_splits==1` 走单趟、不发 stage-2（D2）；短上下文档纳入回归观察 |
| 动态形状下 `.max` 取错 | workspace 不足 | 用 `DynamicPluginTensorDesc.max`，把"为什么不是 `desc.dims`"写进注释；用例覆盖形变 |
| FP16 partial 精度 | FP16 下更易丢精度 | partial **一律 float**（与现 kernel 累积精度一致），FP16 只用于读写 |
| 收益低于判别下限（重踩 #38 的坑） | 白改一轮 | 先跑 P2_2-0 基线 + §12.5 同轮交替；**低于判别下限就不改**（`PROGRESS.md` §5.13b），登记为"未获支持" |
| 顺手做"warp-per-position" | 一次改两个变量 → 读数不可归因 | **本轮不做**；若 split-K 之后仍显瓶颈（每位置一次 `BlockReduceSum` + `__syncthreads`），**另起一轮** |

**回退方案**：split-K 全部落在 `paged_attention_plugin.cu` + 一个新头里，`enqueue` 只有一个
分支点 → 把 `num_splits` 选择退回常量 1（或恢复 `getWorkspaceSize` 返回 0 + 恢复图版本）
即回到现状；旧单趟 kernel 从未删除，因此回退不需要重写代码。

</details>

### 12.8 破坏性动作清单（**动手前一次性确认**；批准本节 ≠ 批准这些动作）

> 依据 `AGENTS.md` §0.2 / §0.5 / §0.7 与 `PROGRESS.md` §2.14 B：
> 计划批准的是**目标**，不是这批动作。

<details><summary>展开：12.8 破坏性动作清单（**动手前一次性确认**；批准本节 ≠ 批准这些动作） 全文</summary>

> **作者决定（2026-09-27）：下面 7 条全部批准**（原话："F1~F4 全按 A，破坏性清单全批"）。
> 第 6 条仍按"默认不删、靠指纹判 stale"执行——它批的是"必要时可以删"，不是"现在就删"。

1. **`src/core/builder.cpp`：`kEngineGraphVersion` 1 → 2** —— **安全必需**：workspace 需求
   从 0 变正数，复用旧引擎会让新 kernel 往 0 字节 workspace 里写。
   影响：所有已缓存引擎判 `stale`，**下次真机第一次重建全部引擎**（GPT-2 主引擎 623 / 709 MB、
   ctxsweep 627 / 475 MB → 分钟级）。
2. **`paged_attention_plugin.hpp`：`kPagedAttentionPluginVersion` `"1"` → `"2"`** ——
   让旧引擎反序列化直接失败（这是**想要**的安全网），代价是残留 `.engine` 必须重建。
3. **`paged_attention_kernel.hpp`：`PagedAttentionKernelArgs` 增字段** —— 改的是**公开结构体**，
   所有调用点（插件 + 测试）同批改。
4. **`getWorkspaceSize()`：0 → 正数**，并删掉"只用 static shared memory"那条注释。
5. **新增头文件 `paged_attention_split.hpp`** —— `mini_trt_llm/CMakeLists.txt` 用 `file(GLOB)`，
   **必须重跑 `cmake -B build ...`**，否则症状是链接期 `undefined reference to vtable`
   （`TROUBLESHOOTING` #24.2，已踩两次）。
6. **可能需要删 `/tmp/mini_trt_llm_gpt2_*.engine` 与 `..._ctxsweep_*.engine`** —— 只在
   指纹机制表现异常时才做；**默认不删**，靠指纹判 stale。
7. **不做**：不改 `AGENTS.md`、不动 `0_resnet18_onnx/` `1_gpt2_onnx/`、不改其它模块的图版本。

</details>

### 12.9 待拍板决策（F1~F4）

| 编号 | 决策 | 选项 A（推荐） | 选项 B | 影响 |
|---|---|---|---|---|
| **F1** | 性能判据 | **设"斜率至少下降 40%"为达标线**（ctx≈960 档） | 只报观测值、不设阈值 | A 让"做完没有"可判定，但阈值要作者认账；B 与 PF-8 / PF-9 一致，代价是"是否达标"永远悬着 |
| **F2** | 分片数策略 | **按上下文自适应**：`num_splits = clamp(ceil(total_len / kTargetChunk), 1, kMaxSplits)`，`kMaxSplits=8`、`kTargetChunk=128` | 固定 8；或做成可配置属性 | A 在短上下文不退化成两次发射；B 实现更简单但短上下文恒多一次发射 |
| **F3** | 是否保留旧单趟 kernel 作 A/B 入口 | **保留**（D5） | 直接替换、不留入口 | A 换来可复核的 A/B（`future_iterations.md` §9.2 先例）；B 更干净，但下次复核只能靠 git |
| **F4** | 是否顺带扩 batch（G2-3） | **不做**（范围收敛，见 §12.1） | 一起做 | A 风险可控、可单独验收；B 两变量混在一起，读数不可归因 |

**作者决定（2026-09-27）：F1~F4 全按选项 A** —— 性能达标线 = ctx≈960 斜率至少降 40%；
分片数按上下文自适应（`kMaxSplits=8` / `kTargetChunk=128`）；保留旧单趟 kernel 作 A/B 入口；
不扩 batch。**已进入 P2_2-2。**

### 12.10 真机执行清单（**已执行**：2026-09-27；命令保留供复跑。Agent 侧无 GPU）

```bash
# 0) 编译（新增文件后必须先 configure，见 §12.8 第 5 条）

<details><summary>展开：12.10 真机执行清单（**已执行**：2026-09-27；命令保留供复跑。Agent 侧无 GPU） 全文</summary>

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# 1) 沙箱可跑：分片策略 host 用例（Agent 自检）
ctest --test-dir build -R 'PagedAttentionSplit*'

# 2) 真机：算子单测（GPU 用例跳过即失败）
#    **`ctest -R` 收的是正则，不是 gtest 的过滤器**——写成 'A.*:B.*'（gtest 语法）
#    会一条都不匹配，ctest 打 "No tests were found!!!" 且**退出码仍是 0**
#    （2026-09-27 真机踩过，见 TROUBLESHOOTING #43）。要么用下面这种正则，
#    要么直接调二进制并用 --gtest_filter。
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure -R 'PagedAttention'
# 等价写法（更贴近意图，且能看到逐条 [ OK ]）：
# MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
#     --gtest_filter='PagedAttention*:Fp16PathTest.PagedAttention*'

# 3) 真机：端到端回归（图版本 bump 后第一次会重建引擎，分钟级，属预期）
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure \
    -R 'Gpt2DecodeConsistency|Gpt2Generate'

# 3b) 推荐：直接跑全量（一次把"有没有漏跑"这件事也验掉；过滤集最容易静默空跑）
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure

# 4) 真机：性能——kernel 级同 session A/B（PP-1，不建引擎，最快）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='PagedAttentionSplitPerf.*'

# 5) 真机：性能——端到端同 session A/B（PP-2，复用 ctxsweep 引擎；含 token 一致性打印）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='Gpt2DecodePerf.ContextLengthSweepSplitVsSinglePass'

# 6) 真机：PF-9 单点观测（跨 session，只作对照，不可与 5) 相减——见 #38）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='Gpt2DecodePerf.ContextLengthSweep'
```

- 真机跑全量仍要带 `MINI_TRT_REQUIRE_GPU=1`；
- **引擎重建是预期**（§12.8 第 1 条），不要当成故障；
- **每轮只改一个变量**；
- 回填要求：状态 + 实测值 + 出处（命令 / 日志）；排查过程写 `docs/TROUBLESHOOTING.md`。

---

</details>

## 13. [OI-INT8-PERCHANNEL-PLAN] `future_iterations.md` §1.5 P4-INT8-a：per-channel 整网退化根因（开发计划）

> 条目事实来源：`docs/future_iterations.md` §1.5（目标 / 做法 / 验收判据）。
> 用例安排：`docs/future_iterations_test_plan.md` §3。本节只写"怎么做"。
> 历史排查：`docs/TROUBLESHOOTING.md` #28 / #29 / #30 / #31（已否证 11 条假设）。

### 13.1 计划对账（AGENTS.md §5 第 0 步）

1. **有没有计划文档**：有。`future_iterations.md` §1.5 已写全目标 / 做法（3 步）/ 验收判据 /
   前置依赖 / 成本校准；`future_iterations_development_plan.md` §3.1 已给出"开工硬前置 +

<details><summary>展开：13.1 计划对账（AGENTS.md §5 第 0 步） 全文</summary>

   改动面 + 破坏性动作预告"。本轮**不新建** per-item 文件（§0 的文档归位规则；该规则 2026-10-01 起
   已被 `docs/dev/<feature>/` 取代，此处保留当时口径），执行细节落进本节。
2. **是否一致**：逐条对照后**全部对上**，只有两处**细化**（不是偏差）：
   - `future_iterations.md` §1.5 做法第 1 条写"与 torch **已折叠 BN** 的模型同点对拍"。本轮改成
     **用 ONNX 官方参考实现直接执行那张 Q/DQ 图**——仍是"同点对拍"，但**取消了"我去折 BN"这一步**。
     理由见 13.3 D1：上一轮 3 次真机往返里有 2 次就是耗在这个折叠上（#30.5）。
   - `future_iterations.md` §1.5 做法第 3 条要求覆盖"3 个下采样卷积 + GAP+fc 段"。本轮把 3 个 `1×1/s2` 下采样卷积
     （`layer2.0` / `layer3.0` / `layer4.0` 的 `downsample.0`）与 **GAP 输出**一并放进探针清单，
     fc 段由 GAP 输出 + logits 覆盖（13.4 的 P1_5-1）。
3. **偏差怎么处理**：先改文档再改代码——本节 13.3 与本节的探针清单就是改后的文档，代码随后。
4. **反向查**（文档与现状矛盾，当场修）：
   - 测试计划 §3 原表把 B1-1 的判据写成"差异落在 FP32 kernel 正常差异量级内"。这句话**没错，
     但缺"谁量、怎么量"**——写"量级内"却不说量级从哪来，就是来路不明的阈值（AGENTS.md §7）。
     本轮把它落地为：**噪声地板由 PT 臂当场量出**（13.3 D4），测试计划 §3 同步改写。
   - 开发计划 §3.1 预告的探针图路径是单数（`models/resnet18/resnet18_qdq_probe.onnx`），
     实际要**两臂各一份**（per-channel / per-tensor）才有 A/B → 以本节 13.4 的路径为准。
   - `future_iterations.md` §1.5 的现状表与 `TROUBLESHOOTING` #29.5 的"顺带否证"段（conv1 死通道
     scale 跨度 1e13、加 1/1024 下限后数值不变）**一致，无矛盾**，不回改。

</details>

### 13.2 目标与范围

**目标**（= `future_iterations.md` §1.5 的验收，二选一，不允许"下次再看"）：


<details><summary>展开：13.2 目标与范围 全文</summary>

1. 定位到**从第几层开始分叉**并说明机制（该机制还要能被算子级 / block 级最小复现解释）；
2. 或证明所有可查方向均已查空，逐条写明"为什么这条路不能再查"。

**本轮要回答的那一问**：已知——算子级（单卷积 / 最小残差 block）两臂等价，整网级 per-channel
明显更差（余量子集 54.5% vs 100%）；且**同一批 scale 的整网模拟两臂都是 100%**（#30.2）。
所以"per-channel 更差"不可能来自数学本身，只能来自 **TRT 的执行**。
于是判据落成一句可测的话：**TRT 的 per-channel 臂在第几层开始偏离"忠实执行同一张图"。**

**范围**：只诊断，不改产品行为（正式产物仍按实测选 per_tensor）；不调任何现有阈值；
不把本轮产生的新阈值复用到别的精度上。

**明确的非目标**：不去"修" TRT；不改 GPT-2 / PagedAttention 侧任何东西。

</details>

### 13.3 [OI-INT8-PERCHANNEL-DESIGN] 设计决策（D1~D6）

**D1 —— 数值标尺 = ONNX 官方参考实现（`onnx.reference.ReferenceEvaluator`），
不用"自己折 BN 的 torch 模型"。**

<details><summary>展开：13.3 [OI-INT8-PERCHANNEL-DESIGN] 设计决策（D1~D6） 全文</summary>


- 图里 BN **已经折进 Conv**（这份 ONNX 的 42 个张量里没有 running stats）。直接执行这张图，
  就**根本不存在"折叠"这一步** → 从源头消掉 #30.5 那类"探针自身的错"。
- 它是对本项目代码**完全独立**的第三方实现（对应 `PROGRESS.md` §2.13 的"参考必须唯一且独立"）：
  我们比的是"TRT 有没有照着 ONNX 语义跑"，不是"我重写的模拟对不对"。
- **已知的实现细节（必须写进工具注释与自检）**：本图是 opset **17**，而 `ReferenceEvaluator`
  只带 `DequantizeLinear` 的 **19 / 21** 实现 → 参考侧把**副本**的默认 opset 提到 21。
  语义不变的理由：本图用到的 Q/DQ 语义（int8、对称、`axis`、round-half-even）在 17/19/21 之间没有变化；
  这条由 `qdq_reference.py --self-test` 用一个最小 Q/DQ 图**逐位自证**，不靠这句话。

**D2 —— 探"量化前"的 float 张量。**

挂的是每个 Conv 的**原始输出**（`node.output[0]`，即它后面那对 Q/DQ 里 `QuantizeLinear` 的输入）。
**为什么**：量化台阶（本图 conv1 是 `0.0796`）会把 FP32 kernel 的正常差异（`1e-3` 量级）在桶边界
附近放大成 **±1 格**，噪声与待查信号同量级（#30.5 第 3 条 / #30.6）。量化前张量的量级就是 `1e-3`，
可直接判读。

**D3 —— "探到的是量化前"用格点占比自证，而不是再折一遍 BN。**

量化后的张量必然落在 `scale` 的整数格 `{k·s}` 上（占比 ≈100%）；量化前的不会（占比 ≈0%）。
两种情形相差约 100 个百分点 → 判据取中点 `0.5`，对阈值不敏感。
这条直接替代旧仪器的 BN 折叠自证，并额外抓一种仪器失效：**TRT 有可能把"量化后再反量化"的值
当成那个图输出交回来**（融合把 Q/DQ 提前了）。这种情况曲线会全程贴在 `1e-3` 以下、看起来"正常"，
只有格点判据能识破。

**D4 —— 噪声地板当场由 PT 臂量出，不预设绝对阈值。**

PT 臂（权重 per-tensor）是已知健康的对照（余量子集 100%）。它与参考的逐层差异就是
"TRT kernel vs ONNX 语义"的正常差异。于是：

```
noise_floor = max over layers( PT 臂的 max_abs )
diverged(L)  ⟺  PC 臂的 max_abs(L) > kDivergenceFactor × noise_floor
```

`kDivergenceFactor = 100` 的出处：实测的失败幅度是"让 `margin ≥ 5` 的样本翻类"，即 logits 被扰动
**O(1)**（#29.4），而正常逐层差异是 **1e-3** 量级（P4-2 实测 logits `9.5e-6`、逐层可达 `1e-3`）——
两者相差 3 个数量级，100× 仍低一个数量级，留足余量。**没有用任何"期望值"来定它**。

**D5 —— 探针清单 = 20 个 Conv 原始输出 + GAP 输出 + 契约输出 `output`。**

为什么这样就够：残差 `Add` 的输出是已探两个张量的线性组合、`Relu` 是逐元素裁剪，
**两者都能由已探张量推出**；再加输出面只会让 TRT 的融合进一步变形。`GlobalAveragePool`
的输出是 `#30.3` 第 2 条点名要覆盖的"GAP + fc 段"的入口。

**D6 —— 必须验证"探针图下退化仍然复现"，否则本轮结论无效。**

挂额外图输出会阻止/改变 TRT 的融合（尤其 `Conv → Quantize` 的尾融合）。所以同一轮必须再跑一次
**探针图下的 256 张余量子集一致率**：仍应看到 PC ≈ 54.5% / PT ≈ 100%（口径与正式产物一致）。
不复现 → 仪器改变了现象，下一轮退到"单点探针"（一次只挂一层）。

</details>

### 13.4 任务分解（P1_5-0 ~ P1_5-5）

| ID | 任务 | 交付物 | 层 | 依赖 |
|---|---|---|---|---|
| **P1_5-0** | **先修仪器**：参考工具 + 探针图变换各自带自检（缺一条就退出），并进 ctest | `mini_trt_llm/tools/validate/qdq_reference.py`（含 `--self-test`）、`mini_trt_llm/tools/convert/add_probe_outputs.py`（含 4 道图不变性护栏） | H | 无 |
| **P1_5-1** | 产出**两臂探针图**：`add_probe_outputs.py` 从既有 Q/DQ 产物**只追加图输出**、其余一字不改 | `models/resnet18/resnet18_qdq_probe_per_tensor.onnx`；per-channel 臂需先用 `quantize_resnet18.py --weight-scope per_channel` 产出 `..._per_channel.onnx`（**新路径，不覆盖正式产物**）再追加 | H | P1_5-0 |
| **P1_5-2** | 用参考实现把探针张量落盘（纯 CPU，不联网） | `/tmp/mini_trt_llm_int8_probe/{pc,pt}/` 下每个探针张量一份 `.f32.bin` + `probe_index.txt` + `meta.json`（含 onnx SHA256） | H | P1_5-1 |
| **P1_5-3** | 真机用例：确定性 + 逐层误差曲线 + 格点自证 + 复现对照 | `mini_trt_llm/tests/test_resnet18_int8_probe.cpp`（B1-1~B1-4） | G | P1_5-2 |
| **P1_5-4** | **真机执行**（作者，见 13.9） | `/tmp/mini_trt_llm_resnet18_qdq_probe_{pc,pt}.engine`、用例日志、`probe_report.json` | G | P1_5-3 |
| **P1_5-5** | 回填：结论（第 N 层 + 机制，或"查空"清单）写进 `TROUBLESHOOTING.md` 新节；同步 `future_iterations.md` §1.5 状态与测试计划 §3 的回填表 | 文档 | — | P1_5-4 |

**每轮只改一个变量**（`PROGRESS.md` §2.14 C）。本轮**不改任何产品代码**——若结论指向产品侧行为，
那是**下一轮**的事，且按 §0.5 需重新报破坏性动作清单。

### 13.5 测量口径（写死，免得两侧各写一份）

| 项 | 口径 |
|---|---|
| 输入 | `assets/legacy/resnet18_onnx/calib_data/`（500 张，**已归一化**）按文件名排序；前 8 张拼成一个 `batch=8` 张量 |
| 比较对象 | 同一张图：**TRT 引擎输出** vs **ONNX 参考实现输出**（逐张量、逐元素） |
| 差异 | `max_abs`（主）+ `max_rel`（辅，分母取 `max\|reference\|`，沿用 `tests/diff_stats.hpp` 的唯一定义） |
| 逐层曲线 | 按 ONNX 图中的拓扑序（= 用例里的 `probe_index.txt` 顺序）逐行打印，**不做单点比较**（`future_iterations.md` §1.5 做法第 2 条） |
| 确定性 | 同一引擎 + 同一输入连跑 2 次，**逐位相同**（先证确定性，再谈误差曲线） |
| 复现对照 | 与正式产物**同一套**分层口径：`margin ≥ 5` 的余量子集一致率、整体一致率 |

### 13.6 验收判据
> **判据的唯一出处 = 测试计划**（`docs/future_iterations_test_plan.md` 的 per-channel 用例节）：本节保留的是**设计侧视角**，与测试计划重复的行**以测试计划为准**。

判据的唯一来源是 `docs/future_iterations.md` + OI-INT8-PERCHANNEL 的"二选一"，可执行形式见测试计划 §3。
本节的硬约束只有两条：

1. **仪器不过不取数**：B1-1（格点自证）/ B1-2（确定性）任一不过 → 本轮作废，先修仪器；
2. **现象不复现不解释**：B1-4 不过（探针图下 PC 不再更差）→ 结论只能是"探针改变了现象"，
   不许拿 B1-3 的曲线去解释原来的退化。

### 13.7 风险与回退

| 风险 | 影响 | 缓解 / 回退 |
|---|---|---|
| 挂图输出改变 TRT 融合 → 现象不复现 | 曲线测的不是原来那个现象 | B1-4 是硬门；不复现就退"单点探针"（每轮只挂 1 层，多轮逼近） |
| TRT 交回的是"量化后再反量化"的值 | 曲线看起来"很干净"，实则测错对象 | D3 的格点判据（两臂都查） |
| `ReferenceEvaluator` 的 opset 21 副本与 opset 17 原文语义不一致 | 标尺失真 | 自检用最小 Q/DQ 图逐位比手算值（D1）；不靠"应该一样"这句话 |
| 参考在 CPU 上跑 256 张太慢 | 真机往返被拖长 | **参考只需跑前 8 张**（逐层曲线）；复现对照用的是**引擎之间的 A/B**（不需要参考） |
| 又把预算耗在仪器上 | 真机往返浪费 | P1_5-0 把仪器自检前置到 host 侧（沙箱即可跑）；真机只做"取数" |

### 13.8 破坏性动作清单（**动手前一次性列给作者**）

1. **新增** `mini_trt_llm/tools/convert/add_probe_outputs.py`；
2. **新增** `mini_trt_llm/tools/validate/qdq_reference.py`；
3. **新增** `mini_trt_llm/tests/test_resnet18_int8_probe.cpp`；
4. **修改** `mini_trt_llm/tests/CMakeLists.txt`（注册 `qdq_reference_selftest` 这一条 host ctest 项）；
5. **新增** `models/resnet18/resnet18_qdq_per_channel.onnx` 与两份探针图（**新文件，不覆盖正式产物**）；
6. **修改** 文档：本节、测试计划 §3、`future_iterations.md` §1.5 / §0.1 / §0.3、`PROGRESS.md`、
   `TROUBLESHOOTING.md`（新增节）；
7. **不删除**任何文件、**不碰** git、**不改** `.gitignore`、**不联网**；
8. 真机侧：`/tmp/mini_trt_llm_resnet18_qdq_probe_{pc,pt}.engine` 是**新路径**，不会顶掉正式引擎缓存；
   若指纹判定为 stale 而重建，属预期。

### 13.9 真机执行清单（**已执行**：2026-09-27 B1 四条真机全绿；命令保留供复跑。Agent 侧无 GPU，见 `PROGRESS.md` §5.10）

```bash
# 0) 构建（**新增了 .cpp，必须先重新 configure**：GLOB 只在 configure 时求值）

<details><summary>展开：13.9 真机执行清单（**已执行**：2026-09-27 B1 四条真机全绿；命令保留供复跑。Agent 侧无 GPU，见 `PROGRESS 全文</summary>

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j"$(nproc)"

# 0b) 沙箱即可跑：参考工具的自检（不需要 GPU / 不需要产物）
python3 mini_trt_llm/tools/validate/qdq_reference.py --self-test
ctest --test-dir build -R qdq_reference_selftest --output-on-failure

# 1) per-channel 臂的 Q/DQ 图（**新路径，不覆盖正式产物**；纯 CPU，不联网）
#    ⚠️ 这一步**故意**用默认的 `--weight-range-source torchvision`：产出的就是 #46 那个
#    "错源"样本，B1-4 要复现的正是它。它的身份 = **#46 的复现样本，不是候选基线**
#    （见 §13.11）。要拿 per-channel 做正确性对比/选型，必须另加 `--weight-range-source onnx`
#    并输出到**另一个路径**，别覆盖这一份。
python3 mini_trt_llm/tools/convert/quantize_resnet18.py \
    --onnx assets/legacy/resnet18_onnx/resnet18.onnx \
    --calib-dir assets/legacy/resnet18_onnx/calib_data \
    --weight-scope per_channel \
    --output models/resnet18/resnet18_qdq_per_channel.onnx

# 2) 两臂探针图（只追加图输出；秒级）
python3 mini_trt_llm/tools/convert/add_probe_outputs.py \
    --onnx models/resnet18/resnet18_qdq.onnx \
    --output models/resnet18/resnet18_qdq_probe_per_tensor.onnx
python3 mini_trt_llm/tools/convert/add_probe_outputs.py \
    --onnx models/resnet18/resnet18_qdq_per_channel.onnx \
    --output models/resnet18/resnet18_qdq_probe_per_channel.onnx

# 3) 参考落盘（纯 CPU；只跑前 8 张，秒级）
python3 mini_trt_llm/tools/validate/qdq_reference.py \
    --onnx models/resnet18/resnet18_qdq_probe_per_tensor.onnx \
    --calib-dir assets/legacy/resnet18_onnx/calib_data --num-images 8 \
    --output-dir /tmp/mini_trt_llm_int8_probe/pt
python3 mini_trt_llm/tools/validate/qdq_reference.py \
    --onnx models/resnet18/resnet18_qdq_probe_per_channel.onnx \
    --calib-dir assets/legacy/resnet18_onnx/calib_data --num-images 8 \
    --output-dir /tmp/mini_trt_llm_int8_probe/pc

# 4) 真机：B1 四条（**跳过即失败**）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='Int8Probe*'

# 5) 真机：全量（确认没有连带回归）
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure

# 6) 回填：把 B1-3 的逐层曲线 + B1-4 的两个一致率贴回测试计划 §3 与本文件 §13.10
```

- `ctest -R` 收的是**正则**，不是 gtest 过滤器（`TROUBLESHOOTING` #43）；
  上面第 4 步直接用二进制，就是为了避开这个坑；
- 每个引擎都是**分钟级**（第一次必然重建）；
- **每轮只改一个变量**；
- 回填要求：状态 + 实测值 + 出处（命令 / 日志）；排查过程写 `docs/TROUBLESHOOTING.md`。

</details>

### 13.10 结果回填（已回填）

> 回填要求：只写"状态 + 实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md` + TS-046。
> **本节已于 2026-09-27 全部回填**（B1 四条真机全绿）；按纪律，**没有实测过的行不许写"通过"**——

<details><summary>展开：13.10 结果回填（已回填） 全文</summary>

> 若将来复跑出别的结果，改的是这些行，不是判据。

| 项 | 状态 | 实测值 / 出处 |
|---|---|---|
| P1_5-0 仪器自检（host） | ✅ **沙箱通过**（2026-09-27） | `qdq_reference.py --self-test`：最小 Q/DQ 图逐位比手算 ONNX 语义（**差 0**）、per-channel/per-tensor 可分辨、index/meta 格式、配对查找（自检当场抓出"用 producer 当 consumer"的错）。`add_probe_outputs.py --self-test`：探针集 = Conv 输出 + GAP 输出、图其余部分逐字节不变、二次追加与非目标图都被拒。两者已进 ctest：`qdq_reference_selftest` / `add_probe_outputs_selftest`（**Passed**） |
| P1_5-1 两臂探针图（host） | ✅ **沙箱产出**（2026-09-27） | 各 **21** 个探针输出（20 个 Conv 量化前输出 + GAP 输出），加契约输出 `output` 共 22 个引擎输出；**node / initializer / input / opset 逐字节未变**（脚本自证 + 独立复核：与产物图的 node/initializer/output 三份序列化逐字节相同） |
| P1_5-2 参考落盘（host） | ✅ **沙箱产出**（2026-09-27） | 每臂 42 个张量 = 引擎输出 22 + 量化后（参考独有）20；`probe_index.txt` 带 role/paired 两列；`--num-images 8` 约 **7 s**/臂 |
| P1_5-3 用例编译（host） | ✅ **沙箱可编译并注册**（2026-09-27） | `cmake --build build` 通过；沙箱 `ctest` **264 条 / 0 失败**（原 259 + 本轮 5：2 条 host 自检 + 3 条 GPU 用例，GPU 在沙箱显式跳过） |
| **根因（离线，本机 CPU）** | ✅ **已定位并反证**（2026-09-27） | 权重 scale 取自**未折 BN** 的 torchvision 权重、量化对象是**已折 BN** 的 ONNX 权重（折叠系数逐通道 0.05~19.9）。ONNX 官方参考实现、64 张真实图：**PT 60.9%/100%、PC(错源) 25.0%/54.5%、PC(改源) 57.8%/100%**；PC(错源) 与 #29.2/#29.5 记的真机 TRT 数字**逐位相同**。逐层曲线：分叉**从 conv1 就开始**，在 `layer4.1.conv2`（折叠跨度最大）放大到 16.4。完整记录见 `TROUBLESHOOTING.md` #46 |
| 默认产物是否被改动 | ✅ **未改动** | 用默认参数重新生成到临时路径，与 `models/resnet18/resnet18_qdq.onnx` 的 node / initializer / output **逐字节相同** |
| **首跑（2026-09-27，用户真机）** | 🟡 **3 条全红，但红在用例自身的绑定**（已修，待复跑） | `RunProbeEngine` 用 `ICudaEngine::getTensorShape` 取形状——**引擎上动态维是 -1**，`size_t` 一转就是天文数字 → `显存分配失败：input`。改用 `IExecutionContext::getTensorShape` 并加"任何维 ≤ 0 即报错"的校验。附带观察到**探针图确实改了 tactic**（产物图 44 层/38 Int8/4 个 i8i8 → 探针图 78 层/74 Int8/**0 个 i8i8**），已加逐层 ONELINE 落盘。见 `TROUBLESHOOTING.md` **#47.1 / #47.2** |
| **文件级证据（离线）** | ✅ **已取得**（2026-09-27） | 读 ONNX 里的 int8 权重常量、数被 clamp 到 ±127 的比例：**PT 3.919% / PC(错源) 16.188% / PC(改源) 0.044%**（对称量化下"每通道约 1 个"才对）。**坏值已经烘进文件** → 任何忠实后端都会复现，**与 TRT 的 tactic 选择无关**（正好补上 #47.2 那个风险）。见 #47.3 |
| B1-1 探针自证 | ✅ **真机通过**（2026-09-27） | `conv1`：PT `d_pre=8.34e-07` vs `d_post=0.0398`；PC `d_pre=1.43e-06` vs `d_post=0.0398` → 探到的是**量化前**张量（差 4.7 个数量级） |
| B1-2 确定性 | ✅ **真机通过**（2026-09-27） | 22 个张量两次运行**逐位相同**（704 ms，引擎 cache hit） |
| B1-3 逐层曲线 | ✅ **真机通过**（2026-09-27） | 引擎 vs **自己的图**：逐层最大 `max_abs` PT **0.2714**（`layer4.1.conv2`，相对 2.3%）、PC **0.168**；`conv1` 仅差 **8.3e-07**。两臂都**未**越过 `100 × 噪声地板(0.2714)` → **两个引擎都忠实执行了各自的图**（判读已写进用例输出） |
| B1-4 复现对照 | ✅ **真机通过（硬门）**（2026-09-27） | 256 张、探针图：整体 PT 99 / PC 26；**余量子集（n=12）PT 12/12 = 100%、PC 6/12 = 50%** —— 与正式产物口径（PT 100% / PC 54.5%，同为"6 张对上"）一致 → 探针图是现象的有效模型（尽管它把 `i8i8` 从 4 变成 0，见 #47.2） |
| **`future_iterations.md` §1.5 验收（二选一）** | ✅ **走分支一：第 0 层 + 机制明确** | 机制 = 权重 scale 取自**未折 BN** 的权重、量化对象是**已折 BN** 的权重；三条独立证据互咬（引擎忠实 / 探针探对对象 / 现象复现）+ 文件级饱和统计 + 改源后 100% |

---

</details>

### 13.11 [OI-INT8-PERCHANNEL-ARTIFACTS] 产物身份与两件"暂不做"的事项（2026-09-27 作者指示：只钉身份，不动产物）

`future_iterations.md` §1.5 结案后留下两个**都不是默认行为**的选择。作者 2026-09-27 决定：**这一轮只把产物的"身份"
钉死，两件都暂不做**。本节把"为什么暂不做"和"将来要做时该连什么一起做"写清楚——否则下个会话

<details><summary>展开：13.11 [OI-INT8-PERCHANNEL-ARTIFACTS] 产物身份与两件"暂 全文</summary>

看到盘上那份 per-channel 图，很可能把它当成"per-channel 的基线"去用。

**产物身份（钉死）**：

| 产物 | 身份 | 依据 |
|---|---|---|
| `models/resnet18/resnet18_qdq.onnx` | **正式产物**（per_tensor + 默认 `torchvision` 源） | `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §4/§7；默认路径逐字节可复现（§13.10） |
| `models/resnet18/resnet18_qdq_per_channel.onnx` + 其探针图 | **`TROUBLESHOOTING.md` #46 的复现样本，不是候选基线** | 它按"错源"生成：权重 scale 取自**未折 BN** 的权重，16.19% 的 int8 权重被 clamp 饱和（#47.3）。**B1-4 的红是设计**——它要的就是"PC 比 PT 差" |
| `/tmp/resnet18_qdq_per_channel_fixed.onnx`（仅 /tmp，未入 `models/`） | 离线实验件（改源后的 per-channel 图，已验证余量子集 100%） | #46.2 第 5 步 |

**两件暂不做的事项**：

| # | 事项 | 收益 | 真实成本（为什么现在不做） |
|---|---|---|---|
| ① | 把 `--weight-range-source` **默认**切到 `onnx` | 语义更正确：scale 与"被量化的张量"一致。实测 per-tensor 的饱和权重 **3.919% → 0.000%（20 个，每张权重恰好 1 个 = 教科书形态）** | **换了默认就等于换了正式产物** → `docs/dev/REQ-008-int8-qdq/phase4_int8_plan.md` §4 的判据行（整体 37.9% / 余量子集 12/12）、`PROGRESS.md` §3.0d、`phase4_test_plan` R2.6、C 批交叉校验的 n 与分子、以及 `.meta.json` 里的裁剪值/预设**全部要真机重测回填**（§5 计划对账纪律）。而**"更准"的证据不足**：64 张、余量子集仅 11 张的图上，改源前后判据与一致率**完全一样**（60.9% / 100%，`max_abs` 21.736 → 21.556）。也就是说换默认的理由只能是"**更对**"，不能是"更准"——而下这个判断**不需要**换默认，显式传参即可 |
| ② | 重生成 per-channel 产物（改源） | 不在盘上留"已知错误"的文件 | **零功能收益**：默认路径上没有任何代码/测试/工具读它（`ResNet18Int8*` 读的是 per_tensor 那份）。**但它是 B1-4 的承重件**——重生成后现象消失，**B1-4 会立刻变红**。所以它不是"一条命令"，而要打包三件事：㈠ 重生成 per-channel 图与探针图；㈡ **退役或改写 B1-4**（从"断言 PC 更差"改成"两臂都对 FP32 全一致"之类；按 §7 这属于"证明期望值本身错"，允许，但**必须把依据写下来**）；㈢ 可选：把坏的那份**钉成回归夹具**（13 MB，唯一用途是复现历史 bug） |

**触发条件（将来要做时从这里接）**：

1. **先有 `future_iterations.md` §1.6 的验收集**（带真值标签、样本量够）。当前那批判别力不足的图**得不出**"per-channel
   和 per-tensor 谁更好"，而这正是①②两个决定共同缺的那块证据。
2. 然后**一次**把"重生成 per-channel + 退役/改写 B1-4 + 重新评估粒度"打包做（细粒度上
   per-channel 理论上不会更差，但要不要切是另一件事，需要新证据）；
3. 若最终决定切默认源，再单独做 ①＋全套数值回填（那是独立的一轮）。

**这一轮实际做的只有一件事**：把身份写进本文档、`PROGRESS.md` §3.0j 的产物表与
`quantize_resnet18.py` 的告警（生成"错源"产物时直接打印"本产物的身份是 #46 的复现样本，
不是候选基线"）。**没有重生成任何产物、没有改任何默认值。**

---

*本计划只含执行安排与纪律；各条目目标与触发条件的唯一来源仍是 `docs/future_iterations.md`。*

</details>
