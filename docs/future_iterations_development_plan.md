# future_iterations 开发计划（触发驱动的分批执行计划）

> **状态**：2026-09-26 产出，**未开工**。本文件是 `docs/future_iterations.md` 的**执行层**。
>
> **分工（避免两处来源漂移）**：
> - **排序 / 级别定义 / 触发条件 / 目标 / 做法 / 验收判据** → `docs/future_iterations.md`
>   （§0 优先级、§1.5 P4-INT8-a、§1.6 P4-INT8-b、§11 缺口索引）；
> - **本文件只回答**：轮到某条时"改哪些文件、分几步、每步怎么自检、哪一步要你批"。
>   条目内部的做法与验收**不复制过来**——复制出来的第二份必然漂移。
>
> 配套测试计划：`docs/future_iterations_test_plan.md`（用例 → 判据 → 出处 → 环境 → 状态）。

---

## 0. 计划对账（AGENTS.md §5 第 0 步）

### 0.1 有没有计划文档

**有。** 能力条目与触发条件在 `docs/future_iterations.md`（2026-09-26 已重定优先级，含 §0 总表）；
本文件是它的执行层，测试侧在 `docs/future_iterations_test_plan.md`。
两条**已立项**条目（P4-INT8-a → §1.5、P4-INT8-b → §1.6）的目标 / 做法 / 验收已在该文件里写全。

### 0.2 逐条对照：本文件的任务 / 接口 / 验收 vs 现状

| 本次要做的 | 现状 | 是否一致 |
|---|---|---|
| 把 31 个条目按触发条件分批 | `future_iterations.md` §0.1 已给出级别与理由 | 一致，本文件只做"批次化"（同级内按依赖排序） |
| 给"可立即开工"的条目写可执行步骤 | §5.1 与 §1.6 离线子项此前只有"工作内容"级描述 | **有偏差**：缺文件级改动面、步序、自检点 → 本文件 §2 补齐 |
| 给"触发即做"的条目定开工前提 | §1.5 已有做法与成本校准 | 一致；本文件补"先修仪器"的硬前置与真机往返预算 |
| 给"需外部前置"的条目定第一步 | §0.1 只有一行备注 | **有意保持粗粒度**：触发后各条按 AGENTS.md §5 升格为独立 phase 计划，本文件不预写细节（预写必然过期） |
| 测试侧判据 | 各条目只写了主判据，没有用例清单与分层 | **有偏差** → `future_iterations_test_plan.md` |

### 0.3 偏差怎么处理

先补文档（本文件 + 测试计划），再动手；条目内部做法仍以 `future_iterations.md` 为准，本文件不覆盖它。
任何一次实际开工前，重新跑一次本节的对照（现状会变，尤其"前置依赖是否还在"）。

### 0.4 反向查：文档与现状矛盾之处（当场修）

- **【已修，2026-09-26】我上一轮把 §5.1 的数据前置写错了。** `future_iterations.md` §0.1 原写
  "`vocab.json` + `merges.txt` 本地没有 → 缺文件时本条降为 P2"。查证后是**我错**：
  本地 HF 缓存里有 gpt2 的完整 tokenizer 文件——
  `~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/{vocab.json,merges.txt,tokenizer.json}`，
  且实测 `transformers 4.44.0` 能 `local_files_only=True` 离线加载并给出参考 token id
  （`"The quick brown fox"` → `[464, 2068, 7586, 21831]`，`vocab_size = 50257`）。
  **结论：§5.1 保持 P1，不需要联网。** 我当时的判断只查了 `models/gpt2/`，把"仓库里没有"
  当成了"机器上没有"——这正是 `AGENTS.md` §7 禁止的"观测缺口当成现象"。
- **【已修，2026-09-26，同一次自检发现】** `future_iterations.md` §5.1 原先没写"byte-level BPE 需要
  **两个**文件"这个接口事实，而 `BaseTokenizer::Load` 只有一个路径参数（见 §2.1 的 F2）。
  已在该条目补"接口事实"一行（入参语义 = 目录，不改基类）。这不是矛盾，是缺口——
  但它属于"下个会话照文档做就一定会卡住"的那类缺口，所以当场补而不是留给执行时。

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
| **A** | 无外部前置，且触发已成立 | §5.1（BPE Tokenizer）、§1.6 的离线子项 | **能**（见 §2） |
| **B** | 无外部前置，但触发未成立（触发即做） | §1.5（P4-INT8-a） | 触发后能（见 §3） |
| **C** | 有外部前置（联网 / 新硬件 / 先补测量 / 需求未定） | §1.2、§1.6 整条、§2.1、§2.3、§2.4、§3.1、§3.2、§4.1、§4.2、§6.1~§6.4、§9.2、§9.3、§10.1 | 不能（见 §4） |
| **D** | 冻结备查 | §1.1、§9.1 | **不做**（见 §5） |

**批次规则**：一批一次收口——每批结束时跑真机全量、回填文档、把新增的"坑"写进
`docs/TROUBLESHOOTING.md`（结论留 `PROGRESS.md`）。批次内可以并行，批次间不并行，
以便"一次真机往返只验一批"（`AGENTS.md` §7"每轮只改一个变量"的同一精神）。

---

## 2. 批次 A：现在就能开工

### 2.1 A1 = BPE Tokenizer（`future_iterations.md` §5.1，P1）

**目标（一句话）**：让框架具备"文本进 / 文本出"的 GPT-2 路径——目前 `LLMRunner` 只收发 token id，
`tokenizer_` 仅在构造时校验（`include/mini_trt_llm/core/llm_runner.hpp:16`）。

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

**明确不做**：`tiktoken`（§5.2）、chat template（§5.3）、tokenizer 级批处理、性能优化。

**破坏性动作**：**无删除、无覆盖既有文件**。实际动到的既有文件有三个，全是**追加**：
`tests/CMakeLists.txt`（注册 ctest 项）、`tests/test_gpt2_generate.cpp`（加一条桥接用例）、
`include/mini_trt_llm/utils/json.hpp`（补 `\uXXXX` 转义——原先该分支直接报错，属"打开以前走不通的路径"）。
**已执行（2026-09-26）**，其余全部是新增文件。

### 2.2 A2 = INT8 判据的离线口径定义（`future_iterations.md` §1.6 的离线子项，P1）

**目标（一句话）**：把"INT8 怎么算合格"写成可执行的判据规格 + 脚本，**不下载任何数据**。

**前置**：无。现有资产：`models/resnet18/*.meta.json`（已有 `sha256` 字段体例）、
`scripts/ref_resnet18.py`（基线生成）、`tests/test_resnet18_int8.cpp`（余量分层统计的 C++ 现役实现）。

**改动面（文件级）**：

| 文件 | 动作 |
|---|---|
| `mini_trt_llm/tools/validate/int8_eval.py` | 新增：读 logits + meta → 输出**分层报告**（每层：率 + 样本量 n） |
| `mini_trt_llm/tools/validate/README.md` | 新增：验收集规格（来源 / 版本 / SHA256 / 与 `calib_data` 的重叠排除规则） |
| `docs/phase4_int8_plan.md` §4 | 改：把新规格接进判据表（**先改文档**） |
| `docs/phase4_test_plan.md` R2.6 | 改：判据出处指向新规格 |
| `mini_trt_llm/tests/CMakeLists.txt` | 加：脚本自检项（`--self-test`，缺 Python/资产返回 77） |

**步骤**：

1. **A2-1 写规格**：验收集"必须有什么字段才能被引用"（来源 / 版本 / SHA256 / 标签 / 与标定集的重叠排除）。
2. **A2-2 写分层统计**：口径与 C++ 现役实现**必须一致**（同一 `margin` 定义、同一分桶边界来源）；
   两者不一致时以实测交叉校验为准（测试计划 A2-5）。
3. **A2-3 加护栏自检**：脚本对下面 **7 类**输入必须**拒绝**，每种都要有一份故意改坏的输入把它打出来
   （`AGENTS.md` §2.13「护栏必须有用例证明它会拦人」）：
   ① 缺 provenance（`manifest_sha256` 缺失）；② 验收集与标定集重叠（文件名或 `sha256` 命中）；
   ③ 字段类型错（`num_samples` 是字符串）；④ 清单被改过（`sha256` 不符）；
   ⑤ logits 大小与 `shape` 不符；⑥ 报告只报率不报 n；⑦ 率与分子分母不自洽。
   **已实现并全部通过**（2026-09-26，`--self-test`，沙箱可跑、不联网）。
4. **A2-4 不下载**：规格里可以写"将来到哪里取数据"，但本轮**不执行**任何下载。

**明确不做**：下载验收集、定绝对误差阈值（那需要真数据，见 §4 的 C 组）。

**破坏性动作**：改动 `docs/phase4_int8_plan.md` §4 与 `docs/phase4_test_plan.md` R2.6 的**判据表文字**
（不删条目、不改判据本身）——按 `AGENTS.md` §0.5，动手前把这两处改动一次性列给你确认。
**已执行（2026-09-26 获你批准）**：两处均为**追加**指向新规格与脚本，未改动任何既有判据；
另按同一批批准把 `int8_eval_selftest` 注册进 `tests/CMakeLists.txt`。

---

## 3. 批次 B：触发即做

### 3.1 B1 = P4-INT8-a（`future_iterations.md` §1.5）

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
| `docs/future_iterations.md` §1.5 | 回填状态 |

**真机预算**：每轮 1 次往返；本轮自带仪器自证，目标 ≤3 轮。
**每轮只改一个变量**（`PROGRESS.md` §2.14 C）。

**破坏性动作（预告，执行前逐条确认）**：

1. 修改 `quantize_resnet18.py`（**脚本**，不是配置文件）；
2. 新增 `models/resnet18/resnet18_qdq_probe.onnx`（新文件，不覆盖）；
3. **若**需要重新生成正式 `resnet18_qdq.onnx` 或覆盖它 → **单独确认**；
4. 删除 `/tmp/mini_trt_llm_resnet18_*.engine` 以强制重建（缓存只按路径名区分）。

---

## 4. 批次 C：需要外部前置（只到"预备"深度）

**共同纪律**：触发后**先产独立 phase 计划**（`AGENTS.md` §5 第 0 步），再动手；
本表只写"前置是什么 + 触发后的第一步"，不预写做法细节（预写必然过期，且会与将来那份 phase 计划形成两份来源）。

| 条目 | 前置是什么 | 触发后的第一步 | 要联网吗 |
|---|---|---|---|
| §1.6 整条（INT8 验收集） | 带真值标签的验收集 | **先获批**，再下载并登记 SHA256 + 重叠排除（规格来自 A2） | **要** |
| §1.2 LLM INT8 / INT4 | `PagedAttentionPlugin` 只支持 `kFLOAT`/`kHALF`（`paged_attention_plugin.cu:265`）→ 先扩 INT8 KV cache；INT4 需先评估 sm_75 kernel | 先做"PagedAttention INT8 KV cache"的设计 + 契约用例 | 否 |
| §2.1 显存池 | 无前置，但**缺测量** | 先量一次分配开销（按 `G6` 口径），再决定是否实现 | 否 |
| §2.3 Continuous Batching | 与 `G2-3`（`batch = 1` 限定）耦合 | 先扩 `LLMRunner` 的 batch（含多序列 block 分配与 `context_lens`） | 否 |
| §2.4 CV 动态分辨率 | 无前置，但会牵动 profile 与缓存 | 先改 `AddCvOptimizationProfile` 的接受/拒绝语义（host 可测） | 否 |
| §3.1 Encoder-Decoder / §3.2 ViT | 需要目标模型（HF 权重 + 导出） | 先定模型与 `config.json` 契约 | 可能（取权重） |
| §4.1 GroupNorm / InstanceNorm、§4.2 SiLU / SwiGLU | 需要用到它们的模型（CV / LLaMA 系列） | 先接模型，再写 Plugin（`LayerNorm` 一半已作废，见 §5） | 可能 |
| §6.1 转换工具增强 | 出现新权重来源 | 先定"要支持谁"，再动脚本 | 可能 |
| §6.2 ONNX custom op | 真做子图替换（§10.2） | 先定替换目标子图 | 否 |
| §6.3 Nsight 一键 target | 需要按 `G6` 口径做可复现测量 | 先定测量协议（≥3 次构建 / ≥20 次推理、报中位数与极差） | 否 |
| §6.4 CI | 无前置（本地脚本部分），GitHub Actions 属联网 | 先做本地 `ctest` 脚本，联网部分**先获批** | 部分 |
| §9.2 Sampler 高性能 kernel | **profile 从未做过** | 先做一次 decode 性能 profile，确认 sampler 占比 | 否 |
| §9.3 采样器参考数据固化 | 无前置 | 把 `scripts/ref_sampler.py` 输出落成 `.bin` 供 C++ 载入 | 否 |
| §10.1 ONNX / 原生 I/O 契约统一 | 要让 ONNX 路径接进 `LLMRunner` | 先加 `Cast`（把 `input_ids` 降到 INT32）并补契约用例 | 否 |

---

## 5. 批次 D：冻结备查

| 条目 | 冻结原因 | 解冻条件 |
|---|---|---|
| §1.1 ResNet18 INT8 校准（implicit calibration / `IInt8Calibrator`） | 已被 Q/DQ 显式量化取代；该 API 路线自 TRT 10.12 起弃用 | 若将来某个模型在 Q/DQ 路线上**确实做不到**（例如算子不支持 `QuantizeLinear`），再评估 |
| §9.1 PagedAttention 的 Prefill 阶段 | 触发条件（"Phase 2 走单引擎"）已被现状否证：`LLMRunner` 收 prefill + decode 两个引擎（`llm_runner.hpp:56`） | 真要合成"单引擎含 Prefill"时 |

**冻结 ≠ 删除**：两个条目及其历史理由都保留（`AGENTS.md` §0.5 / §0.6）。

---

## 6. 全局开工纪律（每批都适用）

1. **四步走**：计划对账 → 破坏性动作**一次性清单**确认 → 真机全量回归 → 回填文档。
2. **新增源文件后重跑 configure**：`mini_trt_llm/CMakeLists.txt` 与 `tests/CMakeLists.txt` 都用
   `file(GLOB ...)`，GLOB 只在 configure 时求值；漏跑的症状是链接期 `undefined reference to vtable`。
3. **产物或建图变了先删引擎缓存**：`/tmp/mini_trt_llm_*.engine` 只按路径名区分，不随代码失效。
4. **真机口径固定**：`MINI_TRT_REQUIRE_GPU=1`（否则 GPU 用例静默跳过，等于白跑）。
   当前基线：**沙箱 215 条 / 0 失败**（**89 条跳过、126 条实际执行**，含新增的脚本项与 host 用例）；
   真机上次实测 182 条 / 0 跳过 / 1 红（FP16 NaN 复现器，按设计红），加入新用例后应为 204，
   **待真机复验**（`AGENTS.md` §7：未跑过的不写"通过"）。
5. **跳过或失败都要显式**：`MINI_TRT_SKIP_IF_NO_CUDA` / ctest 的 77；缺资产 → 跳过并打印探测结果。
6. **不擅自删旧模块**：`0_resnet18_onnx/`、`1_gpt2_onnx/` 归作者（Phase 5 已永久取消）。
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

## 8. 真机执行清单（**待作者执行**；Agent 侧无 GPU，见 `PROGRESS.md` §5.10）

> 工作区 = 仓库根目录。先构建：`cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j$(nproc)`。
> **顺序不能反**：§8.1（全量）→ §8.2（文本端到端）→ §8.3（INT8 交叉校验）。

### 8.0 前置检查（30 秒，缺一项后面的用例会**跳过**而不是失败）

```bash
ls ~/.cache/huggingface/hub/models--gpt2/snapshots/*/vocab.json   # 缺 → export MINI_TRT_GPT2_TOKENIZER_DIR=<你的 HF gpt2 目录>
ls models/gpt2/config.json models/gpt2/model.safetensors
ls models/resnet18/resnet18_qdq.onnx models/resnet18/config.json
ls 0_resnet18_onnx/calib_data | head -3
```

### 8.1 真机全量（改动面：无）

```bash
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build --output-on-failure
```

- **期望**：**204 条**，其中：
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
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
    --gtest_filter='*DumpsLogitsAndCppReportForCrossCheck*'

# C-2 用 Python 侧实现算同一批数据（用例输出里会打印等价命令）
python3 mini_trt_llm/tools/validate/int8_eval.py \
  --fp32-logits /tmp/mini_trt_llm_int8_crosscheck/fp32.f32.bin \
  --int8-logits /tmp/mini_trt_llm_int8_crosscheck/int8.f32.bin \
  --meta /tmp/mini_trt_llm_int8_crosscheck/meta.json \
  --calib-dir 0_resnet18_onnx/calib_data --legacy-mode \
  --json-out /tmp/mini_trt_llm_int8_crosscheck/py_report.json

# C-3 比对两侧口径（不一致就是真问题）
ctest --test-dir build -R int8_crosscheck --output-on-failure
```

- **期望**：C-3 PASSED，并打印"整体 a/b == 整体 a/b、余量子集 12/12 == 12/12、逐桶一致"。
- **C 为什么用 `--legacy-mode`**：当前的测试图与标定集**同源**（都用 `calib_data`），严格模式下脚本会拒绝。
  legacy 模式不放松判据，只在报告里写明"**不满足 §1.6 规格、一致率会被高估**，仅用于口径对齐"。
- **C-3 报差异时**：这是口径漂移（真问题）——**先查两侧口径，不许改阈值**（`AGENTS.md` §7）。

### 8.4 回填（把结果给我）

把三条命令的关键输出贴回来（§8.1 的总数 / §8.2 的判据 / §8.3 的 C-1 与 C-3 行），
我据此回填 `PROGRESS.md` §3.5 / §3.0e / §6.5 与两份计划的 §8 / §3。

### 8.5 新红处置：`Gpt2OnnxTest.MatchesAcrossProfileShapes`（seq=512 逐行 argmax）——**已按方案 B 结案（真机复跑通过）**

**为什么单开一节**：它不是本次交付引入的（改动面不触及该路径，见 `TROUBLESHOOTING.md` #34.2），
但它现在是全量报告里除"按设计红"之外的唯一红，必须有人接手——否则下个会话会把它当成噪声。

```bash
# 复跑：确认五个形状的"不可判行"与 ExpectedUndecidableRows() 登记表逐行一致（§34.9）
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

## 9. 执行结果（待回填）

> 回填要求：只写"状态 + 产出 + 判据实测值 + 出处（命令 / 日志）"；排查过程写 `docs/TROUBLESHOOTING.md`。

| 任务 | 状态 | 产出 | 判据实测值 / 出处 |
|---|---|---|---|
| A1 BPE Tokenizer | ✅ **完成**（2026-09-26） | `tokenizer/bpe_tokenizer.{hpp,cpp}`；`utils/json.hpp` 补 `\uXXXX`；`tools/make_tokenizer_golden.py` + `tests/data/gpt2_tokenizer_golden.json`；`tests/test_bpe_tokenizer.cpp`（8 条）+ 参考自证 1 条 + `test_json.cpp`（6 条）+ `test_gpt2_generate.cpp` 的桥接用例 1 条；ctest 项 `tokenizer_golden_check` | `Encode` 与 HF **逐 token 全等**（21 样本：basic 3 / whitespace 5 / utf8 10 / long 1 / edge 2；长文本 306 token）；`Decode` 一致；4 条 `Load` 负例全部拒绝；`ctest -R tokenizer_golden_check` Passed 3.21 s；沙箱全量 **204 条 / 0 失败**。排查过程 → `TROUBLESHOOTING.md` #33 |
| A2 INT8 判据离线规格 | ✅ **完成**（2026-09-26） | `tools/validate/README.md`（规格）、`tools/validate/int8_eval.py`（评估 + 自检 + `--legacy-mode`）、ctest 项 `int8_eval_selftest`；`phase4_int8_plan.md` §4 与 `phase4_test_plan.md` R2.6 后续行已加指向 | `--self-test` 全绿（3 项分层数学 + 7 道护栏 + 1 项 legacy 标注）；真实 `calib_data`（500 文件）smoke 通过；沙箱全量 **204 条 / 0 失败** |
| A2-5 INT8 口径交叉校验 | ✅ **真机通过**（2026-09-26，作者执行） | C++ 侧 `ResNet18Int8AccuracyTest.DumpsLogitsAndCppReportForCrossCheck`；`tools/validate/int8_eval.py --legacy-mode` + `crosscheck_reports.py`；ctest 项 `int8_crosscheck` / `int8_crosscheck_selftest` | 沙箱：`--self-test` 5 项全过；**真机：三步全过（作者回报"4. pass"）**，说明两侧实现对同一批 logits 给出**同一组 n 与分子** |
| B 文本端到端 | ✅ **真机通过**（2026-09-26，作者执行） | `Gpt2GenerateTest.RealGpt2TextPromptEndToEnd` | **真机：PASSED（作者回报"3. pass"）**——分词 / 生成 / 解码文本三段判据全过，即"文本进 → 文本出"链路成立 |
| A 真机全量 | ✅ **215 条 / 1 红**（2026-09-26 全量重跑确认） | —— | 唯一红 = `RealGpt2Fp16GreedyMatchesReferenceTokens`（按设计）。更早的 204 条 / 2 红 是修正判据前的快照：ONNX argmax 那条机制见 `TROUBLESHOOTING.md` #34.6、判据见 #34.9、复跑实测见 #34.9 末尾 |
| B1 P4-INT8-a | 未触发 | —— | —— |
| C 组 | 未触发 | —— | —— |

---

*本计划不含对话过程，只含执行安排与纪律；事实与判据的唯一来源仍是 `docs/future_iterations.md`。*

---

## 10. §9.2 Sampler 高性能 kernel（开发计划）

> **本文档是它唯一的正式计划落点**（2026-09-26 作者决定：后续迭代中每个事项的开发/测试计划
> **都并入 `future_iterations*_plan.md` 家族，不再单独新建 per-item 文件**）。
> 配套测试计划见 `future_iterations_test_plan.md` §9。

### 10.1 计划对账（AGENTS.md §5 第 0 步）

#### 10.1.1 有没有计划文档

**没有。** 覆盖本次工作的只有 `future_iterations.md` §9.2 的条目（背景 / 工作内容 / 触发时机），
没有开发计划与测试计划。→ 本文件 + 本文件同级的 `future_iterations_test_plan.md` §9 即为它们。

#### 10.1.2 逐条对照：§9.2 的描述 vs 代码现状

| §9.2 的说法 | 代码证据 | 对上了吗 |
|---|---|---|
| "Top-K / Top-P 目前每步都要对整行 `vocab_size` 做一次降序排序" | `src/sampler/sampler_kernels.cu`：`PrepareSortInputKernel`（物化 fp32 key + 下标）→ `cub::DeviceSegmentedRadixSort::SortPairsDescending`（逐行整段排序）→ `TopKSampleKernel` / `TopPSampleKernel` | **一致** |
| "是明显的性能瓶颈（`vocab_size` 可达 128K）" | 排序 O(V log V) + 物化两份 [B,V] 中间缓冲；Top-P kernel 还要两趟全行扫描 | 一致（但**占比未测**，见 0.3 偏差 1） |
| "工作内容：warp-level Top-K 选择、bitonic sort" | 现状无任何 warp 级选择逻辑 | 一致（尚未做） |
| "与后续 continuous batching 配合" | `LLMRunner` 仍限定 `batch = 1`（缺口 G2-3） | **偏差 2**：本次只保证 kernel 层支持 batch>1（现状已支持多行），**不接**调度 |
| 接口形态 | `sampler_common.hpp` 只有设备侧 API、k/p 是 per-batch 张量、workspace 由调用方提供（`PROGRESS.md` §2.12 的约定） | **本次不改**（见 §2 D1） |
| workspace 谁分配 | `LLMRunner` 构造时按 `TopKSamplerWorkspaceBytes(1, vocab)` 一次性分配（`llm_runner.cpp:181`），测试同样查询该函数 | 说明**缩小 workspace 是兼容改动**（调用方只查大小） |

#### 10.1.3 偏差怎么处理（写在前面，动手前先认账）

- **偏差 1：触发条件（decode 性能 profile）至今没测过。** §9.2 原文的触发时机是"先用 `nsys`/`ncu`
  定位到 sampler 占比显著"。本次由作者决定直接推进 → 处理方式：把"**先测基线**"作为本计划的第一个任务
  **P9_2-0**，这样既给性能结论一个 before 数，也保留一个合法出口：
  **若基线显示 sampler 占比可以忽略，则只落地"正确性/覆盖"改进（FP16 + 大 vocab 覆盖），不换实现**。
- **偏差 2：continuous batching 的对接不在本次范围。** 它依赖 §2.3 + 缺口 G2-3；本次只在 kernel 层
  保证多行正确（既有能力）。

#### 10.1.4 反向查：文档与代码的矛盾（当场记）

- `future_iterations.md` §11 的 **P1.5-a** 说"Top-K / Top-P 的 FP16 分支未覆盖（Greedy 已覆盖）"——
  核对 `tests/test_fp16_paths.cpp`：**属实**（只有 `GreedySamplerMatchesArgmaxOnFp16Logits`）。
  本次顺带补齐，完成后 **P1.5-a 可关闭**并回填 §11。
- `PROGRESS.md` §2.12 的"采样器只有设备侧 API"——核对代码：**仍成立**，本次不动该形态。
- 既有用例的**并列语义**（"降序排序时相同 key 保持原下标顺序"依赖 CUB 的稳定性）——代码里没写明，
  属**隐含契约**。本次实现必须显式保持（见 §2 设计第 4 条）并在测试里锁住。

---

### 10.2 目标与范围

**目标（2026-09-26 由 §10.11 收窄，原文保留在下面的"做/不做"里以便追溯）**：
Top-K 侧把"整行降序排序"换成"每行一个 block 的部分选择（top-M）+ 只对选出的候选排序"；
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

### 10.3 接口与设计（D1~D4）

**D1｜API 形态不变**：只允许改 `TopKSamplerWorkspaceBytes` / `TopPSamplerWorkspaceBytes` 的**返回大小**
与注释（调用方只查询大小，见 §0.2 表）。签名、参数结构、错误约定（返回 `cudaError_t`、越界/空指针返回非
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

### 10.4 任务分解（P9_2-0 ~ P9_2-8）

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

### 10.5 验收标准（每条都要能回答"凭什么"）

1. **语义回归（主判据）**：`tests/test_sampler.cpp` 既有 11 条用例**不改判据、全部通过**；
   另有 `test_e2e_mini_decoder.cpp` 的 Top-K（k=1）与 `test_fp16_paths.cpp` 的 Greedy 用例通过。
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

### 10.8 真机执行清单（**2026-09-27：四步全部已执行**）

> 已执行结果：第 1 步与第 3 步（`ctest` 全量）**228 条 / 226 通过 / 1 红 / 1 跳过**，
> 红 = 按设计的 FP16 复现器、跳过 = `int8_crosscheck`（缺报告 → 77）。
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

### 10.9 结果回填（待回填）

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
   跨 session 的 ±25% 漂移（`phase3_test_plan.md` §5 记过同一现象）足以把 13× 变成 10×。
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
这符合 §9.4 登记的机制**：两条实现对同一行的累计和只差舍入，但采样命中的是"第一个累计 ≥ target
的**下标**"；12.8 万词表上 nucleus 内相邻台阶只有 ~1e-5·total 的间距，而串行累加 11 万项的舍入误差
量级是 `ε·√N`（≈1e-5·total）——两者同量级，命中下标因此可能挪 1~2 位，token 标签就换了。
分布层面这只是一次 ~1e-5 的 CDF 扰动（S-21 在 3σ 下通过、S-15 在带 +1 元素边界的 nucleus 内通过）。
**n 只有 3 次抽样，不足以谈"差异率"**，所以它只打印、不判定。
**排查过程完整留痕**：`docs/TROUBLESHOOTING.md` **#35**（两把尺子的合法性论证、延迟/吞吐的
判别法、串行重扫的 cycle 估算与可选优化 P9_2-5b）。

---

### 10.12 P9_2-5b：把 Top-P 的收尾段也并行化（2026-09-27 立项，同日实施）

#### 10.12.1 计划对账（AGENTS.md §5 第 0 步）

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
   §9.4 登记的"允许差异"类别不变（仍只有浮点累加顺序；这次把"分块 + 块间串行合并"扩成
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


### 10.10 P9_2-2~4 结果：正确性达标、**性能不达标**（2026-09-26 真机）

**正确性**：`TopKFastMatchesLegacyTokens`（fast vs legacy **逐 token 相同**，4 形状）与
`TopKFastPoisonsRowsAboveContractLimit` 均真机通过 → 语义等价成立。

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


### 10.11 决策（2026-09-26）：先做 Top-P，且**改用"保留排序、并行化行内数学"的方案**

**作者决定**：跳过 Top-K 重做、直接做 **P9_2-5（Top-P）**——它是基线里的绝对大头（6.6~22 ms，
是 top-k 的 10 倍量级）。

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
---

*本计划只含执行安排与纪律；条目目标与触发条件的唯一来源仍是 `docs/future_iterations.md` §9.2。*
