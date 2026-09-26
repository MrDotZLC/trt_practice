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

**目标**：把 Top-K / Top-P 的"整行降序排序"换成"**每行一个 block 的部分选择（top-M）+ 只对选出的
候选做 bitonic 排序**"，在**语义不变**的前提下降低每步采样的开销；同时补齐 FP16 与大 vocab 的覆盖。

**做**：

1. 先测基线（P9_2-0），据此决定"是否换实现"以及性能判据的阈值（**阈值必须由实测给出**）；
2. 新 kernel：`TopKSelectKernel`（block-per-row，strided 扫描 + 块内归并出 top-M，M 自适应）；
3. 对 M 个候选做 `BlockRadixSort`/bitonic 降序；
4. Top-K 采样改走新路径；Top-P 在**语义等价**的前提下走新路径，并在"nucleus 超出 M"时**回退**到既有 CUB 路径；
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
5. Top-P 的 nucleus：在排好序的候选前缀上累计；若**累计未达 p**（nucleus 超出 M）→ 该行
   **回退**到既有 CUB 路径（正确性优先），并把回退行数作为**观测指标**打印（不作判据）。

**D3｜隐式契约显式化（并列语义）**：既有实现依赖"相同 logit 时保持原下标顺序"（等价于"并列取小下标"）。
新实现必须在块内排序时**显式**保证同 key 按小下标优先，并在测试里锁住（见测试计划 S-4）。

**D4｜数值路径不变**：softmax 用 `__expf`、减去行 max 稳定化、逆变换 CDF 比较用 `>=`、
Top-P 在保留前缀内**重新归一化**——逐条与现状对齐（这些是本阶段"语义不变"的具体含义）。

---

### 10.4 任务分解（P9_2-0 ~ P9_2-8）

| 任务 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| **P9_2-0** | **基线测量**：现有实现的采样耗时与占比（`nsys`/`ncu` + 单测微基准），vocab ∈ {50257, 128000}、batch ∈ {1, 8} | 真机 | 基线表（中位数/极差）→ 决定是否继续换实现、并给出性能阈值。**仪器已就位（2026-09-26）**：`tests/test_sampler.cpp` 的 `SamplerPerf.ThroughputByShape`（CudaTimer + warmup 3 + 采样 21 次，报中位数/极差/min-max，并给 greedy 作"一趟扫描"参照）；**基线数据待真机运行** |
| P9_2-1 | 固定"语义清单"：把现有实现的语义写成可核对条目（含并列、p 截断、重新归一化、随机数消费） | —— | 测试计划 §3 的判据表 |
| P9_2-2 | top-M 选择 kernel（含阈值迭代与覆盖率统计） | P9_2-1 | `TopKSelectKernel` + host 可测的覆盖率逻辑 |
| P9_2-3 | 候选排序（bitonic/radix，稳定同 key→小下标） | P9_2-2 | 排序 kernel |
| P9_2-4 | Top-K 接新路径 | P9_2-3 | 采样结果与语义清单逐条一致 |
| P9_2-5 | Top-P 接新路径 + 回退路径 | P9_2-4 | 覆盖率/回退行数可观测 |
| P9_2-6 | FP16 分支覆盖（补 P1.5-a） | P9_2-4/5 | 新增用例 |
| P9_2-7 | 大 vocab 覆盖（≥50257、合成 128K） | P9_2-4/5 | 新增用例 |
| P9_2-8 | 性能复测 + 文档回填（含 §11 的 P1.5-a 关闭） | 全部 | 本文件 §8、`PROGRESS.md`、`future_iterations.md` |

---

### 10.5 验收标准（每条都要能回答"凭什么"）

1. **语义回归（主判据）**：`tests/test_sampler.cpp` 既有 11 条用例**不改判据、全部通过**；
   另有 `test_e2e_mini_decoder.cpp` 的 Top-K（k=1）与 `test_fp16_paths.cpp` 的 Greedy 用例通过。
2. **FP16 覆盖（补 P1.5-a）**：新增 Top-K / Top-P 的 FP16 用例，判据沿用既有分布口径
   （与解析 softmax 概率的 **3σ** 比较，出处：`TopKDistributionMatchesSoftmaxProbabilities` 的注释）。
3. **大 vocab 覆盖**：vocab = 128000（合成）与 50257（真实形状）下，Top-K 结果必落在解析 top-K 集合内；
   Top-P 结果必落在解析 nucleus 内。**这两条不设阈值**，是集合成员关系判定（无阈值即无出处问题）。
4. **性能**：报告 vocab ∈ {50257, 128000} × batch ∈ {1, 8} 的**中位数与极差（≥20 次）**；
   相对 P9_2-0 基线的加速比阈值**由基线实测给出**，写进测试计划时附"哪次实测"。**不许先写一个数字。**
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

### 10.7 破坏性动作清单（**预告，执行前逐条确认**）

1. **修改** `mini_trt_llm/src/sampler/sampler_kernels.cu`（新增 kernel + 替换 Top-K/Top-P 的排序路径；
   **既有 CUB 路径保留为 Top-P 的回退路径**，不删除）；
2. **修改** `mini_trt_llm/include/mini_trt_llm/sampler/sampler_common.hpp`——**只改注释与
   `WorkspaceBytes` 的语义说明**，不动签名；
3. **新增** `mini_trt_llm/tests/test_sampler_large_vocab.cpp`（或并入 `test_sampler.cpp`，看规模）；
4. **修改** `mini_trt_llm/tests/test_fp16_paths.cpp`（补 Top-K / Top-P 的 FP16 用例）；
5. **可能修改** `mini_trt_llm/tests/CMakeLists.txt`（若新增测试文件需注册；GLOB 只需重跑 configure）；
6. **不改**：`llm_runner.cpp` 的调用形态、`sampler_common.hpp` 的 API、greedy 实现。

---

### 10.8 真机执行清单（待作者执行）

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
| P9_2-0 基线测量 | 🟡 **仪器已完成，待真机取数** | `SamplerPerf.ThroughputByShape`：沙箱显式跳过（无 GPU）；沙箱全量 216 条 / 0 失败；真机命令见 §10.8 |
| P9_2-1 ~ P9_2-7 | 未开始 | —— |
| P9_2-8 回填 | 未开始 | —— |

---

*本计划只含执行安排与纪律；条目目标与触发条件的唯一来源仍是 `docs/future_iterations.md` §9.2。*
