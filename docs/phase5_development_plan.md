# Phase 5 开发计划：下线两个历史示例工程

> **状态**：2026-09-27 立项（同日的"永久取消"决定已改回，见 `PROGRESS.md` §4.6）。
> **尚未开工**：本文只是计划；按 `AGENTS.md` §0.7，执行要作者点名到具体阶段，删除动作还要逐条确认。
>
> **事实归属**：现状、测试基线、产物尺寸一律看 `PROGRESS.md` 的「当前基线」，本文不复制数字；
> 排查与实验记录见 `TROUBLESHOOTING.md`（本文 §3 的实验若要长期留痕，另开一条 `TS-xxx`）。

---

## 0. 计划对账（`AGENTS.md` §5 第 0 步）

### 0.1 有没有计划文档

**有，即本文。** 原 `mini_trt_llm_design.md` 的 Phase 5 段（**冻结**）只写了"从根 CMakeLists 移除
两个 `add_subdirectory` + 删除或归档旧目录"，**没有**资产处置、没有覆盖损失的可见性、没有回退方案
——按那份原文直接动手会把 ONNX / INT8 用例的来源一起删掉。因此重写为本文，冻结段只作历史记录。

### 0.2 逐条对照：任务 / 接口 / 验收 vs 现状

| 本次要做 | 现状 | 是否一致 |
|---|---|---|
| 把两个历史示例工程从仓库路径里下线 | 根 `CMakeLists.txt` 的两行 `add_subdirectory` 早已注释；两个目录只剩源码 + 本地资产 | 一致（"下线"在做，只是没删目录） |
| 保住现有判据 | 两个目录提供了 GPT-2 logits 基线、ONNX 图、INT8 验收集、`resnet18_convert_selftest` 的夹具 | **有偏差**：它们不是"没人用的旧代码"，删前必须先迁资产 |
| 覆盖损失可被发现 | ctest 把 Skipped 记成 Passed；`MINI_TRT_REQUIRE_GPU=1` 只管"无设备"，不管"缺资产" | **有偏差** → 阶段 0 |

### 0.3 偏差怎么处理

**先补文档与闸门、再动代码与文件**。本文已给出阶段顺序；任何删除动作开工前按 `AGENTS.md` §0.5
一次性列清单确认。

### 0.4 反向查：文档与现状矛盾之处

- `future_iterations_development_plan.md` §6.6 第 6 条写"不擅自删旧模块（Phase 5 **已永久取消**）"
  → 与本次立项矛盾，**同批修正**（2026-09-27）：改为"未经点名批准不得删除/移动；Phase 5 已重新立项，
  方案见 `docs/phase5_development_plan.md`"。
- 冻结文档（`mini_trt_llm_design.md` §3、`phase0_development_plan.md`、`phase4_development_plan.md`）
  里的"Phase 5 已永久取消 / Agent 不要删除"**已按"冲突修正"批注**：原文保留（可加删除线）、
  只追加带日期的更新注，指向本文；规则见 `docs/README.md` §7 的例外条。结论由本文取代。
- 我此前口头说过"`future_iterations.md` §11 有'永久取消'表述"——**那是错的**：该文件里没有这句；
  真正的落点是 `PROGRESS.md`（6 处）与 `future_iterations_development_plan.md` §6.6（1 处）。

---

## 1. 目标与范围

### 目标

让仓库路径里不再出现 `0_resnet18_onnx` / `1_gpt2_onnx`，同时满足三条硬约束：

1. **不丢任何判据**（现有用例的覆盖面不减少）；
2. **覆盖损失必须可见**——不许把"用例没了"变成"静默跳过"；
3. **资产仍可重建**，且重建入口在仓库里有明确位置。

### 非目标（明确不做）

- 不恢复旧模块的构建（根 `CMakeLists.txt` 的注释项按本文阶段 3 直接删掉，不重建）；
- 不改动旧模块的**业务逻辑**（它们不参与构建，改逻辑没有收益）；
- 不重算任何基线数字（除非阶段 1 的软链接方案被否，见 §5 与 §10）。

---

## 2. 资产盘点（要迁走的东西）

| 资产 | 体积 | 现在被谁用 | 可否离线重建 |
|---|---|---|---|
| `0_resnet18_onnx/resnet18.onnx` | 46.7 MB | 12 个 CV 用例；`onnx_to_mini_trt_llm.py` 与 `quantize_resnet18.py` 的**输入**；`resnet18_convert_selftest` 的夹具；GPT-2 错误路径用例借它当"外来 I/O 名" | 可（`load_model.py`，torchvision 权重已缓存） |
| `0_resnet18_onnx/calib_data/` | 288 MB / 500 张 | INT8 判据的验收集（余量子集一致率）、`Int8ProbeTest`、`qdq_reference.py`、`int8_eval.py` 的重叠排除基准 | **不可**（需联网下载 tiny-imagenet） |
| `1_gpt2_onnx/gpt2.onnx` | 652 MB | `Gpt2OnnxTest`（3 条）、`inspect_onnx.py` 的结构基线 | 可（HF gpt2 权重本地已有） |
| `1_gpt2_onnx/ref_output.bin` | 804 KB | GPT-2 prefill logits 的**唯一**精度基线 | 技术可重建，但会失去与历史实测数字的可比性 |
| 两个目录的源码等 | 22 个 tracked 文件 | 不参与构建 | 已在 git 历史里 |

三份 `models/resnet18/*.json` 里记着 `"onnx": "0_resnet18_onnx/resnet18.onnx"`，
属**产物来源（provenance）记录**——迁移时怎么处理见 §5。

---

## 3. 依据：2026-09-27 的改名实验

**做法**：把两个目录原地改名（不删、不复制），跑沙箱全量，再改名还原。

| 项 | 改名前 | 改名后 |
|---|---|---|
| ctest 总结 | 265 条，**100% passed，0 failed** | 265 条，**100% passed，0 failed** |
| `ResNet18OnnxBuildTest.RejectsUnknownSubgraphName` | Passed | **Skipped** |
| `onnx_graph_probe` | Passed 8.18 s | **Skipped 0.30 s** |
| `resnet18_convert_selftest` | Passed 0.89 s | **Skipped 0.35 s** |
| gtest 计数 | PASSED 144 / SKIPPED 112 | PASSED 143 / SKIPPED 113 |

**三条结论**：

1. **ctest 的通过信号对这次删除完全不可见**（还是 265 / 0 failed / 100%），因为 Skipped 记作 Passed。
2. `MINI_TRT_REQUIRE_GPU=1` **拦不住**：它只把"无设备"跳过变成失败，资产缺失走裸 `GTEST_SKIP`
   （见 `tests/test_gpu_guard.hpp`）。
3. **静态分析会漏**：`resnet18_convert_selftest` 对 `resnet18.onnx` 的依赖写在
   `tests/CMakeLists.txt` 的 `add_test` 参数里，grep 测试源码抓不到。所以阶段 0 的闸门与
   "期望跳过集合"是必需的，不能靠人力核对。

---

## 4. 阶段 0：让覆盖损失可见（**前置，纯新增**）

> 不做这一步，后面任何一步的删除都无法验证——**它是本计划的第一步，也是风险最低的一步**。

1. **P5-0-1 加资产闸门**：新增 `MINI_TRT_REQUIRE_ASSETS`（与 `MINI_TRT_REQUIRE_GPU` 同形态）。
   必须是**宏**：`GTEST_SKIP()` / `GTEST_FAIL()` 展开后都是 `return`，封装成函数只会退出那个函数，
   用例体会继续往下跑（`tests/test_gpu_guard.hpp` 里记着这条教训）。
2. **P5-0-2 钉住"期望跳过集合"**：一次真机全量导出跳过集合 → 落成清单文件 → 加一个 ctest 比对项，
   集合一变就红。**判据是"集合逐条相等"**，不是"条数相等"（改名不改条数，正是本计划的坑）。
3. **P5-0-3 给闸门本身加用例**（`PROGRESS.md` §2.13「护栏必须有用例证明它会拦人」）：
   故意把资产路径指向空目录，断言 REQUIRE 模式下**失败**而非跳过；非 REQUIRE 模式下仍然跳过。
4. **P5-0-4 改记录**（改文档结论，需作者点名）：`PROGRESS.md` §4.6 / §6.6 的"Phase 5 永久取消"
   与 `future_iterations_development_plan.md` §6.6 第 6 条。**已于 2026-09-27 获批并执行**。

### 4.1 执行状态（2026-09-28，沙箱）

| 项 | 状态 | 证据 / 落点 |
|---|---|---|
| P5-0-1 资产闸门 | ✅ 完成 | 新增 `mini_trt_llm/tests/test_asset_guard.hpp`：`RequireAssets()` + `MINI_TRT_SKIP_IF_MISSING_ASSET(...)` 宏（必须是宏，理由同 GPU 闸门）。**27 处**"旧模块资产"跳过点改走闸门（9 个测试文件） |
| P5-0-1b 脚本项闸门 | ✅ 完成 | `tools/inspect_onnx.py` / `tools/convert/onnx_to_mini_trt_llm.py` 的缺资产分支同样受 `MINI_TRT_REQUIRE_ASSETS` 控制（返回 1 而不是 77）。**两处同名同义实现，改一处要两处同改**（代码注释里已写明） |
| P5-0-2 期望跳过集合 | ✅ **完成（2026-09-28）** | 基线来自真机全量（267 / 1 红 / **0 跳过** / 301.72 s，从 `LastTest.log` 逐条解析）。落点：`mini_trt_llm/tests/data/expected_skips.txt`（空集合 = "真机全量不允许任何覆盖跳过"）+ `mini_trt_llm/tools/check_skips.py`（解析日志、忽略 `asset_gate_*` 探针段与"无 GPU"跳过、未登记的跳过判红、`--update` 重钉、`--self-test` 6 项）+ ctest 项 `check_skips_selftest`。**判定口径**：比对本身不进 ctest（要吃一次完整跑的日志），只注册自检 |
| P5-0-3 闸门自证 | ✅ 完成 | 新增两条 ctest 项 `asset_gate_skips_without_require` / `asset_gate_fails_with_require`：两者都从**空目录**跑 `ResNet18OnnxBuildTest.RejectsUnknownSubgraphName`（相对候选全落空 → 必走缺资产分支），后者用 `WILL_FAIL` 断言"它真的红了"。**闸门被删 → 该 test 立刻变红** |
| P5-0-4 改记录 | ✅ 完成 | 2026-09-27（本文 §0.4 有清单） |

**验收实测（2026-09-28，沙箱）**

| 检查 | 结果 |
|---|---|
| 全量（资产在位，不设变量） | **267 条 / 0 失败**（原 265 + 本批 2 条） |
| **跳过集合零变化** | 与加闸门前逐条 `diff` **为空**（`SKIP_SET_IDENTICAL`） |
| 空目录、不设变量 | 该用例 `[ SKIPPED ]`，退出码 0 |
| 空目录、`MINI_TRT_REQUIRE_ASSETS=1` | 该用例 `[ FAILED ]`，退出码 1 |
| **两个目录改名 + 变量置 1**（复现 §3 的实验） | 上一轮那 3 处"静默跳过"**全部变红**：`onnx_graph_probe` / `resnet18_convert_selftest`（ctest Failed）+ `ResNet18OnnxBuildTest.RejectsUnknownSubgraphName`（gtest Failed） |

> **真机首跑修正（2026-09-28，`TROUBLESHOOTING.md` + TS-049）**：`asset_gate_skips_without_require`
> 原先**继承环境里的 `MINI_TRT_REQUIRE_ASSETS`**，而真机验收命令本身就带 `=1` → 探针"应当跳过"
> 的前提被自己破坏，探针反被判红。已改为显式 `MINI_TRT_REQUIRE_ASSETS=0`（两条自证项现在都自己
> 钉住变量，互不依赖环境）。**教训**：自证 / 探针类 test 的前提若与环境变量有关，必须自己钉死。

**P5-0-2 怎么用**（比对本身不进 ctest：它要吃一次完整跑的日志，在 ctest 里跑会递归）：

```bash
# 真机：跑完全量后比对（基线 = 空集合）
MINI_TRT_REQUIRE_GPU=1 MINI_TRT_REQUIRE_ASSETS=1 ctest --test-dir build --output-on-failure
python3 mini_trt_llm/tools/check_skips.py --log build/Testing/Temporary/LastTest.log
```

判据与两条已知边界：① `asset_gate_*` 探针段里的跳过**一律忽略**（那是探针的设计）；
② "无 GPU"跳过**默认忽略并打印**——判据是 `CudaProbeToString()` 里的
`cudaGetDeviceCount -> err=`，**不能只认 "No CUDA device available" 那句固定文案**：
传了自定义说明的调用（如 `MINI_TRT_SKIP_IF_NO_CUDA("解析 ONNX 需要 CUDA…")`）不带它
——2026-09-28 这个工具自己在沙箱日志上抓到 2 条漏判，已修并把该形态加进自检。

**覆盖范围（2026-09-28 第二批之后）**：闸门覆盖**全部"资产缺失型"跳过点**——共 **62 处 /
15 个测试文件**（旧模块资产 27 处 + `models/` 产物与 tokenizer 目录 35 处），用的仍是同一道闸门
（`MINI_TRT_REQUIRE_ASSETS` + 同一个宏），**没有新增机制**。

**有意不覆盖的三类**（跳过语义保持不变）：

1. 缺 **Python 包**（`onnx` / `transformers` / `python3`）——属环境，不是资产；
2. 缺仓库内的脚本文件（`tools/make_tiny_onnx.py`）——那是仓库缺陷，不该走"缺资产跳过"；
3. `int8_crosscheck` 缺**流程产物**（报告）——按设计就是"先全量、后跑 C"，缺报告跳过是对的。

**第二批验收（2026-09-28，沙箱）**：全量仍 **267 条 / 0 失败**；**跳过集合与加闸门前逐条相同**
（默认行为零变化）。新覆盖面确实生效：把 `models/gpt2` 改名后，
`Gpt2WeightContractTest.RealConvertedArtifactIsComplete` 在不设变量时 SKIP（退出码 0）、
设 `MINI_TRT_REQUIRE_ASSETS=1` 时 **FAIL**（退出码 1，报文带资产准备指引）。

---

## 5. 阶段 1：资产迁移（**行为零变化**）

目标布局：

```text
assets/legacy/
├── resnet18.onnx
├── resnet18_calib_data/          # 500 张，保持原文件名
├── gpt2.onnx
├── gpt2_ref_output.bin
└── README.md                     # 来源 / 获取方式 / SHA256 / 重建命令 / 为什么留着
```

- 用 `mv`（同盘、秒级、不复制）；`.gitignore` 增补对应忽略规则，维持"大产物不入库"的既有规矩。
- **验收判据是"零变化"**：迁移后立刻跑沙箱 ctest，条数与跳过集合必须与迁移前**逐条一致**。
- 需要改路径的位置（共约 15 处代码 / 脚本 / 构建点）：

| 类别 | 位置 |
|---|---|
| C++ 测试的 `FindFile({...})` | `test_resnet18_onnx` / `test_resnet18_native` / `test_resnet18_fp16` / `test_resnet18_int8` / `test_resnet18_int8_probe` / `test_cv_runner` / `test_gpt2_onnx` / `test_gpt2_onnx_error_paths` / `test_gpt2_prefill_accuracy` |
| 构建 | `mini_trt_llm/tests/CMakeLists.txt`（`resnet18_convert_selftest` 的 `--onnx`） |
| 工具 | `inspect_onnx.py`、`onnx_to_mini_trt_llm.py`、`quantize_resnet18.py`、`validate/{README.md,int8_eval.py,qdq_reference.py}`、`scripts/ref_resnet18.py` |
| provenance | `models/resnet18/config.json`、`resnet18_qdq.meta.json`、`resnet18_qdq_per_channel.meta.json` |

**两个附带决策**（建议值，开工前请作者确认）：

1. **路径解析先收敛**：现在 9 个测试文件各写 4 个相对路径候选，对工作目录敏感（`TS-025` 就是这类
   helper 出的坑）。建议新增共享 helper：`MINI_TRT_ASSET_DIR` 环境变量 → `assets/` 默认 → 旧路径兜底。
   **这是独立一项改造**，不要混进纯搬迁，否则出问题无法归因。
2. **provenance 用软链接过渡**：在仓库根留 `0_resnet18_onnx → assets/legacy` 与
   `1_gpt2_onnx → assets/legacy` 两条软链接，让老路径继续可解析，`*.meta.json` 一字不动。
   好处是**零数字变动、零产物重生成**；代价是"目录名还在"（只是变成链接）。
   若要彻底去掉名字，则必须改 provenance 字符串并说明原因（`AGENTS.md` §7），并评估是否重生成产物。

### 5.1 执行状态（2026-09-28，沙箱）

**采用的形式：软链接保老路径，`mini_trt_llm/` 零改动。** 与本节的偏差：原文设想"迁资产 + 改
15 处路径"；实际按作者给的"**不能影响 `mini_trt_llm`**"原则改为——数据搬到 `assets/legacy/`，
老路径用**相对软链接**指回来，于是 `mini_trt_llm/` 的代码、测试路径，以及
`models/resnet18/*.meta.json` 的 provenance 字符串**一行都不用改**。代价是**目录名还没消失**
（`0_resnet18_onnx/` / `1_gpt2_onnx/` 仍在，只是不再持有数据）。

| 资产 | 新位置 | 老路径 |
|---|---|---|
| `resnet18.onnx` | `assets/legacy/resnet18_onnx/resnet18.onnx` | `0_resnet18_onnx/resnet18.onnx`（软链接） |
| `calib_data/`（500 张） | `assets/legacy/resnet18_onnx/calib_data/` | `0_resnet18_onnx/calib_data`（软链接） |
| `gpt2.onnx` | `assets/legacy/gpt2_onnx/gpt2.onnx` | `1_gpt2_onnx/gpt2.onnx`（软链接） |
| `ref_output.bin` | `assets/legacy/gpt2_onnx/ref_output.bin` | `1_gpt2_onnx/ref_output.bin`（软链接） |

> **更新（2026-09-28，阶段 3 之后）**：上表的"老路径（软链接）"是**中间态**——那 4 条相对软链接
> 已随两个目录一起删除（阶段 3）。现在代码 / 测试 / 工具 / provenance 一律直接指向
> `assets/legacy/`，不再有任何 `0_resnet18_onnx` / `1_gpt2_onnx` 路径。

配套：新增 `assets/legacy/README.md`（来源 / 重建命令 / 链接表 / 何时才可以删）；`.gitignore`
增补一行**精确路径** `/0_resnet18_onnx/calib_data`——软链接不是目录，带斜杠的 `**/calib_data/`
匹配不到它，不加这行 `git add -A` 会把那条链接收进版本库。

**验收（2026-09-28，沙箱）**

| 检查 | 结果 |
|---|---|
| 迁移后全量 | **267 条 / 0 失败** |
| 跳过集合 | 与迁移前**逐条相同** |
| 软链接可用（沙箱能验的两条） | `onnx_graph_probe` Passed（2.42 s，经链接读 652 MB 的 `gpt2.onnx`）；`resnet18_convert_selftest` Passed（0.69 s） |
| 链接链是承重的 | 把 `assets/legacy` 改名后，`MINI_TRT_REQUIRE_ASSETS=1` 下 `ResNet18OnnxBuildTest.RejectsUnknownSubgraphName` **判失败**（退出码 1）——阶段 0 的闸门能抓到"断链"这种缺资产 |

**本批未做**：本节原列的 15 处路径改写**没有做**（软链接方案下不需要）。**要彻底去掉那两个目录名**，
必须在阶段 3 改它们，而其中 9 处位于 `mini_trt_llm/tests/`——那会**触碰 `mini_trt_llm`**，
边界问题见 §11。

---

## 6. 阶段 2：解除"借夹具"耦合

`Gpt2OnnxErrorTest.RejectsGraphWithForeignIoNames` 现在借 `resnet18.onnx` 当"外来 I/O 名"样本
——这正是 `PROGRESS.md` §2.13 记过的坑（借别的模型的产物当负例夹具 = 把两边契约耦合起来）。

做法：改用 `tools/make_tiny_onnx.py` 现造一张 I/O 名不匹配的小图（它已能造
`input_ids → not_logits`，扩展成本低），彻底断开这条依赖。

**执行状态（2026-09-28）：✅ 已执行**。生成器本就支持 `--input-name/--output-name`，所以
那条用例改成现场生成 `input → output` 的单节点图（对 LLM 契约两个名字都不符），
不再依赖 `assets/legacy/` 里的任何文件。夹具生成失败（缺 `python3` / `onnx`）仍按"缺环境"
跳过——与同一文件里的 G1c 用例同口径（这两类**有意不进**资产闸门，见 §4.1 的覆盖边界）。
**副作用（好事）**：该用例从此与本地资产解耦——即使将来把 `assets/legacy/` 整体搬走 / 退役，
它照样能跑；若它在真机上因为生成夹具失败而跳过，`check_skips.py` 会判红（基线为空）。

---

## 7. 阶段 3：删除源码目录（**需逐条批准**）

**执行状态（2026-09-28）：✅ 已执行（作者 2026-09-28 "全部确认"）**，实际动作与本清单一致，
另有两处必要的顺带项（都写在下面）：

| 动作 | 结果 |
|---|---|
| 删除 22 个跟踪文件（两个目录的 `CMakeLists.txt` / `src/*` / `README.md` / `stats.sql` / `requirments.txt`） | 已删；`git status` 显示 22 条 ` D`；可用 `git show ba3ea7a:<路径>` 复核 |
| 删除根 `CMakeLists.txt` 的旧模块注释 | 删了 **3 行**（原清单只写"第 13/14 行两行"，实际连同其标题行"# 旧模块：后续 Phase 5 移除"一起删，避免留孤零零的注释） |
| 删除 4 条软链接 | 已删（两个目录随之消失） |
| 迁 3 个重建脚本 | → `assets/legacy/scripts/{resnet18_load_model.py, gpt2_load_model.py, resnet18_prepare_calib_data.py}`；**改的是输出目录**（原来靠脚本同目录 / 当前工作目录，现在按 `__file__` 定位到 `assets/legacy/{resnet18_onnx,gpt2_onnx}/`） |
| 改路径 | 20 处：9 个测试文件 + `tests/CMakeLists.txt` + 6 个工具/脚本（含 `--calib-dir` 默认值与命令示例）+ 3 处 provenance + 活文档 |
| provenance | `models/resnet18/{config.json, resnet18_qdq.meta.json, resnet18_qdq_per_channel.meta.json}` 的 `"onnx"` 改为新路径；**`onnx_sha256` 未动**（内容没变，hash 才是真身份）。**为什么可以改**：该字段是产图脚本写的来源记录，同一份文件换了位置；不留死路径比"保留旧字符串"更有用 |
| 历史引用 | 两处产品代码注释里"依据历史工程"的引用补了出处（`git show ba3ea7a:...`），避免删除后证据不可追溯 |
| 额外清理 | 删掉仓库根的 `Testing/`（作者那次"从仓库根跑 ctest"留下的 ctest 临时目录） |

**验收（2026-09-28，沙箱）**：全量 **267 条 / 0 失败**；**跳过集合与迁移前逐条相同**；
缺资产（把 `assets/legacy` 改名）时 `ResNet18OnnxBuildTest.RejectsUnknownSubgraphName` 判失败、
`onnx_graph_probe` ctest 项 Failed。**真机复跑待做**（见 §11）。

**真机验收（2026-09-28，作者执行）**：`MINI_TRT_REQUIRE_GPU=1 MINI_TRT_REQUIRE_ASSETS=1 ctest
--test-dir build --output-on-failure` → **267 条 / 1 红 / 0 跳过 / 301.72 s**，唯一红 = 按设计的
`RealGpt2Fp16GreedyMatchesReferenceTokens`（GPT-2 FP16 NaN 复现器）。
**这次跑等于完成了本阶段的"覆盖零损失"证明**：两个闸门都开着，缺资产与无设备都会**判失败**，
而除那条按设计的红外没有别的红。**跳过集合由 `build/Testing/Temporary/LastTest.log` 逐条解析
确认**：ctest 级 `***Skipped` = 0；gtest 级唯一一条 `[  SKIPPED ]` 来自
`asset_gate_skips_without_require` 探针**故意**从空目录跑（设计如此）→ **真实覆盖跳过 = 0**。
耗时从首跑的 892 s 回落到 301.72 s 也印证了"首跑的一次性引擎重建已结束"（`TS-049` 末尾）。

待删清单（执行前按 `AGENTS.md` §0.5 一次性列给作者确认）：

- `0_resnet18_onnx/{CMakeLists.txt, README.md, stats.sql, src/*}`
- `1_gpt2_onnx/{CMakeLists.txt, requirments.txt, src/*}`
- 根 `CMakeLists.txt` 第 13/14 行的两行注释

**必须保留并迁进 `mini_trt_llm/tools/`**（它们是资产的**生成脚本**，删掉等于放弃"能重建"）：

- `0_resnet18_onnx/load_model.py`、`1_gpt2_onnx/load_model.py`；
- `0_resnet18_onnx/prepare_calib_data.py`（`calib_data` 唯一的重建入口，且需联网 → 属要批准的动作）。

---

## 8. 阶段 4：验证与回填

1. **沙箱**：`ctest` 条数与跳过集合与基线**逐条一致**（阶段 0 的闸门到位后，这条才有意义）。
2. **真机**：`MINI_TRT_REQUIRE_GPU=1 MINI_TRT_REQUIRE_ASSETS=1 ctest --output-on-failure` 全量，
   红 / 跳过集合与基线逐条对比。
3. **全仓检索**：`rg '0_resnet18_onnx|1_gpt2_onnx'` 只应剩 `assets/legacy/README.md` 的历史说明
   与 git 历史。
4. **回填**：`PROGRESS.md` §4.6 / §6.6 / §6.5 / §7、`docs/README.md` 的文档地图与引用面统计。
   **边界（2026-09-28 明示）**：**冻结文档**（`mini_trt_llm_design.md`、`phase0_*`~`phase4_*`、
   `TROUBLESHOOTING.md`）里仍写着 `0_resnet18_onnx/` / `1_gpt2_onnx/` 的命令与叙述——
   按 `docs/README.md` §7 的冻结规则**不改**。**要跑命令请以 `PROGRESS.md` §6.5 / §7 与
   `assets/legacy/README.md` 为准**，冻结文档里的路径只作历史记录。

---

## 9. 验收判据（每条都要能回答"凭什么"）

| 判据 | 凭什么这么定 |
|---|---|
| 阶段 1 后沙箱跳过集合**逐条不变** | 迁移不改行为，任何变化都说明改漏或改错；用集合而非条数，因为本计划的坑正是"条数不变、用例换人" |
| 阶段 3 后真机跳过集合 = 阶段 0 钉住的基线 | 基线由阶段 0 用真机全量量出；删除的唯一可接受后果是"零变化" |
| 闸门本身有正反用例 | `PROGRESS.md` §2.13：护栏没有用例证明"它会拦人"就等于没有护栏 |
| 重建脚本在仓库内可定位 | 资产的"可重建"若只写在文档里、脚本却删了，等于不可重建 |

---

## 10. 风险与回退

| 风险 | 影响 | 回退 / 缓解 |
|---|---|---|
| `calib_data` 删了无法离线重建 | INT8 判据失去验收集 | 全程 `mv` 不复制不删除；迁移后立刻验证新位置可读 |
| provenance 字符串改动触发产物重生成 | 全套 INT8 数字要重测回填 | 优先用软链接（阶段 1 决策 2），不动字符串 |
| 路径候选散在 9 个测试文件里，改漏 = 静默跳过 | 覆盖静默下降且 CI 不报 | 阶段 0 的闸门 + 期望跳过集合；改漏立刻红 |
| 文档引用面漏改 | 下个会话照旧路径找不到东西 | 阶段 4 做一次全仓 `rg` 验收 |
| 抢跑（未点名就删除） | 不可逆的数据损失 | `AGENTS.md` §0.6 / §0.7：删除必须点名到具体文件并一次性确认 |
| `gtest_discover_tests` 的 5 s 发现超时在满负载构建下会**误报构建失败** | 看起来像代码错，实际是探针进程启动超时 | 2026-09-28 撞到一次，重跑即过；`--gtest_list_tests` 实测仅 0.08 s，说明是负载噪声。若频繁出现再考虑调大 `TEST_DISCOVERY_TIMEOUT`（构建脚本改动，需单独批准） |

---

## 11. 待作者拍板的开放项

1. **阶段顺序是否就按 0 → 1 → 2 → 3 → 4**？阶段 0 是纯新增，建议先做。
2. **阶段 1 的两条附带决策**：路径 helper 是否一并收敛；provenance 用软链接还是改字符串。
3. **资产新位置**：`assets/legacy/` 是否合适（与 `models/` 的语义区分：`models/` 放**转换产物**，
   `assets/` 放**作为输入的原始资产**）。
4. **要不要为本文 §3 的实验补一条 `TROUBLESHOOTING.md` 记录**（它证明的是"覆盖损失对 CI 不可见"
   这一类问题，价值超出本计划本身）。
5. ~~**资产闸门是否扩展到全部资产跳过点**~~ **已决（2026-09-28）：扩展，且复用同一道闸门**
   （作者："保证功能的前提下，尽可能复用原有 gate"）。已覆盖 62 处 / 15 个文件，覆盖边界与
   有意不覆盖的三类见 §4.1。
6. ~~**P5-0-2 的真机基线**~~ **已完成（2026-09-28）**：基线 = 真机全量 267 / 1 红 / **0 跳过**
   （跳过集合实测为空）；落点为 `tests/data/expected_skips.txt` + `tools/check_skips.py`
   + ctest 项 `check_skips_selftest`（用法见 §4.1）。
7. **"不能影响 `mini_trt_llm`"的边界**：① **语义不退化**（允许改测试里的路径 / 夹具，判据与覆盖
   不变——本批阶段 1 采用的就是这个读法）；还是 ② **文件面完全不动**（`mini_trt_llm/` 一行不改）？
   读法②与"**彻底去掉 `0_resnet18_onnx` / `1_gpt2_onnx` 这两个名字**"不可兼得：那两个目录被
   9 个测试文件的 15 处路径引用。**这一条决定阶段 3 能不能做。**
