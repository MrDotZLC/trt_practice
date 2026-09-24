# Phase 3 开发计划：GPT-2 ONNX + Plugin（方案 B）

> **定位**：让"ONNX → engine"这条路成为一等公民——能一键构建、失败可诊断，并能与
> Phase 2 的原生构建（方案 A）**交叉验证**。
>
> **状态**：**待确认 §2 的决策（尤其 D1）**。§0.2 记录了一处与设计文档描述的重大不一致，
> 它直接决定了本阶段的范围。

---

## 0. 开工前的事实核对（实测，非推测）

### 0.1 现有资产

`1_gpt2_onnx/gpt2.onnx`：652 MB，opset 17，输入 `input_ids [batch, seq_len]`（两维皆动态），
输出 `logits [batch, seq_len, 50257]`（`use_cache=False`，无 past/KV 输入输出）。

实测算子分布（2411 个节点、149 个 initializer）：

```
Constant 926   Unsqueeze 252   Shape 234   Gather 172   Concat 149   Reshape 148
Mul 73   Add 63   Slice 63   Squeeze 51   Gemm 48   Transpose 48   Sqrt 36   Cast 28
LayerNormalization 25   MatMul 25   Split 12   Div 12   Softmax 12   Pow 12
```

initializer 命名带 `model.transformer.` 前缀，**与 `models/gpt2/model.safetensors` 逐比特一致**
（已抽样核对 `wte.weight` / `h.0.attn.c_attn.weight` / `h.5.mlp.c_proj.weight` / `ln_f.bias`，
maxdiff = 0.0），Conv1D 同样是 `[in, out]` 约定，LM head 单独物化为 `[768, 50257]`。

→ **方案 A 与方案 B 用的是同一份数值，因此"两者输出对齐"是一个干净的判据**：
不需要外部基线，任何不一致都指向实现差异。

### 0.2 与 `mini_trt_llm_design.md` §3 描述的重大不一致

设计文档写的 Phase 3 是"对 `gpt2.onnx` 替换 **RoPE / RMSNorm / Attention** 子图"。
但实测该图**不含 RMSNorm、也不含 RoPE**（用的是 LayerNormalization 25 处 + 学习式位置编码）。
那两类替换在这个模型上**没有替换对象**。

可替换对象的盘点（也见 §0.3 的结论）：

| ONNX 里的算子 | 能替换成什么 | 替换的收益 |
|---|---|---|
| `LayerNormalization` × 25 | 自研 LayerNorm 插件 | **无**——TRT 原生已支持，且方案 A 用的就是原生层；换掉只会引入精度风险 |
| `Softmax` × 12 + `Gemm/MatMul/Transpose` 的注意力分解 | `IAttention` 或自研注意力插件 | 可能省几个 kernel，但该图是整段前向、没有 KV cache，**`PagedAttention` 用不上**（它只做 decode） |
| `Tanh` × 12（gelu_new） | `kGELU_TANH` | 无——已经是原生激活 |
| `Gather/Gemm/Reshape` 等 | —— | 属图优化，交给 TRT |

→ **结论：本模型上"子图替换"没有正确性收益**，只有潜在的（未量化的）性能收益。
这正是 D1 需要你定夺的地方。

### 0.3 现状与可复用资产

- `EngineBuilder::BuildFromOnnx(onnx_path, engine_path, plugin_ops)` 已能 parse + build，
  `plugin_ops` 处的子图替换是 TODO；
- `1_gpt2_onnx/src/builder.cpp` 是历史上跑通过的 TRT 工程，其 optimization profile /
  精度设置可作参照（避免我们重新踩一遍）；
- `PluginRegistry`（Phase 1）已具备"按名查找 creator"的能力，是子图替换的落点。

---

## 1. 阶段目标与非目标

### 目标

1. **ONNX 路径可用**：`BuildFromOnnx` 支持动态 batch/seq 的 profile 设置、精度选择、
   失败可诊断（parse 错误逐条打印），并能被测试与 runner 复用。
2. **与方案 A 交叉验证**：同一 prompt 下，ONNX engine 与原生 engine 的 logits 按
   Phase 2 确立的口径对齐（cosine + 相对界 + 逐位置 argmax）。
3. **子图替换机制**：范围由 D1 决定。

### 非目标

- **不改**：BPE tokenizer、批处理、INT8。
- **不做**：把 ONNX 路径接进 `LLMRunner`（该图没有 KV cache 输入，做不了 decode——
  见 D3）。

---

## 2. 待确认决策项

| 编号 | 决策项 | 建议 | 备选与理由 |
|---|---|---|---|
| **D1** | 本阶段"子图替换"做到什么程度 | **先做 C（识别+统计+断言），并用数据决定要不要做 B**：把 ONNX 里的注意力/归一化模式识别出来、打印与断言，作为 host 用例固化；同时量一次"ONNX 路径 vs 方案 A"的构建耗时与推理延迟，**用实测数据决定替换是否值得** | A（完全不做替换）：最快，但丢掉了"验证替换机制"这一能力；B（真做替换，如注意力子图 → `IAttention`）：机制价值真实，但 §0.2 表明正确性收益为零、性能收益未量化，属于"先做再找理由" |
| **D2** | 与方案 A 的对齐判据 | 沿用 Phase 2 实测口径：`cosine > 0.999999`、`max_abs/max\|ref\| < 1e-5`、逐位置 argmax 一致 | 若 ONNX 图与原生图的算子分解不同（如 Gemm vs MatMul、不同的 softmax 实现），差异可能更大——届时**按实测报告再定**，不预先放宽 |
| **D3** | 是否要求 ONNX 路径支持 decode | **不支持**（本阶段） | 该 ONNX 是整段前向、无 past 输入。要支持 decode 必须**重新导出带 KV cache 的 ONNX**，那是新工作量且与"方案 B 便于复用现有 ONNX"的初衷相悖 |
| **D4** | profile 与精度设置 | 复用 `EngineBuilder` 既有能力（`AddLlmOptimizationProfiles` + `Precision`），不另起一套 | 参照 `1_gpt2_onnx/src/builder.cpp` 校验一遍，避免历史工程里已解决过的坑重踩 |

---

## 3. 任务分解（草案，取决于 D1）

| ID | 任务 | 依赖 | 预估 |
|---|---|---|---|
| **P3-0** | 把 §0.1 的算子统计固化成可复现脚本（`tools/inspect_onnx.py` 或 host 用例），作为"图变了要知道"的探针 | 无 | 0.5 天 |
| **P3-1** | `BuildFromOnnx` 落地：动态 profile、精度、错误可诊断、输出名校验 | 无 | 1 天 |
| **P3-2** | 与方案 A 的数值对齐用例（真机；含 `ref_output.bin` 三方对照） | P3-1 | 1 天 |
| **P3-3** | 子图识别与断言（D1 选 C）；若 D1 选 B，则在此加替换实现与对照用例 | P3-1 | 1~3 天 |
| **P3-4** | 性能对照：ONNX 路径 vs 方案 A 的构建耗时 / prefill 延迟（用数据支撑 D1 的后续决定） | P3-1 | 0.5 天 |
| **P3-5** | 文档收口（本文件执行结果、PROGRESS、TROUBLESHOOTING） | 全部 | 0.5 天 |

---

## 4. 验收标准（2026-09-25 回填）

- [x] `ctest` 沙箱全绿：**140 用例 0 失败**（GPU 用例自动跳过）。
- [x] 算子统计探针**已接入 `ctest`**（2026-09-25，用户授权修改 `tests/CMakeLists.txt`）：
  注册为 `onnx_graph_probe`，实测**真的执行**（2.66 s，非 skip）。
  环境语义按"缺环境 ≠ 图有问题"设计：缺 Python 解释器则不注册该 test；
  缺 `onnx` 包或缺 `1_gpt2_onnx/gpt2.onnx` 时返回 `SKIP_RETURN_CODE=77` → ctest 报 Skipped。
  负路径已验：`inspect_onnx.py <不存在路径> --check --skip-if-missing` 退出码 = 77。
- [x] ONNX 路径能对 `1_gpt2_onnx/gpt2.onnx` 建出可推理 engine；失败路径有可诊断信息
  （parse 错误逐条打印 + I/O 名校验；失败路径用例见 `docs/phase3_test_plan.md` G1a/G1b）。
- [x] 与方案 A 的 logits 按 D2 对齐（真机：相对偏差 `5.66e-07`，阈值 `1e-5`），
  并与 `ref_output.bin` 三方对照（ONNX/原生对 HF 均为 `9.92e-05`，逐位置 argmax 一致）。
- [x] 子图替换范围按 D1=C 落实：只做**识别 + 断言**（探针的 `--check` 断言注意力 12 块 /
  LayerNorm 25 / 位置编码为学习式），不做替换；不靠人眼。
- [x] 本文件（§P3 执行结果 + 本节）与 `docs/PROGRESS.md` 的执行结果已回填，且与代码一致。

**未完成项汇总**：仅剩 `docs/phase3_test_plan.md` §5 的 G5 / G6（按触发条件处理）。
§4 的五条验收标准**全部达成**；测试缺口 G1a/G1b/G1c/G3/G4/G4b/G7 均已关闭。

---

## 5. 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| 652 MB ONNX 解析慢、占内存 | 每次测试构建引擎分钟级；沙箱里难以跑 | 引擎文件按路径缓存复用；解析类用例进真机；host 探针只做结构统计（可用 onnx python 侧或 TRT parser 的元信息） |
| ONNX 与原生两条图的算子分解不同 → 数值差异大于预期 | 对齐判据可能达不到 D2 | **必须先按实测报告再定阈值**（AGENTS.md §7），不允许为了对齐而放宽 |
| 子图替换（若做 B）引入回归 | 正确性风险 | 替换前后各留一份 engine，做三方对照；替换只在识别与断言稳定之后做 |
| 历史工程 `1_gpt2_onnx` 的经验被忽略 | 重复踩坑 | 开工前先读它的 `src/builder.cpp` 与 README 里的坑 |

---

## 6. 文件清单（草案）

```text
mini_trt_llm/
├── src/core/builder.cpp            # P3-1：BuildFromOnnx 落地（子图替换 TODO → 实现或明确延后）
├── include/mini_trt_llm/plugins/   # P3-3：若做替换，新增子图替换器接口
├── tests/test_gpt2_onnx.cpp        # P3-2：与方案 A 对齐
├── tests/test_onnx_subgraph.cpp    # P3-3：子图识别断言（host 可跑的部分）
tools/
└── inspect_onnx.py                 # P3-0：算子统计探针
```

### P3-0 执行结果（2026-09-25）

落地在 `mini_trt_llm/tools/inspect_onnx.py`（**与 §6 写的根 `tools/` 有一处偏差**：
放 `mini_trt_llm/tools/` 与既有的 `convert/` 并列，工具集中在一处更好找）。

用法与实现要点：

- `python inspect_onnx.py <onnx> [--check] [--json]`，`--check` 与**内置基线**对比，
  图变了就返回非零；
- 基线记录的是**结构性事实**（opset / 输入输出名 / initializer 数 / 关键算子计数），
  不是随手定的期望值：它们是子图识别与对齐判据的前提；
- 基线里还有一组 `absent_ops`（RMSNorm / RoPE / Attention 等）：这类算子一旦出现，
  说明图变了、§0.2 的"本模型无替换对象"结论需要重审——把那条结论变成了**可断言的护栏**；
- 实测通过：`opset 17`、`input_ids[3]`、`logits[3]`、149 个 initializer、
  `LayerNormalization 25 / Tanh 12 / Softmax 12 / Gemm 48 / MatMul 25` 与基线一致。

### P3 执行结果汇总（2026-09-25）

决策全部按 §2 的建议落地（D1=C、D2/D3/D4 按建议），A/B/C/D 四组确认项已冻结。

| ID | 状态 | 产出 |
|---|---|---|
| P3-0 | ✅ | `mini_trt_llm/tools/inspect_onnx.py`（结构基线 + `absent_ops` 护栏） |
| P3-1 | ✅ | `BuildFromOnnx(model_dir, onnx_path, engine_path, subgraph_names)`：复用方案 A 的 profile/精度语义；I/O 名校验（`input_ids` / `logits`）；parse 错误逐条打印；`subgraph_names` 写错即失败 |
| P3-2 | ✅ 完成（真机已验证） | `tests/test_gpt2_onnx.cpp`：ONNX vs 原生 vs HF 参考三方对照 + 逐位置 argmax |
| P3-3 | ✅ | 探针新增子图识别断言：注意力块（Softmax 12 / MatMul 25 / Transpose 48）、LayerNorm 25、位置编码为学习式（无 RoPE 类算子） |
| P3-4 | ✅ 完成 | 目标按计划原文是"**用数据支撑 D1 的后续决定**"= 记录基线数据，已达成：构建 13.61 s / 16.28 s，prefill(4 token) 4.72 ms / 6.25 ms。**"谁更快"从未属于本阶段验收项**，其结论未定（两次运行方向相反、±25% 小于构建间噪声）→ 归属 `docs/future_iterations.md` §10.2，触发时先按缺口 G6 建测量方法 |
| P3-5 | ✅ | 本节 + PROGRESS + `requirements.txt` 补 `onnx` |

**D4 的历史工程核对结论**（`1_gpt2_onnx/src/builder.cpp`）：

- 它当年用的是**单个** profile，取值 min `[1,1]` / opt `[1,64]` / max `[4,512]`——
  与 `EngineBuilder::Config` 的 prefill 默认值**完全一致**，所以"复用既有能力"是复用而非改写；
- 它记过一条坑的注释："`Dims2`，不是 `Dims4`"（该 ONNX 的 `input_ids` 是 2-D）；
- 因此 ONNX 路径只挂 **prefill 那一组** profile：该图是整段前向、无 KV 输入，
  多挂一组 decode profile 只会让 TRT 白编译一份。

**为什么 ONNX 构建要求 `model_dir`**（对 §6 文件清单的一处说明）：profile 规则按
`config.json` 的 `architecture` 选择，与方案 A **同一套**语义。若只接 `onnx_path` 而从
动态轴猜范围，两条路的 profile 可能不同，"ONNX 与原生对齐"这个判据就失去意义。

**真机实测（2026-09-25，`Gpt2OnnxTest.MatchesNativeBuildOnSamePrompt`）**：

```
ONNX vs 原生 : max_abs 6.10e-05 / max_abs÷max|ref| 5.66e-07 / cosine 1
ONNX vs HF   : max_abs 9.92e-05 / 9.19e-07 / cosine 1
原生  vs HF  : max_abs 9.92e-05 / 9.19e-07 / cosine 1
prefill(4 token): ONNX 4.545 ms  vs  原生 3.717 ms
```

判读：

1. **对齐判据通过**：方案 B 与方案 A 的相对偏差 `5.66e-07`，比冻结阈值 `1e-5` 还宽约 18 倍余量；
   两者逐位置 argmax 一致。
2. ONNX 与原生对 HF 的偏差**数值完全相同**（`9.91821e-05`）——说明两条路各自的误差都是 1e-6 量级，
   ONNX 与原生之间的差异（`6.10e-05`）只来自图的分解方式（`Gemm` vs `MatMul`、softmax 实现不同），
   完全在预期内，未触发"阈值超标"分支。
3. **性能数据不能支撑结论**（此处已修正一次误读）：第一次运行显示 ONNX 慢 22%，
   删除缓存引擎后重测显示 ONNX 快 24%——**方向翻转，差异小于构建间噪声**
   （TRT 的 tactic 选择依赖构建时机器状态）。因此"要不要做子图替换"仍属**未定**，
   需要先建立可复现的测量方法（缺口 G6）；相关判断依据记入 `docs/future_iterations.md` §10.2。

**沙箱状态**：`ctest` 133 用例 0 失败（新增的 ONNX 对拍用例在沙箱跳过）。

**状态：Phase 3 完成**（测试口径与未覆盖缺口见 `docs/phase3_test_plan.md`）。

---

*文档版本：v1.0（开工前，待确认 §2 决策项）*
