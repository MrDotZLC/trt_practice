# Phase 3 测试计划：GPT-2 ONNX + Plugin（方案 B）

> **定位**：Phase 3 的**任务与决策**见 `docs/phase3_development_plan.md`；本文档定义
> **测试分层、用例、判据出处与缺口**。
>
> **状态说明**：Phase 3 的用例在本文档成文之前就已实现并真机通过（开发计划 §P3 执行结果）。
> 因此本文档的价值是三件事：把**判据的出处**写清（交接需要）、把**尚未覆盖的缺口**列全
> （下一次要做的是这些，而不是重复已通过的）、给出**可复现的执行方式**。
>
> 判定一律区分"**已验证**"与"**未验证**"——未跑过的不写"通过"（AGENTS.md §7）。

---

## 1. 被测链路

```
config.json + gpt2.onnx
        │
        ├─► tools/inspect_onnx.py --check        # 静态：图结构与子图形态（不需要 GPU）
        │
        └─► EngineBuilder::BuildFromOnnx(model_dir, onnx_path, engine_path, subgraph_names)
                    │  ├─ 参数/文件校验（子图名、config、ONNX 可读）
                    │  ├─ nvonnxparser 解析 + 逐条错误打印
                    │  ├─ I/O 契约校验（input_ids / logits）
                    │  ├─ profile：按 architecture 选规则 → 只挂 prefill 一组
                    │  └─ buildSerializedNetwork → .engine
                    ▼
              Engine(engine_path) → SetInputShape → SetTensorAddress → Enqueue → logits
                                                              │
                                                              └─► 与方案 A 的 engine、
                                                                  `ref_output.bin` 三方对照
```

---

## 2. 测试分层

| 层 | 覆盖什么 | 在哪跑 | 现状 |
|---|---|---|---|
| **L0 静态/结构** | 图的算子分布、I/O 名、子图形态（注意力块 / LayerNorm / 位置编码） | **沙箱**（Python，无需 GPU） | ✅ 已实现并实测通过 |
| **L1a 参数与文件校验** | `subgraph_names` 写错、`config.json` 缺失或非法、ONNX 不可读 | **沙箱**：这三类都被刻意排在 `createInferBuilder` **之前**，因此不需要 GPU | ✅ 已补齐（`tests/test_gpt2_onnx_error_paths.cpp`，4 条） |
| **L1b 图契约校验** | 图里没有 `logits` 输出、没有 `input_ids` 输入 | 真机（解析需要 `createInferBuilder`） | ⚠️ 输入侧已覆盖（G1b，用 `resnet18.onnx`）；输出侧仍缺（G1c） |
| **L2 数值** | ONNX 路径 vs 方案 A vs HF 参考（`ref_output.bin`） | 真机 | ✅ 已验证 |
| **L3 性能** | 构建耗时、prefill 延迟（vs 方案 A） | 真机 | ⚠️ **延迟已验证；构建耗时未测到**（见缺口 G2） |

---

## 3. 用例清单（现状）

| 用例 | 覆盖 | 判据 | 出处 | 状态 |
|---|---|---|---|---|
| `onnx_graph_probe`（ctest） | L0：opset / I-O 名 / initializer 数 / 关键算子计数 / `absent_ops` 护栏 | 与内置基线逐项相等 | 开发计划 §0.1 / G7 | ✅ **已接入 ctest** 并实测执行 |
| `inspect_onnx.py --check`（子图识别） | L0：注意力 12 块（Softmax 12 / MatMul 25 / Transpose 48）、LayerNorm 25、位置编码为学习式 | 计数相等 + 无 RoPE 类算子 | D1=C / A4 | ✅ 实测通过 |
| `Gpt2OnnxTest.MatchesNativeBuildOnSamePrompt` | L2：ONNX vs 原生 vs HF 三方对照 + 逐位置 argmax | `cosine > 0.999999`、`max_abs/max\|ref\| < 1e-5`、argmax 一致 | D2 + A1–A5 冻结 | ✅ 真机通过 |
| 同上（诊断输出） | L3：构建耗时 + prefill(4 token) 延迟 | 不作判据，仅记录 | P3-4 | ⚠️ 已测两次，但**两次结论相反**（见 §3.1） |

| `Gpt2OnnxTest.Fp16PathsAgree`（G3） | L2：FP16 下两条路是否一致 | `cosine > 0.999`、相对界 `< 5e-3`（D6 的 FP16 档）+ 逐位置 argmax 一致 | D6 | ✅ 真机通过（**实测值未采集**，阈值未收紧） |
| `Gpt2OnnxTest.MatchesAcrossProfileShapes`（G4 + G4b） | L2：`(batch,seq) ∈ {(1,1),(1,64),(1,512),(2,4),(2,64)}` 两条路对照 | 同 D2（`cosine > 0.999999`、相对界 `< 1e-5`）+ 逐行 argmax | D2 | ✅ 真机通过（**实测值未采集**，阈值未收紧） |

**实测值（2026-09-25）**：

```
ONNX vs 原生 : max_abs 6.10e-05 / max_abs÷max|ref| 5.66e-07 / cosine 1
ONNX vs HF   : max_abs 9.92e-05 / max_abs÷max|ref| 9.19e-07 / cosine 1
原生  vs HF  : max_abs 9.92e-05 / max_abs÷max|ref| 9.19e-07 / cosine 1
```

---

### 3.1 L3 的两次测量（结论不可复现）

| 运行 | 构建 ONNX / 原生 | prefill ONNX / 原生 | ONNX vs HF（max_abs / 相对） |
|---|---|---|---|
| A（复用缓存引擎） | 未测到 / 未测到 | 4.545 ms / 3.717 ms（ONNX 慢 22%） | 9.92e-05 / 9.19e-07 |
| B（两引擎均新构建） | 13.61 s / 16.28 s | 4.716 ms / 6.250 ms（ONNX 快 24%） | 7.63e-05 / 7.07e-07 |

**两条结论**：

1. **性能方向翻转**（±25%），说明差异小于"构建间噪声"。已知机制：TRT 的 kernel tactic
   选择依赖构建时机器状态，重新构建会选到不同 tactic。**因此两次数据都不能用来判断
   "ONNX 路径更快还是更慢"，更不能用来支撑"要不要做子图替换"的结论。**
2. **末位精度也随构建变化**（`9.92e-05` ↔ `7.63e-05`），但相对量级始终在 `7e-7 ~ 1e-6`，
   相对冻结阈值 `1e-5` 仍有 **约 14 倍余量** → L2 的判据不受影响、结论稳定。

**由此产生的新缺口 G6**：L3 目前**没有可复现的测量方法**。要能支撑任何结论，至少需要：
固定机器状态、同一 session 内多次构建（≥3）与多次推理（≥20），并**报告离散度**
（例如中位数与极差），而不是报一个点值。在此之前，`docs/future_iterations.md` §10.2 的
性能结论一律视为"未定"。

## 4. 判据与出处（阈值纪律）

| 判据 | 出处 | 余量 |
|---|---|---|
| `cosine > 0.999999` | D6（FP32 `cosine ≥ 0.9999`）+ Phase 2 实测收紧（`cosine` 实测 = 1.0） | 实测值即上界 |
| `max_abs / max\|ref\| < 1e-5` | 同上：D6 原文为 `1e-3`，按 Phase 2 实测（`9.19e-07`）收紧到 10 倍余量 | ONNX vs 原生实测 `5.66e-07` → 约 18 倍余量 |
| 逐位置 argmax 一致 | 「token 是产品的语义判据，数值是定位判据」 | 一致 |

**超标时的处置（§7 冻结）**：先量"与正确性无关的差异"（同一 engine 重复运行、不同 profile
的差异、两条路的分解差异），再拿数据找用户定阈值——**不允许为了让用例变绿而放宽**。
本轮未触发该分支（三条判据均大幅达标）。

---

## 5. 缺口（尚未覆盖，下一次要做的正是这些）

| ID | 缺口 | 为什么重要 | 建议做法 |
|---|---|---|---|
| **G1a** | ~~L1a 失败路径无用例~~ **已补齐**（2026-09-25） | —— | `tests/test_gpt2_onnx_error_paths.cpp`：4 条（子图名写错 / 缺 config.json / config 缺 architecture / ONNX 不可读），**沙箱执行**，并断言"失败不留 engine 文件" |
| **G1b** | ~~L1b 的"输入名不符"无用例~~ **已补齐**（2026-09-25） | —— | `RejectsGraphWithForeignIoNames`：直接用仓库已有的 `0_resnet18_onnx/resnet18.onnx` 当"外来 I/O 名"夹具（它的 I/O 是 `input`/`output`），真机执行（解析需要 GPU）。**不需要造 ONNX** |
| **G1c** | ~~L1b 的"输出名不符"无用例~~ **已补齐**（2026-09-25） | —— | `RejectsGraphWithoutLogitsOutput`：用新增的 `tools/make_tiny_onnx.py` 生成 `input_ids → not_logits` 的单节点图（Python 造夹具，避开手写 protobuf），真机执行（解析需 GPU）；缺 python3/onnx 时跳过 |
| **G2** | ~~构建耗时未真正测到~~ **已补齐**（2026-09-25，运行 B） | —— | 已删除缓存引擎后重测：ONNX 13.61 s / 原生 16.28 s（原生侧计时也一并补上，避免单边数据） |
| **G6** | **L3 无可靠测量方法**：两次运行结论相反（见 §3.1） | 性能数据不可复现时，任何"谁更快"的结论都是从噪声里读出来的——本次已犯过一次 | 固定机器状态 + 同一 session 内 ≥3 次构建、≥20 次推理，报中位数与极差；先做方法，再谈结论 |
| **G3** | ~~ONNX 路径只测了 FP32~~ **已补齐**（2026-09-25，真机通过） | —— | `Gpt2OnnxTest.Fp16PathsAgree`：FP16 下两条路对照 + 逐位置 argmax；阈值用 D6 的 FP16 档。**实测值未采集**，故未收紧（收紧需要一次带诊断输出的运行） |
| **G4** | ~~ONNX 路径只测了 `[1,4]`~~ **已补齐**（2026-09-25，真机通过） | —— | `Gpt2OnnxTest.MatchesAcrossProfileShapes`：`seq ∈ {1, 64(opt), 512(max)}` 两条路对照 + 逐位置 argmax。**实测值未采集**，阈值未收紧。**batch 维仍未覆盖**（profile 允许 1..4）——留作 G4b |
| **G5** | 子图识别只做"计数"，未做"拓扑/邻接" | A4 已明确只做三类计数级识别；计数相同但连接关系不同（例如把 LayerNorm 挪到别处）不会被发现 | 若将来要做替换（`docs/future_iterations.md` §10.2），再升级为邻接级识别 |

---

## 6. 执行方式

```bash
# L0：图结构与子图识别（沙箱即可，不需要 GPU）
python3 mini_trt_llm/tools/inspect_onnx.py 1_gpt2_onnx/gpt2.onnx --check

# 沙箱全量（含 host 用例；GPU 用例自动跳过）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# L2/L3：真机（需确认后执行）
./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='Gpt2OnnxTest.*'

# 想测真实构建耗时（缺口 G2）：先清掉缓存的引擎，否则用例会复用
rm -f /tmp/mini_trt_llm_gpt2_onnx.engine
./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='Gpt2OnnxTest.*' \
  | grep 'P3-4'
```

> `rm -f /tmp/...engine` 只删测试自己产生的临时引擎文件，不涉及仓库内容；
> 按 AGENTS.md §0.2，这类删除仍会先向你说明再执行。

---

## 7. 通过标准（Phase 3 整体）

- [x] L0 探针 `--check` 通过（结构基线 + 三类子图识别）。
- [x] 沙箱 `ctest` 全绿：**141 用例 0 失败**（GPU 用例自动跳过；含新增的 `onnx_graph_probe`）。
- [x] L2 三方对照通过，三条判据均大幅达标。
- [x] L3 构建耗时已测到（13.61 s / 16.28 s）——但**测量方法不足以支撑结论**（缺口 G6）。
- [x] G1a 失败路径（4 条，沙箱）——已补齐并通过。
- [x] G1b 输入名不符（用 resnet18.onnx 当夹具）——已补齐，真机执行。
- [x] G3（FP16）/ G4（shape）——已补齐，真机通过（实测值未采集）。
- [ ] G1c（输出名不符）/ G4b（batch 维）/ G5（拓扑级识别）/ G6（性能测量方法）—— 未完成，登记在 §5。

---

*文档版本：v1.0（Phase 3 完成后补记；§5 的缺口按 AGENTS.md §7 显式标注为未验证）*
