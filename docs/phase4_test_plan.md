# Phase 4 测试计划：ResNet18（CV 路径）

> **状态**：P4-0 产出（2026-09-25）；用例随 P4-1～P4-8 落地后在本文件 §7 回填结果。
>
> **与开发计划的分工（唯一来源原则，`PROGRESS.md` §2.13）**：
> - **任务、顺序、依赖、验收、决策（D1～D4）** → `docs/phase4_development_plan.md`（本文件不复述）；
> - **历史工程的事实与结论** → 同上 §1；
> - **分层、用例清单、判据出处、执行口径、覆盖缺口** → 本文件；
> - 排查过程 → `docs/TROUBLESHOOTING.md`；整体进度 → `docs/PROGRESS.md`。
>
> 已确认的决策（细节见开发计划 §0.5）：**D1** 先 ONNX 再原生 ｜ **D2** INT8 纳入、走 Q/DQ ｜
> **D3** 基线用 torchvision FP32 ｜ **D4** 前处理由 `CVRunner` 负责。

---

## 1. 分层与环境

| 层 | 测什么 | 环境 | 为什么在这一层 |
|---|---|---|---|
| **L0 host 契约** | `config.json` 解析；**转换产物的权重名集合与形状**；前处理的纯函数 | **沙箱** | 不需要 CUDA；转换/前处理是后面所有数值结论的地基，越早进 CI 越好 |
| **L1 建网** | 原生网络与 ONNX 网络的 I/O 契约（名字/形状/动态轴）、层数与关键 shape 冒烟、缺权重必须失败 | 真机（`createInferBuilder` 需要 CUDA） | 建网错误绝大多数只在真机暴露（Phase 2 的教训） |
| **L2 数值** | 各路径 vs torchvision 基线；原生 vs ONNX 互拍；FP16；INT8 | 真机 | 需要真实引擎执行 |
| **L3 端到端** | `CVRunner::Infer`（batch 1/8/16）、超范围拒绝、`Benchmark` 可跑 | 真机 | Runner 是产品路径，与"引擎能跑"是两件事（Phase 2 的 E2e 同理） |
| **L4 性能（可选）** | batch 扫描的中位数与极差 | 真机 | 只有要报性能时才做；口径见 `future_iterations.md` §11 的 G6 |

---

## 2. 用例清单

**这张表是索引**：用例名以代码为准（`tests/test_resnet18_*.cpp`、`tests/test_cv_*`）。
编号 `R<层>.<序>` 供开发计划与结果表引用。

### L0（沙箱）

| 编号 | 用例 | 验证点 | 判据 |
|---|---|---|---|
| R0.1 | `ResNet18ConfigTest.LoadsCnnConfig` | `models/resnet18/config.json`：`model_type=resnet18`、`architecture=cnn`、`weight_map` 非空 | 解析成功 + 字段值正确 |
| R0.2 | `ResNet18WeightContractTest.ConvertedArtifactCoversEveryWeight` | **三集合相等**：safetensors 的键 == `weight_map` 的键 == 由 ResNet18 结构推出的 42 个期望名；且每个名字都能经**真实路径**（`WeightLoader::GetWeight`）取到 | 集合全等；`weight_map` 的值 == 键（identity 是本项目当前的产物形态）。⚠️ 反向覆盖要**自己读 safetensors 文件头**（`GetTensorNames()` 是空实现，PROGRESS §5.2），否则只能单向验"map 里的键都存在"，发现不了"文件里多出没人引用的张量" |
| R0.2b | `ResNet18WeightContractTest.ShapesAreSelfConsistent` | 20 个卷积的 weight 必须 4 维、bias 长度 == 输出通道数；`fc.weight` 与 `config.hyper_params.num_classes` 一致 | 全过；`fc.weight = [1000, 512]`（形状错位最难从数值反查） |
| R0.2c | `ResNet18WeightContractTest.SourceMetadataMatchesReality` | `source` 元数据必须与产物自洽：`opset`/`tensor_count`/`conv_count`/`batch_norm`/`fc_weight_layout`，且布局声明与文件里的实际形状对得上 | 逐字段相等；`fc.weight` 第 0 维 == 1000（与 `out_in_transB` 声明一致） |
| R0.2d | ctest `resnet18_convert_selftest`（`onnx_to_mini_trt_llm.py --self-test`） | **护栏有没有牙齿**：把真模型按 5 种方式改坏，逐个确认自检会拦（未消费的 initializer / 逻辑名冲突 / bias 长度不符 / 缺 fc / 不支持的带权算子） | 5 条全部"自检生效"；任一条没拦住即失败 |
| R0.2e | 转换**可复现性**（脚本级，见 §7 记录） | 同输入两次转换必须逐字节相同 | `model.safetensors` 与 `config.json` 两次 SHA256 全一致（且与仓库产物一致） |
| R0.3 | `CvRunnerPreprocessTest.MatchesBaselineNormalization` | 前处理公式 `(pixel/255 - mean[c]) / std[c]`，入参 **NCHW `[0,255]`**（**没有** HWC→CHW 这一步：形参名与 P4-1 契约输入都是 NCHW） | 与 P4-1 落盘的 `ref_pixels_b8.normalized` 逐元素一致；**必须用 batch=8 测**（batch=1 时通道下标算错也看不出来，见 #23.1）。**实测 `max_abs = 0`** |
| R0.4 | `CvRunnerPreprocessTest.HandlesPerChannelAndBoundaries` | 三通道 mean/std **逐通道不同**、像素边界 0 与 255、参数非法返回空 | 逐通道正确；把通道顺序写反必须产生不同结果 |
| R0.5 | `OnnxIoContractTest.CnnUsesInputAndOutput` | ONNX I/O 契约映射：`cnn` → `input`/`output` | 逐字段断言（**P4-2 已实现并通过**） |
| R0.6 | `OnnxIoContractTest.LlmKeepsInputIdsAndLogits` | 默认分支必须保持 Phase 3 的 `input_ids`/`logits`（含 `encoder_decoder` 与未知 architecture） | 三个 architecture 逐条断言（**P4-2 已实现并通过**） |
| R0.7 | `ResNet18OnnxBuildTest.RejectsUnknownSubgraphName` | 未登记的子图名必须被拒（该校验在 `createInferBuilder` **之前**，所以能在沙箱跑） | 返回 false（**P4-2 已实现并通过**）。⚠️ 名字必须是白名单外的：第一版误用 `attention`（已知名）导致用例走到建 builder 那步报 CUDA 错 |
| R0.8 | `ResNet18BaselineTest.RampInputMatchesFormulaBitExact` | 基线的 ramp 输入与公式**逐位一致**（公式在 Python 与 C++ 各一份，防漂移） | 全部元素 `==`，0 处不符（**P4-1 补验，沙箱通过**） |
| R0.9 | `ResNet18BaselineTest.PixelsNormalizationMatchesFormula` | `normalized` == 对 `contract_input` 施加 ImageNet mean/std（C++ 独立算） | `max_abs < 1e-6`；**实测 = 0**（逐位相同）。同时校验像素质落在 `[0,255]`（**P4-1 补验**） |
| R0.10 | `ResNet18BaselineTest.MetaArgmaxMatchesLogits` | 元数据 `argmax_per_sample` 必须能从 logits 现算出来 | 两套基线逐行相等（**P4-1 补验**） |
| R0.11 | `CvRunnerTest.RejectsNullEngineWithoutCuda` | **构造期失败必须能被 `ok()` 查到**：空引擎时 `ok()==false`、`Infer` 返回空、`Benchmark` 返回零值统计（而不是崩或给假数字） | 三条断言全过；**不需要 GPU**（构造函数在碰 CUDA 前就返回），所以放在沙箱（**P4-3 补验**） |

> R0.3/R0.4 是**纯 host 函数**用例，不碰 CUDA——这是"把能在沙箱跑的都摘出来"的既有纪律
> （Phase 1.5 L1 曾吃过"以为能在沙箱跑、其实要 CUDA"的亏）。

### L1（真机）

| 编号 | 用例 | 验证点 | 判据 |
|---|---|---|---|
| R1.1 | `ResNet18NetworkBuildTest.BuildsWithExpectedIo` | 原生网络的输入名 `input` / 输出名 `output`、形状 `[B,3,224,224]` / `[B,1000]`、动态轴只在 batch | 名字与维数逐条断言 + 逐层 `getDimensions()` 冒烟（延迟报告陷阱，见 Phase 2） |
| R1.2 | `ResNet18NetworkBuildTest.MissingWeightFailsTheBuild` | `weight_map` 指向不存在的键必须建网失败 | 返回 false，不产出 engine |
| R1.3 | `ResNet18OnnxBuildTest.BuildsFromCnnConfig` | `BuildFromOnnx` 在 `architecture=cnn` 下接受 `input`/`output` 图 | 引擎构建成功 + profile 挂上 |
| R1.4 | `ResNet18OnnxBuildTest.RejectsLlmConfigForCnnGraph` | 拿 `decoder_only` 的 config 配 ResNet18 图必须被拒 | 返回 false，错误信息指向 I/O 契约。⏳ **未落地**（要解析 ONNX → 需 CUDA），留给 P4-6 |
| R1.5 | `ResNet18OnnxProfileTest.AcceptsBatchRangeAndRejectsOutOfRange` | CV profile（min/opt/max = 1/8/16）：范围内 **真跑** batch 1/8/16，范围外（17）**显式失败** | 三次 `Enqueue` 都返回 `[batch,1000]`；`SetInputShape(17,...)` 返回 false。**P4-2 已实现并通过**。⚠️ 输入按目标 batch **重新生成**，不要对别的 batch 的张量切片（#22.2 的越界教训） |
| R1.5b | `ResNet18NativeProfileTest.AcceptsBatchRangeAndRejectsOutOfRange` | 同上，但跑在**原生引擎**上 | 同上。**P4-5 补验**：profile 挂在网络输入维度上，而原生维度是手写的、与 ONNX 图无关——"同一段 profile 代码"不足以当证据 |

### L2（真机）

| 编号 | 用例 | 验证点 | 判据（出处见 §3） |
|---|---|---|---|
| R2.1 | `ResNet18OnnxAccuracyTest.MatchesBaselineOnRampInput` | ONNX 引擎 vs `ref_ramp_b8.bin`（**未归一化**的合成 ramp，直接喂） | FP32 阈值 + 逐个样本 argmax 一致 |
| R2.2 | `ResNet18OnnxAccuracyTest.MatchesBaselineOnPixels` | ONNX 引擎 vs `ref_pixels_b8.bin`（喂**归一化后**的张量） | 同上；注意此路 argmax 判别力有限（§4.1） |
| R2.3 | `ResNet18NativeAccuracyTest.MatchesOnnxPath` | 原生 vs ONNX（**逐位相同的权重**、同一输入） | **阈值 `max_abs < 1e-5`**：实测 `1.07288e-06`（rel 1.19e-7），取约 10 倍留构建间余量。**刻意比 R2.4 紧一个数量级**——这条路没有 BN 折叠那项差异，用同一把尺子等于白白放宽 |
| R2.4 | `ResNet18NativeAccuracyTest.MatchesBaseline` | 原生 vs torchvision 基线 | 同 R2.1 的阈值与 argmax 判据 |
| R2.5a | `ResNet18Fp16PathTest.OnnxEngineMatchesFp32Baseline` | FP16 引擎（ONNX 路径）vs FP32 基线（ramp） | `max_abs < 0.1` + **argmax 逐样本一致**（双判据）。**P4-6 已实现并通过** |
| R2.5b | `ResNet18Fp16PathTest.NativeMatchesOnnxInFp16` | 原生 FP16 vs ONNX FP16（同精度、同一份权重）+ **打印三角证据** | `max_abs < 0.05` + argmax 一致；同时打印两条 FP16 各自离原生 FP32 的距离（**P4-6**） |
| R2.5c | `ResNet18Fp16PathTest.CvRunnerOnFp16EngineMatchesFp32Baseline` | CVRunner 驱动 FP16 引擎（pixels 输入） | 同 R2.5a 口径；验"调用方接口在 FP16 下不变"（**P4-6**） |
| R2.5d | `ResNet18Fp16PathTest.NativeEngineMatchesFp32Baseline` | **原生 FP16 vs 外部基线**（torchvision FP32，ramp） | 同 R2.5a 口径。**P4-6 补验**：此前原生 FP16 与外部真值之间只有传递推断（0.0076+0.031），而"两条路共享同一个 bug"恰好能躲过互拍、躲不过外部基线 |
| R2.5e | R2.5b 内的 **I/O 契约比对** | 两条 FP16 引擎的 `input`/`output` 名字与**声明精度**必须一致 | 逐字段相等；实测两条都是 `input=FP32, output=FP32`（弱类型网络下 FP16 引擎的 I/O 常被 TRT 定成 FP32，见 #18）——调用方靠声明精度决定喂什么，不一致则"同一份调用代码"不成立（**P4-6 补验**） |
| R2.6 | `ResNet18Int8EngineTest.IsActuallyInt8` + `ResNet18Int8AccuracyTest.{RampInputIsOutOfDistribution,Top1AgreementOnRealImages}`（D2 选②） | Q/DQ INT8 引擎：① 层信息自证在跑 INT8；② ramp 只记录不判（分布外）；③ 真实图按**分层**判 | ① ≥20 层含 `Format/Datatype: Int8` 且 ≥1 个 `i8i8` tactic（对照 FP32 引擎为 0）；③ **主判据：FP32 余量子集（margin≥5）一致率 ≥90%**（实测 12/12），整体一致率 ≥30% 仅作下界（实测 38.3%） |

### L3（真机）

| 编号 | 用例 | 验证点 | 判据 |
|---|---|---|---|
| R3.1 | `CvRunnerTest.InferMatchesBaseline` | `CVRunner::Infer` 在 batch 1 / 8 上端到端 | 与基线一致（同 L2 判据）+ 输出 shape `[B,1000]` |
| R3.2 | `CvRunnerTest.RejectsBatchOutsideProfile` | batch 0 / 17 / 超大 | **显式失败**（对应 E3.3 的纪律：宁可报错，不许用错形状跑出结果） |
| R3.3 | `CvRunnerTest.BenchmarkReportsFiniteStats` | `Benchmark` 可跑且统计字段有效（mean/p50/p99 > 0、p50 ≤ p99） | 只验"统计自洽"，**不把性能数字当判据** |
| R3.4 | `CvRunnerTest.RejectsMeanStdSizeMismatch` | `mean`/`std` 长度与输入通道数不符时，必须在**构造期**被拒（`ok()==false`），而不是拿错常量算出垃圾 | 太短（1 个）与太长（4 个）都返回 false；`Infer` 随后返回空（**P4-3 补验**） |
| R3.5 | `CvRunnerTest.InferMatchesBaselineOnNativeEngine` | CVRunner 驱动**原生引擎**（与 R3.1 同一套前处理契约） | 与 P4-1 基线一致；阈值同 R3.1（`1e-4`）。**P4-5 补验**——"同名 I/O"不等于"同契约"，两套引擎的维度来源不同 |

---

## 3. 判据与出处（每条都要能回答"凭什么"）

| 判据 | 出处 | 关键说明 |
|---|---|---|
| torchvision FP32 基线数值 | P4-1 产出：`ref_output.bin` + `ref_meta.json` | 元数据必须含：生成脚本、torch/torchvision 版本、**权重文件 SHA256**、输入来源与 shape、逐样本 argmax |
| **FP32 阈值** | **已定量**：`max_abs < 1e-4` ≈ 最大无关差异的 5 倍 | 无关差异实测：BN 折叠 **1.9e-5**、torch FP32 CPU-vs-GPU **7.6e-6**；观测值 TRT FP32 vs torch 基线 **9.5e-6**（ramp）/ **1.3e-5**（pixels）。三处同量级。阈值仍能拦住"精度选错"：FP16 引擎实测 **3.3e-2**（大 330 倍）。踩坑与完整推导见 `TROUBLESHOOTING.md` #21 |
| 逐个样本 argmax 一致 | 历史工程判据（`0_resnet18_onnx/src/main.cpp:156-158`）+ 分类任务语义 | 与数值阈值**并列**，不互相替代 |
| FP16 | **已按实测另定两档**（不再沿用 D4）：① **FP16 引擎 vs FP32 基线** `max_abs < 0.1`（≈ 实测上界 0.027 的 4 倍）；② **两条 FP16 路径互拍** `max_abs < 0.05`（≈ 实测 0.0076 的 6.6 倍）。两档都**并列要求 argmax 逐样本一致** | **阈值不跨精度复用**（`PROGRESS.md` §7）。FP16 的噪声底实测 0.018~0.027（纯舍入），因此 D4 的 `rel < 1e-3` 对"网络级 FP16 vs FP32"根本不适用。完整测量表、三角证据与"为什么这次放宽合规"见 `TROUBLESHOOTING.md` #26 |
| INT8 | 需先测再定（argmax 一致 + top-1 一致率） | 校准集用现成的 500 张真实图；**不引用历史数字**（历史没留） |
| 前处理一致性 | R0.3 | 与基线**同一套参数**；这是"误差归因"的前提（否则数值差会被误判成引擎错） |
| batch 超范围 | 必须显式失败 | 与 Phase 1.5 的 E3.3 同一纪律 |

**若 FP16/INT8 不达标**：按 AGENTS.md §7 只允许三选一（继续查并标"原因未知" / 证明期望值本身错并给出
独立依据 / 标"已知失败 + 原因未知"保持红色）。**禁止**调阈值、删断言、降级成打印。

---

## 4. 输入与基线

| 输入集 | 来源 | 用途 |
|---|---|---|
| 合成 ramp | `input[i] = (i % 255) / 255.f`（与历史工程 `src/main.cpp:120-123` 同式），**不归一化** | 用于 **L2 引擎级**对拍（R2.1/R2.4）；不依赖任何图像文件 |
| 像素质 pixels | 由 `0_resnet18_onnx/calib_data/` 前 8 张**反归一化**回 `[0,255]` 的 float32 NCHW | 用于 **L3 `CVRunner`** 对拍（R3.1）与 **R0.3 前处理公式**交叉验证；`CVRunner` 的输入契约就是它 |

> **两套输入的归一化状态不同，不能混用**：ramp 未归一化（历史工程直接喂原始值），
> pixels 需要消费方按 ImageNet mean/std 归一化（D4）。混淆这两者会得到"engine 错了"的假象。

### 4.1 已产出的基线产物（P4-1，2026-09-25）

| 产物 | 位置 | 大小 |
|---|---|---|
| 基线 logits（ramp / pixels） | `models/resnet18/ref_{ramp,pixels}_b8.bin` | 32 KB each |
| 元数据（含权重与产物 SHA256、逐样本 argmax） | `models/resnet18/ref_{ramp,pixels}_b8.meta.json` | 1.7 KB each |
| 契约输入张量（ramp 的原始输入 / pixels 的 `[0,255]` 输入） | `models/resnet18/inputs/*.contract_input.f32.bin` | 4.8 MB each |
| pixels 的**归一化后**张量（Python 侧前处理输出，供 R0.3 对拍） | `models/resnet18/inputs/ref_pixels_b8.normalized.f32.bin` | 4.8 MB |

**版本控制口径（沿用仓库既有规矩）**：`.gitignore` 忽略 `*.bin`，且仓库**不跟踪任何
`*.onnx` / `*.safetensors` / `calib_data/`**（GPT-2 的模型目录与 `0_resnet18_onnx/` 同样是本地件）。
因此这些 `.bin` 产物**不入库**；可入库的是 `.meta.json`（记录 SHA256，用来回答"基线有没有被改过"）。
**跑 R2.x / R3.x 的前提**：本机存在 `models/resnet18/`、`0_resnet18_onnx/calib_data/`
与 torchvision 权重缓存（`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`）——
缺了就该 **skip**（与 GPT-2 用例缺 `models/gpt2` 时跳过的口径一致），不是失败。

**基线判别力的一处已知限制**：pixels 的 8 个样本里有 5 个 argmax = 1、其余为 376/392/5
（这些图来自 tiny-imagenet 的 64×64 放大，本身分类就不稳）。所以
**argmax 一致这条判据在真实图上判别力有限**，数值阈值才是主判据；argmax 用来兜"整体跑偏"。

**基线自证要求**（同 `PROGRESS.md` §2.13 对参考实现的纪律）：

1. 同一脚本跑两次，`ref_output.bin` **逐位一致**；
2. 权重来源固定（本地缓存 `resnet18-f37072fd.pth`，**生成基线不需要联网**）；
3. 输入张量本身要落盘（否则无法回答"这个 argmax 是对哪张图算的"）。

> ⚠️ **真实图数量有限 ≠ 数据集精度**：用几张图只能做 sanity，
> **不得**据此声称 top-1 精度。真正的数据集评估不在本阶段范围（见 §5）。

---

## 5. 执行方式

```bash
# 沙箱：L0（前处理与转换产物的 host 契约；GPU 用例会跳过且带探测结果）
cmake --build build -j$(nproc) && ctest --test-dir build --output-on-failure

# 真机：L1–L3（必须带闸门，否则无设备时静默跳过 = 白跑）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
  --gtest_filter='ResNet18*Test.*:CvRunnerTest.*'

# 基线生成（P4-1；只跑一次，产物入库或按 .gitignore 口径处理）
python3 scripts/ref_resnet18.py --input ramp --output models/resnet18/ref_output.bin
```

**注意**：

1. 引擎缓存**只按路径名区分、不随代码或开关失效**——改了建图/开关后必须清理（`TROUBLESHOOTING.md` #19）；
2. 新增的 GPU 跳过点必须用 `MINI_TRT_SKIP_IF_NO_CUDA` **宏**（`GTEST_SKIP` 封装成函数会只退出函数，见 `PROGRESS.md` §2.13）；
3. 真机全量约需数分钟（每个用例都要建引擎；ResNet18 比 GPT-2 小得多）。

---

## 6. 覆盖缺口（本阶段已知不做，登记于此避免失传）

| 缺口 | 影响 | 触发条件 |
|---|---|---|
| **数据集级精度未评估** | 只能说"与 torchvision 一致"，不能说"top-1 = X%" | 需要真实业务精度结论时（要 ImageNet 验证集与完整评估脚本） |
| **动态分辨率** | 空间维必须固定 224×224 | `AddCvOptimizationProfile` 目前显式拒绝；见 `future_iterations.md` §2.4 |
| **图像解码/缩放** | 只吃"已解码并 resize 好的 NCHW" | 要接真实图片文件/摄像头时（stb/libjpeg + GPU resize） |
| **多模型 / 多输入 CV** | 只覆盖 ResNet18 单输入单输出 | 接入 ViT/Detectron 类模型时 |
| **INT8 的 per-channel vs per-tensor 细节** | D2 选②时只验 argmax 与一致率，不深究量化方案 | 要压 INT8 精度时 |
| **性能数字** | L4 不做则无性能结论 | 要回答"INT8 到底快多少"时（按 G6 口径） |

---

## 7. 结果回填

| 编号 | 状态 | 实测值 / 出处 |
|---|---|---|
| R0.1 | ⚪ 未单独落地 | config 解析断言已由 R0.2 系列承担（`ResNet18WeightContractTest` 内部 `ModelConfig::Load` 后断言 `model_type=resnet18` / `architecture=cnn`）；不另建重复用例 |
| R0.2 | ✅ | 三集合相等（42 个键）+ 每个名字经 `WeightLoader::GetWeight` 取到；沙箱通过（P4-4） |
| R0.2b | ✅ | 20 个卷积 bias 与输出通道一致、`fc.weight = [1000,512]` 与 `num_classes` 一致；沙箱通过（P4-4） |
| R0.2c | ✅ | `source` 五个字段与产物自洽、fc 布局声明与形状一致；沙箱通过（P4-4） |
| R0.2d | ✅ | 5 道护栏全部"自检生效"（ctest 已注册，缺 ONNX/onnx 包时返回 77 跳过）；沙箱通过（P4-4） |
| R0.2e | ✅ | 两次转换的两个产物 SHA256 与仓库产物三者一致（P4-4） |
| R0.3 | ✅ | 前处理 vs P4-1 基线 `max_abs = 0`（逐位相同）；沙箱通过（P4-3）。**这条用例抓出了 #23.1 的通道下标 bug** |
| R0.4 | ✅ | 逐通道 mean/std + 0/255 边界 + 非法参数返回空；沙箱通过（P4-3） |
| R0.5 | ✅ | `cnn → input/output`；沙箱通过（P4-2） |
| R0.6 | ✅ | 三个 architecture 都保持 `input_ids`/`logits`；沙箱通过（P4-2） |
| R0.7 | ✅ | 未登记子图名被拒；沙箱通过（P4-2） |
| R0.8 | ✅ | ramp 公式 C++ vs Python 逐位相同（P4-1 补验） |
| R0.9 | ✅ | 归一化公式 `max_abs = 0`（逐位相同）（P4-1 补验） |
| R0.10 | ✅ | 两套基线的 meta argmax 与 logits 自洽（P4-1 补验） |
| R0.11 | ✅ | 空引擎 → `ok()==false`、`Infer` 空、`Benchmark` 零值；沙箱通过（P4-3 补验） |
| R1.1 | ✅ | 真机：原生网络 52 层；I/O = `input`/`output`（与 ONNX 路径同一份契约常量）；输出 `[-1, 1000]`；逐层 `getDimensions()` 冒烟通过（P4-5） |
| R1.2 | ✅ | 真机：临时目录把 `conv1.weight` 指向不存在的 source key（safetensors 用软链）→ 建网失败（P4-5） |
| R1.2b | ✅ | 真机：`kPrefill` / `kDecode` 被拒——CV 没有 prefill/decode 之分，静默忽略会建出语义不明的引擎（P4-5，计划外补充） |
| R1.3 | ✅ | 真机：ONNX 引擎构建成功（51 MB FP32），I/O = `input`/`output` 均 FP32；`getNbIOTensors()==2`、输出第 1 维 1000（P4-2） |
| R1.4 | ✅ | 真机：`decoder_only` 的 config 配 ResNet18 图被 I/O 契约拦下、且不产出引擎（P4-6） |
| R1.5 | ✅ | 真机：batch 1/8/16 各真跑一次均通过，batch=17 被拒（P4-2） |
| R1.5b | ✅ | 真机：**原生引擎**上的同一组 profile 验收（batch 1/8/16 真跑、17 被拒）。为什么不复用 R1.5：profile 挂在**网络输入维度**上，而原生网络的维度是手写的、与 ONNX 图无关（P4-5 补验） |
| R2.1 | ✅ | 真机：ONNX vs 基线（ramp）`max_abs = 9.53674e-06`、`max_rel = 1.05e-06`、argmax 0/8 不一致（P4-2） |
| R2.2 | ✅ | 真机：ONNX vs 基线（pixels 归一化输入）`max_abs = 1.33514e-05`、`max_rel = 5.36e-07`、argmax 0/8 不一致（P4-2） |
| R2.3 | ✅ | 真机：原生 vs ONNX（同一份权重）`max_abs = 1.07288e-06`（rel 1.19e-7）、argmax 0/8 不一致；阈值 `1e-5`（P4-5） |
| R2.4 | ✅ | 真机：原生 vs torchvision 基线 `max_abs = 1.04904e-05`（rel 1.16e-6）、argmax 0/8 不一致；阈值 `1e-4`（P4-5） |
| R2.5a | ✅ | 真机：ONNX-FP16 vs torchvision-FP32（ramp）`max_abs = 0.03092`（rel 3.42e-3）、argmax 0/8 不一致；阈值 0.1（P4-6） |
| R2.5b | ✅ | 真机：原生-FP16 vs ONNX-FP16 `max_abs = 0.00757`（rel 8.4e-4）、argmax 0/8 不一致；三角证据 `原生-FP16 vs 原生-FP32 = 0.03295`、`ONNX-FP16 vs 原生-FP32 = 0.03092`；阈值 0.05（P4-6，第一版 1e-4 是错的，见 #26） |
| R2.5c | ✅ | 真机：CVRunner + ONNX-FP16 vs FP32 基线（pixels）`max_abs = 0.06207`（rel 2.49e-3）、argmax 0/8 不一致（P4-6） |
| R2.5d | ✅ | 真机：原生-FP16 vs torchvision-FP32（ramp）`max_abs = 0.0329475`（rel 3.64e-3）、argmax 0/8 不一致（P4-6 补验；与三角推断一致） |
| R2.5e | ✅ | 真机：两条 FP16 引擎的 I/O 名字与声明精度逐字段相等，均为 `input=FP32, output=FP32`（P4-6 补验） |
| R2.6 | ✅ | 真机：QDQ 引擎 43 层 / **38 层含 Int8** / **4 层 `i8i8` tactic**（对照 FP32 引擎 0 层 Int8）；真实图 256 张整体一致 38.3%、**余量子集 12/12 = 100%**、`max_abs = 21.6`；ramp 只记录（0/8 不一致）。产物形态 `prequant_dq`（13.3 MB）。**开放项**：per-channel 整网退化（`TROUBLESHOOTING.md` #29/#30/#31，P4-INT8-a）。**"整体 38.3%"的成因已用交叉统计固化进用例输出**：按 FP32 余量分层 → `<1: 23.6%`、`1~2: 39.7%`、`2~5: 73.7%`、`5~10: 100%`、`>10: 100%`（58% 的样本余量<1，即类别本身不可判） |
| R3.1 | ✅ | 真机：`CVRunner::Infer` batch=1 与 8 均 `max_abs = 1.33514e-05`（rel 8.9e-7 / 5.4e-7），与 P4-1 基线一致（P4-3） |
| R3.2 | ✅ | 真机：batch=17、batch=0、元素数不匹配均返回空；同一 runner 在合法输入上返回 `[1,1000]`（有对照）（P4-3） |
| R3.3 | ✅ | 真机：batch=8 时 `mean=8.357ms p50=8.231ms p99=8.592ms throughput=957.2 img/s`；只验统计自洽；非法参数返回零值（P4-3，两次运行数值差异属正常抖动，不作为判据） |
| R3.4 | ✅ | 真机：mean/std 长度 1 与 4（输入通道 3）均 `ok()==false` 且 `Infer` 返回空（P4-3 补验） |
| R3.5 | ✅ | 真机：CVRunner 驱动**原生引擎** batch=8 `max_abs = 1.33514e-05`（与 ONNX 引擎那条同值）。Runner 按**引擎声明的** I/O 契约工作，两套引擎的维度来源不同（手写 vs 解析），"同名 ≠ 同契约"（P4-5 补验） |

**回填要求**：只写"状态 + 实测值 + 出处（命令/日志）"；排查过程写 `TROUBLESHOOTING.md`。
