# Test Plan

<!--
执行口径（沿用 REQ-016 的约定，出处 `docs/TROUBLESHOOTING.md` + TS-055）：
清单顺序 = 测试文件里 `TEST(...)` 的书写顺序；名字 / 条数一一对应；真机按清单逐条打勾。
**每条用例只归一类**（Unit / Integration / Regression / Failure），避免同一名字出现两次。
本环境（Windows 沙箱）无 GPU / 无编译器：Unit 与 Failure 里的 host 项与 Python 项**可沙箱跑**，
Integration 与需要 GPU 的项**只能真机跑**；结果一律先记"未验证"。
-->

## Requirement Traceability

行长口径：行数 = Included 条数 + Acceptance Criteria 条数 = 4 + 5 = **9**。

| 需求条目 | 判据 | 用例 | 结果 |
|---|---|---|---|
| Included 1 路线选型与实现（路线 C） | Python 产出 int8 权重 + 清单；C++ 只读常量挂 DQ，且清单外不碰 | U1、I1 | 未验证 |
| Included 2 自证低精度 | 引擎层信息里出现 `Format/Datatype: Int8` | I2 | 未验证 |
| Included 3 与 FP32 的数值对比 | 同 prompt 同采样策略下逐 token 一致率 + prefill logits 相对/绝对偏差（阈值 P4 先量再定） | 见"待真机窗口的数值对照" | 未验证 |
| Included 4 显存 / 体积对比 | 同配置下 INT8 引擎体积 < FP32 | I3 | 未验证 |
| AC1 端到端可跑 | 两张图（prefill / decode）都能构建成功并完成生成 | I1 + 既有 `RealGpt2*` 用例 | 未验证 |
| AC2 自证低精度 | 同 Included 2（层信息 + 数量） | I2 | 未验证 |
| AC3 数值判据有出处 | 阈值旁写出处（P4 实测分布 / 分位），不得复用视觉模型阈值 | 见"待真机窗口的数值对照" | 未验证 |
| AC4 资源下降 | 同 Included 4，且报**实测**值不报估计 | I3 | 未验证 |
| AC5 不回归 | 既有 FP32 / FP16 路径行为不变；沙箱全绿；真机不出现新红 | R1 | 未验证 |

## Unit Test

| # | 用例 | 判据 | 出处 |
|---|---|---|---|
| U1 | `QuantSpecTest.LoadsValidManifestAndResolvesBothNames` | 合法清单载入成功；`Find` 对 TRT 名与文件 key **双向命中**、未知名返回 nullptr；相对路径按清单目录解析 | `design.md` 的 Data Structure 契约表 |
| U2 | `quantize_gpt2_selftest`（ctest，`quantize_gpt2.py --self-test`） | 6 道护栏（缺 config / 已存在产物 / 缺张量 / 非 2-D / 产物被改坏 / 清单 sha256 造假）+ 命名空间与 scale 身份自检全过 | `TROUBLESHOOTING.md` + TS-056 |

## Integration Test

需要 GPU / TensorRT；沙箱内 `MINI_TRT_SKIP_IF_NO_CUDA()` 显式跳过。

| # | 用例 | 判据 |
|---|---|---|
| I1 | `Gpt2Int8WeightsTest.BuildsBothStagesAndConsumesWholeManifest` | prefill 与 decode 两张图用**同一份清单**都能建出来；"清单全消费"校验通过 |
| I2 | `Gpt2Int8WeightsTest.EngineLayerInfoShowsInt8` | 层信息里含 Int8 张量的层数 ≥ 1（DQ 没被构建期常量折叠 —— **D7 的小模型信号**） |
| I3 | `Gpt2Int8WeightsTest.FullGpt2Int8EngineIsSmallerThanFp32` | **D7 的正式判据**：真实 GPT-2 的 INT8 引擎体积 < 同配置 FP32（缺 `models/gpt2/quant_int8.json` 则跳过，且 `MINI_TRT_REQUIRE_ASSETS=1` 时判失败） |

## Regression Test

| # | 用例 | 判据 |
|---|---|---|
| R1 | 既有 GPT-2 用例（不新增）：小模型类（沙箱）+ `RealGpt2*`（真机） | 清单为空时建图路径逐字未改（`quant == nullptr` → 走原调用），既有用例**全绿、无新红**；真机全量按 `MINI_TRT_REQUIRE_GPU=1` 跑 |

## Failure Test

全部是 host 项（清单是纯文本契约，拒绝面不需要 GPU）。

| # | 用例 | 判据 |
|---|---|---|
| F1 | `QuantSpecTest.RejectsUnsupportedFormatVersion` | `format_version != 1` → 拒绝 |
| F2 | `QuantSpecTest.RejectsNonPerTensorGranularity` | `granularity = per_channel` → 拒绝（本轮只支持 per_tensor） |
| F3 | `QuantSpecTest.RejectsNonSymmetricScheme` | `scheme != symmetric_per_tensor` → 拒绝 |
| F4 | `QuantSpecTest.RejectsNonZeroZeroPoint` | `zero_point != 0` → 拒绝 |
| F5 | `QuantSpecTest.RejectsEmptyOrNonArrayEntries` | `entries` 为空或不是数组 → 拒绝 |
| F6 | `QuantSpecTest.RejectsBadScales` | `scales` 长度 ≠ 1、为 0、为负 → 拒绝；合法值通过（防"假绿"） |
| F7 | `QuantSpecTest.RejectsMissingRequiredFields` | 缺 `int8_weights.path` / `weights_source.path` / `tensor` / `source_key` → 拒绝 |
| F8 | `QuantSpecTest.RejectsDuplicateEntries` | 同 `tensor` 或同 `source_key` 出现两次 → 拒绝 |
| F9 | `QuantSpecTest.FailsOnMissingManifestFile` | 清单文件不存在 → 载入失败（不静默成功） |
| F10 | `Gpt2Int8WeightsTest.RejectsManifestEntryOutsideModel` | 清单点名了图里没有的张量 → **构建失败**（"全消费"校验；不得静默退化成纯 FP32） |
| F11 | `Gpt2Int8WeightsTest.RejectsMissingInt8WeightsFile` | 清单在、int8 产物缺 → **构建失败** |

## Expected Result

- U1 / U2 / F1–F9：**沙箱可执行**，必须全过（不需要 GPU）。
- I1 / I2 / I3 与 R1 的真机部分：**需真机**（GPU + TensorRT + `models/gpt2` 资产），本次环境无法执行，
  一律记"未验证"。
- R1 的沙箱部分：既有用例在沙箱里跑（GPU 项跳过），**不得出现新红**。
- 判据与阈值纪律：任何数值阈值必须写出来源（`AGENTS.md` §7）；本轮**没有任何**放宽阈值或跳过用例的动作。

## Actual Result

- U2（`quantize_gpt2.py --self-test`）：**已跑通**（6 道护栏 + `--only` 正向断言 + 命名空间/scale/身份自检）。
- 其余：**未执行**（本环境无编译器 / 无 GPU；C++ 用例连编译都还没做）。

### 待真机窗口的数值对照（AC1 / AC3，无新增用例文件）

口径由作者 2026-10-05 定（P1-6）：**短 + 长各一档**——短 = prompt 4、长 = prompt 960（与 PF-9 同口径）。
步骤：同 prompt、同采样策略下跑 INT8 与 FP32，报 ① 逐 token 一致率（含样本量）② prefill logits 的相对偏差
**与绝对差**（并写明比较对象的形状 / 布局）。**阈值必须先量再定**（P4 记的是 `N/A`，故阈值本轮为空，
真机取到 FP32 自身的构建间波动之后再定稿）。

## Status

| 级别 | 条数 | 状态 |
|---|---|---|
| Unit | 2 | U2 已过；U1 待编译 + 运行 |
| Integration | 3 | **未执行**（需真机） |
| Regression | 1 | 沙箱部分待编译；真机部分未执行 |
| Failure | 11 | 待编译 + 运行 |

**P6 的 Exit Gate 尚未满足**：四类用例都要求**实际通过**，而本环境连编译都不具备（与 `REQ-016` 同源）。
因此本文件先落"用例 + 判据 + 口径"，**不写通过**；真机窗口按清单逐条打勾后再回填 `## Actual Result`
与本节状态。
