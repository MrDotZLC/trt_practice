# Phase 1.5 测试计划：全流程测试基建与收尾

> **补记性质（2026-09-25）**：本文件是**事后补写**的。Phase 1.5 当初**有意不写独立测试计划**，
> 理由是"设计已在 `docs/phase1_test_plan.md`（E1–E4），再写一份只会复述一遍、形成两处来源"
> （见 `PROGRESS.md` §2.13「唯一来源」）。现在补，是因为那个理由只对"复制设计"成立，而这些信息
> 此前没有归属：**用例清单（用例名 → 判据 → 出处 → 环境 → 状态）、执行方式、覆盖缺口**——
> 它们散落在两份文档里，新会话要读两份才能回答"Phase 1.5 到底测了什么、凭什么算过"。
> Phase 2 / Phase 3 都有各自的测试计划，缺这一份会让跨阶段比对时少一环。
>
> **为避免重开那个坑，本文件不复制任何设计**：E1–E4 的目的、做法、网络结构、表格一字不搬，
> 只给指针。设计要改就改 `docs/phase1_test_plan.md` §4（唯一来源）；本文件只在
> "用例→判据→状态"这一层与它对齐。

---

## 0. 文档分工（唯一来源）

| 内容 | 唯一来源 | 本文件的角色 |
|---|---|---|
| E1–E4 的**设计**（目的、做法、网络结构、每条用例的验证点） | `docs/phase1_test_plan.md` §4 | 只引用编号（E1.1～E4.5），不复述 |
| Phase 1.5 的**任务、依赖、验收、风险** | `docs/phase1_5_development_plan.md` §2/§3/§5/§6 | 引用结论 |
| **用例清单、判据出处、执行口径、覆盖缺口、结果快照** | **本文件** | 唯一来源 |
| 缺陷的排查过程 | `docs/TROUBLESHOOTING.md` #5 / #6 / #7 / #8 / #9 | 引用编号 |
| 当前整体进度与跨阶段状态 | `docs/PROGRESS.md` | 引用 |

---

## 1. 分层与执行环境

Phase 1 / 2 / 3 都采用"按**需要的运行时能力**分层"的口径，Phase 1.5 同样适用；
分层的意义是"哪一层能在沙箱里跑、哪一层必须上真机"。

| 层 | 测什么 | 环境 | 为什么在这一层 |
|---|---|---|---|
| **S 支撑层（host 契约）** | 配置解析 / 权重名与 dtype 矩阵 / 指针别名 / 参考实现自证 | **沙箱** | 不需要 CUDA，成本最低、进 CI 才有价值 |
| **E1 单算子闭环** | 算子经**真实路径**（safetensors → `weight_map` → 常量 → Plugin → engine → 推理） | 真机（`createInferBuilder` 需要 CUDA） | 与算子单测的分工见 `phase1_test_plan.md` §2：单测验"算法对不对"，本层验"挂得上、取得对" |
| **E2 多算子链路** | 算子之间的 shape / dtype / 布局契约 + 整体数值 | 真机 | 任一处契约不一致只在链路里暴露 |
| **E3 动态 shape** | Prefill/Decode 两组 optimization profile 真正生效、超范围必须拒绝 | 真机 | profile 是 TRT 构建期能力 |
| **E4 错误路径** | 失败必须是**可控失败**（返回 false / 抛异常），不产半成品 engine、不静默成功 | 3 条沙箱 + 2 条真机 | 能在沙箱跑的就不推给真机 |

> 层名刻意用 **S / E1–E4**，不用 `L1`/`L2`：`phase1_test_plan.md` 里的 `L1`/`L2` 指的是
> "算子单测 / 集成测试"那一套分层，两套编号混用会让人对着旧文档读错层。

> **2026-09-25 起真机口径变更**：跑真机用例必须带 `MINI_TRT_REQUIRE_GPU=1`
> （跳过即失败）。理由见 §4 与 `TROUBLESHOOTING.md` #19：无 GPU 时用例会 `GTEST_SKIP`，
> 而"静默跳过"曾经掩盖过真缺陷。

---

## 2. 用例清单

**这张表是索引，不是定义**：每条用例的"验证点"以代码与 `phase1_test_plan.md` §4 为准。
状态列的"真机 ✅"来自 2026-09-25 的全量真机运行（`MINI_TRT_REQUIRE_GPU=1`，146 条 / 0 跳过）。

### E1 单算子最小模型闭环（4 条，`tests/test_e2e_single_op.cpp`，真机）

| 用例 | 编号 | 验证点 | 判据 | 状态 |
|---|---|---|---|---|
| `E2eSingleOpTest.RmsNormThroughRealWeightLoader` | E1.1 | 权重经 `weight_map` 命中；`eps`/`hidden_size` 经 engine 往返仍正确 | FP32 绝对误差 < `1e-4`（`phase1_5_development_plan.md` §5） | 真机 ✅ |
| `E2eSingleOpTest.RoPEThroughRealBuilder` | E1.2 | `position_ids` 作输入；`rotary_dim < head_size` 时尾部保持原值 | 同上 | 真机 ✅ |
| `E2eSingleOpTest.PagedAttentionThroughRealBuilder` | E1.3 | `block_tables`/`context_lens` 作输入；跨多个物理块寻址 | 同上 | 真机 ✅ |
| `E2eSingleOpTest.LogitsFeedSamplerWithoutHostRoundTrip` | E1.4 | engine 输出 logits 直接进采样器（不经 host）；固定 seed 可复现 | 与 CPU argmax 一致 | 真机 ✅ |

### E2 多算子串联（2 条，`tests/test_e2e_mini_decoder.cpp`，真机）

| 用例 | 验证点 | 判据 | 状态 |
|---|---|---|---|
| `E2eMiniDecoderTest.FullChainMatchesIndependentCpuReference` | 完整链路（设计见 `phase1_test_plan.md` §4 E2）的接口契约与整体数值；权重全部来自 safetensors | 对独立 CPU 参考；FP16 走 D4 档（`rel < 1e-3`，**阈值不跨精度复用**，见 §3） | 真机 ✅ |
| `E2eMiniDecoderTest.RejectsMissingWeightFromMap` | `weight_map` 指向不存在的 key 时必须建网失败 | 返回 false，不产出 engine | 真机 ✅ |

### E3 动态 shape（3 条，`tests/test_e2e_dynamic_shape.cpp`，真机）

| 用例 | 编号 | 输入 / 期望 | 状态 |
|---|---|---|---|
| `E2eDynamicShapeTest.PrefillAcceptsVaryingSeqLen` | E3.1 | `[B,S]`，S 在 prefill 范围内变化 → 同一 engine 接受且输出正确 | 真机 ✅ |
| `E2eDynamicShapeTest.DecodeAcceptsVaryingBatch` | E3.2 | `[B,1]`，B 在 decode 范围内变化 | 真机 ✅ |
| `E2eDynamicShapeTest.OutOfRangeShapeIsRejected` | E3.3 | 超范围时 `SetInputShape` 失败并给出可诊断错误 | 真机 ✅ |

### E4 错误路径（5 条，`tests/test_e2e_error_paths.cpp`）

| 用例 | 编号 | 场景 | 判据 | 环境 / 状态 |
|---|---|---|---|---|
| `E2eErrorPathTest.MissingConfigFailsWithoutProducingEngine` | E4.1 | 无 `config.json` | 返回 false 且**不产出 engine 文件** | 沙箱 ✅ |
| `E2eErrorPathTest.UnregisteredModelTypeFails` | E4.2 | `model_type` 未注册 | 返回 false + 明确日志 | 沙箱 ✅ |
| `E2eErrorPathTest.MissingWeightsFileFails` | E4.3 | 缺 `model.safetensors` | `Load` 失败 → 返回 false | 沙箱 ✅ |
| `E2eErrorPathTest.UnknownWeightMapKeyFails` | E4.4 | `weight_map` 指向不存在的 key | 返回 false，**不得建出错误 engine** | 真机 ✅ |
| `E2eErrorPathTest.BuilderFailurePropagatesAndLeavesNoEngine` | E4.5 | 插件配置非法等构建期失败 | 失败向上传播且不落 engine | 真机 ✅ |

### 一并收口的覆盖缺口（3 条 FP16，`tests/test_fp16_paths.cpp`，真机）

| 用例 | 覆盖的缺口（出处 `phase1_5_development_plan.md` §0.3） | 状态 |
|---|---|---|
| `Fp16PathTest.RoPEHandlesFp16WithGqaAndMultipleBatches` | RoPE 缺 FP16 / GQA / `batch > 1` | 真机 ✅ |
| `Fp16PathTest.PagedAttentionMatchesFp16Reference` | PagedAttention 缺 FP16 | 真机 ✅ |
| `Fp16PathTest.GreedySamplerMatchesArgmaxOnFp16Logits` | Sampler 缺 FP16 | 真机 ✅ |

### S 支撑层：参考实现自证与权重转换回归（20 条，沙箱）

| 文件 | 条数 | 作用 | 状态 |
|---|---|---|---|
| `tests/test_reference_helpers.cpp` | 11 | **参考实现自身的 meta-test**：参考实现是裁决对错的标尺，标尺错了会给出错误裁决（#9 吃过一次），所以它必须自证。其中 `LayerNorm*` / `GeluNew*` 三条是 **Phase 2** 期间补的（同一文件，跨阶段共用） | 沙箱 ✅ |
| `tests/test_safetensors_loader.cpp` | 9 | P1.5-0 的 8 条回归（dtype 组合矩阵 / 指针不别名 / 零拷贝 / 缓存稳定）+ 1 条早于本阶段的缺文件用例 | 沙箱 ✅ |

### 夹具（不是用例）

| 文件 | 作用 |
|---|---|
| `tests/e2e_safetensors_writer.{hpp,cpp}` | 测试用 safetensors 写入 helper（F32 / F16 / BF16），让端到端夹具自包含、不依赖 Python |
| `tests/e2e_fixture.{hpp,cpp}` | 临时模型目录（`mkdtemp` + 析构清理），组装 `config.json` + `model.safetensors` |

---

## 3. 判据与出处（"凭什么算过"）

| 判据 | 出处 | 备注 |
|---|---|---|
| E1 的 FP32 绝对误差 < `1e-4` | `phase1_5_development_plan.md` §5 验收标准 | |
| FP16 数值按 **D4**（相对界 `< 1e-3`） | `docs/phase1_development_plan.md` D4 | **阈值不跨精度复用**：把 FP16 的尺子套到 FP32 上等于放宽约 1000 倍（`PROGRESS.md` §7） |
| 错误路径：返回 false / 抛异常 **且不产出 engine 文件** | `phase1_test_plan.md` §4 E4 | "不产出半成品"与"返回 false"同等重要 |
| 退出码/失败语义：不在 enqueue 期才发现问题 | `TROUBLESHOOTING.md` #19 | 反例：诊断输出未绑定曾让失败推到 enqueue |

---

## 4. 执行方式

```bash
# 沙箱：S 层（host 用例真实执行；GPU 用例跳过，且跳过信息里带 cudaGetDeviceCount 的探测结果）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# 真机：E1–E4（**必须带 MINI_TRT_REQUIRE_GPU=1**，否则无设备时会静默跳过 = 白跑）
MINI_TRT_REQUIRE_GPU=1 ./build/mini_trt_llm/tests/mini_trt_llm_tests \
  --gtest_filter='E2eSingleOpTest.*:E2eMiniDecoderTest.*:E2eDynamicShapeTest.*:E2eErrorPathTest.*:Fp16PathTest.*:ReferenceHelpersTest.*:SafetensorsLoaderTest.*'

# 或者直接跑全量（含 Phase 2/3 用例）
MINI_TRT_REQUIRE_GPU=1 ctest --test-dir build
```

**注意**：

1. 引擎文件按 `model_type + shape` 缓存在临时目录 / `/tmp`；**改动建图或构建开关后必须清理**，
   否则缓存会让"新代码 + 旧引擎"给出假结果（`TROUBLESHOOTING.md` #19）。
2. 每个 E2E 用例都要跑一次 `buildSerializedNetwork`（数秒～数十秒），全量真机约 3～9 分钟
   （取决于缓存是否命中）。
3. `MINI_TRT_REQUIRE_GPU` 是 2026-09-25 引入的闸门：默认（未设置）无设备时跳过，
   设置后**跳过即失败**。沙箱不要设它，否则 CI 必然红（`PROGRESS.md` §5.7）。

---

## 5. 覆盖缺口（当时未登记，现补；只做索引，不排期）

| 缺口 | 是什么 | 现状 / 触发条件 |
|---|---|---|
| **E2 只到"单步 decode"** | 完整自回归循环要等 `LLMRunner`，属 Phase 2 范围（`phase1_5_development_plan.md` §6.5） | Phase 2 已交付并由 `Gpt2GenerateTest.*` 覆盖 |
| **P1.5-4 缩减完成** | 交付了多权重 BF16 路径的数值验证；完整 `RMSNorm → QKV → RoPE → PagedAttention → LM Head` 链路与 `ref_mini_block.py` 有意留后 | 理由与定位见 `phase1_5_development_plan.md` §0.1（该阶段的目的是"组件能组合"，不是"预演某个具体模型"） |
| **Top-K / Top-P 的 FP16 分支未覆盖** | Greedy 已覆盖 FP16；Top-K/Top-P 同属采样器同一处 dtype 分派 | 风险低；留待真实采样路径（Phase 2 已用 Greedy 覆盖端到端） |
| **采样器分布级数据未固化** | `scripts/ref_sampler.py` 能打印 HF 风格截断语义，但输出没有固化成数据文件供测试载入 | 见 `future_iterations.md` §9.3（增强项，非组件能力） |
| **E3 只验"接受/拒绝"** | 未覆盖"同 engine 内多次切换 profile 后的数值一致性" | 未触发；需要时再补 |

---

## 6. 通过标准与结果快照

**通过标准**：`phase1_5_development_plan.md` §5 的 8 条（本文件不复制，只记状态）。

| 验收项 | 状态 |
|---|---|
| P1.5-0 三个缺陷各有回归用例且不需要 GPU | ✅ |
| 沙箱 `ctest` 全绿、GPU 用例跳过不产生噪声 | ✅ 146 条 / 0 失败 / 64 跳过（2026-09-25） |
| E4 五条在沙箱内实际执行并通过 | ✅（3 条沙箱；另 2 条需 CUDA，真机通过） |
| E1 四条在真机通过，FP32 绝对误差 < `1e-4` | ✅ |
| E2 的 logits 与独立参考在阈值内，FP16 按 D4 | ✅ |
| E3 三条在真机通过且覆盖 opt 形状 | ✅ |
| 未配 profile 而输入含动态维时报可诊断错误 | ✅（E3.3） |
| P1.5-7 四项文档动作完成 | ✅ |

**实测快照（2026-09-25）**：真机全量 `MINI_TRT_REQUIRE_GPU=1 ctest` = **146 条 / 0 跳过 / 1 红**，
唯一红是 Phase 2 的 FP16 NaN 复现器（`RealGpt2Fp16GreedyMatchesReferenceTokens`，按设计红）；
Phase 1.5 自己的 37 条（E1 4 + E2 2 + E3 3 + E4 5 + FP16 3 + 参考自证 11 + 权重回归 9）**全绿**。

**过程中修掉的缺陷**（编号可追）：`TROUBLESHOOTING.md` #5 / #6 / #7（产品代码）
与 #8 / #9（真机复验暴露的 Plugin 反序列化与参考实现缺陷）。
