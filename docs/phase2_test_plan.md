# Phase 2 测试计划：GPT-2 原生构建（方案 A）

> **定位**：Phase 2 的**任务与决策**见 `docs/phase2_development_plan.md`；本文档定义
> **测试分层、用例、判据出处与缺口**。
>
> **状态说明（补记性质）**：Phase 2 的用例在本文档成文之前就已实现并真机通过，因此本文档
> 的价值是四点：**判据的出处**（交接需要）、**分层与分工**（哪些能在沙箱跑）、
> **缺口清单**（当时没有登记，现补）、**可复现的执行方式**（含运行前提）。
> 判定一律区分"**已验证**"与"**未验证**"——未跑过的不写"通过"（AGENTS.md §7）。

---

## 1. 被测链路

```
models/gpt2/{config.json, model.safetensors}
        │
        ├─► ModelConfig::Load + GPT2Config::FromModelConfig     # L0：配置契约
        ├─► WeightLoader（weight_map 映射 148 个权重）           # L0：权重契约
        │
        └─► EngineBuilder::BuildFromConfig(model_dir, engine_path, stage)
                    │  ├─ SetupBuilder（精度 / workspace）
                    │  ├─ IModelBuilder::Build(network, weights, config, BuildOptions)
                    │  └─ optimization profile（按 stage 只挂一组）
                    ▼
              .engine ──► Engine ──► Prefill / Decode 推理
                                          │
                                          ├─► 写 / 追加 分页 KV Cache
                                          └─► LLMRunner 自回归循环（循环内零 H2D/D2H）
                                                       └─► Sampler（Greedy / Top-K / Top-P）
```

---

## 2. 测试分层

| 层 | 覆盖什么 | 在哪跑 | 用例 |
|---|---|---|---|
| **L0 host 契约** | 配置解析与校验、权重名集合（148）、真实转换产物的权重与关键形状、块分配器 | **沙箱**（不需要 GPU） | `test_gpt2_config.cpp`、`test_paged_kv_cache.cpp`（BlockAllocator 部分） |
| **L1 建网** | prefill / decode 网络的输入输出契约、逐层 shape 冒烟、缺权重必须在建网阶段失败 | 真机（`createInferBuilder` 需要 CUDA，实测确认） | `test_gpt2_network_build.cpp` |

> ⚠️ **2026-09-25 更正**：`test_gpt2_network_build.cpp` 里曾写"建网络不需要 CUDA，可以在沙箱 / CI
> 里跑"——**该注释是错的**，与上表（以及沙箱实测：`createInferBuilder` 报
> `CUDA initialization failure with error: 35`）矛盾。这条错误假设让两条本该红的输出数断言在
> 沙箱里被静默跳过，缺陷一路滑到提交（见 `docs/TROUBLESHOOTING.md` #19）。注释已改，
> 并且"无设备才跳过、有设备却建不出 builder 判失败"已成为硬约束。
| **L2 数值** | prefill 对 `ref_output.bin`；decode 与 prefill 同位置一致；分页 cache 写入 / 追加；FP16 分支 | 真机 | `test_gpt2_prefill_accuracy.cpp`、`test_gpt2_decode_consistency.cpp`、`test_paged_kv_cache.cpp`（GPU 部分）、`test_fp16_paths.cpp`（Phase 1 遗留 + 复用） |
| **L3 端到端** | 带 KV Cache 的生成 vs 每步全序列重算；真实 GPT-2 的 8 个贪心 token 与 HF 基线；`temperature` 拒绝 | 真机 | `test_gpt2_generate.cpp` |

**分层由实测决定**："建网"被单独摘出来（L1）是因为 `createInferBuilder` **即使只建网也需要
CUDA 初始化**——把它与"建引擎 + 数值"混在一起时，真机失败要同时排查两者；拆开后定位面小得多。

---

## 3. 用例清单

| 用例 | 层 | 判据 | 出处 | 状态 |
|---|---|---|---|---|
| `Gpt2ConfigTest.ParsesNativeConfig` / `RejectsMissingOrInconsistentHyperParams` | L0 | 必需字段缺失/非法必须失败（不用默认值兜底） | §4.6 / D4（G2） | ✅ 沙箱 |
| `Gpt2ConfigTest.WeightNamesMatchRealModelContract` | L0 | 12 层 → 148 个权重名，不含 `attn.bias`、共享权重时不索取 `lm_head.weight` | §0.1 实测 | ✅ 沙箱 |
| `Gpt2WeightContractTest.SyntheticFixtureResolvesEveryWeight` / `RealConvertedArtifactIsComplete` | L0 | 每个权重都能经 `weight_map` 解析到；真实产物的 148 个权重存在且关键形状匹配（Conv1D `[in,out]`） | §0.1 / 4.2 | ✅ 沙箱（真实产物存在时执行） |
| `Gpt2NetworkBuildTest.PrefillNetworkBuildsWithExpectedIo` 等 4 条 | L1 | I/O 契约（名字/维度/动态轴）+ 逐层 shape 冒烟 + 缺权重建网即失败 | §4.6 | ✅ 真机 |
| `Gpt2PrefillAccuracyTest.RealGpt2LogitsMatchReference` | L2 | `cosine > 0.999999`、`max_abs/max\|ref\| < 1e-5`、逐位置 argmax 一致 | D6 + 实测收紧 | ✅ 真机（`cosine = 1.0`、`9.19e-07`） |
| `Gpt2DecodeConsistencyTest.DecodeStepMatchesPrefillAtSamePosition` | L2 | decode 一步 == prefill 同位置；分层 K/V 差异用于定位 | §4.10（P2-6） | ✅ 真机 |
| `Gpt2DecodeConsistencyTest.TwoStepDecodeMatchesPrefillAfterAppend` | L2 | 追加一次后再 decode 仍一致（三个独立量：步1 logits / 追加的 K/V / 步2 logits） | §4.10 + TROUBLESHOOTING #16 | ✅ 真机（该用例即 #16 的判别实验） |
| `Gpt2DecodeConsistencyTest.DecodeWithEmptyCacheMatchesSingleTokenPrefill` | L2 | 单 key 时两条路几乎逐位一致（硬判据 `1e-5`） | §4.10 | ✅ 真机 |
| `PagedKVCacheTest.*`（现 4 条） | L2 | 写入走块表 / 跨块追加（**两层都验**）/ 越界拒绝 / **层数不匹配必须被拒且不留副作用** | §4.8（P2-4） | ✅ 真机（2026-09-25 复验 4/4）。⚠️ 更正：此前这条写"✅ 真机"时，`AppendCrossesBlockBoundary...` **自 d6af2eb 起就不可能通过**（传 1 对 vs 契约要求每层一对），沙箱跳过掩盖了它——详见 `docs/TROUBLESHOOTING.md` #20 |
| `Gpt2GenerateTest.RunnerMatchesFullRecomputeWithoutCache` | L3 | runner（带 cache）与每步全序列重算**逐 token 一致** | §4.11 | ✅ 真机 |
| `Gpt2GenerateTest.RealGpt2GreedyMatchesReferenceTokens` | L3 | 与 HF 基线逐 token 一致：`[274, 389, 257, 1049, 835, 284, 651, 257]` | §0.3 / §4.11 | ✅ 真机 |
| `Gpt2GenerateTest.RejectsUnsupportedTemperature` | L3 | `temperature != 1.0` 必须返回失败（不静默忽略） | D5 | ✅ 真机 |

---

## 4. 判据与出处（阈值纪律）

| 判据 | 出处 | 实测与余量 |
|---|---|---|
| `cosine > 0.999999`、`max_abs/max\|ref\| < 1e-5`（FP32 对拍） | D6（FP32 `cosine ≥ 0.9999`、相对界 `< 1e-3`）+ **首次实测后收紧**（`cosine = 1.0`、`9.19e-07`） | 相对界实测 `9.19e-07` → 余量约 10 倍；收紧无需额外证据，**放宽必须按 §7 先量"无关差异"** |
| 解码一致性 `1e-5`（绝对） | 实测模型敏感性：一次性 softmax vs online softmax 在 float32 下差 `6e-8`、模型对扰动放大倍数 ≈ 1（numpy 复刻） | 该阈值"比无关差异宽 100 倍、比当时观测的 `1.25e-3` 严 100 倍"，正是它抓出了 #15 / #16 |
| FP16：`cosine ≥ 0.999`、相对界 `< 5e-3` | D6 的 FP16 档 | 真实 GPT-2 的 FP16 **端到端不稳定（NaN）→ 不适用**：GPT-2 的推荐精度为 **FP32**（见 G2-1 与 `TROUBLESHOOTING` §18.1）。算子/网络层的 FP16 覆盖（Phase 1.5 的 `Fp16PathTest`）仍然有效 |
| 语义判据：逐位置 argmax / 逐 token 一致 | "token 是产品判据，数值是定位判据" | 真实模型 8/8 命中 |

**关于"用单点测量支撑结论"**：Phase 3 曾因此误读性能数据（见 `docs/phase3_test_plan.md` §3.1），
该教训同样适用于本阶段——**任何从测量得出的结论都要说明离散度**。

---

## 5. 缺口（当时未登记，现补）

| ID | 缺口 | 影响 | 触发条件 / 做法 |
|---|---|---|---|
| **G2-1** | ~~真实 GPT-2 的 FP16 端到端未测~~ **用例已就位并执行；结果：FP16 端到端数值不稳定（NaN），按政策不修，登记为已知限制** | 这条缺口**首跑即立功**：不仅抓出缓冲按假定精度分配导致的非法访存（已修复），还暴露出"GPT-2 在弱类型 FP16 下端到端不可用"这一性质 | 用例保留为**复现器**（`RealGpt2Fp16GreedyMatchesReferenceTokens` 预期失败；`Fp16PrefillOutputsDiagnostic` 为逐输出诊断仪器，**自 2026-09-25 起需显式打开 `export_diagnostics` 并用独立引擎路径**——诊断输出会改 I/O 契约，见 `docs/TROUBLESHOOTING.md` #19）。结论、证据与理由见 §18.1；后续解决路径见 `docs/future_iterations.md` §1.4 | 方案：新增 `Gpt2GenerateTest.RealGpt2Fp16GreedyMatchesReferenceTokens` —— FP16 下 `EngineBuilder` + **`LLMRunner::Config::is_half = true`**（cache 精度必须与引擎激活精度一致，否则 PagedAttention 按错误宽度读 cache），8 个贪心 token 与 HF 基线逐 token 对照。判据用 D6 的 FP16 档（`cosine ≥ 0.999`、相对界 `< 5e-3`）+ **语义判据：8 个 token 必须全中**；实测值打印出来供后续按"实测收敛"处理 |
| **G2-2** | **P2-0 的专用"多权重要素"用例未写**（计划里叫 `test_gpt2_weight_plumbing.cpp`） | 计划中的该项没有独立用例 | **以更强证据覆盖**，故不补：148 个权重经真实 `weight_map` 建出 engine，且数值与 HF 对拍通过——这比"逐个 `addConstant` 后比对"强。若将来出现"权重静默失效"的疑似缺陷，再补该用例 |
| **G2-3** | `LLMRunner` **只支持 `batch = 1`**（有意限定，见计划 §4.11） | 批处理场景不可用 | 需要批处理时再扩（届时同时引入多序列 block 分配、各自 `context_lens` 与采样参数） |
| **G2-4** | EOS 无法在循环内早停（**已知 workaround**，非缺陷） | EOS 前仍跑满 `max_new_tokens`，多余计算被丢弃；语义正确 | 见 `docs/PROGRESS.md` §5.0（含后续可选方案） |

---

## 6. 执行方式

```bash
# 前提：先生成模型目录（否则真机用例跳过）
python3 mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py \
    --model_name_or_path <HF gpt2 目录> --output_dir models/gpt2

# 沙箱：L0（host 用例真实执行，GPU 用例自动跳过）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# 真机：L1 / L2 / L3（引擎文件在 /tmp 缓存复用；删掉会强制重建）
./build/mini_trt_llm/tests/mini_trt_llm_tests \
  --gtest_filter='Gpt2NetworkBuildTest.*:Gpt2PrefillAccuracyTest.*:Gpt2DecodeConsistencyTest.*:Gpt2GenerateTest.*:PagedKVCacheTest.*'
```

---

## 7. 通过标准

- [x] L0 在沙箱内全绿（含真实转换产物的 148 权重与形状校验）。
- [x] L1 建网契约在真机通过。
- [x] L2：prefill 对 `ref_output.bin` 达标；解码一致性（单步 / 两步 / 空 cache）达标。
- [x] L3：真实 GPT-2 贪心 8 token 与 HF 基线逐 token 一致；`temperature` 拒绝生效。
- [ ] G2-1（FP16 端到端，**进行中**）/ G2-2（不补，已说明理由）/ G2-3（有意限定）/ G2-4（已知 workaround）。

---

*文档版本：v1.0（Phase 2 完成后补记；§5 的缺口显式标注为未验证）*
