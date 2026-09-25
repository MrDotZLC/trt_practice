# Phase 1 全流程测试计划：从模型加载到 Plugin 调用

> **范围**：验证「模型目录 → 配置/权重加载 → builder 注册分发 → Plugin 挂载 → engine 构建与序列化
> → 反序列化 → 推理 → 采样 → 输出比对」这条**完整链路**，而不是单个算子的数值。
>
> **为什么需要这一层**：Phase 1 的算子单测（L1）验证的是「kernel 算得对不对」，L2 只对 RMSNorm
> 验证了「Plugin 挂到 engine 上能不能跑」。但两者都没有覆盖**真实使用路径**——权重是从 Safetensors
> 经 `weight_map` 取出来的、网络是 builder 按 config 搭的、engine 是存盘再读回来的。
> Phase 1 抓到的两个真实缺陷（`supportsFormatCombination` 越界、RoPE 部分旋转尾部未写入）中，
> 前者恰恰只有「真实 engine 构建」才会触发——这正是本计划要常态化的覆盖。

> **本文档只描述"测什么、为什么这么测"（用例设计与链路）**。
> E1–E4 的任务归属、开发顺序、产出与验收以 `docs/phase1_5_development_plan.md` 为准，
> 本文档不再单独维护进度清单，避免两处各记一份。

---

## 1. 被测链路

```
model_dir/
  config.json ──► ModelConfig::Load
  model.safetensors ──► WeightLoader::Load + SetWeightMap + SetDefaultPrecision
                              │
        ModelRegistry::Get(model_type) ──► IModelBuilder::Build(network, weights, config)
                              │                        │
                              │                        └─► addPluginV3(RMSNorm / RoPE / PagedAttention)
                              │                             weight 经 GetWeight() 取出后挂成常量层输入
                              ▼
                  EngineBuilder::BuildFromConfig ──► buildSerializedNetwork ──► .engine 文件
                              ▼
                  Engine(engine_path) 反序列化 ──► SetInputShape / SetTensorAddress / Enqueue
                              ▼
                  logits ──► LaunchGreedy/TopK/TopPSampler ──► token_ids
```

每一步对应的现有类都已在位，但**链路上有三个缺口会让上面的图跑不通**，见 §3。

---

## 2. 与单算子单测的分工

单算子单测已经存在且保持不动，本计划不重复它们：

| 层次 | 已验证内容 | 文件 |
|---|---|---|
| L0 Host 契约 | 元数据、序列化往返、`supportsFormatCombination`、参数拒绝 | `tests/test_{rmsnorm,rope,paged_attention,sampler}*.cpp` |
| L1 Kernel 数值 | 与 CPU 参考逐元素对比 | 同上（GPU 门控） |
| L2 Engine 集成 | **仅 RMSNorm**：真实 engine 构建 + 序列化 + 反序列化 + 推理 | `tests/test_rmsnorm_integration.cpp` |

**本计划补的是 L2 之外的两块**：

1. **L2 扩面**：RoPE / PagedAttention / Sampler 也要走真实 engine 链路。
2. **L3 全流程**：算子按真实调用方式组装（经 config + safetensors + builder），而不是测试里手工 `addPluginV3`。

---

## 3. 现有能力盘点与缺口

### 3.1 已就绪

| 环节 | 现有能力 |
|---|---|
| 配置加载 | `ModelConfig::Load(model_dir)` 读 `config.json` |
| 权重加载 | `WeightLoader::Load/SetWeightMap/GetWeight/GetWeightBySourceKey`，支持 BF16→FP16/FP32 |
| 分发 | `ModelRegistry::Register/Get`；`EngineBuilder::RegisterModelBuilder` 可注入 |
| 构建 | `EngineBuilder::BuildFromConfig(model_dir, engine_path)` 已打通全链路 |
| 推理 | `Engine` 封装 runtime/engine/context，支持 `SetInputShape` / `SetTensorAddress` / `Enqueue` |
| 采样 | `Launch{Greedy,TopK,TopP}Sampler` 设备侧 API |

### 3.2 缺口（**这些是本计划的前置改造项**，不补则 E2/E3 做不了）

| # | 缺口 | 影响 | 证据 |
|---|---|---|---|
| G1 | **Optimization profile 完全未实现**。`EngineBuilder` 声明了 `AddCvOptimizationProfile` / `AddLlmOptimizationProfiles`，但 `builder.cpp` 里**没有任何定义**，也没被调用；`BuildFromConfig` 里是一句 `// TODO: 根据 architecture 添加 optimization profile`。 | 动态 shape（动态 batch / 动态 seq_len）的 engine 构建不出来，E3 全部阻塞；E1/E2 只能跑**全静态 shape**。 | `grep "bool EngineBuilder::Add" src/core/builder.cpp` 返回 0 |
| G2 | **C++ 侧没有 Safetensors 写入能力**。`SafetensorsLoader` 是只读封装；现有夹具 `third_party/safetensors-cpp/gen/model.safetensors` 是 Python 生成的。 | 测试无法自包含地造出 `model.safetensors`，只能提交二进制夹具（污染仓库、不可 review）。 | `safetensors.hh` 有写入 API（见 `serialize-example.cc`），但未被封装 |
| G3 | **`LLMRunner` 仍是 stub**（`Generate()` 抛 not implemented）。 | 无法通过 runner 做多步自回归；E4 只能做到「单步 prefill→logits→采样」，完整 decode 循环留给 Phase 2。 | `src/core/llm_runner.cpp` |

**处理原则**：G1、G2 属于「测试可行性的前提」，建议**在写 E2E 用例之前先补**（见 §6 顺序）；
G3 不补，E4 按单步设计，完整循环由 Phase 2 的 `LLMRunner` 覆盖。

---

## 4. 用例设计

图例：**阻塞项** 标注必须先解决的缺口；**GPU** 表示需真机。

### E1：单算子最小模型闭环（4 条）

**目的**：验证「算子能通过**真实使用路径**被调用」——权重从 safetensors 取出、经 `weight_map` 映射、
挂成常量输入，而不是测试里手工构造 `Weights`。

**做法**：每个算子写一个最小 `IModelBuilder`（测试专用，注册名为 `e2e_<op>`），
网络 = 输入 + 常量权重 + 该 Plugin + 输出；目录下生成 `config.json` + `model.safetensors`。
调用 `EngineBuilder::BuildFromConfig` 生成 `.engine`，再用 `Engine` 加载推理，与参考对比。

| 用例 | 算子 | 关键验证点 | 阻塞项 | GPU |
|---|---|---|---|---|
| E1.1 | RMSNorm | 权重经 `weight_map` 命中；`eps`/`hidden_size` 经 engine 往返后仍正确 | G2 | ✓ |
| E1.2 | RoPE | `position_ids` 作为网络输入传入；`rotary_dim < head_size` 时尾部保持原值 | G2 | ✓ |
| E1.3 | PagedAttention | `block_tables`/`context_lens` 作为输入；跨多个物理块的块表寻址 | G2 | ✓ |
| E1.4 | Sampler | 网络输出 logits 后接采样器；固定 seed 结果可复现 | G2 | ✓ |

> E1.4 的采样器不是 Plugin（是自由函数），因此它验证的是「engine 输出 logits → 采样器 → token」
> 这一段衔接，而不是 `addPluginV3` 路径。

### E2：多算子串联（mini decoder step）（1 条，推荐优先）

**目的**：这是真正的「全流程」——一条 commit 就能验证算子之间的**接口契约**（shape/dtype/布局）
以及整体数值。

**网络结构**（解码一步，全静态 shape）：

```
hidden [B,1,H] ──► RMSNorm ──► QKV 线性层(MatMul+Add) ──► RoPE(q,k) ──┐
                                                                      ├─► PagedAttention(decode)
key_cache / value_cache（网络输入） ───────────────────────────────────┘
                                                       │
                                      RMSNorm ◄────────┘
                                         │
                                  LM Head(MatMul) ──► logits [B,V] ──► Top-K Sampler ──► token_ids
```

| 关键验证点 | 说明 |
|---|---|
| 权重全部来自 safetensors | 通过 `weight_map` 映射 QKV / LM Head / 两处 RMSNorm 权重，验证映射方向正确 |
| 算子间形状契约 | RoPE 输出喂给 PagedAttention、PagedAttention 输出喂给 RMSNorm，任一处 shape/布局不一致都会在构建期或数值上暴露 |
| 整体数值 | 与 PyTorch 搭建的等价网络对比（见 §5），FP16 走 Q13 阈值 |
| 采样衔接 | logits 直接留在 device 上交给采样器，不经过 host |

**阻塞项**：G2（夹具）、G1（若用动态 batch 则也需要）。**GPU**：✓。

### E3：Prefill / Decode 双 profile（阻塞于 G1）

**目的**：验证动态 shape 下 `EngineBuilder::Config` 里的 prefill/decode 两组范围能真正生效。

| 用例 | 输入 | 期望 |
|---|---|---|
| E3.1 | Prefill：`[B, S]`，S 在 `[min_prefill_seq_len, max_prefill_seq_len]` 内变化 | 同一 engine 接受不同 S，输出正确 |
| E3.2 | Decode：`[B, 1]`，B 在 `[min_decode_batch, max_decode_batch]` 内变化 | 同一 engine 接受不同 B |
| E3.3 | 超出 profile 范围 | `SetInputShape` 失败并给出明确错误，而不是静默出错 |

**前置**：先实现 `AddCvOptimizationProfile` / `AddLlmOptimizationProfiles` 并在 `BuildFromConfig` 中按
`model_config.architecture` 选择调用（`cnn` → CV profile，`decoder_only` → prefill+decode 双 profile）。

### E4：错误路径（host 侧即可跑，无需 GPU）

**目的**：确认失败是**可控失败**（返回 false / 抛异常）而不是崩溃、产出半成品 engine、或静默成功。

| 用例 | 场景 | 期望 |
|---|---|---|
| E4.1 | `model_dir` 下无 `config.json` | `BuildFromConfig` 返回 false，不产出 engine 文件 |
| E4.2 | `config.json` 的 `model_type` 未注册 | 返回 false，日志给出 `No registered builder for model type: X` |
| E4.3 | 目录下缺 `model.safetensors` | `WeightLoader::Load` 失败 → 返回 false |
| E4.4 | `weight_map` 指向不存在的 source key | builder 取权重失败 → 返回 false（不得建出错误 engine） |
| E4.5 | Plugin 配置非法（如 PagedAttention 未配 `block_size`） | 构建失败且错误信息指向具体插件 |

> ⚠️ **2026-09-25 更正**：这里原写"E4 全部可在沙箱内执行，是这批用例里唯一能进 CI 的部分"，
> 实测**不成立**：E4.4（`weight_map` 指向不存在的 key）与 E4.5（构建期失败传播）都要走
> `createInferBuilder`，无 GPU 时同样 `GTEST_SKIP`。实际口径是 **3 条沙箱（E4.1/E4.2/E4.3）
> + 2 条真机（E4.4/E4.5）**，与 `PROGRESS.md` §3.10 的记录一致。
> 用例 → 环境的完整映射见 `docs/phase1_5_test_plan.md` §2。

---

## 5. 测试夹具与参考输出

### 5.1 模型目录夹具

统一封装成测试侧的 helper（建议 `tests/e2e_fixture.hpp` + `.cpp`）：

```cpp
struct MiniModelFixture {
    std::string dir;                 // /tmp/mini_trt_llm_e2e_XXXX
    std::map<std::string, std::vector<float>> tensors;  // 张量名 → 数据
    std::string config_json;         // model_type / architecture / hyper_params / weight_map
    static MiniModelFixture Create(const std::string& model_type, ...);
};
```

- 目录用 `mkdtemp` 创建，测试结束清理，避免残留污染。
- `config.json` 直接字符串写盘。
- `model.safetensors` 由 **G2 补齐的写入 helper** 生成（不提交二进制、不依赖 Python）。
- 张量数据用确定性伪随机（`sin(k*index)`），保证与参考实现输入一致。

### 5.2 参考输出

| 用例 | 参考来源 |
|---|---|
| E1.* | 复用 `tests/test_reference.hpp` 的 CPU 双精度实现 |
| E2 | `scripts/ref_mini_block.py`：用 PyTorch 搭同结构网络，导出参考 logits / token 到 `.bin`；C++ 侧载入比对 |
| E3 | 同 E2，另加不同 shape 的组合 |

**为什么 E2 需要 Python 参考**：E1 的参考实现可以直接复用，但多算子串联后的数值依赖每层的
布局与 dtype 提升规则，用 PyTorch 作为独立实现比对，才能避免「自己写的参考和被测实现犯同一个错」。
`scripts/ref_rope.py` 已经用这个思路验证过 RoPE 约定（与 HF 零差异），可作模板。

---

## 6. 实施顺序（依赖驱动）

```
Step 1  G2：补 Safetensors 写入 helper ────────────┐
Step 2  E4：错误路径用例（纯 host，先拿到 CI 防线） │ 可并行
Step 3  E1：四个算子的单算子闭环 ──────────────────┘
Step 4  E2：多算子串联（含 scripts/ref_mini_block.py）
Step 5  G1：实现 optimization profile
Step 6  E3：动态 shape / 双 profile
```

Step 1 与 Step 2 互不依赖可并行；Step 5 是 E3 的硬前置，若 Phase 2 更早需要动态 shape，
应把它提到 Phase 2 开头做。

---

## 7. 通过标准

- E4 全绿，且在**沙箱内**即可执行（进 CI）。
- E1/E2/E3 在真机执行时全绿；沙箱内以 `GTEST_SKIP` 跳过，不产生噪声。
- 精度：FP32 绝对误差 < `1e-4`（全流程链路更长，比单算子的 `1e-5` 放宽一档）；
  FP16 相对误差 < `1e-3` + 小值绝对误差 Guardrail（D4）。
- 任一新用例失败必须同步补一条针对性回归用例，并在 `docs/TROUBLESHOOTING.md` 留下定位路径。

---

## 8. 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| G1 未实现导致 E3 长期阻塞 | 动态 shape 无覆盖，Phase 2 的真实模型又强依赖它 | 把 profile 实现纳入 Phase 2 的第一批任务，不要等测试来推 |
| 全流程用例依赖真机 | 沙箱内只有 E4 有信号 | E1/E2 尽量把「构建期」与「运行期」拆开：构建期错误（权重缺失、shape 不匹配）在无 GPU 时也应能通过 `GTEST_SKIP` 之外的方式暴露 |
| PyTorch 参考与实现同错 | 串联数值偏差被掩盖 | 参考脚本必须用**独立实现**（HF 算子组装），不复用我们的公式推导 |
| 测试时间膨胀 | 每个用例都要建 engine，累积耗时长 | 同一 fixture 的多个断言合并到一条用例；engine 文件按 `model_type+shape` 缓存复用 |

---

*文档版本：v1.0*  
*关联文档：`docs/phase1_development_plan.md`（§7 测试策略）、`docs/phase0_model_loading_test_plan.md`（T3 DummyBuilder 端到端，即本计划 E1 的前身）、`docs/TROUBLESHOOTING.md`*
