# Phase 1 开发方案：Plugin 基础

> 目标：实现 TensorRT-LLM 推理所需的基础自定义 Plugin，跑通 `IPluginV3` 模板、序列化、注册、测试流程，为 Phase 2 GPT-2 原生构建提供算子基础。

---

## 1. 阶段目标

1. 完善 `IPluginV3` 基类封装，适配 TensorRT 10.x。
2. 实现 `RMSNormPlugin`。
3. 实现 `RoPEPlugin`。
4. 实现基础版 `PagedAttentionPlugin`（仅 Decoding 阶段，支持 GQA/MHA）。
5. 实现 Sampler CUDA Kernels（Greedy / Top-K / Top-P）。
6. 为每个 Plugin / Kernel 编写单元测试。

**产出物**
- 4 个 Plugin 类 + Creator + 注册逻辑
- 4 个 CUDA kernel 实现
- 对应单元测试
- 更新后的 `docs/PROGRESS.md`

---

## 2. 架构决策

### 2.1 Plugin API：IPluginV3（TRT 10.x）

- **为什么**：TensorRT 10.15.1 推荐接口，V2 属于 legacy。
- 必须实现的接口：
  - `getOutputDataTypes`
  - `getOutputShapes`
  - `configurePlugin`
  - `getWorkspaceSize`
  - `enqueue`
  - `getSerializationSize` / `serialize`
  - `clone`

### 2.2 Plugin 命名

- 层名加 `mini_trt_llm_` 前缀。
- Creator 名用 `MiniTrtLlmXxxPluginCreator`。
- 避免与 TRT 原生层重名。

### 2.3 动态 Shape

- 所有 Plugin 显式支持动态 `batch_size` 和动态 `seq_len`。
- 通过 `IPluginV3::getOutputShapes` 输出动态维度表达式。

### 2.4 Scratch 显存

- 所有中间缓存严格通过 `getWorkspaceSize()` 申请。
- `enqueue` 中禁止 `cudaMalloc`。

### 2.5 精度

- 默认 FP16，支持 FP32 fallback。
- INT8 放到后续迭代。

---

## 3. 目录结构

```
mini_trt_llm/
├── include/mini_trt_llm/plugins/
│   ├── iplugin_v3_base.hpp      # 已存在，需完善
│   ├── plugin_registry.hpp       # 已存在
│   ├── rmsnorm_plugin.hpp
│   ├── rope_plugin.hpp
│   ├── paged_attention_plugin.hpp
│   └── sampler/
│       ├── greedy_sampler.hpp    # 已存在，需补 CUDA kernel
│       ├── topk_sampler.hpp      # 已存在，需补 CUDA kernel
│       └── topp_sampler.hpp      # 已存在，需补 CUDA kernel
├── src/plugins/
│   ├── plugin_registry.cpp       # 已存在，需注册新 plugin
│   ├── rmsnorm_plugin.cpp
│   ├── rope_plugin.cpp
│   ├── paged_attention_plugin.cpp
│   └── sampler/
│       ├── greedy_sampler.cu
│       ├── topk_sampler.cu
│       └── topp_sampler.cu
└── tests/
    ├── test_rmsnorm_plugin.cpp
    ├── test_rope_plugin.cpp
    ├── test_paged_attention_plugin.cpp
    └── test_sampler.cpp
```

---

## 4. 各 Plugin 详细设计

### 4.1 RMSNormPlugin

**功能**：对输入做 RMSNorm，公式：
```
rms = sqrt(mean(x^2) + eps)
output = x / rms * weight
```

**输入**
- `input`: `[batch_size, seq_len, hidden_size]`，FP16/FP32
- `weight`: `[hidden_size]`，FP16/FP32

**输出**
- `output`: `[batch_size, seq_len, hidden_size]`

**属性**
- `eps`: float，默认 1e-6
- `hidden_size`: int

**CUDA Kernel 设计**
- 每个 row（`hidden_size` 个元素）一个 block 或 warp。
- 使用 warp-level reduction 计算平方和。
- 向量化读写：`half2` / `float4`。

**测试**
- 与 PyTorch `rms_norm` 参考实现逐元素对比，误差 < 1e-3（FP16）。
- 测试动态 shape：`batch_size=1,2`，`seq_len=1,64,128`，`hidden_size=768`。

---

### 4.2 RoPEPlugin

**功能**：对 Query/Key 应用 Rotary Position Embedding。

**输入**
- `query`: `[batch_size, num_heads, seq_len, head_size]`
- `key`: `[batch_size, num_kv_heads, seq_len, head_size]`
- `position_ids`: `[batch_size, seq_len]`，int32

**输出**
- `rotated_query`: 同 query shape
- `rotated_key`: 同 key shape

**属性**
- `head_size`: int
- `num_heads`: int
- `num_kv_heads`: int
- `rotary_dim`: int，默认等于 head_size
- `base`: float，默认 10000.0

**CUDA Kernel 设计**
- 每个 (batch, head, position) 一个 thread 或 warp。
- 预计算 `sin/cos` 到 workspace 或按公式实时计算。
- 支持 paired rotation：对相邻两个维度做旋转。

**测试**
- 与 HuggingFace `transformers.models.llama.modeling_llama.apply_rotary_pos_emb` 对比。
- 测试 `position_ids` 非连续情况。

---

### 4.3 PagedAttentionPlugin

**功能**：对已经存储在 Paged KV Cache 上的 Key/Value 做 attention 计算。

**输入**
- `query`: `[batch_size, num_heads, 1, head_size]`（decode 阶段）或 `[batch_size, num_heads, seq_len, head_size]`（prefill 阶段）
- `key_cache`: `[num_blocks, block_size, num_kv_heads, head_size]`
- `value_cache`: 同 key_cache
- `block_tables`: `[batch_size, max_blocks_per_seq]`，int32
- `context_lens`: `[batch_size]`，int32
- `max_context_len`: scalar int32

**输出**
- `output`: `[batch_size, num_heads, seq_len, head_size]`

**属性**
- `num_heads`: int
- `num_kv_heads`: int
- `head_size`: int
- `block_size`: int，**强制显式配置，无默认值**
- `scale`: float，默认 `1.0 / sqrt(head_size)`

**CUDA Kernel 设计（基础版）**
- 先实现 **Decoding 阶段**（`seq_len == 1`），Prefill 阶段计入后续迭代。
- 每个 (batch, head) 一个 block。
- 每个 block 按 `block_table` 读取历史 K/V，计算 softmax(QK^T / scale)V。
- 使用 shared memory 缓存当前 query 和部分 K/V。
- 支持 **GQA/MHA**：每个 query head 通过 `num_heads / num_kv_heads` 找到对应的 kv head；MQA 作为 GQA 的 num_kv_heads=1 特例。

**测试**
- 构造简单的 2D K/V 序列，与 PyTorch `F.scaled_dot_product_attention` 在因果 mask 下对比。
- 验证 block table 分页读取正确。

---

### 4.4 Sampler CUDA Kernels

#### 4.4.1 Greedy Sampler

**功能**：取 logits 中最大值的索引。

**输入**
- `logits`: `[batch_size, vocab_size]`，FP16/FP32

**输出**
- `token_ids`: `[batch_size]`，int32

**CUDA Kernel**
- 每个 batch 一个 block，block 内 argmax reduction。

#### 4.4.2 Top-K Sampler

**功能**：在 Top-K 个候选内按 softmax 概率采样。

**输入**
- `logits`: `[batch_size, vocab_size]`
- `k`: `[batch_size]`，int32，每个 sample 独立配置

**输出**
- `token_ids`: `[batch_size]`

**CUDA Kernel**
- 每个 batch 一个 block。
- 使用 shared memory 做局部 Top-K（如 bitonic sort 或 selection algorithm）。
- 对 Top-K 做 softmax 后按 cumsum + random 采样。

#### 4.4.3 Top-P Sampler

**功能**：按累积概率截断到 Top-P，再采样。

**输入**
- `logits`: `[batch_size, vocab_size]`
- `p`: `[batch_size]`，float32，每个 sample 独立配置

**输出**
- `token_ids`: `[batch_size]`

**CUDA Kernel**
- **第一阶段：使用 CUB 实现**，保证逻辑与参考实现一致。
- 全局排序 + softmax + cumsum，找到 cutoff。
- 在 cutoff 内采样。
- CUB 调用显式传入 `cudaStream_t`，避免默认 stream 误用。
- 手写高性能版本（bitonic sort、warp-level reduction 等）计入后续迭代。

**测试**
- Greedy：与 `torch.argmax` 对比。
- Top-K / Top-P：与 HuggingFace `TopKLogitsWarper` / `TopPLogitsWarper` + `torch.multinomial` 对比分布。

---

## 5. IPluginV3 基类完善

当前 `include/mini_trt_llm/plugins/iplugin_v3_base.hpp` 可能只有空壳。需要补全：

```cpp
class MiniTrtLlmPluginBase : public nvinfer1::IPluginV3 {
 public:
    // 公共序列化辅助函数
    virtual void setPluginNamespace(const char* pluginNamespace) noexcept override;
    virtual const char* getPluginNamespace() const noexcept override;
    // 子类只需实现业务相关接口

 protected:
    std::string namespace_;
};
```

是否需要 template / CRTP 模式来减少重复代码，需 review 后决定。

---

## 6. Plugin 注册

在 `src/plugins/plugin_registry.cpp` 中新增：

```cpp
REGISTER_TENSORRT_PLUGIN(MiniTrtLlmRmsnormPluginCreator);
REGISTER_TENSORRT_PLUGIN(MiniTrtLlmRopePluginCreator);
REGISTER_TENSORRT_PLUGIN(MiniTrtLlmPagedAttentionPluginCreator);
```

并通过 `PluginRegistry` 或静态初始化确保测试能获取到 Creator。

---

## 7. 测试策略

### 7.1 测试分层

| 层级 | 内容 | 运行环境 |
|---|---|---|
| L1：Kernel 单测 | 直接调用 CUDA kernel，对比 PyTorch | GPU 真机 |
| L2：Plugin 单测 | 通过 TRT network 构建 engine，跑 inference | GPU 真机 |
| L3：集成测试 | 多个 plugin 组合成一个子网络 | GPU 真机 |

### 7.2 参考输出生成与对比策略

为每个 plugin 写 Python 脚本生成参考输出：
- `scripts/ref_rmsnorm.py`
- `scripts/ref_rope.py`
- `scripts/ref_paged_attn.py`
- `scripts/ref_sampler.py`

**对比策略（分层）**
- **Plugin/算子层**：与 PyTorch/HF 参考输出做逐元素对比。
- **端到端 Generation 层**：对比分布 / Logits 统计量（如 top-5 token 重合率、KL 散度、生成序列的统计特征），不做逐 token 强制一致。

### 7.3 精度标准

- FP32：绝对误差 < 1e-5
- FP16：相对误差 < 1e-3（0.1%），并配合绝对误差 Guardrail：
  - 若参考值绝对值较小（如 |ref| < 1e-4），改用绝对误差 < 1e-4。
  - 避免相对误差在接近零时被放大导致误判。

---

## 8. 开发顺序

建议按依赖顺序推进：

```
Week 1: IPluginV3 基类 + RMSNormPlugin
Week 2: RoPEPlugin
Week 3: Sampler Kernels（Greedy / Top-K / Top-P）
Week 4: PagedAttentionPlugin（Decoding only, MHA）
Week 5: 集成测试 + 文档更新
```

实际执行时可并行：RMSNorm 和 RoPE 相互独立，Sampler 和 PagedAttention 也可并行。

---

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| IPluginV3 接口细节与 TRT 10.15.1 头文件不完全一致 | 编译/链接失败 | 边实现边对照 TRT 10 头文件和示例 |
| PagedAttention 核性能差 | Decode 阶段吞吐低 | 先实现正确性，后续用 warp/block 优化 |
| Top-K/Top-P 核复杂、易错 | 采样结果与参考不一致 | 先用 CUB 实现正确版本，再手写优化 |
| 沙箱无 GPU，无法本地验证 | 开发迭代慢 | 代码提交后用户在 WSL2 真机跑，或用 nsys/ncu 分析 |
| RoPE 的 position_ids 处理 | 与 HF 实现对不齐 | 与 `transformers` 实现逐 case diff |
| GPU 验证需人工触发 | 沙箱无 GPU，所有 CUDA/TensorRT 测试必须在真机运行 | 任何验证/测试任务执行前需经人工确认 |

---

## 10. 关键决策确认清单

> 本节合并自原 `phase1_pending_confirmations.md`。15 项待确认问题已全部确认，无遗留阻塞项，Phase 1 可进入编码。

### 10.1 阶段级决策（D1–D5）

| 编号 | 决策项 | 确认结论 |
|---|---|---|
| D1 | PagedAttention 范围 | Decoding 阶段 GQA/MHA，Prefill 计入后续迭代 |
| D2 | Sampler 实现路线 | 先 CUB 保正确性，手写优化 kernel 计入后续迭代 |
| D3 | 对比策略 | Plugin/算子层逐元素对比，Generation 层对比分布/Logits 统计量 |
| D4 | FP16 精度标准 | 相对误差 < 1e-3，配合绝对误差 Guardrail |
| D5 | GPU 验证 | 所有 CUDA/TensorRT 测试需人工确认后执行 |

### 10.2 细化确认项（Q1–Q15）

| 编号 | 问题 | 确认结论 | 关键理由 |
|---|---|---|---|
| Q1 | RMSNorm 是否支持 bias | 不支持，只做标准 RMSNorm | 保持标准实现；后续模型确有需要时再扩展 |
| Q2 | RMSNorm weight 的位置 | 作为 Plugin 输入（第二输入） | 依赖 TRT 常量折叠，无动态内存开销；与 WeightLoader 的常量层路径一致，便于后续 INT8 统一处理 scale |
| Q3 | RoPE `rotary_dim` | 默认等于 `head_size`，可通过属性配置 | 兼容部分旋转变体（GPT-NeoX / CodeLLaMA 等） |
| Q4 | RoPE `position_ids` | 作为 Plugin 输入 | 与 HuggingFace 行为一致，支持非连续 position 与 KV Cache 场景 |
| Q5 | PagedAttention `block_size` | 强制显式配置，无默认值 | 避免默认值在跨模型/跨场景时引入隐式问题 |
| Q6 | PagedAttention `scale` | 作为 Plugin 属性，默认 `1/sqrt(head_size)` | 属性化更简洁，覆盖绝大多数模型 |
| Q7 | Sampler `k` / `p` | per-batch tensor `[batch_size]` | 兼容连续批处理，每请求可独立配置；kernel 内索引开销可忽略 |
| Q8 | Sampler 随机种子 | host 传 seed + offset，device 用 Philox 类确定性算法 | 兼顾测试可复现与避免 host-device 频繁同步 |
| Q9 | 是否支持 ALiBi / 其他位置编码 | Phase 1 不支持 | 聚焦 RoPE + PagedAttention 这条最通用路径 |
| Q10 | Plugin 标量属性存储精度 | 统一用 float | 精度足够，运行时再按需 cast |
| Q11 | Top-K/Top-P 的 `vocab_size` 上限 | 假设 ≤ 128K，按此设计 shared memory | 覆盖当前目标模型；超大 vocab 后续再优化 |
| Q12 | 参考输出的随机 seed | 全部使用固定 seed（如 42） | 参考数据可复现，便于回归对比 |
| Q13 | 单元测试是否要求逐 bit 一致 | 不要求，按相对/绝对误差阈值判定 | FP16 末位差异不应导致测试不稳定 |
| Q14 | serialize / deserialize 完成度 | Phase 1 完整实现并测试 | IPluginV3 接口硬性要求，尽早验证可省后续返工 |
| Q15 | CUB vs Thrust | 选 CUB | CUB 天然显式传 `cudaStream_t`，避免 Thrust 误用默认流；性能基线更接近后续手写 kernel |

---

## 11. GPU 验证与人工确认流程

- 沙箱环境无 GPU，无法本地运行 CUDA/TensorRT 测试。
- 所有涉及 GPU 的编译后测试、nsys/ncu profile、精度验证，必须在用户 WSL2 真机上由用户手动触发。
- 代码提交前，Agent 只做：
  1. 静态代码检查（编译通过、语法正确）。
  2. 非 CUDA 分支的 host 侧测试。
  3. 代码 review 级别的逻辑检查。
- 任何 GPU 测试命令在执行前必须获得用户明确确认。

---

## 12. 与 Phase 2 的衔接

Phase 1 完成后，Phase 2 的 `GPT2ModelBuilder` 将直接使用：
- `RMSNormPlugin`（如果 GPT-2 改用 RMSNorm；否则可能先不用）
- `RoPEPlugin`
- `PagedAttentionPlugin`
- `GreedySampler` / `TopKSampler` / `TopPSampler`

因此 Phase 1 的接口设计必须预留 GPT-2 所需的 shape 和属性。
