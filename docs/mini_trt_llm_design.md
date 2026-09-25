# mini_trt_llm 需求分析、方案设计与实现路径（v1.0）

> 目标：用 `mini_trt_llm` 完全替换现有 `0_resnet18_onnx` 与 `1_gpt2_onnx` 两个 ONNX 模型加载模块。  
> “替换”含义：不是删除 ResNet18 / GPT-2 的推理能力，而是把两个独立的 ONNX 加载示例模块合并进统一的 `mini_trt_llm` 多模态推理框架，最终从根 `CMakeLists.txt` 中移除 `0_resnet18_onnx` 与 `1_gpt2_onnx` 的 `add_subdirectory`，由 `mini_trt_llm` 提供等价的推理能力。

> 确认前提（final）：
> - 替换范围：两个模块都替换；
> - 模型来源：同时支持 **原生权重加载（方案 A）** 与 **ONNX + 自定义 Plugin（方案 B）**，方案 B 需规避插件与原生命名冲突；
> - Tokenizer：引入 **SentencePiece 源码嵌入（v0.2.0）**，并预留多模态 tokenizer 定制化接口；
> - 权重格式：**Safetensors**，使用 `syoyo/safetensors-cpp` 解析；
> - 转换工具：`mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`，HF checkpoint → `config.json + model.safetensors`；
> - 迁移策略：保守并行，先共存后移除；
> - 精度基准：LLM 仅与 `ref_output.bin`（PyTorch FP32）对比；
> - INT8 校准：ResNet18 / LLM 的 INT8 支持延后到后续迭代，Phase 4 先完成 FP32/FP16；
> - 模型范围：架构预留 Encoder-Decoder 接口，Phase 0 只实现 CNN + Decoder-only Transformer；
> - 注意力：MHA / GQA / MQA / Sliding Window 全支持；
> - 配置格式：**JSON**；
> - 动态 shape：LLM 区分 **Prefill / Decode** 两个 OptimizationProfile；CV Phase 0 只支持动态 batch；
> - TensorRT 版本：**10.15.1（版本宏 v101501）**，Plugin API 使用 **`IPluginV3`**；
> - CMake 选项：`BUILD_TESTS=OFF`，`BUILD_EXAMPLES=OFF`，CI 显式 `-DBUILD_TESTS=ON`；
> - 错误处理：构造/初始化抛异常，运行时推理接口返回 bool；
> - 日志：新增 `MINI_TRT_LOG_INFO/WARN/ERROR` 独立业务日志宏；
> - 内存池：Phase 0 简单 `cudaMalloc/cudaFree` 封装，池化后续迭代。

> 实现偏差说明：本文档 API 草图以 `nlohmann::json` 表述配置结构。Phase 0 因网络受限无法引入外部依赖，实际实现使用自研 `utils/json.hpp`（`JsonValue`）；后续迭代替换为 `nlohmann/json` 单头文件。

---

## 1. 需求分析

### 1.1 项目定位

`mini_trt_llm` 是一个**面向 NVIDIA Turing / sm_75 的极简多模态 TensorRT 推理框架**，首批支持：

- **CV 模型**：以 ResNet18 为代表的图像分类模型；
- **LLM 模型**：以 GPT-2 为代表的自回归文本生成模型；
- **未来可扩展**：Encoder-Decoder、ViT、Audio、多模态融合等（通过统一的 `BaseTokenizer`、`IModelBuilder`、`WeightMapper` 扩展接口预留）。

### 1.2 功能需求

| ID | 需求 | 说明 | 优先级 |
|---|---|---|---|
| F1 | 双模式模型加载 | 支持原生 TRT Network 构建（方案 A）与 ONNX 解析+插件替换（方案 B） | P0 |
| F2 | ResNet18 图像分类 | 替换 `0_resnet18_onnx`，支持 FP32/FP16，输出 1000 维 logits；INT8 延后 | P1 |
| F3 | GPT-2 文本生成 | 替换 `1_gpt2_onnx`，支持动态 batch/seq_len，Prefill → Decode 自回归 | P0 |
| F4 | 自定义 Plugin 体系 | RoPE、RMSNorm、PagedAttention、Sampler 等，基于 `IPluginV3` | P0 |
| F5 | Paged KV Cache | BlockAllocator + KV Cache 管理器，Decode 阶段零 Host-Device 同步拷贝 | P0 |
| F6 | CUDA Sampler | Greedy / Top-K / Top-P 采样核 | P0 |
| F7 | C++ Tokenizer | SentencePiece 源码嵌入，预留多模态 tokenizer 抽象接口 | P1 |
| F8 | Safetensors 权重加载 | 原生构建时从 Safetensors 读取模型权重 | P0 |
| F9 | 精度对比 | GPT-2 输出与 `ref_output.bin`（PyTorch FP32）对比 cosine / max abs diff | P0 |
| F10 | Benchmark | mean / p50 / p99 延迟与 throughput，CUDA Event 计时 | P0 |
| F11 | 配置驱动模型构建 | JSON 配置描述模型结构，减少硬编码 | P0 |
| F12 | Dummy model | 超小模型用于快速回归测试 | P1 |
| F13 | Python 转换工具 | `mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`：HF checkpoint → config.json + model.safetensors | P1 |
| F14 | INT8 校准（后续迭代） | ResNet18 / LLM 的 INT8 支持，含 Calibration | P2 |
| F15 | CV 动态分辨率（后续迭代） | 输入分辨率动态变化 | P2 |

### 1.3 非功能需求

- **硬件约束**：仅适配 `sm_75`（GTX 1660 Ti Mobile），`CMAKE_CUDA_ARCHITECTURES=75`。
- **精度支持**：FP32、FP16；**禁用 FP8** 及 Ampere/Hopper 独占特性；INT8 后续迭代。
- **数据类型**：权重接受 F32 / F16 / BF16，BF16 在 sm_75 下需转换为 FP32/FP16 后参与计算。
- **性能约束**：Decode 自回归循环内禁止 `cudaMemcpy` 同步拷贝；Plugin `enqueue` 内禁止 `cudaMalloc`。
- **工程规范**：C++17、命名空间 `mini_trt_llm`、Google C++ 风格、RAII 管理 GPU 资源。
- **代码风格**：4 空格缩进、80 列换行、K&R 大括号，遵循根目录 `.clang-format`。
- **构建选项**：`BUILD_TESTS=OFF`，`BUILD_EXAMPLES=OFF`，CI 显式 `-DBUILD_TESTS=ON`。

---

## 2. 方案设计

### 2.1 目录结构

```text
mini_trt_llm/
├── CMakeLists.txt
├── include/mini_trt_llm/
│   ├── core/
│   │   ├── precision.hpp
│   │   ├── engine.hpp
│   │   ├── builder.hpp
│   │   ├── model_config.hpp
│   │   ├── model_registry.hpp
│   │   ├── imodel_builder.hpp
│   │   ├── native_builder.hpp
│   │   ├── onnx_builder.hpp
│   │   ├── llm_runner.hpp
│   │   └── cv_runner.hpp
│   ├── plugins/
│   │   ├── plugin_registry.hpp
│   │   ├── iplugin_v3_base.hpp
│   │   ├── rope_plugin.hpp
│   │   ├── rmsnorm_plugin.hpp
│   │   ├── paged_attention_plugin.hpp
│   │   └── sampler_plugin.hpp
│   ├── kv_cache/
│   │   ├── block_allocator.hpp
│   │   └── paged_kv_cache.hpp
│   ├── sampler/
│   │   ├── greedy_sampler.hpp
│   │   ├── topk_sampler.hpp
│   │   └── topp_sampler.hpp
│   ├── tokenizer/
│   │   ├── base_tokenizer.hpp
│   │   └── sentencepiece_tokenizer.hpp
│   └── utils/
│       ├── cuda_check.hpp
│       ├── logger.hpp
│       ├── timer.hpp
│       ├── memory_pool.hpp
│       ├── io.hpp
│       └── safetensors_loader.hpp
├── src/
│   ├── core/
│   ├── plugins/
│   ├── kv_cache/
│   ├── sampler/
│   ├── tokenizer/
│   └── utils/
├── tests/              # test_*.cpp 汇总为单一目标 mini_trt_llm_tests
├── third_party/
│   ├── sentencepiece/        # 源码嵌入
│   └── safetensors-cpp/      # 源码嵌入或 add_subdirectory
└── tools/
    └── convert/
        └── hf_to_mini_trt_llm.py
```

> 测试约定：`mini_trt_llm/tests/` 下所有 `test_*.cpp` 汇总编译为单一目标 `mini_trt_llm_tests`，不按模块拆分为独立二进制。全量执行用 `ctest --test-dir build --output-on-failure`，选择性执行用 `--gtest_filter=`。

### 2.2 模块依赖关系

```text
                 ┌─────────────────┐
                 │   applications  │
                 └────────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ LLMRunner│    │ CVRunner │    │ Benchmark│
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌─────────────────┐
              │ Engine (runtime)│
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐ ┌────────────┐ ┌──────────┐
   │NativeBuilder│ │OnnxBuilder │ │ Plugins  │
   └────┬─────┘ └─────┬──────┘ └────┬─────┘
        │             │             │
        ▼             ▼             ▼
   ModelRegistry  Subgraph      IPluginV3
   (GPT2/ResNet)  Replacer
```

### 2.3 核心类设计

#### 2.3.1 `mini_trt_llm::Engine`

```cpp
class Engine {
 public:
  explicit Engine(const std::string& engine_path, Logger& logger);
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  void SetInputShape(const std::string& name, nvinfer1::Dims dims);
  void SetTensorAddress(const std::string& name, void* ptr);

  bool Enqueue(cudaStream_t stream);
  void Synchronize(cudaStream_t stream);

  struct BenchResult {
    float mean_ms;
    float p50_ms;
    float p99_ms;
    float throughput;
  };
  BenchResult Benchmark(int batch_size, int seq_len,
                        int n_warmup, int n_run);

 private:
  Logger& logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};
```

#### 2.3.2 `mini_trt_llm::EngineBuilder`

```cpp
class EngineBuilder {
 public:
  struct Config {
    Precision precision = Precision::FP16;
    size_t workspace_bytes = 1UL << 30;

    // CV dynamic batch
    int opt_batch = 1;
    int max_batch = 16;

    // LLM prefill profile
    int opt_prefill_batch = 1;
    int max_prefill_batch = 4;
    int opt_prefill_seq_len = 64;
    int max_prefill_seq_len = 512;

    // LLM decode profile
    int opt_decode_batch = 1;
    int max_decode_batch = 4;
    int opt_decode_seq_len = 1;
    int max_decode_seq_len = 512;
  };

  explicit EngineBuilder(Logger& logger, const Config& config);

  // 方案 A：从 Safetensors + JSON config 构建
  bool BuildFromConfig(const std::string& model_dir,
                       const std::string& engine_path);

  // 方案 B：从 ONNX + Plugin 替换构建
  bool BuildFromOnnx(const std::string& onnx_path,
                     const std::string& engine_path,
                     const std::vector<std::string>& plugin_ops);

 private:
  Logger& logger_;
  Config config_;
  std::unique_ptr<ModelRegistry> registry_;
};
```

#### 2.3.3 `mini_trt_llm::IModelBuilder`

```cpp
class IModelBuilder {
 public:
  virtual ~IModelBuilder() = default;

  virtual bool Build(nvinfer1::INetworkDefinition* network,
                     const WeightLoader& weights,
                     const ModelConfig& config) = 0;

  virtual std::string Name() const = 0;
};
```

#### 2.3.4 `mini_trt_llm::ModelConfig`

```cpp
struct ModelConfig {
  std::string model_type;       // "gpt2", "resnet18", ...
  std::string architecture;     // "decoder_only", "cnn", "encoder_decoder"
  nlohmann::json hyper_params;  // model-specific params
  nlohmann::json weight_map;    // source_key -> trt_name mapping
};
```

#### 2.3.5 `mini_trt_llm::WeightLoader`

```cpp
class WeightLoader {
 public:
  virtual ~WeightLoader() = default;

  virtual bool Load(const std::string& safetensors_path) = 0;

  // 返回权重指针与字节数；内部完成 BF16->FP32/FP16 转换
  virtual const void* GetWeight(const std::string& trt_name,
                                nvinfer1::DataType target_type,
                                size_t* bytes) = 0;

  // 按 JSON weight_map 做 key 映射
  virtual void SetWeightMap(const nlohmann::json& map) = 0;
};
```

#### 2.3.6 `mini_trt_llm::LLMRunner`

```cpp
class LLMRunner {
 public:
  struct GenerateOptions {
    int max_new_tokens = 20;
    float temperature = 1.0f;
    int top_k = 1;
    float top_p = 1.0f;
  };

  LLMRunner(std::shared_ptr<Engine> prefill_engine,
            std::shared_ptr<Engine> decode_engine,
            std::shared_ptr<BaseTokenizer> tokenizer);

  std::vector<int64_t> Generate(const std::vector<int64_t>& input_ids,
                                const GenerateOptions& options);

 private:
  std::shared_ptr<Engine> prefill_engine_;
  std::shared_ptr<Engine> decode_engine_;
  std::shared_ptr<BaseTokenizer> tokenizer_;
  std::unique_ptr<PagedKVCache> kv_cache_;
};
```

#### 2.3.7 `mini_trt_llm::CVRunner`

```cpp
class CVRunner {
 public:
  CVRunner(std::shared_ptr<Engine> engine,
           const std::vector<float>& mean,
           const std::vector<float>& std);

  std::vector<float> Infer(const std::vector<float>& image_nchw,
                           int batch_size);

  Engine::BenchResult Benchmark(int batch_size, int n_warmup, int n_run);

 private:
  std::shared_ptr<Engine> engine_;
  std::vector<float> mean_;
  std::vector<float> std_;
};
```

### 2.4 Tokenizer 多模态抽象设计

```cpp
class BaseTokenizer {
 public:
  virtual ~BaseTokenizer() = default;
  virtual bool Load(const std::string& vocab_path) = 0;
  virtual std::vector<int64_t> Encode(const std::string& text) const = 0;
  virtual std::string Decode(const std::vector<int64_t>& ids) const = 0;
  virtual size_t VocabSize() const = 0;
};

class SentencePieceTokenizer : public BaseTokenizer {
 public:
  bool Load(const std::string& vocab_path) override;
  std::vector<int64_t> Encode(const std::string& text) const override;
  std::string Decode(const std::vector<int64_t>& ids) const override;
  size_t VocabSize() const override;

 private:
  // sentencepiece::SentenceProcessor sp_;
};

// 未来可扩展：
// class TiktokenTokenizer : public BaseTokenizer { ... };
// class ClipTokenizer : public BaseTokenizer { ... };
```

### 2.5 Plugin 命名与冲突规避（方案 B 关键）

为避免 ONNX 中插件层与 TensorRT 原生命名冲突，约定：

- **Plugin 层名前缀**：`mini_trt_llm_`，例如 `mini_trt_llm_rope_0`。
- **Plugin Creator 名**：`MiniTrtLlmRoPEPluginCreator`、`MiniTrtLlmRMSNormPluginCreator`。
- **Plugin 命名空间/版本**：统一使用 `IPluginV3` 接口（TRT 10.x）。
- **ONNX 节点替换策略**：
  - 若 ONNX 已导出为 custom op（domain = `mini_trt_llm`），直接通过 `PluginRegistry` 创建对应 Plugin。
  - 若 ONNX 使用原生 op 组合（如 `Split` + `Sin/Cos` + `Mul` + `Add` 实现 RoPE），`OnnxBuilder` 按子图模式匹配并替换为 Plugin。

```cpp
OnnxBuilder builder(logger, config);
builder.RegisterSubgraphReplacer(std::make_unique<RoPESubgraphReplacer>());
builder.RegisterSubgraphReplacer(std::make_unique<RMSNormSubgraphReplacer>());
builder.BuildFromOnnx(onnx_path, engine_path);
```

### 2.6 原生网络构建（方案 A）

#### GPT-2

从 Safetensors 读取权重后，用 TRT Network API 构建：

1. Token Embedding + Positional Embedding（或 RoPE 替代位置编码）；
2. Transformer Block × N：
   - RMSNorm / LayerNorm
   - QKV Projection（MatMul + Add）
   - RoPE Plugin
   - PagedAttention Plugin（支持 MHA/GQA/MQA）
   - FFN（MatMul → Add → GELU → MatMul → Add）
   - Residual Connection
3. Final RMSNorm / LayerNorm + LM Head；
4. Sampler Plugin（Greedy / Top-K / Top-P）。

#### ResNet18

1. Conv + BN + ReLU 堆叠；
2. MaxPool / AvgPool；
3. 全连接输出 1000 维；
4. 权重从 Safetensors 加载。

### 2.7 精度验证策略

- **GPT-2**：构建 FP32 engine，输入 `"The quick brown fox"`，与 `1_gpt2_onnx/ref_output.bin` 对比：
  - `cosine_sim >= 0.9999`
  - `max_abs_diff < 1e-4`（具体阈值待首次运行后确认）
- **ResNet18**：使用现有校准数据与 ImageNet 1000 分类输出对比（FP32/FP16 阶段以功能正确性为主）。
- **Dummy model**：固定随机输入，对比 PyTorch 同等结构输出。

---

## 3. 实现路径

### Phase 0：基础设施与通用化骨架（2 周）

1. 创建 `mini_trt_llm/` 目录结构。
2. 实现 `utils/cuda_check.hpp`、`utils/logger.hpp`、`utils/timer.hpp`、`utils/memory_pool.hpp`、`utils/io.hpp`。
3. 实现 `utils/safetensors_loader.hpp`：基于 `syoyo/safetensors-cpp` 的封装。
4. 迁移并完善 `Precision` 枚举到 `core/precision.hpp`。
5. 实现 `core/engine.hpp`：封装 TRT runtime/engine/context。
6. 实现 `core/model_config.hpp`、`core/imodel_builder.hpp`、`core/model_registry.hpp`。
7. 实现 `core/builder.hpp`：统一 EngineBuilder 入口。
8. 接入 SentencePiece 源码嵌入（`third_party/sentencepiece`），禁用非必要功能并写 README.md。
9. 搭建 `mini_trt_llm/CMakeLists.txt`，编译为 static library + tests（`BUILD_TESTS` 默认 OFF）。
10. 根 `CMakeLists.txt` 新增 `add_subdirectory(mini_trt_llm)`，旧模块暂时保留。

**产出**：可编译的 `libmini_trt_llm.a`，基础单元测试通过，通用化骨架可用。

### Phase 1：Plugin 基础（1.5 周）

1. 实现 `PluginRegistry` 与 `IPluginV3` 插件基类封装。
2. 实现并测试 `RoPEPlugin`。
3. 实现并测试 `RMSNormPlugin`。
4. 实现并测试 `PagedAttentionPlugin`（支持 MHA/GQA/MQA）。
5. 实现 `Sampler` CUDA kernels（Greedy + Top-K + Top-P）。

**产出**：`test_rmsnorm_plugin.cpp`、`test_rope_plugin.cpp`、`test_paged_attention_plugin.cpp`、`test_sampler.cpp` 用例通过。

### Phase 2：GPT-2 原生构建（方案 A）（2 周）

1. 实现 `GPT2ModelBuilder` 并注册到 `ModelRegistry`。
2. 实现 Safetensors 权重加载器与 JSON weight_map 映射。
3. 实现 `NativeBuilder::BuildFromConfig`：读取 JSON config + Safetensors → TRT network。
4. 实现 `LLMRunner`：Prefill engine + Decode engine 双引擎切换。
5. 集成 SentencePiece tokenizer（多模态接口预留）。
6. 与 `ref_output.bin` 对比精度并调优。

**产出**：`test_gpt2_native.cpp` 用例输出与 PyTorch FP32 对齐。

### Phase 3：GPT-2 ONNX + Plugin 构建（方案 B）（1.5 周）

> **实测修正（2026-09-25）**：`1_gpt2_onnx/gpt2.onnx` **不含 RMSNorm、也不含 RoPE**
> （实测用 LayerNormalization + 学习式位置编码），因此本节"替换 RoPE / RMSNorm 子图"
> **在这张图上没有替换对象**；Phase 3 实际交付的是"ONNX 路径可用 + 与方案 A 数值对齐 +
> 子图识别断言"。推导与实测见 `docs/phase3_development_plan.md` §0.2 与 §P3 执行结果。

1. 实现 `OnnxBuilder` 与 subgraph replacer。
2. 对现有 `1_gpt2_onnx/gpt2.onnx` 进行 RoPE/RMSNorm/Attention 子图替换。
3. 验证替换后 engine 与原生构建输出一致。

**产出**：`test_gpt2_onnx_plugin.cpp` 用例通过，与方案 A 输出对齐。

### Phase 4：ResNet18 替换（1 周）

1. 实现 `ResNet18ModelBuilder` 并注册到 `ModelRegistry`。
2. 从 Safetensors 加载 ResNet18 权重。
3. 支持 FP32/FP16（INT8 延后到后续迭代）。
4. 实现 `CVRunner`。
5. 与 `0_resnet18_onnx` 输出对比。

**产出**：`test_resnet18.cpp` 用例通过。

### Phase 5：清理旧模块（0.5 周）—— **❌ 已永久取消（2026-09-26 用户决定）**

> 本节保留为历史设计记录。**Phase 5 不做**：旧模块由作者本人按需处理，Agent 不要删除或移动
> （它们同时是 ONNX / INT8 用例的本地产物来源）。


1. 从根 `CMakeLists.txt` 移除 `0_resnet18_onnx` 与 `1_gpt2_onnx`。
2. 删除旧目录（或移动到 `archive/` 备份）。
3. 更新 `README.md` 与构建说明。

**产出**：项目仅保留 `mini_trt_llm` 作为核心模块。

---

## 4. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 自定义 Plugin 在 TRT 10.x `IPluginV3` 下兼容性 | 高 | 严格按 TRT 10.15.1 `IPluginV3` 接口实现，并做版本宏隔离 |
| 原生构建 GPT-2 与 PyTorch 输出不一致 | 高 | 分算子逐层对比，定位差异（RMSNorm epsilon、RoPE base、attention scale 等） |
| SentencePiece 与 GPT-2 Python tokenizer 不对齐 | 中 | GPT-2 实际用 BPE，SentencePiece 行为可能不同，需用 Python tokenizer 生成用例 diff，必要时实现 BPE tokenizer |
| BF16 权重转换开销 | 低 | 构建时一次性转换，运行时无额外开销 |
| sm_75 下 FP16 无 Tensor Core 加速有限 | 中 | INT8 延后到后续迭代，作为性能优化重点 |
| PagedAttention 在 Turing 上无 FlashAttention 优化 | 中 | 先实现功能正确版本，后续再 hand-tune kernel |

---

## 5. 已确认决策清单

- [x] 架构预留 Encoder-Decoder，Phase 0 只做 CNN + Decoder-only。
- [x] 注意力支持 MHA / GQA / MQA / Sliding Window。
- [x] 配置文件格式 JSON。
- [x] LLM 区分 Prefill / Decode OptimizationProfile；CV Phase 0 只动态 batch。
- [x] TensorRT 10.15.1，Plugin API 用 `IPluginV3`。
- [x] SentencePiece v0.2.0 源码嵌入，禁用非必要功能并写 README.md。
- [x] Safetensors 解析用 `syoyo/safetensors-cpp`。
- [x] 转换工具 `mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`。
- [x] 权重接受 F32 / F16 / BF16。
- [x] 权重映射优先 JSON 配置，C++ 硬编码兜底。
- [x] 需要 Dummy model 用于回归测试。
- [x] `BUILD_TESTS=OFF`，`BUILD_EXAMPLES=OFF`，CI 显式开启。
- [x] 错误处理：构造/初始化抛异常，运行时返回 bool。
- [x] 日志：新增 `MINI_TRT_LOG_INFO/WARN/ERROR` 宏。
- [x] Phase 0 内存池用简单 `cudaMalloc/cudaFree` 封装。

---

*文档版本：v1.0*  
*详见 `docs/phase0_development_plan.md` 与 `docs/future_iterations.md`。*
