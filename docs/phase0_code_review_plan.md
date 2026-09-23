# Phase 0 代码 Review 方案

> 目标：系统梳理 Phase 0 全部代码，便于逐模块审查骨架质量、风险点与可测试性。

---

## 1. Review 范围

**包含在 Phase 0 内**
- 构建系统（根 + `mini_trt_llm/CMakeLists.txt` + `tests/CMakeLists.txt`）
- Utils 基础设施
- Safetensors 加载器
- Tokenizer 抽象 + SentencePiece 嵌入
- Engine / EngineBuilder / ModelRegistry / IModelBuilder / ModelConfig 框架
- 单元测试
- 第三方依赖接入

**不包含（后续 Phase）**
- 任何真实模型构建器（`GPT2ModelBuilder`、`ResNet18ModelBuilder`）
- 任何自定义 Plugin（RMSNorm、RoPE、PagedAttention）
- Sampler CUDA Kernel
- Runner 的 `Prefill → Decode` 自回归循环
- INT8 校准、内存池化、Profiler 高级功能

---

## 2. Review 顺序与建议

推荐按“底层 → 框架 → 入口 → 测试”的顺序 review，避免一开始就陷入高层接口细节。

```
1. 构建系统
2. Utils（cuda_check / logger / timer / memory_pool / io / json）
3. SafetensorsLoader
4. Tokenizer 抽象 + SentencePieceTokenizer
5. Precision / WeightLoader
6. ModelConfig / ModelRegistry / IModelBuilder
7. EngineBuilder
8. Engine
9. 单元测试
10. 工具脚本（hf_to_mini_trt_llm.py）
```

---

## 3. 逐模块 Review 清单

### 3.1 构建系统

**文件**
- `CMakeLists.txt`
- `mini_trt_llm/CMakeLists.txt`
- `mini_trt_llm/tests/CMakeLists.txt`

**Review 重点**
- [ ] `sm_75` 是否被正确硬编码，是否暴露可配置开关。
- [ ] TensorRT / CUDA 库查找路径是否合理，是否兼容用户环境。
- [ ] 第三方依赖（sentencepiece / safetensors-cpp / googletest）的选项覆盖是否完整。
- [ ] `BUILD_TESTS=OFF` 默认是否合理。
- [ ] static library 的 include / link 是否干净。

**关键问题**
- 是否应该把 `CMAKE_CUDA_ARCHITECTURES` 做成可配置而不是写死 75？
- 如果用户 TensorRT 不在 `/usr/lib/x86_64-linux-gnu`，构建是否会优雅失败？

---

### 3.2 Utils

#### 3.2.1 `cuda_check.hpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/utils/cuda_check.hpp`

**Review 重点**
- [ ] `CUDA_CHECK` / `NVINFER_CHECK` 宏行为是否一致。
- [ ] 错误信息是否包含文件、行号、错误码描述。
- [ ] 失败时是否抛异常或终止，是否符合项目整体错误处理策略。

#### 3.2.2 `logger.hpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/utils/logger.hpp`

**Review 重点**
- [ ] `MINI_TRT_LOG_*` 宏是否线程安全。
- [ ] 日志级别是否可运行时切换。
- [ ] 是否所有宏都带 `do { ... } while (0)` 或等价结构。

#### 3.2.3 `timer.hpp` / `timer.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/utils/timer.hpp`
- `mini_trt_llm/src/utils/timer.cpp`

**Review 重点**
- [ ] `CudaTimer` 是否遵循 RAII。
- [ ] `Start` / `Stop` 是否在同一个 stream 上配对。
- [ ] 返回值单位是否清晰（ms / s）。

#### 3.2.4 `memory_pool.hpp` / `memory_pool.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/utils/memory_pool.hpp`
- `mini_trt_llm/src/utils/memory_pool.cpp`

**Review 重点**
- [ ] `DeviceBuffer` / `PinnedBuffer` 是否正确使用 RAII（构造、析构、移动）。
- [ ] `Resize` 是否在缩小时释放并重分配，是否有浪费。
- [ ] 是否处理了 `cudaMalloc` / `cudaMallocHost` 失败。
- [ ] 命名是否有误导（当前只是 buffer，不是真正的 pool）。

**关键问题**
- 文件名叫 `memory_pool`，但实现只是 RAII buffer。是否应该改名（如 `device_buffer.hpp`），还是后续真的要实现池化？

#### 3.2.5 `io.hpp` / `io.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/utils/io.hpp`
- `mini_trt_llm/src/utils/io.cpp`

**Review 重点**
- [ ] `ReadFile` / `WriteFile` 是否支持大文件。
- [ ] 异常信息是否清晰。
- [ ] `WriteFile` 是否自动创建父目录。

#### 3.2.6 `json.hpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/utils/json.hpp`

**Review 重点**
- [ ] 解析能力边界在哪里（嵌套深度、数组、字符串转义、数字类型）。
- [ ] 错误处理是否健全（未匹配的括号、非法字符）。
- [ ] 类型转换 API（`AsInt` / `AsString` / `AsObject`）是否在类型错误时给出明确反馈。

**已知问题**
- 自研 JSON 解析器能力有限，复杂配置可能解析失败。这是记录在案的 trade-off。

---

### 3.3 SafetensorsLoader

**文件**
- `mini_trt_llm/include/mini_trt_llm/utils/safetensors_loader.hpp`
- `mini_trt_llm/src/utils/safetensors_loader.cpp`

**Review 重点**
- [ ] dtype 映射是否完整，未支持类型是否有 fallback 并记录警告。
- [ ] `GetRawData` 在 mmap 与非 mmap 模式下地址计算是否正确。
- [ ] `GetConvertedData` 的转换结果生命周期是否明确（内部缓冲区会被下一次调用覆盖）。
- [ ] BF16 → FP16 目前是简单截断，是否应加 TODO 或 round-nearest。
- [ ] `GetTensorNames()` 返回空的问题是否已明确标注。

**关键问题**
- `conversion_buffer_` 是 `DeviceBuffer`，但 BF16 转换在 CPU 上做？确认 `DeviceBuffer` 实际分配的是 host 还是 device 内存。
- `SafetensorsToTrtDtype` 对 `kFLOAT64` 返回 `kFLOAT` 并警告，这是否合理？

---

### 3.4 Tokenizer

#### 3.4.1 `base_tokenizer.hpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/tokenizer/base_tokenizer.hpp`

**Review 重点**
- [ ] 接口是否足够通用，能覆盖 BPE / SentencePiece / Tiktoken。
- [ ] `Encode` / `Decode` 的返回值和异常语义是否清晰。

#### 3.4.2 `sentencepiece_tokenizer.hpp` / `.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/tokenizer/sentencepiece_tokenizer.hpp`
- `mini_trt_llm/src/tokenizer/sentencepiece_tokenizer.cpp`

**Review 重点**
- [ ] SentencePiece 初始化和模型加载是否健壮。
- [ ] `Encode` / `Decode` 是否正确处理 special tokens。
- [ ] 与 GPT-2 BPE 不对齐的问题是否在注释或文档中说明。

---

### 3.5 Precision / WeightLoader

#### 3.5.1 `precision.hpp` / `precision.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/core/precision.hpp`
- `mini_trt_llm/src/core/precision.cpp`

**Review 重点**
- [ ] `Precision` enum 是否与 TensorRT `DataType` 正确映射。
- [ ] 是否预留了 INT8，但当前未实现校准。

#### 3.5.2 `weight_loader.hpp` / `weight_loader.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/core/weight_loader.hpp`
- `mini_trt_llm/src/core/weight_loader.cpp`

**Review 重点**
- [ ] `Load()` 是否硬编码 `model.safetensors`，是否应支持分片或自定义名。
- [ ] `weight_map` 映射方向是否正确（JSON 里是 TRT name → source key，代码里是否一致）。
- [ ] `GetWeight` 兜底逻辑（直接用 trt_name 当 source key）是否会导致意外行为。
- [ ] 转换后的权重指针生命周期是否足够让 builder 使用。

---

### 3.6 ModelConfig / ModelRegistry / IModelBuilder

#### 3.6.1 `model_config.hpp` / `model_config.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/core/model_config.hpp`
- `mini_trt_llm/src/core/model_config.cpp`

**Review 重点**
- [ ] `ModelConfig::Load()` 是否只读取 `config.json`，路径拼接是否正确。
- [ ] `hyper_params` 和 `weight_map` 作为 `JsonValue` 是否足够表达未来需求。
- [ ] 缺少字段时是否给出明确错误。

#### 3.6.2 `model_registry.hpp` / `.cpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/core/model_registry.hpp`
- `mini_trt_llm/src/core/model_registry.cpp`

**Review 重点**
- [ ] 注册表是否线程安全（当前不是，是否可接受）。
- [ ] `Get()` 未找到时返回 `nullptr`，调用方是否都检查。

#### 3.6.3 `imodel_builder.hpp`

**文件**
- `mini_trt_llm/include/mini_trt_llm/core/imodel_builder.hpp`

**Review 重点**
- [ ] 接口是否足够表达 GPT-2 / ResNet18 的需求。
- [ ] `Build` 返回 `bool` 而不是异常，错误信息如何传递。

---

### 3.7 EngineBuilder

**文件**
- `mini_trt_llm/include/mini_trt_llm/core/builder.hpp`
- `mini_trt_llm/src/core/builder.cpp`

**Review 重点**
- [ ] `BuildFromConfig` 流程是否完整：config → registry → weights → build → serialize → save。
- [ ] `BuildFromOnnx` 是否解析 ONNX 并构建 engine，plugin_ops 占位是否明确。
- [ ] `SetupBuilder` 中 workspace、FP16 flag 设置是否合理。
- [ ] `SerializeAndSave` 对 `nullptr` 的处理是否安全。
- [ ] 动态 shape 的 optimization profile 还未实现，是否有 TODO 标注。

**关键问题**
- `BuildFromConfig` 在 `builder_impl->Build()` 失败时，是否已创建的网络资源会泄漏？当前用 `unique_ptr` 管理，应无问题。
- `BuildFromOnnx` 的 `plugin_ops` 参数当前被 `(void)plugin_ops;` 忽略，Phase 3 再实现，这一点是否清晰。

---

### 3.8 Engine

**文件**
- `mini_trt_llm/include/mini_trt_llm/core/engine.hpp`
- `mini_trt_llm/src/core/engine.cpp`

**Review 重点**
- [ ] `Engine` 构造函数是否正确反序列化 engine 文件。
- [ ] `SetInputShape` / `SetTensorAddress` / `Enqueue` 是否正确调用 TRT 10.x API。
- [ ] `Benchmark` 是否合理处理 warmup / run / stream 同步。
- [ ] throughput 计算假设 `batch_size * seq_len` 是否合理。

**关键问题**
- `Benchmark` 内部创建临时 stream 并销毁，是否应在调用方传入 stream 更灵活？
- latency 单位在 `BenchResult` 中是否统一为 ms？

---

### 3.9 单元测试

**文件**
- `mini_trt_llm/tests/*.cpp`
- `mini_trt_llm/tests/CMakeLists.txt`

**Review 重点**
- [ ] 测试覆盖是否完整：utils / loader / config / registry / engine 都有测试。
- [ ] CUDA 相关测试在沙箱环境失败，是否应加 `GTEST_SKIP` 保护。
- [ ] 测试是否过于简单（如 `SafetensorsLoaderTest` 只测文件不存在）。
- [ ] 是否所有测试都能独立运行，不互相污染。

---

### 3.10 工具脚本

**文件**
- `mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`

**Review 重点**
- [ ] 支持的格式是否符合预期。
- [ ] 分片模型加载是否正确，不会重复加载或遗漏。
- [ ] 错误处理是否友好。
- [ ] 是否有 `--help` 和使用示例。

---

## 4. 通用 Review Checklist

对每一文件都可以问：

- [ ] **正确性**：代码是否按预期工作，边界条件是否处理。
- [ ] **错误处理**：所有 CUDA / TensorRT / 文件 / 解析 API 调用是否检查返回值或抛异常。
- [ ] **资源管理**：RAII 是否到位，是否有内存泄漏、double free、悬空指针。
- [ ] **性能**：是否有不必要的拷贝、同步、host-device 往返。
- [ ] **可读性**：命名是否符合 Google C++ Style，函数是否单一职责。
- [ ] **注释**：是否写“Why 而非 What”，魔数和非直观逻辑是否解释清楚。
- [ ] **可测试性**：是否容易写单元测试，是否依赖全局状态。
- [ ] **可扩展性**：接口是否为 Phase 1/2/3/4 预留了扩展点。

---

## 5. 已知风险点（Review 时重点关注）

| 风险 | 位置 | 当前状态 |
|---|---|---|
| 自研 JSON 解析器能力有限 | `utils/json.hpp` | 已记录，后续换 nlohmann/json |
| `SafetensorsLoader::GetTensorNames()` 返回空 | `utils/safetensors_loader.cpp` | 已记录，库接口限制 |
| BF16 → FP16 简单截断 | `utils/safetensors_loader.cpp` | 已记录，可优化为 round-nearest |
| `memory_pool` 命名误导 | `utils/memory_pool.hpp` | 需 review 决定是否改名 |
| SentencePiece 与 GPT-2 BPE 不对齐 | `tokenizer/sentencepiece_tokenizer.cpp` | 已记录，后续加 BpeTokenizer |
| 无真实 builder，框架未闭环 | `core/builder.cpp` | Phase 2/4 填补 |
| CUDA 测试在沙箱失败 | `tests/*.cpp` | 需真机验证 |

---

## 6. Review 输出建议

Review 完成后，建议按模块输出：
1. **OK**：无需修改。
2. **Minor**：可改可不改，记录即可。
3. **Major**：影响后续 Phase，需在本次或下次迭代修复。
4. **Blocker**：必须修复，否则影响编译/运行/安全。

并更新 `docs/PROGRESS.md` 的“已知问题与坑”和“下一步计划”。
