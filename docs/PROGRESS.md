# mini_trt_llm 项目进度交接文档

> 最后更新：2026-09-23  
> 当前阶段：Phase 0 完成 + 注释规范修复完成，准备进入 Phase 1

---

## 1. 项目目标

构建一个面向 NVIDIA Turing / `sm_75` 的极简多模态 TensorRT 推理框架，以 `mini_trt_llm` 统一替换现有 `0_resnet18_onnx` 与 `1_gpt2_onnx` 两个独立 ONNX 加载模块。

---

## 2. 当前架构与关键决策

### 2.1 定位：多模态推理框架，而非单一 LLM 引擎

- **为什么**：旧项目同时包含 CV（ResNet18）和 LLM（GPT-2）两个示例。若只替换 LLM，ResNet18 会 orphaned；若各做各的，代码重复。
- **结果**：`mini_trt_llm` 统一承载 CV + LLM，未来预留 Encoder-Decoder / ViT / 多模态接口。

### 2.2 模型构建：配置驱动 + 模型注册表

- **决策**：用 JSON `config.json` 描述模型结构，通过 `ModelRegistry` 分发到具体 `IModelBuilder`（`GPT2ModelBuilder`、`ResNet18ModelBuilder` 等）。
- **为什么选 A 不选 B**：
  - **A（配置驱动 + 注册表）**：每新增模型只需写一个 builder 子类并注册，不修改 `EngineBuilder` 类。
  - **B（硬编码 `BuildGPT2` / `BuildResNet18`）**：扩展性差，每加模型都要改核心 builder。

### 2.3 模型来源：双模式支持

- **方案 A（原生构建）**：从 Safetensors + JSON config 用 TRT Network API 手搭网络。
- **方案 B（ONNX + Plugin）**：解析 ONNX，按子图模式替换为自定义 Plugin。
- **为什么同时支持**：方案 A 是完全替换 ONNX 加载的最终目标；方案 B 是过渡路径，便于复用现有 ONNX 模型并逐步替换算子。
- **命名冲突规避**：所有 Plugin 层名加 `mini_trt_llm_` 前缀，Creator 名用 `MiniTrtLlmXxxPluginCreator`，避免与 TRT 原生层重名。

### 2.4 权重格式：Safetensors

- **为什么选 Safetensors 不选 PyTorch `.bin`**：
  - Safetensors 加载安全（无 pickle 反序列化风险）。
  - 支持 mmap，Header 与张量元数据分离，便于 C++ 解析。
  - 已成 HuggingFace 标准导出格式。
- **解析库**：`syoyo/safetensors-cpp`（轻量，不拉整个官方库）。

### 2.5 Tokenizer：SentencePiece 源码嵌入 + 多模态抽象

- **为什么选 SentencePiece 先接入**：首批模型 GPT-2 虽用 BPE，但 SentencePiece 是更通用的子词分词器，LLaMA/T5 等后续模型直接可用。
- **为什么源码嵌入**：避免依赖系统包版本不一致，构建自包含。
- **抽象接口**：`BaseTokenizer` 预留，未来可扩展 BPE / Tiktoken / CLIP tokenizer。

### 2.6 Plugin API：IPluginV3（TRT 10.x）

- **为什么选 IPluginV3 不选 IPluginV2DynamicExt**：
  - 当前环境 TensorRT 为 **10.1.0**，V3 是推荐接口。
  - V2 虽兼容性好，但在 TRT 10.x 下属于 legacy，长期维护成本高。

### 2.7 日志：独立业务日志宏

- **决策**：新增 `MINI_TRT_LOG_INFO/WARN/ERROR`，不直接复用 `common/logger.hpp`（TRT `ILogger` 接口）。
- **为什么**：TRT logger 绑定 TensorRT 回调接口，打业务日志语义别扭；独立宏可灵活桥接或独立输出。

### 2.8 测试框架：GoogleTest 源码嵌入

- **为什么选源码嵌入不选系统包**：系统未安装 `libgtest-dev`，且网络受限无法 `FetchContent`；源码嵌入最稳定。

### 2.9 构建选项

- `BUILD_TESTS=OFF`（默认），`BUILD_EXAMPLES=OFF`（默认）。
- **为什么默认 OFF**：日常编译更快；CI 显式 `-DBUILD_TESTS=ON`。

### 2.10 INT8 与性能优化延后

- **INT8 校准**：ResNet18 / LLM 的 INT8 支持放到后续迭代。
- **内存池**：Phase 0 仅 `cudaMalloc/cudaFree` RAII 封装，池化实现后续迭代。
- **为什么延后**：Phase 0 优先搭骨架和可编译性，过早做 INT8/池化会拖慢基础验证。

### 2.11 代码注释规范

- **决策**：启用 `cpp-comment-style` skill，要求注释写"Why 而非 What"，必须覆盖魔数、workaround、非直观逻辑和公共 API；禁止复述代码、逐行翻译和遗留调试注释。
- **为什么**：项目代码会长期维护并交给多轮会话接力，清晰的"Why"注释比代码本身更能降低接手成本。

---

## 3. 已完成的部分

### 3.1 目录与构建

- `mini_trt_llm/CMakeLists.txt`：C++17 + CUDA17、`sm_75`、static library、第三方依赖接入。
- `mini_trt_llm/tests/CMakeLists.txt`：GoogleTest 集成。
- 根 `CMakeLists.txt`：加入 `mini_trt_llm`，旧模块已注释掉。

### 3.2 Utils 基础设施

| 文件 | 说明 |
|---|---|
| `include/mini_trt_llm/utils/cuda_check.hpp` | `CUDA_CHECK`、`NVINFER_CHECK` 宏 |
| `include/mini_trt_llm/utils/logger.hpp` | `MINI_TRT_LOG_*` 业务日志宏 |
| `include/mini_trt_llm/utils/timer.hpp` + `src/utils/timer.cpp` | CUDA Event 计时器（已补注释） |
| `include/mini_trt_llm/utils/memory_pool.hpp` + `src/utils/memory_pool.cpp` | `DeviceBuffer` / `PinnedBuffer` RAII 封装（已补注释） |
| `include/mini_trt_llm/utils/io.hpp` + `src/utils/io.cpp` | 文件读写 + JSON 加载 |
| `include/mini_trt_llm/utils/json.hpp` | 自研极简 JSON 解析器（已补注释） |
| `include/mini_trt_llm/utils/safetensors_loader.hpp` + `src/utils/safetensors_loader.cpp` | Safetensors 加载 + BF16→FP32/FP16 转换（已补注释） |

### 3.3 Core 通用化骨架

| 文件 | 说明 |
|---|---|
| `include/mini_trt_llm/core/precision.hpp` + `src/core/precision.cpp` | 精度枚举与 TRT 映射（已补注释） |
| `include/mini_trt_llm/core/builder.hpp` + `src/core/builder.cpp` | 统一 EngineBuilder（已补默认值注释） |
| `include/mini_trt_llm/core/engine.hpp` + `src/core/engine.cpp` | Engine 封装 + Benchmark（已补统计注释） |
| `include/mini_trt_llm/core/imodel_builder.hpp` | 模型构建器抽象接口 |
| `include/mini_trt_llm/core/model_config.hpp` + `src/core/model_config.cpp` | JSON 配置加载 |
| `include/mini_trt_llm/core/model_registry.hpp` + `src/core/model_registry.cpp` | 模型注册表 |
| `include/mini_trt_llm/core/weight_loader.hpp` + `src/core/weight_loader.cpp` | Safetensors 权重加载 + weight_map 映射 |
| `include/mini_trt_llm/core/llm_runner.hpp/.cpp` | Phase 0 仅接口声明 |
| `include/mini_trt_llm/core/cv_runner.hpp/.cpp` | Phase 0 仅接口声明 |

### 3.4 其他模块占位

- `kv_cache/`、`plugins/`、`sampler/`、`tokenizer/` 头文件与空实现已就位，供 Phase 1/2/4 填充。

### 3.5 测试

- `mini_trt_llm/tests/test_*.cpp`：覆盖 cuda_check、logger、timer、memory_pool、io、model_config、model_registry、safetensors_loader、engine。
- 当前状态：用户本地 **100% tests passed**。

### 3.6 工具与文档

- `mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`：占位脚本。
- `requirements.txt`：转换工具依赖（已移到项目根目录）。
- `docs/mini_trt_llm_design.md`：v1.0 设计文档。
- `docs/phase0_development_plan.md`：Phase 0 开发计划。
- `docs/future_iterations.md`：后续迭代计划。
- `mini_trt_llm/third_party/sentencepiece/README.md`：记录禁用功能。
- `docs/PROGRESS.md`：本交接文档（已按 `progress-summary` skill 更新）。

### 3.7 第三方依赖

- `mini_trt_llm/third_party/sentencepiece`：v0.2.0，源码嵌入。
- `mini_trt_llm/third_party/safetensors-cpp`：main 分支，源码嵌入。
- `mini_trt_llm/third_party/googletest`：release 版本，源码嵌入。

### 3.8 注释规范修复

- 已按 `cpp-comment-style` skill 检查并修复 Phase 0 代码注释：
  - 公共 API（`precision.hpp`、`timer.hpp`、`json.hpp`）补齐类级/函数级注释。
  - 纯英文注释改为中文（`builder.hpp`、`memory_pool.cpp`）。
  - 魔数与边界逻辑补充"Why"注释（`builder.hpp` 默认值、`engine.cpp` 统计、`safetensors_loader.cpp` BF16 位运算）。

---

## 4. 进行中 / 未完成的部分

### 4.1 Phase 1：Plugin 基础（未开始）

- 完善 `IPluginV3` 基类，补齐 TRT 10.x 接口。
- 实现 `RMSNormPlugin`。
- 实现 `RoPEPlugin`。
- 实现 `PagedAttentionPlugin`（基础版，先 Greedy）。
- 实现 Sampler CUDA Kernels（Greedy / Top-K / Top-P）。
- 为每个 Plugin 写单元测试。

### 4.2 Phase 2：GPT-2 原生构建（未开始）

- 实现 `GPT2ModelBuilder`。
- 从 Safetensors 加载 GPT-2 权重并构建 TRT network。
- 实现 `LLMRunner` 的 `Prefill → Decode` 自回归循环。
- 与 `1_gpt2_onnx/ref_output.bin` 对比精度。

### 4.3 Phase 3：GPT-2 ONNX + Plugin（未开始）

- 实现 `OnnxBuilder` + subgraph replacer。
- 对 `1_gpt2_onnx/gpt2.onnx` 替换 RoPE / RMSNorm / Attention 子图。
- 验证与方案 A 输出一致。

### 4.4 Phase 4：ResNet18 替换（未开始）

- 实现 `ResNet18ModelBuilder`。
- 支持 FP32 / FP16（INT8 延后）。
- 实现 `CVRunner`。

### 4.5 Phase 5：清理旧模块（未开始）

- 从根 `CMakeLists.txt` 彻底删除旧模块 `add_subdirectory`（当前仅注释）。
- 删除 `0_resnet18_onnx/` 与 `1_gpt2_onnx/`（或移入 `archive/`）。
- 更新 `README.md`。

---

## 5. 已知问题与坑

### 5.1 自研 JSON 解析器能力有限

- **问题**：Phase 0 用 `include/mini_trt_llm/utils/json.hpp` 自研极简解析器，仅支持基础类型和简单嵌套。
- **影响**：复杂配置可能解析失败。
- **Workaround**：后续迭代替换为 `nlohmann/json` 单头文件。

### 5.2 `SafetensorsLoader::GetTensorNames()` 返回空

- **问题**：`syoyo/safetensors-cpp` 的 `ordered_dict` 不暴露 key 列表遍历接口。
- **影响**：无法枚举所有张量名。
- **Workaround**：当前只按名查询权重，不影响功能；后续可换库或自研解析。

### 5.3 SentencePiece 与 GPT-2 BPE 可能不对齐

- **问题**：GPT-2 原生用 BPE，SentencePiece 行为可能与之有差异。
- **影响**：Tokenizer 结果可能与 Python `transformers` 不完全一致。
- **Workaround**：后续实现 `BpeTokenizer : BaseTokenizer`，与 Python tokenizer 逐 case diff。

### 5.4 BF16 → FP16 简单截断

- **问题**：`safetensors_loader.cpp` 中 BF16→FP16 直接截断尾数，非 round-nearest。
- **影响**：精度损失略大。
- **Workaround**：后续优化为 round-nearest 转换；或优先用 FP32 权重。

### 5.5 沙箱环境无法访问 GPU

- **问题**：Agent 运行环境的 `nvidia-smi` 报 `GPU access blocked by the operating system`，CUDA API 无法初始化。
- **影响**：Agent 无法本地验证 CUDA 相关测试。
- **Workaround**：CUDA 测试必须在用户真实 WSL2 环境手动运行验证。

### 5.6 `.gitmodules` 曾出现重复条目

- **问题**：根目录与 `mini_trt_llm/third_party/` 下曾同时存在 submodule 条目。
- **状态**：已清理，当前仅保留 `mini_trt_llm/third_party/sentencepiece` 与 `mini_trt_llm/third_party/safetensors-cpp`。
- **注意**：新增第三方依赖时避免在根目录再建 submodule。

---

## 6. 下一步计划

**Phase 1 第一步：实现 RMSNormPlugin + RoPEPlugin + 单元测试**

理由：这两个插件逻辑相对独立、计算简单，适合先把 `IPluginV3` 模板、序列化、注册、测试流程跑通。之后再做更复杂的 `PagedAttentionPlugin` 和 Sampler。

需要人工确认：
- 是否从 RMSNorm + RoPE 开始？
- 每个插件是否都需要与 PyTorch 参考输出逐 bit 对比？
- PagedAttention 是否先只支持 MHA，GQA/MQA 后续再加？

---

## 7. 重要环境信息

| 项目 | 版本 / 说明 |
|---|---|
| 操作系统 | Ubuntu on WSL2 |
| GPU | NVIDIA GeForce GTX 1660 Ti Mobile（Turing） |
| Compute Capability | `sm_75` |
| TensorRT | 10.1.0（v101501） |
| CUDA Toolkit | 12.6.85 |
| GCC | 13.3.0 |
| C++ 标准 | C++17 |
| CMake | >= 3.18 |
| SentencePiece | v0.2.0 |
| safetensors-cpp | main（commit af90b6c） |
| GoogleTest | release（源码嵌入） |
| Python 依赖 | `transformers>=4.40`, `safetensors>=0.4`, `torch>=2.0` |

---

*本文档用于新会话快速接手项目，不记录对话过程，只保留决策与状态。*
