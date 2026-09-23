# mini_trt_llm 项目进度交接文档

> 最后更新：2026-09-24  
> 当前阶段：Phase 1 进行中（IPluginV3 基类 + RMSNormPlugin 已完成并接入注册表，下一步 RoPEPlugin）

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
  - 当前环境 TensorRT 为 **10.15.1**，V3 是推荐接口。
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

- `mini_trt_llm/CMakeLists.txt`：C++17 + CUDA C++17、`sm_75`、static library、第三方依赖接入。
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
- 覆盖度局限：`test_safetensors_loader.cpp` 目前只有「文件不存在返回 false」一个负向用例，真实文件解析、ONNX→Engine、DummyBuilder 端到端尚未实现（见 `docs/phase0_model_loading_test_plan.md`）。

### 3.6 工具与文档

- `mini_trt_llm/tools/convert/hf_to_mini_trt_llm.py`：HF checkpoint → `config.json + model.safetensors` 转换脚本（真实实现的唯一落点）。
- `requirements.txt`：转换工具依赖（已移到项目根目录）。
- `docs/mini_trt_llm_design.md`：v1.0 设计文档。
- `docs/phase0_development_plan.md`：Phase 0 开发计划。
- `docs/phase0_code_review_plan.md`：Phase 0 代码 review 方案（review 由用户本人执行，尚未完成）。
- `docs/phase0_model_loading_test_plan.md`：Phase 0 模型加载测试方案（T1–T3 尚未实施）。
- `docs/phase1_development_plan.md`：Phase 1 开发方案 + 关键决策确认清单（含合并后的 15 项决策）。
- `docs/TROUBLESHOOTING.md`：问题排查记录（现象 / 定位路径 / 根因 / 修复 / 回归防护）。
- `docs/future_iterations.md`：后续迭代计划。
- `mini_trt_llm/third_party/sentencepiece/README.md`：记录禁用功能。
- `docs/PROGRESS.md`：本交接文档（已按 `progress-summary` skill 更新）。

> 历史文档：`docs/phase1_pending_confirmations.md` 的内容已全部合并进 `docs/phase1_development_plan.md` §10，原文件已删除。

### 3.7 第三方依赖

- `mini_trt_llm/third_party/sentencepiece`：v0.2.0，源码嵌入。
- `mini_trt_llm/third_party/safetensors-cpp`：main 分支，源码嵌入。
- `mini_trt_llm/third_party/googletest`：release 版本，源码嵌入。

### 3.8 注释规范修复

- 已按 `cpp-comment-style` skill 检查并修复 Phase 0 代码注释：
  - 公共 API（`precision.hpp`、`timer.hpp`、`json.hpp`）补齐类级/函数级注释。
  - 纯英文注释改为中文（`builder.hpp`、`memory_pool.cpp`）。
  - 魔数与边界逻辑补充"Why"注释（`builder.hpp` 默认值、`engine.cpp` 统计、`safetensors_loader.cpp` BF16 位运算）。

### 3.9 Phase 1 插件进展

| 文件 | 说明 |
|---|---|
| `include/mini_trt_llm/plugins/rmsnorm_kernel.hpp` | `LaunchRmsNorm` kernel 启动接口（单独暴露以便 L1 层直测 kernel，绕过 engine 构建） |
| `include/mini_trt_llm/plugins/rmsnorm_plugin.hpp` | `RmsNormPlugin` + `RmsNormPluginCreator` |
| `src/plugins/rmsnorm_plugin.cu` | CUDA kernel + Plugin 实现 + Creator + `REGISTER_TENSORRT_PLUGIN` 静态注册 |
| `tests/test_rmsnorm_plugin.cpp` | 14 个用例：10 个 host 侧、4 个 GPU 门控（无 GPU 时 `GTEST_SKIP`） |
| `tests/test_rmsnorm_integration.cpp` | L2 集成用例：真实 TRT network → engine 序列化/反序列化 → 推理对比 CPU 参考；外加注册表登记用例 |
| `tests/test_gpu_guard.hpp` | 共享的 `HasCudaDevice()` 门控，供所有 GPU 用例复用 |
| `tests/test_reference.hpp` | 共享的 CPU 参考实现与精度判定（相对误差 + 小值绝对误差 Guardrail） |

- 实现要点：一行一个 block，warp shuffle 归约；FP32 走 `float4`、FP16 走 8×half（16B）向量化，`hidden_size` 不能整除时回退标量 kernel。
- 已确认决策的落地：不带 bias；weight 作为第二输入；`eps` / `hidden_size` 按 float / int32 序列化；serialize↔deserialize 往返已单测覆盖。
- `getWorkspaceSize()` 返回 0，`enqueue` 内无任何分配；失败以错误码返回而非抛异常（`enqueue` 为 `noexcept`）。
- 注册表接入：`PluginRegistry::RegisterAllPlugins()` 由 stub 改为登记 `GetRmsNormPluginCreator()`，
  与 `REGISTER_TENSORRT_PLUGIN` 的 TRT 全局注册并存（前者给本框架按名查找，后者给 engine 反序列化）。
- 构建改动：`mini_trt_llm/CMakeLists.txt` 的源文件 glob 增加 `src/*.cu`，否则 nvcc 产物不会进静态库。
- 验证状态：沙箱内 `ctest` 32 个用例 **0 失败**（11 个 GPU 用例自动跳过）；用户在 WSL2 真机上跑
  `--gtest_filter='RmsNorm*'` **全部通过**，即 kernel 数值与 engine 集成均已验证。

---

## 4. 进行中 / 未完成的部分

### 4.1 Phase 1：Plugin 基础（进行中）

- 决策状态：15 项待确认问题已全部关闭，无遗留阻塞项（详见 `docs/phase1_development_plan.md` §10）。
- ✅ 完善 `IPluginV3` 基类，补齐 TRT 10.x 接口。
- ✅ 实现 `RMSNormPlugin` + 单元测试（host 侧用例已通过；GPU 用例待真机验证）。
- ✅ `RMSNormPlugin` 接入 `PluginRegistry`，并补 L2 集成测试（真实 TRT network → engine 序列化/反序列化 → 推理）。
- ⬜ 实现 `RoPEPlugin`。
- ⬜ 实现 `PagedAttentionPlugin`（Decoding 阶段 GQA/MHA）。
- ⬜ 实现 Sampler CUDA Kernels（Greedy / Top-K / Top-P）。
- ⬜ 为其余 Plugin / Kernel 写单元测试。

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

### 5.7 Phase 0 utils 测试缺少 GPU 门控（已修复）

- **问题**：`CudaCheckTest`、`DeviceBufferTest`、`PinnedBufferTest`、`CudaTimerTest` 共 6 个用例直接调用 CUDA API 且未做环境判断，在无 GPU 环境下抛 `cudaErrorInsufficientDriver` 而失败，而不是跳过。
- **影响**：无 GPU 的 CI / 沙箱里 `ctest` 永远不绿，真实回归信号被固定噪声淹没（用户真机上这 6 个用例是过的）。
- **Workaround**：已抽出共享的 `tests/test_gpu_guard.hpp`（`test_support::HasCudaDevice()`）给这批用例加门控；
  例外的 `CudaCheckTest.InvalidDeviceThrows` 刻意不门控，因为它验证的是 `CUDA_CHECK` 的失败路径。
- **排查过程**：见 `docs/TROUBLESHOOTING.md` #1。

### 5.8 `supportsFormatCombination` 越界读取导致 engine 构建失败（已修复）

- **问题**：`supportsFormatCombination` 扫描了 `inOut[pos+1..]`——TensorRT 未初始化的位置，导致所有格式组合都被判为不支持，`buildSerializedNetwork` 报 `could not find any supported formats consistent with input/output data types`。
- **影响**：Plugin 无法构建成 engine；host 侧单测完全无感，只有真机能复现。
- **Workaround**：已改为只与 `inOut[0]` 比对；新增回归用例 `RmsNormPluginTest.IgnoresInvalidDescriptorsAfterPos`。
- **排查过程**：见 `docs/TROUBLESHOOTING.md` #2。

---

## 6. 下一步计划

**Phase 1 第三步：实现 RoPEPlugin + 单元测试**

理由：RMSNorm 已把 `IPluginV3` 模板、序列化、注册、测试流程跑通，RoPE 与之相互独立、可复用同一套脚手架，适合作为下一个算子。RoPE 需要处理 `position_ids` 与部分旋转（`rotary_dim < head_size`），复杂度高于 RMSNorm。

Phase 1 关键技术决策（已确认，完整清单 D1–D5 + Q1–Q15 见 `docs/phase1_development_plan.md` §10），要点：

- PagedAttention 先做 Decoding 阶段 GQA/MHA，Prefill 后续迭代。
- Sampler 先用 **CUB**（非 Thrust）保证正确性，手写高性能 kernel 后续迭代。
- Plugin/算子层与 PyTorch/HF 参考输出逐元素对比；Generation 层对比分布/Logits 统计量。
- FP16 相对误差 < 1e-3，配合绝对误差 Guardrail（不要求逐 bit 一致）。
- RMSNorm 不支持 bias；weight 作为 Plugin 输入，依赖 TRT 常量折叠。
- PagedAttention `block_size` 强制显式配置，无默认值；`scale` 作为属性。
- Sampler k/p 使用 per-batch tensor，支持连续批处理；随机数用 host seed + device Philox。
- Plugin 标量属性统一用 float 存储；Top-K/Top-P 按 `vocab_size <= 128K` 设计。
- 参考输出全部使用固定 seed；Phase 1 需完整实现并测试 serialize/deserialize。
- 所有 GPU 验证/测试任务需经人工确认后执行。

---

## 7. 重要环境信息

| 项目 | 版本 / 说明 |
|---|---|
| 操作系统 | Ubuntu on WSL2 |
| GPU | NVIDIA GeForce GTX 1660 Ti Mobile（Turing） |
| Compute Capability | `sm_75` |
| TensorRT | 10.15.1（版本宏 v101501，`libnvinfer.so.10.15.1`） |
| CUDA Toolkit | 12.6.85 |
| GCC | 13.3.0 |
| C++ 标准 | C++17 |
| CMake | 3.28.3（要求 >= 3.18） |
| SentencePiece | v0.2.0 |
| safetensors-cpp | main（commit af90b6c） |
| GoogleTest | release（源码嵌入） |
| Python 依赖 | `transformers>=4.40`, `safetensors>=0.4`, `torch>=2.0` |

---

*本文档用于新会话快速接手项目，不记录对话过程，只保留决策与状态。*
