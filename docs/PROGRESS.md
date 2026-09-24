# mini_trt_llm 项目进度交接文档

> 最后更新：2026-09-24  
> 当前阶段：**Phase 1.5 已完成**（全流程测试基建与收尾）；下一步 Phase 2（GPT-2 原生构建）

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

### 2.12 Phase 1 算子层的接口约定（实现时确立，勿回改）

以下五条是实现 Phase 1 Plugin 时定下的约定，其中前三条与 `phase1_development_plan.md` 的早期表述**不一致**，
是经用户确认后的有意偏离，后续会话不要按 plan 原文"修正"回去：

- **形状优先、属性兜底**：`configurePlugin` 一律以输入形状为权威来源推导 `num_heads` / `head_size` 等；
  只有对应维度是动态轴（`<= 0`）时才回退到属性值。因此针对属性的校验用例必须构造动态轴。
- **RoPE 采用 half-split 约定**（`out[j] = x[j]cos - x[j+h]sin`），而非 plan §4.2 字面写的"相邻两维配对"。
  **为什么**：plan 同节指定的测试参考是 HuggingFace `apply_rotary_pos_emb`，而它是 half-split；
  已用 `scripts/ref_rope.py` 交叉验证，最大差异 `0.000e+00`。GPT-NeoX 风格留作后续属性开关。
- **RoPE 的 head 配置不做序列化属性**：只序列化 `rotary_dim` 与 `base`，`num_heads` / `num_kv_heads` /
  `head_size` 全部从形状推导。**为什么**：做成属性会多一份可能与形状失配的状态。
- **PagedAttention 不含 `max_context_len` 输入**：plan §4.3 列了该标量输入，但 online softmax 按
  per-batch `context_lens[b]` 循环，全局上界用不到，保留会成为死参数。
- **采样器只有设备侧 API**：`sampler/sampler_common.hpp` 的 `Launch*Sampler` 是唯一接口，token 结果直接写回
  device。**为什么**：Phase 0 遗留的「标量 k/p + host `std::vector` 输出」与 Q7（per-batch tensor）及
  「Decode 全程驻留显存」冲突，已删除（见 §5.9）。实现文件落在 `src/sampler/sampler_kernels.cu`,
  与 plan §3 写的 `src/plugins/sampler/*.cu` 不同——采样器不是 TensorRT Plugin，放在 `src/sampler/` 更贴合实际分层。

另有一条 kernel 实现纪律（来自 `docs/TROUBLESHOOTING.md` #4）：
**输出与输入分离的 kernel，只要存在"部分写入"路径，就必须显式处理未覆盖区间**。

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
- `scripts/ref_rope.py` / `scripts/ref_sampler.py`：参考语义自检脚本（RoPE 与 HuggingFace 交叉验证、采样器截断语义与理论概率）。
- `requirements.txt`：转换工具依赖（已移到项目根目录）。
- `docs/mini_trt_llm_design.md`：v1.0 设计文档。
- `docs/phase0_development_plan.md`：Phase 0 开发计划。
- `docs/phase0_code_review_plan.md`：Phase 0 代码 review 方案（review 由用户本人执行，尚未完成）。
- `docs/phase0_model_loading_test_plan.md`：Phase 0 模型加载测试方案（T1–T3 尚未实施）。
- `docs/phase1_development_plan.md`：Phase 1 开发方案 + 关键决策确认清单（含合并后的 15 项决策）。
- `docs/phase1_test_plan.md`：Phase 1 全流程测试计划（模型加载 → builder 分发 → Plugin 挂载 → engine 构建/反序列化 → 推理 → 采样），含前置改造清单（G1/G2/G3）与实施顺序。
- `docs/phase1_5_development_plan.md`：Phase 1.5 开发计划（P1.5-0 ~ P1.5-7 的任务、依赖、验收）。
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

### 3.9 Phase 1 插件与采样器

| 文件 | 说明 |
|---|---|
| `include/mini_trt_llm/plugins/rmsnorm_{kernel,plugin}.hpp` + `src/plugins/rmsnorm_plugin.cu` | RMSNorm Plugin（一行一 block，FP32 `float4` / FP16 8×half 向量化，不能整除时回退标量） |
| `include/mini_trt_llm/plugins/rope_{kernel,plugin}.hpp` + `src/plugins/rope_plugin.cu` | RoPE Plugin（half-split 约定，双输入双输出，`position_ids` 作为输入） |
| `include/mini_trt_llm/plugins/paged_attention_{kernel,plugin}.hpp` + `src/plugins/paged_attention_plugin.cu` | PagedAttention Plugin（仅 Decoding，GQA/MHA，online softmax 单趟扫描） |
| `include/mini_trt_llm/sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu` | Greedy / Top-K / Top-P 采样器（设备侧 API，内联 Philox 随机源） |
| `include/mini_trt_llm/utils/cuda_dtype.cuh` / `cuda_reduce.cuh` | 多 kernel 共用的 dtype 转换与 block 归约 |
| `tests/test_{rmsnorm_plugin,rmsnorm_integration,rope_plugin,paged_attention_plugin,sampler}.cpp` | 算子单测 + L2 集成测试 |
| `tests/test_gpu_guard.hpp` / `test_reference.hpp` | 共享的 GPU 门控与 CPU 参考实现 |

- 已确认决策的落地：RMSNorm 不带 bias、weight 作第二输入；RoPE `rotary_dim` 默认 `head_size` 且可配置、`position_ids` 作输入；PagedAttention `block_size` 强制显式配置、`scale` 为属性（默认 `1/sqrt(head_size)`）；采样器 k/p 为 per-batch tensor、随机源为 host seed + device Philox。
- 统一的接口纪律：所有 Plugin 的 `getWorkspaceSize()` 返回 0、`enqueue` 内零分配、失败以错误码返回（`enqueue` 是 `noexcept`）；`supportsFormatCombination` 一律只读 `inOut[0..pos]`。
- 注册表接入：`PluginRegistry::RegisterAllPlugins()` 登记三个 Plugin 的 creator，与 `REGISTER_TENSORRT_PLUGIN` 的 TRT 全局注册并存（前者给本框架按名查找，后者给 engine 反序列化）。
- 构建改动：`mini_trt_llm/CMakeLists.txt` 的源文件 glob 增加 `src/*.cu`，否则 nvcc 产物不会进静态库。
- 验证状态：沙箱内 `ctest` 67 个用例 **0 失败**（26 个 GPU 用例自动跳过）；
  用户在 WSL2 真机上跑 `--gtest_filter='RoPE*:PagedAttention*:Sampler*'` **全部通过**，
  加上此前已通过的 `RmsNorm*`，**Phase 1 全部算子（kernel 数值 + engine 集成）均已在真机验证**。
  过程中修复了一个只在部分旋转下暴露的 RoPE 缺陷（见 `docs/TROUBLESHOOTING.md` #4）。
  ⚠️ 该数字是 Phase 1 收尾时的快照，**当前总数见 §3.10**。
- 参考数据脚本：`scripts/ref_rope.py`（与 HuggingFace `apply_rotary_pos_emb` 交叉验证，最大差异 0.0）、
  `scripts/ref_sampler.py`（Top-K/Top-P 截断语义与理论概率）。

### 3.10 Phase 1.5：全流程测试基建与收尾

| 文件 | 说明 |
|---|---|
| `tests/e2e_safetensors_writer.{hpp,cpp}` | 测试用 Safetensors 写入 helper（B/F16/BF16），让端到端夹具自包含、不依赖 Python |
| `tests/e2e_fixture.{hpp,cpp}` | 临时模型目录（`mkdtemp` + 析构清理），组装 `config.json` + `model.safetensors` |
| `tests/test_e2e_error_paths.cpp` | E4：5 条错误路径用例（3 条 host 侧可进 CI，2 条需 GPU） |
| `tests/test_e2e_single_op.cpp` | E1：4 条单算子闭环（RMSNorm / RoPE / PagedAttention / Sampler），重量经真实 `WeightLoader` 取用 |
| `tests/test_e2e_mini_decoder.cpp` | E2：**完整链路** `RMSNorm → QKV → Slice/Reshape → RoPE → PagedAttention → RMSNorm → LM Head → Top-K Sampler`，四权重均以 BF16 存储，与独立 CPU 参考对比 |
| `tests/test_fp16_paths.cpp` | FP16 覆盖缺口：RoPE（含 GQA + batch>1 + 部分旋转）、PagedAttention、Greedy Sampler |
| `tests/test_reference_helpers.cpp` | 参考实现自身的 meta-test（7 条 host 用例），含 `RopeIsBatchAware` 回归 |
| `tests/test_e2e_dynamic_shape.cpp` | E3：Prefill 变长序列 / Decode 变 batch / 超范围拒绝 |
| `mini_trt_llm/tests/test_safetensors_loader.cpp` | P1.5-0 的 8 条回归用例（dtype 组合矩阵、指针不别名、零拷贝） |

- **产品代码修复**（详见 `docs/TROUBLESHOOTING.md` #5 / #6 / #7）：
  - `SafetensorsLoader` 转换路径的 3 个缺陷：BF16 被误判为同类型而返回原始数据、往设备内存从主机侧写、
    BF16→FP16 位截断在数值上错误；转换缓存改为按 key 隔离。
  - `IModelBuilder::Build(const WeightLoader&)` 原先取不到权重（`GetWeight` 非 const），
    读取路径改为 const + `mutable` 缓存。
  - `EngineBuilder::BuildFromConfig` 调整校验顺序：纯数据校验前置于 `createInferBuilder`。
- **动态 shape 打通**：`AddCvOptimizationProfile` / `AddLlmOptimizationProfiles` 由"只声明未定义"
  变为已实现并接入；新增 `Engine::SetOptimizationProfile` 支持多 profile 切换。
- **真机复验发现并修复的缺陷**：反序列化后的 Plugin 从未执行 `configurePlugin`，
  而 RoPE / PagedAttention 的 head 配置是靠构建期从形状推导的、不作为序列化属性，
  导致 `onShapeChange` 把正常形状误判为"改了 head 配置"，所有 RoPE 端到端用例在 enqueue 失败。
  已改为以运行期形状为准刷新，并补 4 条 **host 侧**回归用例（详见 `docs/TROUBLESHOOTING.md` #8）。
- **第二次真机复验发现并修复的缺陷**：参考实现 `ReferenceRoPE` 漏了 batch 维度，导致
  `Fp16PathTest` 在 batch>1 时误判为 kernel 出错。已把参考实现收敛为唯一来源、加 batch 参数
  与入口断言，并补 7 条 host 侧 meta-test（详见 `docs/TROUBLESHOOTING.md` #9）。
- 验证状态：沙箱内 `ctest` **104 个用例 0 失败**（40 个 GPU 用例自动跳过，64 个 host 用例实际执行）；
  参考实现这一层由 `test_reference_helpers.cpp` 在 CI 内自证。
- **真机验证**：E1 / E2 / E3 与 E4 的 2 条 GPU 用例、FP16 覆盖用例（共 5 条）**全部通过**。
  Phase 1.5 完成闭环——进入 Phase 2 建模前所需的组件（插件 / 采样器 / 权重加载 / 动态 shape /
  端到端骨架）均已实现并在真机验证。

---

## 4. 进行中 / 未完成的部分

### 4.1 Phase 1：Plugin 基础（已完成）

- 决策状态：15 项待确认问题已全部关闭，无遗留阻塞项（详见 `docs/phase1_development_plan.md` §10）。
- ✅ 完善 `IPluginV3` 基类，补齐 TRT 10.x 接口。
- ✅ 实现 `RMSNormPlugin` + 单元测试（GPU 用例已在真机验证通过）。
- ✅ `RMSNormPlugin` 接入 `PluginRegistry`，并补 L2 集成测试（真实 TRT network → engine 序列化/反序列化 → 推理）。
- ✅ 实现 `RoPEPlugin` + 单元测试。
- ✅ 实现 `PagedAttentionPlugin`（Decoding 阶段 GQA/MHA）+ 单元测试。
- ✅ 实现 Sampler CUDA Kernels（Greedy / Top-K / Top-P）+ 单元测试。
- ✅ 全部 GPU 用例已在用户 WSL2 真机验证通过。

Phase 1 明确不在本次范围内、留待后续的项：

- PagedAttention 的 **Prefill 阶段**（query 序列长度 > 1）——需要因果 mask 与分块，与 Decode 路径 kernel 结构差异大。
- 采样器分布级对比（D3）已落地为两件事：C++ 侧的统计检验（词频收敛到解析 softmax 概率），以及
  `scripts/ref_sampler.py` 打印的 HF 风格截断语义；尚未做的是把 Python 输出固化成数据文件供测试载入。
- Sampler 的手写高性能 kernel（Phase 1 用 CUB 分段排序保证正确性）。

### 4.2 Phase 1.5：全流程测试基建与收尾（已完成）

- ✅ P1.5-0：修复 `SafetensorsLoader` 转换路径的 3 个缺陷（详见 `docs/TROUBLESHOOTING.md` #5）。
- ✅ P1.5-1 / P1.5-2：Safetensors 写入 helper；E4 错误路径用例。
- ✅ P1.5-3：E1 单算子闭环（4 条）。
- ⚠️ P1.5-4：E2 **缩减完成**——交付了多权重 BF16 路径的数值验证，
  完整 `RMSNorm → QKV → RoPE → PagedAttention → LM Head` 链路与 `ref_mini_block.py` 有意留后
  （理由见 `docs/phase1_5_development_plan.md` §0.1）。
- ✅ P1.5-5 / P1.5-6：Optimization profile 实现；E3 动态 shape 测试。
- ✅ P1.5-7：文档归位 4/4，含把 `phase0_development_plan.md` 的验收判据改写为可执行形式，
  并补上 Phase 0 遗漏的「Optimization profile 能力可用」一条。

真机复验：E1 / E2 / E3 与 E4 的 2 条用例**已通过**。

### 4.3 Phase 2：GPT-2 原生构建（未开始）

- 实现 `GPT2ModelBuilder`。
- 从 Safetensors 加载 GPT-2 权重并构建 TRT network。
- 实现 `LLMRunner` 的 `Prefill → Decode` 自回归循环。
- 与 `1_gpt2_onnx/ref_output.bin` 对比精度。

### 4.4 Phase 3：GPT-2 ONNX + Plugin（未开始）

- 实现 `OnnxBuilder` + subgraph replacer。
- 对 `1_gpt2_onnx/gpt2.onnx` 替换 RoPE / RMSNorm / Attention 子图。
- 验证与方案 A 输出一致。

### 4.5 Phase 4：ResNet18 替换（未开始）

- 实现 `ResNet18ModelBuilder`。
- 支持 FP32 / FP16（INT8 延后）。
- 实现 `CVRunner`。

### 4.6 Phase 5：清理旧模块（未开始）

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

### 5.9 采样器曾存在两套 API（已清理）

- **问题（已解决）**：Phase 0 留下的 `sampler/{greedy,topk,topp}_sampler.{hpp,cpp}` 声明的是「标量 k/p + host `std::vector` 输出」的接口，与已确认的 Q7（per-batch tensor）和「Decode 全程驻留显存」冲突，函数体仍是 `throw not implemented`。
- **解决**：经用户授权删除 6 个旧桩文件，采样器 API 统一收敛到 `sampler/sampler_common.hpp` + `src/sampler/sampler_kernels.cu`。

### 5.10 GPU 用例在沙箱内无法执行（已确认为环境限制，非缺陷）

- **问题**：所有 kernel 数值与 engine 集成用例都需要 GPU，沙箱内只会 `GTEST_SKIP`。
- **影响**：Agent 侧的结论上限是「编译通过 + 契约自洽 + host 侧逻辑正确」。
- **现状**：Phase 1 全部 GPU 用例已由用户在真机验证通过；这是**流程约束而非遗留缺陷**，
  后续 Phase 每完成一个算子，都需要同样走一遍真机验证。

---

## 6. 下一步计划

**Phase 2：GPT-2 原生构建（方案 A）**

理由：Phase 1 的算子层（RMSNorm / RoPE / PagedAttention / Sampler）已齐备，可以开始搭真实模型。注意 GPT-2 使用**学习式位置编码**而非 RoPE，因此 Phase 2 先不依赖 RoPE Plugin；RoPE 是给后续 LLaMA 类模型准备的。

Phase 1.5 为其扫清的前置：dynamic shape 的 optimization profile 已打通（Phase 2 的 Prefill/Decode
双引擎直接依赖它）；多权重取用路径已有端到端验证，且修掉了会让 GPT-2 静默建错的转换缓冲区缺陷。

**Phase 2 开工前建议先做**：

1. 真机复验 Phase 1.5 的 E1/E2/E3（命令见 `docs/phase1_5_development_plan.md`）。
2. 补齐 E2 的完整链路与 `ref_mini_block.py`——它是 GPT-2 子图的缩微版，可当作 GPT-2 接线的脚手架。

> 开发流程提醒：Phase 1 的经验是「沙箱内 host 用例通过不代表真机没问题」——
> RoPE 的部分旋转缺陷只有真机 kernel 执行才暴露。后续每个 Phase 结束都应走一遍真机验证。

> 本节只保留下一步入口。Phase 1 的 15 项已确认决策见 `docs/phase1_development_plan.md` §10，
> Phase 1 确立的接口约定见本文档 §2.12，均已归档，不再在此重复。

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
| Python 依赖 | `torch 2.5.1+cu121`（已装，`scripts/` 下的参考脚本直接可跑）；`transformers>=4.40`, `safetensors>=0.4` |

---

*本文档用于新会话快速接手项目，不记录对话过程，只保留决策与状态。*
