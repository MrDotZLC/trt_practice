# Phase 0 开发计划：基础设施与通用化骨架

> 周期：2 周  
> 目标：搭建 `mini_trt_llm` 可编译基础框架，完成通用化骨架（ModelConfig / ModelRegistry / IModelBuilder / WeightLoader / Engine / Builder），接入 SentencePiece 与 Safetensors 解析，实现基础单元测试。

> 注：本文档的 API 草图中出现 `nlohmann::json`，为设计意图表述。Phase 0 实际实现采用自研极简解析器 `utils/json.hpp`（`JsonValue`），原因是该阶段网络受限、无法引入外部依赖；后续迭代替换为 `nlohmann/json` 单头文件。

---

## 1. 任务总览

| 任务 ID | 任务名 | 预估工期 | 依赖 |
|---|---|---|---|
| P0-1 | 目录与 CMake 骨架 | 2 天 | 无 |
| P0-2 | 第三方依赖接入（SentencePiece + safetensors-cpp） | 2 天 | P0-1 |
| P0-3 | Utils 基础设施 | 2 天 | P0-1 |
| P0-4 | Core 通用化骨架 | 3 天 | P0-3 |
| P0-5 | 基础单元测试 | 2 天 | P0-2, P0-3, P0-4 |
| P0-6 | 根 CMake 接入与验证 | 1 天 | P0-1 |

---

## 2. 任务详细说明

### P0-1：目录与 CMake 骨架

**目标**：建立目录结构，配置 `mini_trt_llm/CMakeLists.txt`，支持 CUDA + C++17，编译为 static library。

**新增/修改文件**：
- `mini_trt_llm/CMakeLists.txt`
- `mini_trt_llm/include/mini_trt_llm/` 目录下各子目录（已创建）
- `mini_trt_llm/src/` 目录下各子目录（已创建）
- `mini_trt_llm/tests/CMakeLists.txt`

**CMake 关键配置**：
```cmake
project(mini_trt_llm LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

# TRT / CUDA discovery
find_path(TRT_INCLUDE_DIR NvInfer.h ...)
find_path(CUDA_INCLUDE_DIR cuda_runtime.h ...)
find_library(...)

# SentencePiece
add_subdirectory(third_party/sentencepiece)
# 禁用测试/Python/Perl绑定等

# Safetensors
# syoyo/safetensors-cpp 为 header-only 或 small lib，视情况 add_subdirectory / include

add_library(mini_trt_llm STATIC ...)
target_link_libraries(mini_trt_llm ... sentencepiece ...)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

**验收标准**：
- `cmake -B build -DCMAKE_BUILD_TYPE=Release` 成功配置。
- `cmake --build build --target mini_trt_llm` 成功编译空库（或仅含桩代码）。

---

### P0-2：第三方依赖接入

#### 2.1 SentencePiece 源码嵌入

**目标**：将 SentencePiece v0.2.0 源码放入 `mini_trt_llm/third_party/sentencepiece/`，通过 CMake 编译，禁用不需要的功能。

**新增/修改文件**：
- `mini_trt_llm/third_party/sentencepiece/`（源码，后续由用户或脚本放入）
- `mini_trt_llm/third_party/sentencepiece/README.md`（记录禁用功能）
- `mini_trt_llm/CMakeLists.txt`（接入 `add_subdirectory`）

**需禁用功能**（在 CMake 中关闭）：
- `SPM_BUILD_TESTS=OFF`
- `SPM_ENABLE_SHARED=OFF`（优先静态链接）
- `SPM_ENABLE_TCMALLOC=OFF`
- `SPM_ENABLE_NFKC_COMPILE=OFF`
- Python/Perl/Ruby bindings 关闭

**README.md 内容模板**：
```markdown
# SentencePiece 使用说明

版本：v0.2.0

## 禁用功能

为减少编译体积，以下功能已关闭：
- 单元测试
- Shared library
- tcmalloc
- Python/Perl/Ruby bindings
- NFKC compile-time normalization

仅使用 C++ static library 的核心 encode/decode 功能。
```

#### 2.2 safetensors-cpp 接入

**目标**：引入 `syoyo/safetensors-cpp`，封装为 `utils/safetensors_loader.hpp`。

**新增/修改文件**：
- `mini_trt_llm/third_party/safetensors-cpp/`（源码）
- `mini_trt_llm/include/mini_trt_llm/utils/safetensors_loader.hpp`
- `mini_trt_llm/src/utils/safetensors_loader.cpp`

**验收标准**：
- SentencePiece 可编译通过，tokenizer 接口可正常链接。
- Safetensors 可读取 `.safetensors` 文件头与张量元数据。

---

### P0-3：Utils 基础设施

**目标**：实现所有 Utils 模块，为 Core 和 Plugin 提供基础能力。

#### 3.1 `utils/cuda_check.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/utils/cuda_check.hpp`

**功能**：
```cpp
#define CUDA_CHECK(call) ...
#define NVINFER_CHECK(call) ...
#define CUDA_CHECK_LAST() ...
```
- 包装 `cudaError_t` 与 `nvinfer1::IBuilder`/`IExecutionContext` 返回值。
- 异常信息包含文件、行号、错误码。

#### 3.2 `utils/logger.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/utils/logger.hpp`

**功能**：
```cpp
#define MINI_TRT_LOG_INFO(...)  ...
#define MINI_TRT_LOG_WARN(...)  ...
#define MINI_TRT_LOG_ERROR(...) ...
```
- 内部可桥接到 `common/logger.hpp` 的 TRT logger，也可独立输出到 `std::cerr`。
- 支持格式化字符串（可用 `fmt` 或 `std::format` 若 C++20；C++17 用 `std::ostringstream` 封装）。

#### 3.3 `utils/timer.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/utils/timer.hpp`
- `mini_trt_llm/src/utils/timer.cpp`

**功能**：
```cpp
class CudaTimer {
 public:
  void Start(cudaStream_t stream);
  float Stop(cudaStream_t stream);  // ms
};
```

#### 3.4 `utils/memory_pool.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/utils/memory_pool.hpp`
- `mini_trt_llm/src/utils/memory_pool.cpp`

**功能**：
```cpp
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();
  void* data();
  size_t size() const;
  void Resize(size_t bytes);
};

class PinnedBuffer { ... };
```
- Phase 0 仅做 `cudaMalloc/cudaFree` 封装，RAII 管理。

#### 3.5 `utils/io.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/utils/io.hpp`
- `mini_trt_llm/src/utils/io.cpp`

**功能**：
- `ReadFile(const std::string& path) -> std::vector<char>`
- `WriteFile(const std::string& path, const void* data, size_t bytes)`
- `LoadJson(const std::string& path) -> nlohmann::json`

**验收标准**：
- 所有 Utils 模块有对应单元测试并通过。

---

### P0-4：Core 通用化骨架

#### 4.1 `core/precision.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/precision.hpp`

**功能**：
```cpp
enum class Precision { FP32, FP16, INT8 };
const char* PrecisionStr(Precision p);
nvinfer1::DataType ToTrtDataType(Precision p);
```

#### 4.2 `core/engine.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/engine.hpp`
- `mini_trt_llm/src/core/engine.cpp`

**功能**：
- 封装 `IRuntime / ICudaEngine / IExecutionContext`。
- 支持动态 shape、tensor address 绑定、`enqueueV3`、stream 同步。
- Benchmark 接口。

#### 4.3 `core/model_config.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/model_config.hpp`

**功能**：
```cpp
struct ModelConfig {
  std::string model_type;
  std::string architecture;
  nlohmann::json hyper_params;
  nlohmann::json weight_map;
};

ModelConfig LoadModelConfig(const std::string& model_dir);
```

#### 4.4 `core/imodel_builder.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/imodel_builder.hpp`

**功能**：
```cpp
class IModelBuilder {
 public:
  virtual ~IModelBuilder() = default;
  virtual std::string Name() const = 0;
  virtual bool Build(nvinfer1::INetworkDefinition* network,
                     const WeightLoader& weights,
                     const ModelConfig& config) = 0;
};
```

#### 4.5 `core/model_registry.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/model_registry.hpp`
- `mini_trt_llm/src/core/model_registry.cpp`

**功能**：
```cpp
class ModelRegistry {
 public:
  void Register(const std::string& name, std::shared_ptr<IModelBuilder> builder);
  std::shared_ptr<IModelBuilder> Get(const std::string& name) const;
  std::vector<std::string> List() const;
};
```

#### 4.6 `core/weight_loader.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/weight_loader.hpp`
- `mini_trt_llm/src/core/weight_loader.cpp`

**功能**：
```cpp
class WeightLoader {
 public:
  bool Load(const std::string& safetensors_path);
  void SetWeightMap(const nlohmann::json& map);

  // 内部按需转换 BF16/F32/F16
  const void* GetWeight(const std::string& trt_name,
                        nvinfer1::DataType target_type,
                        size_t* bytes);
};
```

#### 4.7 `core/builder.hpp`

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/builder.hpp`
- `mini_trt_llm/src/core/builder.cpp`

**功能**：
```cpp
class EngineBuilder {
 public:
  struct Config { ... };  // 含 Prefill/Decode / CV dynamic batch profile
  explicit EngineBuilder(Logger& logger, const Config& config);

  bool BuildFromConfig(const std::string& model_dir,
                       const std::string& engine_path);
  bool BuildFromOnnx(const std::string& onnx_path,
                     const std::string& engine_path,
                     const std::vector<std::string>& plugin_ops);

 private:
  std::unique_ptr<ModelRegistry> registry_;
};
```

#### 4.8 `core/runner` 桩代码

**新增文件**：
- `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp`
- `mini_trt_llm/include/mini_trt_llm/core/cv_runner.hpp`
- 仅声明类，Phase 1/2 再实现。

**验收标准**：
- `EngineBuilder::BuildFromConfig` 能读取 JSON config、调用 `ModelRegistry`、构建空 network（可序列化为 engine）。
- `Engine` 能反序列化并执行一次空推理（绑定输入输出地址）。

---

### P0-5：基础单元测试

**目标**：为 Utils 和 Core 骨架写单元测试。

**新增文件**：
- `mini_trt_llm/tests/CMakeLists.txt`
- `mini_trt_llm/tests/test_cuda_check.cpp`
- `mini_trt_llm/tests/test_logger.cpp`
- `mini_trt_llm/tests/test_timer.cpp`
- `mini_trt_llm/tests/test_memory_pool.cpp`
- `mini_trt_llm/tests/test_io.cpp`
- `mini_trt_llm/tests/test_safetensors_loader.cpp`
- `mini_trt_llm/tests/test_model_config.cpp`
- `mini_trt_llm/tests/test_model_registry.cpp`
- `mini_trt_llm/tests/test_engine.cpp`

**测试框架**：GoogleTest（若未安装，Phase 0 先检查可用性；如不可用，用 Catch2 或简单 assert 兜底）。

**验收标准**：
- `cmake -B build -DBUILD_TESTS=ON` 配置成功。
- `ctest --output-on-failure` 全部通过。

---

### P0-6：根 CMake 接入与验证

**目标**：把 `mini_trt_llm` 接入根项目，旧模块暂时保留。

**修改文件**：
- `/home/mr_zlc/trt_practice/CMakeLists.txt`

**修改内容**：
```cmake
add_subdirectory(mini_trt_llm)
# 0_resnet18_onnx 与 1_gpt2_onnx 暂时保留
```

**验收标准**：
- 根目录 `cmake -B build` 成功。
- `cmake --build build --target mini_trt_llm` 成功。
- 旧模块 `trt_resnet18`、`trt_gpt2` 仍可构建。

---

## 3. 文件清单（Phase 0）

```text
mini_trt_llm/
├── CMakeLists.txt
├── include/mini_trt_llm/
│   ├── core/
│   │   ├── precision.hpp
│   │   ├── engine.hpp
│   │   ├── model_config.hpp
│   │   ├── imodel_builder.hpp
│   │   ├── model_registry.hpp
│   │   ├── weight_loader.hpp
│   │   ├── builder.hpp
│   │   ├── llm_runner.hpp
│   │   └── cv_runner.hpp
│   ├── plugins/
│   │   ├── plugin_registry.hpp
│   │   └── iplugin_v3_base.hpp
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
│   │   ├── precision.cpp
│   │   ├── engine.cpp
│   │   ├── model_config.cpp
│   │   ├── model_registry.cpp
│   │   ├── weight_loader.cpp
│   │   └── builder.cpp
│   ├── utils/
│   │   ├── timer.cpp
│   │   ├── memory_pool.cpp
│   │   ├── io.cpp
│   │   └── safetensors_loader.cpp
│   └── tokenizer/
│       └── sentencepiece_tokenizer.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_cuda_check.cpp
│   ├── test_logger.cpp
│   ├── test_timer.cpp
│   ├── test_memory_pool.cpp
│   ├── test_io.cpp
│   ├── test_safetensors_loader.cpp
│   ├── test_model_config.cpp
│   ├── test_model_registry.cpp
│   └── test_engine.cpp
├── third_party/
│   ├── sentencepiece/        # 源码 + README.md
│   └── safetensors-cpp/      # 源码
└── tools/
    └── convert/              # Phase 0 可先放占位脚本
```

---

## 4. 验收标准（Phase 0 整体）

> **判据写法约定（自本节起沿用）**：Phase 0 最初的判据以"文件存在 / 命令能跑通"为准，
> 结果是 `AddCvOptimizationProfile`、`AddLlmOptimizationProfiles` 这类**只声明未定义**的方法
> 也被算作通过——而 `BuildFromConfig` 的动态 shape 能力实际上完全不可用，
> 一直遗留到 Phase 1.5 才补齐（详见 `docs/TROUBLESHOOTING.md` #5 的教训）。
>
> 因此本节改写为**可执行判据**：每条都写明用什么命令或用例验证、以及判定通过的标准。
> 后续 Phase 的验收标准一律采用这种写法，不再接受"代码/文件已存在"作为通过依据。

| # | 判据 | 验证方式（可执行） | 结果 |
|---|---|---|---|
| 1 | 根目录 CMake 配置成功 | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75` | ✅ |
| 2 | `mini_trt_llm` 目标编译成功 | `cmake --build build --target mini_trt_llm` 返回 0 | ✅ |
| 3 | 单元测试全部通过 | `ctest --test-dir build`，判据为 **0 failed**；GPU 用例以 `GTEST_SKIP` 跳过不计为失败 | ✅ 当前 92 用例 |
| 4 | SentencePiece 只编静态库且禁用非必要功能 | 产物中**无** `libsentencepiece.so`（只有 `.a`）；`grep '^SPM_' build/CMakeCache.txt` 中 `BUILD_TEST` / `ENABLE_SHARED` / `ENABLE_TCMALLOC` / `ENABLE_NFKC_COMPILE` / `BUILD_PYTHON_BINDINGS` 均为 OFF | ✅ |
| 5 | Safetensors 能读取 header 与张量元数据 | `--gtest_filter='SafetensorsLoaderTest.*'`：须覆盖真实文件解析、dtype 与 shape 校验（不满足于"文件不存在返回 false"这类负向用例） | ✅ 9 用例 |
| 6 | `EngineBuilder` 骨架可扩展：新增模型无需修改 Builder 类 | 测试侧独立注册 4 个 `IModelBuilder` 并成功构建 engine，且 `EngineBuilder` 源码未被改动 | ✅ `E2eSingleOpTest` |
| 7 | **Optimization profile 能力可用**（Phase 0 遗漏、Phase 1.5 补齐的新增判据） | 两个 `Add*OptimizationProfile` 方法**有定义**且被 `BuildFromConfig` 按 architecture 调用；`--gtest_filter='E2eDynamicShapeTest.*'` 通过 | ✅ Phase 1.5 补齐 |
| 8 | 旧模块 `trt_resnet18` / `trt_gpt2` 仍可构建 | **已失效**：根 `CMakeLists.txt` 自 Phase 1 起已注释掉这两个 `add_subdirectory`，当前无法通过根构建验证 | ⚠️ 见下 |

### 4.1 关于 #7 的说明

这一条是补写出来的，不是新增需求：`EngineBuilder::Config` 里 CV / Prefill / Decode 三组
`min/opt/max_*` 字段从 Phase 0 就在，设计文档也把「Prefill / Decode 双 OptimizationProfile」
列为已确认前提，但两个 profile 方法在 Phase 0 只写了声明、没有定义、也没有任何调用点。
Phase 1.5 补齐后（2 个定义 + `BuildFromConfig` 中的调用），本条才真正成立。

### 4.2 关于 #8 的说明

Phase 0 的迁移策略是「保守并行，先共存后移除」，所以当时要求旧模块仍可构建。
Phase 1 起旧模块已从根 `CMakeLists.txt` 注释掉，该判据**已不再适用**——它的目的（确认新框架
不破坏旧能力）已经完成，而旧模块本就计划在 Phase 5 移除。两条出路，需要在 Phase 5 一并决定：

- 若希望在移除前一直保有「旧模块可构建」的保证，应恢复 `add_subdirectory`；
- 否则按原计划在 Phase 5 直接归档/删除，本条判据随之作废。

当前默认按后者处理（不恢复），因为它会拖慢日常构建。

> 补充：真机相关的验证（CUDA / TensorRT 执行）无法在 Agent 沙箱内复现，这一类判据统一由用户在
> WSL2 环境执行，并在 `docs/PROGRESS.md` 的「真机验证记录」中留痕；沙箱内只跑 host 侧用例。

---

## 5. 注意事项

- Phase 0 **不写 Plugin 实现**（仅放桩头文件），避免并行工作冲突。
- Phase 0 **不写 LLM/CV Runner 实现**（仅声明类）。
- Phase 0 优先保证**可编译、可测试、可扩展**，不求功能完整。
- 所有新代码遵循 `.clang-format`，提交前运行 `clang-format -i`。

---

*文档版本：v1.0*  
*关联文档：`docs/mini_trt_llm_design.md`、`docs/future_iterations.md`*
