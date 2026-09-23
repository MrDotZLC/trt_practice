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

> 以下验收项已由用户在本地 WSL2 真机环境确认通过（Agent 沙箱无 GPU，无法复现 CUDA 相关验证）。

- [x] `cmake -B build -DCMAKE_BUILD_TYPE=Release` 根目录配置成功。
- [x] `cmake --build build --target mini_trt_llm` 编译成功。
- [x] `cmake -B build -DBUILD_TESTS=ON && cmake --build build && ctest` 全部通过。
- [x] 旧模块 `trt_resnet18`、`trt_gpt2` 仍可正常构建。
- [x] SentencePiece 只编译核心 static lib，README.md 记录禁用功能。
- [x] Safetensors 能读取 header 与张量元数据。
- [x] `EngineBuilder` 通用化骨架可扩展（注册新的 IModelBuilder 不修改 Builder 类）。

---

## 5. 注意事项

- Phase 0 **不写 Plugin 实现**（仅放桩头文件），避免并行工作冲突。
- Phase 0 **不写 LLM/CV Runner 实现**（仅声明类）。
- Phase 0 优先保证**可编译、可测试、可扩展**，不求功能完整。
- 所有新代码遵循 `.clang-format`，提交前运行 `clang-format -i`。

---

*文档版本：v1.0*  
*关联文档：`docs/mini_trt_llm_design.md`、`docs/future_iterations.md`*
