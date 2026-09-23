# Phase 0 模型加载测试方案

> 目标：在不进入 Phase 1 的前提下，验证 Phase 0 骨架的“模型加载”路径是否可闭环测试。

> **实施状态（截至 2026-09-24）**：本方案尚未实施，T1/T2/T3 全部为待办。
> - T1：`test_safetensors_loader.cpp` 当前只有 `LoadNonExistentReturnsFalse` 一个负向用例，真实 safetensors 文件解析用例未落地。
> - T2：尚未创建 `test_onnx_build.cpp`，`test_engine.cpp` 中也未追加 ONNX 构建用例。
> - T3：尚未创建 `test_builder_e2e.cpp`，DummyBuilder 端到端链路未验证。

## 1. 背景

Phase 0 已搭建完成：
- `EngineBuilder` 支持 `BuildFromConfig`（Safetensors + JSON）和 `BuildFromOnnx` 两条路径。
- `ModelRegistry`、`IModelBuilder`、`WeightLoader`、`SafetensorsLoader`、`ModelConfig` 已就位。
- 但当前没有真实的 `GPT2ModelBuilder` / `ResNet18ModelBuilder`，因此 `BuildFromConfig` 对真实 model_type 会失败。

本方案计划补充 3 组测试，使 Phase 0 的加载链路具备可运行、可验证的测试覆盖。

## 2. 测试范围

| 编号 | 测试目标 | 验证路径 | 依赖 GPU | 预计耗时 |
|---|---|---|---|---|
| T1 | 真实 Safetensors 文件加载 | `SafetensorsLoader` | 否 | < 1s |
| T2 | ONNX 解析并生成 Engine | `EngineBuilder::BuildFromOnnx` | 是（builder 需要 CUDA） | 数十秒 ~ 数分钟 |
| T3 | 配置驱动 + DummyBuilder 端到端 | `EngineBuilder::BuildFromConfig` + `IModelBuilder` | 是（builder 需要 CUDA） | < 10s |

## 3. 测试用例

### T1. Safetensors 真实文件加载测试

**文件**：`mini_trt_llm/tests/test_safetensors_loader.cpp`

**测试数据**：复用仓库内置的 `mini_trt_llm/third_party/safetensors-cpp/gen/model.safetensors`，其中包含：
- `weight1`: shape `(8, 8)`, dtype `float64`
- `weight2`: shape `(16, 16)`, dtype `float64`

**用例**：

| 用例名 | 输入 | 期望结果 |
|---|---|---|
| `LoadExistingFile` | 上述 safetensors 路径 | `LoadFromFile` 返回 true |
| `HasTensorExisting` | 查询 `weight1` / `weight2` | 返回 true |
| `HasTensorMissing` | 查询 `not_exist` | 返回 false |
| `GetTensorInfoWeight1` | 读取 `weight1` 信息 | dtype 为 `kFLOAT64`，shape 为 `{8, 8}` |
| `GetRawDataWeight1` | 读取 `weight1` 原始数据 | 指针非空，字节数为 `8 * 8 * 8 = 512` |
| `GetTensorInfoWeight2` | 读取 `weight2` 信息 | dtype 为 `kFLOAT64`，shape 为 `{16, 16}` |

**注意**：当前 `GetConvertedData` 仅支持 BF16 → FP32/FP16 转换，float64 会 fallback 为 kFLOAT 并返回原始指针。本测试不验证 float64 → FP32 转换语义，只验证文件解析与元数据读取。

---

### T2. ONNX 解析并生成 Engine 测试

**文件**：`mini_trt_llm/tests/test_engine.cpp`（追加）或新建 `test_onnx_build.cpp`

**测试数据**：
- 主测试：`1_gpt2_onnx/gpt2.onnx`（约 622 MB，真实 GPT-2 ONNX）
- 备选：`0_resnet18_onnx/resnet18.onnx`（约 45 MB，CNN ONNX）

**用例**：

| 用例名 | 输入 | 期望结果 |
|---|---|---|
| `BuildFromOnnxValid` | 有效的 `resnet18.onnx` 路径 | `BuildFromOnnx` 返回 true，生成的 `.engine` 文件存在且大小 > 0 |
| `BuildFromOnnxMissingFile` | 不存在的 ONNX 路径 | `BuildFromOnnx` 返回 false |

**实现要点**：
- 使用 `EngineBuilder` 默认 `Config`（FP16、workspace 1GB）。
- 输出 engine 放到 `/tmp/mini_trt_llm_test_*/` 临时目录，测试结束后清理。
- `plugin_ops` 参数传空列表，当前不替换任何子图。

**环境要求**：
- 需要真实 CUDA 驱动与 TensorRT。
- 沙箱环境会因 `CUDA driver version is insufficient` 失败，需在用户 WSL2 环境运行。

---

### T3. 配置驱动 + DummyBuilder 端到端测试

**文件**：新建 `mini_trt_llm/tests/test_builder_e2e.cpp`

**设计思路**：
用一个最小的合法 TRT network 模拟真实 builder，验证 `EngineBuilder::BuildFromConfig` 的整条链路：
`config.json` → `ModelConfig::Load` → `ModelRegistry::Get` → `IModelBuilder::Build` → `buildSerializedNetwork` → 写 engine 文件。

**DummyBuilder 实现**：
构造一个可序列化的最小网络：
- 输入：`input`，shape `(1, 1)`，FP32
- 常量：`const`，shape `(1, 1)`，值 `[1.0f]`
- `ElementWise` 加法层：`output = input + const`
- 标记 `output` 为网络输出

**测试数据**：
在临时目录 `/tmp/test_mini_trt_e2e_*/` 下创建：
- `config.json`：
  ```json
  {
      "model_type": "dummy",
      "architecture": "test",
      "hyper_params": {"value": 1.0},
      "weight_map": {}
  }
  ```
- `model.safetensors`：空文件或包含一个占位张量（`WeightLoader::Load` 要求文件存在）。
  由于 DummyBuilder 不读取权重，可创建一个只含一个最小张量的 safetensors 文件。

**用例**：

| 用例名 | 输入 | 期望结果 |
|---|---|---|
| `BuildFromConfigWithDummyBuilder` | 临时目录 + 已注册 DummyBuilder | `BuildFromConfig` 返回 true，engine 文件生成 |
| `BuildFromConfigMissingBuilder` | 临时目录 + 未注册任何 builder | 返回 false，并记录 `No registered builder for model type: dummy` |
| `BuildFromConfigMissingConfig` | 目录不存在 `config.json` | 返回 false |
| `BuildFromConfigMissingWeights` | 目录存在 config.json 但无 `model.safetensors` | `WeightLoader::Load` 失败，返回 false |

**环境要求**：
- 需要真实 CUDA 驱动（因为要走 `buildSerializedNetwork`）。
- 沙箱环境会失败，但代码逻辑本身可在无 GPU 时部分验证到 `ModelConfig::Load` 阶段。

## 4. 测试分组建议

为避免无 GPU 环境拖慢日常 CI，建议将测试分为两组：

| 分组 | 包含用例 | 运行方式 |
|---|---|---|
| `host_only` | T1 全部 | 默认运行，不依赖 GPU |
| `gpu_required` | T2、T3 | 仅在 WSL2 真机运行，可加 `--gtest_filter=*Onnx*:*BuilderE2E*` 或 CMake 选项控制 |

具体实现时，可用环境变量或编译宏控制 GPU 测试是否启用：
```cpp
bool HasGpu() {
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess;
}
```
GPU 测试开头 `GTEST_SKIP() << "No GPU available";`。

## 5. 预期收益

- 验证 `SafetensorsLoader` 对真实文件的解析能力。
- 验证 `BuildFromOnnx` 能走通 ONNX → Engine 的完整流程。
- 验证 `BuildFromConfig` 的注册表分发、配置解析、权重加载、序列化全链路。
- 为 Phase 1/2 提供一个最小可工作的测试模板（DummyBuilder 模式）。

## 6. 风险与已知问题

| 问题 | 影响 | Workaround |
|---|---|---|
| 沙箱无 GPU 驱动 | T2、T3 无法在当前 Agent 环境运行 | 代码照常提交，真机 WSL2 运行验证 |
| `SafetensorsLoader::GetTensorNames()` 返回空 | 不影响 T1，因为 T1 按名查询 | 已在 PROGRESS.md 中记录 |
| float64 权重转换未实现 | T1 不验证转换，仅验证元数据 | 真实 LLM 权重一般为 FP16/BF16，不触发此路径 |
| ONNX build 可能耗时较长 | T2 首次运行可能数分钟 | 优先用 `resnet18.onnx`（45MB）做 smoke test |
