# AGENTS.md - Codex 与 AI Agent 项目开发指南

本项目（`trt_practice`）正在演进并实现一个**极简、高性能的 C++ TensorRT-LLM 推理引擎**（`mini_trt_llm`）。

所有 AI 编程助手（Codex、Claude、Cursor 等）在进行代码生成、重构、CUDA Kernel 编写或 C++ 实现时，**必须严格遵守**本指南中的硬件限制、工程规范与设计模式。

## 0. 权限注意事项
1. 在以下操作前，必须停止并请求我的确认：删除文件、修改配置文件、执行 git push 或涉及外部网络的操作。
2. 除单测外的测试任务，必须请求我的确认。
3. 问题排查路径要留痕，让我知道你怎么排查的。

---

## 1. 本地硬件与开发环境限制

生成的 CUDA 代码与 CMake 配置必须严格适配当前开发机环境：

- **操作系统 / 运行时**：Ubuntu on WSL2 (Windows 10/11)
- **GPU 显卡架构**：NVIDIA GeForce GTX 1660 Ti 移动版 (Turing 架构，无 Tensor Core)
- **CUDA 算力架构 (Compute Capability)**：`sm_75`
- **编译器支持**：C++17 / C++20，`nvcc`
- **推理运行时**：TensorRT `10.15.1`（已确认支持 `IPluginV3`）
- **CUDA Toolkit**：12.6.85（`CUDART_VERSION 12060`）
- **支持的数据精度**：FP16 (`half`)、INT8、FP32
- **不支持的技术特性**：FP8、Hopper/Ampere 架构独占的 Transformer Engine 特性（如 Tensor Core FP8 / FP4 指令）。
- **核心工具链与分析备用策略**：
  - CMake >= 3.18
  - **Nsight Systems (`nsys`)**：系统级 Timeline 分析。
  - **Nsight Compute (`ncu`)**：Kernel 细粒度分析。*注意：若 WSL2 环境因 Performance Counter 权限或驱动问题导致 GUI / CLI 无法直接交互剖析，应采用无头导出策略（`-o output_name` 导出 `.ncu-rep` 文件），拷贝至 Windows宿主机，通过 Windows 端 Nsight Compute GUI 打开分析。*
---

## 2. 目标架构与目录结构 (`mini_trt_llm`)

项目核心逻辑收拢在 `mini_trt_llm/` 子目录下，各模块保持高度解耦：

```text
mini_trt_llm/
├── CMakeLists.txt      # 模块构建脚本
├── include/            # C++ 头文件
│   ├── core/           # 执行上下文、Runner、Engine 封装
│   ├── kv_cache/       # BlockAllocator、Paged KVCache 管理器
│   ├── plugins/        # TensorRT 自定义插件 (RoPE, RMSNorm, PagedAttention 等)
│   ├── sampler/        # Greedy / Top-P / Top-K CUDA 采样器
│   ├── tokenizer/      # BaseTokenizer 抽象 + SentencePiece 实现
│   └── utils/          # CUDA 错误检查、Profiler、显存池等
├── src/                # 实现文件 (.cpp, .cu)
│   ├── core/
│   ├── kv_cache/
│   ├── plugins/
│   ├── sampler/
│   ├── tokenizer/
│   └── utils/
├── tests/              # 单元测试 (GoogleTest)
├── third_party/        # 源码嵌入：sentencepiece / safetensors-cpp / googletest
└── tools/              # 模型转换脚本 (convert/hf_to_mini_trt_llm.py)

```

---

## 3. 核心开发规范

### A. CUDA 与性能标准

1. **高性能 CUDA 编写**：优先采用 grid-stride loops、显存对齐访问、合并访存（Coalesced Access）与向量化读写（如 `float4`、`half2`）。
2. **严谨的错误处理**：所有 CUDA API 与 TensorRT API 调用**必须**包裹错误检查宏（如 `CUDA_CHECK(...)`、`NVINFER_CHECK(...)`）。
3. **零无谓 Host-Device 拷贝**：Decode 阶段的自回归生成循环必须全程在 GPU 显存内完成，禁止在 loop 内部调用 `cudaMemcpy`Sync。

### B. TensorRT Plugin 设计规范

1. 自定义插件必须继承 `nvinfer1::IPluginV3`（按 `IPluginV3OneCore` / `IPluginV3OneBuild` / `IPluginV3OneRuntime` 能力拆分实现），不使用 legacy 的 `IPluginV2DynamicExt`。
2. 显式支持 Dynamic Shapes（动态 `batch_size` 与动态 `seq_len`）。
3. 动态 Scratch 显存必须严格通过 TensorRT 的 `getWorkspaceSize()` 分配，禁止在 `enqueue` 运行时临时分配 GPU 显存（如 `cudaMalloc`）。

### C. C++ 工程规范

1. **C++ 标准**：C++17。遵循 RAII 原则管理 CUDA Stream/Event 与 GPU 资源，推荐使用智能指针（`std::unique_ptr`）。
2. **命名空间与风格**：统一使用命名空间 `mini_trt_llm`，遵循 Google C++ 代码风格。

---

## 4. 常用构建与测试指令

### CMake 构建

```bash
# 配置 CMake（针对 sm_75 编译）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75

# 编译 LLM Engine 模块
cmake --build build -j$(nproc)

```

### 运行测试与性能分析

> 自检边界：编译与单元测试属于日常开发自检，Agent 可直接执行，无需事先确认；
> 其余测试与验证任务（性能 Profile、精度验证、端到端等）按 §0 规则需先请求确认。

#### 1. 单元测试

所有 `mini_trt_llm/tests/test_*.cpp` 汇总编译为单一目标 `mini_trt_llm_tests`，不按模块拆分为独立二进制。

```bash
# 运行全部用例
ctest --test-dir build --output-on-failure

# 只跑指定用例（gtest filter）
./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='RmsNorm*'
```

涉及 GPU / TensorRT 运行时的用例在沙箱内无法执行（无 GPU 访问），需在 WSL2 真机运行。

#### 2. Nsight Systems 系统级 Profile

> 以下 Profile 与精度验证属于 §0 规则 2 中「单测外的测试任务」，执行前必须先请求确认。

```bash
# 按 gtest filter 选中要 profile 的用例（Phase 2 接入 LLMRunner 后可换成端到端用例）
nsys profile --stats=true -o trt_llm_trace \
    ./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='*Plugin*'
```

#### 3. Nsight Compute (ncu) Kernel 级 Profile（WSL2 导出备用策略）

```bash
# 策略：直接在 WSL2 导出 .ncu-rep 报告文件
ncu --set full -o ncu_report_kernel \
    ./build/mini_trt_llm/tests/mini_trt_llm_tests --gtest_filter='RmsNorm*'
# 拷贝至宿主机并在 Windows 版 Nsight Compute 打开：
# 报告输出路径：./ncu_report_kernel.ncu-rep
```

---

## 5. Incremental 开发推进流程

生成新功能代码时，请按以下顺序分步推进：

1. **头文件定义 (`.h` / `.hpp`)**：声明轻量化接口并附带清晰的 C++ 注释。
2. **CUDA / C++ 实现 (`.cu` / `.cpp`)**：编写 Kernel 算法与边界检查。
3. **单元测试 (`tests/`)**：提供极简单元测试验证正确性与边界情况（可直接执行，无需事先确认）。
4. **集成**：接入 `LLMRunner` 执行主循环（`Prefill` -> `Decode`）。

## 6. 进度维护

当完成一个阶段性任务，或做出重要架构决策时，
更新 docs/PROGRESS.md 的对应部分。

### 更新规则
请把当前进度总结成一份可以交接给新会话的文档，包含以下部分：

  1. 项目目标（一句话）
  2. 当前架构与关键决策（含为什么这么选）
  3. 已完成的部分（文件 / 模块级别）
  4. 进行中 / 未完成的部分
  5. 已知问题与坑（含 workaround）
  6. 下一步计划
  7. 重要的环境信息（TensorRT 版本、CUDA 版本、依赖库版本等）

  要求：
  - 只保留决策和状态，不要复述对话过程
  - 涉及取舍的地方写清楚“为什么选 A 不选 B”
  - 排查过程要留痕：在「已知问题与坑」记录定位路径（用过的命令、关键日志、对比数据），而不只写结论
  - 输出为 Markdown，更新到 docs/PROGRESS.md。
