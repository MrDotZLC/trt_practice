# AGENTS.md - Codex 与 AI Agent 项目开发指南

本项目（`trt_practice`）正在演进并实现一个**极简、高性能的 C++ TensorRT-LLM 推理引擎**（`mini_trt_llm`）。

所有 AI 编程助手（Codex、Claude、Cursor 等）在进行代码生成、重构、CUDA Kernel 编写或 C++ 实现时，**必须严格遵守**本指南中的硬件限制、工程规范与设计模式。

序列0的条例永远有效。

## 0. 权限注意事项
0. 本指南中，“我”只指代作者本人。
1. 禁止修改AGENTS.md，每次修改都需我明确许可。
2. 在以下操作前，必须停止并请求我的确认：删除文件、修改配置文件、执行 git push 或涉及外部网络的操作。
3. 除单测外的测试任务，必须请求我的确认。
4. 问题排查路径要留痕，记录到 `docs/TROUBLESHOOTING.md`，让我知道你怎么排查的。
5. **批准目标 ≠ 批准手段**：计划文档里写了"要重写 / 删除 X"，不等于那个具体操作已获批。
   删除或覆盖现有文件时，每个具体动作都要单独确认；动手前把本轮的破坏性动作
   **一次性列成清单**给我确认（不要每步问一次，也不要因为"看起来琐碎"而省略）。
6. **不要自己当"这东西没人用"的裁判**：在代码里查不到引用不足以判定安全，
   我可能还有代码之外的用法。这类判断归我。

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

> 以下 Profile 与精度验证属于 §0 规则 3 中「单测外的测试任务」，执行前必须先请求确认。

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

0. **计划对账（在写任何代码之前）**：见本节末尾的「计划对账」。
1. **头文件定义 (`.h` / `.hpp`)**：声明轻量化接口并附带清晰的 C++ 注释。
2. **CUDA / C++ 实现 (`.cu` / `.cpp`)**：编写 Kernel 算法与边界检查。
3. **单元测试 (`tests/`)**：提供极简单元测试验证正确性与边界情况（可直接执行，无需事先确认）。
4. **集成**：接入 `LLMRunner` 执行主循环（`Prefill` -> `Decode`）。

### 计划对账（每次开发前必做，结论要说出来）

任何一次（含继续上一轮未完成的任务）动手写代码之前，先完成并**在回复里明确说出**下面四件事：

1. **有没有计划文档**：这次要做的事是否已被 `docs/phaseN_development_plan.md` 覆盖。
   没有 → 先产出计划文档并等确认，不要直接开工。
2. **是否一致**：把这次要做的任务、接口、验收判据，逐条与文档描述对照，说出
   "哪几条对上了、哪几条有偏差"。
3. **偏差怎么处理**：不一致时**先改文档再改代码**（或在动手前说明差异并取得确认）。
   禁止"代码先走、文档后补"——文档一旦落后于代码，下一个会话就会照文档把修正改回去。
4. **反向也要查**：文档里与代码现状矛盾的地方属于必须当场修的问题，
   修复时要连原因一起写进去（只写结论的文档等于给下一个人埋雷）。

**为什么要有这一步**：本项目已经出现过两次代价很高的偏差——计划文档里写着"6 个输入"、
代码已是"每层一对 cache 输入"（照文档改回去就是恢复一个真 bug）；以及口头确认过的接口改动
没有回到文档里。两者都会让下一个会话在错误的基线上开工。

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
  - 排查过程要留痕：定位路径（用过的命令、关键日志、对比数据）写进 `docs/TROUBLESHOOTING.md`；
    PROGRESS 的「已知问题与坑」只保留结论（问题 / 影响 / Workaround）并指向该文件，避免交接文档膨胀成流水账
  - 输出为 Markdown，更新到 docs/PROGRESS.md。

---

## 7. 验证与证据纪律

**当实测与期望不一致时，只允许三种动作**：

1. 继续查，状态明确写成"原因未知"；
2. 证明**期望值本身**错——必须给出独立依据（参考实现 / 实测敏感性数据 / 设计文档出处），
   并把推导过程写进代码注释与文档；
3. 标成"已知失败 + 原因未知"，**保持红色**。

**禁止**：调大阈值、删除断言、把断言降级成打印、跳过用例。
放宽期望值只会掩盖最难查的那类问题——本项目的真 bug（`TROUBLESHOOTING` #4 / #5 / #8 / #10 / #15）
全是数值正确性问题，放宽阈值恰好只对它们生效。

**阈值规则**：

- 每个数值阈值旁边必须写清出处（哪次实测 / 哪个标准 / 哪份文档）。可以松，但不能来路不明。
- 阈值不跨精度复用：`phase1_development_plan.md` D4 的 `相对误差 < 1e-3` 是 **FP16** 标准，
  套到 FP32 上等于把尺子放宽约 1000 倍。
- 放宽阈值前必须先量"与正确性无关的差异"（算法不同 / 累加顺序不同 / kernel 不同），
  阈值取其合理倍数。观测值若比它高出几个数量级，说明另有原因，**此时唯一的动作是查**。
- 验收时要能回答"这个阈值凭什么这么定"，而不是只看"绿了没有"。

**诊断规则**：

- 诊断输出必须说明比较对象是什么（比了哪两个张量、各自的布局与形状），
  否则会输出"看起来像真故障"的数字，把排查方向带偏（见 `TROUBLESHOOTING` #15 的 `13.8` / `175`）。
- 诊断代码本身也要自证：D2H 读回的范围必须落在同一段分配内。
- 关键诊断同时给**绝对差与相对差**：相对差在小值上会被放大，单看它会误判严重程度。
