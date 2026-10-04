# AGENTS.md - Codex 与 AI Agent 项目开发指南

本项目（`trt_practice`）正在演进并实现一个**极简、高性能、支持多模态的 C++ TensorRT-LLM 推理引擎**（`mini_trt_llm`）。

所有 AI 编程助手（Codex、Claude、Cursor 等）在进行代码生成、重构、CUDA Kernel 编写或 C++ 实现时，**必须严格遵守**本指南中的硬件限制、工程规范与设计模式。

序列0的条例永远有效。

## 0. 权限注意事项
0. 本指南中，“我”只指代作者本人。
1. 禁止修改AGENTS.md，每次修改都需我明确许可，许可指令为“授权修改AGENTS.md一次”。
2. 在以下操作前，必须停止并请求我的确认：删除文件、修改配置文件、执行 git commit（含任何形式的
   自动提交）与 git push、涉及外部网络的操作，以及下列两档文档与技能文件的修改。
   （AGENTS.md 本身仍按第 1 条的口令规则。）
   - **提交口径**：需要提交时先提醒我并给出完整 commit msg 与文件清单，得到确认后才执行；
     禁止 "add + commit" 连做（细则见技能的 Commit Rules）。
   - **文档口径（分级）**：**结构 / 契约类**——`requirement` / `analysis` / `design` / `review` /
     `*_interface_spec` / `PROGRESS`——**先确认再改**；
     **阶段状态回填类**——各 feature 的 `STATE` / `test_plan` / `benchmark*` / `summary` /
     `interview_notes`，以及 `TROUBLESHOOTING`（**追加式**留痕，与第 4 条"排查留痕"一致）——
     在该 Phase 已获授权的前提下可自动更新，但必须在当轮回复里**逐条列出**改了哪几处。
   - **技能口径**：`.agents/skills/**` **绝对禁止**修改；每次修改都需我明确许可，许可指令为
     “授权修改SKILL一次”（一次许可 = 整个技能目录算一次修改）。
3. 除单测外的测试任务，必须请求我的确认。
4. 问题排查路径要留痕，记录到 `docs/TROUBLESHOOTING.md`，让我知道你怎么排查的。
5. **批准目标 ≠ 批准手段**：计划文档里写了"要重写 / 删除 X"，不等于那个具体操作已获批。
   删除或覆盖现有文件时，每个具体动作都要单独确认；动手前把本轮的破坏性动作
   **一次性列成清单**给我确认（不要每步问一次，也不要因为"看起来琐碎"而省略）。
6. **不要自己当"这东西没人用"的裁判**：在代码里查不到引用不足以判定安全，
   我可能还有代码之外的用法。这类判断归我。
7. **没我明确同意，不许开工**：只有当我点名批准了**具体某项**（条目号 / 文件 / 任务号），
   才允许开始那项工作。**开工包含写计划文档、改代码、跑实验、真机任务**——不只是改产品代码。
   - **笼统的话不算批准**："开始吧" "继续" "看着办" 这类没有指明范围的话，
     不构成对任何具体工作的许可；不确定就停下来问"要我现在做 X 吗"。
   - **批准范围不自动外扩**：批准写计划 ≠ 批准改代码；批准改代码 ≠ 批准 bump 图版本 /
     删缓存 / 跑真机 / 扩测试面。
   - **已经点名的范围就是上限**，不要"顺手"多做一件。
   - **为什么**：我有自己的节奏与验收安排。Agent 抢跑会产生我事先没有预期的改动，
     而这类改动一旦落盘（尤其是产品代码与文档结论），评审与回滚的成本由我承担。

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

**技能清单（本仓库自带，均在 `.agents/skills/`）**

| 技能 | 入口 | 何时使用 |
| --- | --- | --- |
| `trt-inference-engineering` | `.agents/skills/trt-inference-engineering/SKILL.md` | 功能开发 / Bug 修复 / 评审 / benchmark；**本指南各处的"技能"均指它** |
| `cpp-comment-style` | `.agents/skills/cpp-comment-style/SKILL.md` | 生成或修改 C++ 注释时（对应 §3 的 C++ 工程规范） |
| `progress-summary` | `.agents/skills/progress-summary/SKILL.md` | 生成可交接的进度总结时（对应 §6 的 `docs/PROGRESS.md`） |

**流程权威**：阶段链、Gate 判据、artifact 落点与模板、各阶段自检清单——**全部以上表中
`trt-inference-engineering` 为准**。
本节**只**规定与权限、项目事实有关的部分，**不复述**技能内容。

- **开工资格**：技能里的"自动进入下一阶段"（例如 P0 目标明确即自动进 P1）**不构成开工许可**。
  写计划文档 / 改代码 / 跑实验 / 真机任务，仍须按 §0.7 由作者点名；未点名时停在当前阶段。
- **Gate 与 §0.7 并行**：Gate 决定"设计是否被接受"，§0.7 决定"是否允许开始"。两者都要过。
- **单测外测试的授权范围**：技能规定的**基准与测量阶段**（P4 Baseline / P7 Benchmark）视为
  **已授权、可自动执行**；**其余单测外测试**（真机任务、精度验证、端到端等）仍按 §0.3
  **先请求确认**。
- **性能判定的本机参数**：技能 P7 与 Gate-B 已内置"先声明本次口径的判别下限、低于下限只能写
  **无显著差异**"的要求（见技能 `.agents/skills/trt-inference-engineering/phases/p7_benchmark.md`）；本机这类测量的判别下限约 ±400~600 µs
  （见 `docs/PROGRESS.md` §5.13b），因此 Gate-B 的"≥5% 自动接受"必须同时满足该下限。
- **改代码的前置**：技能 P5 要求先写 Implementation Plan（当前模块 / 预计文件 / 测试方式）并记进
  对应条目的 `STATE.md`；该步完成后再按下面四步推进。
- **提交规则**：按技能的 Commit Rules（`.agents/skills/trt-inference-engineering/workflows/feature.md`）
  执行——**每个 commit 只完成一个 Phase 子任务**，推荐格式 `[feature][Phase-X] description`，
  **禁止**一个 commit 跨越多个无关功能。无 Phase 归属的改动（如纯文档整理）单独成笔，不与功能改动混提。
  **无 Phase 归属的改动**（纯文档整理 / 规则维护）：feature 标签用 **`[文档修改]`**（英文 `[docs]`
  亦可），不带 Phase 段。
- **产物完备性**：按技能的 Required Artifacts（10 件：`STATE` / `requirement` / `analysis` / `design` /
  `review` / `benchmark_before` / `benchmark` / `test_plan` / `summary` / `interview_notes`）执行。
  `docs/dev/` 里的历史条目目前只有四件套（`STATE` / `requirement` / `design` / `test_plan`），
  **不构成对新条目的豁免**。
- **面试笔记的层级**：技能 P9 产出的是**每个条目**的 `interview_notes.md`（放在该条目目录）；
  项目级面试总结仍是 `docs/interview_summary.md`（材料类、**不参与 SSOT**），两者不互相替代。

**C++ 侧按以下顺序分步推进：**

0. **计划对账（在写任何代码之前）**：见本节末尾的「计划对账」。
1. **头文件定义 (`.h` / `.hpp`)**：声明轻量化接口并附带清晰的 C++ 注释。
2. **CUDA / C++ 实现 (`.cu` / `.cpp`)**：编写 Kernel 算法与边界检查。
3. **单元测试 (`tests/`)**：提供极简单元测试验证正确性与边界情况（可直接执行，无需事先确认）。
4. **集成**：接入 `LLMRunner` 执行主循环（`Prefill` -> `Decode`）。

### 计划对账（每次开发前必做，结论要说出来）

任何一次（含继续上一轮未完成的任务）动手写代码之前，先完成并**在回复里明确说出**下面五件事：

1. **有没有计划文档**：这次要做的事是否已被 `docs/dev/<feature>/` 覆盖（`requirement.md` +
   `analysis.md` + `design.md`；Bugfix 走 `requirement.md` + `analysis.md` + 候选修复方案）。
   没有 → 先产出计划文档并等确认，不要直接开工。
   （历史阶段文档已按功能迁入 `docs/dev/REQ-NNN-*/`，只作判据出处，**不再作为新工作的计划落点**。）
2. **是否一致**：把这次要做的任务、接口、验收判据，逐条与文档描述对照，说出
   "哪几条对上了、哪几条有偏差"。
3. **偏差怎么处理**：不一致时**先改文档再改代码**（或在动手前说明差异并取得确认）。
   禁止"代码先走、文档后补"——文档一旦落后于代码，下一个会话就会照文档把修正改回去。
4. **反向也要查**：文档里与代码现状矛盾的地方属于必须当场修的问题，
   修复时要连原因一起写进去（只写结论的文档等于给下一个人埋雷）。
5. **来源缺失就停（不许推演实现）**：把计划 / 设计里每条“已定 / 必须拒绝”的条目按
   「兑现它需要什么输入或接口 → 系统里到底有没有」走一遍；走到“**找不到来源**”就标“待定”并
   停下来问我，**不得**自行选定一个来源把方案补完——缺来源不等于可以挑一个来源。
   （依据：`docs/dev/REQ-016-continuous-batching` 的 `TS-051` 第 4 条：`prompt_len > n_positions`
   当时被压缩成“入口加一条检查”才被批准，批次内的改动因此从“贴合既有设计”越界成“改设计契约”。）

**为什么要有这一步**：本项目已经出现过两次代价很高的偏差——计划文档里写着"6 个输入"、
代码已是"每层一对 cache 输入"（照文档改回去就是恢复一个真 bug）；以及口头确认过的接口改动
没有回到文档里。两者都会让下一个会话在错误的基线上开工。

## 6. 进度维护

当完成一个阶段性任务，或做出重要架构决策时，
更新 docs/PROGRESS.md 的对应部分。

> **与 §5 的分工**：**每个 Phase 结束**更新 `docs/dev/<feature>/STATE.md`（进行状态 / blocker）；
> **阶段或批次收口**才回填 `PROGRESS.md`（结论与基线）。两者都要写，落点不同、时效不同。

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
- 阈值不跨精度复用：`docs/dev/REQ-002-plugins/phase1_development_plan.md` D4 的 `相对误差 < 1e-3` 是 **FP16** 标准，
  套到 FP32 上等于把尺子放宽约 1000 倍。
- 放宽阈值前必须先量"与正确性无关的差异"（算法不同 / 累加顺序不同 / kernel 不同），
  阈值取其合理倍数。观测值若比它高出几个数量级，说明另有原因，**此时唯一的动作是查**。
- 验收时要能回答"这个阈值凭什么这么定"，而不是只看"绿了没有"。

**诊断规则**：

- 诊断输出必须说明比较对象是什么（比了哪两个张量、各自的布局与形状），
  否则会输出"看起来像真故障"的数字，把排查方向带偏（见 `TROUBLESHOOTING` #15 的 `13.8` / `175`）。
- 诊断代码本身也要自证：D2H 读回的范围必须落在同一段分配内。
- 关键诊断同时给**绝对差与相对差**：相对差在小值上会被放大，单看它会误判严重程度。
