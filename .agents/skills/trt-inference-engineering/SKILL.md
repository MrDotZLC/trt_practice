---
name: trt-inference-engineering
description: 在 C++/CUDA/TensorRT/LLM 推理项目里做功能开发、Bug 修复、评审与 benchmark 时使用，按 P0–P9 / B0–B4 流程推进并维护 docs/dev 下各 feature 的 artifact。单纯解释代码或讨论概念时不触发。
---

# TensorRT Inference Engineering Skill

## Purpose

该 Skill 用于在 C++ / CUDA / TensorRT / LLM Runtime 项目中，
按照工程化流程开发、优化和维护推理系统。

目标：

- 保证功能开发具有明确设计依据
- 保证代码修改具有测试验证
- 保证性能优化具有可复现 benchmark
- 保证长期项目维护具有状态恢复能力

## Trigger

路由以 frontmatter 的 description 为权威摘要，本节列出细化的触发条件。

### Feature Workflow

以下请求触发 Feature Workflow：

- 实现 XXX 功能
- 增加 XXX 支持
- 支持 XXX
- 优化 XXX 性能
- 扩展 XXX 模块

### Bugfix Workflow

以下请求触发 Bugfix Workflow：

- 修复 XXX
- XXX 崩溃
- XXX 结果错误
- XXX 性能退化

### Do Not Trigger

以下情况不启动 Workflow：

- 单纯解释代码
- 技术概念讨论
- 阅读代码分析
- 学习问题

### Ambiguous Request

如果无法判断属于 Feature 或 Bugfix：

必须询问用户。

禁止自动选择。

## Directory Model

该 Skill 使用双目录模型。

### Skill Directory

路径：

<project_root>/.agents/skills/trt-inference-engineering/

用途：

保存：

- Workflow规则
- Phase规则
- Checklist规则
- Template定义

约束：

- 只读
- 禁止写入开发状态
- 禁止生成项目artifact

### Artifact Directory

路径：

<project_root>/docs/dev/<feature>/

用途：

保存当前项目开发状态。

标准结构：

STATE.md

requirement.md

analysis.md

design.md

review.md

benchmark_before.md

benchmark.md

test_plan.md

summary.md

interview_notes.md

规则：

所有开发过程产物必须写入 Artifact Directory。

### Naming Conventions

STATE.md 的 `phase` 字段使用连字符标识（feature 用 `P0-Requirement` … `P9-Interview`，bugfix 用 `B0-Reproduce` … `B4-Summary`），便于机器比对；正文叙述与状态机使用空格形式（`P3 Review`）。

`status` 字段取值限定为 `in-progress` / `waiting-human-gate` / `completed`；结束状态由 `status` 表达，不写 `phase: DONE`。

## Global Rules

### Mandatory Rules

所有 Workflow 必须遵守：

1. 先分析，再设计，再修改代码。

2. 修改代码前必须存在对应设计artifact。

3. 新功能必须包含测试。

4. 性能优化必须包含baseline。

5. 每个Phase结束必须更新STATE.md。

6. 每次修改必须保持项目可编译。

7. 唯一来源 + 指针式登记：每条规则只有一个"措辞权威"层（权限 / 项目事实 → AGENTS.md；流程机制 →
   本技能；本 feature 的实例与证据 → 该 feature 的文档）。规则从低层升到高层时，**同一次动作**里
   必须把低层副本降级为指针（规则名 + 出处 + 本层实例），不复述条文。

8. **枚举判据不得转述（只能引用 + 逐项打勾）**：凡本技能 / `AGENTS.md` 以**枚举清单**给出的判据
   （如"三处留痕""四份 checklists""N 条不变量"…），产出里**不得自己重抄清单**，只能写"出处 + 逐项
   一行（项 / 状态 / 证据指针）"，且**行数必须等于权威清单的行数**；缺行即未完成。
   先例：`phases/p3_review.md` 的 Exit Gate 已要求三张表的行数等于对应集合条目数 —— 本条把它推广到
   所有枚举判据。

9. **例外条款必须自带"不得跳过哪几项"**：任何 N/A / 降级 / 豁免 / 例外条款旁，必须逐项点名
   "本条款不得用于跳过以下判据：…"。清单外的跳过一律按 `AGENTS.md` §7 处理（禁止跳过判据）。

10. **异常路径要留"判据对照"，且下一阶段入口先复核**：阶段产出里凡出现"不适用 / 未满足 / 放行"，
    必须在 `STATE.md` 的 `## 判据对照` 一节留行（判据 / 状态 / 证据或"谁放行、放行范围"）；
    **全绿阶段不必写**（轻量口径）。下一阶段的 `## Entry` 先复核该节：存在"未满足（阻塞）"项时
    **不得开工**，除非作者明确放行并把放行范围记在该节里。**触发条件**：凡发生"依赖缺失 / 降级 /
    N-A 放行 / 阶段切换"，必须对表一次（用第 8 条的格式），留档即在对照块里 —— 口头说"对过了"不算。

### Forbidden Rules

禁止：

1. 未生成 analysis.md 前修改源码。

2. 未通过 Human Gate 进入下一阶段。

3. 新功能没有测试直接提交。

4. 性能优化没有benchmark直接声称提升。

5. Bugfix过程中顺手重构架构。

6. 猜测不存在的feature状态。

7. 修改Skill自身文件（绝对禁止；每次修改都需作者明确许可，许可指令为“授权修改SKILL一次”）。

## Workflow Routing

### Feature Workflow

适用于：新增能力、新增模块、性能优化、架构增强。

阶段顺序：P0 Requirement → P1 Analysis → P2 Design → P3 Review → Gate-A → P4 Baseline → P5 Implementation → P6 Test → P7 Benchmark → Gate-B → P8 Documentation → P9 Interview。

详细流程、各 Gate 判据与所需 artifact 以 [workflows/feature.md](workflows/feature.md) 为准，本节不复述。

Phase 执行规则：进入 Phase 时只读取对应的 `phases/pX_xxx.md`（见 [phases/](phases/)），禁止一次加载所有 Phase。

### Bugfix Workflow

适用于：Bug 修复、崩溃修复、正确性修复。

阶段顺序：B0 Reproduce → B1 Diagnose → B2 Minimal Fix → B3 Regression → B4 Summary。

详细流程以 [workflows/bugfix.md](workflows/bugfix.md) 为准，本节不复述。

## Human Gate Protocol Summary

该Skill包含两个Human Gate。

### Gate-A

位置：

P3 Review之后

输入：

design.md + review.md

规则：

P0 Blocker：

必须暂停，返回 P2。

P1 Risk：

必须暂停等待确认。

P2 Quality：

记录即可继续。

### Gate-B

位置：

P7 Benchmark之后

判据以 [phases/p7_benchmark.md](phases/p7_benchmark.md) 的 Performance Decision 为唯一权威（含判别下限前置）；缺依赖导致 P7 不可执行时，以 [phases/p4_baseline.md](phases/p4_baseline.md) 的 Dependency Missing 为唯一权威。

规则：

性能提升 >=5%，且 >= 判别下限：

自动接受。

低于判别下限：

只能写"无显著差异"，等待确认。

性能变化 0~5%：

等待确认。

性能下降：

等待确认。

## Session Recovery

Skill启动时：

Step 1:

搜索：

<project_root>/docs/dev/*/STATE.md

Step 2:

如果只有一个feature：

恢复该feature。

Step 3:

如果存在多个feature：

读取所有STATE.md。

展示：

- feature
- workflow
- phase
- status

等待用户选择。

Step 4:

如果：

status = waiting-human-gate

恢复Human Gate状态。

Step 5:

如果STATE.md缺失：

根据artifact推断状态。

重新生成STATE.md。

禁止：

猜测恢复目标。

## 会话收尾（Handoff）

收尾回复里凡出现"本该在动手前说、我在这里补记"这类自我披露，**必须同时产出一条 Recovery Note
（写进该 feature 的 STATE.md）或一条流程改动**；只披露不落载体不算闭环。

## Multi Feature Handling

当 `<project_root>/docs/dev/` 下存在多个 feature 时。

用户输入：

继续开发

必须展示：

Feature:

Workflow:

Phase:

Status:

等待用户选择。

禁止：

默认选择最近修改项目。

### 写冲突

当多条进行中 feature 需要修改同一个文件时。

规则：

- 必须由用户指定串行顺序，禁止并行开工。
- 未指定顺序前，后开工的一方停在当前阶段，不动代码。
- 冲突关系记入各方 STATE.md 的 Current Blockers（写清与哪条 feature、哪个文件冲突）。
- 先开工的一方收口后，后开工的一方重新对齐基线再开工。
- 禁止用"代码里查不到引用"判定自己可以先改：用户的代码外用法不在搜索范围内。

## File Index

| 场景 | 读取 |
| --- | --- |
| 走 Feature 流程 | [workflows/feature.md](workflows/feature.md) |
| 走 Bugfix 流程 | [workflows/bugfix.md](workflows/bugfix.md) |
| P0-Requirement 阶段 | [phases/p0_requirement.md](phases/p0_requirement.md) |
| P1-Analysis 阶段 | [phases/p1_analysis.md](phases/p1_analysis.md) |
| P2-Design 阶段 | [phases/p2_design.md](phases/p2_design.md) |
| P3-Review 阶段 | [phases/p3_review.md](phases/p3_review.md) |
| P4-Baseline 阶段 | [phases/p4_baseline.md](phases/p4_baseline.md) |
| P5-Implementation 阶段 | [phases/p5_implementation.md](phases/p5_implementation.md) |
| P6-Test 阶段 | [phases/p6_test.md](phases/p6_test.md) |
| P7-Benchmark 阶段 | [phases/p7_benchmark.md](phases/p7_benchmark.md) |
| P8-Documentation 阶段 | [phases/p8_documentation.md](phases/p8_documentation.md) |
| P9-Interview 阶段 | [phases/p9_interview.md](phases/p9_interview.md) |
| 提交前自检 C++ | [checklists/cpp.md](checklists/cpp.md) |
| 提交前自检 CUDA | [checklists/cuda.md](checklists/cuda.md) |
| 提交前自检 TensorRT | [checklists/tensorrt.md](checklists/tensorrt.md) |
| 提交前自检 LLM Runtime | [checklists/llm_runtime.md](checklists/llm_runtime.md) |
| 生成 requirement.md | [templates/requirement.md](templates/requirement.md) |
| 生成 analysis.md | [templates/analysis.md](templates/analysis.md) |
| 生成 design.md | [templates/design.md](templates/design.md) |
| 生成 review.md | [templates/review.md](templates/review.md) |
| 生成 benchmark_before.md | [templates/benchmark_before.md](templates/benchmark_before.md) |
| 生成 benchmark.md | [templates/benchmark.md](templates/benchmark.md) |
| 生成 test_plan.md | [templates/test_plan.md](templates/test_plan.md) |
| 生成 summary.md | [templates/summary.md](templates/summary.md) |
| 生成 interview_notes.md | [templates/interview_notes.md](templates/interview_notes.md) |
| 维护 STATE.md | [templates/STATE.md](templates/STATE.md) |
