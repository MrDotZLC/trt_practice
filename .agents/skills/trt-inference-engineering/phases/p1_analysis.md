# Phase P1 Analysis

## Purpose

分析当前系统实现。

回答：

- 当前代码如何工作
- 数据如何流动
- 修改点在哪里

禁止提出最终实现方案。

## Entry

读取：

requirement.md

## Actions

### 1. Repository Analysis

分析：

- 项目结构
- 模块依赖
- 调用链
- 数据流

### 2. Runtime Analysis

针对TensorRT / LLM Runtime：

分析：

- Engine生命周期
- Context生命周期
- Memory ownership
- CUDA Stream关系
- Kernel调用路径

### 3. Generate analysis.md

必须包含：

章节骨架以 [templates/analysis.md](../templates/analysis.md) 为准（本文件不复述）。

### 4. 术语表

必须包含：

`## Terminology`

规则：

- `requirement.md` 里的每一个模糊名词（如"动态 batch""最小连续批"）必须给出本项目的**可验证定义**。
- 定义必须能回答"怎样算做到了、怎样算没做到"。
- 未定义的模糊名词不得进入 design.md，也不得原样抄进 Requirement Coverage 表的"需求条目"列。

## Rules

analysis.md：

允许：

描述已有实现。

禁止：

出现：

- 修改计划

- 新架构设计

- 新代码方案

## Exit Gate

Self Check：

必须回答：

1. 修改入口在哪里？

2. 数据如何流动？

3. 哪些模块受影响？

如果无法回答：

回P0。

## Output

生成：

docs/dev/<feature>/analysis.md

更新：

STATE.md

phase: P2-Design
