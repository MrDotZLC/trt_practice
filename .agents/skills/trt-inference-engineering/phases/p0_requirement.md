# Phase P0 Requirement

## Purpose

明确功能目标、范围和验收标准。

本阶段只回答：

- 为什么做
- 要解决什么问题
- 成功标准是什么

禁止进入：

- 文件修改
- 架构设计
- 代码方案

## Entry

读取：

用户需求。

如果已有状态：

读取：

docs/dev//STATE.md

## Actions

### 1. 创建feature目录

创建：

docs/dev//

### 2. 创建requirement.md

模板：

templates/requirement.md

### 3. 明确需求

记录：

#### Problem

当前存在的问题。

#### Goal

希望达到的目标。

#### Scope

包含：

- 功能范围
- 不包含范围

#### Acceptance Criteria

必须可验证。

例如：

- 支持dynamic batch
- 单元测试通过
- latency降低

## Rules

requirement.md：

允许：

描述用户目标。

禁止：

出现：

- cpp文件名
- class名称
- 函数名
- 修改方案

需求文档不能被已有代码限制。

## Exit Gate

### Self Check

确认：

- requirement.md存在
- Problem明确
- Goal明确
- Acceptance Criteria可测试

### Human Gate

触发条件：

以下任意成立：

- 目标存在多个解释
- 涉及架构级变化
- 验收标准无法确定

否则：

自动进入P1。

## Output

生成：

docs/dev//requirement.md

更新：

STATE.md

workflow: feature  
phase: P1-Analysis
