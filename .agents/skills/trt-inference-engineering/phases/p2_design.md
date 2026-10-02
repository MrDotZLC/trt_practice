# Phase P2 Design

## Purpose

设计实现方案。

输出：

可执行设计文档。

## Entry

读取：

requirement.md

analysis.md

## Actions

生成：

design.md

## Design Requirements

必须包含：

### Architecture

说明：

新增模块和关系。

### Module Design

格式：

| Module | Responsibility | Dependency |
||||

### Data Structure

说明：

核心结构。

例如：

- KV Cache block
- Request state
- Tensor metadata

### Runtime Flow

必须包含：

ASCII流程图。

例如：

Request  
↓  
Scheduler  
↓  
Engine Context  
↓  
Kernel  
↓  
Output

### Resource Lifecycle

说明：

- CUDA memory
- TensorRT object
- Stream
- Event

生命周期。

### Performance Consideration

说明：

- compute
- memory
- communication

### Trade-off

至少包含：

两个方案比较。

### Requirement Coverage

必须包含覆盖表：

| 需求条目 | 设计落点（章节） | 交付里程碑 | 验证 Phase |
|---|---|---|---|

规则：

- `requirement.md` 的每条 Included 与每条 Acceptance Criteria 都必须有一行。
- 落点必须是本设计的章节名，不接受只写里程碑名或需求原话。
- 未纳入本轮的条目必须写进 requirement 的 Excluded 并给出理由，禁止只写"后续再说"。
- 只有名字、没有流程 / 状态机 / 判据的落点，视为未覆盖。
- 引入里程碑分期时，每个里程碑必须至少承担一条需求条目的完整交付；分期不得把需求条目悬空。

## Rules

设计阶段：

禁止修改源码。

## Exit Gate

Self Check：

检查：

- Requirement Coverage 表存在，且无空落点
- 是否明确接口
- 是否说明资源生命周期
- 是否说明性能影响

Human Gate：

无。设计确认与评审结论一起在 P3 Review 之后的 Gate-A 处理。

## Output

生成：

docs/dev/<feature>/design.md

更新 STATE.md 字段：

- phase: P3-Review
- status: in-progress
