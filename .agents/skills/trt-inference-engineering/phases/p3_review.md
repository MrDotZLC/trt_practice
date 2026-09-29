# Phase P3 Review

## Purpose

对设计进行工程审查。



# Entry

读取：


design.md



加载：



checklists/cpp.md

checklists/cuda.md

checklists/tensorrt.md

checklists/llm_runtime.md





# Actions

生成：


review.md





# Review Classification

## P0 Blocker

必须修复。

例如：

- TensorRT Context生命周期错误
- CUDA memory ownership不明确
- 数据竞争

结果：

阻止进入P4。



## P1 Risk

需要人工确认。

例如：

- 性能风险
- 异常恢复不足



## P2 Quality

记录即可。

例如：

- 命名风格
- 文档完善



# Exit Gate

## Self Check

review.md必须包含：

| Issue | Level | Action |
||||



## Human Gate

Gate-B规则：

### P0存在

暂停。

### P0不存在，P1存在

暂停。

### P0/P1不存在

自动进入P4。



# Failure Route

P0失败：

返回P2。

设计重新修改。



# Output

生成：


docs/dev//review.md



更新：

STATE。