# Bugfix Workflow

## Purpose

用于：

- 修复已有错误
- 修复崩溃
- 修复结果异常
- 修复性能回退

核心原则：

最小修改。

# Workflow State Machine

B0 Reproduce

↓

B1 Diagnose

↓

B2 Minimal Fix

↓

B3 Regression

↓

B4 Summary

# B0 Reproduce

目标：

确认问题真实存在。

必须产生：

failure test

*

error log

要求：

failure test必须可以运行。

记录：

- 输入条件
- 运行环境
- 错误输出
- 复现步骤

禁止：

未复现直接修改代码。

# B1 Diagnose

目标：

定位根因。

必须分析：

- 调用链
- 数据流
- 生命周期
- 错误位置

禁止：

直接修改。

# B2 Minimal Fix

目标：

最小范围修复。

默认限制：

Files <=3

Lines <=100

如果预计超过：

触发：

Bugfix → Feature Decision。

禁止自动转换。

# B3 Regression

必须验证：

1. 原问题消失。

2. 原有功能正常。

3. 新修改没有引入回归。

# B4 Summary

输出：

summary.md

包含：

- Root Cause
- Fix
- Test Result
- Limitation

# Forbidden

Bugfix禁止：

- 架构重构
- 新增功能
- API重新设计
- 大规模代码整理

如果需要：

创建Feature Workflow。
