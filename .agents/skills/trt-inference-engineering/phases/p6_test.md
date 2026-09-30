# Phase P6 Test

## Purpose

验证功能正确性。

禁止：

功能未验证直接benchmark。

# Entry

读取：

implementation changes

# Actions

创建：

test_plan.md

# Test Levels

## Unit Test

验证：

- 单函数
- Kernel
- Plugin

## Integration Test

验证：

- Engine build
- Runtime execution
- End-to-end flow

## Regression Test

验证：

已有功能未损坏。

## Failure Test

验证：

异常输入。

例如：

- empty input
- invalid shape
- memory limit

# Exit Gate

Self Check:

必须满足：

- Unit Test通过
- Integration Test通过
- Regression Test通过
- Failure Case通过

## Hard Rule

P6未通过：

禁止进入P7 Benchmark。

# Failure Route

测试失败：

返回P5。

连续2次失败：

暂停，请求用户介入。

# Output

生成：

test_plan.md

更新：

STATE:

phase: P7-Benchmark
