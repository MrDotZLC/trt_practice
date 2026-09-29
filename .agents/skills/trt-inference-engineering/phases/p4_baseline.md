# Phase P4 Baseline

## Purpose

在修改代码前建立性能基线。



# Entry

条件：

design通过。

review无Blocker。



# Rule

如果改动涉及性能：

必须执行Baseline。

Baseline必须发生在：


Phase P5 Implementation  
之前





# Actions

运行当前版本。

记录：

## Environment

必须包含：

- Git commit hash
- GPU型号
- CUDA版本
- TensorRT版本



## Workload

必须固定：

- batch size
- sequence length
- dtype
- input shape



## Measurement

要求：

Warmup：


> =10



Iteration：


> =100



统计：

- median
- P95



# Output

生成：


benchmark_before.md





# Failure Handling

## Timeout / OOM

调整workload。

必须记录原因。



## Dependency Missing

例如：

无GPU。

写：


N/A: reason



P7改为外部参考比较。



## Continuous Failure

连续3次失败：

暂停。

请求用户介入。



# Exit Gate

Self Check:

benchmark_before.md包含：

- environment
- workload
- measurement



Human Gate:

无。



# Output

更新：

STATE:


phase: P5-Implementation