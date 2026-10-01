# Design

> **性质**：历史设计回填（2026-10-01）。从 `docs/dev/REQ-001-bootstrap/phase0_development_plan.md` §2 任务详细说明 / §3 文件清单抽取，
> **未新增当时的决定**。需求见 `requirement.md`，判据与用例见 `test_plan.md`。

## Overview

建立可编译、可测试的骨架，作为后续所有阶段的公共底座。顺序：目录与构建 → 依赖接入 →
基础设施 → 引擎构建骨架 → 基础测试 → 根构建接入。

## Architecture

```text
include/mini_trt_llm/{core,kv_cache,plugins,sampler,tokenizer,utils}/   ← 头文件按模块分层
src/{core,kv_cache,plugins,sampler,tokenizer,utils}/                    ← 实现
tests/                                                                  ← 汇总为单一测试目标
third_party/                                                            ← 源码嵌入的依赖
```

构建目标为**静态库**；CUDA 与 C++ 均按 C++17 编译；目标架构固定为 `sm_75`。

## Module Design

| 模块 | 职责 | 依赖 |
|---|---|---|
| 目录与 CMake（P0-1） | 目录结构、库目标、测试目标 | — |
| 第三方依赖（P0-2） | 分词库以**静态**方式嵌入并关掉非必要功能；张量文件解析库源码嵌入 | 构建系统 |
| 基础设施（P0-3） | CUDA / TensorRT 错误检查宏、日志、文件 IO、计时、显存池 | — |
| 引擎骨架（P0-4） | 精度枚举、引擎封装、构建入口、模型配置、权重加载、注册表与构建器抽象 | 基础设施 |
| 基础测试（P0-5） | 测试目标与首批判据 | 以上全部 |
| 根构建接入（P0-6） | 根 `CMakeLists.txt` 接入模块 | — |

## Data Structure

```text
Precision { FP32, FP16, INT8 }        ← 与运行时数据类型一一对应
ModelConfig                           ← 由 config.json 解析出的模型描述
IModelBuilder                         ← 抽象：Name() + Build(network, weights, config, options)
ModelRegistry                         ← 名称 → 构建器
```

## Runtime Flow

```text
config.json → ModelConfig::Load → ModelRegistry::Get → IModelBuilder::Build
      → buildSerializedNetwork → 序列化落盘 → 运行时加载 → 绑定 → 推理
```

## Resource Lifecycle

- 显存与固定内存缓冲用 RAII 封装，析构即释放。
- 引擎为可序列化产物：构建 → 落盘 → 由运行时反序列化。
- 第三方依赖以源码嵌入，生命周期随构建产物，不依赖宿主机安装。

## Performance Consideration

骨架阶段**不涉及**性能设计；当时的取舍全部围绕"可编译、可测、可扩展"。

## Trade-off

| 决策 | 选择 | 为什么 |
|---|---|---|
| 库形态 | **静态库** | 便于嵌入与分发，避免运行时依赖污染 |
| 依赖接入 | **源码嵌入 / 静态** | 与"换机器要能重建"的项目约束一致；不被宿主机安装左右 |
| 模型分发 | **注册表 + 抽象接口** | 新增模型不改构建入口——这条后来被写成可执行判据（见 `test_plan.md`） |
| 判据写法 | **可执行判据** | 初版"文件存在即通过"让只声明未定义的方法也算过，动态 shape 能力实际不可用 |
