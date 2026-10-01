# Project Summary

> **性质**：历史交付总结回填（2026-10-01）。来源：本目录 `phase3_development_plan.md` §4 / §5。

## Problem

框架只有原生一条路径；外部图这条现实存在的模型来源完全没被纳入，且缺少"图变了要知道"的护栏。

## Solution

把外部图路径打通（动态 profile、精度、失败可诊断、输出名校验），与原生路径交叉验证，
并用图结构探针把"图变了"变成自动化告警。

## Architecture

```text
外部图 → 解析 → I/O 契约校验（按 architecture 分支）→ profile（只挂前向一组）→ 引擎
原生路径 ————————————————————————————┘
              ↓ 同一输入、同一口径对拍
          图结构探针（进 ctest）
```

## Implementation

探针固化 → 构建入口落地 → 数值对齐用例 → 子图识别与断言 → 性能对照（未定）→ 文档收口。

## Performance

**结论 = 未定**：两次测量方向相反、差异小于构建间噪声。详见 `benchmark_before.md` 与 `benchmark.md`。

## Limitation

- 子图替换**只做识别 + 断言**（决策 D1=C），不做真替换——可替换面在本模型上收益为零或未量化。
- 外部图**不支持解码**（该图无 past 输入）。
- 两个缺口转开放项：子图识别只到计数级、性能测量方法缺失。

## Future Work

- 可复现测量方法 → `docs/dev/REQ-013-perf-profile/`（已交付）。
- 子图替换的前置对照（PF-7）→ `docs/dev/REQ-019-onnx-subgraph/`。
