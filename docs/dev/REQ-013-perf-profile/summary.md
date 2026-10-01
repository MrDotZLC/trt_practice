# Project Summary

> **性质**：历史交付总结回填（2026-10-01）。来源：本目录 `future_iterations_development_plan.md` §11.5.2。

## Problem

性能结论只能靠单次点值，跨 session 不可比；"时间花在哪"没有可归因的粒度，导致优化无从排序。

## Solution

建立"同一 session 内可复现"的测量方法 + 一键采集入口，并用它把解码时间分解到可归因的粒度。

## Architecture

```text
采集入口（不进默认构建）→ 报告落文件 → 分桶脚本（含自检）→ 结论（含偏差 / 构建态 / 条件）
协议：同 session、同轮交替、报中位数与分位、先给判别下限
```

## Implementation

协议成文 → 采集入口（系统级 / kernel 级）→ 分桶脚本 + 自检 → 两次实测（PF-8 采样器占比、PF-9 注意力占比）
→ 结论回填。

## Performance

三条结论（见 `benchmark.md`）：采样器占比（贪心 1.25% / Top-K 17.7% / Top-P 20.1%）；
长上下文下注意力约每步 80%；显存分配**代价不足以支撑优化**（触发未获支持）。

## Limitation

- **本机拿不到 GPU kernel 时间线** → 逐 kernel 分解记为**能力边界**。
- 跨构建对照（PF-7）当时**没跑**，已移交 `docs/dev/REQ-019-onnx-subgraph/` 的前置。

## Future Work

- 长上下文 attention → `docs/dev/REQ-014-attention-splitk/`（已交付）。
- 跨构建对照 PF-7 → 子图替换的前置。
