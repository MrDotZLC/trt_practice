# Requirement

> **性质**：历史需求回填（2026-10-01）。内容从下列来源抽取，**未新增任何当时的决定**。
> **来源**：`docs/dev/REQ-007-resnet18/phase4_development_plan.md` §2 目标与范围 / §6 验收标准 / §8 范围外。
> **判据的另一半**：见本目录 `test_plan.md`（原出处：`docs/dev/REQ-007-resnet18/phase4_test_plan.md`）
> **设计**：见本目录 `design.md`。
> **谁引用这里**：`scripts/ref_resnet18.py`（3 处：PH4-LEGACY-FINDINGS / PH4-LEGACY-INHERIT、
> PH4-TEST-INPUTS、PH4-CRITERIA）、`tools/convert/onnx_to_mini_trt_llm.py`（PH4-ONNX-SOURCE-DECISION）、
> `include/mini_trt_llm/core/builder.hpp`（PH4-CV-PROFILE）。
> **数字与结论**：不在本文重复。
> **状态**：已交付。

## Problem

框架此前只覆盖语言模型。视觉模型是另一条独立的现实需求，且历史工程没有留下可用的外部基线
（它当年的"精度验证"是自相对，输入还是合成数据），因此"数值对齐"这件事缺少标尺。

## Goal

让框架能**独立承载该视觉模型的构建与推理**，数值与外部基线对齐，
且输入/输出/profile 契约与历史工程一致。

## Scope

### Included

| 分类 | 内容 |
|---|---|
| **必做** | 外部参考基线；外部图路径打通；CV 运行时封装；原生构建器 + 权重转换工具；三方对拍；文档收口 |
| **按需求拆出** | 低精度量化见 `docs/dev/REQ-008-int8-qdq/`（2026-10-01 按需求归属拆出） |

### Excluded

- **动态分辨率**（空间维固定，留作开放项）。
- 数据增强 / 后处理（top-k 标签等业务语义）。
- 多模型（只做一个视觉模型）。
- 旧示例工程的处置（当时列为取消，后重新立项）。

## Acceptance Criteria

来源：`docs/dev/REQ-007-resnet18/phase4_development_plan.md` §6（7 条）。

1. 基线可复现（同脚本两次运行逐位一致，SHA256 全一致），元数据完整自洽。
2. 外部图路径能在批量 1 / 8 / 16 上推理成功，超范围批量被拒。
3. 原生路径建图成功，**与外部图路径逐值对拍达标**，两者都与外部基线一致。
4. 逐样本 argmax 在所有路径与精度下一致（低精度见 `docs/dev/REQ-008-int8-qdq/`）。
5. 运行时封装端到端可用；超范围批量、尺寸不符、构造失败均**显式失败**而非静默。
6. 沙箱与真机的测试口径明确（含跳过策略）。
7. **低精度达标** —— 判据与实测见 `docs/dev/REQ-008-int8-qdq/`（2026-10-01 拆出）。

## Constraints

- 基线必须先于一切：没有可复现的外部基线，所有"对齐"结论都无依据。
- 前处理契约先定死，否则数值误差会被误判成"引擎错"，把排查方向带偏。
- 低精度路线必须避开已废弃的隐式校准（见 `docs/dev/REQ-008-int8-qdq/design.md`）。
- 判据不许跨精度复用（视觉侧的低精度阈值不能套到语言侧）。

## 历史演进（抽取时保留，勿按旧值验收）

- **低精度判据的演进**（曾按"分布外输入一致性"判、实测后作废并改为分层一致率）见
  `docs/dev/REQ-008-int8-qdq/requirement.md` 的「历史演进」；过程证据在 `TROUBLESHOOTING.md`。
- 本条目交付期间踩到的"量化尺子量错对象"事件，后来成为一条**通用纪律**（量化尺度必须取自
  被量化那张张量本身），并写入 `PROGRESS.md` §2.18。
