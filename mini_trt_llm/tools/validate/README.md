# INT8 判据的验收集规格（离线口径）

> **定位**：这里定义"INT8 怎么算合格"所需的**输入规格**与**报告字段**。
> 目标 / 做法 / 验收判据的**唯一来源**是 `docs/future_iterations.md` §1.6；本目录只把它们落成可执行的东西。
> 本目录**不下载任何数据**——见 §4 的"前置"。

---

## 1. 为什么需要这份规格

当前 INT8 的判据是**分层的**（`docs/phase4_int8_plan.md` §4）：

| 判据 | 现状 | 弱点 |
|---|---|---|
| **主判据**：FP32 有余量子集（`top1 − top2 ≥ 5`）的 top-1 一致率 `≥ 90%` | 实测 12/12 = 100% | 有判别力的样本只有 11~12 张，率的**分辨力**弱 |
| **下界**：整体一致率 `≥ 30%` | 实测 38.3% | 这批图（tiny-imagenet 放大图）FP32 自身摇摆，主要在测测试集噪声 |
| 数值上界 | **未定**（实测 `max_abs ≈ 21.6`，故意不作判据） | 被少数样本放大；且验收集**没有真值标签** → 只能判"与 FP32 是否一致"，判不了"对不对" |

所以本规格要补的三件事，正是 `docs/future_iterations.md` §1.6 的目标：

1. 验收集**带真值标签**，从而能报 top-1 **正确率**（而不只是与 FP32 的一致率）；
2. 每个率都必须带**样本量 n**；
3. 绝对误差只在**有余量**的子集上报 **p50 / p95 / p99 分布**，不报全样本 `max`。

---

## 2. 输入规格

### 2.1 logits dump

脚本**不产生 logits**（那需要 GPU 与引擎），它只消费 dump：

| 项 | 规定 |
|---|---|
| 文件格式 | 裸小端 `float32`，行主序，无头部 |
| 形状 | `[N, num_classes]`，`N` 必须等于 meta 的 `validation_set.num_samples` |
| 命名 | 约定 `*.f32.bin`（与 `models/resnet18/inputs/*.f32.bin` 的既有命名一致） |
| 来源侧 | FP32 引擎与 INT8 引擎**各自跑同一批输入**（同一份预处理张量），先按 batch 分块再拼接 |

### 2.2 meta JSON

沿用 `models/resnet18/*.meta.json` 的既有风格（`script` / 版本 / `sha256` / `shape`），
新增验收集必需字段。**必需字段缺一即拒绝**（不是警告）：

```jsonc
{
  "validation_set": {
    "name": "imagenet-val-subset",          // 必需
    "version": "2026-09-26",                // 必需
    "num_samples": 512,                     // 必需；必须与 logits 行数一致
    "manifest_path": "...",                 // 必需；每行一个样本
    "manifest_sha256": "...",               // 必需
    "source": { "url": "...", "retrieved_utc": "...", "license": "..." },  // 必需
    "preprocessing": {                      // 必需：否则无法复现
      "resize": [224, 224], "layout": "NCHW", "dtype": "float32",
      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
      "pixel_range": [0, 255]
    }
  },
  "calibration_set": {                       // 必需：重叠排除要用它
    "dir": "0_resnet18_onnx/calib_data",
    "num_samples": 500,
    "manifest_path": "...", "manifest_sha256": "..."
  },
  "labels": { "num_classes": 1000, "source": "..." }   // 必需
}
```

### 2.3 样本清单（manifest）

JSON 数组，每个元素：

```jsonc
[{ "file": "val/ILSVRC2012_val_00000001.bin", "sha256": "...", "label": 3 }, ...]
```

- **验收集**清单：`file` / `sha256` / `label` **三者都必需**（缺 `label` 就无法报正确率，缺 `sha256` 就无法做重叠排除）；
- **标定集**清单：`file` / `sha256` 必需，`label` 可缺（标定不需要真值）。

---

## 3. 重叠排除规则（为什么必须做）

**规则**：只要某个验证样本的 **文件名（basename）或 `sha256`** 与标定集任一元素相同 → **整份评估拒绝执行**。

**为什么**：标定集参与过 scale 的定标（`QuantizeLinear` 的 bin 边界由它决定）。
若它同时出现在验收集里，那些样本的量化误差会被系统性低估，一致率 / 正确率都会被**高估**——
这是"指标看起来更好"的典型来源，而不是模型真的更好。

**拒绝而不是"剔除后继续"**：静默剔除会让人误以为评估覆盖了整个验收集；
排除规则属于数据准备阶段该做的事，不该在报告阶段悄悄发生。

---

## 4. 前置与不做的事

### 4.1 前置（需要你批准的外部动作）

真正带真值标签的 ImageNet 验证子集**需要联网获取**（`AGENTS.md` §0.2：联网操作必须单独获批）。
未获批时**可以先做**本目录已有的东西：规格（本文件）、脚本（`int8_eval.py`）、脚本自检、
以及用**合成 logits** 验证分层数学——这些都不需要任何外部数据。

### 4.2 明确不做

- 不做绝对数值的**任意输入保证**：本判据只覆盖该验证集分布内的 top-1 正确率与一致率；
- 不把 INT8 的阈值套到 FP16 / FP32 上（跨精度复用，见 `AGENTS.md` §7）；
- 不用整体一致率当质量指标（它只是"没崩坏"的下界）；
- 不在报告里给出"没有样本量的率"。

---

## 5. 报告字段

`int8_eval.py --json-out` 产出的结构（每个率**必须**带 `n`）：

```jsonc
{
  "thresholds": { "confident_margin": 5.0, "bucket_edges": [1,2,5,10],
                  "provenance": "tests/test_resnet18_int8.cpp + docs/phase4_int8_plan.md §4" },
  "provenance": { "validation_set": {...}, "calibration_set": {...}, "labels": {...} },
  "overall":    { "n": 512, "agree": 196, "agree_rate": 0.383, "correct": ..., "top1_accuracy": ... },
  "confident":  { "n": 12, "agree": 12, "agree_rate": 1.0, ... },
  "strata":     [ { "bucket": "margin<1", "lo": null, "hi": 1.0, "n": 297, "agree": 70, ... }, ... ],
  "max_abs_on_confident": { "n": 12, "p50": ..., "p95": ..., "p99": ... },
  "not_covered": [ "...只覆盖该验证集分布内的分类指标..." ]
}
```

**判读纪律**（`AGENTS.md` §7）：报告里每个阈值都要能回答"凭什么这么定"；
本文件与脚本共同保证"率必带 n"和"阈值必带出处"。
