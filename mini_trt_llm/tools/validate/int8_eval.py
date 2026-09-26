#!/usr/bin/env python3
"""INT8 判据评估：把"一致率"升级为"带样本量、带真值标签、带分布"的报告。

为什么需要它（背景与判据口径的唯一来源是 `docs/future_iterations.md` §1.6）：
当前 INT8 的主判据是"FP32 有余量子集的一致率"，它有三个已知弱点——
有判别力的样本只有 11~12 张、验收集没有真值标签（只能判"与 FP32 是否一致"、
判不了"对不对"）、绝对误差界未定。本脚本负责把这三件事变成可复现的报告。

它**不产生 logits**（那需要 GPU 与引擎），只消费 dump；规格见同目录 README.md。

用法：
    python3 int8_eval.py --fp32-logits fp32.f32.bin --int8-logits int8.f32.bin \
        --meta meta.json --calib-dir 0_resnet18_onnx/calib_data --json-out report.json
    python3 int8_eval.py --self-test        # 护栏自证 + 分层数学自证（不需要任何外部数据）
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from array import array

# 阈值出处：tests/test_resnet18_int8.cpp（kConfidentMargin / kBucketEdges）与
# docs/phase4_int8_plan.md §4。**改动它们等于改判据**，所以报告里必须回显出处。
DEFAULT_CONFIDENT_MARGIN = 5.0
DEFAULT_BUCKET_EDGES = (1.0, 2.0, 5.0, 10.0)
THRESHOLD_PROVENANCE = "tests/test_resnet18_int8.cpp + docs/phase4_int8_plan.md §4"

# 报告里必须写清的"本判据不覆盖什么"（§1.6 验收判据第 3 条）。
NOT_COVERED = [
    "只覆盖该验证集分布内的分类一致率与 top-1 正确率，不构成对任意输入的数值保证",
    "不覆盖 FP16 / FP32：INT8 的阈值不跨精度复用",
    "整体一致率只是'没崩坏'的下界，不是质量指标（这批图的 FP32 自身摇摆）",
]


class SpecError(Exception):
    """输入不满足规格。规格不满足时**拒绝执行**，不降级成警告。"""


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SpecError("文件不存在：{}".format(path))
    except json.JSONDecodeError as exc:
        raise SpecError("JSON 解析失败：{}（{}）".format(path, exc))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_keys(obj, keys, where):
    """必需字段缺一即拒绝——"警告 + 继续"会让报告看起来有效但其实不可复核。"""
    if not isinstance(obj, dict):
        raise SpecError("{} 应为对象，实际是 {}".format(where, type(obj).__name__))
    missing = [k for k in keys if k not in obj or obj[k] in (None, "")]
    if missing:
        raise SpecError("{} 缺必需字段：{}".format(where, ", ".join(missing)))


def require_number(obj, key, where):
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError("{} 的 {} 必须是数字，实际是 {!r}".format(where, key, value))
    return value


def validate_meta(meta, calib_dir=None, legacy_mode=False):
    """校验验收集 meta 的必需字段，返回归一化后的拷贝。

    `calib_dir` 非空表示标定集清单由命令行现场枚举（此时不再要求 meta 里给 manifest_path）——
    这个分支必须在**校验期**就知道，不能等校验后再补，否则合法的调用会被误拒。
    `legacy_mode` 用于历史口径交叉校验：允许清单缺 `manifest_sha256`（由本脚本现算），
    其余必需字段照旧——legacy 也不能省略"验的是什么"。
    """
    require_keys(meta, ["validation_set", "calibration_set", "labels"], "meta")
    validation = meta["validation_set"]
    require_keys(
        validation,
        ["name", "version", "num_samples", "manifest_path", "source", "preprocessing"],
        "meta.validation_set",
    )
    if not legacy_mode:
        require_keys(validation, ["manifest_sha256"], "meta.validation_set")
    # num_samples 用 require_number 而不是 int()：字符串 "512" 会被静默接受，那正是要拦的输入。
    num_samples = require_number(validation, "num_samples", "meta.validation_set")
    if int(num_samples) != num_samples or num_samples <= 0:
        raise SpecError("meta.validation_set.num_samples 必须是正整数，实际是 {!r}".format(num_samples))
    require_keys(validation["preprocessing"], ["resize", "layout", "dtype", "mean", "std"], "meta.validation_set.preprocessing")
    require_keys(validation["source"], ["url", "retrieved_utc", "license"], "meta.validation_set.source")

    calibration = meta["calibration_set"]
    require_keys(calibration, ["dir", "num_samples"], "meta.calibration_set")
    # 标定集清单允许两种来源：meta 内的 manifest，或命令行 --calib-dir 现场枚举。
    if not calib_dir and "manifest_path" not in calibration:
        raise SpecError("meta.calibration_set 需要 manifest_path（或用 --calib-dir 现场枚举）")

    require_keys(meta["labels"], ["num_classes", "source"], "meta.labels")
    return meta


def load_manifest(path, need_label, where, legacy_mode=False):
    entries = load_json(path)
    if not isinstance(entries, list) or not entries:
        raise SpecError("{} 应为非空数组：{}".format(where, path))
    base = os.path.dirname(os.path.abspath(path))
    for idx, entry in enumerate(entries):
        require_keys(entry, ["file"], "{}[{}]".format(where, idx))
        if "sha256" not in entry or entry["sha256"] in (None, ""):
            # 只有 legacy 允许缺哈希：此时由本脚本读原文件现算并写回 entry，
            # 报告的 provenance 里就能看到"验的到底是哪份数据"。
            if not legacy_mode:
                raise SpecError("{}[{}] 缺 sha256（严格模式必需；历史口径请用 --legacy-mode）".format(where, idx))
            resolved = entry["file"] if os.path.isabs(entry["file"]) else os.path.join(base, entry["file"])
            if not os.path.isfile(resolved):
                raise SpecError("{}[{}] 的文件不可读，无法现算 sha256：{}".format(where, idx, resolved))
            entry["sha256"] = sha256_file(resolved)
        if need_label and "label" not in entry:
            raise SpecError("{}[{}] 缺 label：没有真值标签就无法回答'对不对'".format(where, idx))
    return entries


def manifest_from_dir(directory):
    """现场枚举一个目录，产出 {file, sha256} 清单（标定集用）。"""
    if not os.path.isdir(directory):
        raise SpecError("目录不存在：{}".format(directory))
    entries = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if os.path.isfile(full) and name.endswith(".bin"):
            entries.append({"file": name, "sha256": sha256_file(full)})
    if not entries:
        raise SpecError("目录里没有 .bin：{}".format(directory))
    return entries


def overlap_violations(validation_entries, calibration_entries):
    """按文件名或 sha256 判重叠。重叠即拒绝，不静默剔除——静默剔除会让报告看起来覆盖了整个验收集。"""
    calib_names = {os.path.basename(e["file"]) for e in calibration_entries}
    calib_hashes = {e["sha256"] for e in calibration_entries}
    hits = []
    for entry in validation_entries:
        name = os.path.basename(entry["file"])
        if name in calib_names:
            hits.append("{}（文件名命中标定集）".format(name))
        elif entry["sha256"] in calib_hashes:
            hits.append("{}（sha256 命中标定集）".format(name))
    return hits


def read_logits(path, rows, cols):
    """读裸 float32 行主序矩阵；长度不符即拒绝（读错形状会安静地给出错误的率）。"""
    with open(path, "rb") as handle:
        raw = handle.read()
    expected = rows * cols * 4
    if len(raw) != expected:
        raise SpecError(
            "logits 大小不符：{} 实际 {} 字节，按 num_samples={} × num_classes={} 应为 {}".format(
                path, len(raw), rows, cols, expected
            )
        )
    flat = array("f")
    flat.frombytes(raw)
    if sys.byteorder != "little":  # dump 约定是小端
        flat.byteswap()
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


def argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def top1_margin(row):
    order = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
    return row[order[0]], row[order[0]] - row[order[1]]


def percentile(sorted_values, fraction):
    """nearest-rank 分位：n 小的时候 p95/p99 会等于 max，这是**要写进报告的已知限制**。"""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def summarize(samples, bucket, lo, hi):
    n = len(samples)
    agree = sum(1 for s in samples if s["agree"])
    labeled = [s for s in samples if s["correct"] is not None]
    summary = {
        "bucket": bucket,
        "lo": lo,
        "hi": hi,
        "n": n,
        "agree": agree,
        "agree_rate": (agree / n) if n else None,
    }
    if labeled:
        correct = sum(1 for s in labeled if s["correct"])
        summary["n_labeled"] = len(labeled)
        summary["correct"] = correct
        summary["top1_accuracy"] = correct / len(labeled)
    else:
        summary["n_labeled"] = 0
        summary["correct"] = None
        summary["top1_accuracy"] = None
    return summary


def stratified_report(fp32_rows, int8_rows, labels, edges, confident_margin):
    samples = []
    for i, (fp32_row, int8_row) in enumerate(zip(fp32_rows, int8_rows)):
        top1_value, margin = top1_margin(fp32_row)
        del top1_value
        pred_int8 = argmax(int8_row)
        correct = None if labels is None else int(pred_int8 == labels[i])
        samples.append(
            {
                "index": i,
                "margin": margin,
                "agree": pred_int8 == argmax(fp32_row),
                "correct": correct,
                "max_abs": max(abs(a - b) for a, b in zip(fp32_row, int8_row)),
            }
        )

    strata = []
    lower = None
    for edge in list(edges) + [None]:
        if edge is None:
            bucket = samples if lower is None else [s for s in samples if s["margin"] >= lower]
            name = "margin>={}".format(lower) if lower is not None else "margin<{}".format(edges[0])
            strata.append(summarize(bucket, name, lower, None))
            break
        bucket = [s for s in samples if (lower is None or s["margin"] >= lower) and s["margin"] < edge]
        name = "margin<{}".format(edge) if lower is None else "{}<=margin<{}".format(lower, edge)
        strata.append(summarize(bucket, name, lower, edge))
        lower = edge

    confident = [s for s in samples if s["margin"] >= confident_margin]
    report = {
        "thresholds": {
            "confident_margin": confident_margin,
            "bucket_edges": list(edges),
            "provenance": THRESHOLD_PROVENANCE,
        },
        "overall": summarize(samples, "all", None, None),
        "confident": summarize(confident, "margin>={}".format(confident_margin), confident_margin, None),
        "strata": strata,
        "not_covered": list(NOT_COVERED),
    }
    abs_values = sorted(s["max_abs"] for s in confident)
    report["max_abs_on_confident"] = {
        "n": len(abs_values),
        "p50": percentile(abs_values, 0.50),
        "p95": percentile(abs_values, 0.95),
        "p99": percentile(abs_values, 0.99),
        "note": "分位数取 nearest-rank；n 小时 p95/p99 会等于 max",
    }
    return report


def require_sample_counts(report):
    """护栏：任何"率"都必须带样本量与分子，且三者自洽。只报率不报 n 的结论不可复核。"""
    sections = [("overall", report.get("overall")), ("confident", report.get("confident"))]
    sections += [("strata[{}]".format(i), s) for i, s in enumerate(report.get("strata", []))]
    for where, section in sections:
        if not isinstance(section, dict):
            raise SpecError("{} 缺失".format(where))
        if "agree_rate" not in section:
            raise SpecError("{} 没有 agree_rate".format(where))
        for key in ("n", "agree"):
            if not isinstance(section.get(key), int):
                raise SpecError("{} 的 {} 必须是整数（只报率不报 n 不可接受）".format(where, key))
        if section["agree"] > section["n"]:
            raise SpecError("{} 的分子 {} 大于分母 {}".format(where, section["agree"], section["n"]))
        if section["n"] > 0:
            expected = section["agree"] / section["n"]
            if abs(section["agree_rate"] - expected) > 1e-9:
                raise SpecError(
                    "{} 的率与分子分母不自洽：{} != {}/{}".format(where, section["agree_rate"], section["agree"], section["n"])
                )


def render_text(report):
    lines = []
    thresholds = report["thresholds"]
    lines.append("[INT8 评估] 阈值出处：{}".format(thresholds["provenance"]))
    lines.append("    置信余量 = {}，分桶边界 = {}".format(thresholds["confident_margin"], thresholds["bucket_edges"]))
    for key in ("overall", "confident"):
        section = report[key]
        lines.append(
            "    {:<10} n={:<5} agree={:<5} rate={}".format(
                key, section["n"], section["agree"], _fmt_rate(section)
            )
        )
        if section.get("top1_accuracy") is not None:
            lines.append("        真值标签：n={} correct={} top1={:.4f}".format(section["n_labeled"], section["correct"], section["top1_accuracy"]))
    lines.append("    按 FP32 余量分层（率旁边必须能看见 n）：")
    for stratum in report["strata"]:
        lines.append("        {:<22} n={:<5} agree={:<5} rate={}".format(stratum["bucket"], stratum["n"], stratum["agree"], _fmt_rate(stratum)))
    stats = report["max_abs_on_confident"]
    lines.append("    余量子集 max_abs：n={} p50={} p95={} p99={}（{}）".format(
        stats["n"], _fmt_num(stats["p50"]), _fmt_num(stats["p95"]), _fmt_num(stats["p99"]), stats["note"]))
    if report.get("label_note"):
        lines.append("    ⚠ {}".format(report["label_note"]))
    return "\n".join(lines)


def _fmt_rate(section):
    if section["agree_rate"] is None:
        return "n/a（该层无样本）"
    return "{:.4f}（{}/{}）".format(section["agree_rate"], section["agree"], section["n"])


def _fmt_num(value):
    return "n/a" if value is None else "{:.6g}".format(value)


def evaluate(args):
    legacy = bool(getattr(args, "legacy_mode", False))
    meta = validate_meta(load_json(args.meta), calib_dir=args.calib_dir, legacy_mode=legacy)
    validation = meta["validation_set"]
    calibration = meta["calibration_set"]

    if args.calib_dir:
        calib_entries = manifest_from_dir(args.calib_dir)
        calibration["_runtime_manifest"] = {
            "source": "命令行 --calib-dir 现场枚举",
            "dir": args.calib_dir,
            "num_samples": len(calib_entries),
        }
    else:
        calib_entries = load_manifest(calibration["manifest_path"], need_label=False, where="calibration manifest")

    manifest_path = validation["manifest_path"]
    if not os.path.isabs(manifest_path) and args.meta_root:
        manifest_path = os.path.join(args.meta_root, manifest_path)
    # 记了 sha256 就必须验它——"要求提供哈希却不核对"只是装饰，不是护栏。
    # legacy 模式允许 meta 不带清单哈希（交叉校验的清单是 C++ 现场生成的），此时记下实算值。
    actual_sha = sha256_file(manifest_path)
    if not legacy:
        if actual_sha != validation["manifest_sha256"]:
            raise SpecError(
                "验收集清单 sha256 不符：meta 记 {}，实际 {}（清单被改过，或 meta 与清单不同步）".format(
                    validation["manifest_sha256"], actual_sha
                )
            )
    validation_entries = load_manifest(
        manifest_path,
        need_label=not (args.allow_missing_labels or legacy),
        legacy_mode=legacy,
        where="validation manifest",
    )
    if len(validation_entries) != int(validation["num_samples"]):
        raise SpecError(
            "manifest 条目数 {} 与 meta.validation_set.num_samples {} 不一致".format(
                len(validation_entries), int(validation["num_samples"])
            )
        )
    if not (args.allow_missing_labels or legacy):
        labels = [int(e["label"]) for e in validation_entries]
    else:
        labels = None

    hits = overlap_violations(validation_entries, calib_entries)
    if hits and not legacy:
        raise SpecError(
            "验收集与标定集重叠 {} 个样本，拒绝执行（标定集污染验证集会让一致率被高估）：{}".format(
                len(hits), "; ".join(hits[:5]) + ("..." if len(hits) > 5 else "")
            )
        )

    rows = int(validation["num_samples"])
    cols = int(meta["labels"]["num_classes"])
    fp32_rows = read_logits(args.fp32_logits, rows, cols)
    int8_rows = read_logits(args.int8_logits, rows, cols)

    edges = tuple(float(x) for x in args.bucket_edges.split(",")) if args.bucket_edges else DEFAULT_BUCKET_EDGES
    confident_margin = args.confident_margin if args.confident_margin is not None else DEFAULT_CONFIDENT_MARGIN
    report = stratified_report(fp32_rows, int8_rows, labels, edges, confident_margin)
    # legacy 模式下 meta 允许不带清单哈希，此时用现算值回填——provenance 里必须能查到"验的是哪份清单"。
    manifest_provenance = {k: validation[k] for k in ("name", "version", "num_samples")}
    manifest_provenance["manifest_path"] = manifest_path
    manifest_provenance["manifest_sha256"] = validation.get("manifest_sha256", actual_sha)
    report["provenance"] = {
        "validation_set": manifest_provenance,
        "calibration_set": calibration.get("_runtime_manifest") or {
            "manifest_path": calibration["manifest_path"],
            "manifest_sha256": calibration["manifest_sha256"],
            "num_samples": calibration["num_samples"],
        },
        "labels": meta["labels"],
    }
    if labels is None:
        report["label_note"] = "本报告不含真值标签：只回答'与 FP32 是否一致'，不回答'对不对'（不满足 §1.6 规格）"
        report["spec_compliance"] = {"requires_truth_labels": False, "ref": "docs/future_iterations.md §1.6"}
    else:
        report["spec_compliance"] = {"requires_truth_labels": True, "ref": "docs/future_iterations.md §1.6"}
    if legacy:
        report["spec_compliance"].update(
            {
                "legacy_mode": True,
                "calib_overlap_violations": len(hits),
                "manifest_sha256_computed": actual_sha,
                "note": (
                    "**历史口径交叉校验**：本报告的验收集与标定集同源（重叠 {} 个样本），"
                    "因此一致率会被系统性高估；它**不满足 §1.6 规格**，只用于与 C++ 侧的现役统计对齐口径。".format(len(hits))
                ),
            }
        )
    else:
        report["spec_compliance"]["legacy_mode"] = False
    require_sample_counts(report)
    return report


# ---------------------------------------------------------------------------
# 自检：护栏必须有用例证明它会拦人（AGENTS.md §2.13），分层数学必须自证。
# ---------------------------------------------------------------------------


def _write_logits(path, rows):
    flat = array("f")
    for row in rows:
        flat.extend(row)
    with open(path, "wb") as handle:
        handle.write(flat.tobytes())


def _synthetic_case(tmp):
    """构造已知答案的合成数据：每个样本的余量、是否一致、是否正确都是设计好的。"""
    fp32 = [
        [10.0, 0.0, 0.0, 0.0],    # margin 10 → 置信；INT8 一致
        [6.0, 4.0, 0.0, 0.0],     # margin 2  → 桶 [2,5)；一致
        [3.0, 2.5, 0.0, 0.0],     # margin 0.5→ 桶 <1；INT8 翻成 class 1
        [5.5, 4.0, 0.0, 0.0],     # margin 1.5→ 桶 [1,2)；一致
        [12.0, 1.0, 0.0, 0.0],    # margin 11 → 置信；INT8 翻成 class 3
        [2.0, 1.5, 0.0, 0.0],     # margin 0.5→ 桶 <1；一致
    ]
    int8 = [
        [10.25, 0.0, 0.0, 0.0],
        [6.0, 4.0, 0.0, 0.0],
        [2.5, 3.0, 0.0, 0.0],
        [5.5, 4.0, 0.0, 0.0],
        [6.0, 1.0, 0.0, 12.0],
        [2.0, 1.5, 0.0, 0.0],
    ]
    labels = [0, 0, 0, 0, 0, 0]

    fp32_path = os.path.join(tmp, "fp32.f32.bin")
    int8_path = os.path.join(tmp, "int8.f32.bin")
    _write_logits(fp32_path, fp32)
    _write_logits(int8_path, int8)

    val_manifest = os.path.join(tmp, "val.json")
    with open(val_manifest, "w", encoding="utf-8") as handle:
        json.dump(
            [{"file": "val_{}.bin".format(i), "sha256": "v{:02d}".format(i), "label": labels[i]} for i in range(len(labels))],
            handle,
        )
    calib_manifest = os.path.join(tmp, "calib.json")
    with open(calib_manifest, "w", encoding="utf-8") as handle:
        json.dump([{"file": "calib_{}.bin".format(i), "sha256": "c{:02d}".format(i)} for i in range(3)], handle)

    base_meta = {
        "validation_set": {
            "name": "synthetic",
            "version": "self-test",
            "num_samples": len(labels),
            "manifest_path": val_manifest,
            "manifest_sha256": sha256_file(val_manifest),
            "source": {"url": "n/a", "retrieved_utc": "n/a", "license": "n/a"},
            "preprocessing": {
                "resize": [224, 224], "layout": "NCHW", "dtype": "float32",
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            },
        },
        "calibration_set": {
            "dir": tmp,
            "num_samples": 3,
            "manifest_path": calib_manifest,
            "manifest_sha256": sha256_file(calib_manifest),
        },
        "labels": {"num_classes": 4, "source": "synthetic"},
    }
    meta_path = os.path.join(tmp, "meta.json")

    def write_meta(meta):
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        return meta_path

    return {
        "fp32": fp32_path, "int8": int8_path, "meta": write_meta(base_meta),
        "base_meta": base_meta, "write_meta": write_meta, "tmp": tmp, "val_manifest": val_manifest,
    }


def _args_for(case):
    return argparse.Namespace(
        fp32_logits=case["fp32"], int8_logits=case["int8"], meta=case["meta"], calib_dir=None,
        meta_root=None, allow_missing_labels=False, bucket_edges=None, confident_margin=None, json_out=None,
        self_test=False, legacy_mode=False,
    )


def _expect_spec_error(label, fn):
    try:
        fn()
    except SpecError:
        print("  [PASS] {}（被拒绝）".format(label))
        return
    raise AssertionError("{} 本应被拒绝，但通过了".format(label))


def self_test():
    print("[int8_eval --self-test]")
    with tempfile.TemporaryDirectory() as tmp:
        case = _synthetic_case(tmp)
        report = evaluate(_args_for(case))

        overall = report["overall"]
        assert overall["n"] == 6 and overall["agree"] == 4, overall
        assert abs(overall["agree_rate"] - 4 / 6) < 1e-9, overall
        assert overall["correct"] == 4 and abs(overall["top1_accuracy"] - 4 / 6) < 1e-9, overall
        confident = report["confident"]
        assert confident["n"] == 2 and confident["agree"] == 1, confident
        print("  [PASS] 整体与置信子集的 n/率/正确率符合构造")

        by_bucket = {s["bucket"]: s for s in report["strata"]}
        assert by_bucket["margin<1.0"]["n"] == 2 and by_bucket["margin<1.0"]["agree"] == 1, by_bucket
        assert by_bucket["1.0<=margin<2.0"]["n"] == 1, by_bucket
        assert by_bucket["2.0<=margin<5.0"]["n"] == 1, by_bucket
        assert by_bucket["5.0<=margin<10.0"]["n"] == 0, by_bucket
        assert by_bucket["margin>=10.0"]["n"] == 2 and by_bucket["margin>=10.0"]["agree"] == 1, by_bucket
        print("  [PASS] 分层分桶的 n 与分子符合构造")

        stats = report["max_abs_on_confident"]
        # s0 的逐样本 max_abs = |10.25-10| = 0.25；s4 被翻到 class 3（12.0 vs 0.0）→ 12.0。
        # p95/p99 在 n=2 时都等于 max，这正是 report 里 note 写明的已知限制。
        assert stats["n"] == 2 and abs(stats["p50"] - 0.25) < 1e-6 and abs(stats["p99"] - 12.0) < 1e-6, stats
        assert abs(stats["p95"] - 12.0) < 1e-6, stats
        print("  [PASS] 余量子集 max_abs 的分位数符合构造（p50=0.25, p95=p99=12.0，n=2 的已知限制）")

        # 护栏 1：缺 provenance（manifest_sha256）
        def missing_provenance():
            meta = json.loads(json.dumps(case["base_meta"]))
            del meta["validation_set"]["manifest_sha256"]
            case["write_meta"](meta)
            try:
                evaluate(_args_for(case))
            finally:
                case["write_meta"](case["base_meta"])

        _expect_spec_error("缺 manifest_sha256", missing_provenance)

        # 护栏 2：与标定集重叠（文件名命中）
        def overlap():
            meta = json.loads(json.dumps(case["base_meta"]))
            with open(case["val_manifest"], "w", encoding="utf-8") as handle:
                json.dump([{"file": "calib_0001.bin", "sha256": "v00", "label": 0}], handle)
            meta["validation_set"]["num_samples"] = 1
            meta["validation_set"]["manifest_sha256"] = sha256_file(case["val_manifest"])
            case["write_meta"](meta)
            try:
                evaluate(_args_for(case))
            finally:
                # 整个用例重建（清单与 fp32/int8 logits 都回到自检的构造），避免残留影响后续护栏
                case.update(_synthetic_case(tmp))

        _expect_spec_error("验收集与标定集重叠", overlap)

        # 护栏 3：字段类型错（num_samples 是字符串）
        def wrong_type():
            meta = json.loads(json.dumps(case["base_meta"]))
            meta["validation_set"]["num_samples"] = "6"
            case["write_meta"](meta)
            try:
                evaluate(_args_for(case))
            finally:
                case["write_meta"](case["base_meta"])

        _expect_spec_error("num_samples 是字符串", wrong_type)

        # 护栏 4：清单被改过（sha256 不符）
        def manifest_tampered():
            meta = json.loads(json.dumps(case["base_meta"]))
            with open(case["val_manifest"], "a", encoding="utf-8") as handle:
                handle.write("\n")
            case["write_meta"](meta)
            try:
                evaluate(_args_for(case))
            finally:
                case.update(_synthetic_case(tmp))

        _expect_spec_error("验收集清单 sha256 不符", manifest_tampered)

        # 护栏 5：logits 大小与形状不符
        def wrong_size():
            truncated = os.path.join(tmp, "trunc.f32.bin")
            with open(case["fp32"], "rb") as src, open(truncated, "wb") as dst:
                dst.write(src.read(4 * 2))
            args = _args_for(case)
            args.fp32_logits = truncated
            evaluate(args)

        _expect_spec_error("logits 大小不符", wrong_size)

        # 护栏 6：只报率不报 n 的报告必须被拦
        _expect_spec_error(
            "报告只报率不报 n",
            lambda: require_sample_counts({"overall": {"agree_rate": 0.5}, "confident": {}, "strata": []}),
        )
        # 护栏 7：率与分子分母不自洽
        _expect_spec_error(
            "率与分子分母不自洽",
            lambda: require_sample_counts(
                {"overall": {"n": 4, "agree": 1, "agree_rate": 0.5}, "confident": {"n": 0, "agree": 0, "agree_rate": None}, "strata": []}
            ),
        )

        # 无标签模式：必须显式声明，并在报告里写清"不回答对不对"
        args = _args_for(case)
        args.allow_missing_labels = True
        relaxed = evaluate(args)
        assert relaxed["overall"]["top1_accuracy"] is None and relaxed["label_note"], relaxed
        print("  [PASS] --allow-missing-labels 下不产出正确率，且写出未覆盖说明")

        # legacy 模式：允许"与标定集同源 + 清单缺 sha256 + 无标签"，但必须在报告里写明会被高估。
        shared = os.path.join(tmp, "shared_sample.bin")
        with open(shared, "wb") as handle:
            handle.write(b"\x00\x00\x80\x3f")
        legacy_val = os.path.join(tmp, "legacy_val.json")
        legacy_calib = os.path.join(tmp, "legacy_calib.json")
        legacy_meta = os.path.join(tmp, "legacy_meta.json")
        one_row_src = _synthetic_case(tmp)
        with open(legacy_val, "w", encoding="utf-8") as handle:
            json.dump([{"file": shared}], handle)
        with open(legacy_calib, "w", encoding="utf-8") as handle:
            json.dump([{"file": shared, "sha256": sha256_file(shared)}], handle)
        meta = json.loads(json.dumps(one_row_src["base_meta"]))
        meta["validation_set"]["num_samples"] = 1
        meta["validation_set"]["manifest_path"] = legacy_val
        meta["validation_set"].pop("manifest_sha256", None)   # legacy 允许缺
        meta["calibration_set"]["manifest_path"] = legacy_calib
        meta["calibration_set"]["manifest_sha256"] = sha256_file(legacy_calib)
        meta["calibration_set"]["num_samples"] = 1
        with open(legacy_meta, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

        one_fp32 = os.path.join(tmp, "one_fp32.bin")
        one_int8 = os.path.join(tmp, "one_int8.bin")
        _write_logits(one_fp32, [[10.0, 0.0, 0.0, 0.0]])
        _write_logits(one_int8, [[10.0, 0.0, 0.0, 0.0]])
        legacy_args = _args_for(one_row_src)
        legacy_args.fp32_logits = one_fp32
        legacy_args.int8_logits = one_int8
        legacy_args.meta = legacy_meta
        legacy_args.legacy_mode = True
        legacy_report = evaluate(legacy_args)
        compliance = legacy_report["spec_compliance"]
        assert compliance["legacy_mode"] is True, compliance
        assert compliance["calib_overlap_violations"] == 1, compliance
        assert "高估" in compliance["note"], compliance
        print("  [PASS] --legacy-mode 允许同源数据，但报告写明'不满足 §1.6 规格、一致率会被高估'")

    print("[int8_eval --self-test] 全部通过")


def main(argv=None):
    parser = argparse.ArgumentParser(description="INT8 判据评估（规格见同目录 README.md）")
    parser.add_argument("--fp32-logits")
    parser.add_argument("--int8-logits")
    parser.add_argument("--meta")
    parser.add_argument("--calib-dir", help="标定集目录；给定则现场枚举并算 sha256（优先于 meta 内的清单）")
    parser.add_argument("--meta-root", help="meta 内相对路径的解析基准目录")
    parser.add_argument("--allow-missing-labels", action="store_true", help="允许无真值标签（报告会写明不回答'对不对'，不满足 §1.6 规格）")
    parser.add_argument(
        "--legacy-mode",
        action="store_true",
        help=(
            "历史口径交叉校验模式：允许清单缺 sha256（由本脚本现算）、允许无真值标签、"
            "允许验收集与标定集同源。报告会明确标注'不满足 §1.6 规格、一致率会被高估'。"
            "用途只有两个：与 C++ 侧现役统计对齐口径、复现历史数字。"
        ),
    )
    parser.add_argument("--confident-margin", type=float, help="覆盖默认置信余量（默认 {}）".format(DEFAULT_CONFIDENT_MARGIN))
    parser.add_argument("--bucket-edges", help="覆盖默认分桶边界，逗号分隔（默认 {}）".format(",".join(str(int(e)) for e in DEFAULT_BUCKET_EDGES)))
    parser.add_argument("--json-out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    for required in ("fp32_logits", "int8_logits", "meta"):
        if not getattr(args, required):
            parser.error("缺参数 --{}（或使用 --self-test）".format(required.replace("_", "-")))
    try:
        report = evaluate(args)
    except SpecError as exc:
        print("[int8_eval] 拒绝执行：{}".format(exc), file=sys.stderr)
        return 2

    print(render_text(report))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("    报告已写入 {}".format(args.json_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
