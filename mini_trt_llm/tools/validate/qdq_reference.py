#!/usr/bin/env python3
"""用 **ONNX 官方参考实现**执行 Q/DQ 图并落盘指定张量——P4-INT8-a 的数值标尺
（`docs/future_iterations.md` + OI-INT8-PERCHANNEL / future_iterations_development_plan.md + OI-INT8-PERCHANNEL-PLAN）。

## 为什么标尺是"直接执行这张图"，而不是"自己折 BN 的 torch 模型"

上一轮排查（`docs/TROUBLESHOOTING.md` + TS-030-LAYERWISE-PROBE）3 次真机往返里有 **2 次**耗在**探针自身的 BN 折叠错**
上：ONNX 里 BN 已折进 Conv，"卷积输出"这个张量在折叠前后不是同一个东西，第一版连 per-tensor 的
首层都差 96%——那不是 TRT 的问题，是在比两个不同的网络。

这张图的 BN 已经折好了（42 个张量里没有 running stats）。**直接执行它，就根本不存在"折叠"这一步。**
而且 `onnx.reference.ReferenceEvaluator` 是对本项目代码完全独立的第三方实现（`PROGRESS.md` §2.13：
参考必须唯一且独立）——我们比的是"TRT 有没有照着 ONNX 语义跑"。

## 已知实现细节：opset 17 → 21（**必须靠自检兜底，不许只靠这句话**）

本图是 opset **17**，而 `ReferenceEvaluator` 只带 `DequantizeLinear` 的 19 / 21 实现，直接跑会
`RuntimeError: No implementation for operator 'DequantizeLinear' ... found 19, 21, None`。
所以参考侧把**副本**的默认 opset 提到 21。语义不变的理由：本图用到的 Q/DQ 语义（int8、对称、
`axis`、round-half-even）在 17/19/21 之间没有变化。
**这句话由 `--self-test` 逐条验证**（最小 Q/DQ 图逐位比手算值），不是"相信它一样"。

## 落盘格式

`<output-dir>/` 下：

- `<sanitized tensor name>.f32.bin` —— 每个张量一份，raw float32、行主序、按 batch 维拼接；
- `probe_index.txt` —— 每行 `<张量名>\t<元素数>\t<文件名>\t<角色>\t<配对张量>`；
- `meta.json` —— onnx 路径 + SHA256 + opset 处理说明 + 每个张量的形状 + 输入文件清单。

**角色（role）为什么有两类**：只比"探到的张量 vs 参考的**量化前**张量"还不够——万一 TRT 交回来的
是**量化后再反量化**的值（把它当作那个图输出），曲线会全程贴在 `1e-3` 以下、看起来"很干净"，
结论就反了（开发计划 §13.3 D3）。所以参考侧**同时**落每个 Conv 的**量化后**张量，用例做两条比较：
`d_pre = |引擎探针 − 参考量化前|`、`d_post = |引擎探针 − 参考量化后|`，要求 `d_pre ≤ d_post`。
不用"估 scale + 数格点"：那要先估一个 scale，多一个可能出错的环节。

- `role = probe`：这张图**本身就是引擎输出**（量化前张量 / GAP 输出 / 契约输出）；`配对张量` 写 `-`；
- `role = postquant`：**只有参考有**（量化后张量），`配对张量` 列写明它属于哪个 `probe`。

用法：
    python3 mini_trt_llm/tools/validate/qdq_reference.py --self-test
    python3 mini_trt_llm/tools/validate/qdq_reference.py \\
        --onnx models/resnet18/resnet18_qdq_probe_per_tensor.onnx \\
        --calib-dir assets/legacy/resnet18_onnx/calib_data --num-images 8 \\
        --output-dir /tmp/mini_trt_llm_int8_probe/pt
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as H
from onnx.reference import ReferenceEvaluator

# `ReferenceEvaluator` 带 DequantizeLinear 的最低 opset。见模块 docstring。
K_REFERENCE_OPSET = 21


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(name: str) -> str:
    """张量名 → 安全文件名。用例侧必须用**同一套**规则（`test_resnet18_int8_probe.cpp`）。"""
    out = name.strip("/").replace("/", "_")
    return out if out else "tensor"


def reference_model(path: str) -> Tuple[onnx.ModelProto, Dict[str, object]]:
    """加载图并做 opset 提升（只动副本），返回（模型, 说明）。"""
    return reference_model_from_proto(onnx.load(path))


def reference_model_from_proto(model: onnx.ModelProto
                               ) -> Tuple[onnx.ModelProto, Dict[str, object]]:
    """`reference_model` 的内存版（自检用，避免为了自检写一份临时文件）。"""
    copy = onnx.ModelProto()
    copy.CopyFrom(model)
    original = {o.domain or "ai.onnx": o.version for o in copy.opset_import}
    bumped = []
    for o in copy.opset_import:
        if o.domain in ("", "ai.onnx") and o.version < K_REFERENCE_OPSET:
            bumped.append((o.domain or "ai.onnx", o.version, K_REFERENCE_OPSET))
            o.version = K_REFERENCE_OPSET
    return copy, {
        "original_opset": original,
        "bumped_for_reference": bumped,
        "reason": ("ReferenceEvaluator 只实现 DequantizeLinear 的 19/21；本图 Q/DQ 语义"
                   "（int8/对称/axis/round-half-even）在 17→21 之间未变。由 --self-test 逐位兜底。"),
    }


def load_inputs(calib_dir: str, num_images: int) -> Tuple[List[str], np.ndarray]:
    files = sorted(glob.glob(os.path.join(calib_dir, "*.bin")))
    if len(files) < num_images:
        raise SystemExit(f"标定图不足：{len(files)} < {num_images}")
    files = files[:num_images]
    rows = []
    for path in files:
        flat = np.fromfile(path, dtype=np.float32)
        if flat.size != 3 * 224 * 224:
            raise SystemExit(f"{path} 元素数 {flat.size} ≠ 3*224*224")
        rows.append(flat.reshape(1, 3, 224, 224))
    return files, np.concatenate(rows, axis=0)


def postquant_pairs(model: onnx.ModelProto) -> List[Tuple[str, str]]:
    """对每个 Conv：`Conv 输出` → 它后面那对 Q/DQ 的 **DQ 输出**（量化后张量）。

    为什么用"顺着消费者找"而不是拼名字：名字是上游脚本的约定，图形态一变就失效；
    "谁消费谁"是图自身的结构。找不到就**不猜**，直接跳过。
    """
    # 找的是"谁消费了这个张量"，不是"谁生产了它"——第一版写成了 producer，
    # 于是 `producer.get(conv_out)` 拿回 Conv 自己，配对恒为空（自检当场抓住）。
    consumers: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, node)
    pairs: List[Tuple[str, str]] = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        conv_out = node.output[0]
        quant = consumers.get(conv_out)
        if quant is None or quant.op_type != "QuantizeLinear":
            continue
        dequant = consumers.get(quant.output[0])
        if dequant is None or dequant.op_type != "DequantizeLinear":
            continue
        pairs.append((conv_out, dequant.output[0]))
    return pairs


def run_reference(model: onnx.ModelProto, batch: np.ndarray, extra: List[str]
                  ) -> Tuple[List[str], List[np.ndarray]]:
    graph_outputs = [o.name for o in model.graph.output]
    known = set(graph_outputs)
    for name in extra:
        if name in known:
            raise SystemExit(f"{name} 既是图输出又被列为 extra —— 会写出两份同名文件")
    names = graph_outputs + list(extra)
    if not names:
        raise SystemExit("图没有任何输出，没什么可落的")
    evaluator = ReferenceEvaluator(model, verbose=0)
    outputs = evaluator.run(names, {model.graph.input[0].name: batch})
    return names, outputs


def dump(output_dir: str, records: List[Tuple[str, np.ndarray, str, str]],
         meta: Dict[str, object]) -> None:
    """records: (张量名, 数组, 角色, 配对张量名)。"""
    os.makedirs(output_dir, exist_ok=True)
    index_lines = []
    entries = []
    for name, array, role, paired in records:
        if array.dtype != np.float32:
            raise SystemExit(f"{name} 的类型是 {array.dtype}，本工具只落 float32 图输出")
        file_name = sanitize(name) + ".f32.bin"
        data = np.ascontiguousarray(array, dtype=np.float32)
        data.tofile(os.path.join(output_dir, file_name))
        index_lines.append(f"{name}\t{data.size}\t{file_name}\t{role}\t{paired}")
        entries.append({"name": name, "shape": list(array.shape), "elements": int(data.size),
                        "file": file_name, "role": role, "paired": paired})
    with open(os.path.join(output_dir, "probe_index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(index_lines) + "\n")
    meta = dict(meta)
    meta["tensors"] = entries
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def self_test() -> None:
    """护栏自证：标尺自己必须先被校准（`TROUBLESHOOTING.md` + #29.1 / #30.4 的教训）。"""
    failures: List[str] = []

    def build_tiny() -> onnx.ModelProto:
        """输入 (1,2,3,3) → 输入 Q/DQ（per-tensor）→ Conv(4 通道, 权重 per-channel Q/DQ)
        → 输出 Q/DQ。权重故意让第 2 个输出通道量级大 10× ——这样"per-channel 与 per-tensor
        可分辨"是数学必然（沿用 `docs/TROUBLESHOOTING.md` + #29.1 的手法）。"""
        rng = np.random.default_rng(0)
        w = rng.normal(size=(4, 2, 3, 3)).astype(np.float32)
        w[1] *= 100.0
        in_scale = np.float32(2.0)
        out_scale = np.float32(3.0)
        w_scale = (np.abs(w).reshape(4, -1).max(axis=1) / 127.0).astype(np.float32)
        nodes = [
            H.make_node("QuantizeLinear", ["input", "in_scale", "in_zp"], ["in_q"]),
            H.make_node("DequantizeLinear", ["in_q", "in_scale", "in_zp"], ["in_dq"]),
            H.make_node("QuantizeLinear", ["w", "w_scale", "w_zp"], ["w_q"], axis=0),
            H.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"], axis=0),
            H.make_node("Conv", ["in_dq", "w_dq"], ["conv_out"], name="/conv1/Conv",
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            H.make_node("QuantizeLinear", ["conv_out", "out_scale", "out_zp"], ["out_q"]),
            H.make_node("DequantizeLinear", ["out_q", "out_scale", "out_zp"], ["output"]),
        ]
        graph = H.make_graph(
            nodes, "tiny_qdq",
            [H.make_tensor_value_info("input", T.FLOAT, [1, 2, 3, 3])],
            [H.make_tensor_value_info("output", T.FLOAT, [1, 4, 3, 3])],
            initializer=[
                onnx.numpy_helper.from_array(w, "w"),
                onnx.numpy_helper.from_array(np.array(in_scale), "in_scale"),
                onnx.numpy_helper.from_array(np.array(0, np.int8), "in_zp"),
                onnx.numpy_helper.from_array(np.array(out_scale), "out_scale"),
                onnx.numpy_helper.from_array(np.array(0, np.int8), "out_zp"),
                onnx.numpy_helper.from_array(w_scale, "w_scale"),
                onnx.numpy_helper.from_array(np.zeros(4, np.int8), "w_zp"),
            ])
        model = H.make_model(graph, producer_name="qdq_reference --self-test")
        model.opset_import[0].version = 17  # 与真实产物同代，确保走 opset 提升那条路径
        return model

    def quant(x, scale, zp=0):
        """ONNX QuantizeLinear 的语义：round-half-to-even + 饱和到 int8。"""
        return np.clip(np.round(np.asarray(x, np.float32) / np.float32(scale)) + zp,
                       -128, 127).astype(np.int8)

    def conv_same(padded: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """stride=1 / pad=1 的 3×3 卷积（手算，够小）。"""
        out = np.zeros((1, weight.shape[0], 3, 3), np.float32)
        for oc in range(weight.shape[0]):
            for ic in range(weight.shape[1]):
                for i in range(3):
                    for j in range(3):
                        out[0, oc, i, j] += float(
                            (padded[0, ic, i:i + 3, j:j + 3] * weight[oc, ic]).sum())
        return out

    # ① opset 提升后参考实现能跑，且**逐位等于手算的 ONNX 语义**
    model = build_tiny()
    bumped, note = reference_model_from_proto(model)
    if note["bumped_for_reference"] != [("ai.onnx", 17, K_REFERENCE_OPSET)]:
        failures.append(f"opset 提升没有按预期发生：{note['bumped_for_reference']}")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 2, 3, 3)).astype(np.float32)
    # 这张小图里 Conv 的输出就叫 conv_out、它的 DQ 输出就叫 output → 配对应是 [(conv_out, output)]
    pairs = postquant_pairs(bumped)
    if pairs != [("conv_out", "output")]:
        failures.append(f"量化后张量的配对找错了：{pairs}")
    try:
        names, tensors = run_reference(bumped, x, extra=[])
    except Exception as exc:  # noqa: BLE001 —— 自检要把任何异常转成失败
        failures.append(f"参考实现跑最小 Q/DQ 图失败：{type(exc).__name__}: {exc}")
        names, tensors = [], []

    if tensors:
        got = dict(zip(names, tensors))
        init = {i.name: onnx.numpy_helper.to_array(i) for i in bumped.graph.initializer}
        w, w_scale = init["w"], init["w_scale"]

        # 手算：输入量化 → conv（用 DQ 后的权重）→ 输出量化
        in_scale, out_scale = float(init["in_scale"]), float(init["out_scale"])
        in_dq = quant(x, in_scale).astype(np.float32) * in_scale
        # 逐输出通道的权重 scale → 逐通道量化（axis=0）
        w_q = np.clip(np.round(w / w_scale.reshape(-1, 1, 1, 1)), -128, 127).astype(np.int8)
        w_dq = (w_q.astype(np.float32) * w_scale.reshape(-1, 1, 1, 1))
        padded = np.pad(in_dq, ((0, 0), (0, 0), (1, 1), (1, 1)))
        conv = conv_same(padded, w_dq)
        expected = quant(conv, out_scale).astype(np.float32) * out_scale

        diff = float(np.abs(got["output"] - expected).max())
        if diff != 0.0:
            failures.append(f"参考实现与手算的 ONNX 语义不一致：max_abs = {diff}")

        # ② 可分辨性：per-channel 与 per-tensor 在同一张图上必须给出**差得远**的输出
        #    （否则这把尺子根本量不出权重粒度的差别，整条结论就无从谈起 —— `docs/TROUBLESHOOTING.md` + #29.1 的手法）。
        w_pt_scale = float(np.abs(w).max()) / 127.0
        w_pt = np.clip(np.round(w / w_pt_scale), -128, 127).astype(np.float32) * w_pt_scale
        out_pc = quant(conv_same(padded, w_dq), out_scale).astype(np.float32) * out_scale
        out_pt = quant(conv_same(padded, w_pt), out_scale).astype(np.float32) * out_scale
        if float(np.abs(out_pc - out_pt).max()) < 1.0:
            failures.append("构造的权重上 per-channel 与 per-tensor 的输出几乎无差"
                            "——自检失去判别力")

    # ③ index / meta 落盘格式
    with tempfile.TemporaryDirectory() as tmp:
        dump(tmp, [("/a/b", np.zeros(3, np.float32), "probe", "-"),
                   ("conv_out", np.ones((2, 3), np.float32), "postquant", "/a/b")],
             {"onnx": "x"})
        index = open(os.path.join(tmp, "probe_index.txt"), encoding="utf-8").read().strip()
        if "conv_out\t6\tconv_out.f32.bin\tpostquant\t/a/b" not in index.splitlines():
            failures.append(f"probe_index.txt 格式不对：{index!r}")
        meta = json.load(open(os.path.join(tmp, "meta.json"), encoding="utf-8"))
        if [t["name"] for t in meta["tensors"]] != ["/a/b", "conv_out"]:
            failures.append("meta.json 的张量顺序不对（必须与图输出顺序一致）")

    if failures:
        for line in failures:
            print(f"[FAIL] {line}", file=sys.stderr)
        raise SystemExit(1)
    print("[qdq_reference --self-test] OK："
          "参考实现逐位等于手算的 ONNX Q/DQ 语义（0 差）；"
          "per-channel / per-tensor 可分辨；index/meta 格式正确")

def main() -> None:
    parser = argparse.ArgumentParser(description="用 ONNX 官方参考实现落盘 Q/DQ 图的张量")
    parser.add_argument("--onnx", help="Q/DQ 图（通常是探针图）")
    parser.add_argument("--calib-dir", default="assets/legacy/resnet18_onnx/calib_data")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--output-dir", help="落盘目录")
    parser.add_argument("--self-test", action="store_true", help="只跑护栏自证，不需要任何文件")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.onnx or not args.output_dir:
        parser.error("--onnx 与 --output-dir 都是必填（除非用 --self-test）")

    model, opset_note = reference_model(args.onnx)
    files, batch = load_inputs(args.calib_dir, args.num_images)
    pairs = postquant_pairs(model)
    if not pairs:
        raise SystemExit("没找到任何 `Conv 输出 → 量化后` 的配对：这张图可能不是 Q/DQ 图"
                         "（探针用例要靠它做'探到的是量化前'的自证）")
    names, tensors = run_reference(model, batch, extra=[post for _, post in pairs])
    records = [(name, array, "probe", "-")
               for name, array in zip(names[:len(names) - len(pairs)],
                                      tensors[:len(tensors) - len(pairs)])]
    records += [(post, array, "postquant", pre)
                for (pre, post), array in zip(pairs, tensors[len(tensors) - len(pairs):])]
    dump(args.output_dir, records, {
        "onnx": args.onnx,
        "onnx_sha256": sha256_of_file(args.onnx),
        "num_images": len(files),
        "batch_shape": list(batch.shape),
        "input_files": files,
        "reference": "onnx.reference.ReferenceEvaluator",
        "postquant_pairs": [{"probe": pre, "postquant": post} for pre, post in pairs],
        "opset": opset_note,
    })
    print(f"[qdq_reference] {args.onnx} → {args.output_dir}")
    print(f"  张量    : {len(records)} 个 = 引擎输出 {len(names) - len(pairs)}"
          f" + 量化后（参考独有）{len(pairs)}")
    print(f"  输入    : {len(files)} 张，batch 形状 {list(batch.shape)}")
    print(f"  落盘    : probe_index.txt / meta.json + 每个张量一份 .f32.bin")


if __name__ == "__main__":
    main()
