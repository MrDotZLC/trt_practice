#!/usr/bin/env python3
"""给既有的 Q/DQ ONNX 图**追加"量化前"张量作为图输出**，产出探针图（P4-INT8-a / `future_iterations.md` §1.5）。

## 为什么是一个独立的小工具，而不是塞进 `quantize_resnet18.py`

探针图必须与正式产物**逐位同源**——"只多几个图输出，其余一字不改"。若改成"重新跑一遍标定 + 插
Q/DQ，再顺手加探针输出"，探针图与产物图就绑在**两次独立的标定**上；标定一变（分位、标定图数、
权重粒度），两图就不可比，而整条结论都建立在这个可比性上。
这里只做**一次纯图变换**（读产物 → 追加 `graph.output` → 写新路径）：秒级、离线、可自证。

## 探什么

**每个 Conv 的原始输出**——也就是它后面那对 `QuantizeLinear` / `DequantizeLinear` 里 `QuantizeLinear`
的**输入**那个张量。为什么必须是量化前：量化台阶（本图 conv1 是 `0.0796`）会把 FP32 kernel 的
正常差异（`1e-3` 量级）在桶边界附近放大成 **±1 格**，噪声与待查信号同量级
（`docs/TROUBLESHOOTING.md` + TS-030-LAYERWISE-PROBE / TS-030-CLOSURE 已实测）。量化前张量的量级就是 `1e-3`，可直接判读。

再加 `GlobalAveragePool` 的输出（#30.3 第 2 条点名要覆盖的 "GAP + fc" 段的入口）。
残差 `Add` / `Relu` 的输出**不另挂**：`Add` 是已探两个张量的线性组合、`Relu` 是逐元素裁剪，
两者都能由已探张量推出；多挂输出只会让 TRT 的融合进一步变形（开发计划 §13.3 D5）。

## 自检（缺一条就退出）

1. 候选张量**都不是**既有图输出——命中就说明这张图"已经探过"或图形态变了，脚本的前提不成立；
2. `onnx.shape_inference` 能给全部候选张量定出形状（探针输出必须有具体形状，ONNX 不允许空 shape）；
3. 追加后 `onnx.checker` 通过；
4. **`graph.node` / `graph.initializer` / `graph.input` / `opset_import` 逐字节不变**，只有
   `graph.output` 变长——这就是"探针图 = 产物图 + 探针"的证据。少了这条，本次结论就没有地基。

用法：
    python3 mini_trt_llm/tools/convert/add_probe_outputs.py \\
        --onnx models/resnet18/resnet18_qdq.onnx \\
        --output models/resnet18/resnet18_qdq_probe_per_tensor.onnx
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import onnx
from onnx import TensorProto as T
from onnx import helper as H


def _bytes_of_graph_parts(model: onnx.ModelProto) -> Dict[str, List[bytes]]:
    """把"不该变的部分"序列化成字节，用于前后逐字节比对。"""
    return {
        "node": [n.SerializeToString() for n in model.graph.node],
        "initializer": [i.SerializeToString() for i in model.graph.initializer],
        "input": [i.SerializeToString() for i in model.graph.input],
        "opset": [o.SerializeToString() for o in model.opset_import],
    }


def probe_tensor_names(model: onnx.ModelProto) -> List[str]:
    """探针清单（按图拓扑序）：每个 Conv 的原始输出 + GlobalAveragePool 的输出。"""
    names: List[str] = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            names.append(node.output[0])
        elif node.op_type == "GlobalAveragePool":
            names.append(node.output[0])
    if not names:
        raise SystemExit("图里既没有 Conv 也没有 GlobalAveragePool——这张图不是本工具的目标")
    if len(set(names)) != len(names):
        raise SystemExit("探针清单里有重复张量名，图形态与预期不符")
    return names


def shape_from_inference(model: onnx.ModelProto, names: List[str]) -> Dict[str, List[int]]:
    """用静态形状推断给出每个探针张量的形状（dynamic batch 维会是 0）。

    为什么不在图输出上留空 shape：ONNX 的 `TensorProto` 要求 `type.shape` 存在
    （实测：留空会被 `onnx.checker` 判 `Field 'shape' of 'type' is required but missing`）。
    """
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    lookup = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + \
            list(inferred.graph.output):
        lookup[value.name] = value

    shapes: Dict[str, List[int]] = {}
    for name in names:
        value = lookup.get(name)
        if value is None:
            raise SystemExit(f"形状推断没有给出 {name} —— 无法为它声明图输出")
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 0:
            raise SystemExit(f"{name} 的形状为空，无法声明图输出")
        shapes[name] = [d.dim_value for d in dims]
    return shapes


def add_probe_outputs(src: onnx.ModelProto) -> Tuple[onnx.ModelProto, Dict[str, List[int]]]:
    """返回（探针图, 探针形状表）。不修改入参。"""
    model = onnx.ModelProto()
    model.CopyFrom(src)

    existing = {o.name for o in model.graph.output}
    names = probe_tensor_names(model)
    already = sorted(set(names) & existing)
    if already:
        raise SystemExit(
            f"这些张量已经是图输出：{already[:3]}……"
            "（要么这张图已经探过，要么图形态变了）——本工具拒绝二次追加")

    shapes = shape_from_inference(model, names)
    before = _bytes_of_graph_parts(model)

    for name in names:
        model.graph.output.append(H.make_tensor_value_info(name, T.FLOAT, shapes[name]))

    # 护栏：除了 graph.output，别的都必须逐字节不变。
    after = _bytes_of_graph_parts(model)
    for key, old in before.items():
        if after[key] != old:
            raise SystemExit(f"追加输出时意外改动了 graph.{key} —— 探针图不再等于'产物图 + 探针'")
    if len(model.graph.output) != len(existing) + len(names):
        raise SystemExit("图输出数量与预期不符")
    return model, shapes


def self_test() -> None:
    """护栏自证：把"它会拦人"变成可执行的事实（否则护栏等于没有）。"""
    def tiny_qdq() -> onnx.ModelProto:
        """1×1×4×4 → Conv(2 通道) → Relu，权重走对称 per-tensor Q/DQ。"""
        w = H.make_tensor("w", T.FLOAT, [2, 1, 3, 3], [float(i) for i in range(18)])
        w_scale = H.make_tensor("w_scale", T.FLOAT, [], [0.5])
        w_zp = H.make_tensor("w_zp", T.INT8, [], [0])
        nodes = [
            H.make_node("QuantizeLinear", ["w", "w_scale", "w_zp"], ["w_q"], axis=0),
            H.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"], axis=0),
            H.make_node("Conv", ["input", "w_dq"], ["conv_out"], name="/conv1/Conv",
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            H.make_node("Relu", ["conv_out"], ["relu_out"]),
        ]
        graph = H.make_graph(
            nodes, "tiny_qdq",
            [H.make_tensor_value_info("input", T.FLOAT, [1, 1, 4, 4])],
            [H.make_tensor_value_info("relu_out", T.FLOAT, [1, 2, 4, 4])],
            initializer=[w, w_scale, w_zp])
        model = H.make_model(graph, producer_name="add_probe_outputs --self-test")
        model.opset_import[0].version = 13
        return model

    failures: List[str] = []

    # 1) 正常路径：应当只多出 graph.output，别的逐字节不变。
    model = tiny_qdq()
    probed, shapes = add_probe_outputs(model)
    got = sorted(o.name for o in probed.graph.output)
    if got != ["conv_out", "relu_out"]:
        failures.append(f"探针清单不对：{got}")
    if shapes.get("conv_out") != [1, 2, 4, 4]:
        failures.append(f"conv_out 形状推断不对：{shapes.get('conv_out')}")
    if len(model.graph.output) != 1:
        failures.append("add_probe_outputs 改了入参（它必须是纯函数）")

    # 2) 护栏：已经是图输出的张量必须被拒。
    again = tiny_qdq()
    again.graph.output.append(H.make_tensor_value_info("conv_out", T.FLOAT, [1, 2, 4, 4]))
    try:
        add_probe_outputs(again)
        failures.append("已经探过的图没有被拒绝——二次追加会产出重复图输出")
    except SystemExit:
        pass

    # 3) 护栏：没有 Conv / GAP 的图必须被拒（否则它会静默产出空探针集）。
    plain = H.make_graph(
        [H.make_node("Relu", ["input"], ["relu_out"])], "plain",
        [H.make_tensor_value_info("input", T.FLOAT, [1, 1, 4, 4])],
        [H.make_tensor_value_info("relu_out", T.FLOAT, [1, 1, 4, 4])])
    empty = H.make_model(plain, producer_name="add_probe_outputs --self-test")
    empty.opset_import[0].version = 13
    try:
        add_probe_outputs(empty)
        failures.append("没有 Conv/GAP 的图没有被拒绝")
    except SystemExit:
        pass

    if failures:
        for line in failures:
            print(f"[FAIL] {line}", file=sys.stderr)
        raise SystemExit(1)
    print("[add_probe_outputs --self-test] OK：探针集 = Conv 输出 + GAP 输出；"
          "图其余部分逐字节不变；二次追加与非目标图都被拒")


def main() -> None:
    parser = argparse.ArgumentParser(description="给 Q/DQ ONNX 图追加'量化前'探针输出")
    parser.add_argument("--onnx", help="源 Q/DQ ONNX（通常是正式产物）")
    parser.add_argument("--output", help="探针图输出路径（**新路径**，不覆盖源文件）")
    parser.add_argument("--self-test", action="store_true", help="只跑护栏自证，不需要任何文件")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.onnx or not args.output:
        parser.error("--onnx 与 --output 都是必填（除非用 --self-test）")
    if os.path.abspath(args.onnx) == os.path.abspath(args.output):
        parser.error("--output 不能与 --onnx 相同：探针图是**新文件**，不许覆盖产物")

    src = onnx.load(args.onnx)
    probed, shapes = add_probe_outputs(src)
    onnx.checker.check_model(probed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    onnx.save(probed, args.output)

    report = {
        "source_onnx": args.onnx,
        "probe_onnx": args.output,
        "probe_tensors": {name: shapes[name] for name in
                          probe_tensor_names(src)},
        "probe_count": len(shapes),
        "graph_outputs": [o.name for o in probed.graph.output],
    }
    meta_path = os.path.splitext(args.output)[0] + ".probe.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[probe] {args.onnx} → {args.output}")
    print(f"  探针张量 : {len(shapes)} 个（Conv 原始输出 + GlobalAveragePool 输出）")
    print(f"  图其余部分: 逐字节未变（node / initializer / input / opset）")
    print(f"  元数据    : {meta_path}")


if __name__ == "__main__":
    main()
