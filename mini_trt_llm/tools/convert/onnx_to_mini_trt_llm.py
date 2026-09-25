#!/usr/bin/env python3
"""把 ResNet18 的 ONNX 转成 mini_trt_llm 原生路径需要的 `model.safetensors` + `config.json`。

**为什么源是 ONNX 而不是 PyTorch 的 state_dict**（docs/phase4_development_plan.md §3.1 的决定）：
这份 ONNX 在导出时已经把 BatchNorm **折叠进 Conv**（42 个张量里没有 BN 的 running stats，
也没有独立的 Mul/Div）。直接取 ONNX 里的权重，就保证了**原生路径与 ONNX 路径用的是逐位相同的
权重**——"两条路对拍"才谈得上干净。若改从 state_dict 折叠，折叠顺序/精度与当年导出的结果
未必逐位一致，差异会混进对拍结论里。

**逻辑命名**：ONNX 的 Conv 权重名是 `onnx::Conv_193` 这种无意义的名字，但 **Conv 节点名有意义**
（`/layer1/layer1.0/conv1/Conv`、`/layer2/layer2.0/downsample/downsample.0/Conv`），
所以逻辑名从**节点名**推导，而不是从 initializer 名。推导规则见 `_logical_name_from_node`。

**为什么把布局声明写进 config 而不是转换时悄悄转置**：`fc.weight` 存的是 `[out, in]`
（配合 Gemm 的 `transB=1`）。沿用 GPT-2 转换脚本的既有做法（`source.conv1d_layout`），
**保留原始布局 + 在 `source` 里声明**，这样消费方（builder）必须显式处理，而不是猜。

用法：
    python3 mini_trt_llm/tools/convert/onnx_to_mini_trt_llm.py \\
        --onnx 0_resnet18_onnx/resnet18.onnx --output_dir models/resnet18
"""

import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

# 只允许这两类算子携带权重：出现别的带权重算子就该失败，而不是静默丢掉它。
# （静默丢权重的后果是"模型能跑但数值全错"，最难查。）
SUPPORTED_WEIGHT_OPS = {"Conv", "Gemm"}


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_name_from_node(node_name: str, op_type: str) -> str:
    """`/layer1/layer1.0/conv1/Conv` → `layer1.0.conv1`；`/conv1/Conv` → `conv1`；`/fc/Gemm` → `fc`。

    PyTorch 的导出会把 `layer1` 与 `layer1.0` 都写进路径、把下采样写成
    `downsample/downsample.0`。这两处重复段要去掉，否则同一个模块会出现两种写法，
    消费方就得同时兼容——而"兼容两种写法"正是命名漂移的开端。

    末尾的算子名（`Conv` / `Gemm`）按 `op_type` 剥掉，而**不是**写死 `"Conv"`——
    第一版就因为写死了 Conv，`/fc/Gemm` 变成了 `fc.Gemm`，被下面的自检当场拦下。
    """
    parts = [p for p in node_name.split("/") if p]
    if parts and parts[-1] == op_type:
        parts = parts[:-1]
    # 去掉与外层同名的重复段：['layer1','layer1.0','conv1'] → ['layer1.0','conv1']
    if len(parts) >= 2 and parts[1].startswith(parts[0] + "."):
        parts = parts[1:]
    if len(parts) >= 2 and parts[-1].startswith(parts[-2] + "."):
        # ['layer2.0','downsample','downsample.0'] → ['layer2.0','downsample.0']
        parts = parts[:-2] + [parts[-1]]
    return ".".join(parts)


def _shape_of(value_info) -> List[int]:
    dims = value_info.type.tensor_type.shape.dim
    return [d.dim_value if d.HasField("dim_value") else -1 for d in dims]


def collect_weights(model: onnx.ModelProto) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """返回 (逻辑名 → 张量, 逻辑名 → 算子在图中的类型)。"""
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    tensors: Dict[str, np.ndarray] = {}
    kinds: Dict[str, str] = {}

    for node in model.graph.node:
        used = [name for name in node.input if name in initializers]
        if not used:
            continue
        if node.op_type not in SUPPORTED_WEIGHT_OPS:
            raise SystemExit(
                f"带权重的算子 {node.op_type}（节点 {node.name}）不在支持列表 "
                f"{sorted(SUPPORTED_WEIGHT_OPS)} 内——请先扩展转换脚本，不要静默丢弃权重"
            )
        logical = _logical_name_from_node(node.name, node.op_type)
        if logical in kinds:
            raise SystemExit(f"逻辑名冲突：{logical}（来自节点 {node.name}）")
        kinds[logical] = node.op_type

        for index, name in enumerate(used):
            # Conv/Gemm 的输入顺序固定：index 0 = 权重，index 1 = bias。
            suffix = "weight" if index == 0 else "bias"
            key = f"{logical}.{suffix}"
            if key in tensors:
                raise SystemExit(f"键冲突：{key}")
            tensors[key] = initializers[name].astype(np.float32)

    # 反向护栏：ONNX 里的每个 initializer 都必须被消费掉，否则说明有权重被漏掉
    consumed = set()
    for node in model.graph.node:
        consumed.update(n for n in node.input if n in initializers)
    unused = sorted(set(initializers) - consumed)
    if unused:
        raise SystemExit(f"有 initializer 没被任何算子使用，转换会丢权重：{unused}")
    return tensors, kinds


def validate_resnet18(model: onnx.ModelProto, tensors: Dict[str, np.ndarray]) -> None:
    """结构自检：ResNet18 该有的 20 个 Conv + 1 个 Gemm，形状必须自洽。

    这些断言是"转换产物可信"的前提；没有它们，一份形状错位的 safetensors 会被下游
    当成"权重不对"，排查方向就开始跑偏。
    """
    conv_names = [k[:-len(".weight")] for k in tensors if k.endswith(".weight") and "fc" not in k]
    if len(conv_names) != 20:
        raise SystemExit(f"期望 20 个 Conv，实际 {len(conv_names)} 个：{sorted(conv_names)}")
    if "fc.weight" not in tensors or "fc.bias" not in tensors:
        raise SystemExit("缺少 fc.weight / fc.bias")

    for name in conv_names:
        weight = tensors[f"{name}.weight"]
        bias = tensors[f"{name}.bias"]
        if weight.ndim != 4:
            raise SystemExit(f"{name}.weight 应为 4 维，实际 {weight.shape}")
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise SystemExit(
                f"{name} 的 bias 长度 {bias.shape} 与 weight 输出通道 {weight.shape[0]} 不一致"
            )
    fc_weight = tensors["fc.weight"]
    fc_bias = tensors["fc.bias"]
    if fc_weight.ndim != 2 or fc_bias.shape[0] != fc_weight.shape[0]:
        raise SystemExit(f"fc 形状不自洽：weight {fc_weight.shape}, bias {fc_bias.shape}")

    # 输入/输出契约：与 P4-1 基线、P4-2 引擎对拍用的必须是同一份形状
    input_shape = _shape_of(model.graph.input[0])
    output_shape = _shape_of(model.graph.output[0])
    if input_shape[1:] != [3, 224, 224]:
        raise SystemExit(f"输入形状意外：{input_shape}")
    if output_shape[1] != fc_weight.shape[0]:
        raise SystemExit(f"输出类别数 {output_shape[1]} 与 fc.weight 的 {fc_weight.shape[0]} 不一致")


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet18 ONNX → mini_trt_llm safetensors + config")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_type", default="resnet18")
    parser.add_argument("--architecture", default="cnn")
    parser.add_argument("--self-test", action="store_true",
                        help="只跑护栏自检（把真模型按几种方式改坏，确认自检会拦），不写任何产物")
    parser.add_argument("--skip-if-missing", action="store_true",
                        help="ONNX 不存在时以 77 退出（ctest 约定：缺环境 ≠ 失败），与 "
                             "tools/inspect_onnx.py 的 --skip-if-missing 同义")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        if args.skip_if_missing:
            print(f"[convert] 跳过：找不到 {args.onnx}")
            raise SystemExit(77)
        raise SystemExit(f"ONNX 不存在：{args.onnx}")
    model = onnx.load(args.onnx)
    if args.self_test:
        run_self_test(model)
        return

    tensors, kinds = collect_weights(model)
    validate_resnet18(model, tensors)

    os.makedirs(args.output_dir, exist_ok=True)
    safetensors_path = os.path.join(args.output_dir, "model.safetensors")
    # safetensors 要求 C 连续数组；ONNX 取出来的已经是连续 float32。
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, safetensors_path)

    input_shape = _shape_of(model.graph.input[0])
    output_shape = _shape_of(model.graph.output[0])
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)

    # weight_map 方向：逻辑名（TRT 层权重名）→ safetensors 里的 key。
    # 这里两者同名（identity），与 GPT-2 的既有产物一致；显式写出来是为了让消费方
    # 永远走映射而不是硬编码文件里的键名。
    weight_map = {key: key for key in sorted(tensors)}
    config = {
        "model_type": args.model_type,
        "architecture": args.architecture,
        "hyper_params": {
            "num_classes": int(output_shape[1]),
            "input_channels": int(input_shape[1]),
            "input_height": int(input_shape[2]),
            "input_width": int(input_shape[3]),
        },
        "weight_map": weight_map,
        "source": {
            "onnx": os.path.relpath(args.onnx),
            "onnx_sha256": sha256_of_file(args.onnx),
            "opset": opset,
            "batch_norm": "folded_into_conv",  # 导出时就折叠了，见文件头说明
            "tensor_count": len(tensors),
            "conv_count": sum(1 for k in kinds.values() if k == "Conv"),
            # 布局声明：fc.weight 存的是 [out_features, in_features]，对应 Gemm 的 transB=1。
            # 消费方**不要**再转置一次；要转置也必须依据本字段，而不是猜。
            "fc_weight_layout": "out_in_transB",
        },
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[convert] {args.onnx} → {args.output_dir}")
    print(f"  张量数        : {len(tensors)}（Conv {config['source']['conv_count']} 个 + fc）")
    print(f"  safetensors   : {safetensors_path}")
    print(f"  config        : {config_path}（weight_map {len(weight_map)} 条）")
    print(f"  fc.weight     : {tensors['fc.weight'].shape}（{config['source']['fc_weight_layout']}）")
    print(f"  输入/输出契约 : {input_shape} → {output_shape}")


def _expect_rejected(label: str, mutate, model: onnx.ModelProto) -> None:
    """把模型改坏，确认自检**真的会拦**。

    为什么要有这个自检：护栏没有用例证明"它会拦人"，就等于没有护栏——项目里已经吃过
    "声明了却没人核对"的亏（Phase 3 的子图名核对一栏，见 docs/phase3_test_plan.md §5）。
    """
    broken = onnx.ModelProto()
    broken.CopyFrom(model)
    mutate(broken)
    try:
        tensors, _kinds = collect_weights(broken)
        validate_resnet18(broken, tensors)
    except SystemExit as exc:
        print(f"  [自检生效] {label}: {exc}")
        return
    raise SystemExit(f"[自检失效] {label}：改坏之后竟然通过了——这道护栏是假的")


def run_self_test(model: onnx.ModelProto) -> None:
    print("[self-test] 先确认未改坏的模型能通过…")
    tensors, _kinds = collect_weights(model)
    validate_resnet18(model, tensors)
    print(f"  [基线通过] 收集到 {len(tensors)} 个张量")

    def add_unused_initializer(broken: onnx.ModelProto) -> None:
        # 加一个没有任何算子引用的 initializer：转换会**丢掉**它，必须拒绝而不是静默丢
        dangling = onnx.helper.make_tensor("dangling_weight", onnx.TensorProto.FLOAT, [1],
                                           [0.0])
        broken.graph.initializer.append(dangling)

    def rename_conv_to_duplicate(broken: onnx.ModelProto) -> None:
        # 把第二个 Conv 的节点名改成与第一个相同 → 逻辑名冲突
        convs = [n for n in broken.graph.node if n.op_type == "Conv"]
        convs[1].name = convs[0].name

    def corrupt_bias_length(broken: onnx.ModelProto) -> None:
        # 把某个卷积的 bias 长度改成与输出通道不符 → 形状自检必须拦
        conv = next(n for n in broken.graph.node if n.op_type == "Conv")
        bias_name = conv.input[2]
        for index, initializer in enumerate(broken.graph.initializer):
            if initializer.name == bias_name:
                del broken.graph.initializer[index]
                broken.graph.initializer.append(
                    onnx.helper.make_tensor(bias_name, onnx.TensorProto.FLOAT, [1], [0.0]))
                return
        raise SystemExit("夹具构造失败：找不到卷积的 bias initializer")

    def drop_fc(broken: onnx.ModelProto) -> None:
        # 去掉 Gemm 节点 → 缺少 fc
        keep = [n for n in broken.graph.node if n.op_type != "Gemm"]
        del broken.graph.node[:]
        broken.graph.node.extend(keep)
        # 同时删掉 fc 的 initializer，避免只触发"未被引用"那条
        keep_init = [i for i in broken.graph.initializer
                     if not i.name.startswith("fc.")]
        del broken.graph.initializer[:]
        broken.graph.initializer.extend(keep_init)

    def add_unsupported_weight_op(broken: onnx.ModelProto) -> None:
        # 造一个带权重的 MatMul 节点（不在支持列表里）→ 必须拒绝，不能静默丢弃
        weight = onnx.helper.make_tensor("matmul_weight", onnx.TensorProto.FLOAT, [1, 1], [1.0])
        broken.graph.initializer.append(weight)
        node = onnx.helper.make_node("MatMul", ["matmul_weight", "matmul_weight"],
                                     ["matmul_out"], name="/matmul/MatMul")
        broken.graph.node.append(node)

    print("[self-test] 依次改坏模型，确认每道自检都会拦：")
    _expect_rejected("未被引用的 initializer", add_unused_initializer, model)
    _expect_rejected("逻辑名冲突（节点名重复）", rename_conv_to_duplicate, model)
    _expect_rejected("卷积 bias 长度与输出通道不符", corrupt_bias_length, model)
    _expect_rejected("缺少 fc", drop_fc, model)
    _expect_rejected("不支持的带权算子（MatMul）", add_unsupported_weight_op, model)
    print("[self-test] 全部通过：护栏不是摆设")


if __name__ == "__main__":
    main()
