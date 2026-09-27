#!/usr/bin/env python3
"""把 FP32 的 resnet18.onnx 转成**对称 int8 Q/DQ**图，供 TensorRT 建 INT8 引擎（Phase 4 / P4-7）。

## 为什么不用 `torch.ao.quantization` 的现成 PTQ（走过的弯路，别再走）

1. **默认 qconfig 是非对称的**（`get_default_qconfig('fbgemm')` → quint8、zero_point≠0），
   导出后 TensorRT **在解析阶段直接拒**：
   `Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported`。
2. **自建"对称 qconfig"在 torch 里不成立**：fbgemm 后端的 `DTypeConfig` 写明激活只认
   `torch.quint8`（见 `torch/ao/quantization/backend_config/fbgemm.py:32`），
   任何 qint8 激活的 qconfig 会被 `prepare_fx` **静默丢弃**（observer 数 = 0，且不报错）。

所以这里自己插 Q/DQ：**校准统计仍用 torch**（在真模型上挂 hook 收 min/max，不重造标定），
只有"插节点"这一步由本脚本做——它可控、可自检，而且产出的形态正是 TRT 要求的
（激活 per-tensor 对称、权重 per-channel 对称、zero_point 恒为 0、int8）。

## 图形态

对每个 Conv：`x → Q → DQ → Conv(weight 也走 Q→DQ) → Q → DQ → 原有消费者`。
残差 Add / Relu / MaxPool / GlobalAveragePool / Gemm **保持 FP32**（混合精度），
因此量化误差只来自卷积，且每处量化点都显式可见。图的 I/O 名（`input` / `output`）
与契约保持不变——`BuildFromOnnx` 依赖它。

## 自检（缺一条就退出）

1. 所有 Q/DQ 的 zero_point 初值 == 0（int8）；
2. Q/DQ 对数 == 20 个卷积 × 3（输入激活 / 权重 / 输出激活）；
3. `onnx.checker` 通过；
4. **数值预检**：用同一批 scale 在 torch 里做 fake-quant，报告与 FP32 的 argmax 一致性与误差
   ——在建 TRT 引擎之前先知道量化损失有多大（省一轮真机往返）。

用法：
    python3 mini_trt_llm/tools/convert/quantize_resnet18.py \\
        --onnx 0_resnet18_onnx/resnet18.onnx \\
        --calib-dir 0_resnet18_onnx/calib_data \\
        --output models/resnet18/resnet18_qdq.onnx
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision.models as models
from onnx import TensorProto as T
from onnx import helper as H
from onnx import numpy_helper as N

K_CONVS_EXPECTED = 20
ZERO_POINT = np.array(0, dtype=np.int8)


def logical_name(node_name: str, op_type: str) -> str:
    """与 `onnx_to_mini_trt_llm.py` **同一套**命名规则（两处必须一致，否则对不上 torch 模块名）。

    `/layer1/layer1.0/conv1/Conv` → `layer1.0.conv1`；`/conv1/Conv` → `conv1`。
    """
    parts = [p for p in node_name.split("/") if p]
    if parts and parts[-1] == op_type:
        parts = parts[:-1]
    if len(parts) >= 2 and parts[1].startswith(parts[0] + "."):
        parts = parts[1:]
    if len(parts) >= 2 and parts[-1].startswith(parts[-2] + "."):
        parts = parts[:-2] + [parts[-1]]
    return ".".join(parts)


def collect_ranges(calib_files: List[str], batch_limit: int, bins: int = 2048
                   ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """在真模型上统计每个卷积的**输入/输出激活**与**权重**范围。

    为什么用 hook 而不是自己写前向：标定统计（哪些张量、取什么范围）交给 torch 这个既有实现，
    我们只负责把它变成 Q/DQ 节点——把"算法"和"格式"分开，出错时容易定位是哪一层的错。
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    convs = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Conv2d)]
    stats: Dict[str, Dict[str, float]] = {name: {"in_max": 0.0, "out_max": 0.0} for name, _ in convs}
    handles = []

    def make_pre(name: str):
        def hook(_module, inputs):
            x = inputs[0].detach()
            stats[name]["in_max"] = max(stats[name]["in_max"], float(x.abs().max()))
        return hook

    def make_post(name: str):
        def hook(_module, _inputs, output):
            y = output.detach()
            stats[name]["out_max"] = max(stats[name]["out_max"], float(y.abs().max()))
        return hook

    for name, mod in convs:
        handles.append(mod.register_forward_pre_hook(make_pre(name)))
        handles.append(mod.register_forward_hook(make_post(name)))

    with torch.no_grad():
        for path in calib_files[:batch_limit]:
            x = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(1, 3, 224, 224))
            model(x)
    for handle in handles:
        handle.remove()

    for name in stats:
        if stats[name]["in_max"] <= 0.0 or stats[name]["out_max"] <= 0.0:
            raise SystemExit(f"{name} 的激活范围统计为 0，标定数据可能没喂进去")

    # **第二遍**：用第一遍定下的固定上界收集直方图。
    #
    # 为什么必须两遍：第一版在同一个 hook 里用"当时累计的 max"当上界，而累计 max 会不断增长
    # → 每次累加的桶边界都不一样，多张图的直方图**不能相加**，分位统计因此失去意义。
    # （症状：预检看起来还行，真机 INT8 的对拍却差 6 倍以上。）
    hist: Dict[str, Dict[str, np.ndarray]] = {
        name: {"in": np.zeros(bins, dtype=np.float64), "out": np.zeros(bins, dtype=np.float64)}
        for name, _ in convs}

    def make_pre_fixed(name: str):
        def hook(_module, inputs):
            x = inputs[0].detach().abs().flatten()
            hist[name]["in"] += _hist(x, stats[name]["in_max"], bins)
        return hook

    def make_post_fixed(name: str):
        def hook(_module, _inputs, output):
            y = output.detach().abs().flatten()
            hist[name]["out"] += _hist(y, stats[name]["out_max"], bins)
        return hook

    handles = []
    for name, mod in convs:
        handles.append(mod.register_forward_pre_hook(make_pre_fixed(name)))
        handles.append(mod.register_forward_hook(make_post_fixed(name)))
    with torch.no_grad():
        for path in calib_files[:batch_limit]:
            x = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(1, 3, 224, 224))
            model(x)
    for handle in handles:
        handle.remove()

    weights = {name: mod.weight.detach().numpy().astype(np.float32) for name, mod in convs}
    return {"hist": hist, "max": stats}, weights


def _hist(values: torch.Tensor, bound: float, bins: int) -> np.ndarray:
    """把 |x| 累加到 **[0, bound] 的固定桶**里（bound 由第一遍确定，整个标定期间不变）。

    只有桶边界固定，多张图的直方图才可加——这是百分位标定能成立的前提。
    """
    if bound <= 0.0:
        return np.zeros(bins, dtype=np.float64)
    counts = torch.histc(values.clamp(max=bound), bins=bins, min=0.0, max=bound)
    return counts.cpu().numpy().astype(np.float64)


def percentile_clip(values_hist: Dict[str, Dict[str, np.ndarray]],
                    max_ranges: Dict[str, Dict[str, float]], bins: int,
                    percentile: float) -> Dict[str, Dict[str, float]]:
    """从直方图取高分位作为**裁剪值**（不是取 max）。

    为什么要裁剪：单个离群值会把整张张量的 scale 抬高，其余 99.9% 的值都挤在很窄的一段里，
    精度反而更差。这也是实测"用 max 标定时 argmax 掉一个"的根因。
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in values_hist:
        out[name] = {}
        for kind in ("in", "out"):
            counts = values_hist[name][kind]
            total = counts.sum()
            bound = max_ranges[name][f"{kind}_max"]
            if total <= 0 or bound <= 0:
                raise SystemExit(f"{name} 的 {kind} 直方图为空")
            cumulative = np.cumsum(counts)
            # 找第一个累计占比 >= percentile 的桶，取该桶右边界
            idx = int(np.searchsorted(cumulative, percentile / 100.0 * total))
            idx = min(idx, bins - 1)
            out[name][kind] = float((idx + 1) / bins * bound)
    return out


def symmetrize(clip_ranges: Dict[str, Dict[str, float]], weights: Dict[str, np.ndarray],
               weight_scope: str = "per_channel"):
    """把范围换算成**对称 int8** 的 scale：`s = max|x| / 127`（zero_point 恒为 0）。

    激活 per-tensor、权重 per-channel（每个输出通道一条 scale）——这正是 TRT 的
    `IQuantizeLayer` 支持的两种形态（per-channel 只对权重有效）。
    """
    act_scales, weight_scales = {}, {}
    for name, r in clip_ranges.items():
        act_scales[name] = {"in": np.float32(r["in"] / 127.0),
                            "out": np.float32(r["out"] / 127.0)}
        flat = np.abs(weights[name]).reshape(weights[name].shape[0], -1)
        if weight_scope == "per_channel":
            scale = flat.max(axis=1)
            # **给 per-channel scale 设下限**：真实模型里存在"死通道"——实测 conv1 有 8 个输出通道
            # 的 max|w| < 1e-6（最小 1.37e-14，torchvision ResNet18 的已知现象）。若照原样取
            # scale = max|w|/127，该层的 scale 跨度会到 1e13，**TRT 的 INT8 卷积处理不了这种比例**，
            # 那些通道的输出被搞坏（实测：整网 top-1 一致率从 60.9% 掉到 25%）。
            # 下限取该层最大 scale 的 1/1024：死通道的权重远小于它 → round(w/s) = 0 →
            # 整通道量化为 0，这正是它们应有的表示。
            floor_ratio = 1024.0
            scale = np.maximum(scale, scale.max() / floor_ratio)
        else:  # per_tensor：整张权重一个 scale（隔离实验用）
            scale = np.array([flat.max()], dtype=np.float32)
        weight_scales[name] = (scale / 127.0).astype(np.float32)
    return act_scales, weight_scales


def onnx_graph_parameters(model: onnx.ModelProto) -> Dict[str, object]:
    """按**逻辑名**取 ONNX 图里 Conv / Gemm 实际使用的权重与偏置。

    **为什么必须有这个函数（P4-INT8-a 的根因）**：`collect_ranges` 的 `weights` 来自
    **torchvision 模型**，那里的 Conv 还没折 BatchNorm；而插 Q/DQ 的对象是**这份 ONNX 图**里的权重，
    它**已经折过 BN**（`W_folded = W · γ/√(var+ε)`，实测逐通道系数跨度 **0.05 ~ 19.9**）。
    两边不是同一个张量：拿前者算 scale、往后者上量化，等于"尺子量 A、裁剪 B"。
      · **per-tensor**：整张权重一个标量，错的是同一个倍率 → 后果是整体粗一点点（实测 conv1
        用 `1.016` 去量幅度只有 `0.392` 的权重 → 有效位宽少约 1.4 bit）；
      · **per-channel**：错的是**逐通道**倍率 → 系数 >1 的通道直接**溢出饱和**（`round(w/s) > 127`
        被 clamp），系数 <1 的通道变得很粗。这就是"算子级/block 级都等价、整网级 per-channel
        明显更差"的来源——不是 TRT 的处理，而是**产图时用错了张量**。

    正确做法：scale 必须从"**被量化那张张量**"上取。本函数返回的就是它——`fake_quant_check`
    也用它把手上的 torch 模型换成"与图逐位等价"的那个，否则预检量的又是另一张权重。
    """
    initializers = {i.name: N.to_array(i) for i in model.graph.initializer}
    conv_weights: Dict[str, np.ndarray] = {}
    conv_biases: Dict[str, np.ndarray] = {}
    fc_weight = fc_bias = None
    for node in model.graph.node:
        if node.op_type == "Conv":
            name = logical_name(node.name, "Conv")
            if node.input[1] not in initializers:
                raise SystemExit(f"Conv {node.name} 的权重 {node.input[1]} 不在 initializer 里")
            conv_weights[name] = initializers[node.input[1]].astype(np.float32)
            if len(node.input) > 2 and node.input[2] in initializers:
                conv_biases[name] = initializers[node.input[2]].astype(np.float32)
        elif node.op_type == "Gemm":
            fc_weight = initializers[node.input[1]].astype(np.float32)
            if len(node.input) > 2 and node.input[2] in initializers:
                fc_bias = initializers[node.input[2]].astype(np.float32)
    if fc_weight is None or fc_bias is None:
        raise SystemExit("图里没找到带权重与偏置的 Gemm（ResNet18 的 fc）")
    return {"conv_weights": conv_weights, "conv_biases": conv_biases,
            "fc_weight": fc_weight, "fc_bias": fc_bias}


def insert_qdq(model: onnx.ModelProto, act_scales, weight_scales,
               weight_axis: bool = True,
               weight_form: str = "qdq") -> Dict[str, object]:
    """在每个 Conv 的输入激活、权重、输出激活上插 Q/DQ。返回统计信息。

    节点顺序：按原图拓扑顺序重建，遇到 Conv 就在它**前后**各插一对 Q/DQ。
    这样既保持拓扑序，又让"哪对 Q/DQ 属于哪个卷积"一眼可见。
    """
    graph = model.graph
    new_inits: List[onnx.TensorProto] = []
    # prequant_dq 形态要按名字取原始 FP32 权重、并在最后删掉它们（否则文件里白留一份 FP32）
    numpy_helper_map = {i.name: N.to_array(i) for i in graph.initializer}
    drop_initializers: set = set()
    consumers: Dict[str, List[Tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for idx, inp in enumerate(node.input):
            consumers.setdefault(inp, []).append((node, idx))
    graph_outputs = {o.name for o in graph.output}

    def add_scale(name: str, value) -> str:
        new_inits.append(N.from_array(np.atleast_1d(np.asarray(value, dtype=np.float32)), name))
        return name

    def add_zero_point(name: str, size: int) -> str:
        zp = ZERO_POINT if size <= 1 else np.zeros(size, dtype=np.int8)
        new_inits.append(N.from_array(zp, name))
        return name

    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    if len(conv_nodes) != K_CONVS_EXPECTED:
        raise SystemExit(f"期望 {K_CONVS_EXPECTED} 个 Conv，实际 {len(conv_nodes)}")

    final_nodes: List[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type != "Conv":
            final_nodes.append(node)
            continue
        name = logical_name(node.name, "Conv")
        if name not in act_scales:
            raise SystemExit(f"Conv {node.name}（逻辑名 {name}）没有标定数据")

        # ① 输入激活：Q → DQ → 接回卷积输入
        in_scale = add_scale(f"qdq_{name}_in_scale", act_scales[name]["in"])
        in_zp = add_zero_point(f"qdq_{name}_in_zp", 1)
        q_in, dq_in = f"qdq_{name}_in_q", f"qdq_{name}_in"
        final_nodes.append(H.make_node("QuantizeLinear", [node.input[0], in_scale, in_zp],
                                       [q_in], name=f"/qdq_{name}_in/QuantizeLinear"))
        final_nodes.append(H.make_node("DequantizeLinear", [q_in, in_scale, in_zp], [dq_in],
                                       name=f"/qdq_{name}_in/DequantizeLinear"))
        node.input[0] = dq_in

        # ② 权重：per-channel 对称（axis=0 = 输出通道）。TRT 的 per-channel 只对权重有效。
        w_name = node.input[1]
        w_scale = add_scale(f"qdq_{name}_w_scale", weight_scales[name])
        w_zp = add_zero_point(f"qdq_{name}_w_zp", len(weight_scales[name]))
        q_w, dq_w = f"qdq_{name}_w_q", f"qdq_{name}_w"
        w_attrs = {"axis": 0} if weight_axis else {}
        if weight_form == "prequant_dq":
            # **权重只留 DequantizeLinear**：在 Python 侧把权重预先量化成 int8 常量，
            # 图里不再出现 `Q(const)`。理由：NVIDIA 工具链导出的 QDQ 图就是这么做的，
            # 而 `Q(const)→DQ` 在某些后端可能被优化器特殊处理（见 TROUBLESHOOTING #31 的验证）。
            # 用 ONNX 的 int8 常量 + DQ(scale, zp, axis) 表达同一语义。
            w_fp32 = numpy_helper_map[w_name]   # ONNX 的 initializer 名是 onnx::Conv_xxx，不是逻辑名
            per_channel = len(weight_scales[name]) > 1
            s = weight_scales[name]
            s_broadcast = s.reshape(-1, 1, 1, 1) if per_channel else s
            q_values = np.clip(np.round(w_fp32 / s_broadcast), -128, 127).astype(np.int8)
            q_name = f"qdq_{name}_w_int8"
            new_inits.append(N.from_array(q_values, q_name))
            final_nodes.append(H.make_node("DequantizeLinear", [q_name, w_scale, w_zp], [dq_w],
                                           name=f"/qdq_{name}_w/DequantizeLinear", **w_attrs))
            node.input[1] = dq_w
            drop_initializers.add(w_name)
        else:
            final_nodes.append(H.make_node("QuantizeLinear", [w_name, w_scale, w_zp], [q_w],
                                           name=f"/qdq_{name}_w/QuantizeLinear", **w_attrs))
            final_nodes.append(H.make_node("DequantizeLinear", [q_w, w_scale, w_zp], [dq_w],
                                           name=f"/qdq_{name}_w/DequantizeLinear", **w_attrs))
            node.input[1] = dq_w

        final_nodes.append(node)

        # ③ 输出激活：Q → DQ，并把所有消费者改到 DQ 的输出上
        out_name = node.output[0]
        if out_name in graph_outputs:
            raise SystemExit(f"{out_name} 是图输出；本脚本假定卷积输出不是图输出，"
                             f"否则会破坏 I/O 契约")
        out_scale = add_scale(f"qdq_{name}_out_scale", act_scales[name]["out"])
        out_zp = add_zero_point(f"qdq_{name}_out_zp", 1)
        q_out, dq_out = f"qdq_{name}_out_q", f"qdq_{name}_out"
        final_nodes.append(H.make_node("QuantizeLinear", [out_name, out_scale, out_zp], [q_out],
                                       name=f"/qdq_{name}_out/QuantizeLinear"))
        final_nodes.append(H.make_node("DequantizeLinear", [q_out, out_scale, out_zp], [dq_out],
                                       name=f"/qdq_{name}_out/DequantizeLinear"))
        for consumer, idx in consumers.get(out_name, []):
            consumer.input[idx] = dq_out

    del graph.node[:]
    graph.node.extend(final_nodes)
    if drop_initializers:
        keep = [i for i in graph.initializer if i.name not in drop_initializers]
        del graph.initializer[:]
        graph.initializer.extend(keep)
    graph.initializer.extend(new_inits)
    return {"convs": len(conv_nodes), "qdq_pairs": len(conv_nodes) * 3}


def check_qdq(model: onnx.ModelProto, expected_pairs: int) -> Dict[str, int]:
    """自检：Q/DQ 数量、zero_point 必须全 0、dtype 必须是 int8。"""
    graph = model.graph
    q = sum(1 for n in graph.node if n.op_type == "QuantizeLinear")
    dq = sum(1 for n in graph.node if n.op_type == "DequantizeLinear")
    inits = {i.name: i for i in graph.initializer}
    bad_zp = []
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear") or len(node.input) < 3:
            continue
        tensor = inits.get(node.input[2])
        if tensor is None:
            continue
        values = N.to_array(tensor)
        if values.dtype != np.int8 or np.any(values != 0):
            bad_zp.append((node.name, str(values.dtype), values.tolist()[:4]))
    if bad_zp:
        raise SystemExit(f"存在非零或非 int8 的 zero_point（TRT 会直接拒）：{bad_zp[:3]}")
    return {"quantize": q, "dequantize": dq, "expected_q": expected_pairs,
            "expected_dq": expected_pairs}
    return {"quantize": q, "dequantize": dq}


def saturation_stats(model: onnx.ModelProto) -> Dict[str, object]:
    """量化后的权重常量里被 clamp 到 ±127 的比例 —— **尺度是否合理的直接指纹**。

    对称量化在尺度正确时，每张权重里**恰好**只有那个 `max|w|` 元素贴到 ±127
    （per-channel 就是每个输出通道一个）。若被 clamp 的比例远高于这个数，说明 scale **偏小**——
    而 scale 偏小的典型来源正是"拿另一张张量的 max 当尺子"（P4-INT8-a 的根因）。

    实测（`docs/TROUBLESHOOTING.md` #46）：错源 16.19% / 改源 **0.044%** / per-tensor 产物 3.92%。
    本函数只**报数**不做断言——"多少算高"没有普适阈值，而这条数字配 #46.2 那张表足以一眼判断。
    """
    saturated = total = channels_saturated = channels = 0
    worst: List[Tuple[str, int]] = []
    for initializer in model.graph.initializer:
        if not initializer.name.endswith("_w_int8") or initializer.data_type != T.INT8:
            continue
        values = N.to_array(initializer).astype(np.int32)
        flat = values.reshape(values.shape[0], -1)
        hit = np.abs(flat) >= 127
        saturated += int(hit.sum())
        total += int(flat.size)
        channels_saturated += int(hit.any(axis=1).sum())
        channels += int(flat.shape[0])
        if hit.any():
            worst.append((initializer.name, int(hit.sum())))
    worst.sort(key=lambda item: -item[1])
    return {"saturated": saturated, "total": total,
            "saturated_fraction": (saturated / total) if total else 0.0,
            "channels_with_saturation": channels_saturated, "channels": channels,
            "worst_layers": [{"tensor": name, "saturated": count} for name, count in worst[:3]]}


def build_graph_equivalent_model(params: Dict[str, object]):
    """搭一个**与 ONNX 图逐位等价**的 torch 模型：权重取自图（**已折 BN**），BN 置成精确恒等。

    **为什么必须换掉 `models.resnet18(weights=DEFAULT)`**：它的 Conv **还没折 BN**，而图里的权重
    **已经折过**——两者不是同一个张量（逐通道系数跨度 0.05 ~ 19.9）。旧版预检直接拿前者做 fake-quant，
    量的是"另一张权重上的量化"，于是报出的 `max_abs ≈ 3.9` 与真机/参考实现的 `≈ 22` 差了 5 倍多
    （`TROUBLESHOOTING.md` #28 里"预检与实测差 5 倍"那一问，根因就在这里；#30.3 曾把它当成
    "已否证"，那次否证用的是**同样不忠实**的模拟，所以现在要推翻）。

    BN 置成恒等的写法：`γ=1, β=0, μ=0, σ²=1-ε` → `(x-0)/√((1-ε)+ε)·1+0 = x`，**精确**不是近似。
    """
    model = models.resnet18(weights=None).eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
            module.running_mean.zero_()
            module.running_var.fill_(1.0 - float(module.eps))
    convs = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Conv2d)}
    if set(convs) != set(params["conv_weights"]):
        raise SystemExit(f"torch 模型与 ONNX 图的 Conv 集合不一致："
                         f"{sorted(set(convs) ^ set(params['conv_weights']))}")
    with torch.no_grad():
        for name, module in convs.items():
            if name not in params["conv_biases"]:
                raise SystemExit(f"ONNX 图里 {name} 没有 bias；折叠 BN 后每一层都该有 bias")
            if module.bias is None:
                # torchvision 的 Conv 后面接 BN 时 `bias=False` → `module.bias is None`。
                # 图里的 conv 是**折过 BN** 的，每一层都带 bias（折叠把 β 并进来了），
                # 所以这里必须把 bias 挂上，否则前向会少加一项。
                module.bias = nn.Parameter(torch.zeros(module.out_channels, dtype=torch.float32))
            module.weight.copy_(torch.from_numpy(params["conv_weights"][name]))
            module.bias.copy_(torch.from_numpy(params["conv_biases"][name]))
        model.fc.weight.copy_(torch.from_numpy(params["fc_weight"]))
        model.fc.bias.copy_(torch.from_numpy(params["fc_bias"]))
    return model, convs


def fake_quant_check(act_scales, weight_scales, calib_files, params: Dict[str, object],
                     limit: int = 8) -> Dict[str, object]:
    """建 TRT 引擎**之前**先预估量化损失：在**与图等价**的 torch 模型上做 fake-quant。

    为什么值得做：真机建引擎要几分钟，而"量化损失有多大"在 torch 里几秒就能估出来。

    这里模拟的是**图里真实发生的事**：每个卷积的输入按 `in_scale` 量化、输出按 `out_scale` 量化、
    权重按 `weight_scales` 量化。输出量化那一步不能省——图里 `Conv → Q → DQ → Relu/Add`
    是**两处量化**，只做输入量化就不是同一张图（`TROUBLESHOOTING.md` #30.4 的教训）。
    """
    def fake_quant(x: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.clamp(torch.round(x / scale), -127, 127) * scale

    def fake_quant_per_channel(w: torch.Tensor, scales: np.ndarray) -> torch.Tensor:
        # 动态 rank：`view(-1,1,1,1)` 只对 4-D 卷积权重成立，遇到 2-D（Gemm/FC）会直接抛
        # RuntimeError。当前调用点只遍历 nn.Conv2d（都是 4-D），所以**没有触发过**；
        # 但按"不写死隐式假设"的纪律改成通用形式，免得将来复用这个函数时踩坑。
        shape = [-1] + [1] * (w.dim() - 1)
        s = torch.from_numpy(scales).view(*shape).to(w.device)
        return torch.clamp(torch.round(w / s), -127, 127) * s

    ref_model, _ = build_graph_equivalent_model(params)
    q_model, q_convs = build_graph_equivalent_model(params)
    # **先把权重量化干净**，再挂激活 hook —— 不要在 forward hook 里就地改权重：
    # 那会让"参考"与"量化"两次前向的口径混在一起，报出来的差异没有意义。
    for name, module in q_convs.items():
        with torch.no_grad():
            module.weight.copy_(fake_quant_per_channel(module.weight, weight_scales[name]))

    handles = []
    for name, module in q_convs.items():
        scale_in = float(act_scales[name]["in"])
        scale_out = float(act_scales[name]["out"])
        handles.append(module.register_forward_pre_hook(
            lambda _m, inputs, s=scale_in: (fake_quant(inputs[0], s),)))
        handles.append(module.register_forward_hook(
            lambda _m, _inputs, output, s=scale_out: fake_quant(output, s)))

    diffs, mismatches = [], 0
    with torch.no_grad():
        for path in calib_files[:limit]:
            x = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(1, 3, 224, 224))
            ref = ref_model(x)
            got = q_model(x)          # 权重与激活都按同一批 scale 量化过
            diffs.append(float((ref - got).abs().max()))
            mismatches += int(ref.argmax(1).item() != got.argmax(1).item())
    for handle in handles:
        handle.remove()
    return {"checked_images": min(limit, len(calib_files)),
            "max_abs_vs_fp32": max(diffs) if diffs else None,
            "argmax_mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet18 FP32 ONNX → 对称 int8 Q/DQ ONNX")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calib-dir", default="0_resnet18_onnx/calib_data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib-images", type=int, default=500,
                        help="标定图片数（默认 500，与 legacy 校准集一致）")
    parser.add_argument("--calib-percentile", type=float, default=99.9,
                        help="激活裁剪值的分位（默认 99.9）。取 100 表示退化成 max 标定——"
                             "实测那样会让 argmax 掉一个，见 meta 里的 fake_quant 段")
    parser.add_argument("--calib-bins", type=int, default=2048)
    parser.add_argument("--weight-form", choices=["qdq", "prequant_dq"], default="prequant_dq",
                        help="权重的图形态。**默认 prequant_dq**（正式产物的形态）：在 Python 侧把权重"
                             "预量化成 int8 常量、图里只留 DequantizeLinear——ONNX 从 44.7 MB 降到 13.3 MB，"
                             "且实测与 `qdq` 形态**数值等价**（引擎 max_abs 逐位相同，见 TROUBLESHOOTING #31.3）。"
                             "`qdq` = `Q(const)→DQ`，保留用于对照/复现旧产物")
    parser.add_argument("--weight-scope", choices=["per_channel", "per_tensor"],
                        default="per_tensor",
                        help="权重量化粒度。**默认 per_tensor 是实测结论**：在 FP32 有余量的样本上，"
                             "per-tensor 的 top-1 一致率 100%（11/11），per-channel 只有 54.5%（6/11）。"
                             "**per-channel 更差的根因已于 2026-09-27 定位**——它配合默认的 "
                             "`--weight-range-source torchvision` 会拿**未折 BN** 的权重算 scale、"
                             "却量化**已折 BN** 的权重；改用 `--weight-range-source onnx` 后 per-channel "
                             "的余量子集一致率回到 100%（见 docs/TROUBLESHOOTING.md #46）。"
                             "**默认仍保持 per_tensor + torchvision 源**（= 现有正式产物，逐字节不变）")
    parser.add_argument("--weight-range-source", choices=["torchvision", "onnx"],
                        default="torchvision",
                        help="权重 scale 从哪张张量上统计。`torchvision` = 历史行为（来自未折 BN 的 "
                             "torchvision 模型，而 Q/DQ 插在已折 BN 的 ONNX 权重上——**两者不是同一个"
                             "张量**）；`onnx` = 从 ONNX 图里 Conv 实际用的权重上取（**与插入对象一致**）。"
                             "**默认保持 torchvision 以维持正式产物逐位不变**；P4-INT8-a 的根因正是这个"
                             "不一致（见 docs/TROUBLESHOOTING.md #46 与本函数上方 onnx_weight_ranges 的说明）")
    parser.add_argument("--skip-fake-quant", action="store_true")
    parser.add_argument("--fake-quant-images", type=int, default=32,
                        help="fake-quant 预检用多少张图（默认 32）。**别只看这 8 张就下结论**——"
                             "calib_data 是 tiny-imagenet 放大图，分类本身退化，样本太少时"
                             "argmax 一致性的判别力很弱（phase4_test_plan §4.1 同样的坑）")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.bin")))
    if len(files) < args.calib_images:
        raise SystemExit(f"标定图不足：{len(files)} < {args.calib_images}")

    model = onnx.load(args.onnx)
    collected, torchvision_weights = collect_ranges(files, args.calib_images, args.calib_bins)
    clips = percentile_clip(collected["hist"], collected["max"], args.calib_bins,
                            args.calib_percentile)
    params = onnx_graph_parameters(model)
    if set(params["conv_weights"]) != set(torchvision_weights):
        raise SystemExit(f"ONNX 图与 torchvision 的 Conv 集合不一致："
                         f"{sorted(set(params['conv_weights']) ^ set(torchvision_weights))}")
    weights = params["conv_weights"] if args.weight_range_source == "onnx" else torchvision_weights
    # **来源自检（P4-INT8-a 的根因护栏）**：算 scale 的那张张量，必须就是**被量化那张**。
    # 两者形状/数值不一致 = "尺子量 A、裁剪 B"——per-tensor 只会整体偏，per-channel 会逐通道错配
    # （系数 >1 的通道直接 clamp 饱和）。这里只**报**不拦：默认路径（torchvision）历史产物必须
    # 还能逐位复现，是否切换默认要作者拍板（见 docs/TROUBLESHOOTING.md #46.4）。
    mismatch_layers = []
    worst_mismatch = 0.0
    for name, tensor in weights.items():
        target = params["conv_weights"][name]
        if tensor.shape != target.shape:
            mismatch_layers.append(name)
            worst_mismatch = float("inf")
            continue
        scale_ref = max(float(np.abs(target).max()), 1e-30)
        relative = float(np.abs(tensor - target).max()) / scale_ref
        worst_mismatch = max(worst_mismatch, relative)
        if relative > 1e-6:
            mismatch_layers.append(name)
    act_scales, weight_scales = symmetrize(clips, weights, args.weight_scope)

    info = insert_qdq(model, act_scales, weight_scales,
                      weight_axis=(args.weight_scope == "per_channel"),
                      weight_form=args.weight_form)
    counts = check_qdq(model, info["qdq_pairs"])
    onnx.checker.check_model(model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    onnx.save(model, args.output)

    report = {"onnx": args.onnx, "output": args.output, "calib_images": args.calib_images,
              "weight_scope": args.weight_scope,
              "weight_form": args.weight_form,
              "weight_range_source": args.weight_range_source,
              "weight_range_source_check": {
                  "layers_with_mismatch": mismatch_layers,
                  "worst_relative_mismatch": worst_mismatch,
                  "meaning": ("scale 的来源张量 vs 图里被量化的权重张量；不一致 = '尺子量 A、裁剪 B'。"
                              "`torchvision` 源在折过 BN 的图上**必然**不一致（P4-INT8-a 的根因）"),
              },
              "saturated_int8_weights": saturation_stats(model),
              "calib_percentile": args.calib_percentile, "calib_bins": args.calib_bins,
              "clip_values": {k: {kk: float(vv) for kk, vv in v.items()}
                              for k, v in clips.items()},
              "observed_max": collected["max"],
              "qdq_pairs": info["qdq_pairs"], **counts,
              "act_scales": {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in act_scales.items()}}
    if not args.skip_fake_quant:
        report["fake_quant"] = fake_quant_check(act_scales, weight_scales, files, params,
                                                args.fake_quant_images)
    meta_path = os.path.splitext(args.output)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[quantize] {args.onnx} → {args.output}")
    print(f"  Q/DQ 对        : {info['qdq_pairs']}（{info['convs']} 个卷积 × 3）")
    print(f"  zero_point 自检 : 全部为 0 且 int8（TRT 要求对称）")
    saturation = report["saturated_int8_weights"]
    if saturation["total"]:
        print(f"  尺度自检        : 饱和(±127)权重 {saturation['saturated']}/{saturation['total']}"
              f" = {saturation['saturated_fraction'] * 100:.3f}%"
              f"（对称量化下每通道约 1 个；远高于此说明 scale 偏小）")
    if mismatch_layers:
        print(f"  [WARN] scale 的来源与量化对象**不是同一张张量**：{len(mismatch_layers)} 层不一致，"
              f"最大相对差 {worst_mismatch:.4g}")
        print(f"         → 这正是 P4-INT8-a 的根因（docs/TROUBLESHOOTING.md #46）；"
              f"加 --weight-range-source onnx 可修")
        print(f"  [WARN] 因此本产物**带有 #46 那个缺陷**：它的身份是「**#46 的复现样本**」，"
              f"**不是候选基线**。")
        print(f"         别拿它做 per-channel vs per-tensor 的对比、也别把它当成 per-channel 的"
              f"正确性参照——")
        print(f"         要对比就得先用 `--weight-range-source onnx` 重生成（见 "
              f"future_iterations_development_plan.md §13.11）。")
    print(f"  元数据          : {meta_path}")
    if "fake_quant" in report:
        fq = report["fake_quant"]
        print(f"  fake-quant 预估 : {fq['checked_images']} 张，"
              f"max_abs_vs_fp32 = {fq['max_abs_vs_fp32']:.4g}，"
              f"argmax 不一致 = {fq['argmax_mismatches']}/{fq['checked_images']}")


if __name__ == "__main__":
    main()
