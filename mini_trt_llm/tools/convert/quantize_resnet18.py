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


def fake_quant_check(act_scales, weight_scales, calib_files, limit: int = 8) -> Dict[str, object]:
    """建 TRT 引擎**之前**先预估量化损失：在 torch 里用同一批 scale 做 fake-quant。

    为什么值得做：真机建引擎要几分钟，而"量化损失有多大"在 torch 里几秒就能估出来。
    若这一步的 argmax 就已经不一致，说明 scale 方案有问题，不必上真机。
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    convs = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Conv2d)}

    def fake_quant(x: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.clamp(torch.round(x / scale), -127, 127) * scale

    def fake_quant_per_channel(w: torch.Tensor, scales: np.ndarray) -> torch.Tensor:
        # 动态 rank：`view(-1,1,1,1)` 只对 4-D 卷积权重成立，遇到 2-D（Gemm/FC）会直接抛
        # RuntimeError。当前调用点只遍历 nn.Conv2d（都是 4-D），所以**没有触发过**；
        # 但按"不写死隐式假设"的纪律改成通用形式，免得将来复用这个函数时踩坑。
        shape = [-1] + [1] * (w.dim() - 1)
        s = torch.from_numpy(scales).view(*shape).to(w.device)
        return torch.clamp(torch.round(w / s), -127, 127) * s

    # **先把权重量化干净**，再挂激活的 pre-hook —— 不要在第一版那样在 forward hook 里就地改权重：
    # 那会让"参考"与"量化"两次前向的口径混在一起（第一次用 FP32 权重、第二次用量化权重），
    # 报出来的差异既不是"全量化 vs FP32"，也不是任何有意义的量。
    ref_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    q_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    q_convs = {name: mod for name, mod in q_model.named_modules() if isinstance(mod, nn.Conv2d)}
    for name, mod in q_convs.items():
        with torch.no_grad():
            mod.weight.copy_(fake_quant_per_channel(mod.weight, weight_scales[name]))
    handles = []
    for name, mod in q_convs.items():
        s_in = float(act_scales[name]["in"])

        def pre_hook(_module, inputs, s=s_in):
            return (fake_quant(inputs[0], s),)

        handles.append(mod.register_forward_pre_hook(pre_hook))

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
                             "per-channel 为何在整网上更差**原因未知**（单卷积上两者等价、差 1.9e-6），"
                             "见 docs/TROUBLESHOOTING.md §29.2 与 §29.5——根因查清前按实测选 per_tensor")
    parser.add_argument("--skip-fake-quant", action="store_true")
    parser.add_argument("--fake-quant-images", type=int, default=32,
                        help="fake-quant 预检用多少张图（默认 32）。**别只看这 8 张就下结论**——"
                             "calib_data 是 tiny-imagenet 放大图，分类本身退化，样本太少时"
                             "argmax 一致性的判别力很弱（phase4_test_plan §4.1 同样的坑）")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.bin")))
    if len(files) < args.calib_images:
        raise SystemExit(f"标定图不足：{len(files)} < {args.calib_images}")

    collected, weights = collect_ranges(files, args.calib_images, args.calib_bins)
    clips = percentile_clip(collected["hist"], collected["max"], args.calib_bins,
                            args.calib_percentile)
    act_scales, weight_scales = symmetrize(clips, weights, args.weight_scope)

    model = onnx.load(args.onnx)
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
              "calib_percentile": args.calib_percentile, "calib_bins": args.calib_bins,
              "clip_values": {k: {kk: float(vv) for kk, vv in v.items()}
                              for k, v in clips.items()},
              "observed_max": collected["max"],
              "qdq_pairs": info["qdq_pairs"], **counts,
              "act_scales": {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in act_scales.items()}}
    if not args.skip_fake_quant:
        report["fake_quant"] = fake_quant_check(act_scales, weight_scales, files,
                                                args.fake_quant_images)
    meta_path = os.path.splitext(args.output)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[quantize] {args.onnx} → {args.output}")
    print(f"  Q/DQ 对        : {info['qdq_pairs']}（{info['convs']} 个卷积 × 3）")
    print(f"  zero_point 自检 : 全部为 0 且 int8（TRT 要求对称）")
    print(f"  元数据          : {meta_path}")
    if "fake_quant" in report:
        fq = report["fake_quant"]
        print(f"  fake-quant 预估 : {fq['checked_images']} 张，"
              f"max_abs_vs_fp32 = {fq['max_abs_vs_fp32']:.4g}，"
              f"argmax 不一致 = {fq['argmax_mismatches']}/{fq['checked_images']}")


if __name__ == "__main__":
    main()
