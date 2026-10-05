#!/usr/bin/env python3
"""ONNX 图结构探针：把"这张图长什么样"变成可断言的检查。

为什么需要它：方案 B（ONNX + Plugin）的两件事都建立在这张图的结构上——
子图识别的目标、以及与方案 A 的对齐判据。图一旦变化（重新导出、换 opset、
换导出脚本），我们得**立刻知道**，而不是等引擎构建失败或数值对不上才发现。

用法：
    python inspect_onnx.py assets/legacy/gpt2_onnx/gpt2.onnx            # 打印结构摘要
    python inspect_onnx.py assets/legacy/gpt2_onnx/gpt2.onnx --check    # 与内置基线对比

依赖：onnx（见 requirements.txt）。

注意：`--check` 的基线是**首次实测值**（见 docs/dev/REQ-006-gpt2-onnx/phase3_development_plan.md + PH3-ASSETS），
不是"随便定的期望"。图变了就让 `--check` 红着，先判断变化是有意为之还是意外。
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

# ctest 的 SKIP_RETURN_CODE（见 tests/CMakeLists.txt）：缺环境时返回它表示"跳过"，
# 与"图结构不对（返回 1）"区分开——缺 onnx 包或缺 ONNX 文件都不代表图有问题。
SKIP_EXIT_CODE = 77


def require_assets() -> bool:
    """`MINI_TRT_REQUIRE_ASSETS=1` 表示"本环境必须具备测试资产"。

    为什么需要：ctest 把"跳过"记成 Passed，于是**缺资产导致的覆盖下降对 CI 不可见**
    （实测 2026-09-27：把两个历史示例工程改名后全量仍报 265 条 / 100% passed / 0 failed，
    只有跳过集合变了 3 项；见 docs/dev/REQ-009-retire-legacy/phase5_development_plan.md 阶段 0）。设了这个变量，
    缺资产返回 1（失败）而不是 77（跳过）。

    它只管**资产**缺失；缺 Python 包属环境问题，仍按跳过处理。
    **`tools/convert/onnx_to_mini_trt_llm.py` 里有同名同义的实现**，改一处要两处同改。
    """
    return os.environ.get("MINI_TRT_REQUIRE_ASSETS", "") not in ("", "0")

try:
    import onnx
except ImportError:  # pragma: no cover
    if "--skip-if-missing" in sys.argv:
        print("跳过：未安装 onnx 包（pip install -r requirements.txt）")
        sys.exit(SKIP_EXIT_CODE)
    print("需要 onnx 包：pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


# 首次实测基线（2026-09-25，assets/legacy/gpt2_onnx/gpt2.onnx）。
# 只记录"结构性事实"，不记录与实现无关的数字——这些是子图识别与对齐判据的前提。
BASELINE = {
    "opset": 17,
    "inputs": ["input_ids"],
    "outputs": ["logits"],
    "initializer_count": 149,
    # 关键算子计数：注意力与归一化的形态由它们决定
    "ops": {
        "LayerNormalization": 25,
        "Tanh": 12,
        "Softmax": 12,
        "Gemm": 48,
        "MatMul": 25,
        "Split": 12,
        "Transpose": 48,
    },
    # GPT-2 的位置编码是学习式的：既没有 RMSNorm 也没有 RoPE 节点。
    # 这条断言是 Phase 3 §0.2 那个结论的护栏——若将来出现这些算子，
    # 说明图变了（换了模型或换了导出路径），计划里的"无可替换对象"结论需要重审。
    "absent_ops": ["RMSNormalization", "RotaryEmbedding", "RoPE", "Attention"],
}

# 位置编码"学习式"的判据用到的算子名（与 BASELINE["absent_ops"] 同源，改一处要两处同改）。
ROPE_LIKE_OPS = ("RotaryEmbedding", "RoPE")

# T2 允许"穿过"的搬运类算子：导出器换版本时这些会多一层或少一层，
# 但不改变"张量从哪来"的语义。不放进来的算子是**有语义**的（Mul/Add/MatMul/Gemm…），
# 让它们挡住追溯路径正是判据要抓的东西。
PASS_THROUGH_OPS = ("Transpose", "Reshape", "Squeeze", "Unsqueeze", "Cast", "Identity")


def summarize(model) -> dict:
    graph = model.graph
    dims = lambda value: [
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in value.type.tensor_type.shape.dim
    ]
    return {
        "opset": max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
                     default=0),
        "inputs": [f"{i.name}{dims(i)}" for i in graph.input],
        "outputs": [f"{o.name}{dims(o)}" for o in graph.output],
        "input_names": [i.name for i in graph.input],
        "output_names": [o.name for o in graph.output],
        "initializer_count": len(graph.initializer),
        "op_counts": dict(collections.Counter(n.op_type for n in graph.node)),
        "node_count": sum(1 for _ in graph.node),
    }


def recognize_subgraphs(model) -> dict:
    """识别三类子图的存在与形态（A4 冻结的范围：不做通用子图匹配器）。

    识别结果只用于"断言 + 报告"，当前阶段不做替换（D1=C）。
    """
    ops = collections.Counter(n.op_type for n in model.graph.node)
    # (i) 注意力块：QKV 投影用 Gemm，注意力分数用 MatMul + Softmax，再用 Transpose 换轴。
    #     三类计数必须匹配（每个 block 一套），否则说明图的分解方式变了。
    attention_blocks = ops.get("Softmax", 0)
    # (ii) LayerNorm：每 block 两处 + 最后一处
    layernorm = ops.get("LayerNormalization", 0)
    # (iii) 位置编码：GPT-2 是学习式的——wpe 走 Gather，而不是 RoPE 的
    #       Cos/Sin 常量 + Mul/Add 组合。若图上出现 RoPE 特征算子，说明模型换类了。
    rope_like = ops.get("RotaryEmbedding", 0) + ops.get("RoPE", 0)
    return {
        "attention": {
            "blocks(Softmax)": attention_blocks,
            "matmul": ops.get("MatMul", 0),
            "transpose": ops.get("Transpose", 0),
        },
        "layernorm": {"count": layernorm},
        "position_embedding": {"learned_gather_based": rope_like == 0,
                               "rope_like_ops": rope_like},
    }


def check(summary: dict) -> int:
    failures = []
    base = BASELINE

    if summary["opset"] != base["opset"]:
        failures.append(f"opset: {summary['opset']} != {base['opset']}")
    if summary["input_names"] != base["inputs"]:
        failures.append(f"输入: {summary['input_names']} != {base['inputs']}")
    if summary["output_names"] != base["outputs"]:
        failures.append(f"输出: {summary['output_names']} != {base['outputs']}")
    if summary["initializer_count"] != base["initializer_count"]:
        failures.append(
            f"initializer 数: {summary['initializer_count']} != {base['initializer_count']}")
    for op, expected in base["ops"].items():
        actual = summary["op_counts"].get(op, 0)
        if actual != expected:
            failures.append(f"{op} 计数: {actual} != {expected}")
    for op in base["absent_ops"]:
        if summary["op_counts"].get(op, 0) != 0:
            failures.append(
                f"{op} 出现了（{summary['op_counts'][op]} 个）：图结构变了，"
                "Phase 3 §0.2 的\"无替换对象\"结论需要重审")

    if failures:
        print("图结构已偏离基线：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("图结构与基线一致（opset / I-O / 算子分布 / 初始器数量）。")
    return 0


def _producers(graph) -> dict:
    """张量名 → 生产它的节点下标。"""
    out = {}
    for idx, node in enumerate(graph.node):
        for name in node.output:
            if name:
                out[name] = idx
    return out


def _consumers(graph) -> dict:
    """张量名 → 消费它的节点下标列表（保持出现顺序）。"""
    out = {}
    for idx, node in enumerate(graph.node):
        for name in node.input:
            if name:
                out.setdefault(name, []).append(idx)
    return out


def _trace_back(nodes, producers, start_tensor, wanted_ops, max_hops=8):
    """从某个张量往回走，穿过 PASS_THROUGH_OPS，收集 op_type ∈ wanted_ops 的节点下标。

    返回的是**集合**：同一个张量可能经多条路径到达同一类算子（例如 K 与 V 都来自 Split），
    比较集合是否相等比比较"第一个命中的节点"更稳——后者依赖遍历顺序。
    """
    found = set()
    frontier = [start_tensor]
    for _ in range(max_hops):
        nxt = []
        for tensor in frontier:
            idx = producers.get(tensor)
            if idx is None:
                continue
            node = nodes[idx]
            if node.op_type in wanted_ops:
                found.add(idx)
                continue
            if node.op_type in PASS_THROUGH_OPS:
                nxt.extend(t for t in node.input if t)
            # 其它算子：语义边界，停止追溯（正是不让 MatMul/Gemm 穿过的原因）
        if not nxt:
            break
        frontier = nxt
    return found


def _trace_forward(nodes, consumers, start_tensor, wanted_ops, max_hops=8):
    """从某个张量往前走，穿过 PASS_THROUGH_OPS，收集 op_type ∈ wanted_ops 的节点下标。"""
    found = set()
    frontier = [start_tensor]
    for _ in range(max_hops):
        nxt = []
        for tensor in frontier:
            for idx in consumers.get(tensor, []):
                node = nodes[idx]
                if node.op_type in wanted_ops:
                    found.add(idx)
                    continue
                if node.op_type in PASS_THROUGH_OPS:
                    nxt.extend(t for t in node.output if t)
        if not nxt:
            break
        frontier = nxt
    return found


def check_topology(model):
    """连接级判据：T1 块边界不共享 / T2 块内自洽 / T3 输出投影 / T4 位置编码形态。

    **与 `check()` 的分工**：数量（12 个 Softmax、25 处 LayerNorm…）由 `check()` 盯；
    这里只盯"谁喂谁"。两者都要过——`--check` 抓"换模型/换导出器"，这里抓"重新接线"。

    返回 (failures, summary)；failures 为空表示通过。
    """
    graph = model.graph
    nodes = list(graph.node)
    producers = _producers(graph)
    consumers = _consumers(graph)
    failures = []
    summary = {"softmax_blocks": 0, "blocks": []}

    softmax_idx = [i for i, n in enumerate(nodes) if n.op_type == "Softmax"]
    summary["softmax_blocks"] = len(softmax_idx)
    if not softmax_idx:
        failures.append("T1 图里没有 Softmax——注意力块无法切边界")
        return failures, summary

    # T1：score 张量不得被多个 Softmax 共用（共用即"两个块叠在一起"）
    by_score = {}
    for i in softmax_idx:
        ins = [t for t in nodes[i].input if t]
        if not ins:
            failures.append(f"T2 Softmax 无输入（节点 {nodes[i].name or i}）")
            continue
        by_score.setdefault(ins[0], []).append(i)
    for score, users in by_score.items():
        if len(users) > 1:
            failures.append(
                f"T1 score 张量 '{score}' 被 {len(users)} 个 Softmax 共用（块边界重叠）")

    # T2 / T3：逐块核对"Q/K 的 Split == V 的 Split"与"输出经 Gemm 回到主线"
    for i in softmax_idx:
        tag = nodes[i].name or f"#{i}"
        ins = [t for t in nodes[i].input if t]
        if not ins:
            continue
        score_tensor = ins[0]
        score_idx = producers.get(score_tensor)
        if score_idx is None:
            failures.append(f"T2 Softmax({tag}) 的输入 '{score_tensor}' 不是图内节点的输出")
            continue
        score_node = nodes[score_idx]
        if score_node.op_type not in ("MatMul", "Gemm"):
            failures.append(
                f"T2 Softmax({tag}) 的 score 生产者是 {score_node.op_type}，"
                "不是 MatMul/Gemm（QK^T）")
            continue

        qk_splits = set()
        ok = True
        for tensor in [t for t in score_node.input if t]:
            hit = _trace_back(nodes, producers, tensor, {"Split"})
            if not hit:
                failures.append(
                    f"T2 Softmax({tag}) 的 score 输入 '{tensor}' 追不到 Split"
                    "（块边界切不出来）")
                ok = False
            qk_splits |= hit
        if not ok:
            continue

        # V 侧：Softmax 之后第一个 MatMul/Gemm 即"乘 V"那一步
        outs = [t for t in nodes[i].output if t]
        ctx = []
        for t in outs:
            ctx.extend(sorted(_trace_forward(nodes, consumers, t, {"MatMul", "Gemm"})))
        if not ctx:
            failures.append(f"T3 Softmax({tag}) 的下游追不到 MatMul/Gemm（乘 V / 输出投影）")
            continue
        ctx_idx = ctx[0]
        v_splits = set()
        for tensor in [t for t in nodes[ctx_idx].input if t][1:]:
            v_splits |= _trace_back(nodes, producers, tensor, {"Split"})

        if not v_splits:
            failures.append(f"T2 Softmax({tag}) 的 V 侧追不到 Split")
            continue
        if v_splits != qk_splits:
            failures.append(
                f"T2 Softmax({tag}) 的 Q/K 与 V 不来自同一个 Split"
                f"（Q/K←{sorted(qk_splits)}，V←{sorted(v_splits)}）——块内不自洽")

        proj = _trace_forward(nodes, consumers, nodes[ctx_idx].output[0],
                              {"Gemm", "MatMul"})
        if not proj:
            failures.append(f"T3 Softmax({tag}) 的输出没有经 Gemm/MatMul 回到主线")

        summary["blocks"].append({
            "softmax": tag,
            "splits": sorted(qk_splits),
            "ctx_matmul": nodes[ctx_idx].name or ctx_idx,
            "proj": sorted(proj),
        })

    # T4：位置编码仍是学习式（有查表、且没有 RoPE 类算子）
    # **缩窄说明**（见 s1_topology_interface_spec.md §7 第 4 条）：原定义还要求"位置编码先于
    # 第一个注意力块进入主线"，那半条需要真实图才能核对（本机无资产），记为待核实；
    # 这里只断言"有 Gather 且无 RoPE 类算子"——**缩窄不是放宽阈值**，顺序那半条没被写进断言。
    op_counts = {}
    for node in nodes:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    rope_like = sum(op_counts.get(op, 0) for op in ROPE_LIKE_OPS)
    if rope_like:
        failures.append(f"T4 出现 RoPE 类算子（{rope_like} 个）——位置编码不再是学习式查表")
    if op_counts.get("Gather", 0) == 0:
        failures.append("T4 找不到 Gather——位置编码不是学习式查表形态")
    summary["gather"] = op_counts.get("Gather", 0)

    return failures, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX 图结构探针")
    parser.add_argument("onnx_path", help="ONNX 文件路径")
    parser.add_argument("--check", action="store_true",
                        help="与内置基线对比（图变了就返回非零）")
    parser.add_argument("--check-topology", action="store_true",
                        help="连接级判据：块边界不共享 / 块内自洽 / 输出投影 / 位置编码形态")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印摘要")
    parser.add_argument("--skip-if-missing", action="store_true",
                        help="文件不存在时返回 SKIP_EXIT_CODE（供 ctest 使用）")
    args = parser.parse_args()

    path = Path(args.onnx_path)
    if not path.exists():
        if args.skip_if_missing:
            if require_assets():
                print(f"缺资产（MINI_TRT_REQUIRE_ASSETS=1）：找不到 {path}", file=sys.stderr)
                return 1
            print(f"跳过：找不到 {path}（该文件不在版本控制里，属环境依赖）")
            return SKIP_EXIT_CODE
        print(f"找不到 {path}", file=sys.stderr)
        return 1

    # 只读结构，不加载外部权重数据（大模型的权重在 initializer 里，但不解析也没关系）
    model = onnx.load(str(path), load_external_data=False)
    summary = summarize(model)
    subgraphs = recognize_subgraphs(model)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"opset        : {summary['opset']}")
        print(f"输入         : {summary['inputs']}")
        print(f"输出         : {summary['outputs']}")
        print(f"节点数       : {summary['node_count']}")
        print(f"initializer  : {summary['initializer_count']}")
        print("算子分布     :")
        for op, count in sorted(summary["op_counts"].items(), key=lambda kv: -kv[1]):
            print(f"  {op:<24} {count}")

    status = 0
    if args.check:
        status = check(summary)
        # 三项识别断言（注意力 / LayerNorm / 位置编码）。
        # 数字来自 §0.1 的实测：12 个 block、25 处 LayerNorm、位置编码是学习式的。
        problems = []
        if subgraphs["attention"]["blocks(Softmax)"] != 12 or \
                subgraphs["attention"]["matmul"] != 25 or \
                subgraphs["attention"]["transpose"] != 48:
            problems.append(f"注意力块形态变了: {subgraphs['attention']}")
        if subgraphs["layernorm"]["count"] != 25:
            problems.append(f"LayerNorm 数量变了: {subgraphs['layernorm']}")
        if not subgraphs["position_embedding"]["learned_gather_based"]:
            problems.append(
                "位置编码不再是学习式 Gather，而是 RoPE 类算子——"
                "GPT-2 的图变了，方案 B 的结论需重审")
        if problems:
            print("子图识别失败：")
            for item in problems:
                print(f"  - {item}")
            return 1
        print(f"子图识别通过：注意力 {subgraphs['attention']}；"
              f"LayerNorm {subgraphs['layernorm']['count']} 处；"
              f"位置编码为学习式（无 RoPE 类算子）。")

    # 连接级判据（T1–T4）。与 `--check` 的计数判据**各管一半**：两条都要过。
    # 为什么单独一条开关：夹具图的算子分布本来就不等于真实图基线，调 `--check` 必然红——
    # 那不是夹具的问题，所以夹具自检只调这一条（见 s1_topology_interface_spec.md §7.1）。
    if args.check_topology:
        failures, topology = check_topology(model)
        if failures:
            print("拓扑识别失败：")
            for item in failures:
                print(f"  - {item}")
            return 1
        print(f"拓扑识别通过：{topology['softmax_blocks']} 个注意力块——"
              f"块内 Q/K/V 同源、输出经投影回到主线、位置编码为学习式查表"
              f"（Gather {topology['gather']} 处）。")

    return status


if __name__ == "__main__":
    sys.exit(main())
