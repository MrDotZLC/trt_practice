#!/usr/bin/env python3
"""ONNX 图结构探针：把"这张图长什么样"变成可断言的检查。

为什么需要它：方案 B（ONNX + Plugin）的两件事都建立在这张图的结构上——
子图识别的目标、以及与方案 A 的对齐判据。图一旦变化（重新导出、换 opset、
换导出脚本），我们得**立刻知道**，而不是等引擎构建失败或数值对不上才发现。

用法：
    python inspect_onnx.py 1_gpt2_onnx/gpt2.onnx            # 打印结构摘要
    python inspect_onnx.py 1_gpt2_onnx/gpt2.onnx --check    # 与内置基线对比

依赖：onnx（见 requirements.txt）。

注意：`--check` 的基线是**首次实测值**（见 docs/phase3_development_plan.md §0.1），
不是"随便定的期望"。图变了就让 `--check` 红着，先判断变化是有意为之还是意外。
"""

import argparse
import collections
import json
import sys
from pathlib import Path

# ctest 的 SKIP_RETURN_CODE（见 tests/CMakeLists.txt）：缺环境时返回它表示"跳过"，
# 与"图结构不对（返回 1）"区分开——缺 onnx 包或缺 ONNX 文件都不代表图有问题。
SKIP_EXIT_CODE = 77

try:
    import onnx
except ImportError:  # pragma: no cover
    if "--skip-if-missing" in sys.argv:
        print("跳过：未安装 onnx 包（pip install -r requirements.txt）")
        sys.exit(SKIP_EXIT_CODE)
    print("需要 onnx 包：pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


# 首次实测基线（2026-09-25，1_gpt2_onnx/gpt2.onnx）。
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


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX 图结构探针")
    parser.add_argument("onnx_path", help="ONNX 文件路径")
    parser.add_argument("--check", action="store_true",
                        help="与内置基线对比（图变了就返回非零）")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印摘要")
    parser.add_argument("--skip-if-missing", action="store_true",
                        help="文件不存在时返回 SKIP_EXIT_CODE（供 ctest 使用）")
    args = parser.parse_args()

    path = Path(args.onnx_path)
    if not path.exists():
        if args.skip_if_missing:
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
        return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
