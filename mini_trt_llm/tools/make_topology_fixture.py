#!/usr/bin/env python3
"""造"注意力骨架"夹具，用来证明连接级判据（`inspect_onnx.py --check-topology`）**会拦人**。

为什么需要它：S1 把识别从"计数"升级到"连接级"，判据是"能拒绝**计数相同、连接不同**的图"。
**护栏没有反例证明它会拦人，就等于没有护栏**——本项目已经吃过这个亏
（`docs/dev/REQ-006-gpt2-onnx/phase3_test_plan.md` + PH3-GAPS 的 G1c）。

三个夹具（`--self-test` 一次全跑）：

| 夹具 | 造法 | 期望 |
|---|---|---|
| `good`（F3） | 结构正确的两层注意力骨架 | `--check-topology` **通过**（证明它不会把对的判错） |
| `swap-softmax-inputs`（F1） | 两个块的 Softmax **输入对调** | **失败**（T2：V 侧与 Q/K 侧不再来自同一个 Split） |
| `share-score`（F2） | 两个 Softmax **共用**同一个 score 张量（原 score 变成孤儿节点，**计数不变**） | **失败**（T1：块边界重叠） |

两个反例都**不增删节点**，只改接线——这正是旧护栏（只数算子个数）抓不到的那类变化。

用法：
    python make_topology_fixture.py --output /tmp/fixture.onnx --mode good
    python make_topology_fixture.py --self-test            # 三个一起造并逐个断言
    python make_topology_fixture.py --self-test --skip-if-missing

环境语义与 `inspect_onnx.py` 同口径：缺 `onnx` 包时 `--skip-if-missing` 返回 77（跳过），
否则返回 1（失败）。夹具**不依赖 652 MB 资产、不依赖 GPU**，沙箱即可跑。
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP_EXIT_CODE = 77

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # pragma: no cover
    if "--skip-if-missing" in sys.argv:
        print("跳过：未安装 onnx 包（pip install -r requirements.txt）")
        sys.exit(SKIP_EXIT_CODE)
    print("需要 onnx 包：pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

import numpy as np

# 夹具规模刻意取最小：它只证明"接线判据会拦人"，不需要数值意义。
# （真正的块数断言（12）在 `inspect_onnx.py --check` 的基线上，不在这里。）
HIDDEN = 4
SEQ = 2
VOCAB = 8
MAX_POS = 8
NUM_BLOCKS = 2


def _build(mode: str):
    """按 mode 造图。只改接线，不增删节点（反例必须保持计数不变）。"""
    rng = np.random.default_rng(0)

    nodes = []
    initializers = [
        numpy_helper.from_array(rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32), "wte"),
        numpy_helper.from_array(rng.standard_normal((MAX_POS, HIDDEN)).astype(np.float32), "wpe"),
    ]

    nodes.append(helper.make_node("Gather", ["wte", "input_ids"], ["tok_emb"], axis=0))
    nodes.append(helper.make_node("Gather", ["wpe", "position_ids"], ["pos_emb"], axis=0))
    nodes.append(helper.make_node("Add", ["tok_emb", "pos_emb"], ["x_0"]))

    # **夹具的块是"并联"的**（每个块都读同一个 `x_0`），不是真实图的串联残差链。
    # 为什么：跨块重连（F1 / F2）在串联结构里会造出**真实的依赖环**
    # （块 0 的输出喂块 1，块 1 的 score 又喂块 0 的 Softmax）——ONNX 要求 DAG，checker 会直接拒绝，
    # 那样验的就不是判据而是"图不合法"。而 T1–T4 都是**块内局部**判据，不依赖块之间是否串联。
    x = "x_0"
    score_names = []
    softmax_names = []
    proj_names = []
    for i in range(NUM_BLOCKS):
        scale = f"ln{i}_scale"
        initializers.append(
            numpy_helper.from_array(np.ones((HIDDEN,), dtype=np.float32), scale))
        wqkv = f"wqkv{i}"
        initializers.append(
            numpy_helper.from_array(rng.standard_normal((HIDDEN, 3 * HIDDEN)).astype(np.float32), wqkv))
        wproj = f"wproj{i}"
        initializers.append(
            numpy_helper.from_array(rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32), wproj))
        split_sizes = f"split_sizes{i}"
        initializers.append(
            numpy_helper.from_array(np.array([HIDDEN, HIDDEN, HIDDEN], dtype=np.int64), split_sizes))

        nodes.append(helper.make_node("LayerNormalization", [x, scale], [f"h{i}"], axis=-1))
        nodes.append(helper.make_node("MatMul", [f"h{i}", wqkv], [f"qkv{i}"]))
        nodes.append(helper.make_node(
            "Split", [f"qkv{i}", split_sizes], [f"q{i}", f"k{i}", f"v{i}"], axis=2))
        for name in ("q", "k", "v"):
            nodes.append(helper.make_node(
                "Transpose", [f"{name}{i}"], [f"t{name}{i}"], perm=[0, 2, 1]))
        nodes.append(helper.make_node("MatMul", [f"tq{i}", f"tk{i}"], [f"scores{i}"]))
        score_names.append(f"scores{i}")
        nodes.append(helper.make_node("Softmax", [f"scores{i}"], [f"soft{i}"], axis=-1))
        softmax_names.append(f"soft{i}")
        nodes.append(helper.make_node("MatMul", [f"soft{i}", f"tv{i}"], [f"ctx{i}"]))
        nodes.append(helper.make_node("Transpose", [f"ctx{i}"], [f"tctx{i}"], perm=[0, 2, 1]))
        nodes.append(helper.make_node("MatMul", [f"tctx{i}", wproj], [f"proj{i}"]))
        proj_names.append(f"proj{i}")

    # ---- 反例：只改 Softmax 的输入张量，节点集合与各算子计数**不变** ----
    if mode == "swap-softmax-inputs":
        _set_input(nodes, softmax_names[0], 0, score_names[1])
        _set_input(nodes, softmax_names[1], 0, score_names[0])
    elif mode == "share-score":
        # 两个 Softmax 都读 scores1；scores0 成为孤儿输出（仍在图里，所以计数不变）
        _set_input(nodes, softmax_names[0], 0, score_names[1])
    elif mode != "good":
        raise ValueError(f"未知 mode: {mode}")

    acc = proj_names[0]
    for j, name in enumerate(proj_names[1:], start=1):
        out = f"merged_{j}"
        nodes.append(helper.make_node("Add", [acc, name], [out]))
        acc = out
    nodes.append(helper.make_node("Identity", [acc], ["logits"]))

    # 反例是"改接线"造出来的，改完的节点顺序可能不再满足 ONNX 的**拓扑有序**要求
    # （例如 soft0 引用了后面才产出的 scores1）。checker 会直接拒绝这种图，
    # 所以要重排一次：只调整书写顺序，不动任何接线。
    available = {"input_ids", "position_ids"} | {i.name for i in initializers}
    ordered = []
    pending = list(nodes)
    while pending:
        ready = [n for n in pending if all(t in available or not t for t in n.input)]
        if not ready:
            raise AssertionError("夹具的节点图有环，无法拓扑排序")
        for node in ready:
            ordered.append(node)
            available.update(t for t in node.output if t)
            pending.remove(node)

    graph = helper.make_graph(
        nodes=ordered,
        name=f"topology_fixture_{mode}",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch", "seq"]),
            helper.make_tensor_value_info("position_ids", TensorProto.INT32, ["batch", "seq"]),
        ],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                               ["batch", "seq", HIDDEN])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # 与 onnx 1.16+ 兼容的 IR 版本（同 make_tiny_onnx.py）
    onnx.checker.check_model(model)
    return model


def _set_input(nodes, producer_output: str, index: int, new_tensor: str) -> None:
    """把"产出 producer_output 的那个节点"的第 index 个输入换成 new_tensor。"""
    for node in nodes:
        if producer_output in node.output:
            node.input[index] = new_tensor
            return
    raise AssertionError(f"找不到产出 {producer_output} 的节点")


def _run_topology_check(onnx_path: Path) -> int:
    """调 `inspect_onnx.py --check-topology`，返回它的退出码。"""
    probe = Path(__file__).with_name("inspect_onnx.py")
    proc = subprocess.run(
        [sys.executable, str(probe), str(onnx_path), "--check-topology"],
        capture_output=True, text=True)
    if proc.stdout.strip():
        for line in proc.stdout.rstrip().splitlines():
            print(f"      {line}")
    if proc.returncode not in (0, 1):
        print(f"      探针异常退出（{proc.returncode}）：{proc.stderr.strip()}", file=sys.stderr)
    return proc.returncode


def self_test() -> int:
    cases = [("good", 0), ("swap-softmax-inputs", 1), ("share-score", 1)]
    failures = []
    counts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode, expected in cases:
            path = Path(tmp) / f"fixture_{mode}.onnx"
            model = _build(mode)
            # 自证前提：反例必须**不增删算子**，否则验的就不是"计数相同、连接不同"。
            # 没有这条断言，夹具很容易在演化中悄悄改成"多/少一个算子"，
            # 而那样连旧护栏（只数个数）都能抓到，等于白验。
            counts[mode] = sorted(
                (node.op_type, sum(1 for n in model.graph.node if n.op_type == node.op_type))
                for node in model.graph.node
            )
            onnx.save(model, str(path))
            print(f"  [{mode}] 期望退出码 {expected}")
            actual = _run_topology_check(path)
            if actual != expected:
                failures.append(f"{mode}: 退出码 {actual} != 期望 {expected}")
            else:
                print(f"      → 符合期望（{actual}）")

    if len({tuple(v) for v in counts.values()}) != 1:
        failures.append(f"三个夹具的算子计数不一致（前提被破坏）：{counts}")

    if failures:
        print("夹具自检失败：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("夹具自检通过：good 被接受；swap-softmax-inputs / share-score 被拒（计数均未变）。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="注意力骨架夹具（连接级判据的反例来源）")
    parser.add_argument("--output", help="输出 ONNX 路径（与 --self-test 二选一）")
    parser.add_argument("--mode", default="good",
                        choices=["good", "swap-softmax-inputs", "share-score"])
    parser.add_argument("--self-test", action="store_true",
                        help="造三个夹具并逐个断言 --check-topology 的结论")
    parser.add_argument("--skip-if-missing", action="store_true",
                        help="缺环境时返回 SKIP_EXIT_CODE（供 ctest 使用）")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.output:
        parser.error("需要 --output 或 --self-test")
    onnx.save(_build(args.mode), args.output)
    print(f"已生成 {args.output}（mode={args.mode}, blocks={NUM_BLOCKS}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
