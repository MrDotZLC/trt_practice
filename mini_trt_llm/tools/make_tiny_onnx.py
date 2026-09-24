#!/usr/bin/env python3
"""生成极小的 ONNX 夹具，用于测试"图契约不符"的失败路径。

为什么需要它：`BuildFromOnnx` 会校验图的 I/O 名（必须是 `input_ids` / `logits`），
而仓库里的真实模型（GPT-2 / ResNet18）**都满足或都不满足**同一组条件——
要测"满足一半"的情况（例如输入名对、输出名不对），必须自己造一张图。

为什么用 Python 而不是 C++：手写 ONNX protobuf 既冗长又易错，而 `onnx.helper`
几行就能拼出来。生成器只在需要夹具时调用，不进构建流程。

用法：
    python make_tiny_onnx.py --output /tmp/tiny.onnx --output-name not_logits
    python make_tiny_onnx.py --output /tmp/tiny2.onnx --input-name not_input_ids

图的内容刻意做到最简（一个 Identity）：本夹具只用于校验 I/O 名，
真正被解析的图越简单，失败原因就越不可能来自"算子不支持"。
输入用 float：TRT 对 float 的 Identity 支持最稳，避免 int64 之类的额外变量干扰判断。
"""

import argparse
import sys

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:  # pragma: no cover
    print("需要 onnx 包：pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


def build(input_name: str, output_name: str):
    # [batch, seq] 两维皆动态，与真实模型一致
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", [input_name], [output_name])],
        name="tiny_io_fixture",
        inputs=[helper.make_tensor_value_info(
            input_name, TensorProto.FLOAT, ["batch", "seq"])],
        outputs=[helper.make_tensor_value_info(
            output_name, TensorProto.FLOAT, ["batch", "seq"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # 与 onnx 1.16+ 兼容的 IR 版本
    onnx.checker.check_model(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="生成极小 ONNX I/O 夹具")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--input-name", default="input_ids")
    parser.add_argument("--output-name", default="not_logits")
    args = parser.parse_args()

    onnx.save(build(args.input_name, args.output_name), args.output)
    print(f"已生成 {args.output}（input={args.input_name}, output={args.output_name}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
