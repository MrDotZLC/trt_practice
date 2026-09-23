#!/usr/bin/env python3
"""Convert HuggingFace checkpoint to mini_trt_llm format.

Output (目录模式，对应 mini_trt_llm 的模型目录约定):
    {output_dir}/config.json
    {output_dir}/model.safetensors

Usage:
    # 目录模式：产出 config.json + model.safetensors
    python hf_to_mini_trt_llm.py \
        --model_name_or_path gpt2 \
        --output_dir ./gpt2_mini_trt_llm

    # 单文件模式：只产出 .safetensors
    python hf_to_mini_trt_llm.py --model_name_or_path ./ckpt --output model.safetensors

Supported inputs:
    - HuggingFace 模型目录（含 pytorch_model.bin 或分片 *.index.json 或 model.safetensors）
    - 单个 PyTorch 文件：.pth / .pt / .bin / .ckpt
    - 单个 safetensors 文件

Dependencies:
    See requirements.txt at the repository root.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可操作的提示
    print(f"Error: missing dependency ({exc}).", file=sys.stderr)
    print("Install with: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


def _load_single_file(path: Path) -> dict:
    """Load a single PyTorch checkpoint file and return its state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def _load_huggingface_sharded(model_dir: Path, index_name: str) -> dict:
    """Load weights referenced by a shard index file.

    Weights are grouped by shard first so every shard file is read exactly once;
    a naive per-key load would re-read large shards once per tensor.
    """
    index_path = model_dir / index_name
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Empty weight_map in {index_path}")

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_file in weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file, keys in shard_to_keys.items():
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")

        if shard_path.suffix == ".safetensors":
            shard_state = safetensors_load_file(str(shard_path))
        else:
            shard_state = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )

        for key in keys:
            if key not in shard_state:
                raise KeyError(f"Key '{key}' not found in {shard_path}")
            state_dict[key] = shard_state[key]

    return state_dict


def _load_from_directory(model_dir: Path) -> tuple[dict, Path | None]:
    """Load a HuggingFace model directory; return (state_dict, config_path)."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        config_path = None

    # 分片 .bin 优先于单片 .bin，避免只加载到部分权重
    bin_index = model_dir / "pytorch_model.bin.index.json"
    safe_index = model_dir / "model.safetensors.index.json"
    bin_file = model_dir / "pytorch_model.bin"
    safe_file = model_dir / "model.safetensors"

    if bin_index.exists():
        print(f"Loading sharded checkpoint from {model_dir}")
        return _load_huggingface_sharded(model_dir, bin_index.name), config_path
    if safe_index.exists():
        print(f"Loading sharded safetensors checkpoint from {model_dir}")
        return _load_huggingface_sharded(model_dir, safe_index.name), config_path
    if safe_file.exists():
        print(f"Loading safetensors checkpoint from {safe_file}")
        return safetensors_load_file(str(safe_file)), config_path
    if bin_file.exists():
        print(f"Loading HuggingFace checkpoint from {bin_file}")
        return torch.load(bin_file, map_location="cpu", weights_only=False), config_path

    raise FileNotFoundError(
        f"Directory {model_dir} contains none of: pytorch_model.bin, "
        "pytorch_model.bin.index.json, model.safetensors, model.safetensors.index.json"
    )


def load_state_dict(src: str) -> tuple[dict, Path | None]:
    """Resolve `src` (local dir / local file) to a state dict plus optional config."""
    src_path = Path(src)

    if src_path.is_dir():
        return _load_from_directory(src_path)

    if src_path.is_file():
        if src_path.suffix == ".safetensors":
            print(f"Loading safetensors file from {src_path}")
            return safetensors_load_file(str(src_path)), None
        print(f"Loading PyTorch checkpoint from {src_path}")
        return _load_single_file(src_path), None

    # 非本地路径视为 HuggingFace repo id，交由 transformers 拉取（需要网络）
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise FileNotFoundError(
            f"'{src}' is not a local path, and transformers is not installed "
            "for downloading from the HuggingFace Hub."
        )

    print(f"Downloading '{src}' from the HuggingFace Hub")
    model = AutoModelForCausalLM.from_pretrained(src)
    return dict(model.state_dict()), None


def convert(src: str, output_dir: Path | None, output_file: Path | None) -> None:
    state_dict, config_path = load_state_dict(src)

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Loaded checkpoint is {type(state_dict).__name__}, "
            "expected a state dict mapping"
        )

    # 过滤非张量条目（部分 checkpoint 会混入优化器状态 / 元数据）
    tensor_dict = {
        k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)
    }
    if not tensor_dict:
        raise ValueError("No tensors found in checkpoint")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensor_dict, str(output_file))

    if output_dir is not None:
        # mini_trt_llm 的 ModelConfig::Load 要求 config.json 存在；没有源配置时
        # 写一个最小骨架，让后续人工补全模型结构字段。
        dest_config = output_dir / "config.json"
        if config_path is not None:
            shutil.copyfile(config_path, dest_config)
            print(f"Copied config: {dest_config}")
        else:
            skeleton = {
                "model_type": "",
                "architecture": "",
                "hyper_params": {},
                "weight_map": {},
            }
            with open(dest_config, "w", encoding="utf-8") as f:
                json.dump(skeleton, f, indent=4)
            print(f"Wrote config skeleton (needs manual fill-in): {dest_config}")

    total_params = sum(v.numel() for v in tensor_dict.values())
    print(
        f"Saved {len(tensor_dict)} tensors "
        f"({total_params / 1e6:.2f}M parameters) to {output_file}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace checkpoint to mini_trt_llm format"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Local HF dir / checkpoint file / HF repo id",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Alias for the positional input (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for config.json and model.safetensors",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output a single .safetensors file (mutually exclusive with --output_dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    src = args.model_name_or_path or args.input
    if not src:
        print(
            "Error: provide a model via --model_name_or_path or a positional input.",
            file=sys.stderr,
        )
        return 1

    if args.output and args.output_dir:
        print("Error: --output and --output_dir are mutually exclusive.", file=sys.stderr)
        return 1

    # 默认走目录模式，直接产出 mini_trt_llm 可直接加载的模型目录
    if args.output:
        output_dir, output_file = None, Path(args.output)
    else:
        output_dir = Path(args.output_dir or "converted")
        output_file = output_dir / "model.safetensors"

    print(f"[INFO] Model: {src}")
    print(f"[INFO] Output: {output_file}")

    try:
        convert(src, output_dir, output_file)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
