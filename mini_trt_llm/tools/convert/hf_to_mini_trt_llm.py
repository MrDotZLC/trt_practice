#!/usr/bin/env python3
"""Convert HuggingFace checkpoint to mini_trt_llm format.

Output:
    {output_dir}/config.json
    {output_dir}/model.safetensors

Usage:
    python hf_to_mini_trt_llm.py \
        --model_name_or_path gpt2 \
        --output_dir ./gpt2_mini_trt_llm

Dependencies:
    See requirements.txt in the same directory.
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace checkpoint to mini_trt_llm format"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for config.json and model.safetensors",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Model: {args.model_name_or_path}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print("[WARN] Conversion tool is a placeholder in Phase 0.")
    print("       Full implementation will be added in future iterations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
