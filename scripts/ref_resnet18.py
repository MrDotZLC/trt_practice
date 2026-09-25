#!/usr/bin/env python3
"""生成 ResNet18 的外部参考基线（Phase 4 的 P4-1）。

为什么需要它：历史工程 `0_resnet18_onnx/` 的"精度验证"是**自相对**的——FP16/INT8 跟它自己跑的
FP32 比，没有任何外部真值（见 docs/phase4_development_plan.md §1.1/§1.3）。没有独立基线，
"两条路径对齐"就只能证明它们互相一致，证明不了它们对。本脚本用 torchvision 的
ImageNet 预训练权重（与 load_model.py 同源）产出 FP32 logits 作为那条标尺。

两套输入（docs/phase4_test_plan.md §4）：
  ramp   —— 与历史工程 src/main.cpp 同式的合成斜坡，用于与旧实现/旧输入定义对齐；
            它**不做归一化**（历史工程直接喂原始值）。
  pixels —— 从 calib_data 的真实图**反归一化**回 [0,255] 的像素质（float32 NCHW），
            再由本脚本按 ImageNet mean/std 归一化后送模型。它定义了 `CVRunner` 的输入契约：
            **输入 = [0,255] 的 float32 NCHW**，归一化由 CVRunner 自己做（D4）。
            反归一化那一步有四舍五入（得到的是"像素质"，不是原图的逐位副本），
            因此**基线以落盘的 pixels 张量为唯一输入定义**，与原始 JPEG 无关——自洽即可。

确定性：CPU、单线程、固定权重。同一脚本跑两次必须逐位一致（PROGRESS.md §2.13 对参考实现的纪律）。

用法：
    python3 scripts/ref_resnet18.py --input ramp   --output models/resnet18/ref_ramp_b8.bin
    python3 scripts/ref_resnet18.py --input pixels --output models/resnet18/ref_pixels_b8.bin
"""

import argparse
import hashlib
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.models as models

# ImageNet 归一化参数。与 0_resnet18_onnx/prepare_calib_data.py 完全一致——
# 两处若不一致，数值差会被误判成"引擎错"（docs/phase4_test_plan.md §3）。
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

K_CHANNELS = 3
K_SIZE = 224
K_CLASSES = 1000


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_ramp_input(batch: int) -> torch.Tensor:
    """历史工程 src/main.cpp 的合成输入：input[i] = (i % 255) / 255。

    刻意用 float32 除法（而不是先算 double 再转换），与 C++ 的 `float(k) / 255.f` 同语义；
    否则两条路径的输入本身就会差 1 ulp，对拍时无法区分"输入不同"与"实现不同"。
    """
    count = batch * K_CHANNELS * K_SIZE * K_SIZE
    flat = (torch.arange(count, dtype=torch.int64) % 255).to(torch.float32) / 255.0
    return flat.reshape(batch, K_CHANNELS, K_SIZE, K_SIZE)


def load_calib_images(calib_dir: str, batch: int) -> torch.Tensor:
    """读取 calib_data 里前 batch 张（已是 CHW FP32 归一化值）并还原成 [0,255] 像素质。

    为什么反归一化：calib_data 是"归一化之后"的张量，而 CVRunner 的契约是吃原始像素质
    自己做归一化（D4）。要测 CVRunner，就必须有一份归一化**之前**的输入。
    """
    files = sorted(f for f in os.listdir(calib_dir) if f.endswith(".bin"))
    if len(files) < batch:
        raise SystemExit(f"calib_data 只有 {len(files)} 张，不足 batch={batch}")

    pixel_batch = np.empty((batch, K_CHANNELS, K_SIZE, K_SIZE), dtype=np.float32)
    for i in range(batch):
        raw = np.fromfile(os.path.join(calib_dir, files[i]), dtype=np.float32)
        expected = K_CHANNELS * K_SIZE * K_SIZE
        if raw.size != expected:
            raise SystemExit(f"{files[i]} 元素数 {raw.size} != {expected}")
        normalized = raw.reshape(K_CHANNELS, K_SIZE, K_SIZE)
        # 反归一化 → 裁到 [0,255] → 取整；得到的是像素质（uint8 可表示），再存成 float32。
        pixels = (normalized * STD[:, None, None] + MEAN[:, None, None]) * 255.0
        pixel_batch[i] = np.rint(np.clip(pixels, 0.0, 255.0)).astype(np.float32)
    return torch.from_numpy(pixel_batch)


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """[0,255] 像素质 → ImageNet 归一化。这是 CVRunner 的预处理契约的 Python 参考实现。"""
    mean = torch.tensor(MEAN, dtype=torch.float32).view(1, K_CHANNELS, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(1, K_CHANNELS, 1, 1)
    return (pixels / 255.0 - mean) / std


def build_inputs(kind: str, batch: int, calib_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (送入模型的张量, 落盘的"契约输入"张量)。

    - ramp：契约输入就是它本身（未归一化，与历史工程一致）；
    - pixels：契约输入是 [0,255] 像素质，送模型的是它归一化后的结果。
    """
    if kind == "ramp":
        ramp = make_ramp_input(batch)
        return ramp, ramp
    pixels = load_calib_images(calib_dir, batch)
    return normalize_pixels(pixels), pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet18 torchvision FP32 reference baseline")
    parser.add_argument("--input", choices=["ramp", "pixels"], required=True)
    parser.add_argument("--output", required=True, help="logits 输出路径（FP32 raw）")
    parser.add_argument("--batch", type=int, default=8, help="与历史工程一致：8")
    parser.add_argument("--calib-dir", default="0_resnet18_onnx/calib_data")
    parser.add_argument("--input-dir", default=None,
                        help="契约输入张量的落盘目录，默认与 --output 同级的 inputs/")
    args = parser.parse_args()

    # 确定性：CPU 单线程 + 固定权重，保证"同脚本两次运行逐位一致"。
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    weights = models.ResNet18_Weights.DEFAULT
    # 记录权重文件的绝对路径与 SHA256：没有它，"这份基线是哪份权重算的"就无从回答，
    # 而换权重会让所有对拍结论失效（docs/phase4_test_plan.md §3）。
    weights_file = os.path.join(torch.hub.get_dir(), "checkpoints",
                                os.path.basename(weights.url))
    if not os.path.isfile(weights_file):
        raise SystemExit(f"权重文件不存在: {weights_file}（需要联网下载或用缓存）")
    model = models.resnet18(weights=weights).eval().to(torch.float32)

    model_input, contract_input = build_inputs(args.input, args.batch, args.calib_dir)
    with torch.no_grad():
        logits = model(model_input).to(torch.float32).contiguous()

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    input_dir = args.input_dir or os.path.join(out_dir, "inputs")
    os.makedirs(input_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.output))[0]
    contract_path = os.path.join(input_dir, f"{stem}.contract_input.f32.bin")
    contract_input.numpy().astype(np.float32).tofile(contract_path)

    # 归一化后的张量单独落盘：它是"Python 侧前处理"的输出，供 C++ 的 R0.3 用例逐元素对拍。
    normalized_path = None
    if args.input == "pixels":
        normalized_path = os.path.join(input_dir, f"{stem}.normalized.f32.bin")
        model_input.numpy().astype(np.float32).tofile(normalized_path)

    logits.numpy().astype(np.float32).tofile(args.output)

    argmax = logits.argmax(dim=1).tolist()
    meta: Dict[str, object] = {
        "script": os.path.relpath(__file__),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "device": "cpu",
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
        "weights": {
            "name": "ResNet18_Weights.DEFAULT",
            "url": weights.url,
            "cached_file": weights_file,
            "cached_file_sha256": sha256_of_file(weights_file),
        },
        "input": {
            "kind": args.input,
            "batch": args.batch,
            "shape": list(contract_input.shape),
            "dtype": "float32",
            "contract_path": os.path.relpath(contract_path),
            "contract_sha256": sha256_of_file(contract_path),
            "normalized_path": os.path.relpath(normalized_path) if normalized_path else None,
            "normalized_sha256": sha256_of_file(normalized_path) if normalized_path else None,
            "description": ("合成斜坡 (i%255)/255，未归一化（与历史工程 src/main.cpp 同式）"
                            if args.input == "ramp" else
                            "[0,255] 像素质（由 calib_data 反归一化四舍五入得到），"
                            "归一化由消费方按 ImageNet mean/std 完成"),
        },
        "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
        "logits": {
            "path": os.path.relpath(args.output),
            "shape": list(logits.shape),
            "sha256": sha256_of_file(args.output),
            "argmax_per_sample": argmax,
            "min": float(logits.min()),
            "max": float(logits.max()),
            "mean": float(logits.mean()),
        },
    }
    meta_path = os.path.join(out_dir, f"{stem}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[ref_resnet18] input={args.input} batch={args.batch} output={args.output}")
    print(f"  logits sha256 = {meta['logits']['sha256']}")  # type: ignore[index]
    print(f"  argmax        = {argmax}")
    print(f"  logits range  = [{meta['logits']['min']:.4f}, {meta['logits']['max']:.4f}]")  # type: ignore[index]
    print(f"  meta          = {meta_path}")


if __name__ == "__main__":
    main()
