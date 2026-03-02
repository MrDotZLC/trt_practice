# ~/trt_practice/0_resnet18_onnx/prepare_calib_data.py

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# ImageNet 归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image: Image.Image) -> np.ndarray:
    """
    ImageNet 标准预处理：
    1. tiny-imagenet 原始尺寸 64x64，直接 resize 到 224x224
    2. 转 float32，归一化到 [0,1]
    3. 减均值除标准差
    4. HWC → CHW
    """
    if image is None:
        return None
    image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "calib_data")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ILSVRC2012 validation set (streaming)...")
    dataset = load_dataset(
        "zh-plus/tiny-imagenet",
        split="valid",
        streaming=True,
    )

    count = 0
    target = 500
    pbar = tqdm(total=target, desc="Saving calib data", unit="img")

    for sample in dataset:
        arr = preprocess(sample["image"])
        if arr is None:
            continue
        path = os.path.join(out_dir, f"calib_{count:04d}.bin")
        arr.tofile(path)
        count += 1
        pbar.update(1)
        if count >= target:
            break

    pbar.close()
    print(f"Done. {count} files saved to {out_dir}/")
    print(f"Each file: 3x224x224 float32 = {3*224*224*4} bytes")

if __name__ == "__main__":
    main()