#!/usr/bin/env python3
"""重建 `assets/legacy/resnet18_onnx/resnet18.onnx`（Phase 5 从 `0_resnet18_onnx/` 迁来）。

**为什么显式拼输出目录**：原脚本写的是裸文件名 `'resnet18.onnx'`，产物落在"当前工作目录"——
它当年住在 `0_resnet18_onnx/` 下所以不出错，迁到 `assets/legacy/scripts/` 之后会写到别处。
现在统一按 `__file__` 定位到 `assets/legacy/resnet18_onnx/`，与测试读取的位置一致。
"""

import os

import torch
import torchvision.models as models

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resnet18_onnx")
os.makedirs(out_dir, exist_ok=True)
onnx_path = os.path.join(out_dir, "resnet18.onnx")

# 下载预训练模型并导出为ONNX格式
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, onnx_path,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17)
print('done')

# 加载ONNX模型并检查
import onnx
m = onnx.load(onnx_path)
onnx.checker.check_model(m)
print('ONNX check OK')
print('input shape:', m.graph.input[0].type.tensor_type.shape)
