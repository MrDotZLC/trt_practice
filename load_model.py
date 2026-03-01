import torch
import torchvision.models as models
# 下载预训练模型并导出为ONNX格式
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'resnet18.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17)
print('done')

# 加载ONNX模型并检查
import onnx
m = onnx.load('resnet18.onnx')
onnx.checker.check_model(m)
print('ONNX check OK')
print('input shape:', m.graph.input[0].type.tensor_type.shape)
