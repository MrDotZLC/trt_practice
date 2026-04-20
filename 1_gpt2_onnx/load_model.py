# ~/trt_practice/1_gpt2_trt/load_model.py

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Wrapper(nn.Module):
    """
    包装 GPT2Model，固定 use_cache=False，
    只返回 last_hidden_state（Tensor），
    避免 DynamicCache 导致 ONNX export 失败。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids, use_cache=False)
        return out.logits  # [batch, seq_len, 50257]

# ── 1. 下载并加载模型 ──────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2").eval()
wrapper   = GPT2Wrapper(model).eval()

print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
print(f"FP32 size : {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.1f} MB")
print(f"FP16 size : {sum(p.numel() for p in model.parameters()) * 2 / 1024**2:.1f} MB")

# ── 2. 导出 ONNX ───────────────────────────────────────────────────────────
#
# GPT-2 输入：
#   input_ids : [batch, seq_len]  token id 序列
#
# 动态轴：
#   batch   → 支持不同 batch size
#   seq_len → 支持不同序列长度（推理时变长）
#
# 输出：
#   last_hidden_state : [batch, seq_len, vocab_size=50257]
#
# use_cache=False：
#   禁用 KV Cache，简化 ONNX 图（去掉 past_key_values 输入输出）
#   生产环境用 KV Cache 加速自回归，此处先学基础流程

print(__file__)
onnx_path = os.path.join(os.path.dirname(__file__), "gpt2.onnx")

dummy_input = torch.randint(0, 50257, (1, 16))  # batch=1, seq_len=16

print("\nExporting ONNX...")
torch.onnx.export(
    wrapper,
    (dummy_input,),
    onnx_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids" : {0: "batch", 1: "seq_len"},
        "logits"    : {0: "batch", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"ONNX saved: {onnx_path}")

# ── 3. 验证 ONNX ───────────────────────────────────────────────────────────
import onnx
model_onnx = onnx.load(onnx_path)
onnx.checker.check_model(model_onnx)
print("ONNX check passed")
print(f"ONNX size: {os.path.getsize(onnx_path) / 1024**2:.1f} MB")

# ── 4. 保存一组参考输出（用于后续 TRT 精度对比）─────────────────────────────
print("\nGenerating reference output...")
text   = "The quick brown fox"
tokens = tokenizer(text, return_tensors="pt")
wrapper_ref = GPT2Wrapper(model).eval()
with torch.no_grad():
    ref_out = wrapper_ref(tokens["input_ids"])  # [1, seq_len, 50257]

ref_path = os.path.join(os.path.dirname(__file__), "ref_output.bin")
ref_out.numpy().tofile(ref_path)
print(f"Reference output saved: {ref_path}")
print(f"  shape: {ref_out.shape}")
print(f"  seq_len={ref_out.shape[1]}, hidden_size={ref_out.shape[2]}")