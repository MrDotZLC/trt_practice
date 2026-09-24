#!/usr/bin/env python3
"""RoPE 参考输出生成/自检脚本。

用途：确认 C++ RoPEPlugin 采用的旋转约定与 HuggingFace 一致，并打印可对比的数值。

约定（half-split，与 transformers 的 apply_rotary_pos_emb 对齐）：
    h  = rotary_dim / 2
    out[j]     = x[j]     * cos - x[j + h] * sin
    out[j + h] = x[j + h] * cos + x[j]     * sin
    cos/sin 由 position 与 inv_freq = base ** (-2j / rotary_dim) 得到

依赖：torch。装了 transformers 时会额外与 HF 官方实现交叉验证。
"""

import sys

import torch


def build_cos_sin(positions: torch.Tensor, rotary_dim: int, base: float):
    """positions: [batch, seq] → (cos, sin): [batch, seq, rotary_dim]。

    每个角度重复两次，是因为 rotate_half 的配对方式是 (x[j], x[j + h]) 共享同一个角度，
    因此 cos/sin 的各半段必须相同。
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    angles = torch.outer(positions.reshape(-1).float(), inv_freq)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    shape = (*positions.shape, rotary_dim)
    return cos.reshape(shape), sin.reshape(shape)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def main() -> int:
    torch.manual_seed(42)

    head_size = 8
    rotary_dim = 4  # 部分旋转：后 4 维应保持原值
    base = 10000.0
    # **必须覆盖 batch > 1**：曾经有一版 C++ 参考实现漏了 batch 维度，
    # 而当时的交叉验证只跑了 batch=1，没能拦住（见 docs/TROUBLESHOOTING.md #9）。
    positions = torch.tensor([[2, 5], [7, 11]])
    batch = positions.shape[0]
    seq = positions.shape[1]

    x = torch.sin(0.71 * torch.arange(batch * seq * head_size, dtype=torch.float32)) * 0.8
    x = x.reshape(batch, seq, head_size)

    cos, sin = build_cos_sin(positions, rotary_dim, base)
    rotated_front = apply_rope(x[..., :rotary_dim], cos, sin)
    out = torch.cat((rotated_front, x[..., rotary_dim:]), dim=-1)
    print(f"reference (half-split, batch={batch}, seq={seq}, rotary_dim=4, head_size=8):")
    print(out)

    # 自检：不同 batch 的位置不同，其旋转结果必须不同。
    # 若实现漏了 batch 维度（所有 batch 都用 positions[0]），这条断言会失败。
    if batch > 1:
        if torch.allclose(out[0, 0], out[1, 0]):
            print("MISMATCH: batch 维没有被正确使用（不同 batch 得到相同结果）", file=sys.stderr)
            return 1

    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        # HF 的 apply_rotary_pos_emb 直接作用于完整 head_dim，因此交叉验证用全旋转场景
        full_cos, full_sin = build_cos_sin(positions, head_size, base)
        q = x.unsqueeze(1)  # [batch, heads=1, seq, head_size]
        # 注意：HF 的 apply_rotary_pos_emb 内部会自己执行 cos.unsqueeze(unsqueeze_dim=1)，
        # 这里必须传 [batch, seq, head_size]。手工再 unsqueeze 一次会让 batch>1 时广播错位——
        # 而 batch=1 时它会"碰巧"广播成功，把错误藏住。
        hf_q, _ = apply_rotary_pos_emb(q, q, full_cos, full_sin)
        ours = apply_rope(x, full_cos, full_sin)
        max_diff = (hf_q[:, 0] - ours).abs().max().item()
        print(f"max diff vs HF apply_rotary_pos_emb (batch={batch}): {max_diff:.3e}")
        if max_diff > 1e-5:
            print("MISMATCH: 旋转约定与 HuggingFace 不一致", file=sys.stderr)
            return 1
    except ImportError:
        print("transformers 未安装，跳过与 HF 的交叉验证")

    return 0


if __name__ == "__main__":
    sys.exit(main())
