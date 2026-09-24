#!/usr/bin/env python3
"""采样器参考语义生成脚本（Top-K / Top-P）。

用途：把 HuggingFace 风格的 Top-K / Top-P 截断语义与理论概率打印出来，
供 C++ 侧 `test_sampler.cpp` 的分布检验对照。

注意：这里只比对**候选集合与概率**，不比对具体 token。
C++ 采样器用内联 Philox + host seed，与 torch.multinomial 的随机流不同，
逐 token 一致在实现上不可达，D3 也只要求分布级一致。

依赖：torch。
"""

import torch


def top_k_candidates(logits: torch.Tensor, k: int) -> torch.Tensor:
    """返回每个 batch 的 top-k 下标（按 logits 降序）。"""
    return torch.topk(logits, k, dim=-1).indices


def top_p_cutoff(logits: torch.Tensor, p: float) -> torch.Tensor:
    """返回每个 batch 的 nucleus 长度：满足累计概率 >= p 的最短前缀长度（至少 1）。"""
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    # cumsum >= p 的第一个位置 + 1 即前缀长度
    cutoff = (cumulative >= p).float().argmax(dim=-1) + 1
    return cutoff


def main() -> int:
    torch.manual_seed(42)

    vocab_size = 4
    logits = torch.tensor([[0.5, -0.5, 0.0, 1.5]])

    probabilities = torch.softmax(logits, dim=-1)
    print("softmax probabilities:", probabilities.tolist())

    for k in (1, 2, 4):
        candidates = top_k_candidates(logits, k)
        print(f"top-{k} candidates:", candidates.tolist())

    for p in (1e-6, 0.8, 1.0):
        cutoff = top_p_cutoff(logits, p)
        print(f"top-p={p} nucleus length:", cutoff.tolist())

    # k = vocab_size 时不应截断任何 token，可用于校验 C++ 侧的分布检验前提
    print(
        "expectation: k == vocab_size degrades to full softmax sampling;"
        " p == 1.0 keeps the whole vocabulary"
    )
    return 0


if __name__ == "__main__":
    main()
