#!/usr/bin/env python3
"""生成 / 校验 GPT-2 BPE 的参考数据（golden）。

为什么要有它：`BpeTokenizer` 的正确性判据是"**与 HF 逐 token 全等**"（
`docs/future_iterations_test_plan.md` §2.1）。参考实现是裁决对错的标尺，标尺必须可复现、
必须自带出处——所以这份脚本把"哪个 tokenizer、哪几个文件、哪一版 transformers"一起固化下来。

**本脚本不联网**：它用 `local_files_only=True` 从本地快照加载；找不到就报错退出，
绝不静默去下载（`AGENTS.md` §0.2：联网操作必须单独获批）。

用法：
    # 生成（写入仓库内的参考文件，按 F3 决议入库）
    python3 make_tokenizer_golden.py --output ../tests/data/gpt2_tokenizer_golden.json
    # 校验（重新算一遍，逐样本比对已提交的参考；缺 tokenizer 文件时返回 77 → ctest 记为 Skipped）
    python3 make_tokenizer_golden.py --check ../tests/data/gpt2_tokenizer_golden.json --skip-if-missing
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SKIP_RETURN_CODE = 77

# 默认快照：本机 HF 缓存里的 gpt2（2026-09-26 实测存在；换机器要么重建缓存，
# 要么用 --tokenizer-dir 指到你自己留存的 HF gpt2 目录）。
DEFAULT_SNAPSHOT = (
    "/home/mr_zlc/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

# 样本集是**固定**的：覆盖"最容易错"的几类，而不是随机抽文本。
# 空串 / 前导空格 / 连续空格 / 制表符 / 换行 → GPT-2 的 `Ġ` 与 `\s+(?!\S)` 语义；
# 中文 / emoji / 中英混排 → byte-level 回退路径；纯标点 / 数字 → 预切分的分支边界；
# 长文本 → 多次 merge 的深路径。
_PARAGRAPH = (
    "In the beginning the Universe was created. This has made a lot of people very angry "
    "and been widely regarded as a bad move. Many were increasingly of the opinion that "
    "they'd all made a big mistake in coming down from the trees in the first place."
)
# 重复 6 次：目标是 token 数 **≥256**，让 BPE 走多轮 merge（单段只有 ~50 token，深路径覆盖不足）。
_LONG_TEXT = _PARAGRAPH * 6

SAMPLES = [
    # kind 决定 C++ 侧哪条用例跑它：edge（边界）/ basic / whitespace / utf8 / long。
    ("edge", ""),
    ("basic", "The quick brown fox"),
    ("basic", "Hello, world!"),
    ("whitespace", " hello"),
    ("whitespace", "a  b"),
    ("whitespace", "a\tb"),
    ("whitespace", "line1\nline2"),
    ("whitespace", "   "),
    ("basic", "123 456 7.5"),
    ("utf8", "café münchen"),
    ("utf8", "中文 测试"),
    ("utf8", "emoji 🙂"),
    ("utf8", "中英mix: hello 世界! 🙂 123"),
    # 下面几条刻意压在"Unicode 分类是近似"这个已知限制上：日文 / 西里尔是 \p{L}，
    # 全角字母数字是 \p{L}/\p{N}，CJK 标点是"其它"，ZWJ 序列与货币符号是"其它"。
    # 它们不通过就说明分类区间表要补——**判据仍然是 golden 对拍**。
    ("utf8", "日本語のテスト"),
    ("utf8", "Привет мир"),
    ("utf8", "ＡＢＣ１２３"),
    ("utf8", "a、b。c"),
    ("utf8", "👨👩👧 family"),
    ("utf8", "€100 £50"),
    ("long", _LONG_TEXT),
    ("edge", "!"),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(tokenizer_dir):
    """离线加载 HF 的 GPT-2 tokenizer。任何"要不要去下载"的念头都在这里被拦住。"""
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError("tokenizer 目录不存在：{}".format(tokenizer_dir))
    for name in ("vocab.json", "merges.txt"):
        if not os.path.isfile(os.path.join(tokenizer_dir, name)):
            raise FileNotFoundError("缺少 {}：{}".format(name, tokenizer_dir))
    try:
        import transformers
        from transformers import GPT2TokenizerFast
    except ImportError as exc:  # pragma: no cover - 依赖缺失给出可操作提示
        raise RuntimeError("需要 transformers（requirements.txt），当前缺失：{}".format(exc))
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    return tokenizer, transformers.__version__


def build_payload(tokenizer, transformers_version, tokenizer_dir):
    samples = []
    pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    for kind, text in SAMPLES:
        ids = tokenizer(text)["input_ids"]
        samples.append(
            {
                "kind": kind,
                "text": text,
                # pieces 是 **byte-encoded** 的预切分结果（与 HF 的 pre_tokenize_str 同口径）：
                # 存它是为了排错——不一致时能立刻分清是"预切分错"还是"BPE merge 错"。
                "pieces": [piece for piece, _ in pre_tokenizer.pre_tokenize_str(text)],
                "ids": ids,
                "decoded": tokenizer.decode(ids),
                "tokens": tokenizer.convert_ids_to_tokens(ids),
            }
        )
    return {
        "reference": {
            "implementation": "transformers.GPT2TokenizerFast",
            "transformers_version": transformers_version,
            "tokenizer_dir": tokenizer_dir,
            "vocab_json_sha256": sha256_file(os.path.join(tokenizer_dir, "vocab.json")),
            "merges_txt_sha256": sha256_file(os.path.join(tokenizer_dir, "merges.txt")),
            "local_files_only": True,
            "note": "本文件由 tools/make_tokenizer_golden.py 生成；改动它等于改标尺，必须同时说明原因",
        },
        "vocab_size": tokenizer.vocab_size,
        "samples": samples,
    }


def write_payload(payload, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False：让中文 / emoji 以原始 UTF-8 落盘，C++ 侧现有的极简 JSON 解析器
    # 才能直接读（它按字节拷贝字符串，不处理 \u 之外的花样）。
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if "\\u" in text:
        raise RuntimeError("输出里出现了 \\u 转义：ensure_ascii 没生效？")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(text)


def compare(expected, actual):
    """只比"标尺内容"（文本 / ids / 解码结果），不比随环境变化的元信息。"""
    if len(expected["samples"]) != len(actual["samples"]):
        return "样本数不同：{} vs {}".format(len(expected["samples"]), len(actual["samples"]))
    if expected.get("vocab_size") != actual.get("vocab_size"):
        return "vocab_size 不同：{} vs {}".format(expected.get("vocab_size"), actual.get("vocab_size"))
    for i, (want, got) in enumerate(zip(expected["samples"], actual["samples"])):
        for field in ("kind", "text", "pieces", "ids", "decoded"):
            if want.get(field) != got.get(field):
                return "样本[{}] 的 {} 不一致：{!r} vs {!r}".format(i, field, want.get(field), got.get(field))
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="GPT-2 BPE 参考数据生成 / 校验（不联网）")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_SNAPSHOT, help="含 vocab.json + merges.txt 的目录")
    parser.add_argument("--output", help="生成模式：写入路径")
    parser.add_argument("--check", help="校验模式：与已有参考逐样本比对")
    parser.add_argument("--skip-if-missing", action="store_true", help="缺 tokenizer 文件 / 依赖时返回 77（ctest 记为 Skipped）")
    args = parser.parse_args(argv)

    if not args.output and not args.check:
        parser.error("需要 --output 或 --check 之一")

    try:
        tokenizer, transformers_version = load_reference(args.tokenizer_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        if args.skip_if_missing:
            print("[make_tokenizer_golden] 跳过：{}".format(exc))
            return SKIP_RETURN_CODE
        print("[make_tokenizer_golden] 失败：{}".format(exc), file=sys.stderr)
        return 2

    payload = build_payload(tokenizer, transformers_version, args.tokenizer_dir)

    if args.output:
        write_payload(payload, args.output)
        print("[make_tokenizer_golden] 已写出 {}（{} 个样本，vocab_size={}）".format(
            args.output, len(payload["samples"]), payload["vocab_size"]))
        return 0

    with open(args.check, "r", encoding="utf-8") as handle:
        expected = json.load(handle)
    problem = compare(expected, payload)
    if problem:
        print("[make_tokenizer_golden] 校验失败：{}\n"
              "  这说明提交的参考与本地 tokenizer 文件不一致——先判断是参考被改错、"
              "还是本机 tokenizer 文件换了版本，再决定改哪一边。".format(problem), file=sys.stderr)
        return 1
    print("[make_tokenizer_golden] 校验通过：{} 个样本与 {} 一致".format(
        len(payload["samples"]), args.tokenizer_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
