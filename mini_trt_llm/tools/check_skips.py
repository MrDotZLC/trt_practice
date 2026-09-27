#!/usr/bin/env python3
"""比对"期望跳过集合"：把**新增的跳过**变成红。

为什么需要它（Phase 5 阶段 0 的 P5-0-2）：ctest 把 Skipped 记作 **Passed**，所以"一批用例
因为缺资产 / 别的原因不再跑了"这件事对 CI 完全不可见——2026-09-27 实测：把两个历史示例工程
改名后，全量仍报 265 条 / 100% passed / 0 failed，只有跳过集合从 112 变 113
（见 docs/phase5_development_plan.md §3）。资产闸门（`MINI_TRT_REQUIRE_ASSETS`）挡住了
"缺资产"，本脚本挡住**其余**原因：只要出现没被登记过的跳过就判红。

输入是 ctest 的 `LastTest.log`（`ctest --test-dir build` 跑完就有，路径
`<build>/Testing/Temporary/LastTest.log`），因为它同时含各用例的 stdout——gtest 级的
`[  SKIPPED ]` 只有在那里才看得到。

约定：
- **`asset_gate_*` 这两条探针项里的跳过一律忽略**：它们的设计就是"从空目录跑同一条用例，
  看它是否按闸门规则跳过/失败"，跳过是它们的预期结果。
- **因"没有 CUDA 设备"而跳过的，单独计数并默认忽略**（打印出来）：那不是覆盖损失，而
  本脚本的基准是**真机全量**（`MINI_TRT_REQUIRE_GPU=1`，此时这类跳过会变成失败而不是跳过）。

用法：
    python3 mini_trt_llm/tools/check_skips.py --log build/Testing/Temporary/LastTest.log
    python3 mini_trt_llm/tools/check_skips.py --log <log> --update   # 重钉基线（要写理由）
    python3 mini_trt_llm/tools/check_skips.py --self-test

退出码：0 = 没有未登记的跳过；1 = 有（或日志不可信）；2 = 用法错误。
"""

import argparse
import os
import re
import sys
import tempfile

# ctest 在每个用例前打的行：`241/267 Testing: SomeSuite.SomeCase`
SECTION_RE = re.compile(r"^\s*\d+/\d+\s+Testing:\s+(\S.*?)\s*$")
# gtest 的跳过汇总行：`[  SKIPPED ] SomeSuite.SomeCase`（"N tests, listed below:" 不算）
SKIP_RE = re.compile(r"^\[  SKIPPED \] (\S+)\s*$")
# 无 GPU 的跳过标志。`test_gpu_guard.hpp` 的 `NoCudaMessage()` 会把 `CudaProbeToString()` 拼进来，
# 那里面**一定**有 `cudaGetDeviceCount -> err=`。
# **不能只认 "No CUDA device available"**：传了自定义说明的调用（例如
# `MINI_TRT_SKIP_IF_NO_CUDA("解析 ONNX 需要 CUDA（createInferBuilder）")`）不会带那句话
# —— 2026-09-28 实测因此漏判 2 条，被这个工具自己抓出来。
NO_CUDA_MARKERS = ("cudaGetDeviceCount -> err=", "No CUDA device available")
# 看起来不像"一次全量跑"的下限（防止把空跑 / 局部跑当成基线；见 TROUBLESHOOTING + TS-043）
MIN_SECTIONS = 100


def parse_log(text: str):
    """→ (skips: dict[name → item], gpu_skips: list[(name, item)], sections: int)"""
    skips = {}
    gpu_skips = []
    sections = 0
    item = None
    body = []

    def flush(name, lines):
        if name is None:
            return
        if name.startswith("asset_gate_"):
            return  # 探针故意造缺资产，跳过是预期结果
        for line in lines:
            matched = SKIP_RE.match(line.strip())
            if not matched:
                continue
            case = matched.group(1)
            if case in skips:
                continue
            if any(marker in entry for marker in NO_CUDA_MARKERS for entry in lines):
                if (case, name) not in gpu_skips:
                    gpu_skips.append((case, name))
            else:
                skips[case] = name

    for line in text.splitlines():
        matched = SECTION_RE.match(line)
        if matched:
            flush(item, body)
            item = matched.group(1)
            body = []
            sections += 1
            continue
        if item is not None:
            body.append(line)
    flush(item, body)
    return skips, gpu_skips, sections


def load_expected(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    expected = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                expected.add(stripped)
    return expected


def write_expected(path: str, names) -> None:
    header = (
        "# 期望跳过集合：真机全量（GPU + 资产闸门都开）时**允许**出现的 gtest 跳过。\n"
        "# 一行一个 `Suite.Case`；未列出的任何跳过都会让 tools/check_skips.py 判红。\n"
        "# 当前为空 = \"真机全量不允许出现任何覆盖跳过\"（2026-09-28 的基线）。\n"
        "# 改这个文件必须写明理由（为什么这个跳过可接受），并同步 PROGRESS.md 当前基线。\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for name in sorted(names):
            handle.write(name + "\n")


def run_check(log_path: str, expected_path: str, update: bool) -> int:
    if not os.path.isfile(log_path):
        print(f"[FAIL] 找不到日志 {log_path}（先跑 ctest --test-dir build）", file=sys.stderr)
        return 1
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    skips, gpu_skips, sections = parse_log(text)
    print(f"[info] 解析到 {sections} 个用例段；跳过 {len(skips)} 条"
          f"（另有 {len(gpu_skips)} 条因无 GPU 被忽略）")
    for case, item in sorted(gpu_skips)[:5]:
        print(f"[ignore] 无 GPU：{case}（来自 {item}）")
    if len(gpu_skips) > 5:
        print(f"[ignore] …… 另有 {len(gpu_skips) - 5} 条同类，略")

    if sections < MIN_SECTIONS:
        print(f"[FAIL] 只解析到 {sections} 个用例段，看起来不是一次全量跑——"
              f"注意 ctest 的静默空跑（退出码 0 但一条没跑，见 TROUBLESHOOTING + TS-043）",
              file=sys.stderr)
        return 1

    if update:
        write_expected(expected_path, skips.keys())
        print(f"[ok] 已重钉基线：{expected_path}（{len(skips)} 条）")
        return 0

    expected = load_expected(expected_path)
    unexpected = sorted(set(skips) - expected)
    vanished = sorted(expected - set(skips))
    for case in unexpected:
        print(f"[FAIL] 未登记的跳过：{case}（来自 {skips[case]}）", file=sys.stderr)
    for case in vanished:
        print(f"[note] 期望会跳过但这次没跳过：{case}（覆盖反而更多，不判红）")
    if unexpected:
        print("[FAIL] 出现了未登记的跳过——要么修掉它的原因，要么用 --update 重钉并写明理由，"
              "见 PROGRESS.md 的「已知会失败/跳过的测试」", file=sys.stderr)
        return 1
    print(f"[ok] 没有未登记的跳过（期望集合 {len(expected)} 条）")
    return 0


def self_test() -> int:
    """用合成日志自证：护栏必须能证明它会拦人（PROGRESS.md + DEC-TEST-CONVENTIONS）。"""
    failures = []

    def make_log(skip_case="", section="NormalSuite.Case", gpu=False, sections=120):
        lines = []
        for index in range(sections):
            name = f"{section}.{index}" if index else section
            lines.append(f"{index + 1}/{sections} Testing: {name}")
            lines.append(f"\"{name}\" start time: Jan 01 00:00:00 CST")
            if skip_case:
                if gpu:
                    # 用**自定义说明**那种形态（它不带 "No CUDA device available"），
                    # 那是这条判据最容易漏的一类。
                    lines.append("解析 ONNX 需要 CUDA —— cudaGetDeviceCount -> err=35 "
                                 "(cudaErrorInsufficientDriver), count=-1")
                lines.append(f"[  SKIPPED ] {skip_case}")
                lines.append("[  SKIPPED ] 1 test, listed below:")
                lines.append(f"[  SKIPPED ] {skip_case}")
            lines.append("Test Passed.")
        return "\n".join(lines) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        expected = os.path.join(tmp, "expected.txt")
        write_expected(expected, [])

        # ① 干净日志 → 通过
        clean = os.path.join(tmp, "clean.log")
        with open(clean, "w", encoding="utf-8") as handle:
            handle.write(make_log())
        if run_check(clean, expected, update=False) != 0:
            failures.append("干净日志应当通过")

        # ② 普通段里出现跳过 → 必须红（这就是护栏要拦的东西）
        dirty = os.path.join(tmp, "dirty.log")
        with open(dirty, "w", encoding="utf-8") as handle:
            handle.write(make_log(skip_case="NormalSuite.Case", section="NormalSuite.Case"))
        if run_check(dirty, expected, update=False) != 1:
            failures.append("普通段里的跳过必须判红")

        # ③ asset_gate_* 探针段里的跳过 → 忽略（否则探针自己会把检查搞红，见 TS-049）
        probe = os.path.join(tmp, "probe.log")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write(make_log(skip_case="ProbeSuite.Case", section="asset_gate_skips_without_require"))
        if run_check(probe, expected, update=False) != 0:
            failures.append("asset_gate_* 段里的跳过应当被忽略")

        # ④ 无 GPU 造成的跳过 → 忽略但打印
        gpu = os.path.join(tmp, "gpu.log")
        with open(gpu, "w", encoding="utf-8") as handle:
            handle.write(make_log(skip_case="GpuSuite.Case", section="GpuSuite.Case", gpu=True))
        if run_check(gpu, expected, update=False) != 0:
            failures.append("无 GPU 的跳过应当被忽略")

        # ⑤ 局部的 / 空跑的日志 → 判不可信
        tiny = os.path.join(tmp, "tiny.log")
        with open(tiny, "w", encoding="utf-8") as handle:
            handle.write(make_log(sections=3))
        if run_check(tiny, expected, update=False) != 1:
            failures.append("不是全量跑的日志必须判不可信")

        # ⑥ 钉了基线之后同一条跳过就不再算意外
        pinned = os.path.join(tmp, "pinned.txt")
        write_expected(pinned, ["NormalSuite.Case"])
        if run_check(dirty, pinned, update=False) != 0:
            failures.append("已登记的跳过不应当判红")

    if failures:
        for item in failures:
            print(f"[self-test][FAIL] {item}", file=sys.stderr)
        return 1
    print("[self-test][ok] 6 项全过（干净通过 / 普通跳过红 / 探针段忽略 / 无 GPU 忽略 / "
          "非全量日志不可信 / 已登记不红）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="比对期望跳过集合（真机全量用）")
    parser.add_argument("--log", help="ctest 的 LastTest.log 路径")
    parser.add_argument("--expected",
                        default="mini_trt_llm/tests/data/expected_skips.txt",
                        help="期望跳过集合（默认 tests/data/expected_skips.txt）")
    parser.add_argument("--update", action="store_true",
                        help="把本次观测到的跳过写回 --expected（重钉基线，需写明理由）")
    parser.add_argument("--self-test", action="store_true", help="只跑自检")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.log:
        parser.error("要么给 --log，要么用 --self-test")
    return run_check(args.log, args.expected, args.update)


if __name__ == "__main__":
    sys.exit(main())
