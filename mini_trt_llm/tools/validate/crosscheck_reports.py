#!/usr/bin/env python3
"""把 C++ 侧与 Python 侧的 INT8 统计报告放在一起比对（future_iterations A2-5）。

为什么需要它：`int8_eval.py` 是判据规格的第二种实现，而 C++ 的精度用例是第一种。
**两条独立实现对同一批 logits 必须给出同一组数字**（整体的 n/分子、余量子集的 n/分子、
逐桶的 n/分子）。数字不一致说明口径漂移——此时要查口径，**不许改任一侧的阈值**（AGENTS.md §7）。

输入：
    --cpp-report  C++ 用例 DumpsLogitsAndCppReportForCrossCheck 产出的 cpp_report.json
    --py-report   int8_eval.py --json-out 产出的报告

只比"口径数字"（n / 分子 / 阈值），不比名称与文案：两边的分层桶名不同（C++ 用 bucketN），
比的是**同一顺序上的桶**——顺序由同一组 kBucketEdges 决定。
"""

import argparse
import json
import os
import sys
import tempfile

SKIP_RETURN_CODE = 77


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_ratio(where, cpp, py, problems):
    """比对一组 {n, agree}。rate 不单独比：它由这两个数派生，比派生量会掩盖真差异。"""
    for key in ("n", "agree"):
        if cpp.get(key) != py.get(key):
            problems.append("{} 的 {} 不一致：C++ {} vs Python {}".format(where, key, cpp.get(key), py.get(key)))
    if cpp.get("n") in (None, 0) and py.get("n") in (None, 0):
        return
    if cpp.get("n"):
        cpp_rate = cpp["agree"] / cpp["n"]
        py_rate = py["agree"] / py["n"] if py.get("n") else None
        if py_rate is None or abs(cpp_rate - py_rate) > 1e-9:
            problems.append("{} 的一致率不一致：C++ {:.6f} vs Python {}".format(where, cpp_rate, py_rate))


def main(argv=None):
    parser = argparse.ArgumentParser(description="比对 C++ 侧与 Python 侧的 INT8 统计口径")
    parser.add_argument("--cpp-report")
    parser.add_argument("--py-report")
    parser.add_argument("--skip-if-missing", action="store_true", help="报告缺失时返回 77（ctest 记为 Skipped）")
    parser.add_argument("--self-test", action="store_true", help="用合成报告自证：一致时返回 0、不一致时返回 1")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.cpp_report or not args.py_report:
        parser.error("需要 --cpp-report 与 --py-report（或使用 --self-test）")

    for path in (args.cpp_report, args.py_report):
        try:
            with open(path, "r", encoding="utf-8"):
                pass
        except OSError as exc:
            if args.skip_if_missing:
                print("[int8_crosscheck] 跳过：{}（{}）".format(path, exc))
                return SKIP_RETURN_CODE
            print("[int8_crosscheck] 失败：读不到 {}".format(path), file=sys.stderr)
            return 2

    cpp = load(args.cpp_report)
    py = load(args.py_report)
    problems = []

    cpp_thresholds = cpp.get("thresholds", {})
    py_thresholds = py.get("thresholds", {})
    if cpp_thresholds.get("confident_margin") != py_thresholds.get("confident_margin"):
        problems.append(
            "置信余量阈值不一致：C++ {} vs Python {}".format(
                cpp_thresholds.get("confident_margin"), py_thresholds.get("confident_margin")
            )
        )
    if cpp_thresholds.get("bucket_edges") != py_thresholds.get("bucket_edges"):
        problems.append(
            "分桶边界不一致：C++ {} vs Python {}".format(
                cpp_thresholds.get("bucket_edges"), py_thresholds.get("bucket_edges")
            )
        )

    compare_ratio("整体", cpp.get("overall", {}), py.get("overall", {}), problems)
    compare_ratio("余量子集", cpp.get("confident", {}), py.get("confident", {}), problems)

    cpp_strata = cpp.get("strata", [])
    py_strata = py.get("strata", [])
    if len(cpp_strata) != len(py_strata):
        problems.append("分层桶数不一致：C++ {} vs Python {}".format(len(cpp_strata), len(py_strata)))
    for index, (cpp_bucket, py_bucket) in enumerate(zip(cpp_strata, py_strata)):
        compare_ratio(
            "第 {} 桶（C++ {} / Python {}）".format(index, cpp_bucket.get("bucket"), py_bucket.get("bucket")),
            cpp_bucket,
            py_bucket,
            problems,
        )

    if problems:
        print("[int8_crosscheck] 口径不一致（这属于**真问题**：先查两边口径，不许改阈值）：")
        for problem in problems:
            print("  - {}".format(problem))
        return 1

    print(
        "[int8_crosscheck] 通过：整体 {}/{}(C++) == {}/{}(Python)、余量子集 {}/{} == {}/{}、逐桶一致".format(
            cpp["overall"]["agree"], cpp["overall"]["n"],
            py["overall"]["agree"], py["overall"]["n"],
            cpp["confident"].get("agree"), cpp["confident"].get("n"),
            py["confident"].get("agree"), py["confident"].get("n"),
        )
    )
    compliance = py.get("spec_compliance", {})
    if compliance.get("legacy_mode"):
        print("  注意：Python 侧报告来自 --legacy-mode（与标定集同源），只用于口径对齐，不满足 §1.6 规格")
    return 0


def self_test():
    """护栏自证：比对器既能放行一致的口径，也**必须**拦住不一致的口径。"""
    print("[int8_crosscheck --self-test]")
    base = {
        "thresholds": {"confident_margin": 5.0, "bucket_edges": [1.0, 2.0, 5.0, 10.0]},
        "overall": {"n": 256, "agree": 98, "agree_rate": 98 / 256},
        "confident": {"n": 12, "agree": 12, "agree_rate": 1.0},
        "strata": [
            {"bucket": "a", "n": 148, "agree": 35},
            {"bucket": "b", "n": 63, "agree": 25},
            {"bucket": "c", "n": 19, "agree": 14},
            {"bucket": "d", "n": 12, "agree": 12},
            {"bucket": "e", "n": 14, "agree": 12},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        def write(name, payload):
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return path

        cpp_path = write("cpp.json", base)
        same_path = write("same.json", base)
        assert main(["--cpp-report", cpp_path, "--py-report", same_path]) == 0
        print("  [PASS] 口径一致时返回 0")

        # 只动余量子集的分母：这是最危险的一类漂移（率看起来还行，口径已经不同）
        drift = json.loads(json.dumps(base))
        drift["confident"]["n"] = 13
        drift_path = write("drift.json", drift)
        assert main(["--cpp-report", cpp_path, "--py-report", drift_path]) == 1
        print("  [PASS] 余量子集分母漂移时返回 1（并打印差异）")

        threshold_drift = json.loads(json.dumps(base))
        threshold_drift["thresholds"]["confident_margin"] = 3.0
        threshold_path = write("threshold.json", threshold_drift)
        assert main(["--cpp-report", cpp_path, "--py-report", threshold_path]) == 1
        print("  [PASS] 阈值漂移时返回 1")

        bucket_drift = json.loads(json.dumps(base))
        bucket_drift["strata"][0]["agree"] = 34
        bucket_path = write("bucket.json", bucket_drift)
        assert main(["--cpp-report", cpp_path, "--py-report", bucket_path]) == 1
        print("  [PASS] 逐桶分子漂移时返回 1")

        assert main(["--cpp-report", os.path.join(tmp, "missing.json"),
                     "--py-report", same_path, "--skip-if-missing"]) == SKIP_RETURN_CODE
        print("  [PASS] 报告缺失且给了 --skip-if-missing 时返回 77")
    print("[int8_crosscheck --self-test] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
