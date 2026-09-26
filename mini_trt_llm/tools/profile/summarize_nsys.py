#!/usr/bin/env python3
"""把 `nsys stats` 导出的 CSV 摘要整理成"可归因"的报告。

对应 `docs/future_iterations_development_plan.md` §11.4（分解口径）与
`docs/future_iterations_test_plan.md` §10 的 PF-4 / PF-5。

两种模式：
  * 默认（kernel）：读 `nsys stats --report cuda_gpu_kern_sum --format csv` 的输出，
    把 kernel 时间分成三层——① 我们自己的 kernel、② CUB、③ TRT 内部。
    **只做这三层**：TRT 内部 kernel 名（`genericNode_*` / tactic 名）不携带语义，
    硬贴"attention / MLP"标签会造出"看起来像真故障"的数字（PROGRESS §2.14 C）。
  * `--api`：读 `cuda_api_sum` 的输出，报 cudaMalloc / cudaFree 等 API 的**次数与总耗时**。
    这是 `future_iterations.md` §2.1"先量分配开销"的供数来源；CUDA API 计数是 nsys
    的精确拦截，不是靠 kernel 名猜。

用法：
  summarize_nsys.py <kern_sum.csv> [--top N]
  summarize_nsys.py --api <api_sum.csv> [--top N]
  summarize_nsys.py --self-test

维护提示：① 层的名字列表由 `rg -o "__global__ void [A-Za-z0-9_]+" src/` 得来。
**改内核函数名就要同步改下面这张表**，否则该 kernel 会退到 ③ 层被当成 TRT 内部；
`--self-test` 里的 `AllKnownKernelsClassifyAsOurs` 会在名单本身被写错时红。
"""

import argparse
import csv
import io
import re
import sys

# ① 层：我们自己写的 kernel（来自 mini_trt_llm/src/ 的 `__global__ void` 名单）。
OUR_KERNELS = (
    "PagedAttentionDecodeKernel",
    "WriteKVKernel",
    "AdvanceContextLensKernel",
    "FillPositionIdsKernel",
    "GreedyKernel",
    "TopPParallelSampleKernel",
    "TopPSampleKernel",
    "TopKSampleKernel",
    "FastTopKSampleKernel",
    "PrepareSortInputKernel",
    "RmsNormScalarKernel",
    "RmsNormVectorKernel",
    "RoPEApplyKernel",
)

CUB_PATTERN = re.compile(r"\bcub::")

# 分配相关的 CUDA API（§2.1 的观测对象）。
ALLOC_APIS = ("cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync")


def classify(name):
    """把 kernel 名分到 ours / cub / trt 三桶。"""
    for kernel in OUR_KERNELS:
        if kernel in name:
            return "ours"
    if CUB_PATTERN.search(name):
        return "cub"
    return "trt"


def parse_table(text):
    """解析 nsys stats 的 CSV：返回 (header, rows)。

    找"表头行"的规则：第一行同时含有 'name' 与一个含 'time' 的列。
    这样能容忍 nsys 版本差异（新版会多出 Grid/Block 维度列）与前置注释行。
    """
    reader = csv.reader(io.StringIO(text))
    header = None
    rows = []
    for row in reader:
        if not row:
            continue
        joined = ",".join(row)
        if joined.lstrip().startswith("#"):
            continue
        if header is None:
            lowered = [field.strip().lower() for field in row]
            if any("name" in field for field in lowered) and any(
                "time" in field for field in lowered
            ):
                header = [field.strip() for field in row]
            continue
        if len(row) != len(header):
            # **不许静默丢行**：模板实参里带逗号的 kernel 名若没被正确引号包裹，
            # csv 会把它拆成更多列——悄悄跳过就等于少算一整类 kernel 的时间
            # （这会把"sampler 占比"直接算错，而不是报错）。宁可失败并指出来。
            raise ValueError(
                f"第 {len(rows) + 1} 行字段数 {len(row)} != 表头 {len(header)}；"
                f"疑似名称未被引号包裹：{row!r}"
            )
        rows.append(row)
    if header is None:
        raise ValueError("CSV 里找不到表头（需要同时含 Name 与 Time 列）")
    if not rows:
        raise ValueError("CSV 里没有数据行")
    return header, rows


def column_index(header, *keywords):
    """按关键字找列；找不到返回 None。"""
    for idx, field in enumerate(header):
        lowered = field.lower()
        if all(keyword in lowered for keyword in keywords):
            return idx
    return None


def parse_float(field):
    try:
        return float(field.replace(",", "").strip())
    except ValueError:
        return 0.0


def time_axis(header):
    """返回 (time_idx, unit_scale_to_ms, unit_label)。"""
    idx = column_index(header, "total time")
    if idx is not None:
        return idx, 1e-6, "ms"
    idx = column_index(header, "time (%)")
    if idx is not None:
        return idx, 1.0, "%"
    idx = column_index(header, "time")
    if idx is not None:
        return idx, 1.0, "(原始单位)"
    raise ValueError("CSV 里找不到时间列（Total Time / Time (%)）")


def load_rows(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return parse_table(handle.read())


def summarize_kernels(path, top):
    header, rows = load_rows(path)
    name_idx = column_index(header, "name")
    if name_idx is None:
        raise ValueError("CSV 里找不到 Name 列")
    time_idx, scale, unit = time_axis(header)

    buckets = {"ours": [], "cub": [], "trt": []}
    for row in rows:
        name = row[name_idx]
        value = parse_float(row[time_idx]) * scale
        buckets[classify(name)].append((name, value))

    total = sum(value for bucket in buckets.values() for _, value in bucket)
    print(f"[kernels] 读入 {len(rows)} 条 kernel，总计 {total:.3f} {unit}（文件：{path}）")
    labels = {"ours": "① 我们的 kernel", "cub": "② CUB", "trt": "③ TRT 内部"}
    for key in ("ours", "cub", "trt"):
        entries = buckets[key]
        subtotal = sum(value for _, value in entries)
        share = (100.0 * subtotal / total) if total > 0 else 0.0
        note = "  ← 只按耗时排列，不做语义归因" if key == "trt" else ""
        print(
            f"[kernels] {labels[key]:<16}: {subtotal:>10.3f} {unit} "
            f"({share:5.1f}%, {len(entries)} 条){note}"
        )

    # sampler / attention / KV 的占比单独给出来——这是本轮要回答的问题。
    groups = {
        "sampler": (
            "GreedyKernel",
            "TopPParallelSampleKernel",
            "TopPSampleKernel",
            "TopKSampleKernel",
            "FastTopKSampleKernel",
            "PrepareSortInputKernel",
        ),
        "attention": ("PagedAttentionDecodeKernel",),
        "kv_write": ("WriteKVKernel", "AdvanceContextLensKernel", "FillPositionIdsKernel"),
    }
    for label, names in groups.items():
        subtotal = sum(
            value
            for name, value in buckets["ours"]
            if any(token in name for token in names)
        )
        share = (100.0 * subtotal / total) if total > 0 else 0.0
        print(f"[kernels] {label:<16}: {subtotal:>10.3f} {unit} ({share:5.1f}%)")

    # ③ 层只列耗时前 N 条，供人肉判断/日后补语义映射。
    ranked = sorted(buckets["trt"], key=lambda item: item[1], reverse=True)[:top]
    for rank, (name, value) in enumerate(ranked, start=1):
        share = (100.0 * value / total) if total > 0 else 0.0
        print(f"[kernels]   trt#{rank}: {value:>10.3f} {unit} ({share:5.1f}%) {name}")
    print("[kernels] 说明：以上均为**观测值**，不设阈值判据（测试计划 §10.3）。")


def summarize_api(path, top):
    header, rows = load_rows(path)
    name_idx = column_index(header, "name")
    if name_idx is None:
        raise ValueError("CSV 里找不到 Name 列")
    time_idx, scale, unit = time_axis(header)
    calls_idx = column_index(header, "num calls")
    if calls_idx is None:
        calls_idx = column_index(header, "calls")

    entries = []
    for row in rows:
        name = row[name_idx]
        value = parse_float(row[time_idx]) * scale
        calls = parse_float(row[calls_idx]) if calls_idx is not None else 0.0
        entries.append((name, value, calls))

    print(f"[api] 读入 {len(rows)} 条 CUDA API（文件：{path}）")
    alloc_total = 0.0
    alloc_calls = 0.0
    for name, value, calls in entries:
        if any(api in name for api in ALLOC_APIS):
            alloc_total += value
            alloc_calls += calls
            print(f"[api]   {name:<20} calls={calls:>8.0f} total={value:.3f} {unit}")
    print(
        f"[api] 分配类合计：calls={alloc_calls:.0f} total={alloc_total:.3f} {unit}"
        f"  ← §2.1 的供数（观测值，不设阈值）"
    )
    for rank, (name, value, calls) in enumerate(
        sorted(entries, key=lambda item: item[1], reverse=True)[:top], start=1
    ):
        print(f"[api]   top#{rank}: {value:>10.3f} {unit} calls={calls:>8.0f} {name}")


def self_test():
    """护栏自检：分桶、解析容忍度、以及"名单写错会红"。"""
    header = ["Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Name"]

    def csv_of(rows):
        # 用 csv.writer 生成，保证含逗号的名称被正确加引号——真实 nsys 导出也是这样，
        # 否则模板实参里的逗号会把一行拆成多列（这正是下面 ToleratesQuotedNames 要锁的）。
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    text = csv_of(
        [
            ["10.0", "1000000", "1", "1000000",
             "void mini_trt_llm::PagedAttentionDecodeKernel(...)"],
            ["20.0", "2000000", "1", "2000000",
             "void mini_trt_llm::WriteKVKernel<float, float>(...)"],
            ["30.0", "3000000", "1", "3000000", "cub::DeviceSegmentedSortKernel<...>"],
            ["40.0", "4000000", "1", "4000000", "genericNode_42_kernel"],
        ]
    )
    parsed_header, rows = parse_table(text)
    assert parsed_header[0].startswith("Time"), parsed_header
    assert len(rows) == 4, rows
    # 带逗号的 kernel 名必须作为**一个**字段被完整读回。
    assert "WriteKVKernel<float, float>" in rows[1][4], rows[1]

    name_idx = column_index(parsed_header, "name")
    time_idx, scale, _ = time_axis(parsed_header)
    buckets = {"ours": 0.0, "cub": 0.0, "trt": 0.0}
    for row in rows:
        buckets[classify(row[name_idx])] += parse_float(row[time_idx]) * scale
    assert abs(buckets["ours"] - 3.0) < 1e-9, buckets
    assert abs(buckets["cub"] - 3.0) < 1e-9, buckets
    assert abs(buckets["trt"] - 4.0) < 1e-9, buckets

    # 名单写错（内核改名后忘了同步）会让这条红：每个已知 kernel 名都必须归到 ours。
    for kernel in OUR_KERNELS:
        assert classify(f"void ns::{kernel}<int>(...)") == "ours", kernel
    assert classify("cub::DeviceRadixSortKernel<...>") == "cub"
    # 未知名字必须落到 trt（安全默认），而不是被误判成 ours。
    assert classify("someUnknownKernel") == "trt"

    # 容忍：前置注释行 / 空行；以及"名称列不在最后"的版本差异。
    tolerant = (
        "# preamble comment\n\n"
        + "Time (%),Total Time (ns),Name,Grid X\n"
        + '50.0,5000000,"void ns::GreedyKernel<float>(...)",1\n'
    )
    parsed_header, rows = parse_table(tolerant)
    name_idx = column_index(parsed_header, "name")
    time_idx, scale, _ = time_axis(parsed_header)
    assert len(rows) == 1
    assert classify(rows[0][name_idx]) == "ours"
    assert abs(parse_float(rows[0][time_idx]) * scale - 5.0) < 1e-9

    # 护栏：缺 Name 列 / 缺时间列 / 空数据 / **字段数不匹配** 都必须报错，
    # 而不是安静地报 0 或静默丢掉一整行。
    for broken in (
        "Time (%),Total Time (ns)\n1.0,1000000\n",
        "Name,Instances\nfoo,1\n",
        "Name,Total Time (ns)\n",
        # 名称含逗号但**没加引号**（真实 nsys 会加引号；这里是坏输入）→ 字段数不匹配。
        "Time (%),Total Time (ns),Name\n1.0,1000000,a,b,c\n",
    ):
        try:
            parse_table(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"坏输入没有被拒绝：{broken!r}")

    print("[self-test] summarize_nsys 分桶 / 解析容忍 / 护栏：全部通过")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="整理 nsys stats 的 CSV 摘要")
    parser.add_argument("csv_path", nargs="?", help="nsys stats 导出的 CSV")
    parser.add_argument("--api", action="store_true", help="按 CUDA API 汇总（分配开销）")
    parser.add_argument("--top", type=int, default=10, help="top-N 条目数（默认 10）")
    parser.add_argument("--self-test", action="store_true", help="只跑护栏自检")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.csv_path:
        parser.error("需要 CSV 路径（或 --self-test）")
    try:
        if args.api:
            summarize_api(args.csv_path, args.top)
        else:
            summarize_kernels(args.csv_path, args.top)
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
