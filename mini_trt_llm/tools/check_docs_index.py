#!/usr/bin/env python3
"""比对 docs/dev/INDEX.md §1 与各条目 STATE.md 的 phase / status，不一致就判红。

为什么需要它（口径出处：docs/README.md §3 与 §4.4，2026-10-05 定）：
条目状态只有**一个真值**——`docs/dev/<REQ>/STATE.md`；`docs/dev/INDEX.md` §1 是**唯一允许的
快照副本**；`docs/PROGRESS.md` 只写人话与链接，不再复制 phase / status 值。
三层并存本身没问题，问题在"副本没人核对"，而它已经实际漂移过两次：
2026-10-03 REQ-016 走过 Gate-A 进 P5 时 INDEX 未同步；2026-10-05 又发现 PROGRESS §4.7
把 REQ-017 写成"在实现阶段"，而 STATE 与 INDEX 都已是"测试阶段"。
本脚本把这类漂移变成机器可判的红，替代人肉记忆。

判据（全部以 STATE 侧为准）：
1. INDEX 表里每个目录都存在，且该目录 STATE.md 的 `feature` 字段与目录名一致；
2. INDEX 的 `phase` / `status` 与 STATE 的同名字段**逐字相同**；
3. 反向也查：`docs/dev/` 下有 STATE.md 的目录，INDEX 表里必须有行
   （`REQ-000` 这类已退休编号没有目录，不在检查范围）。

用法：
    python3 mini_trt_llm/tools/check_docs_index.py
    python3 mini_trt_llm/tools/check_docs_index.py --verbose
    python3 mini_trt_llm/tools/check_docs_index.py --self-test

退出码：0 = 一致；1 = 有漂移或结构不符；2 = 用法错误。

**尚未接入构建 / ctest**：接入属"改配置文件"，需作者点名后再做（AGENTS.md §0.2）。
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

# INDEX §1 的一行：| `REQ-016` | `REQ-016-continuous-batching` | 进行中 | P5-Implementation | in-progress | 一句话 |
REQ_ROW_RE = re.compile(
    r"^\|\s*`(?P<req>REQ-\d{3})`\s*\|\s*`(?P<dir>[^`]+)`\s*\|(?P<cat>[^|]*)\|"
    r"(?P<phase>[^|]*)\|(?P<status>[^|]*)\|"
)
# STATE 首部的字段表：| phase | P5-Implementation |
FIELD_ROW_RE = re.compile(r"^\|\s*(?P<key>workflow|feature|phase|status|updated)\s*\|\s*(?P<val>[^|]*)\|\s*$")


def parse_index(index_path):
    """返回 {req: {"dir":.., "phase":.., "status":..}}；只认 `REQ-NNN` 开头的数据行。"""
    rows = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        m = REQ_ROW_RE.match(line.strip())
        if not m:
            continue
        rows[m.group("req")] = {
            "dir": m.group("dir").strip(),
            "phase": m.group("phase").strip(),
            "status": m.group("status").strip(),
        }
    return rows


def parse_state(state_path):
    """返回 STATE.md 首部字段表的 {key: value}。"""
    fields = {}
    for line in state_path.read_text(encoding="utf-8").splitlines():
        m = FIELD_ROW_RE.match(line.strip())
        if m:
            fields[m.group("key")] = m.group("val").strip()
        elif fields and line.strip() and not line.lstrip().startswith("|"):
            break  # 字段表在前 15 行内；出了表就停，避免误吃正文里的同名表格
    return fields


def check(index_path, dev_dir):
    """返回 (problems, checked_count)。problems 是可直接打印的中文说明列表。"""
    problems = []
    rows = parse_index(index_path)
    if not rows:
        return ["INDEX 里一行 `REQ-NNN` 条目都没解析到——表格式变了？"], 0

    for req, row in sorted(rows.items()):
        entry_dir = dev_dir / row["dir"]
        state_path = entry_dir / "STATE.md"
        if not state_path.is_file():
            problems.append(f"{req}: INDEX 指向 `{row['dir']}`，但该目录没有 STATE.md")
            continue
        fields = parse_state(state_path)
        if fields.get("feature", "") != row["dir"]:
            problems.append(
                f"{req}: STATE.md 的 feature='{fields.get('feature', '')}'，与目录名 '{row['dir']}' 不一致"
            )
        for key in ("phase", "status"):
            want = row[key]
            got = fields.get(key, "")
            if want != got:
                problems.append(f"{req}: {key} 漂移——INDEX='{want}'，STATE='{got}'")

    # 反向：有 STATE.md 的目录必须在 INDEX 里登记
    for entry_dir in sorted(p for p in dev_dir.iterdir() if p.is_dir()):
        if not (entry_dir / "STATE.md").is_file():
            continue
        if not any(row["dir"] == entry_dir.name for row in rows.values()):
            problems.append(f"{entry_dir.name}: 目录里有 STATE.md，但 INDEX 表里没有对应行")

    return problems, len(rows)


def self_test():
    """最小自证：一对一致的样本必须过，一处 phase 漂移必须被抓到。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dev = root / "docs" / "dev"
        entry = dev / "REQ-999-sample"
        entry.mkdir(parents=True)
        (root / "docs" / "INDEX.md").write_text(
            "| 编号 | 目录 | 类别 | phase | status | 一句话 |\n"
            "|---|---|---|---|---|---|\n"
            "| `REQ-999` | `REQ-999-sample` | 历史 | P5-Implementation | in-progress | 样本 |\n",
            encoding="utf-8",
        )
        state = entry / "STATE.md"
        state.write_text(
            "| 字段 | 取值 |\n|---|---|\n| feature | REQ-999-sample |\n"
            "| phase | P5-Implementation |\n| status | in-progress |\n",
            encoding="utf-8",
        )
        problems, _ = check(root / "docs" / "INDEX.md", dev)
        if problems:
            print("SELFTEST FAIL: 一致样本被判红：", problems)
            return 1
        state.write_text(
            "| 字段 | 取值 |\n|---|---|\n| feature | REQ-999-sample |\n"
            "| phase | P6-Test |\n| status | in-progress |\n",
            encoding="utf-8",
        )
        problems, _ = check(root / "docs" / "INDEX.md", dev)
        if not any("phase 漂移" in p for p in problems):
            print("SELFTEST FAIL: 漂移样本没被抓到：", problems)
            return 1
    print("SELFTEST PASS：一致样本通过、漂移样本被抓到")
    return 0


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="校验 docs/dev/INDEX.md 与各 STATE.md 的 phase / status")
    parser.add_argument("--root", default=str(repo_root), help="仓库根目录（默认按脚本位置推断）")
    parser.add_argument("--verbose", action="store_true", help="打印检查范围与逐条结果")
    parser.add_argument("--self-test", action="store_true", help="跑自证后退出")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    root = Path(args.root).resolve()
    index_path = root / "docs" / "dev" / "INDEX.md"
    dev_dir = root / "docs" / "dev"
    if not index_path.is_file():
        print(f"用法错误：找不到 {index_path}", file=sys.stderr)
        return 2

    problems, checked = check(index_path, dev_dir)
    if args.verbose:
        print(f"检查了 INDEX 的 {checked} 行，对照 {dev_dir} 下各条目的 STATE.md")
    if problems:
        print(f"发现 {len(problems)} 处漂移（口径：STATE.md 是真值，INDEX 是快照）：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK：{checked} 条条目的 phase / status 与 STATE.md 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
