# Development State

| 字段 | 取值 |
| --- | --- |
| workflow | feature |
| feature | REQ-016-continuous-batching |
| phase | P5-Implementation |
| phase_index | 5 |
| status | in-progress |
| updated | 2026-10-03 |
| owner | Codex |

---

## Completed Artifacts

- `requirement.md`（P0-Requirement，2026-10-03 重做）
- `analysis.md`（P1-Analysis，2026-10-03 重做，**新增 Terminology**）
- `design.md`（P2-Design，2026-10-03 重做，**新增 Requirement Coverage、验证策略、D11**）
- `review.md`（P3-Review，2026-10-03 新建，结论 PASS）
- `benchmark_before.md`（P4-Baseline，2026-10-03，**N/A：环境不可用，已按 Dependency Missing 记账**）
- `p5_s1_interface_spec.md`（P5 补充：S1 接口细化，2026-10-03，经作者确认）

旧版产物可用 `git show f9f8502:docs/dev/REQ-016-continuous-batching/<file>` 取回。

---

## Current Blockers

- **P4 / P7 搁置（2026-10-03）**：当前不在 GTX 1660 Ti 环境，无法取基线。按
  `phases/p4_baseline.md` 的 Dependency Missing 记 N/A，`benchmark_before.md` 已写明
  环境恢复后必须补的四项测量；**批上限暂时只能取保守值并标注为暂定**，不得写成实测结论。
- **本沙箱无编译能力**：没有 nvcc / cmake / 任何 C++ 编译器，也没有 TensorRT 与 build 目录，
  因此 P5 的 Exit Gate（"编译通过、无新增 warning"）在本环境**无法执行**；
  代码只能先在真机侧编译。**S1 的三个文件已按此前提落码，全部未编译验证。**
- 三处留痕只完成两处：`benchmark_before.md` 与本文；`docs/PROGRESS.md` 的"已知问题与坑"
  按 §6 的分工要等阶段/批次收口时才补。

---

## Next Action

1. **P5-S1（代码已落，待编译）**：改了 `llm_runner.hpp` / `llm_runner.cpp`，新增 `tests/test_llm_runner_batch.cpp`。
   下一步（真机）：`cmake --build build -j` → 全量 `mini_trt_llm_tests` → 新增的 `LlmRunnerBatchTest.*`；
   编译错误与用例结果都要回填本文与 `test_plan.md`（P6）。
2. **P5-S2 / P5-S3**：S1 通过后再做，不并笔提交。
3. 环境恢复后补 P4，再按 P7 口径判收益；S3 的收益结论以那批数据为准（D10）。

---

## Implementation Plan

（技能 P5 要求：当前修改模块 / 预计文件 / 测试方式）

**当前修改模块**：S1 批量执行 —— 运行时（`LLMRunner`）的批量入口与批量缓冲。**状态：已落码，未编译。**

**预计文件**（软约束 ≤3 个文件 / ≤300 行；超限需在提交说明里写明原因）：

| 文件 | 预计改动 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp` | 新增批量请求/批量结果结构与批量入口声明；保留现有单序列入口不动（AC5） |
| `mini_trt_llm/src/core/llm_runner.cpp` | 批量形状设置、按行绑定、批量采样、结果切分；单序列入口改为调用批量入口 |
| `mini_trt_llm/tests/test_llm_runner_batch.cpp` | 新增 runner 级对拍用例（批量 == 逐条单跑） |

**测试方式**：

1. 沙箱可跑的：与编译器无关的静态检查（本环境**只能做这一层**）。
2. 真机必跑的：`mini_trt_llm_tests` 全量（回归）+ 新增对拍用例（AC1 / AC5）+ 块回收用例（AC3）。
3. 口径：AC1 要求"逐 token 逐位相同"，不接受"接近"。

---

## Phase History

- 2026-10-01: P0 -> P1（第一版）
- 2026-10-01: P1 -> P2（第一版）
- 2026-10-01: P2 -> Gate-A（第一版，未通过）
- 2026-10-02: 作者批准重做 P0–P2；三件产物按代码核实结果重写（第二版）
- 2026-10-02: P2 -> Gate-A（第二版，未通过）
- 2026-10-03: 作者批准"重新开始所有内容"；P0 -> P1 -> P2 -> P3 全量重做（本版）
- 2026-10-03: **Gate-A 通过**（D5 / D7 / D10 已拍板）
- 2026-10-03: P4 -> N/A（无 GPU 环境，搁置）→ P5-Implementation
- 2026-10-03: P5-S1 落码（3 个文件），**未编译验证**（本环境无编译器）

---

## Recovery Notes

- **Gate-A 决定（2026-10-03）**：D5 本 feature 先做；D7 诊断开关默认关；**D10 做 S3**；
  AC2 不下修；D11 补齐与后续 feature 的接口面。
- **P4 为什么是 N/A**：作者当前不在 GTX 1660 Ti 环境，先开发代码、GPU 测试搁置。
  `benchmark_before.md` 写明了恢复后必须补的四项与口径（关诊断）；**不要**把它当成"性能已验证"。
- **代码在本环境无法编译**：写代码可以，但编译/单测/真机验证都要等环境恢复；
  在此之前所有 P5 改动都应视为未验证，不能在文档里写"已通过"。
- **写冲突**：本 feature 与 `REQ-017` / `REQ-019` 共用运行时入口。`REQ-019` 早已登记；
  `REQ-017` 已于 2026-10-03 补登记（记在它的 Current Blockers 与 Recovery Notes）。
- **复核过、本轮仍成立的既有事实**：
  1. 元数据缓冲设备分配容量够就复用；正确性依赖"decode 每步重绑"，不是"指针不变"。
  2. 引擎把 cache 第 0 维声明成 `ceil(n_positions / block_size)`，同一个数又当块表宽度用；
     运行时期物理池可以更大，此关系此前未进任何契约。
  3. 随批增长的显存大头是 **prefill logits**（`S=512` 约 98 MiB/条），不是 K/V 池（64 块约 72 MiB）。
  4. 建图代码不进引擎指纹，**改图必须手工 bump `graph_version`**。
  5. 运行期常驻诊断（同步 D2H + 逐层扫 K/V）**没有任何开关**，与 build 期的
     `export_diagnostics` 不是一回事——D7 要解决的是前者。