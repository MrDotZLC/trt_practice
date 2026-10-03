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
- `design.md`（P2-Design，2026-10-03 重做，**新增 Requirement Coverage、验证策略、D10/D11**）
- `review.md`（P3-Review，2026-10-03 新建，结论 PASS）
- `benchmark_before.md`（P4-Baseline，2026-10-03，**N/A：环境不可用，已按 Dependency Missing 记账**）
- `p5_s1_interface_spec.md`（P5 补充：S1 接口细化，2026-10-03，经作者确认）

旧版产物可用 `git show f9f8502:docs/dev/REQ-016-continuous-batching/<file>` 取回。

---

## Current Blockers

- **P4 / P7 搁置（2026-10-03）**：当前不在 GTX 1660 Ti 环境，无法取基线。按
  `phases/p4_baseline.md` 的 Dependency Missing 记 N/A；`benchmark_before.md` 写明环境恢复后
  必须补的四项测量。**批上限（`max_batch`）暂时只能取保守值并标注"待实测"**，不得写成实测结论。
- **本沙箱无编译能力**：没有 nvcc / cmake / 任何 C++ 编译器，也没有 TensorRT 与 build 目录，
  因此 P5 的 Exit Gate（"编译通过、无新增 warning"）在本环境**无法执行**。
  **S1 的代码改动（含采样器）全部未编译验证。**

---

## Next Action

1. **P5-S1（代码已落，待编译）**：改了 `llm_runner.hpp` / `llm_runner.cpp` /
   `sampler_common.hpp` / `sampler_kernels.cu`，新增 `tests/test_llm_runner_batch.cpp`。
   真机下一步：`cmake --build build -j` → 全量 `mini_trt_llm_tests` → 新增的
   `LlmRunnerBatchTest.*`（8 条）。编译错误与用例结果都要回填本文与 `test_plan.md`（P6）。
2. **P5-S2 / S3**：S1 编译通过后再做，不并笔提交。
3. 环境恢复后补 P4，再按 D10 的两种负载跑 P7。

---

## Implementation Plan

（技能 P5 要求：当前修改模块 / 预计文件 / 测试方式）

**当前修改模块**：S1 批量执行 —— 运行时（`LLMRunner`）的批量入口与批量缓冲；
外加一处**采样器随机流口径修正**（见 Recovery Notes）。**状态：已落码，未编译。**

| 文件 | 实际改动 |
|---|---|
| `mini_trt_llm/include/mini_trt_llm/core/llm_runner.hpp` | `Config` 加 `max_batch`（暂定 2）与 `enable_diagnostics`（D7，默认关）；新增 `GenerateRequest` / `GenerateResult` / `GenerateBatch`；私有函数与成员改成批量形态；新增 `d_seeds_` |
| `mini_trt_llm/src/core/llm_runner.cpp` | 批量入口与缓冲、按行绑定、批量采样、解码循环、结果切分；`Generate` 变单元素批包装；新增 prefill 末行收集 |
| `mini_trt_llm/include/mini_trt_llm/sampler/sampler_common.hpp` | `SamplerArgs` 新增 per-batch `seeds` |
| `mini_trt_llm/src/sampler/sampler_kernels.cu` | 新增 `RowUniform01`；4 个 kernel 签名 + 4 个调用点 + 6 个 launch 改为带 `seeds` |
| `mini_trt_llm/tests/test_llm_runner_batch.cpp`（新增） | 8 条用例（含随机采样的 AC1 对拍） |

**测试方式**：

1. 沙箱：**只能做与编译无关的静态检查**（括号平衡、未使用符号、外部符号签名、include 完整性）——
   本轮已做，抓到并修掉 3 处（漏掉的命名空间收尾大括号、未使用的 `BlocksForTokens`/`DecodeLogitsRow`、
   测试里的 `kPromptLen`）。
2. 真机必跑：`mini_trt_llm_tests` 全量（回归）+ `LlmRunnerBatchTest.*` 8 条。
   **块回收（AC3）的用例属于 S2**——S1 沿用"下次调用开头释放"的形态，跑到第 N 轮时最后一轮的块仍被持有。

---

## Phase History

- 2026-10-01: P0 -> P1（第一版）
- 2026-10-01: P1 -> P2（第一版）
- 2026-10-01: P2 -> Gate-A（第一版，未通过）
- 2026-10-02: 作者批准重做 P0–P2；三件产物按代码核实结果重写（第二版）
- 2026-10-02: P2 -> Gate-A（第二版，未通过）
- 2026-10-03: 作者批准"重新开始所有内容"；P0 -> P1 -> P2 -> P3 全量重做
- 2026-10-03: **Gate-A 通过**（D5 / D7 / D10 已拍板）
- 2026-10-03: P4 -> N/A（无 GPU 环境，搁置）→ P5-Implementation
- 2026-10-03: P5-S1 落码（5 个文件），**未编译验证**
- 2026-10-04: **S3 路线修正**：D12 由"固定槽位 + 全批定长"改为"**活跃批 + 压实**"——前者被证明会让上下文段把正在 generation 的行也算一遍、写回时覆盖其 prompt K/V；新增 D13（S3 padding 为生产路径、S4 packed 为默认路径）
- 2026-10-04: **S4 并入本 feature**（打包路径，默认；`REQ-020` 曾分配后同日撤销，编号作废不复用）
- 2026-10-03: 不变量 1 / 2 / 4 落地：D6 依据注释、D8 构造期 profile 校验、行号同源显式校验
- 2026-10-03: **P5-S2 落码**（6 个文件）：元数据缓冲按 max_batch 预分配、`NumFreeBlocks()`、
  `FreeSequence` 补"压实行 + 重建镜像"（补掉一个被掩盖的洞）、调用内归还（RAII 守卫）、
  D9 预算检查；新增 4 条用例
- 2026-10-03: 随机流改为 per-batch `seeds`（行号不进随机流）→ AC1 对**所有采样策略**成立；
  同时把静态审查反查出的 4 处文档↔代码不一致改齐

---

## Recovery Notes

- **Gate-A 决定（2026-10-03）**：D5 本 feature 先做；D7 诊断开关默认关；**D10 做 S3**；
  AC2 不下修；D11 补齐与后续 feature 的接口面。
- **P4 为什么是 N/A**：作者当前不在 GTX 1660 Ti 环境，先开发代码、GPU 测试搁置。
  `benchmark_before.md` 写明恢复后必须补的四项与口径（关诊断）；**不要**当成"性能已验证"。
- **代码在本环境无法编译**：写代码可以，但编译 / 单测 / 真机验证都要等环境恢复；
  在此之前所有 P5 改动都应视为未验证，**不能在文档里写"已通过"**。
- **随机流口径（2026-10-03，不要回退）**：采样不再用行号做随机输入。`SamplerArgs::seeds`（per-batch）
  非空时走 `RowUniform01` → `Uniform01(seeds[row], offset, 0)`；为空时保留旧行为（单行 / 兼容路径）。
  这条决定 AC1 能不能对所有采样策略成立（原来行号进哈希 → 同一请求换批位置就换输出）。
- **S3 的两条硬前提（2026-10-04）**：
  ① **每次引擎调用只装一种相**——上下文段只装本步新入批的序列、生成段只装本步在跑的序列。
     若让上下文段按整批跑，正在 generation 的行会被算出无意义的 K/V，写回按行号落进**它们自己的块**的
     `0..S-1`，把真实 prompt K/V 覆盖掉（静默错）。padding mask 拦不住（它只作用在图内 attention scores，
     而 K/V 是投影输出）。
  ② **写回必须带行映射**——`WritePrefillKV` 的行数原本取自"缓存已登记序列数"，S1/S2 里恒等于引擎的 B；
     活跃批下上下文段的 `B_new` 小于活跃序列数，必须显式给"引擎第 i 行 → 缓存第 rows[i] 行"。
     不改这一步会拿源缓冲里上一轮的残留行去覆盖别的序列。
- **S2（2026-10-03）**：元数据缓冲**构造期按 max_batch 预分配**（`PagedKVCache::Config::max_batch`，默认 4），
  跨请求不再重分配 → 指针恒定；块生命周期收敛到**调用内**（RAII 守卫），失败路径也归还；
  `FreeSequence` 内部压实行并重建镜像（此前只删 order_，靠"下次登记会整体重建"掩盖）。
- **S1 的批内约束（入口校验）**：prompt 等长（D2=A）；**同一种采样策略**；
  `top_k` / `top_p` / `seed` 均可逐行独立。
- **写冲突**：本 feature 与 `REQ-017` / `REQ-019` 共用运行时入口。`REQ-019` 早已登记；
  `REQ-017` 已于 2026-10-03 补登记。三者不并行改同一文件，本 feature 先做。
- **复核过、仍成立的既有事实**：
  1. 元数据缓冲设备分配容量够就复用；正确性依赖"decode 每步重绑"，不是"指针不变"。
  2. 引擎把 cache 第 0 维声明成 `ceil(n_positions / block_size)`，同一个数又当块表宽度用；
     运行时期物理池可以更大，此关系此前未进任何契约。
  3. 随批增长的显存大头是 **prefill logits**（`S=512` 约 98 MiB/条），不是 K/V 池（64 块约 72 MiB）。
  4. 建图代码不进引擎指纹，**改图必须手工 bump `graph_version`**。
  5. 运行期常驻诊断（同步 D2H + 逐层扫 K/V）**没有任何开关**——D7 要解决的是它。