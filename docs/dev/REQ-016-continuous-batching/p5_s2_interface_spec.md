# P5-S2 接口细化（可评审后再落代码）

<!--
本文件只写"接口形态与改动点"，不含产品代码。作用：让 S2（资源生命周期）的签名、
缓冲归属与失败语义在写代码之前先被作者过一遍。
S1 的方案与实现见 p5_s1_interface_spec.md；S3（请求级调度）另立文件。
-->

## 0. 已核实的既有实现（决定 S2 的改动面）

| 事实 | 证据（2026-10-03 读码） | 对 S2 的意义 |
|---|---|---|
| 元数据缓冲**每次登记都重建** | `paged_kv_cache.cpp` 的 `AllocateSequence` 里 `block_tables_host_.assign(batch * W, 0)` / `context_lens_host_.assign(batch, 0)` + 两个设备缓冲 `Allocate(...)` | S2 要改成**构造期按最大批预分配**，登记时只填自己那一行 |
| 设备侧"容量够就复用" | `utils/memory_pool.cpp` 的 `Allocate`：`size_ >= bytes` 直接返回；**`bytes == 0` 会 Free** | 预分配后不得再用 0 字节调用，否则会把缓冲释放掉 |
| **`FreeSequence` 不重建 host 镜像** | 它只做 `sequences_.erase` + `order_.erase`（行会前移），镜像与设备侧都不动 | **这是一个被掩盖的洞**：当前靠"下一次 `AllocateSequence` 会整体重建"兜住；一旦出现"批内退出"（S3），被移除行之后的块表行就会**错位** |
| 上传口径 | `UploadMetadata` 整块拷 host 镜像到设备 | 预分配后它拷的是最大批尺寸（每请求多拷几百字节，可忽略） |
| 块不足的现有语义 | `AllocateSequence` 先比 `NumFree()` 再分配（不抛异常） | D9 的"先检查后分配"已成立；S2 给它**加上可读的错误信息**（需要多少 / 空闲多少） |
| 块池没有任何可见性接口 | `BlockAllocator::NumFree()` 存在，但 `PagedKVCache` 没有暴露 | AC3 的断言和 D9 的日志都需要它 → S2 加 `NumFreeBlocks()` |

## 1. 接口变化

### 1.1 `PagedKVCache`

```cpp
struct Config {
    ...现有字段...
    // 同时存在的序列数上限（= LLMRunner::Config::max_batch）。
    // **必须与 runner 侧一致**：元数据缓冲按它预分配，超出的行没有地方放。
    int32_t max_batch = 1;
};

// 块池里还有多少块空闲（AC3 的断言依据；D9 的预算日志也用它）。
int32_t NumFreeBlocks() const;
```

语义收紧（**行为契约，不只是实现细节**）：

| 入口 | S2 之后的契约 |
|---|---|
| 构造 | 按 `max_batch` 一次性分配 `block_tables[max_batch × W]` 与 `context_lens[max_batch]`；此后**指针不再变化**（不变量 5） |
| `AllocateSequence(seq_id, max_tokens)` | 只在"批内追加一行"：检查 `order_.size() < max_batch`、块够不够 → 分配块 → 追加 `order_` → 填**自己那一行**的镜像。**不重建、不重分配** |
| `FreeSequence(seq_id)` | 释放块 → 从 `order_` 移除 → **把后面的行整体前移并重建 host 镜像**（补齐上面那个洞）。设备侧由调用方随后 `UploadMetadata` 同步 |
| `UploadMetadata` | 不变（整块拷 host 镜像） |

### 1.2 `LLMRunner`

- 构造期把 `config_.max_batch` 传给 `PagedKVCache::Config::max_batch`（并校验 `num_blocks >= max_blocks_per_seq`，这条 `PagedKVCache` 已自检）。
- **改为"调用内归还"**：用一个作用域守卫（RAII）在 `GenerateBatch` 的任何出口归还本批的序列，替代现在 6 处重复的失败路径 `FreeSequence` 循环，并删掉"开头释放上一轮"的逻辑与成员 `active_seqs_`。
  - 为什么这也算 S2 而不是"顺手重构"：AC3 要求"跑 N 轮后空闲块回到初始值"，而 S1 的"下次调用开头释放"让**最后一轮的块仍被持有** → 不满足；守卫是达成该语义的最小手段。
- **入口预算检查（D9 的完整形态）**：登记任何序列之前先算
  `Σ_b ceil((prompt_len + max_new_b) / block_size)`，与 `NumFreeBlocks()` 比；不足则**整批拒绝**并打印"需要 X 块、空闲 Y 块"。

## 2. 失败语义（S2 新增/收紧的部分）

| 情形 | 行为 |
|---|---|
| 预算不足（总需求 > 空闲） | 整批拒绝，日志给 **需要 / 空闲** 两个数（现在只报"需要"） |
| 批大小 > `PagedKVCache::Config::max_batch` | `AllocateSequence` 返回 false + 明确日志（**新失败路径**，S1 时不可能发生，S2 起是安全网） |
| 任一步失败（绑定 / 入队 / 采样 / 追加） | 由作用域守卫归还本批**已分配**的块；空闲块数回到调用前水位 |
| 调用正常结束 | 同样由守卫归还 → **AC3 成立** |

## 3. 生命周期与缓冲归属

| 资源 | 创建 | 释放 | S2 前后的差别 |
|---|---|---|---|
| K/V 池（`num_blocks` 块） | 构造期 | 析构 | 不变 |
| `block_tables` / `context_lens` 设备缓冲 | 构造期（**按 max_batch**） | 析构 | **S2：不再随批大小重分配** |
| 两者的 host 镜像 | 构造期 | 析构 | S2：尺寸固定为 `max_batch × W` / `max_batch` |
| 每序列的块 | 登记时 | **调用结束或失败时由守卫归还** | **S2：从"下次调用开头释放"改为"调用内归还"** |
| runner 的批量缓冲 | 首次调用按容量水位 | 析构 | 不变（S1 已做） |

**刻意不做的一件事**：`FreeSequence` 之后**不清零**被释放的行。理由：真正防止越界读的是"引擎只读 B 行"这个范围约束；清零并不能提供检测能力（读到 0 号物理块同样不报错），反而多一次 memset。

## 4. 改动点（文件级）

| 文件 | 改动 |
|---|---|
| `include/mini_trt_llm/kv_cache/paged_kv_cache.hpp` | `Config` 加 `max_batch`；加 `NumFreeBlocks()`；更新 `AllocateSequence` / `FreeSequence` 的契约注释 |
| `src/kv_cache/paged_kv_cache.cpp` | 构造期预分配；`AllocateSequence` 去掉重建/重分配、只填自己那行；`FreeSequence` 补"压实行 + 重建镜像"；加 `NumFreeBlocks()` |
| `include/mini_trt_llm/core/llm_runner.hpp` | 删 `active_seqs_` 成员 |
| `src/core/llm_runner.cpp` | 构造期传 `max_batch`；加作用域守卫；加预算检查（D9 完整形态）；删"开头释放上一轮" |
| `tests/test_llm_runner_batch.cpp` | +2 条：块回收（AC3）、失败后水位不变 |
| `tests/test_paged_kv_cache.cpp` | +1 条：`AllocateSequence` / `FreeSequence` 往返后元数据指针不变（不变量 5） |

**规模提示**：6 个文件、预计 200~300 行，超技能的软约束（≤3 文件）——提交说明里要写明原因（这是 S2 的定义性改动，跨 kv_cache 与 runner 两个模块）。

## 5. 测试方式

| 用例 | 判据 | 归属 |
|---|---|---|
| `FreeBlocksReturnAfterBatch` | 记录水位 → 跑一批 → 空闲块数**等于**调用前水位（AC3） | 新增（S2） |
| `FreeBlocksUnchangedAfterFailure` | 用"块不足"触发整批拒绝 → 水位不变（覆盖失败路径的归还） | 新增（S2） |
| `MetadataPointersStableAcrossAllocFree` | 同一 cache 上 `AllocateSequence` → `FreeSequence` → 再 `AllocateSequence`，`block_tables()` / `context_lens()` 返回值不变（不变量 5） | 新增（S2，加在 `test_paged_kv_cache.cpp`） |
| S1 的 8 条 | 必须仍然全绿（尤其 AC5：调用内归还不得改变 token 输出） | 回归 |

## 6. S2 不做的事（边界）

- **不做**批内退出后的连续批调度（S3）：`FreeSequence` 的压实能力是给 S3 备的，但 S2 不引入调度循环。
- **不做**抢占与换出（requirement 的 Excluded）。
- **不改**引擎侧契约（`max_blocks_per_seq` 与 cache 第 0 维的关系不变，D6）。
- **不动** K/V 缓冲的分配（仍是构造期一次）。

## 7. 待作者确认的三点

1. **`PagedKVCache::Config::max_batch` 与 `LLMRunner::Config::max_batch` 必须一致**（由 runner 传下去、并在构造期校验）。是否认可这个"单一来源 + 安全网"的写法？
2. **`FreeSequence` 改为"内部压实行并重建镜像"**（而不是让调用方负责）。这是把 S1 里那个被掩盖的洞补齐；代价是 `FreeSequence` 的语义变重。是否认可？
3. **AC3 的判定口径**：空闲块数**等于**调用前水位（引用相等）。是否认可（而不是"不低于"）？

## 8. 实现时新发现的约束（2026-10-03，落码时记录）

1. **`PagedKVCache::Config::max_batch` 的默认值必须"多序列安全"（定为 4）**。
   若默认 1，既有的多序列用例会被新的容量校验直接拒绝（`test_paged_kv_cache.cpp` 从 `MakeConfig()`
   建的 cache 会登记两条序列）。默认 4 的依据：项目 MVP 引擎 profile 的批上限就是 4
   （`core/builder.hpp` 的 max_prefill_batch / max_decode_batch）。测试里另外**显式**声明了
   `max_batch = 4`，不吃默认值。
2. **AC3 需要一个 runner 级访问器**：`LLMRunner::NumFreeKvBlocks()`。没有它，AC3 只能测到
   `PagedKVCache` 层面（测不到 runner 的守卫是否真的归还）。这是为判据加的 API 面，不是顺手加的。
3. **`FreeSequence` 的压实有直接用例**：`PagedKVCacheTest.FreeSequenceCompactsRemainingRows`
   —— 登记 3 条 → 释放中间那条 → `UploadMetadata` → 读回设备侧镜像逐行核对；
   不变量 5 另有 `MetadataPointersStableAcrossAllocFree`（指针恒定）。
4. **`AllocateSequence` 里必须整行写块表**：该行可能被上一轮的序列用过，尾部残留必须清掉
   （块表宽度是引擎契约的一部分，残留块号会让 PagedAttention 读到别的物理块）。
