# Design

<!--
本版 2026-10-03 重做。与上一版的差别：补齐 `## Requirement Coverage`（P2 的 Exit Gate 产物）、
新增 `## 验证策略`（承载验证型判据的落点）、把 S1/S2/S3 各写成可执行设计而非里程碑名。
-->

## Overview

分三个里程碑，每个都能独立验收：

| 里程碑 | 内容 | 为什么这样切 |
|---|---|---|
| **S1 批量执行** | 一次调用同时跑 N 条序列（prefill + decode），数值与逐条单跑一致 | 先把"批量正确"做扎实；不含调度变量，读数可归因 |
| **S2 资源生命周期** | 元数据缓冲按最大批预分配、块回收、跑 N 轮不泄漏 | 没有它，S3 的调度循环会一直重分配，并让"指针是否稳定"变成隐性依赖 |
| **S3 请求级调度** | 请求在步边界进入 / 退出（最小连续批） | 服务化的真正入口；也是判据 2 的完整形态所在 |
| **S4 打包路径** | 不等长批的 prefill 按各序列真实长度计费（不填充），作为**默认路径** | 消掉填充带来的算力浪费与"非真实位置"；S3 的填充路径转为对照 / 回退 |

S1 内部再分两小步：先只支持**批内 prompt 等长**（D2=A），再决定是否支持不等长。
P5 按里程碑分批实现与提交，不把三个里程碑压成一笔。

## Architecture

```text
调用方（多请求）
      ↓
 ┌─────────────────────────────────────────┐
 │ LLMRunner                               │
 │  ├─ 批状态表（每序列：id/状态/长度/参数） │
 │  ├─ 批量形状设置（B, S）与缓冲            │
 │  └─ 采样（per-sequence 参数）             │
 └───────┬─────────────────────┬───────────┘
         ↓                     ↓
   prefill 引擎           decode 引擎
   （[B,S] 前向，          （[B,1] 前向，
    每层导出 K/V）          读分页 cache）
         ↓                     ↓
 ┌─────────────────────────────────────────┐
 │ PagedKVCache（块池 + 每序列块表/长度）    │
 └─────────────────────────────────────────┘
```

## Module Design

| Module | Responsibility | Dependency |
|---|---|---|
| 批状态表（新增，运行时内部） | 每序列的 id、prompt、已生成长度、采样参数、结果缓冲、状态机 | 无 |
| `LLMRunner` | 批量入口；设 `[B,S]` / `[B,1]` 形状；绑定 per-sequence 张量；聚合结果 | 两张引擎、批状态表 |
| `PagedKVCache` | 元数据缓冲**按最大批预分配**；一次导出整批块表 / 长度；块回收 | `BlockAllocator` |
| `BlockAllocator` | 保持不变（free-list 已够用）；其失败是抛异常，调度入口必须先检查后分配（D9） | — |
| 采样器 | 已支持 per-batch 参数指针，无需改 kernel | 现有 sampler |
| `GPT2ModelBuilder` | S1/S2 **不改**；D2 选 B 时才加 mask 输入 | — |

## Data Structure

```text
SequenceState {
    int32_t seq_id;              // 分页 cache 的键，批内唯一
    int32_t prompt_len;          // S1：批内必须相同（D2=A）
    int32_t generated;           // 已生成 token 数
    int32_t context_len;         // prompt_len + generated（host 记账用）
    SamplerParams { top_k, top_p, seed };
    Status { kWaiting, kRunning, kFinished };
}

Batch {
    std::vector<SequenceState> sequences;   // 下标 = 引擎输入的行号
}
```

设备侧每行一个元素：`context_lens[B]`、`position_ids[B]`、`top_k[B]`、`top_p[B]`。
块表 `[B, max_blocks_per_seq]` 沿用现有布局（不足宽度补 0）。
结果缓冲从"一条序列的 `[max_new]`"改成**每序列一段**（D4）。

## Runtime Flow

### S1 静态批（一批同进同出）

```text
N 条等长请求（batch B = N）
   ↓
批状态表登记（先查空闲块是否够，再分配块 + 写 prompt）→ 整批一次上传元数据
   ↓
一次批量 prefill：[B,S] → 每层 K/V → 写 cache → 收集末行 → 采样 → token_0[B]
   ↓
固定跑满 max_new 步（静态批同进同出）：
   设形状 [B,1] → 填 position_ids[B] → decode → 追加 K/V（整步一次）→ 采样 token_i[B]
   ↓
一次性 D2H → 按序列切分 → 逐条截断（各自 max_new_tokens 与 EOS）
```

### S2 资源生命周期（把缓冲与块的生命周期收进 API）

```text
构造期：按 max_batch 预分配 block_tables[B_max, W] / context_lens[B_max] / top_k[B_max] / top_p[B_max]
        —— 此后跨请求不再重分配，指针恒定
请求开始：先检查（需要块数 ≤ 空闲块数）→ 再分配 → 回填 host 镜像 → 整批上传
请求结束：归还块 与 "从批内移除" 必须同一步完成（否则块池泄漏）
```

### S3 请求级调度（活跃批 + padding mask）

**路线（D12）**：批 = **本步活跃的序列**（不保留空槽）；不等长 prompt 用右填充 + padding mask（D2=B）；
序列退出即从活跃表移除并**压实行号**，逐行缓冲每步重建。
**核心前提：每次引擎调用只装一种相**——这是不出现"覆盖"的必要条件（见本节末）。

状态机（每序列）：

```text
kWaiting ──准入（先检查后分配）──→ kRunning(context) ──首次 prefill 完成──→ kRunning(generation)
   ↑                                        │                                     │
   └── 批满 / 块不足，留在等待队列 ←─────────┴── EOS（设备侧 flag）/ 达到 max_new → kFinished
                                                    （本形态不做抢占与换出）
```

一步的流程：

```text
① 退出：读上一步**异步回读**的 finish flag → 命中的序列归还块、从活跃表移除、压实行号
② 进入：从等待队列取 arrival_step ≤ 当前步 的请求，先检查空闲块（D9），放入活跃表
③ 上下文段：**只装本步新入批的序列**，形状 [B_new, S_step]，padding_bias 按每行真实长度填
④ 生成段：**只装本步处于 generation 的序列**，形状 [B_active, 1]
⑤ 采样：per-sequence 参数；采样 kernel 顺带写设备侧 finish flag
```

**为什么 prefill 与 decode 分两段**：requirement 的 Excluded 排除了"prefill / decode 混批"，
两者的输入形状不同（`[B,S]` 与 `[B,1]`），同一次引擎调用装不下。这也正是 TRT-LLM 的
context / generation 两段式结构——本项目已有的双引擎与它对齐。代价见 D10 与 §Risk。

**为什么"每次调用只装一种相"是硬前提**：若让上下文段按整批跑（含正在 generation 的行），那些行会被算出
无意义的 K/V，而写回按行号落进**它们自己的块**的 `0..S-1` —— 把真实 prompt K/V 覆盖掉，**静默算错**。
padding mask 拦不住：它只作用在图内的 attention scores，而 K/V 是投影输出；且即便输出为 0，写进有效位置
同样是破坏（问题不在写什么值，在写了不该写的位置）。

**写回必须带行映射**：`WritePrefillKV` 原来的行数取自"缓存已登记序列数"，在 S1/S2 里恒等于引擎的 B；
活跃批下上下文段的 `B_new` 小于活跃序列数，因此调用方要显式给出"引擎第 i 行 → 缓存第 `rows[i]` 行"的映射。

**与 TRT-LLM 的关系**：这种"分相 + 活跃批"对应它的 IFB（context / generation 两段式）；不混批（chunked
prefill）仍在 Excluded。**注**：这条对应关系是我的理解，未与 TRT-LLM 源码/文档核对过（2026-10-04），
落地前作为待核实项；但"整批跑会覆盖 running 行的 K/V"是代码级可推的结论，不依赖它。packed 打包另立 S4（见 D13）。

**前提是分路径的（2026-10-04 补）**：上面"每次引擎调用只装一种相"是 **S3 padding 路径**的前提，
理由是"context 段按整批跑 + 按行号写回"会覆盖 running 行自己的 prompt K/V。
S4 的 packed 路径**反过来**把两相装进同一次调用（见本节的 S4 小节与 D14）：那里的 context 段
只含新入批的 token，D12 的覆盖隐患不成立，因此这条前提在 S4 路径下不要求成立。

### S4 混合批（packed 输入 + 分段 attention 分派）

**路线（作者 2026-10-04 口径）**：每一步装**一个** packed 张量 —— context 段的全部 token 在前、
generation 段的 1 token/行在后、无填充；attention 在**同一次调用内按段分派**（context 走 varlen
自注意力，generation 走分页注意力 + 当前 token 自包含）。**顺序约束是硬的**：context 段必须排在
generation 段之前，kernel 才能用一个运行时边界标量切开两条路径。

**与 S3 的差别**：S3 每步最多两次调用（两段式），S4 每步一次（两相混装）；调度**策略**（准入/退出/
压实/D9 预算）不分叉，分叉只在输入打包、注意力实现、写回源寻址三处。

接口级细节（输入契约、段边界张量、packed 与缓存行序的映射、写回与采样、profile、开关与用例）
见 `p5_s4_interface_spec.md`。

**段边界与下标纪律（2026-10-04 作者补充）**：`cu_seqlens` **每段一份、下标从 0 起**，配合**一个段边界标量**
（context 段的序列数）；generation 段那份是退化的（每序列 1 个 token），由标量与行号推出。
**两段的下标禁止混用** —— 任何按行/按序列的输入都按 packed 行序排列（context 行在前），
段内下标与 packed 行下标之间只差一个 `B_ctx`，但这个差必须显式写出来。

## Resource Lifecycle

| 资源 | 何时创建 | 何时释放 |
|---|---|---|
| 分页 cache 与块池 | 运行时构造期（一次） | 析构 |
| 块表 / 长度 / 采样参数设备缓冲 | **构造期按最大批预分配**（S2 的核心改动） | 析构 |
| K/V / logits / token 缓冲 | 首次调用按 `(max_batch, max_seq)` 备好，容量不足才扩 | 析构 |
| 每序列的块 | 请求进入时（**先检查空闲量，避免异常穿透**，D9） | 请求结束（正常 / 出错都要归还） |
| 每序列的结果缓冲 | 随批容量一起备好 | 析构 |
| CUDA stream | 沿用默认流，本轮不引入多流 | — |

要点：**解码循环内不分配、不同步**；块归还必须与"从批内移除"同一步完成，否则块池泄漏。

## Performance Consideration

- **compute**：decode 每步的算子规模随批线性增长；权重读取次数不变 → 批量提高的是
  "每字节显存换来的 token 数"。这是本 feature 的主要收益来源。
- **memory**：真机 GPT-2 下两类占用要分开算。
  - **KV 池**：`num_blocks × block_size × kv_heads × head_size × 元素字节 × 2` × 层数；
    64 块、FP32、12 层时约 72 MiB。**它不是瓶颈。**
  - **prefill logits**：`B × S × V × 元素字节`；`S = 512`、FP32 时 B=1 约 98 MiB、B=4 约 393 MiB。
    **这才是随批增长的主要项**，批上限要按它先估。
  - 另有每序列 K/V 输出缓冲（`[B, kv_heads, S, D]`）与其他工作区，P4 一并量。
- **communication**：无（单卡、单流）。
- **测量口径**：沿用项目既有协议——同 session、同二进制 A/B、逐轮交替、报中位数与四分位，
  并先给出判别下限（`docs/PROGRESS.md` §5.13b，本机约 ±400~600 µs）。
  批量收益远大于该下限才值得写结论。**读数必须在关闭常驻诊断的口径下取得（D7）。**

## 验证策略

本节承载"验证型"判据的落点（不属于模块设计，但必须有可执行的判据与责任人，否则无法收口）。

| 判据 | 怎么验 | 在哪验 | 环境 |
|---|---|---|---|
| AC1 数值一致性 | 同一组请求分别走"批量"与"逐条单跑"，逐 token 逐位比对；不一致即失败（**贪心与随机采样各一条**：`BatchEqualsSequential` / `BatchEqualsSequentialWithTopP`） | P6 用例（runner 级对拍） | 真机 |
| AC2 长度不齐 | 批内各序列 `context_len` 不同时，逐行核对位置索引与语境长度 | P6 用例 | 真机 |
| AC3 资源回收 | 连续跑 N 轮后断言空闲块数等于初始值（引用相等） | P6 用例（host 断言） | 真机 |
| AC4 不回归 | 现有沙箱用例全绿；真机既有用例（FP32 端到端、K/V 缓存、插件）不出现新红 | P6 | 沙箱 + 真机 |
| AC5 单序列语义不变 | batch = 1 与改动前的基线逐 token 比对 | P6 用例 | 真机 |
| AC6 性能可复现 | 先声明判别下限 → 同 session、同二进制 A/B、逐轮交替，报中位数与四分位 | P4 baseline + P7 | 真机 |
| AC7 不浪费 | 同一批内"长度差很大"的两组输入对比：最长那条的长度不改变短序列的代价；判别口径与下限在 P4 声明 | P4 + P7（负载对照） | 真机 |
| AC8 两条路径各自成立且可回退 | 打包路径（默认）与"填充 + 掩码"路径**各自**满足 AC1；默认路径可切回且不改调用方接口（`FallbackSwitchKeepsResults`） | P6 用例 | 真机 |
| AC9 分块与不分块等价 | 同一条 prompt 在 `chunk_limit` 取 1 / 中间值 / ≥ prompt_len 三种切法下 token 逐位相同，且不影响同批其它序列（`ChunkedEqualsWholePrompt` 等，见 `p5_s5_interface_spec.md` §6）；**含位置编码的绝对位置断言** | P6 用例 | 真机 |

**环境约束**：沙箱无 GPU，本表除 AC4 的沙箱一半外都只能在真机执行；
若真机不可用，按 `phases/p4_baseline.md` 的 Dependency Missing 记为 N/A 并三处留痕，**不得**默认通过。

## 不变量（实现时必须显式成立，P6 需要断言覆盖）

1. **池与块表的契约**：引擎把 cache 第 0 维声明为 `ceil(n_positions / block_size)`，这个数同时被用作
   块表宽度；运行时的物理块池**必须 ≥ 块表宽度**。池更大之所以安全，是因为插件用运行期偏移在连续的
   更长缓冲里寻址、且 TRT 不校验输入缓冲容量——这条依据要落成注释与断言（D6）。
2. **profile 归属**：运行时给 prefill / decode 各用一个引擎，且各自**只有一个** profile。
   这条不成立时（例如拿到一个挂了双 profile 的单引擎）必须在构造期失败，而不是静默用错组（D8）。
3. **循环内零 H2D / D2H**：与现状一致，不因批量改动而放宽。
4. **批内行号一致**：块表第 i 行、`context_lens[i]`、采样参数第 i 项、结果第 i 段必须同源，
   不允许各自维护下标。
5. **元数据缓冲指针恒定**：S2 之后，跨请求的设备指针不随批大小变化；
   "每步重绑"从"正确性的依赖"降级为"保险措施"。
6. **分块的位置与记账（S5）**：chunk 内第 `i` 个 token 的位置 = `prompt_done + i`（绝对位置）；
   cache 侧的写入长度是**累加**（每步 `+= chunk 长度`），因此任何时刻的 `context_lens` 都等于
   该序列已写入的 token 数。这一条是分块下的"行号 / 长度同源"（不变量 4 的延伸，D16）。

## Trade-off

### D1 调度形态

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 静态批（S1 先做）** | 一批请求同进同出 | 实现小、读数干净；但短请求要等长请求 |
| B. 连续批（S3 再做） | 请求可在运行中进入 / 退出 | 服务化的形态；但引入"批内长度不齐 + 每步形状变化" |

**决策：先 A 后 B**，A 的数值一致性是 B 的前提。

### D2 批内 prompt 长度不等

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 只支持等长批（S1）** | 不同长度的请求分成不同的批 | **不需要改模型构建器**，最短路径到"批量正确" |
| B. 右填充 + padding mask（S3） | 一次吃下不等长批 | 要给 prefill 图加 mask 输入 → **改 I/O 契约**、绑定列表与检视工具都要动，并且**必须 bump `graph_version`**（否则引擎缓存会判定"未过期"而静默复用一张没有 mask 的旧图） |

**决策：S1 选 A，S3 做 B。** 判据 2 的完整形态落在 S3；若 S3 被裁掉，判据 2 必须相应下修并写明原因。

### D3 每序列采样参数

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 全批统一参数 | 一批共用一个 k / p / seed | 最省；但"同一批里有的贪心、有的 top-p"做不到 |
| **B. 每序列参数** | 设备端 per-batch 数组 | 采样器接口已经是 per-batch 指针，只需把缓冲从 1 元素扩成 B 元素 |

**决策：B**，成本几乎为零。

**实现收窄（2026-10-03）**：三种采样 kernel 各自是"整批一个分支"，因此 **S1 要求批内同一种策略**；
`top_k` / `top_p` / `seed` 均可逐行独立。这是**入口校验层面**的收窄，放宽校验即可扩回；
"同批混策略"才需要改采样器（见 `p5_s1_interface_spec.md` §8）。

**同日追加：随机流改为与批位置无关**。原实现 `Uniform01(seed, offset, 行号)` 把**行号**混进哈希，
同一个请求换个批位置就换一串 token——这会让固定 seed 的评估不可复现、线上问题无法复现。
现改为**每行带自己的 seed、行号不进随机流**（`SamplerArgs::seeds` + `RowUniform01`），于是
**AC1 的"逐 token 完全一致"对所有采样策略成立**，而不只是贪心。代价：`sampler_kernels.cu` 的
4 个 kernel 签名 + 4 个调用点 + 6 个 launch；`seeds` 为空时保留旧行为（只服务单行 / 兼容路径），
因此既有用例逐位不变。

### D4 结果收集

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 每序列独立 token 缓冲** | 每行一段，结束后整块 D2H | 结果切分简单，长度不齐也能对 |
| B. 一个大矩阵 `[B, max_new]` | 一次拷贝 | 形状整齐时省事，但先结束的序列要填充，语义更易错 |

**决策：A**，代价是 B 次拷贝，但都在循环之外一次性发生。

**实现澄清（2026-10-03 落码时确定）**：取 **`[max_new, B_max]` 单次分配、步优先布局**。
每行仍是一段独立结果，只是共用一次分配与一次 D2H。为什么不是"B 个独立缓冲"：采样器写的是
**连续 `[batch]`**，步优先让每步直接写、省掉"写暂存再逐步散播"。S1 是静态批、循环固定跑满
`max_new`，**不存在"先结束的序列要填充"**，所以方案 A 里担心的那个语义风险不成立。

### D5 与另外两条 feature 的写冲突

本 feature 与 `REQ-017-llm-int8-quant`（改权重精度与 cache 元素宽度）、`REQ-019-onnx-subgraph`
（让 ONNX 路径接入运行时）都要改运行时入口。**三者不能并行改同一个文件。**
**决策：本 feature 先做**（它定义批量契约）→ INT8（在契约上扩精度）→ ONNX 接入。
**已拍板（2026-10-03，作者）**：按上述推荐执行。`REQ-017` 与 `REQ-019` 均未开工，
本次串行顺序天然成立；本 feature 交付的批量契约就是后两者的扩展基线。

### D6 块池与引擎 cache 张量维度的关系

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 保留"池 ≥ 宽度"的隐含关系，显式化并断言** | 引擎声明不变；运行时校验并把安全依据写进注释 | 不动模型构建器、不用重建引擎；但安全依赖"插件用运行期偏移寻址"这一实现事实 |
| B. 让引擎把池大小当独立超参声明 | 契约最干净 | 要改模型构建器 + 超参 + 重建引擎，超出本 feature 范围 |

**决策：A**，必须同时落下断言与注释，否则这条又变回隐性依赖。

### D7 常驻诊断的去留

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 保持现状常驻 | 不改代码 | 每请求多一次同步 D2H 与逐层 K/V 扫描，真机 `S=512` 约 18 MiB/请求，且随批线性增长 → 污染吞吐读数 |
| **B. 做成运行期开关，默认关闭，P4/P7 一律关闭** | 保留诊断能力，同时让性能口径干净 | 数值问题定位时需要显式打开 |
| C. 直接删除 | 最干净 | 丢掉已经救过场的能力（FP16 NaN 定位靠它），不划算 |

**决策：B**。并加一条流程要求：P4 的 baseline 与 P7 的对比读数**必须在关闭诊断的口径下**取得；
两版开关状态不一致时读数作废。
**已拍板（2026-10-03，作者）**：开关**默认关闭**。

### D8 profile 归属校验

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 不校验，沿用现状 | 零成本 | 拿到双 profile 引擎时静默用错组，表现为"形状设了却不生效"这类难查现象 |
| **B. 构造期校验两个引擎各自的 profile 数** | 构造期一次检查，失败即拒绝启动 | 与现有精度校验同风格；把一类隐性错配变成启动期失败 |

**决策：B**。

### D9 块分配失败的语义

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 依赖异常，在调用方 catch | 改动最小 | 分配器对 OOM / 非法 id / 重复释放一律 `throw`；调度循环里异常穿透会带走整个批次 |
| **B. 调度入口"先检查后分配"** | 进入批次前先算所需块数与空闲量，不足则该请求留在队列，不进入可能抛异常的路径 | 把失败变成可预期的控制流；不需要改分配器 |

**决策：B**。分配器保持不变；"检查"与"分配"之间不能有其他会改块池的操作。

### D10 S3 的两步式执行代价

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 两步式（先 prefill 子批、再全体 decode）** | 与 Excluded 的"prefill/decode 混批"边界一致 | 每一步最多两次引擎调用；若每步都有新请求进入，"摊薄"收益会被 prefill 调用吃掉一部分 |
| B. 延迟准入：等待队列只在批内**全部结束**时才放行 | 一步只有一次 decode 调用 | 退化成"批间连续"，失去连续批的意义 |

**决策：A**，但把代价写进 §Risk，并在 P4 用"每步都进新请求"与"整批同进同出"两种负载分别测。
**已拍板（2026-10-03，作者）：S3 做。** P4 的两种负载对照仍然要跑，但它只用于**把收益结论写准**，
不再作为 S3 是否交付的门槛——S3 的契约与可扩展性价值独立成立。

### D11 与后续 feature 的接口面（可维护性 / 可扩展性）

`REQ-017`（权重量化与 cache 元素宽度）与 `REQ-019`（ONNX 路径接入运行时）都未开工，
它们将来要在本 feature 定义的批量契约上做加法。为让那两步不必返工，本 feature 明确三件事：

| 接口面 | 本 feature 的约束 | 后续怎么扩 |
|---|---|---|
| 批内行号 | 块表第 i 行 / `context_lens[i]` / 采样参数第 i 项 / 结果第 i 段同源（不变量 4） | 新增 per-sequence 张量时按同一规则挂行号，不另立下标 |
| 元素宽度 | 缓冲按**引擎实际声明的精度**分配，cache 的"源→目标"转换在写入内核里做 | 改成 INT8 只动"声明精度 → 缓冲尺寸"这一处映射，批量逻辑不变 |
| 图与引擎 | 改图必须 bump `graph_version`；引擎指纹不含建图代码 | ONNX 路径接入时复用同一指纹与同一 profile 校验入口 |

### D12 调度期的批组成（活跃批 vs 固定槽位）

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 活跃批（采用）** | 批 = 本步活跃的序列；退出压实行号；逐行缓冲每步重建；**每次引擎调用只装一种相** | **不会算到不该算的行**（不覆盖 running 序列的 K/V）、无空槽浪费；代价是行号每步可能变（靠每步重建 + 不变量 4 的校验兜），且写回需要**行映射** |
| B. 固定槽位 + 全批定长 | 批恒为 `max_batch` 行；空槽 / 非参与行用填充顶位 | 行号终身稳定；但**上下文段会把正在 generation 的行也算一遍**，写回时覆盖它们的 prompt K/V（静默算错）——除非再给写回加行过滤、或给非参与行挂 scratch 块 |

**决策：A（2026-10-04，作者）—— 推翻 2026-10-03 那次（当时选的是 B）。** 推翻的理由：
① 当时选 B 的主要依据是"贴近 TRT-LLM 的 IFB（固定 `max_batch` + 空槽）"，**这条依据不成立**：
   TRT-LLM 的运行时按活跃序列建批（`active_batch_size`）；
② 更要紧的是 B 会引入实测可推的缺陷——上下文段按整批跑会把正在 generation 的行也算一遍，
   写回时覆盖它们**自己**的 prompt K/V（见本节末的说明），属于静默算错；
③ 活跃批同时消掉"无用的行"与"不该碰的行"，且**无空槽算力浪费**。

**附带修正**：S2 交付的压实逻辑**不是备用路径**，而是 S3 的**必经路径**（活跃批靠它把退出后的行号压实）。

### D13 两条路径的定位（S3 padding / S4 packed）—— 两者都是本 feature 的里程碑

| 路径 | 实现 | 定位 |
|---|---|---|
| **S3：padding + mask** | 右填充 `[B,S]` + 加性 `padding_bias`（显式子图注意力） | S3 落地后作为**生产路径**交付；S4 落地后转为**对照 / 回退路径** |
| **S4：packed + 选择性批处理** | 一维 `[T]` + `cu_seqlens`；**同一次调用内**装两相、attention 按段分派（context → varlen；generation → 分页 + 自包含当前 token） | **作者指定为默认路径**（2026-10-04）；**是本 feature 的 S4 里程碑**，不另立条目 |

**"选择性批处理"的准确定义（2026-10-04 作者补充，本设计的依据）**：不是"只有 prefill 走 packed"，
而是**两相共享同一个 packed 输入张量、attention 计算各走各的 kernel**。硬约束：**所有 context token
必须排在 generation token 之前**（例：S0 context、S1 generation、S2 context 的顺序必须是 S0 → S2 → S1），
这样 kernel 才能按位置一次切分两条路径。
这条口径的后果：**每步一次引擎调用**（而非 S3 的两段式），且 packed 行序与缓存行序**不同源**，
必须靠显式映射数组对应起来 —— 不变量 4 的口径随之改写（见 `p5_s4_interface_spec.md` §4）。

**两条路径共用的契约**：prefill 的产出 =「每序列的 prompt K/V」+「每序列末位 logits」；下游（写回 / 采样 / 调度）不分叉。
**两条路径的数值不保证逐位相同**（kernel 不同、浮点累加顺序不同）：AC1 必须在**每条路径内部**成立（批跑 vs 单跑）；
跨路径差异按 `AGENTS.md` §7 写清来源与容差出处。
**"默认用 S4"是设计决定，不是实测结论**：S4 落地后仍应做 P4/P7 的 A/B，给这个默认值一个带判别下限的依据；
若实测 packed 在本机反而更慢，默认值的取舍回到作者（采样器的 fast / legacy 就是这么处理的）。

### D14 S4 的引擎形态（一个混合引擎 vs 两个 packed 引擎）

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 一个混合引擎（采用，作者 2026-10-04 确认）** | 每步一次调用，context-only / generation-only / 混合三种步共用同一张图（`context_seq_count == 0` 即纯 generation） | 直接消掉 D10 说的"每步第二次调用"的收益损失；纯 generation 步与今天的 decode 用法等价；代价是**插件内部两份 attention 实现**，且缓存输入在纯 context 步也要绑定 |
| B. 两个引擎各自 packed | context 引擎不带 cache 输入、generation 引擎带 | 两段仍不在同一次调用里 → "选择性批处理"退化成"各自 packed"，与作者口径不符 |

**决策：A（作者 2026-10-04 确认）。** 段边界是**运行时标量**，TRT 图里没有数据相关的分支，
所以"两段分派"只能落在插件内部：一个 attention 插件读 `context_seq_count`，对边界之前的 token 走
varlen 自注意力、之后的走分页注意力（含当前 token）——**这就是 A1，作者已选定**；
"两个插件 + 图内按运行时标量切片再合并（Select/Concat）"（A2）作为对照方案保留。

**它不替代 `PagedAttentionPlugin`**：S3 回退路径继续用它（该插件只支持 decode：query seq_len 必须为 1）。

**与图的关系**：S4 需要一张**新图**（输入是 packed 张量 + cache 输入，输出是 packed K/V + logits），
`graph_version` 必须 bump，并与 S3 的两套图共存（AC8 的可回退）。

### D15 S5 里程碑：chunked prefill（长 prompt 跨步分批）

| 方案 | 说明 | 取舍 |
|---|---|---|
| A. 不做（原 Excluded 口径） | 长 prompt 必须在一步内送完 | 一条长 prompt 独占一步 → decode 侧尾延迟抖动；主流框架（vLLM 的 chunked prefill、TRT-LLM 的 context chunking）都提供这条能力 |
| **B. 立项为 S5（采用，作者 2026-10-04 改判）** | 一条序列的 prompt 跨多步分批推进；每步的 chunk 与**之前已写进缓存**的 K/V 一起参与注意力 | 压住尾延迟抖动；代价：多出第三种计算模式 + 每序列 prompt 进度状态 + 与 D9 预算 / 退出判据联动 |

**决策：B。** requirement 的 Included 与 AC 已同步（Included 第 8 条、AC9"分块与不分块逐位相同"）。

**依赖与边界**：

- **依赖 S4**：S5 不需要 S4 才能开工，但两者共用同一套 packed 输入契约；S4 先做（打包 + 段分派）能让 S5
  只增加"context 段的 chunk 可以 < prompt 长度"与"读缓存"两件事。
- **仍然排除**：抢占 / 换出；调度层的混批 / 分块优先级策略（例如按分块重排准入）。
- **S5 自己的设计要解的五件事**（2026-10-04 第二遍复评后由四件补为五件，第 ⑤ 条是那轮查出的遗漏）：
  ① context 段读缓存（第三种计算模式：query 数 > 1 且 K/V 来自缓存）；
  ② 活跃表的 prompt 进度字段与"何时算 prefill 完成"；③ 与 D9 块预算、退出判据的联动；
  ④ 判据：AC9 + 分块不得影响同批其它序列；
  ⑤ **位置与采样行集**：chunk 的 token 必须用**绝对位置**（`prompt_done + i`）；且"本步完成 prefill 的行"
  在活跃表里**不保证连续**，采样因此需要显式行列表（两条都见 D16）。

### D16 S5 的实现形态（分块语义 + chunked 分页因果 context kernel）

> **2026-10-04 第二遍复评后重写**。原版把形态写成"统一成分页因果"，但核对代码后确认：
> generation 段的 split-K 是 **decode 专用**（每行 1 个 query、workspace 布局写死），
> 而 context 段的现有 kernel 根本不接 `block_tables` / `context_lens`。原版的表述在实现层面
> 是空壳，且漏了位置编码与采样行集两条会**静默算错**的落点。以下为定稿形态。

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. 新写 chunked 分页因果 context kernel（采用，第二遍复评定稿）** | context 段保持**一个 query 位置一个 block** 的网格（与 S4 的 varlen kernel 同形），K/V 读法改成"分页缓存 `[0, prompt_done)` ++ 本 chunk 自包含"；首 chunk 缓存长度为 0 → **自动退化**成 S4 的 varlen 行为（同一段代码） | 一条实现、一个判据面；**workspace 需求不变**（score 仍在 shared），"复用已构建引擎"的条件可证；代价是首块也吃分页寻址开销 |
| B. 把 generation 段的 split-K 扩到"一行多个 query" | 沿用 `PagedAttentionSplitKernel` + 其 workspace 布局 | 首块可能更快；但 split-K 是 decode 专用（workspace 布局 `[split][batch][head][m,l,acc]` 写死、`has_current_token` 单 token），要重写布局与归并 → `getWorkspaceSize` 变 → 引擎必须重建且 `graph_version` 必须 bump |
| C. 双模式（首块 varlen / 后续分页） | 首块保持 S4 原路径，后续 chunk 走新的分页因果路径 | 首块更快；代价是同一语义两套实现、判据与回归面翻倍，与"少一种模式"的初衷相反 |

**决策：A。** 依据：现状核对显示 B 会动 workspace、C 会翻倍判据面，A 是唯一"新增语义但不动
已构建引擎的 I/O 与 workspace 契约"的方案。若 P4/P7 实测首块代价显著，再退回 C（届时判据不变）。

**S5 的实现形态（逐条定下）**：

- **kernel 形态**：新写 chunked context kernel（替换 `PackedContextAttentionKernel` 的 K/V 读法）。
  网格仍是 `(num_heads, B_ctx, max_seq_len)`，一个 block 负责 `(head, seq, 段内 query 位置 pos)`；
  `pos >= L_c` 的块立即返回（`L_c` 由 `cu_seqlens_ctx` 给出，是设备值）。
  K/V 分两段：① 分页缓存 `[0, prompt_done)`——**注意力时刻**的 `context_lens[seq]` 就是 `prompt_done`
  （runner 的顺序是"先上传元数据、再跑引擎、最后写回"，见 `llm_runner.cpp` 的 ②③④），
  ② 本 chunk 自包含的 `[0, pos]`；因果边界 = `prompt_done + pos`。输出只写这 `L_c` 个 query 位置。
- **key 数上界与"支持集合"**：per-query 的 key 数 = `prompt_done + pos + 1 <= prompt_len <= n_positions`，
  而 `configurePlugin` 已要求 `n_positions <= kPackedAttentionMaxContextSeqLen = 1024`
  （`gpt2_model_builder.cpp` 把 `cfg.n_positions` 作为插件的 `max_seq_len` 传入）。score 数组按该
  编译期常量分配即可。因此原版"`chunk_limit` 必须落在 fused kernel 支持的常量集合内"**改写为可执行的两条**：
  `n_positions <= 1024`（建图期已有校验）＋ `prompt_len <= n_positions`（入口拒绝，见兜底纪律）。
  `chunk_limit` 本身**不是**任何 kernel 的模板常量，它是数据，不需要也不应该进"支持集合"。
- **禁止静默跳过**：现有 context kernel 在 `len > 1024` 时直接 `return`（**不写输出**）——S5 的新 kernel
  不得保留这种形态；越界一律在构造期 / 入口显式拒绝。
- **position_ids 用绝对位置**：context 行的第 `i` 个 chunk token 的位置 = `prompt_done + i`
  （首 chunk 退化为 `0..L_c-1`，与 S4 一致）。依据：模型是**绝对位置查表**
  （`gpt2_model_builder.cpp` 的 `addGather(wpe, position_ids)`）；沿用现状的"段内位置从 0 起"
  会让第二块起查错位置表，且不报错（静默错，AC9 必然变红）。
- **采样行集 = 显式行列表 + 紧凑暂存**：本步参与采样的行 = 生成段前缀 ∪ 本步完成 prefill 的 chunk 行。
  完成的行**不保证**连续（长 prompt 分块中、新准入的短 prompt 本步完成时会出现"洞"），而采样器
  只吃连续 `[count]`；因此 runner 用一张显式行列表把（末位 logits / per-row 采样参数 / EOS 标记 /
  输出 token）在紧凑槽位上聚集与散开 —— **不动 `SampleBatch` 签名，也不动行号纪律**（不变量 4）。
- **写回位置与记账**：写回从 `prompt_done` 起——kernel 用**写回时刻**的 `context_lens[row] + t`
  （该值同样仍是本步之前的已写入长度，不需要新输入）；host 侧记账从"赋值"改成"累加"
  （`context_lens_host_[row] += row_lengths[i]`、`Sequence::length` 同）；预留量校验改用**累计长度**。
- **chunk 期间不出 token**：序列在 prompt 全部 prefill 完成前不参与采样，`max_new` 从那时开始计时，
  否则"分块"会改变可见的生成语义。
- **切法是确定性的**：`chunk_limit` = 引擎 profile 的**单序列上限**（`input_ids` / `position_ids`
  第 1 维 `.max`），**不暴露给调用方**；非末块对齐（恒为 `chunk_limit`）、末块按剩余的实际长度。
  不做"按队列长度自适应"那类调度层策略（那一条仍在 Excluded 里）—— AC1/AC9 的逐位对拍依赖确定性。
- **`chunk_limit` 的来源（2026-10-04 作者授权折入）**：`LLMRunner::Config` 里**没有**
  `max_prefill_seq_len`，`Engine` 类也不暴露 profile 查询，所以"由 profile 推导"必须先补一个
  **`Engine` 的只读 profile 查询接口**（查 `getProfileShape`），**不新增 `LLMRunner::Config` 字段** ——
  依据是项目既有的"按对方查询、不按配置假定"（workspace 版见 `paged_attention_split.hpp` 与
  `PROGRESS.md` §2.15）。查询失败必须**构造期报错**，不许退回猜测的默认值。
- **图与引擎（`graph_version`）**：方案 A 让 I/O 契约与 `getWorkspaceSize` **都不变**，所以存在
  "复用旧引擎"的理论可能；但 `engine_cache.hpp` 的规则是"**任何改动建图 / 精度 / 插件行为的代码变更
  都要 +1**"（先例：1 → 2 正是 `PagedAttentionPlugin::getWorkspaceSize` 从 0 变正数），
  而 S5 改的正是插件对同一绑定的计算语义。**作者 2026-10-04 复核确认：
  `kPackedPrefillGraphVersion` 4 → 5**（代价是真机首次重建 packed 引擎，分钟级；不再保留"沿用 4 +
  写豁免条件"的分支）。
- **兜底纪律（作者 2026-10-04 定；第二遍复评补全检查对象与适用范围）**：分界是"**能否在构造期 /
  入口判定**"——配置 / 形状类（`n_positions > 1024`、`prompt_len > n_positions`、推导不出
  `chunk_limit`、profile 查询失败）**显式拒绝**，错误信息带上实际值与上界；运行期资源类
  （TRT 未提供 workspace 时既有插件退单趟 + WARN）保留既有降级，但**S5 新增路径不得引入新的这类
  静默降级**。末块变短**不算**这种情形（同一个 kernel 按 `cu_seqlens` 处理）。

## Requirement Coverage

| 需求条目 | 设计落点（章节） | 交付里程碑 | 验证 Phase |
|---|---|---|---|
| Included 1：批量 > 1 的 prefill 与 decode 路径 | §Architecture + §Runtime Flow（S1 静态批） | S1 | P6 |
| Included 2：每序列独立的长度记账、位置编码推进与结果收集 | §Data Structure + §Runtime Flow（S1） | S1 | P6 |
| Included 3：每序列独立的采样参数 | §Data Structure（`top_k[B]` / `top_p[B]`）+ §Trade-off D3 | S1 | P6 |
| Included 4：每序列独立的 K/V 分配与回收，跨请求不泄漏 | §Resource Lifecycle + §Runtime Flow（S2） | S2 | P6 |
| Included 5：请求级调度（静态批 → 最小连续批） | §Runtime Flow（S3 槽位模型 + 五步流程）+ D10 + D12 | S1（静态批）+ S3（连续批） | P6 |
| Included 6：批量运行 == 逐条单独运行 | §验证策略（AC1 行） | S1 | P6 |
| AC1 数值一致性 | §验证策略（AC1 行） | S1 | P6 |
| AC2 长度不齐 | §Runtime Flow（S3，padding mask）+ §验证策略（AC2 行） | S3 | P6 |
| AC3 资源回收 | §Resource Lifecycle + §Runtime Flow（S2） | S2 | P6 |
| AC4 不回归 | §验证策略（AC4 行） | S1/S2/S3 每步 | P6 |
| AC5 单序列语义不变 | §验证策略（AC5 行） | S1 | P6 |
| AC6 性能可复现且不自证 | §验证策略（AC6 行）+ §Performance Consideration（测量口径） | S1/S3 | P4 + P7 |
| Included 7：不等长 prefill 按真实长度计费 + 两条路径（打包为默认） | §Runtime Flow（S3 两段式 / S4 packed）+ D10 + D13 | S3 + S4 | P4 + P6 |
| Included 8：长 prompt 的分块推进（S5） | §Runtime Flow（S5 分块模型）+ D15 + D16 | S5 | P6 |
| AC7 不浪费 | §验证策略（AC7 行）+ D10（负载对照口径） | S3 + S4 | P4 + P7 |
| AC8 两条路径各自成立且可回退 | §Runtime Flow（S3 / S4 两条路径）+ D13 + D14 | S3 + S4 | P6 |
| AC9 分块与不分块等价 | §Runtime Flow（S5）+ D16 + `p5_s5_interface_spec.md` §6 | S5 | P6 |

## Risk

| 风险 | 影响 | 缓解 |
|---|---|---|
| 批内长度不齐导致注意力算错 | 结果静默错 | S1 限定等长；补"批量 == 逐条单跑"对拍用例 |
| 块表宽度与批大小混淆 | 越界写 / 读错行 | 块表宽度由引擎契约固定，批大小只改行数；host 侧镜像补 0 |
| 池被放大到超过引擎声明的 cache 维 | 越界读 / 静默错 | D6 的断言 + §不变量 1 的注释；P4 先量池的实际需求 |
| 元数据缓冲重分配后旧地址仍被使用 | 静默读到旧地址 | S2 按最大批预分配；P6 加"多次调用后指针不变"的断言 |
| **prefill logits 显存被低估** | 批上限设得过大 → 真机 OOM | P4 把 logits 缓冲单列为测量项；批上限写成实测值而不是推算值 |
| **S3 的两步式吃掉批量收益** | 白做 | P4 用"每步新增请求"与"整批同进同出"两种负载对照；收益不过判别下限就不做 S3（D10） |
| 常驻诊断污染读数 | 性能结论失真 | D7：P4/P7 一律在关闭诊断的口径下取数 |
| 改图未 bump `graph_version` | 静默复用旧引擎，行为与代码不符 | D2=B 落地时把 bump 写进同一个提交 |
| ~~AC1 在随机采样下不成立~~ **已解决（2026-10-03）** | 判据 1 曾被误读成只对贪心成立 | 根因是随机流把**行号**混进了哈希；已改为 per-batch `seeds`、行号不进随机流 → **AC1 对所有采样策略成立**。回归用例 `BatchEqualsSequentialWithTopP` 锁死该性质 |
