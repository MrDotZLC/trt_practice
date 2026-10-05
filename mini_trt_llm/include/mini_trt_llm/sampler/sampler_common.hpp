#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace mini_trt_llm {

// 采样器公共参数（设备侧 API）。
//
// **本结构体的所有指针字段都必须是设备可读地址**（kernel 直接解引用）。两个"显然"的：
// `logits` 来自引擎输出、`token_ids` 是采样结果——Decode 自回归循环必须全程驻留显存，
// 结果直接写回 device，不允许经过 host（AGENTS.md §3.A.3）。
// 其余几个同样如此、但**容易被当成 host 数组**：`seeds` / `offsets` / `top_k` / `top_p` /
// `eos_hit` 由 `LLMRunner` 上传到常驻设备缓冲（`UploadRowParams*` / 构造期的 `d_*`）后才交给
// 采样器。传 host 数组的表现是真机 `illegal access`，或"读到垃圾随机流"这种不报错的错
// （`TS-053` 的对账清单：这条原先没写明，2026-10-05 补上）。
struct SamplerArgs {
    const void* logits = nullptr;  // [batch_size, vocab_size]，FP16 / FP32
    int32_t* token_ids = nullptr;  // [batch_size]，输出
    int32_t batch_size = 0;
    int32_t vocab_size = 0;
    bool is_half = false;
    // 随机源：host 传 seed + offset，device 侧用 Philox 确定性生成（Q8）。
    // 这样测试可用固定 seed 复现，同时避免 host-device 频繁同步。
    uint64_t seed = 0;
    uint64_t offset = 0;
    // **per-batch 的 seed（[batch_size]，调用方持有、设备缓冲）**。
    // 非空时 kernel 走 `Uniform01(seeds[row], offset, 0)`——**行号不进随机流**，于是同一请求
    // 无论落在批内哪一行、批次怎么组成，token 序列都逐位相同（AC1 对**所有**采样策略成立的前提，
    // 也是工业界的口径：随机性只由 (请求 seed, 步数) 决定）。
    // 为空时退化为旧的 `Uniform01(seed, offset, row)`，只服务单行 / 兼容路径。
    const uint64_t* seeds = nullptr;
    // **per-batch 的随机步号（[batch_size]，调用方持有、设备缓冲）**：非空时第 row 行用 `offsets[row]`，
    // 为空则整批共用标量 `offset`。
    // 为什么必须逐行：连续批调度下，同一次引擎调用里各行的"已生成计数"不同（有的在采第 3 个
    // token、有的才第 0 个）。AC1 要求同一请求无论何时入批都逐位相同 → 随机流只能由
    // `(请求 seed, 该请求自己的步号)` 决定，批组成与行号都不许进哈希
    // （p5_s3_interface_spec.md §5；`BatchEqualsSequentialUnderScheduling` 的判据就是它）。
    const uint64_t* offsets = nullptr;
    // 可选：设备缓冲里"该行采到 EOS"的输出（`[batch_size]`，1 = 命中）。nullptr 表示不需要。
    // 有了它，调度器每步只回读 batch_size 字节就能决定谁退出，不必把 token 拷回主机自己比 EOS
    // （见 design.md D12 与 p5_s3_interface_spec.md §4）。
    int8_t* eos_hit = nullptr;
    // <0 → 不判 EOS（此时整批 eos_hit 都写 0）。
    int32_t eos_token_id = -1;
};

// Top-K / Top-P 的 k 与 p 都是 per-batch tensor（Q7），以支持连续批处理中
// 每个请求独立配置采样参数。
struct TopKSamplerArgs : SamplerArgs {
    const int32_t* top_k = nullptr;  // [batch_size]，设备缓冲（见 SamplerArgs 开头的可读侧说明）
};

struct TopPSamplerArgs : SamplerArgs {
    const float* top_p = nullptr;  // [batch_size]，设备缓冲（同上）
};

// Greedy：取 logits 最大值的下标，并列时取最小下标（与 torch.argmax 一致）。
cudaError_t LaunchGreedySampler(const SamplerArgs& args, cudaStream_t stream);

// Top-K / Top-P 内部按行降序排序，需要调用方提供 workspace（避免在 Decode 循环里
// 反复 cudaMalloc）。用 *WorkspaceBytes() 计算所需大小。
size_t TopKSamplerWorkspaceBytes(int32_t batch_size, int32_t vocab_size);
cudaError_t LaunchTopKSampler(const TopKSamplerArgs& args, cudaStream_t stream,
                              void* workspace, size_t workspace_bytes);

// 快速路径支持的最大 top-k。理由：快速路径每行只用一个 warp（32 线程），
// 每线程在线程本地持有"自己那份有序 top-k"；本地候选上限就是它能支持的最大 k。
constexpr int32_t kTopKFastMaxK = 64;

// Top-K 的**快速路径**：不做整行排序，而是"每行一个 warp：每线程本地 top-k + k 轮 warp 归并"。
//
// **调用方契约**：必须保证每个 batch 的 `top_k ≤ kTopKFastMaxK`。越界行会写哨兵 `-1`
// （而不是静默给错答案），以便用例/调用方立刻发现契约被破坏；生产路径（`LLMRunner`）
// 在 host 侧就知道自己的 `top_k`，因此就近判断、不需要任何 D2H 同步（符合 §3.A.3）。
//
// **语义等价**：元素集合、排序顺序、并列取小下标、以及 Philox 随机数消费方式都与
// `LaunchTopKSampler` 一致 → 同一 `(seed, offset)` 下两者应给出**逐 token 相同**的结果。
// 这条由 `SamplerKernelTest.TopKFastMatchesLegacyTokens` 锁住（不一致就是缺陷）。
//
// 为什么值得单独一条路径：基线实测（`docs/future_iterations_development_plan.md` 的
// `[OI-SAMPLER-KERNEL-ACCEPTANCE]`，P9_2-0 表）里 top-k(k=64) 在 50257 词表上是 greedy 的 12.8 倍、
// 1.2 ms（128K 词表），而整行降序排序是主要成本。
cudaError_t LaunchTopKSamplerFast(const TopKSamplerArgs& args, cudaStream_t stream);

size_t TopPSamplerWorkspaceBytes(int32_t batch_size, int32_t vocab_size);

// Top-P 采样：CUB 分段排序 + **行内并行**的采样 kernel（每行一个 block，P9_2-5）。
//
// 与旧的逐行串行实现（`LaunchTopPSamplerLegacy`）相比，**唯一的语义差异是浮点累加顺序**：
// 旧版逐元素累加 `exp/total` 再与 `p` 比，新版累加 `exp` 再与 `p * Σexp` 比。两者在数学上
// 等价，但极端并列 / 恰好落在阈值边界处 cutoff 可能差一格——这是**允许的差异**，
// 不是回归（登记在 `future_iterations_test_plan.md` §9.4）。随机数消费
// （`Uniform01(seed, offset, row)`）、`>=` 比较、稳定项取 top-1、前缀内重新归一化均保持一致。
//
// workspace 需求与旧版相同（`TopPSamplerWorkspaceBytes` 未变）：排序仍是 CUB，采样 kernel
// 不需要额外显存（见 `AGENTS.md` §3.B.3「enqueue 内零分配」）。
cudaError_t LaunchTopPSampler(const TopPSamplerArgs& args, cudaStream_t stream,
                              void* workspace, size_t workspace_bytes);

// Top-P 的 legacy 逐行串行实现（一行一个线程）。**只用于对照与回归**：A/B 打印、复现
// §10.5 的基线数字、以及"新实现是否真的改变了结果"的观测；不接生产路径。
cudaError_t LaunchTopPSamplerLegacy(const TopPSamplerArgs& args, cudaStream_t stream,
                                    void* workspace, size_t workspace_bytes);

// P9_2-5b（子块级定位）的 **A/B 对照入口**：同一条 CUB 排序 + 相同的第一趟，但采样 kernel
// 用**两级**定位（块和 → 元素，即改动前的形态）。**只用于性能对照，不接生产路径。**
//
// 为什么需要一个专门的入口：这一改动的效果此前一直判不了——分段口径没有判别力
// （`TROUBLESHOOTING.md` + TS-037），配对口径又跨 session/跨协议（TS-038）。把两版编进同一个二进制、
// 在**同一轮里交替测量**，噪声对二者同向、差值可加，结论才干净。
cudaError_t LaunchTopPSamplerTwoLevel(const TopPSamplerArgs& args, cudaStream_t stream,
                                      void* workspace, size_t workspace_bytes);

}  // namespace mini_trt_llm
