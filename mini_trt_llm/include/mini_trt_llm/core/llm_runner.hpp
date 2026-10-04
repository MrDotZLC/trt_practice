#pragma once

#include "mini_trt_llm/core/engine.hpp"
#include "mini_trt_llm/kv_cache/paged_kv_cache.hpp"
#include "mini_trt_llm/tokenizer/base_tokenizer.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace mini_trt_llm {

// LLM 自回归生成 Runner：Prefill → Decode 循环 → 采样。
//
// 只做 token-id 级别的生成（收 token id、还 token id），因此**不依赖 tokenizer**——
// GPT-2 用 BPE，与已嵌入的 SentencePiece 不对齐，把 tokenize 混进来会让
// "数值对不对"和"分词对不对"两个问题纠缠在一起。tokenizer_ 目前仅在构造时校验，
// 文本级接口留待后续迭代。
class LLMRunner {
 public:
    // KV Cache 与引擎输入形状的几何参数。
    //
    // 必须与三者一致：建引擎时的 `hyper_params.block_size`、decode 引擎的
    // `block_tables` 宽度、以及 PagedAttention 插件的 `block_size` 属性。
    // Runner 自己推不出来（它看不到 config.json），所以要求调用方显式给出。
    struct Config {
        int32_t num_layers = 0;
        int32_t num_kv_heads = 0;
        int32_t head_size = 0;
        int32_t block_size = 0;
        // 每序列最多多少块，必须等于引擎侧 block_tables 的第二维
        // （`ceil(n_positions / block_size)`，由 GPT2Config::num_blocks() 推出）。
        int32_t max_blocks_per_seq = 0;
        // 物理块池大小（能同时容纳多少 token 的 K/V）。
        int32_t num_blocks = 0;
        // 位置表长度 n_positions（= config.json 的 `n_positions`，也 = 建图时给 PagedAttention /
        // PackedAttention 插件的 `max_seq_len`）。
        //
        // **为什么必须由调用方给**：它不是任何输入张量的形状 —— 引擎侧只留下
        // `ceil(n_positions / block_size)`（cache 的第 0 维 / block_tables 的第 1 维）这个**上界**，
        // 而 runner 看不到 config.json。与本结构体开头那句"Runner 自己推不出来，所以要求调用方
        // 显式给出"同一口径。
        //
        // **谁用它**：`prompt_len + max_new - 1 > max_positions` 时入口直接拒绝（见 RunScheduler
        // 的入口校验）。少了这道闸，位置编码的 `addGather(wpe, position_ids)` 会越界读 ——
        // TRT **不报错**，只会拿位置表以外的数据算出无意义的 logits（`docs/TROUBLESHOOTING.md`
        // 的 `TS-051`）。S5 的分块让"prompt_len 超过单步形状上界"成为正常路径，这条闸不可省。
        //
        // `0` = 未声明：**packed 模式下构造期直接拒绝**（"不许猜默认值"）；padding 两段式路径下
        // 不检查（那条路径的 S 由引擎 profile 兜住：越界时 `SetInputShape` 会显式失败）。
        int32_t max_positions = 0;
        // 单步每序列最多送多少 token（= 建图时的 `EngineBuilder::Config::max_prefill_seq_len`，
        // **两处必须给同一个值**）。S5 的 `chunk_limit` 就是它 —— **唯一来源**（不再从 profile
        // 反推；见 `p5_s5_interface_spec.md` §2 的 2026-10-05 修订）。
        //
        // **只在 packed 模式下有语义**：非 packed 路径（`kPaddedTwoPhase`）**必须留 0** —— runner
        // 不读也不校验它，填非 0 值没有任何效果（写明是为了避免"填了值却没效果"的困惑）。
        // packed 模式下**必填**，并过五条交叉校验（构造期，任一条不过即拒绝并打印实际值与上界）：
        // ① ≥ 1；② ≤ `max_positions`；③ `L × block_tables.dim0.max ≤ input_ids.dim1.max`
        // （保证每步 Σ 每行 chunk 不越出引擎 profile 的 T 上界）；④ ≤ 插件上限
        // `kPackedAttentionMaxContextSeqLen`；⑤ profile 查询必须成功（引擎是上界的裁决者）。
        // 完整口径（含"单一来源 = 下游单一、声明侧两处"）见 spec §2。
        int32_t max_prefill_seq_len = 0;
        // 与引擎激活精度一致；不一致时 PagedAttention 会按错误宽度读 cache。
        bool is_half = false;
        // 单次批量调用的最大序列数（S1：静态批）。
        // **暂定 2：待 P4 实测 prefill logits 显存后确定**（design.md §Performance Consideration）；
        // 调大它同时受引擎 profile 的 batch 上限约束。
        int32_t max_batch = 2;
        int32_t vocab_size = 0;
        // 常驻诊断（D7）：prefill 之后做一次同步 D2H 打印 logits 统计与逐层 K/V 扫描。
        // **默认关闭**：它会进吞吐/延迟读数（真机 S=512 约 18 MiB/请求），P4/P7 必须在关闭口径下取数。
        bool enable_diagnostics = false;
        // 生成到这个 id 就截断（-1 表示不截断）。
        // 注意截断发生在**循环之后**：循环内不能同步，所以做不到"一见 EOS 就停"。
        int32_t eos_token_id = -1;

        // **prefill 路径选择（S4，2026-10-04）**。
        //   kPaddedTwoPhase —— S3 的 padding + 两段式（**默认**，既有的两条引擎）；
        //   kPackedMixed    —— S4 的 packed 混合批：一个张量装两相、每步一次调用。
        // packed 模式下**只用 `prefill_engine_`**（调用方把那个 packed 引擎放在这个槽位，
        // `decode_engine_` 仍需非空以满足构造期校验，但不会被调用）—— 这样切换不改调用方签名（AC8）。
        // **默认值保持 padding**：翻转要等 P6 把两条路径都跑绿（未编译/未验证的路径不当默认）。
        enum class PrefillMode { kPaddedTwoPhase = 0, kPackedMixed = 1 };
        PrefillMode prefill_mode = PrefillMode::kPaddedTwoPhase;
    };

    struct GenerateOptions {
        int max_new_tokens = 20;
        // 只支持 1.0。其他值直接报错而不是静默忽略（D5）：
        // 静默忽略会让"调参无效"看起来像"模型就是这样"。
        float temperature = 1.0f;
        // 采样策略：top_p < 1 走 Top-P；否则 top_k == 1 走 Greedy，其余走 Top-K。
        int top_k = 1;
        float top_p = 1.0f;
        uint64_t seed = 42;
    };

    // 一条批量请求。
    // S1 的批内约束（入口逐条校验，不满足即整批拒绝）：prompt 等长、同一种采样策略、同一个 seed；
    // top_k / top_p 可以逐序列不同（D3 的实质部分）。
    struct GenerateRequest {
        std::vector<int64_t> input_ids;
        GenerateOptions options;
        // <0 → 由 runner 分配为批内下标；>=0 → 调用方指定，须批内唯一。
        int32_t seq_id = -1;
    };

    struct GenerateResult {
        int32_t seq_id = -1;
        // 新生成的 token（不含 prompt）。ok == false 时该值无意义。
        std::vector<int64_t> tokens;
        bool ok = false;
    };

    LLMRunner(const Config& config, std::shared_ptr<Engine> prefill_engine,
              std::shared_ptr<Engine> decode_engine,
              std::shared_ptr<BaseTokenizer> tokenizer);
    ~LLMRunner();

    LLMRunner(const LLMRunner&) = delete;
    LLMRunner& operator=(const LLMRunner&) = delete;

    // 构造期资源（KV Cache 显存、缓冲）是否就绪。
    bool ok() const { return valid_; }

    // KV 块池里还有多少块空闲。AC3 的判据依据；未就绪时返回 -1。
    int32_t NumFreeKvBlocks() const;

    // 返回**新生成**的 token（不含 prompt）。
    // 约定：成功时至少返回 1 个 token，因此**返回空 vector 表示失败**，原因同时记录日志。
    std::vector<int64_t> Generate(const std::vector<int64_t>& input_ids,
                                  const GenerateOptions& options);

    // 批量入口（S1：静态批，同进同出）。
    // 成功时返回 size() == requests.size() 的结果，且每条至少 1 个 token；
    // **任一条不满足入口约束、或任一步执行失败 → 整批拒绝，返回空 vector**（不做部分成功）。
    std::vector<GenerateResult> GenerateBatch(const std::vector<GenerateRequest>& requests);

    // ---- S3：请求级调度（槽位模型，design.md D12）----
    //
    // 一条请求加上"第几步进入等待队列"。
    // 到达时刻**由调用方给定**而不是看墙上时钟：调度必须可复现，否则 AC1 的"批量 == 逐条单跑"
    // 逐位对拍就没法做（同一输入要给出同一结果）。
    struct SchedulerRequest {
        GenerateRequest request;
        int32_t arrival_step = 0;
    };

    // 跑完整个请求集合（内部逐步调度），返回**按 requests 顺序**回填的结果。
    // 失败语义与 GenerateBatch 一致：任一不可恢复错误 → 整批拒绝（返回空 vector）。
    //
    // 与 GenerateBatch 的区别：批次成员可以在**运行中**变化（某条结束 → 退还槽位 → 新请求入空槽），
    // 而 GenerateBatch 是"一批同进同出"。两者共用同一套逐行缓冲与采样路径。
    // 内部状态（活跃表、finish flag 与它的 pinned 暂存）随实现一起落地。
    std::vector<GenerateResult> RunScheduler(const std::vector<SchedulerRequest>& requests);

    // ---- S3 调度的观测口（**只读**，不参与任何计算）----
    //
    // 为什么需要它：S3 有两条判据在公开接口上**本来不可观测**——
    //   ① "采到 EOS 的序列在下一步就退出"：只看 token 分不出"提前退出"与"跑满 max_new 再截断"
    //      （EOS 之后的 token 反正会被截掉）；
    //   ② "context 段只装本步新入批的行"：runner 不暴露 KV cache，读不回 K/V 做逐位比对。
    // 有了 `steps` 与 `context_rows`，这两条都能在 runner 层用可观测的量固定（见 test_plan.md）。
    // 注意：这些量只反映"调度怎么走的"，**不是**性能指标——性能仍按 P4/P7 的协议测。
    //
    // **跨路径成立的量 vs 只属于某条路径的量（2026-10-04 作者确认）**：
    // S4 走"每步一次调用、两相共享 packed 张量"，`prefill_calls` / `decode_calls` 在 S4 下**失去原义**。
    // 因此约定：`steps` / `max_active` / `context_rows` / `generation_rows` 是**跨路径**口径
    // （两条路径的用例都只能依赖这四个）；`prefill_calls` / `decode_calls` **只属于 S3 的两段式**
    // （S4 下不要读，实现 S4 时也不要去凑它们的语义）。
    struct SchedulerStats {
        int32_t steps = 0;           // 跨路径：本次调用实际走了多少轮循环（空闲跳步只算一轮）
        int32_t max_active = 0;      // 跨路径：同时活跃的最大序列数
        int32_t context_rows = 0;    // 跨路径：Σ context 行 —— 一共写回了几行 prompt K/V
        int32_t generation_rows = 0; // 跨路径：Σ generation 行 —— 一共喂了几行 generation token
        int32_t prefill_calls = 0;   // **仅 S3**：context 段（prefill 引擎）调用次数
        int32_t decode_calls = 0;    // **仅 S3**：generation 段（decode 引擎）调用次数
    };

    // 上一次 RunScheduler 的统计；没跑过、或入口校验直接拒绝时全为 0（GenerateBatch 不写它）。
    const SchedulerStats& scheduler_stats() const { return scheduler_stats_; }

 private:
    // ---- S3 调度的活跃表（行号 = 本步的引擎行号；退出即压实行号）----
    struct ActiveSequence {
        int32_t seq_id = -1;
        int32_t result_slot = -1;  // 回填结果的槽位 = 请求下标（确定性顺序）
        int32_t prompt_len = 0;
        int32_t generated = 0;  // 已采出的 token 数
        int32_t max_new = 0;
        int32_t top_k = 1;
        float top_p = 1.0f;
        uint64_t seed = 0;
        bool finished = false;  // EOS（设备侧 flag 回读）或到达 max_new
    };

    bool ReserveBuffers(int32_t prompt_len, int32_t max_new_tokens, int32_t batch);
    // 把整批 prompt 一次拷进设备并设形状；tokens 是 batch * seq_len 个 id 的扁平数组。
    // `row_lengths[i]` 是第 i 行的**真实**长度：它同时决定 padding_bias（≥ 真实长度的位置填
    // 加性 -1e4，见 p5_s3_interface_spec §3）。静态批传"每行都等于 seq_len"即原行为（全 0）。
    bool BindPrefill(const std::vector<int32_t>& tokens, int32_t batch, int32_t seq_len,
                     const std::vector<int32_t>& row_lengths);
    // input_tokens 指向本步每行的当前 token（行步长 = batch_capacity_）。
    bool BindDecode(const int32_t* input_tokens, int32_t batch);
    // 取 logits 的某一行（设备指针）。两套缓冲形状不同，必须区分：
    //   prefill: [B, S, V]，要的是第 batch_row 条的**第 position 个位置**；padding 路径下它是
    //            该行的真实末位 (L_i - 1)，不是 S-1（S-1 是填充位置，那里的 logits 无意义）；
    //   decode:  [B, V]，每行连续，直接取缓冲首地址即可（不需要逐行寻址）。
    const void* PrefillLogitsRow(int32_t batch_row, int32_t seq_len, int32_t position) const;
    // 采样一整批并写进 token_out（batch 个连续 int32）。
    // from_prefill=true 读"已收集的末行"缓冲，false 读 decode 缓冲。
    // S1 限制：三种策略各自是"整批一个分支"，因此要求批内同策略（见 GenerateBatch 的校验）。
    // `row_offsets` / `eos_hit` 非空时：前者给每行自己的随机步号（调度下各行步号不同），
    // 后者让采样器顺带写"该行采到 EOS"的设备侧标记（p5_s3_interface_spec §4）。
    bool SampleBatch(void* token_out, bool from_prefill, int32_t batch, uint64_t offset,
                     const uint64_t* row_offsets, int8_t* eos_hit, cudaStream_t stream);
    // 把 active[begin, begin+count) 的逐行采样参数（k / p / seed / 随机步号）上传到设备缓冲。
    bool UploadRowParams(const std::vector<ActiveSequence>& active, int32_t begin, int32_t count);
    // 同上，但按**显式顺序**（S4 的 packed 行序：context 行在前、generation 行在后）上传。
    bool UploadRowParamsByOrder(const std::vector<ActiveSequence>& active,
                                const std::vector<int32_t>& active_indices);
    // **S4/S5 的一步**：打包 → 一次 packed 调用 → context 写回 / generation 追加 → 采样。
    // `active` 的行序 = 活跃表序；packed 行序 = context 行在前、generation 行在后。
    // **两段的分工不由调用方传入**：S5 起按 cache 的已写入长度在函数内重算（完成 prefill 的行
    // 可能夹在未完成的行后面），所以这里不再有"生成段前缀行数 / 新入批行数"两个参数 ——
    // S4 的"generation 行 = 活跃表前缀"这个前提在分块下不成立。
    bool RunPackedMixedStep(const std::vector<GenerateRequest>& requests,
                            const std::vector<ActiveSequence>& active, int32_t max_new);

    Config config_;
    bool valid_ = false;
    std::shared_ptr<Engine> prefill_engine_;
    std::shared_ptr<Engine> decode_engine_;
    std::shared_ptr<BaseTokenizer> tokenizer_;
    std::unique_ptr<PagedKVCache> kv_cache_;

    // 复用缓冲：每次请求按需扩容，**解码循环内不分配**。
    DeviceBuffer d_prompt_;       // 输入的 prompt token（[1, S0] INT32）
    DeviceBuffer d_position_;     // position_ids（[1, S0] 或 [1,1] INT32）
    // prefill 的加性 padding bias [B, 1, 1, S] FP32：S1/S2 的批内等长 → 全 0；
    // S3 的调度器会把每行的真实长度填进来（design.md D12 / S3 方案 §3）。
    DeviceBuffer d_padding_bias_;
    DeviceBuffer d_tokens_;       // 生成结果；采样器直接写到对应位置
    DeviceBuffer d_prefill_logits_;
    // prefill 末行的收集缓冲 [B, V]：采样器要求 logits 连续，而 [B,S,V] 的末行是跨步的。
    DeviceBuffer d_prefill_last_logits_;
    DeviceBuffer d_decode_logits_;
    DeviceBuffer d_top_k_;        // per-batch k（本版 batch = 1）
    DeviceBuffer d_top_p_;        // per-batch p
    // per-batch seed（[B]）：让随机流只由 (请求 seed, 步数) 决定，与批位置无关（AC1 的前提）。
    DeviceBuffer d_seeds_;
    DeviceBuffer d_sampler_workspace_;
    // ---- S3 调度：每步重建的逐行缓冲（按 max_batch 备好，循环内不分配）----
    // [max_batch] int32：采样器**按行**输出新 token 的暂存（采样器只认连续 [batch] 输出）。
    // 之后会被搬进按序列聚集的结果缓冲 —— 行号每步都会变，token 不能按行号存放。
    DeviceBuffer d_step_tokens_;
    // [max_batch] int32：生成段的输入（每行自己的上一个 token），每步从结果缓冲里聚集一次。
    DeviceBuffer d_decode_input_;
    DeviceBuffer d_offsets_;        // [max_batch] uint64：per-row 随机步号（= 该行已生成计数）
    DeviceBuffer d_eos_hit_;        // [max_batch] int8：设备侧 finish flag
    DeviceBuffer d_result_tokens_;  // [请求数, max_new] int32：结果按序列聚集
    PinnedBuffer host_eos_;         // finish flag 的 pinned 回读暂存
    // 上一次 RunScheduler 的观测统计（只读出口是 scheduler_stats()）。
    SchedulerStats scheduler_stats_;
    // ---- S4 packed 混合批的逐行缓冲（同样按容量备好，循环内只填不分配）----
    DeviceBuffer d_packed_tokens_;         // [T_max] int32：input_ids[T]
    DeviceBuffer d_packed_positions_;      // [T_max] int32：position_ids[T]
    DeviceBuffer d_cu_seqlens_ctx_;        // [max_batch+1] int32：**段内**前缀和（context 段）
    DeviceBuffer d_context_seq_count_;     // [1] int32：段边界 B_ctx
    DeviceBuffer d_packed_block_tables_;   // [max_batch, max_blocks_per_seq] int32（**按 packed 行序**）
    DeviceBuffer d_packed_context_lens_;   // [max_batch] int32（同上，**推进前**的值）
    // ---- S5 chunked prefill ----
    // 每步最多送多少 prompt token（构造期从 `Config::max_prefill_seq_len` 抄一份，之后只读）。
    // 只在 `prefill_mode == kPackedMixed` 时有意义；该字段未声明 / 越界 / 交叉校验不过时构造期直接
    // 拒绝（`valid_ = false` —— 见构造期那五条的注释与 spec §2）。
    int32_t chunk_limit_ = 0;
    // 本步 packed 的两段行数（由 `RunPackedMixedStep` 填，供 `SchedulerStats` 与结果落位读）。
    // S5 之后两段**不再按活跃表前缀切**：context 段 = 本步拿到 chunk 的行，generation 段 = 本步
    // 能喂 token 的行（= prefill 已完成的行，可能夹在未完成的行后面）。
    int32_t packed_context_rows_ = 0;
    int32_t packed_generation_rows_ = 0;
    // 本步参与采样的活跃行号，**升序**（槽位 = 数组下标）。S5 下"本步完成 prefill 的行"可能被
    // 未完成的行隔开，采样器只吃连续 `[count]`，所以按这张表把末位 logits 与 per-row 参数压紧；
    // 结果落位（`RunScheduler` 的 ⑤）与 finish-flag 回读也按同一张表，避免两处口径漂移。
    std::vector<int32_t> sample_active_indices_;
    // 每层的 K/V 输出缓冲（prefill/decode 各自的形状不同）
    std::vector<std::unique_ptr<DeviceBuffer>> d_prefill_kv_;
    std::vector<std::unique_ptr<DeviceBuffer>> d_decode_kv_;
    // 只增不减的容量水位：决定各缓冲的实际尺寸（S2 会把它改成构造期一次到位）。
    int32_t batch_capacity_ = 0;
    int32_t prompt_capacity_ = 0;
    int32_t token_capacity_ = 0;

    // 引擎**实际声明**的边界精度（弱类型网络下 TRT 决定，不由 weight_dtype 决定；
    // 见 TROUBLESHOOTING + TS-018）。缓冲分配、logits 行步长与采样器都按它们走，
    // 而不是按 config_.is_half——那正是导致 FP16 端到端非法访存的那个假定。
    bool prefill_kv_half_ = false;
    bool decode_kv_half_ = false;
    bool prefill_logits_half_ = false;
    bool decode_logits_half_ = false;

    // 本次请求的采样参数。放在成员里是为了让 SampleInto 不必逐层传参；
    // 每个 Generate 开头覆写，生命周期不超出该次调用。

    int32_t options_top_k_ = 1;
    float options_top_p_ = 1.0f;
    uint64_t options_seed_ = 42;
};

}  // namespace mini_trt_llm
