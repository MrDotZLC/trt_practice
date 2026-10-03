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
