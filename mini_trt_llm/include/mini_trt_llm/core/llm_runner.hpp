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
        int32_t vocab_size = 0;
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

    LLMRunner(const Config& config, std::shared_ptr<Engine> prefill_engine,
              std::shared_ptr<Engine> decode_engine,
              std::shared_ptr<BaseTokenizer> tokenizer);
    ~LLMRunner();

    LLMRunner(const LLMRunner&) = delete;
    LLMRunner& operator=(const LLMRunner&) = delete;

    // 构造期资源（KV Cache 显存、缓冲）是否就绪。
    bool ok() const { return valid_; }

    // 返回**新生成**的 token（不含 prompt）。
    // 约定：成功时至少返回 1 个 token，因此**返回空 vector 表示失败**，原因同时记录日志。
    std::vector<int64_t> Generate(const std::vector<int64_t>& input_ids,
                                  const GenerateOptions& options);

 private:
    bool ReserveBuffers(int32_t prompt_len, int32_t max_new_tokens);
    bool BindPrefill(const std::vector<int32_t>& tokens);
    bool BindDecode(const int32_t* input_token);
    // 取 logits 的某一行（设备指针）。
    // row >= 0 → prefill 输出 [1, S0, V] 的第 row 行；
    // row == -1 → decode 输出 [1, 1, V] 的唯一一行（两套缓冲形状不同，必须区分）。
    const void* LogitsRow(int32_t row) const;
    // 采样一次并写入 token_out；offset 用于让不同步的随机流不重复（Greedy 无关）。
    bool SampleInto(void* token_out, int32_t row, uint64_t offset, cudaStream_t stream);

    Config config_;
    bool valid_ = false;
    std::shared_ptr<Engine> prefill_engine_;
    std::shared_ptr<Engine> decode_engine_;
    std::shared_ptr<BaseTokenizer> tokenizer_;
    std::unique_ptr<PagedKVCache> kv_cache_;

    // 复用缓冲：每次请求按需扩容，**解码循环内不分配**。
    DeviceBuffer d_prompt_;       // 输入的 prompt token（[1, S0] INT32）
    DeviceBuffer d_position_;     // position_ids（[1, S0] 或 [1,1] INT32）
    DeviceBuffer d_tokens_;       // 生成结果；采样器直接写到对应位置
    DeviceBuffer d_prefill_logits_;
    DeviceBuffer d_decode_logits_;
    DeviceBuffer d_top_k_;        // per-batch k（本版 batch = 1）
    DeviceBuffer d_top_p_;        // per-batch p
    DeviceBuffer d_sampler_workspace_;
    // 每层的 K/V 输出缓冲（prefill/decode 各自的形状不同）
    std::vector<std::unique_ptr<DeviceBuffer>> d_prefill_kv_;
    std::vector<std::unique_ptr<DeviceBuffer>> d_decode_kv_;
    int32_t prompt_capacity_ = 0;
    int32_t token_capacity_ = 0;

    // 本次请求的采样参数。放在成员里是为了让 SampleInto 不必逐层传参；
    // 每个 Generate 开头覆写，生命周期不超出该次调用。
    int32_t options_top_k_ = 1;
    float options_top_p_ = 1.0f;
    uint64_t options_seed_ = 42;
};

}  // namespace mini_trt_llm
