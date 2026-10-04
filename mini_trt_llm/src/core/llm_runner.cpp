#include "mini_trt_llm/core/llm_runner.hpp"

#include "mini_trt_llm/core/llm_runner_kernel.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/logger.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace mini_trt_llm {
namespace {

size_t ElementSize(bool is_half) { return is_half ? 2u : 4u; }

// 右填充位置的加性注意力偏置（p5_s3_interface_spec §3 的 0 / -1e4 口径）。
// 为什么是 -1e4 而不是 -inf：exp(-1e4 - max) 直接下溢到 0，屏蔽效果与 -inf 等价，
// 又不会在减法里产生 inf - inf = NaN。
constexpr float kPaddingBias = -1e4f;

// 本批已登记序列的作用域守卫：**任何出口都归还**（正常结束与各条失败路径共用一条路径）。
// 为什么不手工归还：GenerateBatch 有多条失败出口，靠人记得在每个出口 FreeSequence 早晚会漏；
// 而漏一个就是静默泄漏——AC3（跑 N 轮后空闲块回到初始水位）会直接不成立。
class SequenceScope {
 public:
    explicit SequenceScope(PagedKVCache* cache) : cache_(cache) {}
    ~SequenceScope() {
        for (int32_t seq_id : ids_) {
            cache_->FreeSequence(seq_id);
        }
    }

    SequenceScope(const SequenceScope&) = delete;
    SequenceScope& operator=(const SequenceScope&) = delete;

    void Add(int32_t seq_id) { ids_.push_back(seq_id); }
    const std::vector<int32_t>& ids() const { return ids_; }

 private:
    PagedKVCache* cache_;
    std::vector<int32_t> ids_;
};

// **测试专用**的 chunk_limit 覆盖值（说明见 hpp）：生产路径恒为 0（= 不覆盖，用引擎 profile 推导）。
int32_t g_chunk_limit_override = 0;

}  // namespace

void SetChunkLimitOverride(int32_t chunk_limit) noexcept { g_chunk_limit_override = chunk_limit; }

int32_t ChunkLimitOverride() noexcept { return g_chunk_limit_override; }

LLMRunner::LLMRunner(const Config& config, std::shared_ptr<Engine> prefill_engine,
                     std::shared_ptr<Engine> decode_engine,
                     std::shared_ptr<BaseTokenizer> tokenizer)
    : config_(config),
      prefill_engine_(std::move(prefill_engine)),
      decode_engine_(std::move(decode_engine)),
      tokenizer_(std::move(tokenizer)) {
    if (config_.num_layers <= 0 || config_.num_kv_heads <= 0 || config_.head_size <= 0 ||
        config_.block_size <= 0 || config_.max_blocks_per_seq <= 0 ||
        config_.num_blocks <= 0 || config_.vocab_size <= 0 || config_.max_batch <= 0) {
        MINI_TRT_LOG_ERROR("LLMRunner: invalid config");
        return;
    }
    if (prefill_engine_ == nullptr || decode_engine_ == nullptr) {
        MINI_TRT_LOG_ERROR("LLMRunner: prefill/decode engine must not be null");
        return;
    }

    // **校验引擎声明的边界精度，而不是假定它**（TROUBLESHOOTING + TS-018）：
    // 弱类型网络里导出的 K/V 与 logits 的类型**由 TRT 决定，不由 `weight_dtype` 决定**
    // （实测 FP16 引擎下它们是 FP32；试过用 `addCast` 在图上钉死，实测无效）。
    // 而这种不一致的后果是"缓冲越界写 / cache 宽度全错"，**不报错**——
    // 所以必须在这里显式拒绝启动，并打印两边的精度。
    const nvinfer1::DataType expected =
        config_.is_half ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    const auto check = [&](Engine* engine, const char* name,
                           const char* description) -> bool {
        nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
        if (cuda == nullptr) {
            return false;
        }
        const nvinfer1::DataType actual = cuda->getTensorDataType(name);
        if (actual != expected) {
            MINI_TRT_LOG_ERROR("LLMRunner: "
                               << description << " '" << name << "' declares "
                               << (actual == nvinfer1::DataType::kHALF ? "FP16" : "FP32")
                               << " but config says "
                               << (config_.is_half ? "FP16" : "FP32")
                               << " —— cache/缓冲/采样器的宽度会全错，拒绝启动");
            return false;
        }
        return true;
    };
    // **cache 输入必须与 config 一致**：它是引擎与 PagedKVCache 之间的契约（布局与宽度），
    // 不一致会让 PagedAttention 按错误宽度读 cache——保留这道闸，不与"可适配"混为一谈。
    if (!check(decode_engine_.get(), "key_cache_0", "decode KV cache 输入")) {
        return;
    }
    // K/V 与 logits 的**实际**精度：TRT 决定，因此只记录、不假定（`docs/TROUBLESHOOTING.md` + TS-018）。
    // 缓冲按它们分配，KV 写入内核按"源→目标"转换，采样器按 logits 精度读。
    const auto dtype_of = [](Engine* engine, const char* name) {
        return engine->GetCudaEngine()->getTensorDataType(name) == nvinfer1::DataType::kHALF;
    };
    prefill_kv_half_ = dtype_of(prefill_engine_.get(), "k_layer0");
    decode_kv_half_ = dtype_of(decode_engine_.get(), "k_layer0");
    prefill_logits_half_ = dtype_of(prefill_engine_.get(), "logits");
    decode_logits_half_ = dtype_of(decode_engine_.get(), "logits");
    MINI_TRT_LOG_INFO("LLMRunner: 引擎边界精度 prefill K/V="
                      << (prefill_kv_half_ ? "FP16" : "FP32") << ", prefill logits="
                      << (prefill_logits_half_ ? "FP16" : "FP32") << ", decode K/V="
                      << (decode_kv_half_ ? "FP16" : "FP32") << ", decode logits="
                      << (decode_logits_half_ ? "FP16" : "FP32"));
    if (prefill_kv_half_ != decode_kv_half_) {
        MINI_TRT_LOG_ERROR("LLMRunner: prefill 与 decode 的 K/V 输出精度不一致，"
                           "KV Cache 无法用同一路径写入");
        return;
    }

    // D8 / design.md §不变量 2：prefill 与 decode 各用一个**独立构建**的引擎，且各自恰好只有
    // 一个 optimization profile。单引擎构建（kSingle）会把 Prefill / Decode 两组 profile 挂在
    // 同一个引擎上，此时"decode 用 profile 0"就是错的组——而**用错组不会报错**，只会表现为
    // "形状设了却不生效"这类极难查的现象。所以在构造期就拒绝，而不是等真机跑出怪结果。
    const auto check_profile_count = [&](Engine* engine, const char* name) -> bool {
        nvinfer1::ICudaEngine* cuda = engine->GetCudaEngine();
        const int32_t profiles = cuda == nullptr ? 0 : cuda->getNbOptimizationProfiles();
        if (profiles != 1) {
            MINI_TRT_LOG_ERROR("LLMRunner: "
                               << name << " declares " << profiles
                               << " optimization profiles; this runtime requires exactly 1"
                                  "（单引擎同时挂两组 profile 时 decode 并不是 0 号）");
            return false;
        }
        return true;
    };
    if (!check_profile_count(prefill_engine_.get(), "prefill engine") ||
        !check_profile_count(decode_engine_.get(), "decode engine")) {
        return;
    }

    // KV Cache 的创建放在精度查询**之后**：cache 的元素精度（is_half）取自家配置，
    // **源**精度（引擎导出的 K/V）取查询结果——两者不同时由写入内核做转换（`docs/TROUBLESHOOTING.md` + TS-018）。
    PagedKVCache::Config cache_config;
    cache_config.num_blocks = config_.num_blocks;
    cache_config.block_size = config_.block_size;
    cache_config.num_layers = config_.num_layers;
    cache_config.num_kv_heads = config_.num_kv_heads;
    cache_config.head_size = config_.head_size;
    cache_config.is_half = config_.is_half;
    cache_config.source_is_half = prefill_kv_half_;
    cache_config.max_blocks_per_seq = config_.max_blocks_per_seq;
    // 元数据缓冲按 max_batch 预分配（S2）——必须与 runner 侧同源，否则批号超出的行没地方放。
    cache_config.max_batch = config_.max_batch;
    kv_cache_ = std::make_unique<PagedKVCache>(cache_config);
    if (!kv_cache_->valid()) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to create KV cache");
        return;
    }

    // 采样器的 per-batch 参数（k / p）在整个请求里是常量，所以在这里上传一次，
    // 不放进解码循环——循环里只允许有"已经备好"的设备侧动作。
    // ---- S5：chunked prefill 的 chunk_limit（构造期从引擎 profile 推导，不暴露给调用方）----
    // 取 prefill 的 token 维上界：packed 图的 `input_ids` 是 [1, T]，所以是第 1 维的 kMAX。
    // **为什么不从 Config 要一个字段**：那会与真实建的图漂移；项目既有纪律是"按对方查询、
    // 不按配置假定"（workspace 版见 `paged_attention_split.hpp` 与 `PROGRESS.md` §2.15）。
    // 推导失败或越界一律**构造期拒绝**（兜底纪律：不许静默退到慢路径，也不许猜一个默认值）。
    if (config_.prefill_mode == Config::PrefillMode::kPackedMixed) {
        const int32_t from_profile = prefill_engine_->GetProfileDim(
            "input_ids", nvinfer1::OptProfileSelector::kMAX, 1);
        if (from_profile <= 0) {
            MINI_TRT_LOG_ERROR("LLMRunner: cannot derive chunk_limit from the prefill profile "
                               "(input_ids dim 1 = " << from_profile << ")");
            return;
        }
        const int32_t declared = ChunkLimitOverride();
        if (declared > from_profile) {
            MINI_TRT_LOG_ERROR("LLMRunner: chunk_limit override " << declared
                               << " exceeds the engine profile bound " << from_profile);
            return;
        }
        chunk_limit_ = declared > 0 ? declared : from_profile;
        MINI_TRT_LOG_INFO("LLMRunner: chunk_limit = " << chunk_limit_
                        << (declared > 0 ? " (test override)" : " (from engine profile)")
                        << ", profile bound = " << from_profile);
    }
    valid_ = true;
}

LLMRunner::~LLMRunner() = default;

int32_t LLMRunner::NumFreeKvBlocks() const {
    return kv_cache_ ? kv_cache_->NumFreeBlocks() : -1;
}
bool LLMRunner::ReserveBuffers(int32_t prompt_len, int32_t max_new_tokens, int32_t batch) {
    const size_t prefill_kv_elem = ElementSize(prefill_kv_half_);
    const size_t decode_kv_elem = ElementSize(decode_kv_half_);
    const size_t prefill_logits_elem = ElementSize(prefill_logits_half_);
    const size_t decode_logits_elem = ElementSize(decode_logits_half_);
    const size_t kv_layer_elems = static_cast<size_t>(config_.num_kv_heads) * config_.head_size;

    // 容量水位只增不减：够用就复用设备指针（`DeviceBuffer::Allocate` 的语义）。
    // S2 会把这一步挪到构造期；在那之前"每步重绑"仍是正确性的来源（design.md §不变量 5）。
    const bool grew = batch > batch_capacity_ || prompt_len > prompt_capacity_ ||
                      max_new_tokens > token_capacity_;
    batch_capacity_ = std::max(batch_capacity_, batch);
    prompt_capacity_ = std::max(prompt_capacity_, prompt_len);
    token_capacity_ = std::max(token_capacity_, max_new_tokens);
    if (!grew && !d_prefill_kv_.empty() && !d_decode_kv_.empty()) {
        return true;
    }

    const size_t batch_sz = static_cast<size_t>(batch_capacity_);
    const size_t seq_sz = static_cast<size_t>(prompt_capacity_);
    const size_t vocab_sz = static_cast<size_t>(config_.vocab_size);

    // 每层的 K/V 输出缓冲：prefill 是 [B, kv_heads, S, D]，decode 是 [B, kv_heads, 1, D]。
    d_prefill_kv_.clear();
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        for (int32_t which = 0; which < 2; ++which) {
            auto buffer = std::make_unique<DeviceBuffer>();
            if (!buffer->Allocate(kv_layer_elems * seq_sz * batch_sz * prefill_kv_elem)) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate prefill K/V buffer");
                return false;
            }
            d_prefill_kv_.push_back(std::move(buffer));
        }
    }
    d_decode_kv_.clear();
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        for (int32_t which = 0; which < 2; ++which) {
            auto buffer = std::make_unique<DeviceBuffer>();
            if (!buffer->Allocate(kv_layer_elems * batch_sz * decode_kv_elem)) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate decode K/V buffer");
                return false;
            }
            d_decode_kv_.push_back(std::move(buffer));
        }
    }

    // 各张量按其**实际声明精度**分配（docs/TROUBLESHOOTING.md + TS-018）：
    // FP16 引擎里 logits/K/V 实测是 FP32，用一个假定的全局 elem 会直接越界写。
    if (!d_prompt_.Allocate(batch_sz * seq_sz * sizeof(int32_t)) ||
        !d_position_.Allocate(batch_sz * seq_sz * sizeof(int32_t)) ||
        !d_padding_bias_.Allocate(batch_sz * seq_sz * sizeof(float)) ||
        !d_prefill_logits_.Allocate(batch_sz * seq_sz * vocab_sz * prefill_logits_elem) ||
        !d_prefill_last_logits_.Allocate(batch_sz * vocab_sz * prefill_logits_elem) ||
        !d_decode_logits_.Allocate(batch_sz * vocab_sz * decode_logits_elem)) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate prefill/decode buffers");
        return false;
    }
    if (!d_tokens_.Allocate(batch_sz * static_cast<size_t>(token_capacity_) * sizeof(int32_t)) ||
        !d_top_k_.Allocate(batch_sz * sizeof(int32_t)) ||
        !d_top_p_.Allocate(batch_sz * sizeof(float)) ||
        !d_seeds_.Allocate(batch_sz * sizeof(uint64_t))) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate token/sampling buffers");
        return false;
    }
    const size_t workspace = TopKSamplerWorkspaceBytes(batch_capacity_, config_.vocab_size);
    if (d_sampler_workspace_.size() < workspace && !d_sampler_workspace_.Allocate(workspace)) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate sampler workspace");
        return false;
    }
    return true;
}

const void* LLMRunner::PrefillLogitsRow(int32_t batch_row, int32_t seq_len,
                                        int32_t position) const {
    if (seq_len <= 0 || position < 0 || position >= seq_len) {
        return nullptr;
    }
    const size_t elem = ElementSize(prefill_logits_half_);
    // prefill logits 是 [B, S, V]：第 batch_row 条第 position 个位置就是 (batch_row, position)。
    // padding 路径下必须传该行的真实末位 L-1 —— S-1 是填充位置，那里的 logits 不来自真实 token。
    const size_t index = (static_cast<size_t>(batch_row) * static_cast<size_t>(seq_len) +
                          static_cast<size_t>(position)) *
                         static_cast<size_t>(config_.vocab_size);
    return static_cast<const char*>(d_prefill_logits_.data()) + index * elem;
}

bool LLMRunner::SampleBatch(void* token_out, bool from_prefill, int32_t batch, uint64_t offset,
                            const uint64_t* row_offsets, int8_t* eos_hit,
                            cudaStream_t stream) {
    // S1 限制：批内同策略——三种采样 kernel 各自是"整批一个分支"（入口已校验）。
    const bool use_top_p = options_top_p_ < 1.0f;
    const bool use_greedy = !use_top_p && options_top_k_ <= 1;
    const bool logits_half = from_prefill ? prefill_logits_half_ : decode_logits_half_;
    // 采样器契约：logits 必须是连续的 [batch, vocab]。prefill 的末行已收集进专门缓冲；
    // decode 的 [B,1,V] 本身连续。
    const void* logits = from_prefill ? d_prefill_last_logits_.data() : d_decode_logits_.data();
    // 随机流口径：seeds 一直给（行号不进哈希）；row_offsets 只有调度器给（各行的已生成计数不同）。
    // eos_hit 非空时顺带写设备侧 finish flag —— 调度器只回读 batch_size 字节就能决定谁退出。
    const uint64_t* seeds = static_cast<const uint64_t*>(d_seeds_.data());
    const int32_t eos_token_id = (eos_hit != nullptr) ? config_.eos_token_id : -1;

    if (use_greedy) {
        SamplerArgs args;
        args.logits = logits;
        args.token_ids = static_cast<int32_t*>(token_out);
        args.batch_size = batch;
        args.vocab_size = config_.vocab_size;
        args.is_half = logits_half;
        args.seed = options_seed_;
        args.offset = offset;
        args.seeds = seeds;
        args.offsets = row_offsets;
        args.eos_hit = eos_hit;
        args.eos_token_id = eos_token_id;
        return LaunchGreedySampler(args, stream) == cudaSuccess;
    }
    if (use_top_p) {
        TopPSamplerArgs args;
        args.logits = logits;
        args.token_ids = static_cast<int32_t*>(token_out);
        args.batch_size = batch;
        args.vocab_size = config_.vocab_size;
        args.is_half = logits_half;
        args.seed = options_seed_;
        args.offset = offset;
        args.seeds = seeds;
        args.offsets = row_offsets;
        args.eos_hit = eos_hit;
        args.eos_token_id = eos_token_id;
        args.top_p = static_cast<const float*>(d_top_p_.data());
        return LaunchTopPSampler(args, stream, d_sampler_workspace_.data(),
                                 d_sampler_workspace_.size()) == cudaSuccess;
    }
    TopKSamplerArgs args;
    args.logits = logits;
    args.token_ids = static_cast<int32_t*>(token_out);
    args.batch_size = batch;
    args.vocab_size = config_.vocab_size;
    args.is_half = logits_half;
    args.seed = options_seed_;
    args.offset = offset;
    args.seeds = seeds;
    args.offsets = row_offsets;
    args.eos_hit = eos_hit;
    args.eos_token_id = eos_token_id;
    args.top_k = static_cast<const int32_t*>(d_top_k_.data());
    // **快速路径暂时不接生产路径**：真机实测它比旧路径慢 6~9 倍
    // （见 future_iterations_development_plan.md §10.5 的失败记录），根因是 occupancy 与 bank conflict。
    return LaunchTopKSampler(args, stream, d_sampler_workspace_.data(),
                             d_sampler_workspace_.size()) == cudaSuccess;
}

bool LLMRunner::BindPrefill(const std::vector<int32_t>& tokens, int32_t batch, int32_t seq_len,
                            const std::vector<int32_t>& row_lengths) {
    // 顺序要求：先选 profile 再设形状（反过来时 TRT 可能用 profile 的 opt 形状覆盖显式形状）。
    if (!prefill_engine_->SetOptimizationProfile(0, nullptr)) {
        return false;
    }
    if (!prefill_engine_->SetInputShape("input_ids", nvinfer1::Dims{2, {batch, seq_len}}) ||
        !prefill_engine_->SetInputShape("position_ids", nvinfer1::Dims{2, {batch, seq_len}})) {
        return false;
    }

    // 批内等长（D2=A）：每行的位置都是 0..S-1。
    std::vector<int32_t> positions(static_cast<size_t>(batch) * static_cast<size_t>(seq_len));
    for (int32_t b = 0; b < batch; ++b) {
        for (int32_t i = 0; i < seq_len; ++i) {
            positions[static_cast<size_t>(b) * static_cast<size_t>(seq_len) + i] = i;
        }
    }
    // padding bias：S1/S2 的批内等长，全 0 即正确（每行都吃满 S）。
    // S3 的 padding 路径按每行的真实长度填 0 / -1e4（design.md D12 / S3 方案 §3）：
    // 偏置是加在**注意力分数**上的（[B,1,1,S]，作用在 key 上），所以真实位置之后
    // 那些填充位置对任何 query 都被屏蔽；填充位置自己的 logits 没人用（采样取真实末位）。
    if (static_cast<int32_t>(row_lengths.size()) != batch) {
        MINI_TRT_LOG_ERROR("LLMRunner: BindPrefill needs one row length per batch row");
        return false;
    }
    std::vector<float> padding(positions.size(), 0.0f);
    for (int32_t b = 0; b < batch; ++b) {
        const int32_t len = row_lengths[static_cast<size_t>(b)];
        if (len <= 0 || len > seq_len) {
            MINI_TRT_LOG_ERROR("LLMRunner: row length " << len << " out of (0, " << seq_len << "]");
            return false;
        }
        for (int32_t i = len; i < seq_len; ++i) {
            padding[static_cast<size_t>(b) * static_cast<size_t>(seq_len) +
                    static_cast<size_t>(i)] = kPaddingBias;
        }
    }
    if (cudaMemcpyAsync(d_prompt_.data(), tokens.data(), tokens.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_position_.data(), positions.data(),
                        positions.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_padding_bias_.data(), padding.data(),
                        padding.size() * sizeof(float), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess) {
        return false;
    }
    if (!prefill_engine_->SetTensorAddress("input_ids", d_prompt_.data()) ||
        !prefill_engine_->SetTensorAddress("position_ids", d_position_.data()) ||
        !prefill_engine_->SetTensorAddress("padding_bias", d_padding_bias_.data()) ||
        !prefill_engine_->SetTensorAddress("logits", d_prefill_logits_.data())) {
        return false;
    }
    // 每层导出 K/V（prefill 侧用来写 cache），形状由引擎按 [B, kv_heads, S, D] 给出。
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const std::string index = std::to_string(layer);
        if (!prefill_engine_->SetTensorAddress(("k_layer" + index).c_str(),
                                              d_prefill_kv_[static_cast<size_t>(layer) * 2]->data()) ||
            !prefill_engine_->SetTensorAddress(
                ("v_layer" + index).c_str(),
                d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data())) {
            return false;
        }
    }
    return true;
}

bool LLMRunner::BindDecode(const int32_t* input_tokens, int32_t batch) {
    if (!decode_engine_->SetOptimizationProfile(0, nullptr)) {
        return false;
    }
    if (!decode_engine_->SetInputShape("input_ids", nvinfer1::Dims{2, {batch, 1}}) ||
        !decode_engine_->SetInputShape("position_ids", nvinfer1::Dims{2, {batch, 1}}) ||
        !decode_engine_->SetInputShape(
            "block_tables", nvinfer1::Dims{2, {batch, config_.max_blocks_per_seq}}) ||
        !decode_engine_->SetInputShape("context_lens", nvinfer1::Dims{1, {batch}})) {
        return false;
    }
    if (!decode_engine_->SetTensorAddress("input_ids", const_cast<int32_t*>(input_tokens)) ||
        !decode_engine_->SetTensorAddress("position_ids", d_position_.data()) ||
        !decode_engine_->SetTensorAddress("block_tables",
                                          const_cast<int32_t*>(kv_cache_->block_tables())) ||
        !decode_engine_->SetTensorAddress("context_lens",
                                          const_cast<int32_t*>(kv_cache_->context_lens())) ||
        !decode_engine_->SetTensorAddress("logits", d_decode_logits_.data())) {
        return false;
    }
    // 每层一对 cache 输入 + 每层导出的当前 token K/V。
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const std::string index = std::to_string(layer);
        if (!decode_engine_->SetTensorAddress(("key_cache_" + index).c_str(),
                                              kv_cache_->key_cache(layer)) ||
            !decode_engine_->SetTensorAddress(("value_cache_" + index).c_str(),
                                              kv_cache_->value_cache(layer)) ||
            !decode_engine_->SetTensorAddress(("k_layer" + index).c_str(),
                                              d_decode_kv_[static_cast<size_t>(layer) * 2]->data()) ||
            !decode_engine_->SetTensorAddress(
                ("v_layer" + index).c_str(),
                d_decode_kv_[static_cast<size_t>(layer) * 2 + 1]->data())) {
            return false;
        }
    }
    return true;
}

std::vector<LLMRunner::GenerateResult> LLMRunner::GenerateBatch(
    const std::vector<GenerateRequest>& requests) {
    if (!valid_) {
        MINI_TRT_LOG_ERROR("LLMRunner::GenerateBatch called on an invalid runner");
        return {};
    }
    // S4：packed 路径下**没有**"静态批 prefill 写回"这回事（引擎的 I/O 契约是 packed 张量），
    // 拿它去跑 padding 路径只会绑错张量。所以这里明确拒绝，并指向正确的入口。
    if (config_.prefill_mode == Config::PrefillMode::kPackedMixed) {
        MINI_TRT_LOG_ERROR("LLMRunner: prefill_mode = kPackedMixed 时请用 RunScheduler ——"
                           " GenerateBatch 是 padding 路径（S3）的入口；"
                           " packed 路径的\"逐条单跑\"参考实现 = 单请求的 RunScheduler");
        return {};
    }
    const int32_t batch = static_cast<int32_t>(requests.size());
    if (batch <= 0) {
        MINI_TRT_LOG_ERROR("LLMRunner: empty batch");
        return {};
    }
    if (batch > config_.max_batch) {
        MINI_TRT_LOG_ERROR("LLMRunner: batch " << batch << " exceeds max_batch "
                                               << config_.max_batch);
        return {};
    }

    // ---- 1. 逐条校验：S1 的入口契约，任一条不满足即整批拒绝（不做部分成功）----
    const GenerateOptions& first = requests[0].options;
    int32_t prompt_len = -1;
    int32_t max_new = 0;
    for (int32_t b = 0; b < batch; ++b) {
        const GenerateRequest& r = requests[b];
        const int32_t len = static_cast<int32_t>(r.input_ids.size());
        if (len <= 0 || r.options.max_new_tokens <= 0) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << b
                                                     << " has an empty prompt or max_new_tokens <= 0");
            return {};
        }
        if (r.options.temperature != 1.0f) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << b
                                                     << " uses temperature != 1.0 (unsupported)");
            return {};
        }
        if (r.options.top_k < 1 || r.options.top_p <= 0.0f || r.options.top_p > 1.0f) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << b << " has invalid sampling parameters");
            return {};
        }
        // 策略必须整批一致：三种采样 kernel 各自是"整批一个分支"（入口校验）。
        // top_k / top_p / seed 都可以逐行不同：seed 走 per-batch 数组，
        // 行号不进随机流（见 sampler_common.hpp 的 seeds 与 RowUniform01）。
        const bool first_top_p = first.top_p < 1.0f;
        const bool first_topk = !first_top_p && first.top_k > 1;
        const bool row_top_p = r.options.top_p < 1.0f;
        const bool row_topk = !row_top_p && r.options.top_k > 1;
        if (row_top_p != first_top_p || row_topk != first_topk) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << b
                               << " uses a different sampling strategy than request 0"
                                  " —— S1 要求批内同策略（D3 的逐行策略需要改采样器）");
            return {};
        }
        // Top-K 路径的调用方契约：top_k 不得超过 kTopKFastMaxK，越界行会被写哨兵 -1
        // （表现为"生成了 token = -1"这种极难查的现象，必须在入口拦住）。
        if (first_topk && r.options.top_k > kTopKFastMaxK) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << b << " top_k " << r.options.top_k
                                                     << " exceeds kTopKFastMaxK "
                                                     << kTopKFastMaxK);
            return {};
        }
        if (prompt_len < 0) {
            prompt_len = len;
        } else if (len != prompt_len) {
            MINI_TRT_LOG_ERROR("LLMRunner: batch 内 prompt 长度不等（第 0 条 "
                               << prompt_len << " vs 第 " << b << " 条 " << len
                               << "）—— S1 只支持等长批（D2=A）");
            return {};
        }
        max_new = std::max(max_new, r.options.max_new_tokens);
    }
    if (prompt_len > config_.max_blocks_per_seq * config_.block_size) {
        MINI_TRT_LOG_ERROR("LLMRunner: prompt longer than the engine's block table capacity");
        return {};
    }

    // seq_id：<0 由 runner 分配为批内下标；显式指定的必须批内唯一。
    std::vector<int32_t> seq_ids(static_cast<size_t>(batch));
    for (int32_t b = 0; b < batch; ++b) {
        const int32_t id = requests[b].seq_id < 0 ? b : requests[b].seq_id;
        for (int32_t prev = 0; prev < b; ++prev) {
            if (seq_ids[static_cast<size_t>(prev)] == id) {
                MINI_TRT_LOG_ERROR("LLMRunner: duplicate seq_id " << id << " in batch");
                return {};
            }
        }
        seq_ids[static_cast<size_t>(b)] = id;
    }

    if (!ReserveBuffers(prompt_len, max_new, batch)) {
        return {};
    }

    // 采样参数：top_k / top_p / seed 都逐行上传（seed 决定随机流，且行号不进随机流）。
    options_top_k_ = first.top_k;
    options_top_p_ = first.top_p;
    options_seed_ = first.seed;
    {
        std::vector<int32_t> top_k(static_cast<size_t>(batch));
        std::vector<float> top_p(static_cast<size_t>(batch));
        std::vector<uint64_t> seeds(static_cast<size_t>(batch));
        for (int32_t b = 0; b < batch; ++b) {
            top_k[static_cast<size_t>(b)] = requests[b].options.top_k;
            top_p[static_cast<size_t>(b)] = requests[b].options.top_p;
            seeds[static_cast<size_t>(b)] = requests[b].options.seed;
        }
        if (cudaMemcpyAsync(d_top_k_.data(), top_k.data(), top_k.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
            cudaMemcpyAsync(d_top_p_.data(), top_p.data(), top_p.size() * sizeof(float),
                            cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
            cudaMemcpyAsync(d_seeds_.data(), seeds.data(), seeds.size() * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, nullptr) != cudaSuccess) {
            return {};
        }
    }

    // ---- 2. 登记序列：按请求顺序 → 引擎第 i 行就是第 i 条请求 ----
    // D9 的完整形态：**登记之前**把总块需求算清并与空闲量比，不足则整批拒绝并报出两个数。
    // 为什么要在登记前算：AllocateSequence 是先查后分配的，但它只知道自己那一条的需求；
    // 到第 k 条才发现不够时前 k-1 条已经占了块——守卫会归还，但错误信息说不清全局。
    int64_t blocks_needed = 0;
    for (int32_t b = 0; b < batch; ++b) {
        const int64_t tokens = prompt_len + requests[b].options.max_new_tokens;
        blocks_needed += (tokens + config_.block_size - 1) / config_.block_size;
    }
    const int32_t blocks_free = kv_cache_->NumFreeBlocks();
    if (blocks_needed > blocks_free) {
        MINI_TRT_LOG_ERROR("LLMRunner: not enough KV cache blocks —— 需要 " << blocks_needed
                           << " 块，空闲 " << blocks_free << " 块（batch = " << batch << "）");
        return {};
    }

    // 本批已登记的序列由作用域守卫统一归还：**任何出口都归还**（正常结束与各条失败路径共用
    // 一条路径）。AC3 要求"跑 N 轮后空闲块回到初始水位"，靠人记得在每个出口归还早晚会漏。
    SequenceScope scope(kv_cache_);
    for (int32_t b = 0; b < batch; ++b) {
        if (!kv_cache_->AllocateSequence(seq_ids[static_cast<size_t>(b)], prompt_len + max_new)) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate KV blocks for request " << b);
            return {};
        }
        scope.Add(seq_ids[static_cast<size_t>(b)]);
    }

    // design.md §不变量 4（行号同源）：下列五处**必须共享同一个行号**，任何一处另立下标，
    // 都会让"形状看起来对"配上错行的数据——而这类错通常不报错，只让结果悄悄不对：
    //   ① 分页 cache 的批内顺序（= 登记顺序，决定 block_tables 第 i 行与 context_lens[i]）
    //   ② d_top_k_[i] / d_top_p_[i] / d_seeds_[i]（逐行采样参数）
    //   ③ 引擎输出 logits 的第 i 行（prefill 的 [B,S,V] 与 decode 的 [B,V]）
    //   ④ d_tokens_ 第 i 步的第 i 个元素（步优先布局 [max_new, B_max]）
    //   ⑤ 结果 results[i] ↔ seq_ids[i]
    // 前件是"登记顺序 == 请求顺序"，所以在动元数据之前显式校一次。
    {
        const std::vector<int32_t>& registered = scope.ids();
        if (registered.size() != static_cast<size_t>(batch)) {
            MINI_TRT_LOG_ERROR("LLMRunner: batch bookkeeping mismatch (" << registered.size()
                               << " allocated vs " << batch << " requested)");
            return {};
        }
        for (size_t b = 0; b < registered.size(); ++b) {
            if (registered[b] != seq_ids[b]) {
                MINI_TRT_LOG_ERROR("LLMRunner: row index mismatch at " << b
                                   << "（登记顺序必须等于请求顺序）");
                return {};
            }
        }
    }

    if (kv_cache_->UploadMetadata(nullptr) != cudaSuccess) {

        return {};
    }

    // ---- 3. Prefill：整批一次前向 ----
    std::vector<int32_t> flat;
    flat.reserve(static_cast<size_t>(batch) * static_cast<size_t>(prompt_len));
    for (int32_t b = 0; b < batch; ++b) {
        for (int64_t token : requests[b].input_ids) {
            flat.push_back(static_cast<int32_t>(token));
        }
    }
    if (!BindPrefill(flat, batch, prompt_len, prefill_lengths) ||
        !prefill_engine_->Enqueue(nullptr)) {
        MINI_TRT_LOG_ERROR("LLMRunner: prefill failed");

        return {};
    }
    prefill_engine_->Synchronize(nullptr);  // 每个请求一次，代价可接受

    // ---- 4. 把 prompt 的 K/V 写进分页 cache ----
    // 静态批的批内顺序 == 登记顺序 == 请求顺序（上面显式校过），所以这里传**恒等映射**；
    // 但**仍然显式传**：WritePrefillKV 的契约要求带映射，S3 的活跃批下 B_new 小于已登记序列数，
    // 少了映射就会拿源缓冲里上一轮的残留行去覆盖别的序列自己的 prompt K/V。
    // 长度同样逐行给：S1 批内等长，所以每行的真实长度就是 prompt_len（等于 tokens）。
    std::vector<int32_t> prefill_rows(static_cast<size_t>(batch));
    std::vector<int32_t> prefill_lengths(static_cast<size_t>(batch));
    for (int32_t b = 0; b < batch; ++b) {
        prefill_rows[static_cast<size_t>(b)] = b;
        prefill_lengths[static_cast<size_t>(b)] = prompt_len;
    }
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        if (kv_cache_->WritePrefillKV(
                layer, d_prefill_kv_[static_cast<size_t>(layer) * 2]->data(),
                d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data(), prompt_len,
                prefill_rows.data(), batch, prefill_lengths.data(), nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to write prefill K/V for layer " << layer);

            return {};
        }
    }

    // 常驻诊断（D7：默认关闭；P4/P7 一律关闭）。批量下只报第 0 条——
    // 逐条全量扫描会把每请求的同步 D2H 放大 B 倍，正是 D7 要避免的。
    if (config_.enable_diagnostics) {
        const void* row_ptr = PrefillLogitsRow(0, prompt_len, prompt_len - 1);
        const size_t elem = ElementSize(prefill_logits_half_);
        std::vector<char> raw(static_cast<size_t>(config_.vocab_size) * elem);
        if (row_ptr != nullptr &&
            cudaMemcpy(raw.data(), row_ptr, raw.size(), cudaMemcpyDeviceToHost) == cudaSuccess) {
            double sum = 0.0;
            float max_value = -1e30f;
            float first4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            bool has_nan = false;
            for (int32_t i = 0; i < config_.vocab_size; ++i) {
                const float v = prefill_logits_half_
                                    ? __half2float(reinterpret_cast<const __half*>(raw.data())[i])
                                    : reinterpret_cast<const float*>(raw.data())[i];
                if (std::isnan(v) || std::isinf(v)) {
                    has_nan = true;
                }
                if (i < 4) {
                    first4[i] = v;
                }
                max_value = std::max(max_value, v);
                sum += v;
            }
            MINI_TRT_LOG_INFO("LLMRunner 诊断（第 0 条）：prefill 末行 logits 前 4 个 = "
                              << first4[0] << ", " << first4[1] << ", " << first4[2] << ", "
                              << first4[3] << "；max = " << max_value
                              << "；mean = " << sum / config_.vocab_size
                              << "；含 NaN/Inf = " << (has_nan ? "是" : "否"));
        }
    }

    // 采样器要求 logits 连续 [batch, vocab]，而 prefill 的末行在 [B,S,V] 里是跨步的：
    // 先收集成 [B, V]（每请求一次，不在解码循环内）。
    {
        const size_t elem = ElementSize(prefill_logits_half_);
        const size_t row_bytes = static_cast<size_t>(config_.vocab_size) * elem;
        for (int32_t b = 0; b < batch; ++b) {
            const void* src = PrefillLogitsRow(b, prompt_len, prompt_len - 1);
            void* dst = static_cast<char*>(d_prefill_last_logits_.data()) +
                        static_cast<size_t>(b) * row_bytes;
            if (src == nullptr ||
                cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyDeviceToDevice, nullptr) !=
                    cudaSuccess) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to gather prefill logits rows");

                return {};
            }
        }
    }
    // 采样结果按"步"排列：[max_new, batch_capacity_]，第 i 步的 B 个 token 连续，
    // 采样器可以直接写；解码下一步的输入就是上一步的那 B 个元素。
    if (!SampleBatch(static_cast<char*>(d_tokens_.data()), /*from_prefill=*/true, batch,
                     /*offset=*/0, /*row_offsets=*/nullptr, /*eos_hit=*/nullptr, nullptr)) {
        MINI_TRT_LOG_ERROR("LLMRunner: sampling after prefill failed");

        return {};
    }

    // ---- 5. Decode 循环（循环内零 H2D/D2H）----
    // 不变量（进入第 i 次迭代时）：cache 里有各序列 0..S0+i-2 的 K/V，
    // context_lens[b] = S0+i-1；d_tokens_ 第 i-1 步的 B 个元素就是本次要喂进去的 token。
    for (int32_t i = 1; i < max_new; ++i) {
        const int32_t* input_tokens =
            static_cast<const int32_t*>(d_tokens_.data()) +
            static_cast<size_t>(i - 1) * static_cast<size_t>(batch_capacity_);
        if (!BindDecode(input_tokens, batch)) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to bind decode inputs at step " << i);

            return {};
        }
        // position_ids 由设备端的 context_lens 填，避免为整数回主机。
        if (LaunchFillPositionIds(kv_cache_->context_lens(),
                                  static_cast<int32_t*>(d_position_.data()), batch,
                                  nullptr) != cudaSuccess ||
            !decode_engine_->Enqueue(nullptr)) {
            MINI_TRT_LOG_ERROR("LLMRunner: decode failed at step " << i);

            return {};
        }
        std::vector<const void*> keys(static_cast<size_t>(config_.num_layers));
        std::vector<const void*> values(static_cast<size_t>(config_.num_layers));
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            keys[static_cast<size_t>(layer)] =
                d_decode_kv_[static_cast<size_t>(layer) * 2]->data();
            values[static_cast<size_t>(layer)] =
                d_decode_kv_[static_cast<size_t>(layer) * 2 + 1]->data();
        }
        // 整步只推进一次语境长度（逐层推进会被算成 n_layer 倍）。
        if (kv_cache_->AppendDecodeStep(keys, values, batch, nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to append decode K/V at step " << i);

            return {};
        }
        if (!SampleBatch(static_cast<char*>(d_tokens_.data()) +
                             static_cast<size_t>(i) * static_cast<size_t>(batch_capacity_) *
                                 sizeof(int32_t),
                         /*from_prefill=*/false, batch, static_cast<uint64_t>(i),
                         /*row_offsets=*/nullptr, /*eos_hit=*/nullptr, nullptr)) {
            MINI_TRT_LOG_ERROR("LLMRunner: sampling failed at step " << i);

            return {};
        }
    }

    // ---- 6. 一次性取回并切分 ----
    const size_t total =
        static_cast<size_t>(batch_capacity_) * static_cast<size_t>(max_new);
    std::vector<int32_t> raw(total);
    if (cudaMemcpyAsync(raw.data(), d_tokens_.data(), total * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, nullptr) != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to fetch generated tokens");

        return {};
    }

    // EOS 截断放在循环之后（循环内同步会带来 host 往返）。
    // 每条按自己的 max_new_tokens 截断 —— 静态批一起跑满 max，结果各取所需。
    std::vector<GenerateResult> results(static_cast<size_t>(batch));
    for (int32_t b = 0; b < batch; ++b) {
        const int32_t want = requests[b].options.max_new_tokens;
        std::vector<int64_t> tokens;
        for (int32_t i = 0; i < want; ++i) {
            const int32_t token =
                raw[static_cast<size_t>(i) * static_cast<size_t>(batch_capacity_) +
                    static_cast<size_t>(b)];
            if (config_.eos_token_id >= 0 && token == config_.eos_token_id) {
                break;
            }
            tokens.push_back(static_cast<int64_t>(token));
        }
        results[static_cast<size_t>(b)].seq_id = seq_ids[static_cast<size_t>(b)];
        results[static_cast<size_t>(b)].tokens = std::move(tokens);
        results[static_cast<size_t>(b)].ok = !results[static_cast<size_t>(b)].tokens.empty();
    }
    return results;
}

std::vector<int64_t> LLMRunner::Generate(const std::vector<int64_t>& input_ids,
                                         const GenerateOptions& options) {
    // 单序列 = 单元素批：AC5（批量 = 1 时语义不变）由结构保证，而不是靠"小心别改坏"。
    GenerateRequest request;
    request.input_ids = input_ids;
    request.options = options;
    request.seq_id = 0;
    const std::vector<GenerateResult> results = GenerateBatch({request});
    if (results.empty() || !results.front().ok) {
        return {};
    }
    return results.front().tokens;
}

// S3：请求级调度（活跃批 + padding mask）。五步循环见 p5_s3_interface_spec.md §2：
//   ① retire（读上一步异步回读的 finish flag / 到达 max_new → 释放块 + 压实行号）
//   ② admit （按 (arrival_step, 请求下标) 准入，先查空闲块再分配，D9）
//   ③ context 段（只装本步新入批的行，[B_new, S_step]，padding_bias 按真实长度填）
//   ④ generation 段（只装本步在跑的活跃表前缀，[B_active, 1]）
//   ⑤ 采样并交回 finish flag
// **核心：每次引擎调用只装一种相** —— 让 context 段按整批跑会把正在 generation 的行也算一遍、
// 写回时覆盖它们自己的 prompt K/V（D12 的静默错）。
std::vector<LLMRunner::GenerateResult> LLMRunner::RunScheduler(
    const std::vector<SchedulerRequest>& requests) {
    // 观测口每次调用从零开始：入口校验失败时也保持全 0（"上一次的统计"不会骗人）。
    scheduler_stats_ = SchedulerStats{};
    if (!valid_) {
        MINI_TRT_LOG_ERROR("LLMRunner::RunScheduler called on an invalid runner");
        return {};
    }
    // S4：packed 模式下只用 `prefill_engine_`（调用方把那个 packed 引擎放在这个槽位）。
    // 显式记一条，免得有人以为"两条引擎都在跑"。
    if (config_.prefill_mode == Config::PrefillMode::kPackedMixed) {
        MINI_TRT_LOG_INFO("LLMRunner: packed mixed prefill（每步一次调用，只用 prefill_engine_）");
    }
    const int32_t request_count = static_cast<int32_t>(requests.size());
    if (request_count <= 0) {
        MINI_TRT_LOG_ERROR("LLMRunner: empty scheduler request set");
        return {};
    }

    // ---- 1. 入口校验：与 GenerateBatch 同一套契约，但**不要求 prompt 等长**（D2=B 的 padding 路径）----
    const GenerateOptions& first = requests[0].request.options;
    const bool first_top_p = first.top_p < 1.0f;
    const bool first_topk = !first_top_p && first.top_k > 1;
    int32_t max_prompt = 0;
    int32_t max_new = 0;
    int32_t max_arrival = 0;
    for (int32_t i = 0; i < request_count; ++i) {
        const GenerateRequest& r = requests[static_cast<size_t>(i)].request;
        const int32_t len = static_cast<int32_t>(r.input_ids.size());
        if (len <= 0 || r.options.max_new_tokens <= 0) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << i
                                                     << " has an empty prompt or max_new_tokens <= 0");
            return {};
        }
        if (r.options.temperature != 1.0f) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << i
                                                     << " uses temperature != 1.0 (unsupported)");
            return {};
        }
        if (r.options.top_k < 1 || r.options.top_p <= 0.0f || r.options.top_p > 1.0f) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << i << " has invalid sampling parameters");
            return {};
        }
        // 策略必须整批一致（同 GenerateBatch）：三种采样 kernel 各自是"整批一个分支"。
        const bool row_top_p = r.options.top_p < 1.0f;
        const bool row_topk = !row_top_p && r.options.top_k > 1;
        if (row_top_p != first_top_p || row_topk != first_topk) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << i
                               << " uses a different sampling strategy than request 0"
                                  " —— 调度同样要求同一种策略（D3 的逐行策略需要改采样器）");
            return {};
        }
        if (first_topk && r.options.top_k > kTopKFastMaxK) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << i << " top_k " << r.options.top_k
                                                     << " exceeds kTopKFastMaxK "
                                                     << kTopKFastMaxK);
            return {};
        }
        if (requests[static_cast<size_t>(i)].arrival_step < 0) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << i << " has a negative arrival_step");
            return {};
        }
        max_prompt = std::max(max_prompt, len);
        max_new = std::max(max_new, r.options.max_new_tokens);
        max_arrival = std::max(max_arrival, requests[static_cast<size_t>(i)].arrival_step);
    }

    // seq_id：<0 → 请求下标；显式指定的必须唯一（回填按请求下标，seq_id 只用于块池与日志）。
    std::vector<int32_t> seq_ids(static_cast<size_t>(request_count));
    for (int32_t i = 0; i < request_count; ++i) {
        const int32_t id =
            requests[static_cast<size_t>(i)].request.seq_id < 0
                ? i
                : requests[static_cast<size_t>(i)].request.seq_id;
        for (int32_t prev = 0; prev < i; ++prev) {
            if (seq_ids[static_cast<size_t>(prev)] == id) {
                MINI_TRT_LOG_ERROR("LLMRunner: duplicate seq_id " << id << " in scheduler set");
                return {};
            }
        }
        seq_ids[static_cast<size_t>(i)] = id;
    }

    // 池**任何时候**都装不下某条请求 → 永远服务不了：入口直接拒绝。
    // 不拒的话调度循环会在"准入失败"上无限打转（块永远不够）。
    for (int32_t i = 0; i < request_count; ++i) {
        const int32_t len = static_cast<int32_t>(
            requests[static_cast<size_t>(i)].request.input_ids.size());
        const int32_t want = requests[static_cast<size_t>(i)].request.options.max_new_tokens;
        // stride 的上界是 max_prompt（本步最长 prompt）；预留按 stride 算（填充位置也写进 cache）。
        const int32_t need_tokens = std::max(max_prompt, len + want);
        const int64_t need_blocks =
            (need_tokens + config_.block_size - 1) / config_.block_size;
        if (need_blocks > config_.max_blocks_per_seq ||
            need_blocks * config_.block_size >
                static_cast<int64_t>(config_.num_blocks) * config_.block_size) {
            MINI_TRT_LOG_ERROR("LLMRunner: request " << i << " can never fit the KV pool (needs "
                               << need_tokens << " tokens per sequence)");
            return {};
        }
    }

    // 准入顺序：(arrival_step, 请求下标) 的字典序 —— 确定性（p5_s3_interface_spec §5）。
    std::vector<int32_t> waiting(static_cast<size_t>(request_count));
    for (int32_t i = 0; i < request_count; ++i) {
        waiting[static_cast<size_t>(i)] = i;
    }
    std::sort(waiting.begin(), waiting.end(), [&requests](int32_t a, int32_t b) {
        const int32_t step_a = requests[static_cast<size_t>(a)].arrival_step;
        const int32_t step_b = requests[static_cast<size_t>(b)].arrival_step;
        if (step_a != step_b) {
            return step_a < step_b;
        }
        return a < b;
    });

    // ---- 2. 缓冲与逐行状态：全部在循环外备好（循环内不分配、不同步）----
    const int32_t active_capacity = std::min(request_count, config_.max_batch);
    if (!ReserveBuffers(max_prompt, max_new, active_capacity)) {
        return {};
    }
    const size_t capacity_sz = static_cast<size_t>(active_capacity);
    const bool packed_mode = config_.prefill_mode == Config::PrefillMode::kPackedMixed;
    if (!d_step_tokens_.Allocate(capacity_sz * sizeof(int32_t)) ||
        !d_decode_input_.Allocate(capacity_sz * sizeof(int32_t)) ||
        !d_offsets_.Allocate(capacity_sz * sizeof(uint64_t)) ||
        !d_eos_hit_.Allocate(capacity_sz * sizeof(int8_t)) || !host_eos_.Allocate(capacity_sz) ||
        !d_result_tokens_.Allocate(static_cast<size_t>(request_count) *
                                   static_cast<size_t>(max_new) * sizeof(int32_t))) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate scheduler buffers");
        return {};
    }
    // S4 的 packed 缓冲：T 的上界 = batch_capacity_ × prompt_capacity_（与 d_prompt_ 同一个界）
    if (packed_mode) {
        const size_t packed_tokens = capacity_sz * static_cast<size_t>(prompt_capacity_);
        if (!d_packed_tokens_.Allocate(packed_tokens * sizeof(int32_t)) ||
            !d_packed_positions_.Allocate(packed_tokens * sizeof(int32_t)) ||
            !d_cu_seqlens_ctx_.Allocate((capacity_sz + 1) * sizeof(int32_t)) ||
            !d_context_seq_count_.Allocate(sizeof(int32_t)) ||
            !d_packed_block_tables_.Allocate(capacity_sz *
                                             static_cast<size_t>(config_.max_blocks_per_seq) *
                                             sizeof(int32_t)) ||
            !d_packed_context_lens_.Allocate(capacity_sz * sizeof(int32_t))) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate packed buffers");
            return {};
        }
    }
    // 采样策略整批一致（上面校验过），SampleBatch 靠这三个成员选分支。
    options_top_k_ = first.top_k;
    options_top_p_ = first.top_p;
    options_seed_ = first.seed;

    std::vector<GenerateResult> results(static_cast<size_t>(request_count));
    std::vector<int32_t> token_counts(static_cast<size_t>(request_count), 0);
    std::vector<ActiveSequence> active;
    active.reserve(capacity_sz);
    // 块归还的作用域守卫：任何出口都归还（正常路径由 retire 归还；失败路径靠它兜底）。
    // 重复归还是安全的（FreeSequence 对未知 seq_id 直接返回）。
    SequenceScope scope(kv_cache_.get());
    std::vector<const void*> keys(static_cast<size_t>(config_.num_layers));
    std::vector<const void*> values(static_cast<size_t>(config_.num_layers));
    int32_t next_waiting = 0;
    int32_t step = 0;
    // finish flag 的异步回读：不在循环里等，连续 kMaxEosPendingSteps 步没落地才强制同步一次。
    bool eos_pending = false;
    std::vector<int32_t> eos_pending_seqs;
    int32_t eos_pending_steps = 0;
    constexpr int32_t kMaxEosPendingSteps = 4;
    // 防呆上界：正常每步都至少推进一件事；超了说明有 bug，宁可报错也不要挂住。
    // S5 的分块让"一条 prompt 分多步送"成为正常路径（每步最多推进 chunk_limit 个 prompt token，
    // 而 chunk_limit 最小为 1 = 逐 token 一步），所以分块步数最多 ceil(prompt_len / chunk_limit)。
    // 上界里必须把这部分算进去，否则"长 prompt + 小 chunk_limit"会撞上限，把正常结果误报成 bug。
    // 取 Σ ceil(prompt_len_i / chunk_limit) ≤ max_prompt × request_count（松上界，与 chunk_limit
    // 取值无关、永远够用）；分块只在 packed 路径发生（S3 两段式一步就送完整条 prompt），所以只在
    // packed 下放宽 —— 保住 S3 那条对死循环的敏感度。
    const int64_t chunk_step_slack =
        packed_mode ? static_cast<int64_t>(max_prompt) * request_count : 0;
    const int64_t step_limit = static_cast<int64_t>(max_arrival) +
                               static_cast<int64_t>(max_new) * request_count + request_count + 4 +
                               chunk_step_slack;

    while (!active.empty() || next_waiting < request_count) {
        if (active.empty() && next_waiting < request_count) {
            // 没有活跃序列：时钟跳到下一个到达步（步号只影响准入时刻，不影响结果）
            step = std::max(step, requests[static_cast<size_t>(waiting[static_cast<size_t>(next_waiting)])]
                                      .arrival_step);
        }

        // ---- ① retire ----
        if (eos_pending) {
            const cudaError_t query = cudaStreamQuery(nullptr);
            bool landed = (query == cudaSuccess);
            if (query != cudaSuccess && query != cudaErrorNotReady) {
                MINI_TRT_LOG_ERROR("LLMRunner: finish-flag readback failed: "
                                   << cudaGetErrorString(query));
                return {};
            }
            if (!landed && ++eos_pending_steps >= kMaxEosPendingSteps) {
                // 兜底：连续几步都没落地就强制同步一次（退化也只是"晚一步退出"，功能不受影响）
                if (cudaStreamSynchronize(nullptr) == cudaSuccess) {
                    landed = true;
                }
            }
            if (landed) {
                const int8_t* flags = static_cast<const int8_t*>(host_eos_.data());
                for (size_t i = 0; i < eos_pending_seqs.size(); ++i) {
                    if (flags[i] == 0) {
                        continue;
                    }
                    for (ActiveSequence& row : active) {
                        if (row.seq_id == eos_pending_seqs[i]) {
                            row.finished = true;  // EOS：下一步不必再喂它
                        }
                    }
                }
                eos_pending = false;
                eos_pending_steps = 0;
                eos_pending_seqs.clear();
            }
        }
        for (int32_t row = 0; row < static_cast<int32_t>(active.size());) {
            const ActiveSequence& s = active[static_cast<size_t>(row)];
            if (!s.finished && s.generated < s.max_new) {
                ++row;
                continue;
            }
            token_counts[static_cast<size_t>(s.result_slot)] = s.generated;
            results[static_cast<size_t>(s.result_slot)].seq_id = s.seq_id;
            kv_cache_->FreeSequence(s.seq_id);   // 释放块 + 压实行号（S2 的能力）
            active.erase(active.begin() + row);  // 活跃表行号随之压实（不变量 4）
        }
        // 退出并压实之后剩下的行**全部**处于 generation 相：它们就是本步生成段的前缀。
        // 必须在 admit 之前取，admit 会把新入批的行追加到它后面。
        const int32_t generation_rows = static_cast<int32_t>(active.size());

        // ---- ② admit：候选 → 用**最终 stride** 一次算清预算（D9）----
        const int32_t free_slots =
            active_capacity - static_cast<int32_t>(active.size());
        int32_t admit_count = 0;
        while (admit_count < free_slots && next_waiting + admit_count < request_count &&
               requests[static_cast<size_t>(waiting[static_cast<size_t>(next_waiting + admit_count)])]
                       .arrival_step <= step) {
            ++admit_count;
        }
        const auto row_len = [&](int32_t j) {
            return static_cast<int32_t>(
                requests[static_cast<size_t>(waiting[static_cast<size_t>(next_waiting + j)])]
                    .request.input_ids.size());
        };
        int32_t stride = 0;
        // 预算不足就退回"到达最晚"的那条重算 —— stride 会随之变小，所以必须整体重算而不是逐条判。
        while (admit_count > 0) {
            stride = 0;
            for (int32_t j = 0; j < admit_count; ++j) {
                stride = std::max(stride, row_len(j));
            }
            int64_t need_blocks = 0;
            for (int32_t j = 0; j < admit_count; ++j) {
                const int32_t idx = waiting[static_cast<size_t>(next_waiting + j)];
                const int32_t len = static_cast<int32_t>(
                    requests[static_cast<size_t>(idx)].request.input_ids.size());
                const int32_t want =
                    requests[static_cast<size_t>(idx)].request.options.max_new_tokens;
                const int32_t need_tokens = std::max(stride, len + want);
                need_blocks += (need_tokens + config_.block_size - 1) / config_.block_size;
            }
            if (need_blocks <= kv_cache_->NumFreeBlocks()) {
                break;
            }
            --admit_count;  // 块不够 → 留在等待队列（D9），本步少接一条
        }
        for (int32_t j = 0; j < admit_count; ++j) {
            const int32_t idx = waiting[static_cast<size_t>(next_waiting + j)];
            const GenerateRequest& r = requests[static_cast<size_t>(idx)].request;
            const int32_t len = static_cast<int32_t>(r.input_ids.size());
            // 预留按 stride 算：填充位置的 K/V 也会写进 cache（p5_s3_interface_spec §3）
            const int32_t reserve_tokens = std::max(stride, len + r.options.max_new_tokens);
            if (!kv_cache_->AllocateSequence(seq_ids[static_cast<size_t>(idx)], reserve_tokens)) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate KV blocks for request " << idx);
                return {};
            }
            scope.Add(seq_ids[static_cast<size_t>(idx)]);
            ActiveSequence s;
            s.seq_id = seq_ids[static_cast<size_t>(idx)];
            s.result_slot = idx;
            s.prompt_len = len;
            s.max_new = r.options.max_new_tokens;
            s.top_k = r.options.top_k;
            s.top_p = r.options.top_p;
            s.seed = r.options.seed;
            active.push_back(s);
        }
        next_waiting += admit_count;
        scheduler_stats_.max_active =
            std::max(scheduler_stats_.max_active, static_cast<int32_t>(active.size()));
        // 跨路径口径：本步"在跑"的行数（= 生成段的行数）。S3 两段式下它等于 decode_calls 的规模，
        // 但 S4 每步只有一次调用，只有这个量还有意义 —— 见 hpp 里 SchedulerStats 的说明。
        // S5：packed 路径的"喂了几行 generation token"由 `RunPackedMixedStep` 精确给出
        // （它才知道哪几行已完成 prefill），所以这里只在 S3 两段式下记账。
        if (!packed_mode) {
            scheduler_stats_.generation_rows += generation_rows;
        }

        // 块表 / 语境长度每步重建并上传：登记与退出都改了行号（不变量 4 的"同源"就靠这一步）。
        if (kv_cache_->UploadMetadata(nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to upload KV metadata");
            return {};
        }

        if (packed_mode) {
            // **S4**：两相装进一个 packed 张量、每步一次调用（见 RunPackedMixedStep）。
            // 统计只填**跨路径**的两个量（`prefill_calls` / `decode_calls` 是 S3 两段式专有，
            // S4 下不读也不去凑语义 —— 见 SchedulerStats 的字段说明）。
            if (!RunPackedMixedStep(requests, active, generation_rows, new_rows, max_new)) {
                MINI_TRT_LOG_ERROR("LLMRunner: packed mixed step failed");
                return {};
            }
            // S5：两段行数由 `RunPackedMixedStep` 按 cache 已写入长度算出（不再按活跃表前缀）；
            // `context_rows` 的新口径 = Σ 本步拿到 chunk 的行数（同一序列会跨多步出现）。
            scheduler_stats_.context_rows += packed_context_rows_;
            scheduler_stats_.generation_rows += packed_generation_rows_;
        } else {
        // ---- ③ context 段：只装本步新入批的行 ----
        const int32_t new_rows = static_cast<int32_t>(active.size()) - generation_rows;
        if (new_rows > 0) {
            int32_t s_step = 0;
            for (int32_t j = generation_rows; j < static_cast<int32_t>(active.size()); ++j) {
                s_step = std::max(s_step, active[static_cast<size_t>(j)].prompt_len);
            }
            const size_t step_sz = static_cast<size_t>(s_step);
            std::vector<int32_t> flat(static_cast<size_t>(new_rows) * step_sz);
            std::vector<int32_t> lengths(static_cast<size_t>(new_rows));
            std::vector<int32_t> rows(static_cast<size_t>(new_rows));
            for (int32_t j = 0; j < new_rows; ++j) {
                const ActiveSequence& s = active[static_cast<size_t>(generation_rows + j)];
                const GenerateRequest& r = requests[static_cast<size_t>(s.result_slot)].request;
                const int32_t row = kv_cache_->RowOf(s.seq_id);
                // 不变量 4：引擎第 j 行必须就是缓存批的第 (generation_rows + j) 行。
                // 不显式校的话，映射错位只会表现为"结果悄悄不对"，而不是报错。
                if (row != generation_rows + j) {
                    MINI_TRT_LOG_ERROR("LLMRunner: seq " << s.seq_id << " sits at KV row " << row
                                                         << ", expected " << (generation_rows + j)
                                                         << " —— 行号不同源");
                    return {};
                }
                rows[static_cast<size_t>(j)] = row;
                lengths[static_cast<size_t>(j)] = s.prompt_len;
                for (int32_t t = 0; t < s_step; ++t) {
                    // 右填充：真实 token 之后填 0。填充位置的 K/V 会被写进 cache，但它们
                    // 不在语境长度内 → 从不参与注意力，且会被后续 decode 覆盖。
                    flat[static_cast<size_t>(j) * step_sz + static_cast<size_t>(t)] =
                        t < s.prompt_len
                            ? static_cast<int32_t>(r.input_ids[static_cast<size_t>(t)])
                            : 0;
                }
            }
            if (!BindPrefill(flat, new_rows, s_step, lengths) ||
                !prefill_engine_->Enqueue(nullptr)) {
                MINI_TRT_LOG_ERROR("LLMRunner: context segment failed");
                return {};
            }
            // 观测口：Σ B_new 与 context 段调用次数 —— "只装新入批的行"就靠这两个量从结果侧锁定
            // （真按整批跑，context_rows 会大于请求总数）。
            scheduler_stats_.context_rows += new_rows;
            scheduler_stats_.prefill_calls += 1;
            // 末位 logits 要取**每行自己的真实末位**（S_step-1 是填充位置，那里的 logits 无意义）
            const size_t elem = ElementSize(prefill_logits_half_);
            const size_t row_bytes = static_cast<size_t>(config_.vocab_size) * elem;
            for (int32_t j = 0; j < new_rows; ++j) {
                const void* src = PrefillLogitsRow(j, s_step,
                                                   lengths[static_cast<size_t>(j)] - 1);
                void* dst = static_cast<char*>(d_prefill_last_logits_.data()) +
                            static_cast<size_t>(j) * row_bytes;
                if (src == nullptr ||
                    cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyDeviceToDevice, nullptr) !=
                        cudaSuccess) {
                    MINI_TRT_LOG_ERROR("LLMRunner: failed to gather context logits rows");
                    return {};
                }
            }
            // 写回 prompt 的 K/V：显式行映射 + 逐行真实长度（p5_s3_interface_spec §3）
            for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
                if (kv_cache_->WritePrefillKV(
                        layer, d_prefill_kv_[static_cast<size_t>(layer) * 2]->data(),
                        d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data(), s_step,
                        rows.data(), new_rows, lengths.data(), nullptr) != cudaSuccess) {
                    MINI_TRT_LOG_ERROR("LLMRunner: failed to write context K/V for layer " << layer);
                    return {};
                }
            }
            // 采第 0 个 token：各行都刚入批，随机步号都是 0（照样走 per-row 数组）
            if (!UploadRowParams(active, generation_rows, new_rows) ||
                !SampleBatch(static_cast<char*>(d_step_tokens_.data()) +
                                 static_cast<size_t>(generation_rows) * sizeof(int32_t),
                             /*from_prefill=*/true, new_rows, /*offset=*/0,
                             static_cast<const uint64_t*>(d_offsets_.data()) + generation_rows,
                             static_cast<int8_t*>(d_eos_hit_.data()) + generation_rows, nullptr)) {
                MINI_TRT_LOG_ERROR("LLMRunner: context sampling failed");
                return {};
            }
        }

        // ---- ④ generation 段：只装本步在跑的活跃表前缀 ----
        if (generation_rows > 0) {
            // 每行的输入 token = 它自己上一步采出的那个 token。token 按**序列**存在结果缓冲里
            // （行号每步都可能变），所以每步按行聚集一次再喂给引擎。
            for (int32_t row = 0; row < generation_rows; ++row) {
                const ActiveSequence& s = active[static_cast<size_t>(row)];
                const int64_t src_index =
                    static_cast<int64_t>(s.result_slot) * max_new + s.generated - 1;
                if (cudaMemcpyAsync(static_cast<int32_t*>(d_decode_input_.data()) + row,
                                    static_cast<const int32_t*>(d_result_tokens_.data()) +
                                        src_index,
                                    sizeof(int32_t), cudaMemcpyDeviceToDevice, nullptr) !=
                    cudaSuccess) {
                    MINI_TRT_LOG_ERROR("LLMRunner: failed to gather generation input tokens");
                    return {};
                }
            }
            if (!BindDecode(static_cast<const int32_t*>(d_decode_input_.data()), generation_rows)) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to bind generation inputs");
                return {};
            }
            scheduler_stats_.decode_calls += 1;
            // position_ids 取自**推进前**的设备端语境长度，所以必须在 AppendDecodeStep 之前
            if (LaunchFillPositionIds(kv_cache_->context_lens(),
                                      static_cast<int32_t*>(d_position_.data()), generation_rows,
                                      nullptr) != cudaSuccess ||
                !decode_engine_->Enqueue(nullptr)) {
                MINI_TRT_LOG_ERROR("LLMRunner: generation segment failed");
                return {};
            }
            for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
                keys[static_cast<size_t>(layer)] =
                    d_decode_kv_[static_cast<size_t>(layer) * 2]->data();
                values[static_cast<size_t>(layer)] =
                    d_decode_kv_[static_cast<size_t>(layer) * 2 + 1]->data();
            }
            // 只追加、只推进本步在跑的那几行：本步刚入批的 context 行还没算出 decode 的 K/V
            if (kv_cache_->AppendDecodeStep(keys, values, generation_rows, nullptr) != cudaSuccess) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to append generation K/V");
                return {};
            }
            if (!UploadRowParams(active, 0, generation_rows) ||
                !SampleBatch(d_step_tokens_.data(), /*from_prefill=*/false, generation_rows,
                             /*offset=*/0, static_cast<const uint64_t*>(d_offsets_.data()),
                             static_cast<int8_t*>(d_eos_hit_.data()), nullptr)) {
                MINI_TRT_LOG_ERROR("LLMRunner: generation sampling failed");
                return {};
            }
        }

        }  // else：S3 的两段式（packed_mode 分支在上面）
        // ---- ⑤ 结果落位 + 交回 finish flag ----
        // S5：采样结果按 `sample_active_indices_` 的**槽位序**写进 `d_step_tokens_`（槽位 = 下标）；
        // 本步没参与采样的行（分块还没完成）**不出 token**，绝不能给它记一个 token ——
        // 那会让"分块"改变可见的生成语义（AC9 / `ChunkProgressStateIsCorrect`）。S3 仍是恒等行号。
        const auto record_token = [&](int32_t row, int32_t slot) -> bool {
            ActiveSequence& s = active[static_cast<size_t>(row)];
            const int64_t dst_index =
                static_cast<int64_t>(s.result_slot) * max_new + s.generated;
            // 每步每行一次 4 字节 D2D：结果按"序列"聚集，退出/压实都不会挪动它
            if (cudaMemcpyAsync(static_cast<int32_t*>(d_result_tokens_.data()) + dst_index,
                                static_cast<const int32_t*>(d_step_tokens_.data()) +
                                    slot,
                                sizeof(int32_t), cudaMemcpyDeviceToDevice, nullptr) !=
                cudaSuccess) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to record the sampled token");
                return false;
            }
            s.generated += 1;
            if (s.generated >= s.max_new) {
                s.finished = true;  // 下一步的 retire 收口
            }
            return true;
        };
        if (packed_mode) {
            for (size_t slot = 0; slot < sample_active_indices_.size(); ++slot) {
                if (!record_token(sample_active_indices_[slot], static_cast<int32_t>(slot))) {
                    return {};
                }
            }
        } else {
            for (int32_t row = 0; row < static_cast<int32_t>(active.size()); ++row) {
                if (!record_token(row, row)) {
                    return {};
                }
            }
        }
        // 读回 finish flag：**最多一个在飞**。上一步的还没落地就不再发新的 ——
        // 否则两次 D2H 会同时往同一块 pinned 缓冲里写（数据竞争），而 eos_pending_seqs
        // 也只对应当前这块内容。推迟期间退出判定只晚一步（兜底同步见 retire）。
        if (!active.empty() && !eos_pending) {
            eos_pending_seqs.clear();
            if (packed_mode) {
                // 与结果落位同一张表：`d_eos_hit_` 是按**采样槽位序**写的（采样器按 count 续写）。
                for (int32_t row_index : sample_active_indices_) {
                    eos_pending_seqs.push_back(active[static_cast<size_t>(row_index)].seq_id);
                }
            } else {
                for (const ActiveSequence& s : active) {
                    eos_pending_seqs.push_back(s.seq_id);
                }
            }
            const size_t flag_count =
                packed_mode ? sample_active_indices_.size() : active.size();
            if (cudaMemcpyAsync(host_eos_.data(), d_eos_hit_.data(),
                                flag_count * sizeof(int8_t),
                                cudaMemcpyDeviceToHost, nullptr) != cudaSuccess) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to issue the finish-flag readback");
                return {};
            }
            eos_pending = true;
            eos_pending_steps = 0;
        }
        ++step;
        scheduler_stats_.steps += 1;  // 轮次计数：一条序列"提前退出"会让它明显变小
        if (static_cast<int64_t>(step) > step_limit) {
            MINI_TRT_LOG_ERROR("LLMRunner: scheduler step limit (" << step_limit << ") exceeded");
            return {};
        }
    }

    // ---- 3. 一次性取回并切分（循环外）：与 S1 同一口径 —— 结果不含末尾的 EOS ----
    const size_t total = static_cast<size_t>(request_count) * static_cast<size_t>(max_new);
    std::vector<int32_t> raw(total, 0);
    if (cudaMemcpyAsync(raw.data(), d_result_tokens_.data(), total * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, nullptr) != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to fetch scheduled tokens");
        return {};
    }
    for (int32_t i = 0; i < request_count; ++i) {
        const int32_t want = token_counts[static_cast<size_t>(i)];
        std::vector<int64_t> tokens;
        for (int32_t k = 0; k < want; ++k) {
            const int32_t token =
                raw[static_cast<size_t>(i) * static_cast<size_t>(max_new) +
                    static_cast<size_t>(k)];
            if (config_.eos_token_id >= 0 && token == config_.eos_token_id) {
                break;
            }
            tokens.push_back(static_cast<int64_t>(token));
        }
        results[static_cast<size_t>(i)].seq_id = seq_ids[static_cast<size_t>(i)];
        results[static_cast<size_t>(i)].tokens = std::move(tokens);
        results[static_cast<size_t>(i)].ok = !results[static_cast<size_t>(i)].tokens.empty();
    }
    return results;
}

bool LLMRunner::UploadRowParamsByOrder(const std::vector<ActiveSequence>& active,
                                       const std::vector<int32_t>& active_indices) {
    const int32_t count = static_cast<int32_t>(active_indices.size());
    if (count < 0 || count > batch_capacity_) {
        MINI_TRT_LOG_ERROR("LLMRunner: row params out of range");
        return false;
    }
    if (count == 0) {
        return true;
    }
    std::vector<int32_t> top_k(static_cast<size_t>(count));
    std::vector<float> top_p(static_cast<size_t>(count));
    std::vector<uint64_t> seeds(static_cast<size_t>(count));
    std::vector<uint64_t> offsets(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        const int32_t index = active_indices[static_cast<size_t>(i)];
        if (index < 0 || index >= static_cast<int32_t>(active.size())) {
            MINI_TRT_LOG_ERROR("LLMRunner: row param order out of range");
            return false;
        }
        const ActiveSequence& row = active[static_cast<size_t>(index)];
        top_k[static_cast<size_t>(i)] = row.top_k;
        top_p[static_cast<size_t>(i)] = row.top_p;
        seeds[static_cast<size_t>(i)] = row.seed;
        // 随机步号 = 该行自己的已生成计数（与 S3 同一口径，见 SchedulerStats 与 §5 的说明）
        offsets[static_cast<size_t>(i)] = static_cast<uint64_t>(row.generated);
    }
    if (cudaMemcpyAsync(d_top_k_.data(), top_k.data(), top_k.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_top_p_.data(), top_p.data(), top_p.size() * sizeof(float),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_seeds_.data(), seeds.data(), seeds.size() * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_offsets_.data(), offsets.data(), offsets.size() * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to upload per-row sampling params (packed order)");
        return false;
    }
    return true;
}

// **S4/S5 的一步**：打包 → 一次 packed 调用 → context 写回 / generation 追加 → 采样。
//
// 两套下标必须分清（p5_s4_interface_spec.md §3 / p5_s5_interface_spec.md §2）：
//   * **活跃表序**：行号 = cache 行号（S2 的压实保证同源）；
//   * **packed 行序**：**context 行在前**（作者的硬约束），generation 行在后。
//   两者之间的换算在这里显式写出（`packed_order_active` 与 ⑤ 处的采样行集）。
//
// **S5 起，"哪一行属于哪一段"不看活跃表位置，而看 cache 的已写入长度**：
//   `written = SequenceLength(seq_id)`；`written >= prompt_len` ⇒ 已 prefill 完成（generation 段），
//   否则本步送一个 chunk（context 段），`chunk_len = min(prompt_len - written, chunk_limit_)`。
//   这条分类是**单源**的：不新增 `prompt_done` 字段，也不依赖"完成的行恰好排在前面"
//   —— 那条前提在分块下不成立（长 prompt 分块中、它后面的短 prompt 可能已完成）。
bool LLMRunner::RunPackedMixedStep(const std::vector<GenerateRequest>& requests,
                                   const std::vector<ActiveSequence>& active,
                                   int32_t generation_rows, int32_t new_rows, int32_t max_new) {
    // S5：这两个入参（S4 的"生成段前缀行数 / 新入批行数"）不再决定分段 —— 分段与采样行集都在本函数
    // 内按 cache 已写入长度重算。保留参数只是不改调用点（见 STATE 的 S5-2b 计划）。
    (void)generation_rows;
    (void)new_rows;
    const int32_t b_total = static_cast<int32_t>(active.size());
    if (b_total <= 0) {
        return true;
    }
    const size_t width = static_cast<size_t>(config_.max_blocks_per_seq);

    // ---- ① 分类 + 打包（host 侧算偏移；逐行重建）----
    // 分类必须在**写回之前**做完：写完之后 `SequenceLength` 就变成"已写回"的长度了。
    std::vector<int32_t> context_active;     // 本步要送 chunk 的行（活跃行号，升序）
    std::vector<int32_t> generation_active;  // 本步能喂 token 的行（活跃行号，升序）
    context_active.reserve(static_cast<size_t>(b_total));
    generation_active.reserve(static_cast<size_t>(b_total));
    for (int32_t row = 0; row < b_total; ++row) {
        const ActiveSequence& s = active[static_cast<size_t>(row)];
        const int32_t written = kv_cache_->SequenceLength(s.seq_id);
        if (written < 0) {
            MINI_TRT_LOG_ERROR("LLMRunner: seq " << s.seq_id << " is not in the KV batch");
            return false;
        }
        if (written >= s.prompt_len) {
            generation_active.push_back(row);
        } else {
            context_active.push_back(row);
        }
    }
    packed_context_rows_ = static_cast<int32_t>(context_active.size());
    packed_generation_rows_ = static_cast<int32_t>(generation_active.size());

    int32_t t_ctx = 0;
    for (int32_t active_index : context_active) {
        const ActiveSequence& s = active[static_cast<size_t>(active_index)];
        const int32_t written = kv_cache_->SequenceLength(s.seq_id);
        t_ctx += std::min(s.prompt_len - written, chunk_limit_);
    }
    const int32_t t_total = t_ctx + packed_generation_rows_;
    const size_t packed_capacity =
        static_cast<size_t>(batch_capacity_) * static_cast<size_t>(prompt_capacity_);
    if (static_cast<size_t>(t_total) > packed_capacity) {
        MINI_TRT_LOG_ERROR("LLMRunner: packed token count " << t_total << " exceeds capacity "
                                                            << packed_capacity);
        return false;
    }

    std::vector<int32_t> host_tokens(static_cast<size_t>(t_ctx), 0);
    std::vector<int32_t> host_positions(static_cast<size_t>(t_total), 0);
    std::vector<int32_t> host_cu_seqlens(context_active.size() + 1, 0);
    std::vector<int32_t> host_row_lengths(context_active.size(), 0);
    // S5：每行的写回起点（= prompt_done）。首块为 0，第 2 块起是已写入长度 —— 没有它，
    // 第 2 块会从 0 覆盖写，把自己前一段的 K/V 抹掉（静默错）。
    std::vector<int32_t> host_row_starts(context_active.size(), 0);
    std::vector<int32_t> host_block_tables(static_cast<size_t>(b_total) * width, 0);
    std::vector<int32_t> host_context_lens(static_cast<size_t>(b_total), 0);
    std::vector<int32_t> packed_order_active(static_cast<size_t>(b_total), 0);

    int32_t offset = 0;
    int32_t max_row_len = 0;
    for (size_t j = 0; j < context_active.size(); ++j) {
        const int32_t active_index = context_active[j];
        const ActiveSequence& s = active[static_cast<size_t>(active_index)];
        const GenerateRequest& r = requests[static_cast<size_t>(s.result_slot)];
        const int32_t written = kv_cache_->SequenceLength(s.seq_id);
        const int32_t chunk_len = std::min(s.prompt_len - written, chunk_limit_);
        // 不变量 4：cache 行号必须就是活跃行号。不显式校的话，映射错位只表现为"结果悄悄不对"。
        const int32_t cache_row = kv_cache_->RowOf(s.seq_id);
        if (cache_row != active_index) {
            MINI_TRT_LOG_ERROR("LLMRunner: seq " << s.seq_id << " sits at KV row " << cache_row
                                                 << ", expected " << active_index
                                                 << " —— 行号不同源");
            return false;
        }
        host_row_lengths[j] = chunk_len;
        host_row_starts[j] = written;  // 从 prompt_done 续写（首块为 0）
        max_row_len = std::max(max_row_len, chunk_len);
        for (int32_t i = 0; i < chunk_len; ++i) {
            host_tokens[static_cast<size_t>(offset + i)] =
                static_cast<int32_t>(r.input_ids[static_cast<size_t>(written + i)]);
            // **绝对位置**：第 2 块起不能再用段内下标（否则位置表查错，且不报错）。
            host_positions[static_cast<size_t>(offset + i)] = written + i;
        }
        offset += chunk_len;
        // 段内前缀和：context 段在 packed 张量的**最前面**，所以段内值就是绝对偏移
        host_cu_seqlens[j + 1] = offset;
        packed_order_active[j] = active_index;
    }
    for (size_t j = 0; j < generation_active.size(); ++j) {
        const int32_t active_index = generation_active[j];
        const ActiveSequence& s = active[static_cast<size_t>(active_index)];
        // 位置取**推进前**的语境长度（host 镜像在 S3 已被维护成准确的：WritePrefillKV 设定、
        // AppendDecodeStep 逐行 +1、FreeSequence 重建）
        const int32_t len = kv_cache_->SequenceLength(s.seq_id);
        if (len < 0) {
            MINI_TRT_LOG_ERROR("LLMRunner: seq " << s.seq_id << " is not in the KV batch");
            return false;
        }
        host_positions[static_cast<size_t>(t_ctx + j)] = len;
        packed_order_active[context_active.size() + j] = active_index;
    }
    // 采样行集（**槽位 = 本数组下标**）：已完成的 generation 行 ∪ 本步刚好完成 prefill 的 chunk 行。
    // 按活跃行号升序排一次，让 ⑤ 的 gather 与结果落位、EOS 回读三处共用同一个口径。
    sample_active_indices_.clear();
    for (int32_t active_index : generation_active) {
        sample_active_indices_.push_back(active_index);
    }
    for (size_t j = 0; j < context_active.size(); ++j) {
        const ActiveSequence& s = active[static_cast<size_t>(context_active[j])];
        if (host_row_starts[j] + host_row_lengths[j] >= s.prompt_len) {
            sample_active_indices_.push_back(context_active[j]);
        }
    }
    std::sort(sample_active_indices_.begin(), sample_active_indices_.end());
    // 按 packed 行序重建按行输入：**S3 的"缓存镜像直传"在这里失效**（引擎行序 ≠ 缓存行序），
    // 所以 block_tables / context_lens 必须每步重排一遍再传（见 spec §3）。
    for (int32_t j = 0; j < b_total; ++j) {
        const int32_t active_index = packed_order_active[static_cast<size_t>(j)];
        const ActiveSequence& s = active[static_cast<size_t>(active_index)];
        std::vector<int32_t> blocks;
        if (!kv_cache_->GetBlockTable(s.seq_id, &blocks)) {
            MINI_TRT_LOG_ERROR("LLMRunner: missing block table for seq " << s.seq_id);
            return false;
        }
        for (size_t i = 0; i < blocks.size() && i < width; ++i) {
            host_block_tables[static_cast<size_t>(j) * width + i] = blocks[i];
        }
        host_context_lens[static_cast<size_t>(j)] = kv_cache_->SequenceLength(s.seq_id);
    }

    // ---- ② 上传（每步重建；H2D/D2D 都是 async，循环内不做同步拷贝）----
    const int32_t context_seq_count = packed_context_rows_;
    if ((!host_tokens.empty() &&
         cudaMemcpyAsync(d_packed_tokens_.data(), host_tokens.data(),
                         host_tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                         nullptr) != cudaSuccess) ||
        cudaMemcpyAsync(d_packed_positions_.data(), host_positions.data(),
                        host_positions.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_cu_seqlens_ctx_.data(), host_cu_seqlens.data(),
                        host_cu_seqlens.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_context_seq_count_.data(), &context_seq_count, sizeof(int32_t),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_packed_block_tables_.data(), host_block_tables.data(),
                        host_block_tables.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_packed_context_lens_.data(), host_context_lens.data(),
                        host_context_lens.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to upload packed metadata");
        return false;
    }
    // generation 段的 token 在设备上（按序列聚集的历史），逐行搬进 packed 的尾部
    for (int32_t j = 0; j < generation_rows; ++j) {
        const ActiveSequence& s = active[static_cast<size_t>(j)];
        const int64_t src_index =
            static_cast<int64_t>(s.result_slot) * max_new + s.generated - 1;
        if (cudaMemcpyAsync(static_cast<int32_t*>(d_packed_tokens_.data()) + t_ctx + j,
                            static_cast<const int32_t*>(d_result_tokens_.data()) + src_index,
                            sizeof(int32_t), cudaMemcpyDeviceToDevice, nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to gather packed generation tokens");
            return false;
        }
    }

    // ---- ③ 绑定 + 一次 packed 调用 ----
    Engine* engine = prefill_engine_.get();
    if (engine == nullptr || !engine->SetOptimizationProfile(0, nullptr)) {
        return false;
    }
    if (!engine->SetInputShape("input_ids", nvinfer1::Dims{2, {1, t_total}}) ||
        !engine->SetInputShape("position_ids", nvinfer1::Dims{2, {1, t_total}}) ||
        !engine->SetInputShape("block_tables",
                               nvinfer1::Dims{2, {b_total, config_.max_blocks_per_seq}}) ||
        !engine->SetInputShape("context_lens", nvinfer1::Dims{1, {b_total}}) ||
        !engine->SetInputShape("cu_seqlens_ctx", nvinfer1::Dims{1, {new_rows + 1}}) ||
        !engine->SetInputShape("context_seq_count", nvinfer1::Dims{1, {1}})) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to set packed input shapes");
        return false;
    }
    if (!engine->SetTensorAddress("input_ids", d_packed_tokens_.data()) ||
        !engine->SetTensorAddress("position_ids", d_packed_positions_.data()) ||
        !engine->SetTensorAddress("block_tables", d_packed_block_tables_.data()) ||
        !engine->SetTensorAddress("context_lens", d_packed_context_lens_.data()) ||
        !engine->SetTensorAddress("cu_seqlens_ctx", d_cu_seqlens_ctx_.data()) ||
        !engine->SetTensorAddress("context_seq_count", d_context_seq_count_.data()) ||
        !engine->SetTensorAddress("logits", d_prefill_logits_.data())) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to bind packed inputs");
        return false;
    }
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const std::string index = std::to_string(layer);
        if (!engine->SetTensorAddress(("key_cache_" + index).c_str(),
                                      kv_cache_->key_cache(layer)) ||
            !engine->SetTensorAddress(("value_cache_" + index).c_str(),
                                      kv_cache_->value_cache(layer)) ||
            !engine->SetTensorAddress(
                ("k_layer" + index).c_str(),
                d_prefill_kv_[static_cast<size_t>(layer) * 2]->data()) ||
            !engine->SetTensorAddress(
                ("v_layer" + index).c_str(),
                d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data())) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to bind packed cache/KV at layer " << layer);
            return false;
        }
    }
    if (!engine->Enqueue(nullptr)) {
        MINI_TRT_LOG_ERROR("LLMRunner: packed call failed");
        return false;
    }

    // ---- ④ 图外两块 K/V 工作（都吃 packed 张量里的对应行）----
    if (packed_context_rows_ > 0) {
        // 缓存行号显式查询（不靠"追加在尾部"的隐式约定）；S3 的同一套映射机制
        std::vector<int32_t> cache_rows(static_cast<size_t>(packed_context_rows_), -1);
        for (int32_t j = 0; j < packed_context_rows_; ++j) {
            const int32_t active_index = context_active[static_cast<size_t>(j)];
            const int32_t row = kv_cache_->RowOf(active[static_cast<size_t>(active_index)].seq_id);
            if (row != active_index) {
                MINI_TRT_LOG_ERROR("LLMRunner: seq "
                                   << active[static_cast<size_t>(active_index)].seq_id
                                   << " sits at KV row " << row << ", expected " << active_index
                                   << " —— 行号不同源");
                return false;
            }
            cache_rows[static_cast<size_t>(j)] = row;
        }
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            if (kv_cache_->WritePrefillKV(
                    layer, d_prefill_kv_[static_cast<size_t>(layer) * 2]->data(),
                    d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data(), max_row_len,
                    cache_rows.data(), packed_context_rows_, host_row_lengths.data(), nullptr,
                    static_cast<const int32_t*>(d_cu_seqlens_ctx_.data()),
                    context_seq_count, host_row_starts.data()) != cudaSuccess) {
                MINI_TRT_LOG_ERROR("LLMRunner: failed to write packed context K/V at layer "
                                   << layer);
                return false;
            }
        }
    }
    if (packed_generation_rows_ > 0) {
        std::vector<const void*> keys(static_cast<size_t>(config_.num_layers));
        std::vector<const void*> values(static_cast<size_t>(config_.num_layers));
        // S5：generation 段的行**不再是活跃前缀**（完成的行可能夹在未完成的行后面），
        // 所以显式给出目标 cache 行号；`nullptr` 的旧行为（恒等）留给 S1/S2/S3。
        std::vector<int32_t> generation_rows_host(static_cast<size_t>(packed_generation_rows_), -1);
        for (int32_t j = 0; j < packed_generation_rows_; ++j) {
            const int32_t active_index = generation_active[static_cast<size_t>(j)];
            generation_rows_host[static_cast<size_t>(j)] = active_index;
        }
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            keys[static_cast<size_t>(layer)] =
                d_prefill_kv_[static_cast<size_t>(layer) * 2]->data();
            values[static_cast<size_t>(layer)] =
                d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data();
        }
        // 源基址 = `cu_seqlens_ctx[B_ctx]`（= t_ctx），由 kernel 自己从设备读；
        // 目标行集 = 活跃表前缀（= 缓存前缀，走恒等映射）。
        if (kv_cache_->AppendDecodeStep(
                keys, values, packed_generation_rows_, nullptr,
                static_cast<const int32_t*>(d_cu_seqlens_ctx_.data()), context_seq_count,
                generation_rows_host.data()) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to append packed generation K/V");
            return false;
        }
    }

    // ---- ⑤ 采样前聚集：按**采样行集**压紧（S5：本步完成的 chunk 行可能被未完成的行隔开）----
    const size_t elem = ElementSize(prefill_logits_half_);
    const size_t row_bytes = static_cast<size_t>(config_.vocab_size) * elem;
    const int32_t sample_count = static_cast<int32_t>(sample_active_indices_.size());
    // 活跃行号 → packed 行号 → 该行在 packed 张量里的**末位 token 下标**（两段各自的公式）。
    // S5 用显式表而不是 S4 的"j < new_rows"分段判断，因为采样行集已经不连续。
    std::vector<int32_t> packed_row_of_active(static_cast<size_t>(b_total), -1);
    std::vector<int32_t> last_of_packed_row(static_cast<size_t>(b_total), 0);
    for (size_t j = 0; j < context_active.size(); ++j) {
        const int32_t packed_row = static_cast<int32_t>(j);
        packed_row_of_active[static_cast<size_t>(context_active[j])] = packed_row;
        last_of_packed_row[static_cast<size_t>(packed_row)] = host_cu_seqlens[j + 1] - 1;
    }
    for (size_t j = 0; j < generation_active.size(); ++j) {
        const int32_t packed_row = static_cast<int32_t>(context_active.size() + j);
        packed_row_of_active[static_cast<size_t>(generation_active[j])] = packed_row;
        last_of_packed_row[static_cast<size_t>(packed_row)] = t_ctx + static_cast<int32_t>(j);
    }
    for (int32_t slot = 0; slot < sample_count; ++slot) {
        const int32_t active_index = sample_active_indices_[static_cast<size_t>(slot)];
        const int32_t packed_row = packed_row_of_active[static_cast<size_t>(active_index)];
        if (packed_row < 0) {
            MINI_TRT_LOG_ERROR("LLMRunner: sampled row " << active_index
                                                         << " has no packed row");
            return false;
        }
        const int32_t last = last_of_packed_row[static_cast<size_t>(packed_row)];
        const void* src = static_cast<const char*>(d_prefill_logits_.data()) +
                          static_cast<size_t>(last) * row_bytes;
        void* dst = static_cast<char*>(d_prefill_last_logits_.data()) +
                    static_cast<size_t>(slot) * row_bytes;
        if (cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyDeviceToDevice, nullptr) !=
            cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to gather packed logits rows");
            return false;
        }
    }
    if (!UploadRowParamsByOrder(active, sample_active_indices_) ||
        !SampleBatch(static_cast<char*>(d_step_tokens_.data()), /*from_prefill=*/true, sample_count,
                     /*offset=*/0, static_cast<const uint64_t*>(d_offsets_.data()),
                     static_cast<int8_t*>(d_eos_hit_.data()), nullptr)) {
        MINI_TRT_LOG_ERROR("LLMRunner: packed sampling failed");
        return false;
    }
    return true;
}

bool LLMRunner::UploadRowParams(const std::vector<ActiveSequence>& active, int32_t begin,
                                int32_t count) {
    if (begin < 0 || count < 0 ||
        begin + count > static_cast<int32_t>(active.size()) || begin + count > batch_capacity_) {
        MINI_TRT_LOG_ERROR("LLMRunner: row params out of range");
        return false;
    }
    if (count == 0) {
        return true;
    }
    std::vector<int32_t> top_k(static_cast<size_t>(count));
    std::vector<float> top_p(static_cast<size_t>(count));
    std::vector<uint64_t> seeds(static_cast<size_t>(count));
    std::vector<uint64_t> offsets(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        const ActiveSequence& row = active[static_cast<size_t>(begin + i)];
        top_k[static_cast<size_t>(i)] = row.top_k;
        top_p[static_cast<size_t>(i)] = row.top_p;
        seeds[static_cast<size_t>(i)] = row.seed;
        // 随机步号 = 该行**自己的**已生成计数：随机流只由 (该请求 seed, 该请求的步号) 决定，
        // 与批组成、行号、arrival_step 都无关 —— AC1 的逐位对拍就靠这条（p5_s3_interface_spec §5）。
        offsets[static_cast<size_t>(i)] = static_cast<uint64_t>(row.generated);
    }
    const size_t base = static_cast<size_t>(begin);
    if (cudaMemcpyAsync(static_cast<int32_t*>(d_top_k_.data()) + base, top_k.data(),
                        top_k.size() * sizeof(int32_t), cudaMemcpyHostToDevice, nullptr) !=
            cudaSuccess ||
        cudaMemcpyAsync(static_cast<float*>(d_top_p_.data()) + base, top_p.data(),
                        top_p.size() * sizeof(float), cudaMemcpyHostToDevice, nullptr) !=
            cudaSuccess ||
        cudaMemcpyAsync(static_cast<uint64_t*>(d_seeds_.data()) + base, seeds.data(),
                        seeds.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, nullptr) !=
            cudaSuccess ||
        cudaMemcpyAsync(static_cast<uint64_t*>(d_offsets_.data()) + base, offsets.data(),
                        offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, nullptr) !=
            cudaSuccess) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to upload per-row sampling params");
        return false;
    }
    return true;
}

}  // namespace mini_trt_llm
