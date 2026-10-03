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

}  // namespace
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

const void* LLMRunner::PrefillLogitsRow(int32_t batch_row, int32_t seq_len) const {
    if (seq_len <= 0) {
        return nullptr;
    }
    const size_t elem = ElementSize(prefill_logits_half_);
    // prefill logits 是 [B, S, V]：第 batch_row 条的最后一个位置是 (batch_row, S-1)。
    const size_t index = (static_cast<size_t>(batch_row) * static_cast<size_t>(seq_len) +
                          static_cast<size_t>(seq_len - 1)) *
                         static_cast<size_t>(config_.vocab_size);
    return static_cast<const char*>(d_prefill_logits_.data()) + index * elem;
}

bool LLMRunner::SampleBatch(void* token_out, bool from_prefill, int32_t batch, uint64_t offset,
                            cudaStream_t stream) {
    // S1 限制：批内同策略——三种采样 kernel 各自是"整批一个分支"（入口已校验）。
    const bool use_top_p = options_top_p_ < 1.0f;
    const bool use_greedy = !use_top_p && options_top_k_ <= 1;
    const bool logits_half = from_prefill ? prefill_logits_half_ : decode_logits_half_;
    // 采样器契约：logits 必须是连续的 [batch, vocab]。prefill 的末行已收集进专门缓冲；
    // decode 的 [B,1,V] 本身连续。
    const void* logits = from_prefill ? d_prefill_last_logits_.data() : d_decode_logits_.data();

    if (use_greedy) {
        SamplerArgs args;
        args.logits = logits;
        args.token_ids = static_cast<int32_t*>(token_out);
        args.batch_size = batch;
        args.vocab_size = config_.vocab_size;
        args.is_half = logits_half;
        args.seed = options_seed_;
        args.offset = offset;
        args.seeds = static_cast<const uint64_t*>(d_seeds_.data());
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
        args.seeds = static_cast<const uint64_t*>(d_seeds_.data());
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
    args.seeds = static_cast<const uint64_t*>(d_seeds_.data());
    args.top_k = static_cast<const int32_t*>(d_top_k_.data());
    // **快速路径暂时不接生产路径**：真机实测它比旧路径慢 6~9 倍
    // （见 future_iterations_development_plan.md §10.5 的失败记录），根因是 occupancy 与 bank conflict。
    return LaunchTopKSampler(args, stream, d_sampler_workspace_.data(),
                             d_sampler_workspace_.size()) == cudaSuccess;
}

bool LLMRunner::BindPrefill(const std::vector<int32_t>& tokens, int32_t batch, int32_t seq_len) {
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
    if (cudaMemcpyAsync(d_prompt_.data(), tokens.data(), tokens.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_position_.data(), positions.data(),
                        positions.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess) {
        return false;
    }
    if (!prefill_engine_->SetTensorAddress("input_ids", d_prompt_.data()) ||
        !prefill_engine_->SetTensorAddress("position_ids", d_position_.data()) ||
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
    if (!BindPrefill(flat, batch, prompt_len) || !prefill_engine_->Enqueue(nullptr)) {
        MINI_TRT_LOG_ERROR("LLMRunner: prefill failed");

        return {};
    }
    prefill_engine_->Synchronize(nullptr);  // 每个请求一次，代价可接受

    // ---- 4. 把 prompt 的 K/V 写进分页 cache ----
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        if (kv_cache_->WritePrefillKV(
                layer, d_prefill_kv_[static_cast<size_t>(layer) * 2]->data(),
                d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data(), prompt_len,
                nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to write prefill K/V for layer " << layer);

            return {};
        }
    }

    // 常驻诊断（D7：默认关闭；P4/P7 一律关闭）。批量下只报第 0 条——
    // 逐条全量扫描会把每请求的同步 D2H 放大 B 倍，正是 D7 要避免的。
    if (config_.enable_diagnostics) {
        const void* row_ptr = PrefillLogitsRow(0, prompt_len);
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
            const void* src = PrefillLogitsRow(b, prompt_len);
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
                     /*offset=*/0, nullptr)) {
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
        if (kv_cache_->AppendDecodeStep(keys, values, nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to append decode K/V at step " << i);

            return {};
        }
        if (!SampleBatch(static_cast<char*>(d_tokens_.data()) +
                             static_cast<size_t>(i) * static_cast<size_t>(batch_capacity_) *
                                 sizeof(int32_t),
                         /*from_prefill=*/false, batch, static_cast<uint64_t>(i), nullptr)) {
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

}  // namespace mini_trt_llm
