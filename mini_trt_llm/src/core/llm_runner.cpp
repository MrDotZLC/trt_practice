#include "mini_trt_llm/core/llm_runner.hpp"

#include "mini_trt_llm/core/llm_runner_kernel.hpp"
#include "mini_trt_llm/sampler/sampler_common.hpp"
#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/logger.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

namespace mini_trt_llm {
namespace {

size_t ElementSize(bool is_half) { return is_half ? 2u : 4u; }

// 同一个序列在 KV Cache 里的 id：本版 runner 只服务单序列（batch = 1）。
constexpr int32_t kSequenceId = 0;

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
        config_.num_blocks <= 0 || config_.vocab_size <= 0) {
        MINI_TRT_LOG_ERROR("LLMRunner: invalid config");
        return;
    }
    if (prefill_engine_ == nullptr || decode_engine_ == nullptr) {
        MINI_TRT_LOG_ERROR("LLMRunner: prefill/decode engine must not be null");
        return;
    }

    PagedKVCache::Config cache_config;
    cache_config.num_blocks = config_.num_blocks;
    cache_config.block_size = config_.block_size;
    cache_config.num_layers = config_.num_layers;
    cache_config.num_kv_heads = config_.num_kv_heads;
    cache_config.head_size = config_.head_size;
    cache_config.is_half = config_.is_half;
    cache_config.max_blocks_per_seq = config_.max_blocks_per_seq;
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

bool LLMRunner::ReserveBuffers(int32_t prompt_len, int32_t max_new_tokens) {
    const size_t elem = ElementSize(config_.is_half);
    const size_t kv_layer_elems = static_cast<size_t>(config_.num_kv_heads) *
                                 config_.head_size;

    // 每层的 K/V 输出缓冲：prefill 是 [1, kv_heads, S0, D]，decode 是 [1, kv_heads, 1, D]。
    // 都按"最大可能长度"一次性备好，循环里不再分配（AGENTS.md §3.B.3 的同一条精神）。
    if (d_prefill_kv_.size() != static_cast<size_t>(config_.num_layers) * 2 ||
        prompt_capacity_ < prompt_len) {
        d_prefill_kv_.clear();
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            for (int32_t which = 0; which < 2; ++which) {
                auto buffer = std::make_unique<DeviceBuffer>();
                if (!buffer->Allocate(kv_layer_elems * prompt_len * elem)) {
                    MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate prefill K/V buffer");
                    return false;
                }
                d_prefill_kv_.push_back(std::move(buffer));
            }
        }
        if (!d_prompt_.Allocate(static_cast<size_t>(prompt_len) * sizeof(int32_t)) ||
            !d_position_.Allocate(static_cast<size_t>(prompt_len) * sizeof(int32_t)) ||
            !d_prefill_logits_.Allocate(static_cast<size_t>(prompt_len) *
                                        config_.vocab_size * elem)) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate prefill buffers");
            return false;
        }
        prompt_capacity_ = prompt_len;
    }

    if (d_decode_kv_.size() != static_cast<size_t>(config_.num_layers) * 2) {
        d_decode_kv_.clear();
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            for (int32_t which = 0; which < 2; ++which) {
                auto buffer = std::make_unique<DeviceBuffer>();
                if (!buffer->Allocate(kv_layer_elems * elem)) {
                    MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate decode K/V buffer");
                    return false;
                }
                d_decode_kv_.push_back(std::move(buffer));
            }
        }
        if (!d_decode_logits_.Allocate(static_cast<size_t>(config_.vocab_size) * elem)) {
            return false;
        }
    }

    if (token_capacity_ < max_new_tokens) {
        if (!d_tokens_.Allocate(static_cast<size_t>(max_new_tokens) * sizeof(int32_t))) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate token buffer");
            return false;
        }
        token_capacity_ = max_new_tokens;
    }

    if (d_sampler_workspace_.size() <
        TopKSamplerWorkspaceBytes(/*batch_size=*/1, config_.vocab_size)) {
        if (!d_sampler_workspace_.Allocate(
                TopKSamplerWorkspaceBytes(/*batch_size=*/1, config_.vocab_size))) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate sampler workspace");
            return false;
        }
    }
    return true;
}

const void* LLMRunner::LogitsRow(int32_t row) const {
    const size_t elem = ElementSize(config_.is_half);
    // prefill 的 logits 是 [1, S0, V]，需要最后一行；decode 是 [1, 1, V]，row = 0。
    const DeviceBuffer& buffer = row >= 0 ? d_prefill_logits_ : d_decode_logits_;
    const int32_t index = row >= 0 ? row : 0;
    return static_cast<const char*>(buffer.data()) +
           static_cast<size_t>(index) * config_.vocab_size * elem;
}

bool LLMRunner::SampleInto(void* token_out, int32_t row, uint64_t offset,
                           cudaStream_t stream) {
    const bool use_top_p = options_top_p_ < 1.0f;
    const bool use_greedy = !use_top_p && options_top_k_ <= 1;

    if (use_greedy) {
        SamplerArgs args;
        args.logits = LogitsRow(row);
        args.token_ids = static_cast<int32_t*>(token_out);
        args.batch_size = 1;
        args.vocab_size = config_.vocab_size;
        args.is_half = config_.is_half;
        args.seed = options_seed_;
        args.offset = offset;
        return LaunchGreedySampler(args, stream) == cudaSuccess;
    }
    if (use_top_p) {
        TopPSamplerArgs args;
        args.logits = LogitsRow(row);
        args.token_ids = static_cast<int32_t*>(token_out);
        args.batch_size = 1;
        args.vocab_size = config_.vocab_size;
        args.is_half = config_.is_half;
        args.seed = options_seed_;
        args.offset = offset;
        args.top_p = static_cast<const float*>(d_top_p_.data());
        return LaunchTopPSampler(args, stream, d_sampler_workspace_.data(),
                                 d_sampler_workspace_.size()) == cudaSuccess;
    }
    TopKSamplerArgs args;
    args.logits = LogitsRow(row);
    args.token_ids = static_cast<int32_t*>(token_out);
    args.batch_size = 1;
    args.vocab_size = config_.vocab_size;
    args.is_half = config_.is_half;
    args.seed = options_seed_;
    args.offset = offset;
    args.top_k = static_cast<const int32_t*>(d_top_k_.data());
    return LaunchTopKSampler(args, stream, d_sampler_workspace_.data(),
                             d_sampler_workspace_.size()) == cudaSuccess;
}

bool LLMRunner::BindPrefill(const std::vector<int32_t>& tokens) {
    const int32_t seq_len = static_cast<int32_t>(tokens.size());
    // 顺序要求：先选 profile 再设形状。反过来时 TRT 可能用 profile 的 opt 形状
    // 覆盖显式设置的形状，网络就会按另一个序列长度执行（这个坑在
    // tests/test_gpt2_decode_consistency.cpp 里踩过一次）。
    if (!prefill_engine_->SetOptimizationProfile(0, nullptr)) {
        return false;
    }
    if (!prefill_engine_->SetInputShape("input_ids", nvinfer1::Dims{2, {1, seq_len}}) ||
        !prefill_engine_->SetInputShape("position_ids", nvinfer1::Dims{2, {1, seq_len}})) {
        return false;
    }

    std::vector<int32_t> positions(static_cast<size_t>(seq_len));
    for (int32_t i = 0; i < seq_len; ++i) {
        positions[static_cast<size_t>(i)] = i;
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
    // 每层导出 K/V（prefill 侧用来写 cache）
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const std::string index = std::to_string(layer);
        if (!prefill_engine_->SetTensorAddress(("k_layer" + index).c_str(),
                                              d_prefill_kv_[static_cast<size_t>(layer) * 2]
                                                  ->data()) ||
            !prefill_engine_->SetTensorAddress(
                ("v_layer" + index).c_str(),
                d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data())) {
            return false;
        }
    }
    return true;
}

bool LLMRunner::BindDecode(const int32_t* input_token) {
    if (!decode_engine_->SetOptimizationProfile(0, nullptr)) {
        return false;
    }
    if (!decode_engine_->SetInputShape("input_ids", nvinfer1::Dims{2, {1, 1}}) ||
        !decode_engine_->SetInputShape("position_ids", nvinfer1::Dims{2, {1, 1}}) ||
        !decode_engine_->SetInputShape(
            "block_tables", nvinfer1::Dims{2, {1, config_.max_blocks_per_seq}}) ||
        !decode_engine_->SetInputShape("context_lens", nvinfer1::Dims{1, {1}})) {
        return false;
    }
    if (!decode_engine_->SetTensorAddress("input_ids", const_cast<int32_t*>(input_token)) ||
        !decode_engine_->SetTensorAddress("position_ids", d_position_.data()) ||
        !decode_engine_->SetTensorAddress(
            "block_tables",
            const_cast<int32_t*>(kv_cache_->block_tables())) ||
        !decode_engine_->SetTensorAddress(
            "context_lens", const_cast<int32_t*>(kv_cache_->context_lens())) ||
        !decode_engine_->SetTensorAddress("logits", d_decode_logits_.data())) {
        return false;
    }
    // 每层一对 cache 输入 + 每层导出的当前 token K/V
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const std::string index = std::to_string(layer);
        if (!decode_engine_->SetTensorAddress(("key_cache_" + index).c_str(),
                                              kv_cache_->key_cache(layer)) ||
            !decode_engine_->SetTensorAddress(("value_cache_" + index).c_str(),
                                              kv_cache_->value_cache(layer)) ||
            !decode_engine_->SetTensorAddress(("k_layer" + index).c_str(),
                                              d_decode_kv_[static_cast<size_t>(layer) * 2]
                                                  ->data()) ||
            !decode_engine_->SetTensorAddress(
                ("v_layer" + index).c_str(),
                d_decode_kv_[static_cast<size_t>(layer) * 2 + 1]->data())) {
            return false;
        }
    }
    return true;
}

std::vector<int64_t> LLMRunner::Generate(const std::vector<int64_t>& input_ids,
                                         const GenerateOptions& options) {
    if (!valid_) {
        MINI_TRT_LOG_ERROR("LLMRunner::Generate called on an invalid runner");
        return {};
    }
    if (input_ids.empty() || options.max_new_tokens <= 0) {
        MINI_TRT_LOG_ERROR("LLMRunner: prompt must be non-empty and max_new_tokens > 0");
        return {};
    }
    if (options.temperature != 1.0f) {
        // D5：不支持的参数直接报错，不静默忽略。
        MINI_TRT_LOG_ERROR("LLMRunner: temperature != 1.0 is not supported yet (got "
                           << options.temperature << ")");
        return {};
    }
    if (options.top_k < 1 || options.top_p <= 0.0f || options.top_p > 1.0f) {
        MINI_TRT_LOG_ERROR("LLMRunner: invalid sampling parameters");
        return {};
    }

    const int32_t prompt_len = static_cast<int32_t>(input_ids.size());
    if (prompt_len > config_.max_blocks_per_seq * config_.block_size) {
        MINI_TRT_LOG_ERROR("LLMRunner: prompt longer than the engine's block table capacity");
        return {};
    }
    if (!ReserveBuffers(prompt_len, options.max_new_tokens)) {
        return {};
    }

    // 采样参数：整个请求内不变，上传一次（不在循环里）。
    options_top_k_ = options.top_k;
    options_top_p_ = options.top_p;
    options_seed_ = options.seed;
    const int32_t top_k = options.top_k;
    const float top_p = options.top_p;
    if (!d_top_k_.Allocate(sizeof(int32_t)) || !d_top_p_.Allocate(sizeof(float))) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to allocate sampling parameter buffers");
        return {};
    }
    if (cudaMemcpyAsync(d_top_k_.data(), &top_k, sizeof(int32_t),
                        cudaMemcpyHostToDevice, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(d_top_p_.data(), &top_p, sizeof(float), cudaMemcpyHostToDevice,
                        nullptr) != cudaSuccess) {
        return {};
    }

    // ---- 1. 登记序列并同步元数据 ----
    kv_cache_->FreeSequence(kSequenceId);  // 复用 runner 时把上次请求的块还回去
    if (!kv_cache_->AllocateSequence(kSequenceId, prompt_len + options.max_new_tokens)) {
        MINI_TRT_LOG_ERROR("LLMRunner: not enough KV cache blocks for "
                           << (prompt_len + options.max_new_tokens) << " tokens");
        return {};
    }
    if (kv_cache_->UploadMetadata(nullptr) != cudaSuccess) {
        return {};
    }

    std::vector<int32_t> prompt_tokens(input_ids.begin(), input_ids.end());

    // ---- 2. Prefill：整段 prompt 一次前向 ----
    if (!BindPrefill(prompt_tokens)) {
        MINI_TRT_LOG_ERROR("LLMRunner: failed to bind prefill inputs");
        return {};
    }
    if (!prefill_engine_->Enqueue(nullptr)) {
        MINI_TRT_LOG_ERROR("LLMRunner: prefill enqueue failed");
        return {};
    }
    prefill_engine_->Synchronize(nullptr);  // 每个请求一次，代价可接受

    // ---- 3. 把 prompt 的 K/V 写进分页 cache，并采出第 1 个新 token ----
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        if (kv_cache_->WritePrefillKV(
                layer, d_prefill_kv_[static_cast<size_t>(layer) * 2]->data(),
                d_prefill_kv_[static_cast<size_t>(layer) * 2 + 1]->data(), prompt_len,
                nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to write prefill K/V for layer " << layer);
            return {};
        }
    }
    // 采样器直接写进结果缓冲的第 0 位，省掉一次拷贝
    if (!SampleInto(static_cast<char*>(d_tokens_.data()), /*row=*/prompt_len - 1,
                    /*offset=*/0, nullptr)) {
        MINI_TRT_LOG_ERROR("LLMRunner: sampling after prefill failed");
        return {};
    }

    // ---- 4. Decode 循环（循环内零 H2D/D2H）----
    // 不变量（进入第 i 次迭代时）：
    //   cache 里有 token 0..S0+i-2 的 K/V，context_lens = S0+i-1；
    //   d_tokens_[i-1] 就是本次要喂进去的 token，其位置正是 context_lens。
    for (int32_t i = 1; i < options.max_new_tokens; ++i) {
        const int32_t* input_token =
            static_cast<const int32_t*>(d_tokens_.data()) + (i - 1);
        if (!BindDecode(input_token)) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to bind decode inputs");
            return {};
        }
        // position_ids 由设备端的 context_lens 填，避免为这一个整数回主机
        if (LaunchFillPositionIds(kv_cache_->context_lens(),
                                  static_cast<int32_t*>(d_position_.data()),
                                  /*batch_size=*/1, nullptr) != cudaSuccess) {
            return {};
        }
        if (!decode_engine_->Enqueue(nullptr)) {
            MINI_TRT_LOG_ERROR("LLMRunner: decode enqueue failed at step " << i);
            return {};
        }
        // 把当前 token 的 K/V 追加进 cache（必须在 engine 之后，同一 stream 串行）。
        // 用 AppendDecodeStep 而不是逐层 AppendDecodeKV：后者会让语境长度被推进
        // n_layer 次，从而在下一个 token 就发散。
        std::vector<const void*> keys(static_cast<size_t>(config_.num_layers));
        std::vector<const void*> values(static_cast<size_t>(config_.num_layers));
        for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
            keys[static_cast<size_t>(layer)] =
                d_decode_kv_[static_cast<size_t>(layer) * 2]->data();
            values[static_cast<size_t>(layer)] =
                d_decode_kv_[static_cast<size_t>(layer) * 2 + 1]->data();
        }
        if (kv_cache_->AppendDecodeStep(keys, values, nullptr) != cudaSuccess) {
            MINI_TRT_LOG_ERROR("LLMRunner: failed to append decode K/V");
            return {};
        }
        if (!SampleInto(static_cast<char*>(d_tokens_.data()) + i * sizeof(int32_t),
                        /*row=*/-1, /*offset=*/static_cast<uint64_t>(i), nullptr)) {
            MINI_TRT_LOG_ERROR("LLMRunner: sampling failed at step " << i);
            return {};
        }
    }

    // ---- 5. 一次性取回结果 ----
    std::vector<int32_t> generated(static_cast<size_t>(options.max_new_tokens));
    if (cudaMemcpyAsync(generated.data(), d_tokens_.data(),
                        generated.size() * sizeof(int32_t), cudaMemcpyDeviceToHost,
                        nullptr) != cudaSuccess) {
        return {};
    }
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
        return {};
    }

    // EOS 截断放在这里而不是循环里：循环内同步会带来 host 往返，
    // 而"一见 EOS 就停"必须知道 token 的值（设备上拿不到）。
    size_t length = generated.size();
    if (config_.eos_token_id >= 0) {
        for (size_t i = 0; i < generated.size(); ++i) {
            if (generated[i] == config_.eos_token_id) {
                length = i;
                break;
            }
        }
    }
    std::vector<int64_t> result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result.push_back(static_cast<int64_t>(generated[i]));
    }
    return result;
}

}  // namespace mini_trt_llm
