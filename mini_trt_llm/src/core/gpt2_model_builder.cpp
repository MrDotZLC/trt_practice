#include "mini_trt_llm/core/gpt2_model_builder.hpp"

#include "mini_trt_llm/plugins/paged_attention_plugin.hpp"
#include "mini_trt_llm/utils/logger.hpp"

#include <NvInfer.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

namespace {

// GPT-2 的 MLP 把 hidden 放大 4 倍（c_fc: n_embd -> 4 * n_embd）。
constexpr int32_t kMlpExpansion = 4;

// 因果 mask 中被屏蔽位置的值。
// 取 -1e4 而不是 -inf：-inf 在 FP16 里不可表示；-1e4 经 softmax 后严格为 0
// （exp(-1e4) 在任何精度下都下溢），同时离 FP16 上限 65504 仍有足够余量。
constexpr float kCausalMaskValue = -1e4f;

size_t DtypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
            return 1;
        default:
            return 0;
    }
}

// 静态维的乘积；含动态维（<0）时返回 -1，表示无法做静态校验。
int64_t Volume(const nvinfer1::Dims& dims) {
    int64_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return -1;
        }
        volume *= dims.d[i];
    }
    return volume;
}

std::string LayerName(const std::string& suffix) {
    return std::string(kGpt2LayerPrefix) + suffix;
}

nvinfer1::Dims Dims2(int32_t a, int32_t b) {
    return nvinfer1::Dims{2, {a, b}};
}

nvinfer1::Dims Dims3(int32_t a, int32_t b, int32_t c) {
    return nvinfer1::Dims{3, {a, b, c}};
}

nvinfer1::Dims Dims4(int32_t a, int32_t b, int32_t c, int32_t d) {
    return nvinfer1::Dims{4, {a, b, c, d}};
}

// 从 WeightLoader 取权重并挂成常量层。
//
// 这里做两道检查，都是为了把错误挡在**建网阶段**而不是留到运行期：
//   1. 权重取不到 → 立即失败。GPT-2 有 148 个权重，静默跳过任何一个都会建出错误网络；
//   2. 张量元素数与声明的常量形状不一致 → 失败。weight_map 指错张量时形状通常就对不上，
//      这条检查能把"key 改名 / 映射写反"这类问题当场暴露。
nvinfer1::ITensor* AddWeightConstant(nvinfer1::INetworkDefinition* network,
                                     const WeightLoader& weights,
                                     const std::string& name,
                                     nvinfer1::DataType dtype,
                                     const nvinfer1::Dims& dims) {
    size_t bytes = 0;
    const void* data = weights.GetWeight(name, dtype, &bytes);
    if (data == nullptr) {
        MINI_TRT_LOG_ERROR("GPT-2 weight not found: " << name);
        return nullptr;
    }

    const size_t element_size = DtypeSize(dtype);
    if (element_size == 0) {
        MINI_TRT_LOG_ERROR("GPT-2 weight " << name << " has unsupported dtype");
        return nullptr;
    }

    const int64_t declared = Volume(dims);
    const int64_t actual = static_cast<int64_t>(bytes / element_size);
    if (declared >= 0 && declared != actual) {
        MINI_TRT_LOG_ERROR("GPT-2 weight " << name << ": declared shape wants "
                                           << declared << " elements but tensor has "
                                           << actual);
        return nullptr;
    }

    nvinfer1::IConstantLayer* layer =
        network->addConstant(dims, nvinfer1::Weights{dtype, data, actual});
    if (layer == nullptr) {
        MINI_TRT_LOG_ERROR("GPT-2 constant layer failed for: " << name);
        return nullptr;
    }
    layer->setName(LayerName("const_" + name).c_str());
    return layer->getOutput(0);
}

// 用主机侧 float 数据建常量，必要时补一层 Cast 到目标精度。
//
// **不能**把 float 缓冲直接标成 kHALF：nvinfer1::Weights 只带一个裸指针和一个类型标记，
// TRT 会按 kHALF 去解释这段内存（把 4 字节 float 当 2 字节 half 读），
// 结果是"能建出来、数值全错"的静默故障。这里统一先建 FP32 常量再显式转换。
nvinfer1::ITensor* AddFloatConstant(nvinfer1::INetworkDefinition* network,
                                    const float* values, size_t count,
                                    const nvinfer1::Dims& dims,
                                    nvinfer1::DataType target,
                                    const std::string& name) {
    nvinfer1::IConstantLayer* layer = network->addConstant(
        dims, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, values,
                                static_cast<int64_t>(count)});
    if (layer == nullptr) {
        MINI_TRT_LOG_ERROR("GPT-2 build: constant layer failed for " << name);
        return nullptr;
    }
    layer->setName(LayerName(name).c_str());
    if (target == nvinfer1::DataType::kFLOAT) {
        return layer->getOutput(0);
    }
    nvinfer1::ICastLayer* cast = network->addCast(*layer->getOutput(0), target);
    if (cast == nullptr) {
        MINI_TRT_LOG_ERROR("GPT-2 build: cast to target dtype failed for " << name);
        return nullptr;
    }
    cast->setName(LayerName(name + "_cast").c_str());
    return cast->getOutput(0);
}

nvinfer1::ITensor* AddReshape(nvinfer1::INetworkDefinition* network,
                              nvinfer1::ITensor* input, const nvinfer1::Dims& dims,
                              const std::string& name) {
    nvinfer1::IShuffleLayer* layer = network->addShuffle(*input);
    if (layer == nullptr) {
        return nullptr;
    }
    // 0 在 reshape 维度里表示"沿用输入对应维"（TRT 的 placeholder 语义，默认开启），
    // 因此含动态 batch/seq 的画面不需要显式知道这两个值。
    layer->setReshapeDimensions(dims);
    layer->setZeroIsPlaceholder(true);
    layer->setName(name.c_str());
    return layer->getOutput(0);
}

nvinfer1::ITensor* AddTranspose(nvinfer1::INetworkDefinition* network,
                                nvinfer1::ITensor* input, const nvinfer1::Dims& order,
                                const std::string& name) {
    nvinfer1::IShuffleLayer* layer = network->addShuffle(*input);
    if (layer == nullptr) {
        return nullptr;
    }
    nvinfer1::Permutation permutation{};
    for (int32_t i = 0; i < nvinfer1::Dims::MAX_DIMS; ++i) {
        permutation.order[i] = i;
    }
    for (int32_t i = 0; i < order.nbDims; ++i) {
        permutation.order[i] = order.d[i];
    }
    layer->setSecondTranspose(permutation);
    layer->setName(name.c_str());
    return layer->getOutput(0);
}

// 单轴静态切片：[start, start+size)，步长 1。
// 用于把 QKV 的 "3" 那一维拆成 q / k / v。
nvinfer1::ITensor* AddSliceOnAxis(nvinfer1::INetworkDefinition* network,
                                  nvinfer1::ITensor* input, int32_t axis,
                                  int64_t start, int64_t size,
                                  const std::string& name) {
    nvinfer1::ISliceLayer* layer = network->addSlice(
        *input, nvinfer1::Dims{1, {start}}, nvinfer1::Dims{1, {size}},
        nvinfer1::Dims{1, {1}});
    if (layer == nullptr) {
        return nullptr;
    }
    layer->setAxes(nvinfer1::Dims{1, {axis}});
    layer->setName(name.c_str());
    return layer->getOutput(0);
}

nvinfer1::ITensor* AddElementWise(nvinfer1::INetworkDefinition* network,
                                  nvinfer1::ITensor* a, nvinfer1::ITensor* b,
                                  nvinfer1::ElementWiseOperation op,
                                  const std::string& name) {
    nvinfer1::IElementWiseLayer* layer = network->addElementWise(*a, *b, op);
    if (layer == nullptr) {
        return nullptr;
    }
    layer->setName(name.c_str());
    return layer->getOutput(0);
}

// y = x @ w + b。w 必须声明成 rank-3（前导 1），因为 TRT 的矩阵乘要求
// 两个输入的"额外维"个数一致——[B,S,H] @ [H,O] 会因额外维不匹配而失败。
nvinfer1::ITensor* AddLinear(nvinfer1::INetworkDefinition* network,
                             nvinfer1::ITensor* input, nvinfer1::ITensor* weight,
                             nvinfer1::ITensor* bias, const std::string& name) {
    nvinfer1::IMatrixMultiplyLayer* matmul = network->addMatrixMultiply(
        *input, nvinfer1::MatrixOperation::kNONE, *weight,
        nvinfer1::MatrixOperation::kNONE);
    if (matmul == nullptr) {
        return nullptr;
    }
    matmul->setName((name + "_matmul").c_str());
    return AddElementWise(network, matmul->getOutput(0), bias,
                          nvinfer1::ElementWiseOperation::kSUM, name);
}

// LayerNorm（归一化最后一维）。
//
// axesMask 的 bit i 对应第 i 个显式维度、LSB 是 dim 0，所以 [B,S,H] 上归一化 H 是 bit 2。
//
// **显式把该层的计算精度设为 FP32，不依赖默认值**：早先版本靠"TRT 默认就是 FP32"这个
// 假定，而 FP16 引擎的实测结果（第 0 层 K/V 干净、第 1 层起 NaN）把范围指到了包含 LN 的
// 那一段。TRT 头文件里 setComputePrecision 的说明正是为这种场景写的
// （"avoid overflow errors by controlling the normalization computation in mixed
// precision mode"）——既然它存在，就不该赌默认值。
// 排查过程见 docs/TROUBLESHOOTING.md #18。
nvinfer1::ITensor* AddLayerNorm(nvinfer1::INetworkDefinition* network,
                                nvinfer1::ITensor* input, nvinfer1::ITensor* scale,
                                nvinfer1::ITensor* bias, float eps,
                                const std::string& name) {
    const int32_t rank = input->getDimensions().nbDims;
    if (rank <= 0 || rank > 8) {
        MINI_TRT_LOG_ERROR("GPT-2 LayerNorm got unsupported rank: " << rank);
        return nullptr;
    }
    const uint32_t axes_mask = 1u << (rank - 1);
    nvinfer1::INormalizationLayer* layer =
        network->addNormalizationV2(*input, *scale, *bias, axes_mask);
    if (layer == nullptr) {
        return nullptr;
    }
    layer->setEpsilon(eps);
    // 见函数头注释：显式声明 FP32 计算，避免 FP16 下归一化的数值问题
    layer->setComputePrecision(nvinfer1::DataType::kFLOAT);
    layer->setName(name.c_str());
    return layer->getOutput(0);
}

}  // namespace

bool GPT2Config::FromModelConfig(const ModelConfig& config, GPT2Config* out) {
    if (!config.hyper_params.IsObject()) {
        MINI_TRT_LOG_ERROR("GPT-2 config: hyper_params must be an object");
        return false;
    }
    const JsonValue& hp = config.hyper_params;

    // 必需字段：缺失即失败，不用默认值兜底——默认值会把"config 写错"变成"静默建错网络"。
    const char* required[] = {"n_layer", "n_head", "n_embd", "n_positions",
                              "vocab_size", "block_size"};
    for (const char* key : required) {
        if (!hp.Has(key) || !hp[key].IsNumber()) {
            MINI_TRT_LOG_ERROR("GPT-2 config: hyper_params missing numeric '" << key << "'");
            return false;
        }
    }

    GPT2Config parsed;
    parsed.n_layer = hp["n_layer"].AsInt();
    parsed.n_head = hp["n_head"].AsInt();
    parsed.n_embd = hp["n_embd"].AsInt();
    parsed.n_positions = hp["n_positions"].AsInt();
    parsed.vocab_size = hp["vocab_size"].AsInt();
    parsed.block_size = hp["block_size"].AsInt();
    if (hp.Has("layer_norm_epsilon") && hp["layer_norm_epsilon"].IsNumber()) {
        parsed.layer_norm_epsilon =
            static_cast<float>(hp["layer_norm_epsilon"].AsNumber());
    }
    if (hp.Has("tie_word_embeddings") && hp["tie_word_embeddings"].IsBool()) {
        parsed.tie_word_embeddings = hp["tie_word_embeddings"].AsBool();
    }

    if (parsed.n_layer <= 0 || parsed.n_head <= 0 || parsed.n_embd <= 0 ||
        parsed.n_positions <= 0 || parsed.vocab_size <= 0 || parsed.block_size <= 0) {
        MINI_TRT_LOG_ERROR("GPT-2 config: hyper_params must all be positive");
        return false;
    }
    if (parsed.n_embd % parsed.n_head != 0) {
        MINI_TRT_LOG_ERROR("GPT-2 config: n_embd (" << parsed.n_embd
                                                   << ") must be divisible by n_head ("
                                                   << parsed.n_head << ")");
        return false;
    }

    *out = parsed;
    return true;
}

std::vector<std::string> GPT2WeightNames(const GPT2Config& config) {
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(config.n_layer) * 12 + 5);

    names.emplace_back("wte.weight");
    names.emplace_back("wpe.weight");
    for (int32_t i = 0; i < config.n_layer; ++i) {
        const std::string prefix = "h." + std::to_string(i) + ".";
        names.push_back(prefix + "ln_1.weight");
        names.push_back(prefix + "ln_1.bias");
        names.push_back(prefix + "attn.c_attn.weight");
        names.push_back(prefix + "attn.c_attn.bias");
        names.push_back(prefix + "attn.c_proj.weight");
        names.push_back(prefix + "attn.c_proj.bias");
        names.push_back(prefix + "ln_2.weight");
        names.push_back(prefix + "ln_2.bias");
        names.push_back(prefix + "mlp.c_fc.weight");
        names.push_back(prefix + "mlp.c_fc.bias");
        names.push_back(prefix + "mlp.c_proj.weight");
        names.push_back(prefix + "mlp.c_proj.bias");
    }
    names.emplace_back("ln_f.weight");
    names.emplace_back("ln_f.bias");
    if (!config.tie_word_embeddings) {
        names.emplace_back("lm_head.weight");
    }
    return names;
}

bool GPT2ModelBuilder::Build(nvinfer1::INetworkDefinition* network,
                             const WeightLoader& weights, const ModelConfig& config,
                             const BuildOptions& options) {
    GPT2Config cfg;
    if (!GPT2Config::FromModelConfig(config, &cfg)) {
        return false;
    }

    const bool is_decode = options.stage == BuildStage::kDecode;
    // prefill 与 decode 都要把每层的 K/V 导出：前者用来写 cache，
    // 后者是"下一轮追加"的数据来源。
    const bool export_kv = options.stage != BuildStage::kSingle;
    const nvinfer1::DataType dtype = options.weight_dtype;
    const int32_t hidden = cfg.n_embd;
    const int32_t heads = cfg.n_head;
    const int32_t head_size = cfg.head_size();
    const int32_t vocab = cfg.vocab_size;
    const int32_t inner = kMlpExpansion * hidden;
    // 存成成员：常量层的 Weights 需要它的地址在 buildSerializedNetwork 之前一直有效。
    attn_scale_value_ = 1.0f / std::sqrt(static_cast<float>(head_size));

    // ---- 输入 ----
    // 位置编码作为显式输入（与 RoPE 的 Q4 决策一致）：写死成内部常量会让
    // 分段 prefill / 带 history 的调用无法复用同一个引擎。
    // decode 每步只吃一个 token，所以序列维是静态 1；prefill 的序列维是动态的。
    nvinfer1::ITensor* input_ids = network->addInput(
        "input_ids", nvinfer1::DataType::kINT32, is_decode ? Dims2(-1, 1) : Dims2(-1, -1));
    nvinfer1::ITensor* position_ids = network->addInput(
        "position_ids", nvinfer1::DataType::kINT32,
        is_decode ? Dims2(-1, 1) : Dims2(-1, -1));
    if (input_ids == nullptr || position_ids == nullptr) {
        MINI_TRT_LOG_ERROR("GPT-2 build: failed to declare inputs");
        return false;
    }

    // decode 专有输入：分页 KV Cache 与它的元数据。
    // block table 的宽度必须与 PagedKVCache 的 max_blocks_per_seq 一致，
    // 两者都由 ceil(n_positions / block_size) 推出——这里只推导一次，另一处由调用方显式传入。
    nvinfer1::ITensor* block_tables = nullptr;
    nvinfer1::ITensor* context_lens = nullptr;
    // **每层一对 cache 输入**：PagedAttention 插件是"单层注意力"，
    // 它只接收一个 4-D 的 [num_blocks, block_size, kv_heads, head_size]。
    // 如果所有层共用同一个 cache 张量，那么每一层都会去读同一段 cache
    // （即第 0 层那段），层数越多错得越离谱——这一点由
    // tests/test_gpt2_decode_consistency.cpp 的严格对拍抓出来过。
    std::vector<nvinfer1::ITensor*> key_caches(static_cast<size_t>(cfg.n_layer), nullptr);
    std::vector<nvinfer1::ITensor*> value_caches(static_cast<size_t>(cfg.n_layer), nullptr);
    if (is_decode) {
        const int32_t blocks_per_seq = cfg.num_blocks();
        block_tables = network->addInput(
            "block_tables", nvinfer1::DataType::kINT32, Dims2(-1, blocks_per_seq));
        context_lens = network->addInput("context_lens", nvinfer1::DataType::kINT32,
                                        nvinfer1::Dims{1, {-1}});
        const nvinfer1::Dims cache_dims = Dims4(cfg.num_blocks(), cfg.block_size, heads,
                                                head_size);
        for (int32_t layer = 0; layer < cfg.n_layer; ++layer) {
            const std::string suffix = std::to_string(layer);
            key_caches[static_cast<size_t>(layer)] = network->addInput(
                ("key_cache_" + suffix).c_str(), dtype, cache_dims);
            value_caches[static_cast<size_t>(layer)] = network->addInput(
                ("value_cache_" + suffix).c_str(), dtype, cache_dims);
            if (key_caches[static_cast<size_t>(layer)] == nullptr ||
                value_caches[static_cast<size_t>(layer)] == nullptr) {
                MINI_TRT_LOG_ERROR("GPT-2 build: failed to declare cache inputs for layer "
                                   << layer);
                return false;
            }
        }
        if (block_tables == nullptr || context_lens == nullptr) {
            MINI_TRT_LOG_ERROR("GPT-2 build: failed to declare decode inputs");
            return false;
        }
    }

    // ---- 权重常量 ----
    nvinfer1::ITensor* wte =
        AddWeightConstant(network, weights, "wte.weight", dtype, Dims2(vocab, hidden));
    nvinfer1::ITensor* wpe = AddWeightConstant(network, weights, "wpe.weight", dtype,
                                               Dims2(cfg.n_positions, hidden));
    if (wte == nullptr || wpe == nullptr) {
        return false;
    }

    // ---- 词嵌入 + 位置嵌入 ----
    nvinfer1::IGatherLayer* token_embed = network->addGather(*wte, *input_ids, 0);
    nvinfer1::IGatherLayer* position_embed =
        network->addGather(*wpe, *position_ids, 0);
    if (token_embed == nullptr || position_embed == nullptr) {
        MINI_TRT_LOG_ERROR("GPT-2 build: failed to add embedding gather layers");
        return false;
    }
    token_embed->setName(LayerName("token_embed").c_str());
    position_embed->setName(LayerName("position_embed").c_str());
    nvinfer1::ITensor* hidden_state =
        AddElementWise(network, token_embed->getOutput(0), position_embed->getOutput(0),
                       nvinfer1::ElementWiseOperation::kSUM, LayerName("embed_sum"));
    if (hidden_state == nullptr) {
        return false;
    }

    // 因果 mask 常量：[1, 1, n_positions, n_positions]，下三角 0、上三角 kCausalMaskValue。
    // 存成成员缓冲而不是局部变量：IConstantLayer 的 Weights 是裸指针形式传入的，
    // 让缓冲活到 buildSerializedNetwork 之后才安全。
    causal_mask_.assign(static_cast<size_t>(cfg.n_positions) * cfg.n_positions, 0.0f);
    for (int32_t row = 0; row < cfg.n_positions; ++row) {
        for (int32_t col = row + 1; col < cfg.n_positions; ++col) {
            causal_mask_[static_cast<size_t>(row) * cfg.n_positions + col] =
                kCausalMaskValue;
        }
    }
    // mask 始终以 FP32 存、按需 Cast：_FloatConstant 的注释解释了为什么不能直接标 kHALF。
    nvinfer1::ITensor* mask_const =
        AddFloatConstant(network, causal_mask_.data(), causal_mask_.size(),
                         Dims4(1, 1, cfg.n_positions, cfg.n_positions), dtype,
                         "causal_mask");
    if (mask_const == nullptr) {
        return false;
    }

    // ---- Transformer block × n_layer ----
    for (int32_t layer = 0; layer < cfg.n_layer; ++layer) {
        const std::string prefix = "h." + std::to_string(layer) + ".";
        const std::string scope = "h" + std::to_string(layer) + "_";

        nvinfer1::ITensor* ln1_w = AddWeightConstant(
            network, weights, prefix + "ln_1.weight", dtype, Dims3(1, 1, hidden));
        nvinfer1::ITensor* ln1_b = AddWeightConstant(
            network, weights, prefix + "ln_1.bias", dtype, Dims3(1, 1, hidden));
        nvinfer1::ITensor* c_attn_w = AddWeightConstant(
            network, weights, prefix + "attn.c_attn.weight", dtype,
            Dims3(1, hidden, 3 * hidden));
        nvinfer1::ITensor* c_attn_b = AddWeightConstant(
            network, weights, prefix + "attn.c_attn.bias", dtype, Dims3(1, 1, 3 * hidden));
        nvinfer1::ITensor* c_proj_w = AddWeightConstant(
            network, weights, prefix + "attn.c_proj.weight", dtype,
            Dims3(1, hidden, hidden));
        nvinfer1::ITensor* c_proj_b = AddWeightConstant(
            network, weights, prefix + "attn.c_proj.bias", dtype, Dims3(1, 1, hidden));
        nvinfer1::ITensor* ln2_w = AddWeightConstant(
            network, weights, prefix + "ln_2.weight", dtype, Dims3(1, 1, hidden));
        nvinfer1::ITensor* ln2_b = AddWeightConstant(
            network, weights, prefix + "ln_2.bias", dtype, Dims3(1, 1, hidden));
        nvinfer1::ITensor* c_fc_w = AddWeightConstant(
            network, weights, prefix + "mlp.c_fc.weight", dtype,
            Dims3(1, hidden, inner));
        nvinfer1::ITensor* c_fc_b = AddWeightConstant(
            network, weights, prefix + "mlp.c_fc.bias", dtype, Dims3(1, 1, inner));
        nvinfer1::ITensor* c_mlp_w = AddWeightConstant(
            network, weights, prefix + "mlp.c_proj.weight", dtype, Dims3(1, inner, hidden));
        nvinfer1::ITensor* c_mlp_b = AddWeightConstant(
            network, weights, prefix + "mlp.c_proj.bias", dtype, Dims3(1, 1, hidden));
        if (ln1_w == nullptr || ln1_b == nullptr || c_attn_w == nullptr ||
            c_attn_b == nullptr || c_proj_w == nullptr || c_proj_b == nullptr ||
            ln2_w == nullptr || ln2_b == nullptr || c_fc_w == nullptr ||
            c_fc_b == nullptr || c_mlp_w == nullptr || c_mlp_b == nullptr) {
            return false;
        }

        // ---- 自注意力 ----
        nvinfer1::ITensor* norm1 =
            AddLayerNorm(network, hidden_state, ln1_w, ln1_b, cfg.layer_norm_epsilon,
                         LayerName(scope + "ln_1"));
        nvinfer1::ITensor* qkv =
            AddLinear(network, norm1, c_attn_w, c_attn_b, LayerName(scope + "c_attn"));
        if (qkv == nullptr) {
            return false;
        }

        // [B,S,3H] -> [B,S,3,NH,D]：Conv1D 的输出布局本来就是 q|k|v 连续排列，
        // 所以按"3"这一维切片即可，不需要在最后一维上做复杂切片。
        nvinfer1::ITensor* qkv_5d =
            AddReshape(network, qkv, nvinfer1::Dims{5, {0, 0, 3, heads, head_size}},
                       LayerName(scope + "qkv_reshape"));
        if (qkv_5d == nullptr) {
            return false;
        }

        nvinfer1::ITensor* q_kv[3];
        const char* kv_names[3] = {"q", "k", "v"};
        for (int32_t part = 0; part < 3; ++part) {
            nvinfer1::ITensor* slice =
                AddSliceOnAxis(network, qkv_5d, /*axis=*/2, part, 1,
                               LayerName(scope + kv_names[part] + "_split"));
            if (slice == nullptr) {
                return false;
            }
            // [B,S,1,NH,D] -> [B,NH,S,D]：一步 reshape（去掉单维）再一步转置。
            // reshape 与 permutation 分两层做，避免零占位符与置换同时生效时的歧义。
            nvinfer1::ITensor* squeezed = AddReshape(
                network, slice, Dims4(0, 0, heads, head_size),
                LayerName(scope + kv_names[part] + "_squeeze"));
            if (squeezed == nullptr) {
                return false;
            }
            q_kv[part] = AddTranspose(network, squeezed, Dims4(0, 2, 1, 3),
                                      LayerName(scope + kv_names[part] + "_heads"));
            if (q_kv[part] == nullptr) {
                return false;
            }
        }

        // 注意力：decode 走 PagedAttention 插件（含当前 token 的 K/V），
        // prefill 用显式子图（MatMul → mask → Softmax → MatMul）。
        // 两条路都产出 [B, NH, seq, D]，之后的 reshape / c_proj / 残差完全共用。
        nvinfer1::ITensor* attention_out = nullptr;
        if (is_decode) {
            nvinfer1::ITensor* plugin_inputs[7] = {
                q_kv[0],
                key_caches[static_cast<size_t>(layer)],
                value_caches[static_cast<size_t>(layer)],
                block_tables,
                context_lens,
                q_kv[1],
                q_kv[2],
            };
            // scale 传 0 → 插件按 1/sqrt(head_size) 推导（Q6）。
            PagedAttentionPlugin plugin(heads, heads, head_size, cfg.block_size, 0.0f);
            nvinfer1::IPluginV3Layer* plugin_layer =
                network->addPluginV3(plugin_inputs, 7, nullptr, 0, plugin);
            if (plugin_layer == nullptr) {
                MINI_TRT_LOG_ERROR("GPT-2 build: PagedAttention plugin failed at layer "
                                   << layer);
                return false;
            }
            plugin_layer->setName(LayerName(scope + "paged_attention").c_str());
            attention_out = plugin_layer->getOutput(0);
        } else {
            nvinfer1::ITensor* key_t =
                AddTranspose(network, q_kv[1], Dims4(0, 1, 3, 2),
                             LayerName(scope + "k_transpose"));
        nvinfer1::IMatrixMultiplyLayer* scores_mm = network->addMatrixMultiply(
            *q_kv[0], nvinfer1::MatrixOperation::kNONE, *key_t,
            nvinfer1::MatrixOperation::kNONE);
        if (scores_mm == nullptr) {
            MINI_TRT_LOG_ERROR("GPT-2 build: scores matmul failed at layer " << layer);
            return false;
        }
        scores_mm->setName(LayerName(scope + "scores").c_str());

        // 缩放：1/sqrt(head_size)
        nvinfer1::ITensor* scale_const =
            AddFloatConstant(network, &attn_scale_value_, 1, Dims4(1, 1, 1, 1), dtype,
                             scope + "attn_scale");
        if (scale_const == nullptr) {
            return false;
        }
        nvinfer1::ITensor* scores = AddElementWise(
            network, scores_mm->getOutput(0), scale_const,
            nvinfer1::ElementWiseOperation::kPROD, LayerName(scope + "scores_scaled"));
        if (scores == nullptr) {
            return false;
        }

        // 因果 mask：把常量裁成当前 S×S 再加到 scores 上。
        // 这里的 size 必须由运行期形状给出（序列维是动态的），走的是 TRT 的
        // "dynamic slice"：从 scores 的 shape tensor 里取最后两维当 size 输入。
        nvinfer1::IShapeLayer* scores_shape = network->addShape(*scores);
        if (scores_shape == nullptr) {
            return false;
        }
        scores_shape->setName(LayerName(scope + "scores_shape").c_str());
        nvinfer1::ISliceLayer* seq_shape = network->addSlice(
            *scores_shape->getOutput(0), nvinfer1::Dims{1, {2}}, nvinfer1::Dims{1, {2}},
            nvinfer1::Dims{1, {1}});
        seq_shape->setName(LayerName(scope + "seq_shape").c_str());

        nvinfer1::ISliceLayer* mask_slice = network->addSlice(
            *mask_const, nvinfer1::Dims{2, {0, 0}},
            nvinfer1::Dims{2, {cfg.n_positions, cfg.n_positions}},
            nvinfer1::Dims{2, {1, 1}});
        mask_slice->setAxes(nvinfer1::Dims{2, {2, 3}});
        mask_slice->setInput(2, *seq_shape->getOutput(0));
        mask_slice->setName(LayerName(scope + "mask_slice").c_str());

        nvinfer1::ITensor* masked = AddElementWise(
            network, scores, mask_slice->getOutput(0),
            nvinfer1::ElementWiseOperation::kSUM, LayerName(scope + "masked_scores"));
        if (masked == nullptr) {
            return false;
        }

        nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*masked);
        if (softmax == nullptr) {
            return false;
        }
        // scores 是 [B,NH,S,S]，在最后一维（key 方向）上归一化 → bit 3。
        softmax->setAxes(1u << 3);
        softmax->setName(LayerName(scope + "softmax").c_str());

        nvinfer1::IMatrixMultiplyLayer* context_mm = network->addMatrixMultiply(
            *softmax->getOutput(0), nvinfer1::MatrixOperation::kNONE, *q_kv[2],
            nvinfer1::MatrixOperation::kNONE);
        if (context_mm == nullptr) {
            return false;
        }
        context_mm->setName(LayerName(scope + "context").c_str());

            attention_out = context_mm->getOutput(0);
        }

        nvinfer1::ITensor* context = AddTranspose(
            network, attention_out, Dims4(0, 2, 1, 3),
            LayerName(scope + "context_heads"));
        nvinfer1::ITensor* context_flat = AddReshape(
            network, context, Dims3(0, 0, hidden), LayerName(scope + "context_flat"));
        nvinfer1::ITensor* attn_out =
            AddLinear(network, context_flat, c_proj_w, c_proj_b,
                      LayerName(scope + "attn_c_proj"));
        if (attn_out == nullptr) {
            return false;
        }
        nvinfer1::ITensor* after_attn = AddElementWise(
            network, hidden_state, attn_out, nvinfer1::ElementWiseOperation::kSUM,
            LayerName(scope + "attn_residual"));

        // ---- MLP ----
        nvinfer1::ITensor* norm2 =
            AddLayerNorm(network, after_attn, ln2_w, ln2_b, cfg.layer_norm_epsilon,
                         LayerName(scope + "ln_2"));
        nvinfer1::ITensor* mlp =
            AddLinear(network, norm2, c_fc_w, c_fc_b, LayerName(scope + "c_fc"));
        if (mlp == nullptr) {
            return false;
        }
        nvinfer1::IActivationLayer* gelu =
            network->addActivation(*mlp, nvinfer1::ActivationType::kGELU_TANH);
        // 数值定位：第 0 层 MLP 的两个切点（gelu 前 / gelu 后）。
        // "NaN 是 c_fc 造出来的、还是 gelu 造出来的"是这次排查的关键分界，
        // 在图上留两个输出比逐次猜算子便宜（见 TROUBLESHOOTING #18）。
        if (export_kv && layer == 0) {
            mlp->setName("mlp_fc_0");
            network->markOutput(*mlp);
        }
        if (gelu == nullptr) {
            return false;
        }
        // gelu_new 就是 tanh 近似，与 kGELU_TANH 是同一个公式。
        gelu->setName(LayerName(scope + "gelu_new").c_str());
        if (export_kv && layer == 0) {
            gelu->getOutput(0)->setName("mlp_gelu_0");
            network->markOutput(*gelu->getOutput(0));
        }
        nvinfer1::ITensor* mlp_out =
            AddLinear(network, gelu->getOutput(0), c_mlp_w, c_mlp_b,
                      LayerName(scope + "mlp_c_proj"));
        if (mlp_out == nullptr) {
            return false;
        }
        hidden_state = AddElementWise(network, after_attn, mlp_out,
                                      nvinfer1::ElementWiseOperation::kSUM,
                                      LayerName(scope + "mlp_residual"));
        if (hidden_state == nullptr) {
            return false;
        }

        // 数值定位用的中途输出（仅第 0 层，且只在导出 K/V 的 stage）：
        // "某层的 K/V 出现 NaN 时，NaN 是来自本层的注意力还是 MLP" 是排查中反复要回答的问题，
        // 而在图上留两个输出就能直接读出来（成本：每层两份 [B,S,H]，第 0 层可忽略）。
        // 见 docs/TROUBLESHOOTING.md #18 的定位过程。
        if (export_kv && layer == 0) {
            after_attn->setName("attn_res_0");
            hidden_state->setName("mlp_res_0");
            network->markOutput(*after_attn);
            network->markOutput(*hidden_state);
        }

        // 把每层的 K/V 暴露成网络输出：
        //   prefill → 写进分页 cache；
        //   decode  → 下一轮追加进 cache。
        if (export_kv) {
            const std::string k_name = "k_layer" + std::to_string(layer);
            const std::string v_name = "v_layer" + std::to_string(layer);
            // **注意（实测结论，勿照直觉改）**：弱类型网络（`createNetworkV2(0U)`）里
            // 导出的 K/V 与 logits 的实际类型**由 TRT 决定，而非 `weight_dtype`**——
            // FP16 引擎里它们都是 FP32。曾经试过在 markOutput 前插 `addCast` 去"钉死"
            // 类型，**实测无效**（重建后仍是 FP32）：弱类型网络下 Cast 只是精度提示，
            // 锁不住 I/O 类型。消费方必须**查询**引擎声明的类型，不能假定。
            // 排查与证据见 docs/TROUBLESHOOTING.md #18。
            q_kv[1]->setName(k_name.c_str());
            q_kv[2]->setName(v_name.c_str());
            network->markOutput(*q_kv[1]);
            network->markOutput(*q_kv[2]);
        }
    }

    // ---- 最终 LayerNorm + LM head ----
    nvinfer1::ITensor* ln_f_w =
        AddWeightConstant(network, weights, "ln_f.weight", dtype, Dims3(1, 1, hidden));
    nvinfer1::ITensor* ln_f_b =
        AddWeightConstant(network, weights, "ln_f.bias", dtype, Dims3(1, 1, hidden));
    if (ln_f_w == nullptr || ln_f_b == nullptr) {
        return false;
    }
    nvinfer1::ITensor* final_norm =
        AddLayerNorm(network, hidden_state, ln_f_w, ln_f_b, cfg.layer_norm_epsilon,
                     LayerName("ln_f"));

    nvinfer1::ITensor* head_w = nullptr;
    if (cfg.tie_word_embeddings) {
        // 与 wte 共享权重：wte 是 [V,H]，LM head 需要 [H,V]。
        // 在图上转置（TRT 会把它折成常量），而不是让转换工具物化一份
        // 154MB 的冗余拷贝。
        nvinfer1::ITensor* transposed =
            AddTranspose(network, wte, Dims2(1, 0), LayerName("lm_head_transpose"));
        // reshape 目标写全三个静态维，**不能**用 0 占位符：0 占位只在
        // "输出维 i 能对应到输入维 i" 时成立，这里输入是 rank-2 的 [H,V]，
        // 输出第 2 维没有对应的输入维，写 0 会被当成字面量 0（reshape 到空张量）。
        head_w = AddReshape(network, transposed, Dims3(1, hidden, vocab),
                            LayerName("lm_head_weight"));
    } else {
        head_w = AddWeightConstant(network, weights, "lm_head.weight", dtype,
                                   Dims3(1, hidden, vocab));
    }
    if (head_w == nullptr) {
        return false;
    }

    nvinfer1::IMatrixMultiplyLayer* logits_mm = network->addMatrixMultiply(
        *final_norm, nvinfer1::MatrixOperation::kNONE, *head_w,
        nvinfer1::MatrixOperation::kNONE);
    if (logits_mm == nullptr) {
        MINI_TRT_LOG_ERROR("GPT-2 build: LM head matmul failed");
        return false;
    }
    logits_mm->setName(LayerName("logits").c_str());
    // 同 K/V：输出类型由 TRT 决定（FP16 引擎下实测为 FP32），不在图上钉（见上方说明）
    logits_mm->getOutput(0)->setName("logits");
    network->markOutput(*logits_mm->getOutput(0));
    return true;
}

}  // namespace mini_trt_llm
