#pragma once

#include "mini_trt_llm/core/imodel_builder.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mini_trt_llm {

// GPT-2 超参数，来自 config.json 的 hyper_params。
//
// 这些字段都来自转换工具产出的原生 config（见 tools/convert/hf_to_mini_trt_llm.py），
// 不从 HuggingFace 原始字段里猜。
struct GPT2Config {
    int32_t n_layer = 12;
    int32_t n_head = 12;
    int32_t n_embd = 768;
    int32_t n_positions = 1024;
    int32_t vocab_size = 50257;
    float layer_norm_epsilon = 1e-5f;

    // 权重与 LM head 是否共享（GPT-2 为 true：文件里没有 lm_head.weight）。
    bool tie_word_embeddings = true;

    // 每个 Paged KV Cache block 容纳多少 token。
    // 强制显式配置、不提供默认值：block_size 会同时影响 cache 显存布局与
    // PagedAttention 插件的属性，给个隐式默认值只会让跨模型排错变难。
    int32_t block_size = 0;

    int32_t head_size() const { return n_head > 0 ? n_embd / n_head : 0; }

    // KV Cache 的总 block 数：按最大位置数向上取整。
    int32_t num_blocks() const {
        return block_size > 0 ? (n_positions + block_size - 1) / block_size : 0;
    }

    // 解析 hyper_params 并做一致性校验。任一项不合法时返回 false 并记录原因。
    static bool FromModelConfig(const ModelConfig& config, GPT2Config* out);
};

// 构建 GPT-2 网络所需的全部权重名（TRT 侧名字，与 weight_map 的 key 一致）。
//
// 单独暴露它的用途是让 host 侧用例在**没有 GPU** 的情况下校验
// "转换产物是否覆盖了建模所需的每一个权重"——这类漂移（转换工具少导一张表、
// key 改名）在真机上表现为数值错误，在 host 侧却是可以立刻断言的事实。
std::vector<std::string> GPT2WeightNames(const GPT2Config& config);

// 网络里所有层名的统一前缀，便于在 profiler 与报错信息里一眼认出本模型。
inline constexpr char kGpt2LayerPrefix[] = "mini_trt_llm_gpt2_";

class GPT2ModelBuilder : public IModelBuilder {
 public:
    std::string Name() const override { return "gpt2"; }

    bool Build(nvinfer1::INetworkDefinition* network, const WeightLoader& weights,
               const ModelConfig& config, const BuildOptions& options) override;

 private:
    // 建网期间需要存活到 buildSerializedNetwork 之后的主机侧缓冲。
    //
    // nvinfer1::Weights 只持有裸指针，而 addConstant 是否复制数据属于实现细节；
    // 把缓冲挂在 builder 对象上（它的生命周期覆盖整个 BuildFromConfig）比赌
    // "TRT 会复制" 更安全——失效的常量不会崩，只会静默建出错误的网络。
    // 两个缓冲都按 FP32 存放，需要 FP16 时由图内的 Cast 层转换
    // （直接把 float 缓冲标成 kHALF 会让 TRT 按 half 解释 float 的位模式）。
    std::vector<float> causal_mask_;
    float attn_scale_value_ = 1.0f;
};

}  // namespace mini_trt_llm
