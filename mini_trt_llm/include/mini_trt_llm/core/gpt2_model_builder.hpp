#pragma once

#include "mini_trt_llm/core/imodel_builder.hpp"

#include <cstdint>
#include <set>
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
    // 把权重挂成常量层。
    //
    // **与文件级同名函数不是一回事**：本方法多一层"这个权重是否在量化清单里"的分派
    // （REQ-017 路线 C），清单没点名时才落到那条纯 FP32/FP16 的老路径。
    //
    // 为什么用同名成员而不是给每个调用点加参数：改签名要动 14 个调用点，而同名成员在类
    // 作用域里天然遮蔽文件级函数（名字查找先查类作用域），调用点一个字都不用改。
    nvinfer1::ITensor* AddWeightConstant(nvinfer1::INetworkDefinition* network,
                                        const WeightLoader& weights,
                                        const std::string& name,
                                        nvinfer1::DataType dtype,
                                        const nvinfer1::Dims& dims);

    // 取"量化源"——**int8 常量本身**，不挂 DQ。
    //
    // 为什么需要它：有的权重在图上要先做形状操作才参与计算（GPT-2 的 `wte` 既被 gather
    // 又被转置给 lm_head）。DQ 的输出**不是常量**，对它做 `ITransposeLayer` 会退化成
    // "每个 decode 步真的转一遍整张权重"——收益归零且不报错。所以这类权重必须
    // **先做形状操作、后 DQ**，本方法就是给它们留的入口。
    //
    // 契约（调用方靠它区分三种结果）：
    //   返回非空                → 取到 int8 常量，`*out_entry` = 对应条目；
    //   返回空且 `*out_entry` 为空 → 该权重**不在清单里**（走原有 FP32/FP16 路径）；
    //   返回空且 `*out_entry` 非空 → **失败**（已打日志）——调用方必须失败返回，
    //                             绝不能退回 FP32（那正是"声明了量化却没量化"）。
    nvinfer1::ITensor* AddQuantizedWeightSource(nvinfer1::INetworkDefinition* network,
                                                const WeightLoader& weights,
                                                const std::string& name,
                                                const nvinfer1::Dims& dims,
                                                const QuantEntry** out_entry);

    // 在任意（可能是 int8、也可能已经过 gather/转置）张量上挂 DQ，得到可参与计算的浮点张量。
    // 反量化公式：y = q * scale（对称量化，zero_point = 0）。
    nvinfer1::ITensor* AddDequantize(nvinfer1::INetworkDefinition* network,
                                     nvinfer1::ITensor* quantized,
                                     const QuantEntry& entry,
                                     nvinfer1::DataType dtype,
                                     const std::string& name);

    // 建网期间需要存活到 buildSerializedNetwork 之后的主机侧缓冲。
    //
    // nvinfer1::Weights 只持有裸指针，而 addConstant 是否复制数据属于实现细节；
    // 把缓冲挂在 builder 对象上（它的生命周期覆盖整个 BuildFromConfig）比赌
    // "TRT 会复制" 更安全——失效的常量不会崩，只会静默建出错误的网络。
    // 两个缓冲都按 FP32 存放，需要 FP16 时由图内的 Cast 层转换
    // （直接把 float 缓冲标成 kHALF 会让 TRT 按 half 解释 float 的位模式）。
    std::vector<float> causal_mask_;
    float attn_scale_value_ = 1.0f;

    // ---- REQ-017 路线 C ----
    // 本次 Build 的量化清单（由 BuildOptions 透传）。生命周期由 EngineBuilder 保证：
    // 它和 buildSerializedNetwork 在同一个作用域里。
    const QuantSpec* quant_ = nullptr;
    // "清单里哪些条目真的被建进了图"。收尾时用它拒绝"清单有、图里没有"的条目——
    // 少了这道闸，清单与模型不匹配会静默退化成纯 FP32 引擎。
    std::set<std::string> quant_consumed_;
    // DQ 的 scale / zeroPoint 常量缓冲。与 causal_mask_ 同一条理由：nvinfer1::Weights 只持
    // 裸指针，必须活到 buildSerializedNetwork 之后。外层 vector 扩容不会搬内层缓冲的堆地址，
    // 所以已交给 TRT 的指针在后续 push_back 之后仍然有效。
    std::vector<std::vector<char>> quant_const_buffers_;
};

}  // namespace mini_trt_llm
