#pragma once

#include "mini_trt_llm/utils/json.hpp"
#include <string>

namespace mini_trt_llm {

// 描述一个模型的配置。
// 通过 JSON 文件加载，支持配置驱动构建。
struct ModelConfig {
    // 模型类型，如 "gpt2", "resnet18"
    std::string model_type;

    // 架构类型，如 "decoder_only", "cnn", "encoder_decoder"
    std::string architecture;

    // 模型超参数（层数、 hidden size 等）
    JsonValue hyper_params;

    // 权重映射：TRT 层权重名 -> Safetensors 中的 source key
    JsonValue weight_map;

    // 从 model_dir/config.json 加载配置
    static ModelConfig Load(const std::string& model_dir);
};

}  // namespace mini_trt_llm
