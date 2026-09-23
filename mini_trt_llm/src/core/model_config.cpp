#include "mini_trt_llm/core/model_config.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include <stdexcept>

namespace mini_trt_llm {

ModelConfig ModelConfig::Load(const std::string& model_dir) {
    std::string config_path = model_dir;
    if (config_path.back() != '/') {
        config_path += '/';
    }
    config_path += "config.json";

    JsonValue json = LoadJson(config_path);
    if (!json.IsObject()) {
        throw std::runtime_error("config.json must be a JSON object");
    }

    ModelConfig config;
    if (json.Has("model_type")) {
        config.model_type = json["model_type"].AsString();
    } else {
        throw std::runtime_error("config.json missing 'model_type'");
    }

    if (json.Has("architecture")) {
        config.architecture = json["architecture"].AsString();
    } else {
        throw std::runtime_error("config.json missing 'architecture'");
    }

    if (json.Has("hyper_params")) {
        config.hyper_params = json["hyper_params"];
    } else {
        config.hyper_params = JsonValue(JsonValue::Object{});
    }

    if (json.Has("weight_map")) {
        config.weight_map = json["weight_map"];
    } else {
        config.weight_map = JsonValue(JsonValue::Object{});
    }

    MINI_TRT_LOG_INFO("Loaded model config: " << config.model_type
                    << " (" << config.architecture << ")");
    return config;
}

}  // namespace mini_trt_llm
