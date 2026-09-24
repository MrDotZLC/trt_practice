#include "mini_trt_llm/core/weight_loader.hpp"
#include "mini_trt_llm/utils/logger.hpp"

namespace mini_trt_llm {

WeightLoader::WeightLoader() = default;
WeightLoader::~WeightLoader() = default;

bool WeightLoader::Load(const std::string& model_dir) {
    model_dir_ = model_dir;
    if (!model_dir_.empty() && model_dir_.back() != '/') {
        model_dir_ += '/';
    }
    std::string path = model_dir_ + "model.safetensors";
    if (!loader_.LoadFromFile(path)) {
        MINI_TRT_LOG_ERROR("Failed to load weights from: " << path);
        return false;
    }
    return true;
}

void WeightLoader::SetWeightMap(const JsonValue& map) {
    weight_map_.clear();
    if (!map.IsObject()) {
        return;
    }
    for (const auto& [trt_name, source_key_value] : map.AsObject()) {
        if (source_key_value.IsString()) {
            weight_map_[trt_name] = source_key_value.AsString();
        }
    }
}

const void* WeightLoader::GetWeightBySourceKey(const std::string& source_key,
                                               nvinfer1::DataType target_type,
                                               size_t* bytes) const {
    if (!loader_.HasTensor(source_key)) {
        MINI_TRT_LOG_WARN("Weight not found in safetensors: " << source_key);
        return nullptr;
    }
    return loader_.GetConvertedData(source_key, target_type, bytes);
}

const void* WeightLoader::GetWeight(const std::string& trt_name,
                                    nvinfer1::DataType target_type,
                                    size_t* bytes) const {
    auto it = weight_map_.find(trt_name);
    if (it != weight_map_.end()) {
        return GetWeightBySourceKey(it->second, target_type, bytes);
    }

    // 兜底：尝试直接用 trt_name 作为 source key
    return GetWeightBySourceKey(trt_name, target_type, bytes);
}

bool WeightLoader::HasWeight(const std::string& trt_name) const {
    auto it = weight_map_.find(trt_name);
    if (it != weight_map_.end()) {
        return loader_.HasTensor(it->second);
    }
    return loader_.HasTensor(trt_name);
}

}  // namespace mini_trt_llm
