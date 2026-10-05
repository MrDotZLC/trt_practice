#include "mini_trt_llm/core/quant_spec.hpp"

#include "mini_trt_llm/utils/io.hpp"
#include "mini_trt_llm/utils/logger.hpp"

#include <cmath>
#include <filesystem>
#include <string>

namespace mini_trt_llm {
namespace {

// 清单里的路径按"相对清单所在目录"解释；绝对路径原样使用。
// 这样脚本既可以在模型目录内原地生成产物，也可以 `--output-dir` 到别处。
std::string ResolveAgainst(const std::string& base_dir, const std::string& path) {
    const std::filesystem::path candidate(path);
    if (candidate.is_absolute()) {
        return candidate.lexically_normal().string();
    }
    return (std::filesystem::path(base_dir) / candidate).lexically_normal().string();
}

bool GetString(const JsonValue& object, const char* key, std::string* out) {
    if (!object.Has(key) || !object[key].IsString()) {
        return false;
    }
    *out = object[key].AsString();
    return true;
}

// scale 必须是有限正数：0 会让 `y = q * scale` 恒为 0（静默把整层清零），
// 而 inf/NaN 会在建图期就污染整张图且不报错。
bool IsUsableScale(double value) {
    return std::isfinite(value) && value > 0.0;
}

}  // namespace

bool QuantSpec::LoadFromFile(const std::string& path, QuantSpec* out) {
    if (out == nullptr) {
        return false;
    }

    JsonValue json;
    try {
        json = LoadJson(path);
    } catch (const std::exception& e) {
        MINI_TRT_LOG_ERROR("量化清单读取失败: " << path << " (" << e.what() << ")");
        return false;
    }
    if (!json.IsObject()) {
        MINI_TRT_LOG_ERROR("量化清单不是 JSON 对象: " << path);
        return false;
    }

    const int32_t version = (json.Has("format_version") && json["format_version"].IsNumber())
                                ? json["format_version"].AsInt()
                                : -1;
    if (version != 1) {
        MINI_TRT_LOG_ERROR("量化清单 format_version 不支持: " << version
                            << "（本版本只认 1）；路径 " << path);
        return false;
    }

    std::string scheme;
    if (!GetString(json, "scheme", &scheme) || scheme != "symmetric_per_tensor") {
        MINI_TRT_LOG_ERROR("量化清单 scheme 不支持: '" << scheme
                            << "'（本版本只认 symmetric_per_tensor；per_channel 要先做 D2 的"
                            << "前置，见 design.md）");
        return false;
    }

    // 非零零点会让反量化多一个减法；本轮不实现它，**明确拒绝**而不是忽略这个字段——
    // 忽略会建出一张与清单语义不符的图。
    if (!json.Has("zero_point") || !json["zero_point"].IsNumber() ||
        json["zero_point"].AsInt() != 0) {
        MINI_TRT_LOG_ERROR("量化清单的 zero_point 必须是 0（本轮只支持对称量化）");
        return false;
    }

    const std::filesystem::path manifest_dir =
        std::filesystem::path(path).parent_path();
    std::string int8_path;
    if (!json.Has("int8_weights") || !GetString(json["int8_weights"], "path", &int8_path)) {
        MINI_TRT_LOG_ERROR("量化清单缺少 int8_weights.path");
        return false;
    }
    std::string source_path;
    if (!json.Has("weights_source") ||
        !GetString(json["weights_source"], "path", &source_path)) {
        MINI_TRT_LOG_ERROR("量化清单缺少 weights_source.path");
        return false;
    }

    if (!json.Has("entries") || !json["entries"].IsArray() || json["entries"].Size() == 0) {
        MINI_TRT_LOG_ERROR("量化清单的 entries 必须是非空数组");
        return false;
    }

    QuantSpec loaded;
    const auto& raw_entries = json["entries"].AsArray();
    for (size_t i = 0; i < raw_entries.size(); ++i) {
        const JsonValue& raw = raw_entries[i];
        if (!raw.IsObject()) {
            MINI_TRT_LOG_ERROR("量化清单 entries[" << i << "] 不是对象");
            return false;
        }

        QuantEntry entry;
        if (!GetString(raw, "tensor", &entry.tensor) || entry.tensor.empty()) {
            MINI_TRT_LOG_ERROR("量化清单 entries[" << i << "] 缺 tensor");
            return false;
        }
        if (!GetString(raw, "source_key", &entry.source_key) || entry.source_key.empty()) {
            MINI_TRT_LOG_ERROR("量化清单 entries[" << i << "] 缺 source_key");
            return false;
        }
        if (!GetString(raw, "granularity", &entry.granularity) ||
            entry.granularity != "per_tensor") {
            MINI_TRT_LOG_ERROR("量化清单 entries[" << i << "] 的 granularity 不是 per_tensor: '"
                                << entry.granularity << "'");
            return false;
        }
        if (raw.Has("axis") && raw["axis"].IsNumber()) {
            entry.axis = raw["axis"].AsInt();
        }
        if (!raw.Has("scales") || !raw["scales"].IsArray() ||
            raw["scales"].Size() != 1 ||
            !IsUsableScale(raw["scales"][0].IsNumber() ? raw["scales"][0].AsNumber() : 0.0)) {
            MINI_TRT_LOG_ERROR("量化清单 entries[" << i << "] 的 scales 必须是"
                                " 1 个有限正数（tensor=" << entry.tensor << "）");
            return false;
        }
        entry.scales.push_back(static_cast<float>(raw["scales"][0].AsNumber()));

        if (loaded.by_tensor_.count(entry.tensor) != 0 ||
            loaded.by_source_key_.count(entry.source_key) != 0) {
            MINI_TRT_LOG_ERROR("量化清单里出现重复条目: tensor=" << entry.tensor
                                << ", source_key=" << entry.source_key);
            return false;
        }
        loaded.by_tensor_[entry.tensor] = loaded.entries_.size();
        loaded.by_source_key_[entry.source_key] = loaded.entries_.size();
        loaded.entries_.push_back(std::move(entry));
    }

    loaded.int8_weights_path_ =
        ResolveAgainst(manifest_dir.string(), int8_path);
    loaded.weights_source_path_ =
        ResolveAgainst(manifest_dir.string(), source_path);
    *out = std::move(loaded);
    return true;
}

const QuantEntry* QuantSpec::Find(const std::string& name) const {
    auto by_tensor = by_tensor_.find(name);
    if (by_tensor != by_tensor_.end()) {
        return &entries_[by_tensor->second];
    }
    auto by_source = by_source_key_.find(name);
    if (by_source != by_source_key_.end()) {
        return &entries_[by_source->second];
    }
    return nullptr;
}

}  // namespace mini_trt_llm
