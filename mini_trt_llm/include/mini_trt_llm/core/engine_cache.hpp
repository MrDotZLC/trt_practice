#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mini_trt_llm {

// 引擎缓存的"构建指纹"。
//
// **要解决的问题**：`.engine` 缓存一直只按**路径名**复用（`/tmp/mini_trt_llm_*.engine`），
// 代码或建图开关改了它也不会失效，于是"拿旧引擎比新代码"这种事故只能靠人记得删 /tmp 来避免
// （`PROGRESS.md` §2.15 记着的坑；`TROUBLESHOOTING.md` #34 就是被它牵连的一次）。
// 这里把"这份引擎是不是当前配置/当前代码产出的"变成可比对的字符串：
// **指纹不一致 → 重建；没有指纹 → 视为不可信，也重建**。
//
// **指纹是什么、不是什么**：
//   · 是**缓存键**——用于判断"能不能复用"，变更敏感即可；
//   · 不是防篡改手段——用 64-bit FNV-1a 就够，不需要密码学强度（要防篡改请用 sha256）。
//
// **已知限制**（写在这里，免得下一个人以为是 bug）：源文件身份取的是 `size + mtime`，
// 若用"保留 mtime"的方式覆盖模型文件，指纹不会变——此时需要显式删引擎或 bump graph_version。
struct EngineFingerprintInputs {
    // 建的是哪一问切面：single / prefill / decode（不同切面是不同图）。
    std::string stage;
    // 精度与来源：fp32/fp16/int8、config/onnx。int8 由 Q/DQ 图决定，但仍要参与（图不同）。
    std::string precision;
    std::string source_kind;
    // 参与指纹的源文件（如 config.json / model.safetensors / *.onnx）——**身份**取 size+mtime。
    std::vector<std::string> source_files;
    // 影响建图的数值参数（seq/batch 范围、workspace 等）。
    std::vector<std::pair<std::string, int64_t>> numeric_params;
    // 影响建图的开关（export_diagnostics / detailed_profiling 等）。
    std::vector<std::pair<std::string, bool>> flags;
    // 运行环境与代码代次。
    std::string trt_version;
    int32_t cuda_runtime_version = 0;
    // 手工维护的"图版本"：**任何改动建图/精度/插件行为的代码变更都要 +1**。
    // 为什么手工：运行期无法自动察觉"图代码变了"；把编译时间写进指纹虽然能自动，
    // 但会让任何一次无关重编都强制重建引擎（分钟级），代价不成比例。
    int32_t graph_version = 0;
};

// 规范化文本（便于人读、也便于排错时看清"到底哪一项变了"）。
std::string CanonicalFingerprintText(const EngineFingerprintInputs& inputs);

// 64-bit FNV-1a 十六进制；同一个输入必然同一个值，任何一个字段变化都要改变它。
std::string ComputeEngineFingerprint(const EngineFingerprintInputs& inputs);

// 引擎旁的 sidecar 路径：`<engine_path>.fingerprint`。
std::string EngineFingerprintPath(const std::string& engine_path);

// 写 sidecar（含规范化文本，便于人工比对）。
bool WriteEngineFingerprint(const std::string& engine_path, const std::string& fingerprint,
                            const EngineFingerprintInputs& inputs);

// 读回指纹；文件不存在或读不出 → 返回空串（**空串表示"不可信"，调用方必须重建**）。
std::string ReadEngineFingerprint(const std::string& engine_path);

// 缓存是否可复用：引擎文件存在**且** sidecar 指纹与当前指纹相同。
// 缺 sidecar 一律视为不可复用——"旧引擎没有指纹"是常态（本功能之前建的），不能默认信任。
bool EngineCacheIsFresh(const std::string& engine_path, const std::string& fingerprint);

}  // namespace mini_trt_llm
