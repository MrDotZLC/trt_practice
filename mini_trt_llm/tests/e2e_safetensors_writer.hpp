#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace mini_trt_llm {
namespace test_support {

// 测试用的 Safetensors 写入 helper。
//
// 放在 tests/ 而非产品代码：写入能力只有测试夹具需要，放进 include/ 会污染产品 API。
// 有了它，端到端测试可以在临时目录里自包含地造出模型文件——既不用往仓库提交二进制夹具，
// 也不用在测试运行时依赖 Python。
struct TensorSpec {
    enum class Dtype { kF32, kF16, kBF16 };

    std::vector<size_t> shape;
    Dtype dtype = Dtype::kF32;
    // 一律以 float 提供数据，写入时按 dtype 编码成对应位宽。
    std::vector<float> values;
};

// 把 {张量名 → TensorSpec} 写成一份合法的 .safetensors 文件。
// 返回 false 表示编码或写盘失败。
bool WriteSafetensorsFile(const std::string& path,
                          const std::map<std::string, TensorSpec>& tensors);

}  // namespace test_support
}  // namespace mini_trt_llm
