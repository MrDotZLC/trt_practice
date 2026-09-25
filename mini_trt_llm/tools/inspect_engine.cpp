// 读出一个已构建 `.engine` 的 I/O 契约：名字、方向、**声明精度**、维数。
//
// 为什么需要它：TensorRT 在**弱类型网络**里自行决定边界张量的类型——实测 FP16 引擎的
// K/V 与 logits 输出都是 **FP32**，而输入类型则是 `addInput` 时声明的那样（锁得住）。
// 没有这个工具，"引擎到底声明了什么"只能靠推断，而推断在本项目的记录里已被证伪两次
// （见 docs/TROUBLESHOOTING.md #18 / #17）。
//
// 特点：只做 `deserializeCudaEngine` + 读元信息，**不建 context、不推理、不写数据**；
// 因此它也是"引擎文件能否被当前环境反序列化"的最小探针。
//
// 编译（无 CMake 目标，按需手动编译——工具不进构建流程）：
//   g++ -O2 -std=c++17 \
//       -I/usr/include/x86_64-linux-gnu -I/usr/local/cuda/include \
//       mini_trt_llm/tools/inspect_engine.cpp -o /tmp/inspect_engine \
//       -L/usr/lib/x86_64-linux-gnu -lnvinfer
//
// 用法：
//   /tmp/inspect_engine /tmp/mini_trt_llm_gpt2_real_decode_fp16.engine
//
// 注意：反序列化需要 CUDA 初始化（实测无 GPU 环境会报 `CUDA initialization failure
// with error: 35`，与 `createInferBuilder` 同一限制）——所以它只能在真机跑。

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

const char* DtypeName(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kINT64:
            return "INT64";
        default:
            return "其它";
    }
}

std::string DimsToString(const nvinfer1::Dims& dims) {
    std::string out = "[";
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(dims.d[i]);
    }
    return out + "]";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <engine 文件路径>\n";
        return 1;
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in.good()) {
        std::cerr << "打不开: " << argv[1] << "\n";
        return 1;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());

    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (runtime == nullptr) {
        std::cerr << "createInferRuntime 返回空（通常是没有 CUDA 设备）\n";
        return 2;
    }
    nvinfer1::ICudaEngine* engine =
        runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (engine == nullptr) {
        std::cerr << "反序列化失败（引擎可能来自不同的 TRT 版本，"
                     "或缺少自定义 plugin 的 creator）\n";
        delete runtime;
        return 3;
    }

    std::cout << argv[1] << "  (" << blob.size() / 1024 / 1024 << " MB)\n";
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        const bool is_input =
            engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        std::cout << "  " << (is_input ? "输入 " : "输出 ") << name << "  "
                  << DtypeName(engine->getTensorDataType(name)) << "  "
                  << DimsToString(engine->getTensorShape(name)) << "\n";
    }

    delete engine;
    delete runtime;
    return 0;
}
