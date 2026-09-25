#include "mini_trt_llm/core/resnet18_model_builder.hpp"

#include "mini_trt_llm/core/builder.hpp"  // OnnxIoContractFor：两条路共用 I/O 名字
#include "mini_trt_llm/utils/logger.hpp"

#include <NvInfer.h>

#include <cstdint>
#include <string>

namespace mini_trt_llm {
namespace {

// ResNet18 的**结构**常量：它们是模型身份的一部分（与权重形状无关），所以写死在这里；
// 而通道数、类别数这些随权重走的量一律从权重形状推（见 Build）。
constexpr int32_t kStages = 4;
constexpr int32_t kBlocksPerStage = 2;
constexpr int32_t kStemKernel = 7;
constexpr int32_t kStemStride = 2;
constexpr int32_t kStemPad = 3;
constexpr int32_t kPoolKernel = 3;
constexpr int32_t kPoolStride = 2;
constexpr int32_t kPoolPad = 1;
constexpr int32_t kBlockKernel = 3;
constexpr int32_t kBlockPad = 1;
constexpr int32_t kDownsampleKernel = 1;

std::string LayerName(const std::string& name) { return "resnet18_" + name; }

size_t DtypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        default:
            return 0;
    }
}

int64_t Volume(const nvinfer1::Dims& dims) {
    int64_t total = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return -1;  // 含动态轴：交给调用方按实际字节数判断
        }
        total *= dims.d[i];
    }
    return total;
}

// 取权重并挂成常量。**带上"声明形状 vs 实际元素数"的交叉校验**：权重取错（同名不同张量、
// weight_map 写错、转换产物形状漂移）在这里就会失败，而不是等到建出引擎后表现为"数值不对"。
nvinfer1::ITensor* AddWeight(nvinfer1::INetworkDefinition* network,
                             const WeightLoader& weights, const std::string& name,
                             nvinfer1::DataType dtype, const nvinfer1::Dims& dims) {
    size_t bytes = 0;
    const void* data = weights.GetWeight(name, dtype, &bytes);
    if (data == nullptr) {
        MINI_TRT_LOG_ERROR("ResNet18 weight not found: " << name);
        return nullptr;
    }
    const size_t element_size = DtypeSize(dtype);
    if (element_size == 0) {
        MINI_TRT_LOG_ERROR("ResNet18 weight " << name << " has unsupported dtype");
        return nullptr;
    }
    const int64_t declared = Volume(dims);
    const int64_t actual = static_cast<int64_t>(bytes / element_size);
    if (declared >= 0 && declared != actual) {
        MINI_TRT_LOG_ERROR("ResNet18 weight " << name << ": declared shape wants " << declared
                           << " elements but tensor has " << actual);
        return nullptr;
    }
    nvinfer1::IConstantLayer* layer =
        network->addConstant(dims, nvinfer1::Weights{dtype, data, actual});
    if (layer == nullptr) {
        MINI_TRT_LOG_ERROR("ResNet18: constant layer failed for " << name);
        return nullptr;
    }
    layer->setName(LayerName(name).c_str());
    return layer->getOutput(0);
}

nvinfer1::ITensor* AddRelu(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                           const std::string& name) {
    nvinfer1::IActivationLayer* layer =
        network->addActivation(*input, nvinfer1::ActivationType::kRELU);
    if (layer == nullptr) {
        return nullptr;
    }
    layer->setName(LayerName(name).c_str());
    return layer->getOutput(0);
}

// Conv（含 bias）。BN 已在导出时折叠进权重，所以**没有**独立的 BN 层——
// 这也是原生路径能完全不依赖 Plugin 的原因。
nvinfer1::ITensor* AddConv(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                           const WeightLoader& weights, const std::string& prefix,
                           nvinfer1::DataType dtype, int32_t in_channels, int32_t out_channels,
                           int32_t kernel, int32_t stride, int32_t pad) {
    // 卷积的权重/偏置**直接以 nvinfer1::Weights 传给 addConvolutionNd**：
    // 这是该 API 的既有契约（不是 setInput 挂常量）。指针在本次 build 期间保持有效——
    // WeightLoader 的转换缓存已按张量名隔离（TROUBLESHOOTING #5），与 GPT-2 builder 依赖同一保证。
    const size_t element_size = DtypeSize(dtype);
    const int64_t expected_weight = static_cast<int64_t>(out_channels) * in_channels * kernel * kernel;
    size_t weight_bytes = 0;
    size_t bias_bytes = 0;
    const void* weight_data = weights.GetWeight(prefix + ".weight", dtype, &weight_bytes);
    const void* bias_data = weights.GetWeight(prefix + ".bias", dtype, &bias_bytes);
    if (weight_data == nullptr ||
        weight_bytes != static_cast<size_t>(expected_weight) * element_size) {
        MINI_TRT_LOG_ERROR("ResNet18 weight " << prefix << ".weight missing or wrong size (want "
                           << expected_weight << " elements)");
        return nullptr;
    }
    if (bias_data == nullptr ||
        bias_bytes != static_cast<size_t>(out_channels) * element_size) {
        MINI_TRT_LOG_ERROR("ResNet18 weight " << prefix << ".bias missing or wrong size");
        return nullptr;
    }
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(
        *input, out_channels, nvinfer1::DimsHW{kernel, kernel},
        nvinfer1::Weights{dtype, weight_data, expected_weight},
        nvinfer1::Weights{dtype, bias_data, out_channels});
    if (conv == nullptr) {
        return nullptr;
    }
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{pad, pad});
    conv->setName(LayerName(prefix).c_str());
    return conv->getOutput(0);
}

// 一个 BasicBlock：conv1(+relu) → conv2 → (+identity) → relu。
// 注意顺序：**残差相加在最后一个 relu 之前**——与 torchvision / ONNX 图一致；
// 顺序写反（先 relu 再 add）会让数值差一大截且很难从现象上看出来。
nvinfer1::ITensor* AddBasicBlock(nvinfer1::INetworkDefinition* network,
                                 nvinfer1::ITensor* input, const WeightLoader& weights,
                                 const std::string& prefix, nvinfer1::DataType dtype,
                                 int32_t in_channels, int32_t out_channels, int32_t stride,
                                 bool downsample) {
    nvinfer1::ITensor* identity = input;
    if (downsample) {
        identity = AddConv(network, input, weights, prefix + ".downsample.0", dtype,
                           in_channels, out_channels, kDownsampleKernel, stride, 0);
        if (identity == nullptr) {
            return nullptr;
        }
    }

    nvinfer1::ITensor* conv1 =
        AddConv(network, input, weights, prefix + ".conv1", dtype, in_channels, out_channels,
                kBlockKernel, stride, kBlockPad);
    if (conv1 == nullptr) {
        return nullptr;
    }
    nvinfer1::ITensor* relu1 = AddRelu(network, conv1, prefix + ".relu1");
    if (relu1 == nullptr) {
        return nullptr;
    }
    nvinfer1::ITensor* conv2 =
        AddConv(network, relu1, weights, prefix + ".conv2", dtype, out_channels, out_channels,
                kBlockKernel, 1, kBlockPad);
    if (conv2 == nullptr) {
        return nullptr;
    }
    nvinfer1::IElementWiseLayer* sum = network->addElementWise(
        *conv2, *identity, nvinfer1::ElementWiseOperation::kSUM);
    if (sum == nullptr) {
        return nullptr;
    }
    sum->setName(LayerName(prefix + ".add").c_str());
    return AddRelu(network, sum->getOutput(0), prefix + ".relu2");
}

}  // namespace

bool ResNet18ModelBuilder::Build(nvinfer1::INetworkDefinition* network,
                                 const WeightLoader& weights, const ModelConfig& config,
                                 const BuildOptions& options) {
    if (network == nullptr) {
        MINI_TRT_LOG_ERROR("ResNet18 build: network is null");
        return false;
    }
    if (options.stage != BuildStage::kSingle) {
        // CV 没有 prefill/decode 之分：静默忽略会建出一个语义不明的引擎（谁会用它？
        // 为什么只有一个 profile？），所以宁可失败。
        MINI_TRT_LOG_ERROR("ResNet18 build: only BuildStage::kSingle is supported "
                           "(CV models have no prefill/decode split)");
        return false;
    }

    // 输入的空间维只能来自 config（重量里没有"输入分辨率"这个信息）；
    // 通道数与类别数则从权重形状推，避免两处各写一份、彼此漂移。
    if (!config.hyper_params.Has("input_channels") ||
        !config.hyper_params.Has("input_height") ||
        !config.hyper_params.Has("input_width")) {
        MINI_TRT_LOG_ERROR("ResNet18 build: hyper_params must declare input_channels / "
                           "input_height / input_width");
        return false;
    }
    const int32_t width = config.hyper_params["input_width"].AsInt();
    const int32_t height = config.hyper_params["input_height"].AsInt();
    if (width <= 0 || height <= 0) {
        MINI_TRT_LOG_ERROR("ResNet18 build: invalid input size " << width << "x" << height);
        return false;
    }

    // fc 的权重形状给出 [out_features, in_features]（见 config 的 source.fc_weight_layout）。
    // WeightLoader 只暴露"字节数"而不暴露形状，所以：类别数取 config 声明（host 用例
    // R0.2b 已把声明与文件里的实际形状对齐过），in_features 由字节数反推并校验整除。
    if (!config.hyper_params.Has("num_classes")) {
        MINI_TRT_LOG_ERROR("ResNet18 build: hyper_params must declare num_classes");
        return false;
    }
    const int32_t num_classes = config.hyper_params["num_classes"].AsInt();
    size_t fc_bytes = 0;
    if (weights.GetWeight("fc.weight", nvinfer1::DataType::kFLOAT, &fc_bytes) == nullptr ||
        num_classes <= 0 || fc_bytes % (sizeof(float) * static_cast<size_t>(num_classes)) != 0) {
        MINI_TRT_LOG_ERROR("ResNet18 build: fc.weight missing or inconsistent with num_classes="
                           << num_classes);
        return false;
    }
    const int32_t final_channels =
        static_cast<int32_t>(fc_bytes / (sizeof(float) * static_cast<size_t>(num_classes)));
    // ResNet18 的通道宽度是 64 * 2^(stage-1)，最后一级 512；stem 就是它的 1/8。
    // 这样写而不是硬编码 64：宽度由权重决定，将来换个宽度的卷积网也能走同一条代码。
    const int32_t stem_channels = final_channels / (1 << (kStages - 1));
    if (stem_channels <= 0 || stem_channels * (1 << (kStages - 1)) != final_channels) {
        MINI_TRT_LOG_ERROR("ResNet18 build: final channels " << final_channels
                           << " is not 64 * 2^(stage-1) 形式");
        return false;
    }

    // I/O 名字与 ONNX 路径共用同一份契约常量：两条路同名同义，否则"对齐"无从谈起。
    const OnnxIoContract io = OnnxIoContractFor(config.architecture);
    // 输入固定声明成 FP32：像素质由调用方以 float 提供（CVRunner 的契约），内部按
    // weight_dtype 计算由 TRT 插 cast——这样 FP32 / FP16 引擎对调用方的接口一致。
    nvinfer1::ITensor* input = network->addInput(
        io.input_name, nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims{4, {-1, config.hyper_params["input_channels"].AsInt(), height, width}});
    if (input == nullptr) {
        MINI_TRT_LOG_ERROR("ResNet18 build: failed to add input");
        return false;
    }

    const nvinfer1::DataType dtype = options.weight_dtype;
    nvinfer1::ITensor* hidden =
        AddConv(network, input, weights, "conv1", dtype,
                config.hyper_params["input_channels"].AsInt(), stem_channels, kStemKernel,
                kStemStride, kStemPad);
    if (hidden == nullptr) {
        return false;
    }
    hidden = AddRelu(network, hidden, "relu_stem");
    if (hidden == nullptr) {
        return false;
    }
    nvinfer1::IPoolingLayer* pool = network->addPoolingNd(
        *hidden, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{kPoolKernel, kPoolKernel});
    if (pool == nullptr) {
        return false;
    }
    pool->setStrideNd(nvinfer1::DimsHW{kPoolStride, kPoolStride});
    pool->setPaddingNd(nvinfer1::DimsHW{kPoolPad, kPoolPad});
    pool->setName(LayerName("maxpool").c_str());
    hidden = pool->getOutput(0);

    int32_t in_channels = stem_channels;
    for (int32_t stage = 1; stage <= kStages; ++stage) {
        const int32_t out_channels = stem_channels * (1 << (stage - 1));
        for (int32_t block = 0; block < kBlocksPerStage; ++block) {
            // 每个 stage 的第一个 block 做 2 倍下采样；stage 1 不需要（空间已由 stem+pool 降够）
            const bool downsample =
                (stage > 1) && (block == 0) && (out_channels != in_channels);
            const int32_t stride = downsample ? 2 : 1;
            const std::string prefix = "layer" + std::to_string(stage) + "." +
                                       std::to_string(block);
            hidden = AddBasicBlock(network, hidden, weights, prefix, dtype, in_channels,
                                   out_channels, stride, downsample);
            if (hidden == nullptr) {
                return false;
            }
            in_channels = out_channels;
        }
    }
    if (in_channels != final_channels) {
        MINI_TRT_LOG_ERROR("ResNet18 build: final channels " << in_channels
                           << " do not match fc in_features " << final_channels);
        return false;
    }

    // GlobalAveragePool：在 H/W 两维上取平均。用 reduce 而不是定尺寸 pooling，
    // 这样输入分辨率变化时不必改常数（TRT 的 reduce 在参数上更直接表达"全局"）。
    // axes mask 的 bit i 对应第 i 个维度 → bit2|bit3 = H,W。
    nvinfer1::IReduceLayer* gap = network->addReduce(
        *hidden, nvinfer1::ReduceOperation::kAVG, /*axes=*/0xCU, /*keepDims=*/true);
    if (gap == nullptr) {
        return false;
    }
    gap->setName(LayerName("gap").c_str());

    // [B,C,1,1] → [B,C]：0 表示沿用输入对应维（动态 batch 不必显式知道）。
    nvinfer1::IShuffleLayer* flatten = network->addShuffle(*gap->getOutput(0));
    if (flatten == nullptr) {
        return false;
    }
    flatten->setReshapeDimensions(nvinfer1::Dims{2, {0, final_channels}});
    flatten->setZeroIsPlaceholder(true);
    flatten->setName(LayerName("flatten").c_str());

    // fc：权重存的是 [out, in]（config 里声明为 out_in_transB），所以右乘时转置——
    // **不在主机侧转置**，避免"转换时转一次、建图时又转一次"。
    const nvinfer1::Dims fc_dims{
        2, {num_classes, final_channels}};
    nvinfer1::ITensor* fc_weight =
        AddWeight(network, weights, "fc.weight", dtype, fc_dims);
    // bias 声明成 rank-2 `[1, num_classes]` 而不是 rank-1：TRT 的 element-wise 要求两侧
    // **rank 相同**，而 matmul 的输出是 `[B, 1000]`。写成 rank-1 时 Build() 会"成功"，
    // 但引擎构建阶段报 `Assertion x.nbDims == y.nbDims failed`
    // （elementWiseNode.cpp）——又是一次"延迟报错"，见 TROUBLESHOOTING #24。
    nvinfer1::ITensor* fc_bias = AddWeight(network, weights, "fc.bias", dtype,
                                           nvinfer1::Dims{2, {1, num_classes}});
    if (fc_weight == nullptr || fc_bias == nullptr) {
        return false;
    }
    nvinfer1::IMatrixMultiplyLayer* matmul = network->addMatrixMultiply(
        *flatten->getOutput(0), nvinfer1::MatrixOperation::kNONE, *fc_weight,
        nvinfer1::MatrixOperation::kTRANSPOSE);
    if (matmul == nullptr) {
        MINI_TRT_LOG_ERROR("ResNet18 build: fc matmul failed");
        return false;
    }
    matmul->setName(LayerName("fc_matmul").c_str());
    nvinfer1::IElementWiseLayer* logits = network->addElementWise(
        *matmul->getOutput(0), *fc_bias, nvinfer1::ElementWiseOperation::kSUM);
    if (logits == nullptr) {
        return false;
    }
    logits->setName(LayerName("fc_bias_add").c_str());
    logits->getOutput(0)->setName(io.output_name);
    network->markOutput(*logits->getOutput(0));
    return true;
}

}  // namespace mini_trt_llm
