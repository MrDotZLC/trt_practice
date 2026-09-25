#include "mini_trt_llm/core/cv_runner.hpp"

#include "mini_trt_llm/utils/cuda_check.hpp"
#include "mini_trt_llm/utils/logger.hpp"
#include "mini_trt_llm/utils/memory_pool.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace mini_trt_llm {

std::vector<float> NormalizePixelsToNchw(const std::vector<float>& pixels_nchw,
                                        int32_t channels, int32_t pixels_per_channel,
                                        const std::vector<float>& mean,
                                        const std::vector<float>& std) {
    const size_t channel_count = static_cast<size_t>(channels);
    const size_t plane = static_cast<size_t>(pixels_per_channel);
    if (channels <= 0 || pixels_per_channel <= 0 || mean.size() != channel_count ||
        std.size() != channel_count || pixels_nchw.empty() ||
        pixels_nchw.size() % (plane * channel_count) != 0) {
        return {};
    }
    std::vector<float> normalized(pixels_nchw.size());
    for (size_t i = 0; i < pixels_nchw.size(); ++i) {
        // NCHW 下通道下标是 (i / (H*W)) % C。**分母必须是 H*W，不是"总元素数/C"**：
        // 后者等于 B*H*W，batch=1 时恰好等价、多 batch 才错（TROUBLESHOOTING #23）。
        const size_t channel = (i / plane) % channel_count;
        normalized[i] = (pixels_nchw[i] / 255.0f - mean[channel]) / std[channel];
    }
    return normalized;
}

CVRunner::CVRunner(std::shared_ptr<Engine> engine,
                   const std::vector<float>& mean,
                   const std::vector<float>& std)
    : engine_(std::move(engine)), mean_(mean), std_(std) {
    if (engine_ == nullptr) {
        MINI_TRT_LOG_ERROR("CVRunner: engine is null");
        return;
    }
    nvinfer1::ICudaEngine* cuda = engine_->GetCudaEngine();
    if (cuda == nullptr) {
        MINI_TRT_LOG_ERROR("CVRunner: engine has no CUDA engine");
        return;
    }

    // 维度从引擎查（不写死）：输入是 [B, C, H, W]，输出是 [B, num_classes]。
    //
    // **两个 API 各管一段，别用混**：
    //   · 输入的动态轴范围只在 profile 里（`getProfileShape`），取 kOPT 读静态维；
    //   · 输出没有自己的 profile，`getProfileShape("output", ...)` 会返回 Dims{-1,{}}——
    //     输出的静态维要用 `getTensorShape`（batch 维仍是 -1，但 num_classes 是实的）。
    const nvinfer1::Dims input_dims = cuda->getProfileShape(
        "input", /*profileIndex=*/0, nvinfer1::OptProfileSelector::kOPT);
    const nvinfer1::Dims output_dims = cuda->getTensorShape("output");
    if (input_dims.nbDims != 4 || output_dims.nbDims != 2) {
        MINI_TRT_LOG_ERROR("CVRunner: unexpected I/O rank (input "
                           << input_dims.nbDims << "D, output " << output_dims.nbDims
                           << "D); expected 4D input and 2D output");
        return;
    }

    const nvinfer1::Dims min_shape = cuda->getProfileShape(
        "input", /*profileIndex=*/0, nvinfer1::OptProfileSelector::kMIN);
    const nvinfer1::Dims max_shape = cuda->getProfileShape(
        "input", /*profileIndex=*/0, nvinfer1::OptProfileSelector::kMAX);
    if (min_shape.nbDims != 4 || max_shape.nbDims != 4) {
        MINI_TRT_LOG_ERROR("CVRunner: cannot read batch range from the engine profile");
        return;
    }

    channels_ = input_dims.d[1];
    height_ = input_dims.d[2];
    width_ = input_dims.d[3];
    num_classes_ = output_dims.d[1];
    min_batch_ = min_shape.d[0];
    max_batch_ = max_shape.d[0];

    if (mean_.size() != static_cast<size_t>(channels_) ||
        std_.size() != static_cast<size_t>(channels_)) {
        MINI_TRT_LOG_ERROR("CVRunner: mean/std size (" << mean_.size() << "/" << std_.size()
                           << ") must equal the input channel count (" << channels_ << ")");
        return;
    }
    if (channels_ <= 0 || height_ <= 0 || width_ <= 0 || num_classes_ <= 0 ||
        min_batch_ <= 0 || max_batch_ < min_batch_) {
        MINI_TRT_LOG_ERROR("CVRunner: invalid engine contract (C=" << channels_
                           << ", H=" << height_ << ", W=" << width_ << ", classes="
                           << num_classes_ << ", batch=[" << min_batch_ << "," << max_batch_
                           << "])");
        return;
    }
    ok_ = true;
}

CVRunner::~CVRunner() = default;

std::vector<float> CVRunner::Infer(const std::vector<float>& image_nchw,
                                   int32_t batch_size) {
    if (!ok_) {
        MINI_TRT_LOG_ERROR("CVRunner::Infer called on an invalid runner");
        return {};
    }
    // 先验批与尺寸，再谈计算：让 TRT 用错形状跑出结果是最坏情况（E3.3 定下的纪律）。
    if (batch_size < min_batch_ || batch_size > max_batch_) {
        MINI_TRT_LOG_ERROR("CVRunner: batch " << batch_size << " outside the engine profile ["
                           << min_batch_ << ", " << max_batch_ << "]");
        return {};
    }
    const size_t expected = static_cast<size_t>(batch_size) *
                            static_cast<size_t>(channels_) * static_cast<size_t>(height_) *
                            static_cast<size_t>(width_);
    if (image_nchw.size() != expected) {
        MINI_TRT_LOG_ERROR("CVRunner: input has " << image_nchw.size() << " elements, expected "
                           << expected << " for batch " << batch_size);
        return {};
    }

    const std::vector<float> normalized =
        NormalizePixelsToNchw(image_nchw, channels_, height_ * width_, mean_, std_);
    if (normalized.empty()) {
        MINI_TRT_LOG_ERROR("CVRunner: preprocessing failed");
        return {};
    }

    const size_t input_bytes = normalized.size() * sizeof(float);
    const size_t output_count = static_cast<size_t>(batch_size) * num_classes_;
    DeviceBuffer d_input(input_bytes);
    DeviceBuffer d_output(output_count * sizeof(float));
    if (!d_input.Allocate(input_bytes) || !d_output.Allocate(output_count * sizeof(float))) {
        MINI_TRT_LOG_ERROR("CVRunner: failed to allocate device buffers");
        return {};
    }
    // 全程只有两次拷贝（H2D 一次输入、D2H 一次输出），中间不往返主机。
    CUDA_CHECK(cudaMemcpy(d_input.data(), normalized.data(), input_bytes,
                          cudaMemcpyHostToDevice));

    if (!engine_->SetOptimizationProfile(0, nullptr) ||
        !engine_->SetInputShape("input",
                                nvinfer1::Dims4{batch_size, channels_, height_, width_}) ||
        !engine_->SetTensorAddress("input", d_input.data()) ||
        !engine_->SetTensorAddress("output", d_output.data())) {
        MINI_TRT_LOG_ERROR("CVRunner: binding failed");
        return {};
    }
    if (!engine_->Enqueue(nullptr)) {
        // enqueueV3 的返回值必须查：漏绑输出等错误只在这里体现（TROUBLESHOOTING #19）。
        MINI_TRT_LOG_ERROR("CVRunner: enqueue failed");
        return {};
    }
    engine_->Synchronize(nullptr);

    std::vector<float> logits(output_count);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_output.data(), output_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return logits;
}

Engine::BenchResult CVRunner::Benchmark(int32_t batch_size, int32_t n_warmup, int32_t n_run) {
    const Engine::BenchResult zero{0.0f, 0.0f, 0.0f, 0.0f};
    if (!ok_ || batch_size < min_batch_ || batch_size > max_batch_ || n_run <= 0) {
        MINI_TRT_LOG_ERROR("CVRunner::Benchmark: invalid runner or arguments");
        return zero;
    }

    // 绑定一次、重复 enqueue——计时循环里不该再掺进分配与拷贝的形状设置。
    const size_t input_count = static_cast<size_t>(batch_size) *
                               static_cast<size_t>(channels_) *
                               static_cast<size_t>(height_) * static_cast<size_t>(width_);
    const size_t output_count = static_cast<size_t>(batch_size) * num_classes_;
    DeviceBuffer d_input(input_count * sizeof(float));
    DeviceBuffer d_output(output_count * sizeof(float));
    if (!d_input.Allocate(input_count * sizeof(float)) ||
        !d_output.Allocate(output_count * sizeof(float))) {
        MINI_TRT_LOG_ERROR("CVRunner::Benchmark: failed to allocate device buffers");
        return zero;
    }
    std::vector<float> host_input(input_count, 0.25f);  // 固定输入，与历史工程 benchmark 的做法一致
    CUDA_CHECK(cudaMemcpy(d_input.data(), host_input.data(), d_input.size(),
                          cudaMemcpyHostToDevice));
    if (!engine_->SetOptimizationProfile(0, nullptr) ||
        !engine_->SetInputShape("input",
                                nvinfer1::Dims4{batch_size, channels_, height_, width_}) ||
        !engine_->SetTensorAddress("input", d_input.data()) ||
        !engine_->SetTensorAddress("output", d_output.data())) {
        MINI_TRT_LOG_ERROR("CVRunner::Benchmark: binding failed");
        return zero;
    }
    // seq_len 记 1：Engine::Benchmark 用 batch*seq_len 算吞吐，于是这里得到 images/s。
    return engine_->Benchmark(batch_size, /*seq_len=*/1, n_warmup, n_run);
}

}  // namespace mini_trt_llm
