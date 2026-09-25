#pragma once

#include "mini_trt_llm/core/engine.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace mini_trt_llm {

// CV 前处理：`[0,255]` 的 NCHW 像素质 → ImageNet 归一化（`(x/255 - mean[c]) / std[c]`）。
//
// 为什么做成自由函数而不是 CVRunner 的私有成员：它是**纯 host 逻辑**，做成公开接口就能在
// 沙箱里直接与 P4-1 基线脚本的 Python 版公式对拍，不必为了测一个除法去建 GPU 引擎。
//
// `channels` 必须与 `mean` / `std` 的长度相同；`pixels_per_channel` 是**单张图**每个通道的
// 像素数（H*W）。
//
// **为什么必须显式传 `pixels_per_channel`**：NCHW 下通道下标是 `(i / (H*W)) % C`，
// 分母必须是 H*W 而不是 "总元素数/C"——后者等于 B*H*W，**在 batch=1 时恰好等价**，
// 于是写法错了也测不出来，直到多 batch 才暴露（本项目真栽过一次，见 TROUBLESHOOTING #23）。
//
// 参数不合法时返回空 vector（调用方 `CVRunner::Infer` 会因此失败并打日志）。
std::vector<float> NormalizePixelsToNchw(const std::vector<float>& pixels_nchw,
                                        int32_t channels, int32_t pixels_per_channel,
                                        const std::vector<float>& mean,
                                        const std::vector<float>& std);

// CV 推理 Runner（ResNet18 这类单输入单输出的分类网）。
//
// **输入契约（D4 决策）**：`image_nchw` 是 float32、NCHW 排布、取值 `[0,255]` 的原始像素质，
// 归一化由本类完成。HWC→CHW 由调用方负责——形参名与 P4-1 的基线产物
// （`models/resnet18/inputs/*.contract_input.f32.bin`）都是 NCHW，保持同一形态才能直接对拍。
//
// **失败约定**：`Infer` 失败返回**空 vector**、`Benchmark` 失败返回零值统计，**都不抛异常**
// （与 `LLMRunner::Generate` 一致；异常留给构造期与底层库错误）。构造期若失败，用 `ok()` 查——
// 不要指望"构造成功就一定能跑"。
//
// **维度与 batch 范围一律向引擎查询**，不写死 224 / 1000 / 16：引擎自己知道 I/O 形状与
// profile 范围（`getProfileShape`），问它即可。弱类型网络下连 dtype 都要问（见 TROUBLESHOOTING
// #18 / #21）——这正是把"模型是什么"焊进 Runner 会踩的坑。
class CVRunner {
 public:
    CVRunner(std::shared_ptr<Engine> engine,
             const std::vector<float>& mean,
             const std::vector<float>& std);
    ~CVRunner();

    // 构造是否可用：引擎非空且可查询 I/O 契约、mean/std 长度与通道数一致、profile 可用。
    bool ok() const { return ok_; }

    // 推理一批图像。`image_nchw` 须有 `batch_size * channels * height * width` 个元素。
    // 成功返回 `[batch_size, num_classes]` 的 logits（行主序）；失败返回空 vector。
    std::vector<float> Infer(const std::vector<float>& image_nchw,
                             int32_t batch_size);

    // 端到端计时：**含前处理与两次拷贝**，即"用户看到的延迟"。
    // `throughput` 是 **images/s**（内部委托 Engine::Benchmark 并把 seq_len 记作 1，
    // 于是 batch*seq_len == batch == 图像数）。
    Engine::BenchResult Benchmark(int32_t batch_size, int32_t n_warmup, int32_t n_run);

 private:
    std::shared_ptr<Engine> engine_;
    std::vector<float> mean_;
    std::vector<float> std_;
    bool ok_ = false;
    // 以下均由引擎查询得到，不硬编码——换模型（别的输入尺寸/类别数）时不需要改本类。
    int32_t channels_ = 0;
    int32_t height_ = 0;
    int32_t width_ = 0;
    int32_t num_classes_ = 0;
    int32_t min_batch_ = 0;
    int32_t max_batch_ = 0;
};

}  // namespace mini_trt_llm
