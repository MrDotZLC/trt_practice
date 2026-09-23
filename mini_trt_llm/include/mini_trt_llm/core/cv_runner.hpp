#pragma once

#include "mini_trt_llm/core/engine.hpp"
#include <memory>
#include <vector>

namespace mini_trt_llm {

// CV 推理 Runner。
// Phase 0 仅声明，Phase 4 实现。
class CVRunner {
 public:
    CVRunner(std::shared_ptr<Engine> engine,
             const std::vector<float>& mean,
             const std::vector<float>& std);
    ~CVRunner();

    std::vector<float> Infer(const std::vector<float>& image_nchw,
                             int batch_size);

    Engine::BenchResult Benchmark(int batch_size, int n_warmup, int n_run);

 private:
    std::shared_ptr<Engine> engine_;
    std::vector<float> mean_;
    std::vector<float> std_;
};

}  // namespace mini_trt_llm
