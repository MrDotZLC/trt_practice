#include "mini_trt_llm/core/cv_runner.hpp"
#include <stdexcept>

namespace mini_trt_llm {

CVRunner::CVRunner(std::shared_ptr<Engine> engine,
                   const std::vector<float>& mean,
                   const std::vector<float>& std)
    : engine_(engine), mean_(mean), std_(std) {}

CVRunner::~CVRunner() = default;

std::vector<float> CVRunner::Infer(const std::vector<float>& image_nchw,
                                   int batch_size) {
    // Phase 4 实现
    (void)image_nchw;
    (void)batch_size;
    throw std::runtime_error("CVRunner::Infer not implemented in Phase 0");
}

Engine::BenchResult CVRunner::Benchmark(int batch_size, int n_warmup, int n_run) {
    // Phase 4 实现
    (void)batch_size;
    (void)n_warmup;
    (void)n_run;
    throw std::runtime_error("CVRunner::Benchmark not implemented in Phase 0");
}

}  // namespace mini_trt_llm
