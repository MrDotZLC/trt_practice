本项目是基于 CMake 的 C++ 代码库，用来记录 TensorRT 的学习代码。

# 一、章节介绍

0. 用 ResNet18 ONNX 模型构建一个 TensorRT C++ Engine。
1. 用 GPT-2 ONNX 模型构建一个 TensorRT C++ Engine。

# 二、当前现状

`0_resnet18_onnx` 与 `1_gpt2_onnx` 两个示例模块仍保留可用，但项目重心已转向 `mini_trt_llm`——一个面向 NVIDIA Turing / `sm_75` 的极简多模态 TensorRT 推理框架。它的目标是把上述两个独立示例模块的能力合并进来，最终等价替换并在根 `CMakeLists.txt` 中下线旧模块。

进度概览：

- **Phase 0（基础设施与通用化骨架）**：已完成。
- **Phase 1（自定义 Plugin 基础）**：已完成（RMSNorm / RoPE / PagedAttention 插件 + 采样器，真机验证通过）。
- **Phase 1.5（全流程测试基建与收尾）**：已完成（端到端骨架、dynamic shape / optimization profile）。
- **Phase 2（GPT-2 原生构建）**：已完成——真实 GPT-2 贪心生成 8 token 与 HuggingFace 基线逐 token 一致。
- **Phase 3（GPT-2 ONNX 路径）**：已完成——与原生构建的 logits 相对偏差 `5.66e-07`（阈值 `1e-5`）。
- **Phase 4 / 5**（ResNet18 替换 / 清理旧模块）：未开始。

> 详细状态与决策见 `docs/PROGRESS.md`；各阶段的开发/测试计划见 `docs/phase*_*.md`。

主要入口：

- 核心代码：`mini_trt_llm/`
- 进度交接文档：`docs/PROGRESS.md`
- 设计文档：`docs/mini_trt_llm_design.md`
- Phase 2 开发计划：`docs/phase2_development_plan.md`
- Phase 3 开发计划与测试计划：`docs/phase3_development_plan.md` / `docs/phase3_test_plan.md`

## 构建与测试

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

开发环境：Ubuntu on WSL2、TensorRT 10.15.1、CUDA Toolkit 12.6、GPU 为 GTX 1660 Ti Mobile（`sm_75`，无 Tensor Core，不支持 FP8）。
