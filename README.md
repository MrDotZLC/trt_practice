# mini_trt_llm

面向 NVIDIA Turing / `sm_75` 的**极简多模态 TensorRT 推理框架**：LLM（GPT-2）与 CV（ResNet18）
共用同一套构建 / 引擎 / 运行时分层的 C++17 推理库。

它把 TensorRT-LLM 的核心机制从零实现了一遍——配置驱动建图、Paged KV Cache、自研 PagedAttention、
设备侧采样、prefill / decode 双引擎——并对每条结论保留实测出处。

> **边界要说清**：这是**机制层面的极小复刻**，不是 TensorRT-LLM 的生产替代品。
> 没有 in-flight batching / 调度器、没有 LLM 量化、没有服务化层、没有多卡；`LLMRunner` 有意限定
> `batch = 1`。能力对照与新架构的接入路径见 `docs/interview_summary.md` §2.12 与
> `docs/future_iterations.md`。

## 支持的模型与精度

| 模型 | 路径 | 精度 | 说明 |
|---|---|---|---|
| **GPT-2** | 原生（`config.json` + safetensors） | **FP32**（推荐） | `LLMRunner` 全链路：Prefill → Decode → 采样；BPE tokenizer，可文本进文本出 |
| GPT-2 | ONNX | FP32 / FP16 | 只做 prefill 推理与对拍，**不能**接 `LLMRunner` 做自回归生成 |
| **ResNet18** | ONNX / 原生 / `CVRunner` | **FP32 / FP16 / INT8（Q/DQ）** | 输入契约 NCHW float `[0,255]`，归一化在 Runner 内 |

**已知限制（按设计保持红色）**：GPT-2 的 FP16 端到端产生 NaN，推荐用 FP32；真机全量里那一条红
就是它的复现器（见 `docs/PROGRESS.md` §5.11）。原因与三条后续路线见 `docs/future_iterations.md` §1.4。

## 目录结构

```text
mini_trt_llm/            核心库：include/ src/ tests/ tools/ third_party/
common/                  TensorRT 的 ILogger 实现（框架与测试都靠它；由根 CMakeLists 加入 include 路径）
assets/legacy/           历史示例工程迁来的本地资产（ONNX 图 / INT8 验收集 / 精度基线），不入库
models/                  转换产物（config.json 入库；*.safetensors / *.onnx 不入库）
scripts/                 参考实现（torchvision 基线、RoPE / 采样器语义自检）
docs/                    文档（入口见 docs/README.md）
build/                   构建目录（CMake）
```

> **沿革**：本仓库原有两个独立的 ONNX 示例工程 `0_resnet18_onnx/` 与 `1_gpt2_onnx/`，
> 它们的能力已并入本框架；这两个目录随 Phase 5 下线（数据迁到 `assets/legacy/`，
> 重建脚本在 `assets/legacy/scripts/`）。要复核被删除的实现，用删除前的提交：
> `git show ba3ea7a:1_gpt2_onnx/src/builder.cpp`。

## 构建

```bash
# 日常构建（默认不建测试）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j$(nproc)

# 要跑测试就显式打开
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

## 跑测试

```bash
# 沙箱 / CI：host 用例真跑，GPU 用例显式跳过（跳过信息里带 CUDA 探测结果）
ctest --test-dir build --output-on-failure

# 真机（WSL2）：跳过即失败、缺资产即失败——不设这两个变量等于白跑
MINI_TRT_REQUIRE_GPU=1 MINI_TRT_REQUIRE_ASSETS=1 ctest --test-dir build --output-on-failure
```

两条注意：

1. **必须带 `--test-dir build`**。在仓库根直接跑 `ctest` 会打印 `No tests were found!!!`，
   而**退出码仍是 0**——判据是输出里的 `Test #N` 行数，不是退出码（见 `TS-043`）。
2. 真机唯一允许的红是 GPT-2 的 FP16 复现器；出现别的红、或出现**未登记的跳过**，都按回归处理。
   跳过集合由 `mini_trt_llm/tools/check_skips.py` 比对，基线是
   `mini_trt_llm/tests/data/expected_skips.txt`（用法见 `docs/phase5_development_plan.md` §4）。

**测试基线**（沙箱 / 真机的条数、红、跳过）只在一处维护：`docs/PROGRESS.md` 的「当前基线」。

## 跑之前要准备的本地产物

`models/gpt2/`、`models/resnet18/`、`assets/legacy/` 里的二进制**都不入库**，换机器要重建；
每类产物的来源、重建命令与身份（SHA256）见 `docs/PROGRESS.md` §7 与 `assets/legacy/README.md`。
缺它们时用例会**失败**（设了 `MINI_TRT_REQUIRE_ASSETS=1`）或**显式跳过**（未设），不会假装通过。

## 文档入口

- 组织规约与阅读顺序：`docs/README.md`
- 现状 / 架构决策 / 已知坑 / 开放项索引：`docs/PROGRESS.md`
- 排查记录（现象 / 命令 / 证据 / 根因，只增不改）：`docs/TROUBLESHOOTING.md`
- 后续迭代（**触发驱动**，不是待办队列）：`docs/future_iterations.md`
- 面向面试的项目总结与问答：`docs/interview_summary.md`
- 历史阶段的设计与判据出处：`docs/phaseN_*.md`（已冻结，不代表现状）

## 开发环境

Ubuntu on WSL2 · TensorRT **10.15.1**（`IPluginV3`）· CUDA Toolkit **12.6** · GCC 13.3 · C++17 ·
CMake ≥ 3.18 · GPU 为 GTX 1660 Ti Mobile（`sm_75`，**无 Tensor Core**，不支持 FP8 / FP4）。
版本与依赖的完整清单见 `docs/PROGRESS.md` §7。
