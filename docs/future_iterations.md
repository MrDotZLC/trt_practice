# 后续迭代计划

> 本文档记录 `mini_trt_llm` 中已明确延后、但未来需要实现的能力，防止遗忘。  
> 每一项标注预计优先级、大致阶段、关键依赖。

---

## 1. 量化与精度优化

### 1.1 ResNet18 INT8 校准

- **优先级**：P1
- **背景**：现有 `0_resnet18_onnx` 已支持 INT8，含 `calib_data/` 与 `Int8Calibrator`。`mini_trt_llm` 替换后需补齐该能力。
- **工作内容**：
  - 实现 `core/int8_calibrator.hpp/.cpp`，封装 `nvinfer1::IInt8Calibrator`。
  - 支持 `CalibrationDataReader` 读取 `calib_data/*.bin`。
  - `EngineBuilder` 增加 INT8 配置路径。
  - 在 `ModelConfig` 中增加 `quantization` 字段描述每层/全局校准策略。
- **关键依赖**：`0_resnet18_onnx` 的 `calibrator.cpp` 可直接参考。

### 1.2 LLM INT8 / INT4 量化

- **优先级**：P2
- **背景**：sm_75 有 INT8 Tensor Core，LLM INT8 可显著提升吞吐。
- **工作内容**：
  - 支持 INT8 weight-only 或 SmoothQuant。
  - 支持 GPTQ/AWQ 需评估 sm_75 兼容性（可能不支持部分 INT4 kernel）。
- **关键依赖**：PagedAttention Plugin 需支持 INT8 KV Cache。

### 1.3 BF16 原生计算支持

- **优先级**：P3
- **背景**：当前 sm_75 无 BF16 Tensor Core，BF16 权重会转换为 FP32/FP16。
- **工作内容**：若未来迁移到支持 BF16 的 GPU（Ampere+），可直接启用 BF16 engine。

---

## 2. 性能优化

### 2.1 显存池替换简单封装

- **优先级**：P1
- **背景**：Phase 0 的 `DeviceBuffer` 仅封装 `cudaMalloc/cudaFree`，频繁分配有性能开销。
- **工作内容**：
  - 实现基于 freelist 或 arena 的 `MemoryPool`。
  - 支持按大小分桶、按 stream 隔离。
  - 替换 `DeviceBuffer` 内部实现，保持接口不变。

### 2.2 FlashAttention / FlashDecoding

- **优先级**：P2
- **背景**：Turing sm_75 无官方 FlashAttention 优化，但可 hand-tune 基础 attention kernel。
- **工作内容**：
  - 在 `PagedAttentionPlugin` 中实现针对 sm_75 的分块 attention。
  - 支持 decode 阶段的 batching 优化。

### 2.3 Continuous Batching / In-Flight Batching

- **优先级**：P2
- **背景**：服务化部署时，continuous batching 可大幅提升 GPU 利用率。
- **工作内容**：
  - `LLMRunner` 支持多请求队列调度。
  - `PagedKVCache` 支持跨请求动态分配与回收。

### 2.4 CV 动态分辨率

- **优先级**：P2
- **背景**：Phase 0 CV 只支持动态 batch，后续需支持输入分辨率变化。
- **工作内容**：
  - `CVRunner` 支持 `H/W` 动态轴。
  - `EngineBuilder::Config` 增加 CV 动态分辨率 profile。
  - ResNet18 全局池化层天然支持动态分辨率，主要工作在 shape 配置。

---

## 3. 模型与架构扩展

### 3.1 Encoder-Decoder 模型

- **优先级**：P2
- **背景**：架构已预留接口（`ModelConfig::architecture = "encoder_decoder"`）。
- **工作内容**：
  - 实现 `EncoderDecoderModelBuilder`。
  - `LLMRunner` 扩展为 `Seq2SeqRunner`，支持 encoder prefill + decoder cross-attention。
  - KV Cache 需同时管理 self-attention 与 cross-attention cache。

### 3.2 Vision Transformer（ViT）

- **优先级**：P2
- **背景**：ViT 结构与 LLM 高度相似，只是输入为 image patches。
- **工作内容**：
  - 实现 `ViTModelBuilder`。
  - Patch Embedding Plugin。
  - 复用现有 Transformer Block 构建逻辑。

### 3.3 多模态模型（CLIP / LLaVA）

- **优先级**：P3
- **背景**：需要 CV encoder + text encoder/decoder 联合推理。
- **工作内容**：
  - 扩展 `BaseTokenizer` 支持多模态 prompt 模板。
  - `EngineBuilder` 支持多输入网络（image + text）。
  - 跨模态投影层 Plugin。

### 3.4 Audio 模型（Whisper）

- **优先级**：P3
- **背景**：Encoder-Decoder 的特例。
- **工作内容**：
  - 音频特征提取（log-mel spectrogram）预处理。
  - Whisper encoder 与 decoder 构建。

---

## 4. Plugin 与 Kernel 扩展

### 4.1 更多归一化 Plugin

- **优先级**：P1
- **内容**：
  - `LayerNormPlugin`（当前已有 RMSNorm，部分模型用 LayerNorm）。
  - `GroupNormPlugin`、`InstanceNormPlugin`（CV 模型需要）。

### 4.2 更多激活函数 Plugin

- **优先级**：P1
- **内容**：
  - `SiLUPlugin`、`SwiGLUPlugin`（LLaMA/GLM 系列 FFN 需要）。
  - `GELUPlugin` 若 TRT 原生实现性能不足时自定义。

### 4.3 MoE（Mixture of Experts）支持

- **优先级**：P3
- **内容**：
  - Expert 路由 Plugin。
  - 稀疏门控与专家并行。

---

## 5. Tokenizer 扩展

### 5.1 BPE Tokenizer（GPT-2 原生）

- **优先级**：P1
- **背景**：GPT-2 使用 BPE，SentencePiece 行为可能与 Python `tiktoken`/`transformers` 不完全一致。
- **工作内容**：
  - 实现 `BpeTokenizer : public BaseTokenizer`。
  - 从 `vocab.json` + `merges.txt` 加载。
  - 与 Python GPT-2 tokenizer 逐 case 对比。

### 5.2 Tiktoken Tokenizer

- **优先级**：P2
- **背景**：GPT-4 / ChatGLM 等模型使用 tiktoken。
- **工作内容**：
  - 接入 `tiktoken` C++ 实现或自研。

### 5.3 多模态 Prompt Template

- **优先级**：P3
- **背景**：LLaVA 等模型需要 image token 占位与特殊模板。
- **工作内容**：
  - `BaseTokenizer` 增加 `ApplyChatTemplate` 接口。

---

## 6. 工具链与工程效率

### 6.1 Python 转换工具增强

- **优先级**：P1
- **内容**：
  - 支持更多来源：PyTorch `.pt`、HuggingFace、Meta 原始 checkpoint。
  - 支持 INT8 校准数据自动生成。
  - 支持 ONNX 导出（方案 B 需要）。

### 6.2 ONNX 导出与 Plugin Custom Op

- **优先级**：P2
- **背景**：方案 B 需要 ONNX 中带有 `mini_trt_llm` domain 的 custom op。
- **工作内容**：
  - 提供 PyTorch `torch.onnx.register_custom_op_symbolic` 示例。
  - 或提供 `torch.export` + custom decomposition 脚本。

### 6.3 Nsight 一键 Profile Target

- **优先级**：P2
- **内容**：
  - CMake 增加 `profile_gpt2`、`profile_resnet18` 自定义 target。
  - 支持 `nsys profile` 与 `ncu` 导出 `.ncu-rep`。

### 6.4 CI / 自动化测试

- **优先级**：P2
- **内容**：
  - GitHub Actions 或本地脚本：
    - 代码格式检查（`clang-format`）。
    - `cmake -DBUILD_TESTS=ON && ctest`。
    - 下载 dummy model 并跑端到端测试。

---

## 7. 服务化与部署

### 7.1 HTTP / gRPC 推理服务

- **优先级**：P3
- **内容**：
  - 基于 `LLMRunner` 提供 OpenAI-compatible API。
  - 支持流式输出。

### 7.2 多 GPU / Tensor Parallel

- **优先级**：P3
- **内容**：
  - 评估 sm_75 单机多卡价值（GTX 1660 Ti 通常为单卡）。
  - 如未来迁移到多卡环境，再实现 TP/PP。

---

## 8. 已知问题记录

| 问题 | 影响 | 状态 | 计划解决阶段 |
|---|---|---|---|
| BF16 在 sm_75 下需转换 | 无功能影响，有轻微构建时开销 | 已知 | Phase 0 已处理 |
| SentencePiece 与 GPT-2 BPE 不完全一致 | 可能导致 tokenizer 结果偏差 | 已知 | 后续实现 BpeTokenizer |
| PagedAttention 无 FlashAttention 优化 | Decode 延迟较高 | 已知 | 后续 hand-tune |
| INT8 校准未实现 | ResNet18 INT8 能力缺失 | 已知 | 后续迭代 |
| CV 动态分辨率未实现 | 输入尺寸固定 | 已知 | 后续迭代 |

---

*文档版本：v1.0*  
*关联文档：`docs/mini_trt_llm_design.md`、`docs/phase0_development_plan.md`*
