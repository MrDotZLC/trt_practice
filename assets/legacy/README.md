# assets/legacy —— 历史示例工程用到的本地资产

> **为什么在这里**：Phase 5 要把 `0_resnet18_onnx/` 与 `1_gpt2_onnx/` 从仓库路径里下线，
> 但这两个目录里的**数据资产**是现行用例的判据来源，不能跟着目录一起消失。所以数据搬到这里、
> 迁移期曾用**相对软链接**指回原路径（让同批的 `mini_trt_llm/` 路径与 provenance 字符串**一行都不改**）。
> **那 4 条软链接连同两个目录已于 2026-09-28 的阶段 3 删除**——现在代码 / 测试 / 工具 / provenance
> 都直接指向本目录。方案见 `docs/dev/REQ-009-retire-legacy/phase5_development_plan.md`（阶段 1 / 阶段 3）。
>
> **本目录的内容全部不入库**（`.gitignore` 命中 `*.onnx` / `*.bin` / `**/calib_data/`），
> 换机器要按下面的命令重建。

## 布局与来源

| 路径 | 原位置 | 用途 | 重建 |
|---|---|---|---|
| `resnet18_onnx/resnet18.onnx` | 原 `0_resnet18_onnx/resnet18.onnx` | CV 用例的 ONNX 输入；`onnx_to_mini_trt_llm.py` / `quantize_resnet18.py` 的输入 | `python3 assets/legacy/scripts/resnet18_load_model.py`（torchvision 权重需已缓存） |
| `resnet18_onnx/calib_data/` | 原 `0_resnet18_onnx/calib_data/` | INT8 判据的验收集（500 张）；`qdq_reference.py` / `int8_eval.py` 的输入 | `python3 assets/legacy/scripts/resnet18_prepare_calib_data.py`（**需联网**下载 tiny-imagenet，按 `AGENTS.md` §0.2 须先获批） |
| `gpt2_onnx/gpt2.onnx` | 原 `1_gpt2_onnx/gpt2.onnx` | `Gpt2OnnxTest` 的输入；`inspect_onnx.py` 的结构基线 | `python3 assets/legacy/scripts/gpt2_load_model.py` |
| `gpt2_onnx/ref_output.bin` | 原 `1_gpt2_onnx/ref_output.bin` | GPT-2 prefill logits 的精度基线（HF FP32） | 同上（生成时一并落盘） |
| `scripts/*.py` | 原 `{0_resnet18_onnx,1_gpt2_onnx}/load_model.py` 与 `0_resnet18_onnx/prepare_calib_data.py` | 上表三个资产的重建入口 | ——（脚本自己：输出目录已按 `__file__` 定位到本目录下） |

## 怎么发现链接断了

设 `MINI_TRT_REQUIRE_ASSETS=1` 跑全量：缺资产会**判失败**而不是静默跳过——这道闸门就是
Phase 5 阶段 0 加的（见 `docs/dev/REQ-009-retire-legacy/phase5_development_plan.md` 阶段 0；沙箱基线 268 条 / 0 失败）。
全仓共 **62 处**资产跳过点走这道闸门（15 个测试文件）。

## 现在还需不需要 `0_resnet18_onnx` / `1_gpt2_onnx`

**不需要了**：两个目录连同那 4 条软链接已于 2026-09-28 删除（Phase 5 阶段 3），代码 / 测试 /
工具 / provenance 全部改指本目录。要复核被删的实现，用删除前的提交：
`git show ba3ea7a:1_gpt2_onnx/src/builder.cpp`（同理 `0_resnet18_onnx/...`）。

**本目录本身不要删**：它是这些判据（GPT-2 logits 基线、ONNX 图、INT8 验收集）的唯一来源，
而 `calib_data` 还要联网才能重建。
