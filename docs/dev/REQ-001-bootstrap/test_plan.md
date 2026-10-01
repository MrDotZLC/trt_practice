# Test Plan

> **性质**：历史用例回填（2026-10-01）。来源：`docs/dev/REQ-001-bootstrap/phase0_model_loading_test_plan.md` §2/§3
> 与 `docs/dev/REQ-001-bootstrap/phase0_development_plan.md` §4 验收标准。**判据以 `requirement.md` 为准**，本文只列用例。

## Unit Test

T1 存取库真实文件加载（`tests/test_safetensors_loader.cpp`），测试数据为仓库内置的最小张量文件
（两个张量，均为 64 位浮点）：

| 用例 | 输入 | 期望 |
|---|---|---|
| `LoadExistingFile` | 内置文件路径 | 加载返回 true |
| `HasTensorExisting` | 查询两个已存在的张量 | 返回 true |
| `HasTensorMissing` | 查询不存在的名字 | 返回 false |
| `GetTensorInfoWeight1` | 读第一个张量信息 | dtype 与 shape 正确 |
| `GetRawDataWeight1` | 读第一个张量原始数据 | 指针非空、字节数正确 |
| `GetTensorInfoWeight2` | 读第二个张量信息 | dtype 与 shape 正确 |

**明确的非目标**：不验证"64 位浮点 → 32 位"的转换语义，只验证文件解析与元数据读取。

## Integration Test

- **T2**：解析图文件并生成引擎（当时需要 CUDA，属真机用例）。
- **T3**：配置驱动 + 测试用构建器的端到端链路——用可序列化的最小网络模拟真实构建器，
  覆盖 `config.json → 配置解析 → 注册表分发 → 构建 → 落盘`。

## Regression Test

`docs/dev/REQ-001-bootstrap/phase0_development_plan.md` §4 的 8 条验收判据（逐条见 `requirement.md`），
其中第 3 / 5 / 6 / 7 条后来由具体用例承担（测试目标全绿、存取库用例、端到端单算子用例、动态 shape 用例）。

## Failure Test

- 查询不存在的张量名必须返回 false，而不是崩溃或静默返回空数据。
- 第三阶段的错误路径用例（配置缺失 / 未注册模型 / 权重缺失）在当时尚未成体系，由后续阶段补齐。

## Expected Result

构建成功、测试目标全绿、依赖形态符合约束（分词库只有静态产物、非必要功能全关）。

## Actual Result

除第 8 条（"旧模块仍可构建"）**已随旧模块下线而失效**外，其余判据均达成。

## Status

已交付（历史条目）。
