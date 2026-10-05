# S2 自定义算子进图并被执行 —— 接口细化（可评审后再落代码）

<!--
本文件只写"接口形态与改动点"，不含产品代码。它是 `design.md` 的 Milestones 表里
S2 那一行的展开，落点为 design.md 的 Interfaces（构建入口 / 生效判据）。
落代码时本文件不删除；S2 完成后由 `summary.md` 收口。
-->

## 0. 已核实的既有能力（决定 S2 的改动面）

| 事实 | 证据 | 对 S2 的意义 |
|---|---|---|
| 项目有插件注册表，但它是**项目自己的表**（`name → nvinfer1::IPluginCreatorV3One*`），**不是** TRT 的 `IPluginRegistry` | `src/plugins/plugin_registry.cpp` 的 `creators_` / `GetCreator(name)` | S2 需要一个**适配器**把这张表接到 TRT 的 `IPluginRegistry` 上；两者接口不同名同义 |
| 已注册四个 creator | `PluginRegistry::RegisterAllPlugins()`：`GetRmsNormPluginCreator` / `GetRoPEPluginCreator` / `GetPagedAttentionPluginCreator` / `GetPackedAttentionPluginCreator` | 查表有两级键之外的信息可用（名字 + 版本） |
| 算子的名字与版本是编译期常量 | `kRmsNormPluginName = "MiniTrtLlmRmsNorm"`（v `"1"`）、`kRoPEPluginName = "MiniTrtLlmRoPE"`（v `"1"`）、`kPagedAttentionPluginName = "MiniTrtLlmPagedAttention"`（v `"2"`）、`kPackedAttentionPluginName = "MiniTrtLlmPackedAttention"` | ONNX 节点的 `op_type` / 版本必须用这些字符串 |
| **creator 的 namespace 默认是空串** | `rmsnorm_plugin.cu` 的 `RmsNormPluginCreator()` 只填 `fields_`，`namespace_` 从未赋值；`getPluginNamespace()` 返回 `namespace_.c_str()` | ONNX 的自定义域**必须非空**（空域等于 `ai.onnx`）→ 当前命名空间与自定义域对不上（见 §3 硬核实项 2） |
| 解析器构造**没挂注册表** | `src/core/builder.cpp` 的 `BuildFromOnnx`：`createParser(*network, logger_)` | S2 的 C++ 主改动点 |
| 解析失败会被逐条打印 | 同函数的 `parseFromFile` 失败分支：`parser->getError(i)->desc()` | "找不到创建器"会以 parse error 形式出现——**不要**把它误判成"插件没注册" |
| RmsNorm 的 I/O 与字段 | `getNbOutputs() == 1`；`enqueue` 用 `inputs[0]`（x）+ `inputs[1]`（weight）；creator 声明字段 `eps`(kFLOAT32) / `hidden_size`(kINT32) | 夹具图的算子形态按这三条拼（其余插件同理，按各自 creator 的 `getFieldNames()` 读） |
| 引擎逐层信息的读取器与开关都已存在 | `tests/engine_layer_info_support.hpp` 的 `InspectLayerInfo`；`BuildOptions::detailed_profiling` | 生效判据不需要新造工具 |

## 1. 接口形态

### 域名常量（插件公共头）

```cpp
inline constexpr char kMiniTrtLlmPluginNamespace[] = "mini_trt_llm";
```

**唯一来源**：ONNX 节点的 `domain`、适配器的匹配键、文档里的举例都用这一个常量，禁止各处字面量。

### 适配器（把项目注册表接到 TRT）

```cpp
// 只做"按 (op_type, version, namespace) 查表"，不持有 creator 的生命周期
// （生命周期仍由各插件自己的静态实例持有，与 PluginRegistry 的既有约定一致）。
class MiniTrtLlmPluginRegistry final : public nvinfer1::IPluginRegistry {
 public:
    nvinfer1::IPluginCreatorInterface* getPluginCreator(
        const char* plugin_type, const char* plugin_version,
        const char* plugin_namespace) noexcept override;
};
```

匹配规则（必须写死，避免"名字对上了但版本没对"这类静默错）：

| 传入 | 行为 |
|---|---|
| `namespace` 不等于 `kMiniTrtLlmPluginNamespace` | 返回 `nullptr`（**不**回落默认域） |
| `plugin_version` 为空 | 返回该类型的创建器（由调用方负责版本语义） |
| 类型/版本查不到 | 返回 `nullptr` 并打日志（parse 阶段会转成 parse error） |

### 解析（C++ 侧）

```cpp
nvonnxparser::createParser(*network, *logger_, plugin_registry);   // 签名待核实（§3 硬核实项 1）
```

### 夹具图（Python 侧）

```text
python mini_trt_llm/tools/make_custom_op_onnx.py --output /tmp/mini_trt_llm_custom.onnx \
    --op MiniTrtLlmRmsNorm --epsilon 1e-5 --hidden-size 64
```

产出：一张含 `domain="mini_trt_llm"`、`op_type="MiniTrtLlmRmsNorm"` 节点的最小图，
输入 `[batch, seq, hidden_size]`（权重作为 `initializer`），输出与算子声明一致。
**为什么用 RmsNorm 做首个夹具**：它的字段最少（两个）、不涉及 K/V 与分页，
最适合先证明"自定义算子能在外部图路径上被解析并执行"。

## 2. 改动点

| 文件 | 现状 | S2 改动 |
|---|---|---|
| `include/mini_trt_llm/plugins/plugin_registry.hpp` | 项目自有表 | 新增域名常量 + 适配器声明 |
| `src/plugins/plugin_registry.cpp` | 只做名字查表 | 实现适配器（按类型/版本/命名空间查） |
| `src/core/builder.cpp` 的 `BuildFromOnnx` | `createParser(network, logger)` | 传入适配器；失败日志区分"缺创建器"与"其它 parse error" |
| 各 creator（`getPluginNamespace()`） | 默认空串 | **设为 `kMiniTrtLlmPluginNamespace`**（作者 2026-10-06 已裁决接受；连带影响见 §3.1） |
| `tools/make_custom_op_onnx.py` | 不存在 | 新增（§1） |
| `tests/test_onnx_custom_op.cpp` | 不存在 | 新增（§6） |

## 3. 开工前必须核实的四件事（本机无 TRT 头 / 无真机，**本次全部未核实**）

| # | 核什么 | 怎么核 | 若不成立的后果 |
|---|---|---|---|
| 1 | `nvonnxparser::createParser` 是否有带 `IPluginRegistry&` 的重载（TRT 10.15.1） | 读 `NvOnnxParser.h` 的函数声明 | 没有 → 改走自定义 `IPluginFactory` 路线，**本文件与 `design.md` 一起回 P2** |
| 2 | 解析器匹配自定义域的规则：是否要求 `creator->getPluginNamespace() == node.domain` | 读 `NvOnnxParser.h` / `NvInferRuntime.h` 的相关注释与 `IPluginRegistry` 定义；真机上一次最小复现 | 若要求匹配 → **必须**给 creator 设非空 namespace（当前是空串），这会改动插件公共行为（见第 3 条） |
| 3 | 改 creator 的 namespace 对**既有已序列化引擎**的影响 | 真机：拿一个既有 `.engine` 做反序列化（`tools/inspect_engine.cpp` 即最小探针） | 若既有引擎里记了 namespace → 那些引擎反序列化会失败，**必须连带重建**，并在 `PROGRESS` / STATE 里写明"这次改动会让既有引擎失效" |
| 4 | ONNX 属性到 `PluginField` 的映射：`eps`(float) / `hidden_size`(int32) 在节点属性里的表达与类型 | 读 parser 文档/头文件 + 真机最小复现 | 类型不匹配 → 创建器拿到空 data，表现为静默错误值（这类必须**响亮失败**） |

**这四条属于"兑现 S2 需要的输入/接口是否存在"的核对**（`AGENTS.md` §5 第 5 条）。
在它们有结论之前，本文件里 §1 的两处签名与 §2 的 namespace 改动**只是待验证设计**，
不得直接落码。

### 3.1 作者裁决（2026-10-06）

| 事项 | 裁决 | 对上面四条的影响 |
|---|---|---|
| creator 的 `getPluginNamespace()` | **设为 `kMiniTrtLlmPluginNamespace`（"接受"）** | 第 2 条的"若要求匹配"分支**已成既定方案**，不再是待选分支 |
| 改 namespace 可能让既有引擎反序列化失败 | **接受连带重建** | 第 3 条的后果被提前接受；**重建范围**仍待第 3 条核实后才能写死（不知道影响面就不能先声明白名单） |
| 本阶段的真机可用性 | **暂时不真机** | 第 1、4 条核实**推迟**；S2 **不得落码**（规格里两处签名仍属待验证设计） |

## 4. 失败语义

| 情形 | 行为 |
|---|---|
| 图上出现自定义域算子但注册表里没有对应创建器 | parse error 且日志点名 `domain/op/version`（**不得**回落成"忽略该节点"） |
| 属性类型与 `getFieldNames()` 不符 | 构建期失败（创建器必须显式拒绝，不允许用默认值兜底） |
| 引擎构建成功但层信息里**没有**插件层 | 视为 S2 失败（见 §5 的生效判据） |

## 5. 判据与来源

| 判据 | 来源 |
|---|---|
| 图能被解析（自定义算子被识别为插件） | `parseFromFile` 返回值 + 无 parse error（既有代码路径） |
| **替换生效**：引擎里确实执行自研算子 | `InspectLayerInfo` 读出的 ONELINE 原文（工具已存在）；断言以**首次实测的原文**为准，先落盘再定判据（`engine_layer_info_support.hpp` 记的教训：不能凭直觉猜标签） |
| 数值一致 | 夹具图上的插件输出 vs 该插件既有单测里的参考实现；阈值按**同精度**复用既有出处（`AGENTS.md` §7） |
| 不回归 | 既有引擎反序列化与既有用例不新增红；若 §3 第 3 条成立，重建范围必须写进 STATE |

## 6. 测试方式

| 用例 | 判据 | 环境 |
|---|---|---|
| `OnnxCustomOp.RmsNormExecutes` | 夹具图构建成功 + 数值与参考一致 + ONELINE 出现插件层 | 真机 |
| `OnnxCustomOp.RejectsUnknownDomain` | 造一个域对不上的图 → 失败且日志点名 domain/op/version | 真机 |
| `OnnxCustomOp.RejectsBadAttributeType` | 属性类型不符 → 构建期响亮失败（不静默取默认值） | 真机 |
| 既有引擎反序列化（§3 第 3 条） | 探针能反序列化既有 `.engine`（若不能，按结论登记重建） | 真机 |
| 全量回归 | 既有用例不新增红 | 沙箱 + 真机 |

## 7. 待作者确认与待核实

| # | 事项 | 现状 |
|---|---|---|
| 1 | 首个夹具算子选 `MiniTrtLlmRmsNorm` | **未裁决**（不阻塞设计：理由是字段最少、不涉及 K/V；若你要先验注意力，请点名，我改 §1） |
| 2 | §3 的四条核实安排在哪个窗口 | **已定（2026-10-06）：本阶段不真机** → 核实推迟，S2 不落码 |
| 3 | 若第 3 条成立（既有引擎失效） | **已裁决（2026-10-06）：接受连带重建**；重建范围待核实后写死 |

**来源核对结论**：S2 的**判据来源**（层信息工具、插件字段声明、参考实现）都存在；
缺的是**接口是否存在**的四条事实，已按 §5 第 5 条标为待核实并停在这里，未自行选定替代路线。
**裁决后的状态（2026-10-06）**：第 2、3 条已定（见 §3.1），第 1、4 条因"暂时不真机"继续挂起——
即 **P0-1 仍未闭环**，S2 保持在"未满足（阻塞）"。
