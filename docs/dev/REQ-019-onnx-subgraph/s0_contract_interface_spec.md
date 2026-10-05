# S0 契约统一 —— 接口细化（可评审后再落代码）

<!--
本文件只写"接口形态与改动点"，不含产品代码。它是 `design.md` 的 Milestones 表里
S0 那一行的展开，落点为 design.md 的 Interfaces（图重写）与 Trade-off D5。
落代码时本文件不删除；S0 完成后由 `summary.md` 收口。
-->

## 0. 已核实的既有能力（决定 S0 的改动面）

| 事实 | 证据 | 对 S0 的意义 |
|---|---|---|
| ONNX 路径只校验**输入 0 的名字**与"输出名存在" | `src/core/builder.cpp` 的 `BuildFromOnnx`：`getNbInputs() < 1 \|\| getInput(0)->getName() != io.input_name` | 加第二个输入后**不会自动被校验**，必须扩契约结构 |
| 契约结构只有一个输入名字段 | `include/mini_trt_llm/core/builder.hpp` 的 `struct OnnxIoContract { const char* input_name; const char* output_name; }` | 扩字段 = 改公开头 → 调用方同轮改 |
| 非 packed 的 profile **按 dim 下标**对所有动态输入套同一组范围（dim0 = batch，其余 = seq） | 同文件 `AddLlmOptimizationProfiles` 的布局约定注释与 `range_for_dim` | 新增 `position_ids[B,S]` **不需要改 profile 代码**（自动拿到 batch/seq 两组范围）——这是 S0 成本低的关键 |
| ONNX 路径的引擎指纹**含 ONNX 文件身份** | `BuildFromOnnx` 开头的 `MakeFingerprintInputs(model_dir, onnx_path, ...)` | 重写出新图 → 指纹变 → 自动重建，不需要手工 bump 图版本 |
| 识别基线写死了输入名清单 | `tools/inspect_onnx.py` 的 `BASELINE["inputs"] = ["input_ids"]` | **S0 之后 `--check` 必然报红**：要么基线按"图角色"分开，要么把新图排除在旧基线之外（见 §7 待确认第 1 条） |
| 对拍用例按**各引擎声明的契约**准备输入 | `tests/test_gpt2_onnx.cpp` 的 `RunEngine` 注释与实现 | S0 的验收判据天然可表达为"同一份输入能绑定两条路"（AC5） |
| 现有 ONNX 侧的数值/性能测量是"5 次取平均、单次构建" | 同文件 `measure` lambda | 它**不是** S0 的判据来源；S0 只用数值判据（见 §5） |

## 1. 接口形态

### 契约结构（`builder.hpp`）

```cpp
struct OnnxIoContract {
    // 方案 B 的图必须与方案 A 同名同义。S0 之后是**两个**输入：
    //   input_ids     : INT32 [batch, seq]
    //   position_ids  : INT32 [batch, seq]        // 形状按 dim 下标规则，不单独声明
    const char* input_name;            // 保留（既有调用方与 CV 路径继续用它）
    const char* position_input_name;   // 新增；LLM 为 "position_ids"，CV 为 nullptr
    const char* output_name;
    bool require_position_input;       // CV 路径 false；LLM 路径 true
};
```

**为什么不直接换成 vector**：`OnnxIoContractFor` 的调用方（`BuildFromOnnx` 与测试）是少数，
但 vector 会引入堆分配与生命周期问题（返回 `const char*` 的指针语义）。字段式扩展最小、可读，
且 CV 路径（一个输入）用 `require_position_input = false` 表达，不写特例分支。

### 重写入口（Python 侧）

```text
python mini_trt_llm/tools/rewrite_onnx.py <in.onnx> <out.onnx> --contract-unify \
    [--report <path.json>]
```

产物契约：

- **只写新产物**，绝不原地改 `<in.onnx>`。
- `--report` 落的 JSON 至少含 `io_signature_before` / `io_signature_after`（名字 + dtype + 形状）
  与 `renamed_tensors`（必须为空数组）。
- 退出码：0 = 成功；非 0 = 规则不成立（§2 的前置探针失败、边界名变化等），并在 stderr 说明原因。

## 2. 图重写规则

### R1 dtype 统一（必做）

把图输入 `input_ids` 的声明类型从 INT64 改成 INT32，并在**仍需要 INT64 的消费点**（主要是
`Gather` / `Range` 的 indices）插入 `Cast(to=INT64)`。

**为什么"图内插 Cast"不算替代方案**：图输入的 dtype 由图声明决定，调用方喂什么类型与图内有没有
Cast 无关——图内插 Cast 改不了调用方要喂的类型（见 `design.md` 的 D5 方案 C）。

### R2 `position_ids` 提升为显式输入（必做）

新增图输入 `position_ids`（INT32，`[batch, seq]`），并把图内**位置编码的来源**改为由它驱动。

**开工前第一步是形态探针**（本机无图资产，**本次未核对真实图的形态**）：

```text
python mini_trt_llm/tools/inspect_onnx.py <资产图> --json     # 现有工具，列出节点与 I/O
```

按探针结果二选一，二者都要保证与改写前**数值逐位一致**（S0 不引入近似）：

| 形态 | 判据（探针看到什么） | 改写方式 |
|---|---|---|
| A. 位置索引来自常量 initializer | `wpe` 的查表 `Gather` 的 indices 输入是 initializer | 把该 initializer 输入换成新图输入 `position_ids` |
| B. 位置索引由图内 `Range` 生成 | 存在 `Range(0, seq_len, 1)` 之类的子图 | 把该 `Range` 的输出整体替换为 `position_ids`（`Range` 及其下游剩余部分保留） |

**两种形态都不成立时停止并上报**（不允许临机发明第三种改写方式）。

## 3. 改动点

| 文件 | 现状 | S0 改动 |
|---|---|---|
| `include/mini_trt_llm/core/builder.hpp` | `OnnxIoContract` 一个输入名 | 按 §1 扩字段（公开头） |
| `src/core/builder.cpp` 的 `OnnxIoContractFor` | LLM 只有 `input_ids` | LLM 增加 `position_ids` + `require_position_input = true` |
| `src/core/builder.cpp` 的 `BuildFromOnnx` | 只查输入 0 的名字 | 逐个核对声明的输入名与 **dtype**；缺 `position_ids` 即失败 |
| `src/core/builder.cpp` 的 profile | —— | **不改**（§0 第 3 行） |
| `tools/rewrite_onnx.py` | 不存在 | 新增（§1、§2） |
| `tools/inspect_onnx.py` | 基线写死单输入 | **按裁决 B（2026-10-06）**：`--check` 只对**源图**生效（基线不动）；重写后的图由重写脚本的 `RewriteReport` 自检 + S0 的 host 用例覆盖 |
| `tests/test_gpt2_onnx.cpp` | 按各引擎声明准备输入 | 改为**同一份 `input_ids` + `position_ids`（INT32）**喂两条路；保留逐形状对比 |

## 4. 失败语义（响亮拒绝，不静默降级）

| 情形 | 行为 |
|---|---|
| 图里缺 `position_ids` 输入，或 dtype 不是 INT32 | `BuildFromOnnx` 直接失败并打印实际名字与 dtype |
| 重写后边界张量名有变化（`renamed_tensors` 非空） | 重写脚本非零退出；**不产出半成品图** |
| 探针形态既不是 A 也不是 B | 停止并上报（§2） |
| 输入 dtype 在图中被多处依赖、Cast 点无法穷尽 | 停止并上报；不允许"先插一个 Cast 试试" |

## 5. 判据与来源

| 判据 | 来源 |
|---|---|
| 同一份输入可绑定两条路 | 修改后的 `tests/test_gpt2_onnx.cpp`（AC5） |
| 数值不退化 | **同精度**的既有阈值：FP32 用该用例的 `cosine > 0.999999` 与 `max_abs ÷ max|ref| < 1e-5`；FP16 用其自身出处（禁止跨精度复用，`AGENTS.md` §7） |
| 引擎可用性 | 构建 → 序列化 → 反序列化（`tools/inspect_engine.cpp` 是最小探针） |
| 重写未改边界 | `RewriteReport.renamed_tensors == []` |
| 不回归 | 沙箱 `ctest` 全绿；真机既有用例不出现新红 |

**S0 不使用性能判据**：dtype 与输入数量都不改变计算量级，性能结论留给 PF-7。

## 6. 测试方式

| 用例 | 判据 | 环境 |
|---|---|---|
| 重写脚本的 host 用例（对夹具图） | 输出图输入为 INT32 双输入；`renamed_tensors` 为空；规则不成立时报错 | 沙箱（需 `onnx` 包） |
| `Gpt2OnnxTest.*`（既有对拍用例改造） | 同一份输入喂两条路，数值判据不变 | 真机 |
| `--check` 基线用例 | §7 第 1 条裁决后的口径 | 沙箱（需 `onnx` + 资产） |

## 7. 待作者确认与待核实

| # | 事项 | 现状 |
|---|---|---|
| 1 | **基线归属**：`inspect_onnx.py --check` 现在把输入名写死为 `["input_ids"]`，S0 后必然报红 | **已裁决（作者 2026-10-06）= B**：`--check` 只对源图生效，重写图由重写器自检与 S0 的 host 用例覆盖。理由是探针的职责是"图变没变"，而重写图是我们自己生成的产物 |
| 2 | 真实图的位置编码形态（§2 的 A / B） | **未核实**：本机无 652MB 资产。开工第一步跑探针 |
| 3 | `input_ids` 在图内是否真的存在需要 INT64 的消费点 | **未核实**：同上 |

**来源核对结论**：S0 的每条"已定"都有来源（阈值来自既有用例、profile 行为来自代码、指纹来自代码）；
两条"未核实"已按 `AGENTS.md` §5 第 5 条标为**待定**并停在这里，未自行选定形态。
