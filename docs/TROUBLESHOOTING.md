# 排查记录（Troubleshooting Log）

> 用途：记录问题排查的完整路径——现象、用过的命令、关键证据、根因、修复与回归防护。
>
> 与 `docs/PROGRESS.md` 的分工：
> - PROGRESS 的「已知问题与坑」只保留结论（问题 / 影响 / Workaround）并指向本文件对应条目；
> - 排查过程（怎么定位的、查了哪些头文件或文档、跑了哪些命令）写在本文件，避免交接文档膨胀成流水账。
>
> 编号只增不改，新记录追加在末尾。

---

## 1. Phase 0 utils 测试缺少 GPU 门控（已修复）

- **日期**：2026-09-24
- **现象**：沙箱内 `ctest` 有 6 个用例失败——`CudaCheckTest.SuccessDoesNotThrow`、
  `DeviceBufferTest.AllocateAndFree`、`DeviceBufferTest.Resize`、`PinnedBufferTest.AllocateAndFree`、
  `CudaTimerTest.CreateDestroy`、`CudaTimerTest.StartStop`，报错统一为
  `CUDA driver version is insufficient for CUDA runtime version (cudaErrorInsufficientDriver)`。
- **排查路径**：
  1. `ctest --test-dir build --output-on-failure`，看到 29 个用例中 6 个失败；
  2. 逐个查看失败输出，报错完全一致，且全部来自直接调用 CUDA API 的用例；
  3. 用 `--gtest_filter='RmsNorm*'` 做对照——新写的 RMSNorm 用例经 `GTEST_SKIP` 正常跳过，
     据此确认这 6 个失败与当次改动无关，属于环境差异而非代码缺陷。
- **根因**：这批 Phase 0 用例没有区分「环境不具备」与「代码有 bug」，无 GPU 时直接抛异常导致失败，而非跳过。
- **修复**：抽出 `mini_trt_llm/tests/test_gpu_guard.hpp` 的 `test_support::HasCudaDevice()`，
  给上述 6 个用例加跳过门控。
  例外：`CudaCheckTest.InvalidDeviceThrows` 刻意**不**门控——它验证的是 `CUDA_CHECK` 的失败路径，
  有无 GPU 都应当抛异常，加门控反而削弱覆盖。
- **回归防护**：无 GPU 环境下 `ctest` 全绿，真实回归信号不再被固定噪声淹没。

---

## 2. `supportsFormatCombination` 越界读取导致 engine 构建失败（已修复）

- **日期**：2026-09-24
- **现象**：真机跑 L2 集成用例 `RmsNormIntegrationTest.SerializedEngineInferenceMatchesCpuReference` 时，
  `IBuilder::buildSerializedNetwork` 返回 nullptr：

  ```
  Error Code 9: Internal Error ((Unnamed Layer* 1) [PluginV3_V3ONE]:
  could not find any supported formats consistent with input/output data types)
  ```
- **排查路径**：
  1. 报错函数是 `reportPluginError`（`optimizer/gpu/jit/pluginV3Builder.cpp`），落在**格式搜索**阶段，
     据此排除 kernel 计算问题，把范围收敛到 Plugin 的构建期接口；
  2. 回到 `supportsFormatCombination` 实现，原写法是
     `for (i = 0; i < nbInputs + nbOutputs; ++i)`——扫描了整个 `inOut` 数组；
  3. 查 `NvInferRuntime.h` 中该接口的说明，明确写着不要读 `pos` 之后的内容：

     > The override should not inspect inOut[pos+1..nbInputs+nbOutputs-1], which will have invalid values.
     > In other words, the decision for pos must be based on inOut[0..pos] only.
- **根因**：TensorRT 按 `pos` 递增调用该接口，`inOut[pos+1..]` 是未初始化内存。全数组扫描把随机值
  当作「类型/格式不一致」，于是**所有**格式组合都被判为不支持，格式搜索必然失败。
- **修复**：改为只与 `inOut[0]` 比对（即 TRT 文档给出的 polymorphic 写法）：

  ```cpp
  if (pos == 0) return true;
  return inOut[pos].desc.format == inOut[0].desc.format &&
         inOut[pos].desc.type == inOut[0].desc.type;
  ```
- **回归防护**：新增单测 `RmsNormPluginTest.IgnoresInvalidDescriptorsAfterPos`，在按契约不可读的位置
  塞入不兼容的 type/format，确认它们被忽略。
  同时修正了旧单测 `SupportsLinearFp32AndFp16Only`——它原先断言「`in_out[1]` 改成 HALF 后 `pos=2`
  应返回 false」，按正确契约 `pos=2` 只与 `inOut[0]` 比对，返回 true 才对，该断言实际是在奖励错误实现。
- **教训**：此类问题 host 侧单测复现不了（需要真实的格式搜索流程），只能靠真机 L2 集成测试兜住。
  后续 `RoPEPlugin` / `PagedAttentionPlugin` 的 `supportsFormatCombination` 一律按同一写法实现。

---

## 3. RoPE 的 GQA 校验用例打不中（已修复）

- **日期**：2026-09-24
- **现象**：`ctest` 报 `RoPEPluginTest.ConfigureRejectsInvalidRotaryDim` 失败：用例传 `num_heads=4 / num_kv_heads=3`
  期望 `configurePlugin` 因 GQA 不整除而返回 1，实际返回 0。
- **排查路径**：
  1. 读 `RoPEPlugin::configurePlugin` 实现，发现 head 配置是**由输入形状推导**的：
     `num_heads_ = query_dims.d[1]`、`num_kv_heads_ = key_dims.d[1]`，仅在对应维度为动态轴（`<= 0`）时才回退到属性值；
  2. 该用例把 key 的形状写成静态的 `[1, 4, 3, 8]`，于是 `num_kv_heads_` 被覆盖成 4，与 `num_heads_=4` 整除，校验自然通过；
  3. 结论是**测试预期写错了**，不是实现有缺陷——静态形状下不存在"属性与形状不一致"这种状态。
- **修复**：
  1. 从该用例中移除不整除分支，改为断言 shape 覆盖属性（`num_kv_heads()` 应等于形状里的 4）；
  2. 新增 `RoPEPluginTest.ConfigureRejectsNonDivisibleGqaHeads`，把 kv head 维度声明为动态轴（`-1`），
     此时属性才是唯一来源，GQA 整除校验才可被真正触发。
- **教训**：Plugin 的 `configurePlugin` 必须明确"形状权威还是属性权威"。本项目统一取**形状优先、属性兜底**，
  因此针对属性的校验用例必须构造动态轴，否则测的是另一条分支。

---

## 4. 部分旋转时 RoPE 输出尾部未被写入（已修复）

- **日期**：2026-09-24
- **现象**：真机跑 `RoPEKernelTest.SupportsPartialRotaryDim` 失败——`key index 15` 处
  `WithinTolerance` 为 false。同文件的另外两个 RoPE kernel 用例（`rotary_dim == head_size`）
  全部通过，只有部分旋转的用例挂掉。
- **排查路径**：
  1. 下标 15 是最后一行最后一维（`head_size = 8` 的第 7 维）。本用例 `rotary_dim = 4`，
     该维属于**不参与旋转**的尾部区间；
  2. 回看 kernel：原实现按"旋转对"分配线程，`total_pairs = rows * (rotary_dim / 2)`，
     每线程只写 `row_out[pair]` 与 `row_out[pair + half_rotary]`，即只覆盖 `[0, rotary_dim)`；
  3. `query_out` / `key_out` 是与输入**分离**的 buffer，`[rotary_dim, head_size)` 没有任何线程写入，
     残留未初始化数据。`rotary_dim == head_size` 时整行恰好都被覆盖，缺陷因此被掩盖；
  4. CPU 参考实现以 `*output = input` 起手，天然保留尾部，所以只可能是 kernel 侧错。
- **修复**：线程分配改为"一个线程负责一个输出元素"，非旋转区间显式写 `row_out[dim] = row_in[dim]`；
  旋转区间按 `pair = dim < half_rotary ? dim : dim - half_rotary` 定位配对维度。
- **回归防护**：`SupportsPartialRotaryDim` 保留在真机用例中；另外 `scripts/ref_rope.py` 与
  HuggingFace `apply_rotary_pos_emb` 交叉验证，最大差异 `0.000e+00`，同时确认尾部 4 维保持原值。
- **教训**：输出与输入分离的 kernel，只要存在"部分写入"的代码路径，就必须显式处理未被覆盖的区间。
  这一条已同步适用于后续所有 Plugin。

---

## 5. `SafetensorsLoader` 权重转换路径的三个缺陷（已修复）

- **日期**：2026-09-24
- **发现方式**：实现 Phase 1.5 的多权端到端测试时，先读代码核对"取权重的指针能否活到 engine 构建"，
  结果发现该函数根本无法正常工作。**这不是测试发现的，是读代码发现的**——原先没有任何用例覆盖 BF16 路径。
- **缺陷 1（最严重）：把 BF16 原始数据当 FP32 返回**
  - `SafetensorsToTrtDtype` 对 `kBFLOAT16` / `kFLOAT64` 没有分支，走 `default` 回退成 `kFLOAT`。
  - `GetConvertedData` 用 `src_trt_type == target_type` 判断"类型相同、可零拷贝"，于是
    "源是 BF16、目标 kFLOAT"被误判为同类型，直接返回 2 字节的 BF16 原始指针，调用方却按 4 字节读。
  - **修法**：不再用 `SafetensorsToTrtDtype` 做这个判断，改为显式的 `IsDirectCopy(src, target)`
    只承认 F32→F32、F16→F16 两种真正无需转换的情况。
- **缺陷 2：往设备内存里从主机侧写**
  - 转换结果写进成员 `conversion_buffer_`（`DeviceBuffer`，底层是 `cudaMalloc`），
    但转换循环是普通主机代码 `dst_f32[i] = ...`。这会段错误，且 `Resize()` 本身就需要 CUDA 上下文。
  - **修法**：换成按张量名索引的主机侧缓存 `std::map<std::string, std::vector<char>>`。
- **缺陷 3：BF16 → FP16 用位截断，数值上是错的**
  - 原注释写"BF16 与 FP16 的指数位相同，直接保留高 16 位"——**这个前提是错的**：
    BF16 是 1+8+7，FP16 是 1+5+10，指数宽度与偏置都不同。
  - **修法**：统一先还原成 FP32 再降到目标精度，顺带补齐 FP16→FP32 与 FP32→FP16
    （原实现遇到 FP16 源 + kFLOAT 目标直接报错返回 nullptr）。
- **附带修复**：转换结果原先共用一个缓冲区，取第二个权重会覆盖第一个；而 `nvinfer1::Weights`
  只存裸指针、到 `buildSerializedNetwork` 才读数据，会**静默构建出错误网络**。缓存按 key 分开后消除。
- **回归防护**：`test_safetensors_loader.cpp` 新增 8 条 host 用例，覆盖 dtype 组合矩阵、
  关键断言 `ConversionResultsForDifferentTensorsDoNotAlias`（取第二个后第一个内容不变）与
  `Bf16ConvertsToFp32`（字节数必须按目标类型算）。这些用例**不需要 GPU**。
- **教训**：整条 BF16 路径此前没有任何用例覆盖，却一直被文档标注为"已实现"。
  "有代码"不等于"能用"——P1.5-7 修正 Phase 0 验收判据正是针对这一点。

---

## 6. `IModelBuilder::Build` 拿不到权重（已修复）

- **日期**：2026-09-24
- **现象**：写第一个端到端 builder 时编译失败：`passing 'const mini_trt_llm::WeightLoader' as
  'this' argument discards qualifiers`。
- **排查路径**：`IModelBuilder::Build` 的签名是 `const WeightLoader&`，而 `WeightLoader::GetWeight`
  是非 const——因为它需要写转换缓存。也就是说**任何 builder 都无法读取权重**，
  包括 Phase 2 的 `GPT2ModelBuilder`。
- **根因**：接口设计问题。读取权重在语义上是只读操作，转换缓存是纯粹的实现细节，
  不应影响方法的 const 性。
- **修复**：
  1. `SafetensorsLoader::GetConvertedData` 改为 const，缓存成员声明为 `mutable`；
  2. `WeightLoader::GetWeight` / `GetWeightBySourceKey` 相应改为 const。
- **教训**：这类"签名对不上"的问题会在写下第一个真实使用者时立刻暴露，
  所以端到端用例的价值不只是验证数值，也在于**强迫接口被真实调用一次**。

---

## 7. `BuildFromConfig` 的失败顺序让错误路径无法在无 GPU 环境下测试（已调整）

- **日期**：2026-09-24
- **现象**：设计 E4 错误路径用例时发现，"权重文件缺失"这类场景在沙箱里跑不到——
  它们排在 `createInferBuilder` 之后，而该调用需要 GPU 驱动。
- **排查路径**：读 `BuildFromConfig` 的语句顺序：`ModelConfig::Load` → 注册表查找 →
  **`SetupBuilder`（创建 IBuilder）** → `WeightLoader::Load` → build。
- **根因**：纯数据校验（配置文件、权重文件）被排在了昂贵的 TRT 初始化之后。
- **调整**：把 `WeightLoader::Load` / `SetWeightMap` / `SetDefaultPrecision` 提到 `SetupBuilder` 之前。
- **收益**：其一，缺失权重时不再需要初始化 TRT，失败更快、错误信息更直接；
  其二，E4 中 3 条纯数据校验用例得以在沙箱内实际执行（另外 2 条仍需 GPU，因为要走到 `Build`）。
- **教训**：**校验顺序决定了错误路径的可测性**。把廉价且无外部依赖的检查前置，
  既省时间又让更多失败分支进入 CI。

---

## 8. 反序列化后的 Plugin 在 `onShapeChange` 上被误判为形状不一致（已修复）

- **日期**：2026-09-24
- **现象**：真机跑 Phase 1.5 的端到端用例，RoPE 相关的三条全部在 enqueue 失败：

  ```
  [ERROR]   RoPE: shape change alters head configuration
  [TRT ERROR] [pluginV3Runner.cpp::onShapeChange::96] Assertion pluginUtils::isSuccess(status) failed
  ```

  同一批用例里 RMSNorm 的通过、RoPE 的失败，是定位的关键线索。
- **排查路径**：
  1. 报错来自插件自己的 `onShapeChange`，说明失败在**运行期形状校验**，不是 kernel 计算；
  2. 对照通过/失败用例的差异：RMSNorm 序列化了 `hidden_size`，而 RoPE 只序列化
     `rotary_dim` / `base`，head 配置（`num_heads` / `num_kv_heads` / `head_size`）靠
     `configurePlugin` 从形状推导——**差别就在"哪些属性被序列化"**；
  3. 确认调用顺序：engine 是存盘后重新加载的，TRT 为反序列化实例创建的是 runtime-phase plugin，
     **不会再调用 `configurePlugin`**；首个 enqueue 前只会调 `onShapeChange`。此时
     `head_size_` 仍是默认值 0，于是 `8 != 0` 被判成"形状改了 head 配置"。
- **根因**：§2.12 里定的"head 配置不做序列化属性、从形状推导"对**构建期实例**成立，
  对**运行期实例**不成立——后者根本没经历过"推导"那一步。
- **影响面**：`RoPEPlugin` 与 `PagedAttentionPlugin` 同时中招（两者都只序列化部分属性）；
  PagedAttention 还会因 `num_kv_heads_ == 0` 在 enqueue 里除零。`RmsNormPlugin` 因序列化了
  `hidden_size` 而幸免——这也解释了为什么 Phase 1 只有 RMSNorm 做了 L2 集成测试时没有暴露。
- **修复**：
  1. 两者的 `onShapeChange` 改为**以运行期形状为准刷新**缓存，仅在真正矛盾时失败
     （GQA 不整除、`rotary_dim` 非法、cache block 维与 `block_size` 不符）；
  2. `PagedAttentionPlugin::enqueue` 增加兜底：当 `num_kv_heads_` 尚未被刷新时，从 cache 形状取。
- **回归防护**：新增 4 条**不需要 GPU** 的用例——
  `RoPEPluginTest.DeserializedPluginRefreshesHeadConfigFromShape`、
  `RoPEPluginTest.DeserializedPluginStillRejectsInconsistentShape`、
  `PagedAttentionPluginTest.DeserializedPluginRefreshesHeadConfigFromShape`、
  `PagedAttentionPluginTest.DeserializedPluginRejectsCacheBlockMismatch`。
- **教训**：带序列化的组件必须区分**构建期实例**与**运行期实例**。凡是"从形状推导"得到的状态，
  运行期入口都必须能自行推导，不能依赖只发生在构建期的初始化。此前的 L0 用例只覆盖了构建期那一半，
  因此这类缺陷只能靠真机暴露——新增的 4 条用例把这一半补上了，且仍留在 CI 可跑的范围内。

---

## 9. 参考实现漏了 batch 维度，把失败误指到 kernel（已修复）

- **日期**：2026-09-24
- **现象**：真机跑 `Fp16PathTest.RoPEHandlesFp16WithGqaAndMultipleBatches`，
  `query index 64` 处 `WithinTolerance` 失败。
- **排查路径**：
  1. 失败下标 64 恰好是 **batch 1 的首元素**（`heads=4 × seq=2 = 每 batch 8 行`，第 8 行 ×
     `head_size=8` = 64）。**"错误正好从 batch 边界开始"直接指向 batch 维度的处理**，
     这是本次定位最快的一步；
  2. 进一步收敛范围：kernel 的 batch 分解在 batch=1 时走的是同一段代码，而 E2 全链路
     （batch=1）里 RoPE 段是通过的 → 可以先排除 kernel；
  3. 读参考实现，发现 `ReferenceRoPE` 的签名是 `positions [seq]`，内部只查 `positions[s]`，
     **没有 batch 维度**；而调用方传入的是 `[batch * seq_len]` 的扁平数组。签名与实参形状
     不一致，却没有任何断言——于是从 batch 1 起所有位置都用错，且完全静默。
- **根因**：参考实现的契约既没有显式表达（参数声明 `[seq]`，实参 `[batch*seq]`），
  也没有前置断言；加之此前所有 RoPE 用例都是 batch=1，问题被完全掩盖。
- **修复**：
  1. 参考实现收敛到 `tests/test_reference.hpp` 作为**唯一来源**。原先
     `e2e_mini_decoder_reference.hpp` 与 `test_rope_plugin.cpp` 各存一份 RoPE 参考，
     两处独立维护必然漂移——这正是分裂出错误版本的土壤；
  2. `ReferenceRoPE` 增加 `batch_size` 参数，并在入口断言
     `positions.size() == batch * seq_len`、input 形状、`rotary_dim` 合法性，
     违反即抛 `std::invalid_argument`；
  3. `ReferenceRmsNorm` / `ReferencePagedAttentionDecode` 同样加前置断言；
  4. `test_rope_plugin.cpp` 的局部实现改为委托给共享实现；
  5. 补齐同一问题的最后一处：`test_reference.hpp` 里 `CpuRmsNorm`（浮点）与
     `ReferenceRmsNorm`（双精度）**同时存在**，又是一份算子两份参考。已把前者改为
     委托后者的薄封装，消除该文件内部的自相矛盾（否则等于在同一个文件里违反本条教训）。
- **回归防护**：
  1. 新增 `tests/test_reference_helpers.cpp`：7 条 **host 侧**用例，可进 CI。核心是
     `RopeIsBatchAware` —— 用手算规模（batch=2 / seq=1 / head_size=2 / position={0,1}）断言
     batch 1 应得到 `{cos1, sin1}`；旧实现会返回恒等结果，被这条用例拦下。
     其余覆盖形状不匹配抛异常、RMSNorm 手算值、单 token 注意力等于 V、GQA 共享 kv head 等；
  2. `scripts/ref_rope.py` 扩到 batch=2，并加入自检（不同 batch 必须得到不同结果）。
     **顺带发现并修掉脚本自身的错误**：HF 的 `apply_rotary_pos_emb` 内部会自行
     `cos.unsqueeze(unsqueeze_dim=1)`，脚本原先又手工 unsqueeze 了一次，batch=1 时靠广播
    碰巧通过、batch>1 时广播错位。修好后 batch=2 的交叉验证差异为 `0.000e+00`。
  3. 新增配置预检用例 `RopeAcceptsFp16TestConfiguration`：在 host 侧复刻这次失败的 GPU 用例
     的维度配置（batch=2 / heads=4 / kv_heads=2 / seq=2 / head_size=8 / rotary_dim=4 /
     位置 `{2,5,7,11}`），断言参考实现接受该配置且输出逐 batch 可区分。
     **这类"用例配置 vs 参考契约"的不一致本可以在 CI 阶段发现**，不必再花一次真机往返。
- **复验结果**：修复后真机 `--gtest_filter='Fp16PathTest.*:E2eMiniDecoderTest.*'` 的 5 条用例
  **全部通过**，与 HF 的 `apply_rotary_pos_emb` 在 batch=2 下差异为 `0.000e+00`。
- **教训**：
  1. **参考实现也是被测对象**。它是裁决对错的标尺，标尺错了会给出错误裁决。参考实现是纯 host
     代码，完全可以进 CI——把它纳入 CI 的成本远低于一次真机往返；
  2. **契约要断言，不要靠约定**。"声明 `[seq]`、实参 `[batch*seq]`"这种不一致，编译期和
     运行期都不会报错；
  3. **`batch=1` 会藏住一整类 bug**。本次两个缺陷（C++ 参考、Python 交叉验证脚本）在 batch=1
     时都表现为"通过"。凡是带 batch 维的算子，用例必须覆盖 `batch > 1`；
  4. **参考实现应当唯一**。"参考与被测必须独立"针对的是参考 vs 实现；同一算子的两份参考
     彼此之间只会漂移。
