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
