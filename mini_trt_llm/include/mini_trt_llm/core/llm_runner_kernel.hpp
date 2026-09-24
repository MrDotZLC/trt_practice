#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mini_trt_llm {

// 把 KV Cache 的设备端语境长度搬进引擎的 position_ids 输入。
//
// **为什么需要它**：decode 每步的 `position_ids[b]` 等于"当前 token 的位置"，
// 也就是 cache 里已有的 token 数（`context_lens[b]`）。而 `context_lens` 每步都在
// 设备端被 `AppendDecodeKV` 推进（那样才能避免自回归循环里出现 H2D 拷贝）。
// 若为这一个整数回主机绕一圈，就等于把"循环内零 H2D/D2H"破了——所以用一次极短的
// kernel 在设备上搬。
//
// position_ids 的布局是 [batch, 1]，即每步只填一个位置；batch 维与 context_lens 对齐。
cudaError_t LaunchFillPositionIds(const int32_t* context_lens, int32_t* position_ids,
                                  int32_t batch_size, cudaStream_t stream);

}  // namespace mini_trt_llm
