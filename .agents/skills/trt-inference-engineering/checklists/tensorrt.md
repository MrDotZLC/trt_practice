# TensorRT Engineering Checklist

---

## Engine Lifecycle

- [P0] ICudaEngine生命周期是否明确？

- [P0] IExecutionContext生命周期是否明确？

- [P0] Runtime / Engine / Context ownership是否正确？

---

## Tensor

- [P0] Tensor shape是否明确？

- [P0] Dynamic shape profile是否覆盖输入范围？

- [P1] Tensor dtype转换是否正确？

- [P2] Layout是否合理？

---

## Plugin

- [P0] Plugin creator注册是否正确？

- [P0] Plugin serialize/deserialize是否完整？

- [P0] Plugin enqueue中的stream是否正确？

- [P1] Plugin workspace管理是否合理？

- [P2] Plugin是否可以进一步融合？

---

## Execution

- [P0] Binding index是否正确？

- [P0] CUDA stream是否传递正确？

- [P1] Async execution是否正确？

- [P2] 是否支持进一步优化？
