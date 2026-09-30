# CUDA Engineering Checklist

---

# Kernel Correctness

- [P0] Kernel边界条件是否正确？

- [P0] 是否存在越界访问？

- [P0] Synchronization 是否正确？

  检查：

  - __syncthreads()
  - warp primitive

- [P1] 不同shape是否覆盖测试？

- [P2] Kernel代码可读性是否良好？

---

# Memory

- [P0] Global Memory访问是否安全？

- [P0] 是否存在race condition？

- [P1] Memory coalescing是否优化？

- [P1] Shared Memory是否存在bank conflict？

- [P2] 是否可以进一步减少访存？

---

# Performance

- [P0] 优化是否有benchmark证明？

- [P1] 是否分析compute bound / memory bound？

- [P1] 是否考虑occupancy？

- [P2] 是否进行了进一步profiling？

---

# CUDA Runtime

- [P0] Stream生命周期是否正确？

- [P0] Event同步是否正确？

- [P1] Async API是否正确使用？

- [P2] 是否避免不必要同步？
