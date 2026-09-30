# LLM Runtime Checklist

---

# Request Scheduling

- [P0] Request生命周期是否明确？

- [P0] Prefill / Decode流程是否区分？

- [P0] Batch状态是否一致？

---

# KV Cache

- [P0] KV Cache ownership是否明确？

- [P0] Block管理是否正确？

- [P0] Cache eviction策略是否明确？

- [P1] Memory fragmentation是否考虑？

- [P1] Long context是否测试？

- [P2] 是否支持进一步优化？

---

# Continuous Batching

- [P0] Dynamic request加入/退出是否安全？

- [P0] Scheduler状态是否一致？

- [P1] Batch调度策略是否合理？

- [P2] 是否考虑token级调度？

---

# Sampling

- [P0] Sampling结果是否正确？

- [P1] CUDA kernel是否验证？

- [P2] 是否支持更多策略？
