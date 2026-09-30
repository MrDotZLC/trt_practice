# C++ Engineering Checklist

说明：

每个检查项必须标注等级：

- P0: 不通过禁止进入下一阶段
- P1: 需要人工确认
- P2: 质量优化

---

# Resource Management

- [P0] RAII 是否用于管理资源生命周期？

- [P0] Ownership 是否明确？

  示例：

  - unique_ptr
  - shared_ptr
  - reference
  - raw pointer

- [P0] 是否存在悬空引用风险？

- [P1] 异常路径是否释放资源？

- [P2] 命名是否符合项目规范？

---

# Concurrency

- [P0] 多线程访问是否存在数据竞争？

- [P0] Mutex / Lock 生命周期是否正确？

- [P1] 是否存在不必要锁竞争？

- [P2] 是否可以进一步优化并发模型？

---

# Interface Design

- [P0] API 输入输出是否明确？

- [P0] 错误处理方式是否统一？

- [P1] 接口是否容易扩展？

- [P2] 是否符合代码风格？

---

# Build

- [P0] Debug/Release 是否均可编译？

- [P1] 是否引入额外依赖？

- [P2] CMake结构是否清晰？
