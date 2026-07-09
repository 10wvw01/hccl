# R9 全量 ST 回归验证结果

## 日期
2026-07-02

## 验证目的
验证 AllGatherMesh1D 模板修复（tx/rx 分离 sliceSize + LocalDataCopy 避免重叠）对全部使用 `InsTempAllGatherMesh1D` 的 executor 无副作用。

## 验证方法
逐测试套件运行，覆盖所有使用 AllGatherMesh1D 基类的 executor（Broadcast 3级/parallel、AllGather 3级/AICPU/parallel/sole、AllReduce 3级/parallel、ReduceScatter 3级/AICPU、Reduce）。

## 验证结果

| 测试套件 | 测试数 | 退出码 | 结果 |
|----------|--------|--------|------|
| AllGather 3Level | 23 | 0 | PASSED |
| AllReduce Multilevel | 23 | 0 | PASSED |
| Broadcast | 22 | 0 | PASSED |
| Broadcast 3Level | 11 | 0 | PASSED |
| AllGather AICPU | 17 | 0 | PASSED |
| AllReduce | 18 | 0 | PASSED |
| ReduceScatter 3Level | 16 | 0 | PASSED |
| ReduceScatter AICPU | 26 | 0 | PASSED |
| Reduce | 6 | 0 | PASSED |
| **合计** | **162** | — | **全部 PASSED** |

## 结论
AllGatherMesh1D 模板修复对全部 9 个测试套件 162 个用例无副作用，向后兼容性确认。

## 注意
一次性运行全部套件会因内存不足导致 segfault，需逐套件运行。
