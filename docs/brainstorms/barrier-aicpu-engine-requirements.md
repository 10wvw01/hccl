# Barrier AICPU 引擎支持需求文档

## 目录

- [问题背景](#问题背景)
- [目标](#目标)
- [方案对比](#方案对比)
  - [方案全览](#方案全览)
  - [排除分析](#排除分析)
  - [候选方案对比](#候选方案对比)
  - [规模趋势总结](#规模趋势总结)
- [选定方案：串行两级 Mesh1D(框内) + NHR(框间)，全 AICPU](#选定方案串行两级-mesh1d框内--nhr框间全-aicpu)
  - [执行流程](#执行流程)
  - [选型理由](#选型理由)
- [范围边界](#范围边界)
  - [In Scope](#in-scope)
  - [Out of Scope](#out-of-scope)
- [场景路由](#场景路由)
- [涉及文件](#涉及文件)
- [关键设计决策](#关键设计决策)
  - [NHR AICPU 模板复用 DPU 版的 RunNHRBarrier 逻辑](#nhr-aicpu-模板复用-dpu-版的-runnhrbarrier-逻辑)
  - [Executor 复用 SequenceExecutor](#executor-复用-sequenceexecutor)
  - [barrier_op.cc 引擎白名单](#barrier_opcc-引擎白名单)
- [成功标准](#成功标准)
- [风险分析](#风险分析)
- [实现方案](#实现方案)
  - [1. 新增 ins_temp_barrier_nhr_aicpu.h](#1-新增-ins_temp_barrier_nhr_aicpuh)
  - [2. 新增 ins_temp_barrier_nhr_aicpu.cc](#2-新增-ins_temp_barrier_nhr_aicpucc)
  - [3. 修改 template/aicpu/CMakeLists.txt](#3-修改-templateaicpucmakeliststxt)
  - [4. 新增 executor/ins_v2_barrier_sole_executor.h 和 .cc](#4-新增-executorins_v2_barrier_sole_executorh-和-cc)
  - [5. 修改 executor/ins_v2_barrier_sequence_executor.cc](#5-修改-executorins_v2_barrier_sequence_executorcc)
  - [6. 修改 selector/barrier_auto_selector.h](#6-修改-selectorbarrier_auto_selectorh)
  - [7. 修改 selector/barrier_auto_selector.cc](#7-修改-selectorbarrier_auto_selectorcc)
  - [算法选择路由](#算法选择路由)
  - [8. 修改 barrier_op.cc](#8-修改-barrier_opcc)
  - [改动量汇总](#改动量汇总)
- [Barrier AICPU 全流程调用链](#barrier-aicpu-全流程调用链)
- [测试方案](#测试方案)
- [架构分析](#架构分析)
  - [整体架构图](#整体架构图)
  - [三层正交分离](#三层正交分离)
  - [两条引擎路径共存](#两条引擎路径共存)
  - [三种算法形态](#三种算法形态)
  - [架构优点](#架构优点)
  - [架构风险](#架构风险)

## 问题背景

### Barrier 算子介绍

Barrier 是**集合通信同步原语**：通信域内所有 rank 到达 Barrier 调用点后，才能继续往下执行。不搬运任何数据，纯同步。

**典型用途**：

- **流水线同步**：多个集合通信算子之间插入 Barrier，保证前一个全部完成才开始下一个
- **阶段对齐**：训练迭代之间、前向/反向之间插入 Barrier，保证所有 rank 处于同一阶段
- **资源安全**：保证前面算子对 buffer 的读写全部完成后，才允许后续算子覆盖

**原理**：利用已有 channel 通信链路，发**空 slice**（无数据）的 SendRecv/Send/Recv，只做 Notify signal 的收发，不搬运数据。

```
Rank0 ──signal──> Rank1     "我到了"
Rank0 <──signal── Rank1     "我也到了"
（所有 rank 互相确认到达后，Barrier 完成）
```

**实现方式对比**：

| 方式 | 实现 | 缺点 |
|------|------|------|
| 旧方案（hcomm） | AllReduce(SUM, 8 bytes) | 传 32 字节无用数据 + 做 SUM 计算，浪费带宽和算力 |
| 新方案（hccl_2039） | 空 slice signal 同步 | 无数据搬运，纯信号，零带宽浪费 |

**拓扑与执行**：950 芯片多级拓扑下，Barrier 分框内框间两级执行：

```
先框间：NHR 算法，log(M) 步串行，每步与一个远端 rank 做 signal 同步
后框内：Mesh1D 算法，1 步全并行，同时与所有本地 rank 做 signal 同步
```

框内走 HCCS（微秒级），框间走网络（十微秒级），分级执行充分利用快链路。

**引擎支持**：

| 引擎 | 框内 | 框间 | 状态 |
|------|------|------|------|
| HostDPU | AICPU Mesh1D | DPU NHR | 已实现 |
| AICPU | AICPU Mesh1D | AICPU NHR | 本次新增 |
| CCU | — | — | 不支持 |
| AIV | — | — | 不支持 |

**性能特征**：

- **无数据搬运**：不占用带宽，不占用计算资源
- **延迟取决于步数**：两级 = log(M)×网络延迟 + 1×HCCS延迟
- **线程开销**：框内 Mesh1D 需 N_intra-1 个线程，框间 NHR 需 1 个线程
- **channel 开销**：框内 N_intra-1 条 + 框间 log(M) 条

### 当前问题

当前 barrier 算子在新流程中仅实现了 HostDPU 引擎（框内 AICPU Mesh1D + 框间 DPU NHR）。950 芯片支持多种引擎（HostDPU、AICPU、CCU、AIV），但 AICPU 引擎尚未实现，对应场景回退到旧 `HcclBarrier`（hcomm 的 AllReduce(count=8) 实现），浪费带宽和计算资源，且语义不清晰。

需要为 barrier 增加 AICPU 引擎支持，消除对应场景对旧 AllReduce 回退的依赖。

## 目标

- 新增 AICPU 引擎的 barrier 实现，覆盖 950 芯片 AICPU 引擎场景
- 不搬运数据，纯 signal 同步，消除 AllReduce 回退的带宽/计算浪费
- 与现有 HostDPU 路径共存，selector 按引擎类型自动选择
- CCU/AIV 引擎本次不涉及

## 方案对比

以 32 rank、4 pod、每 pod 8 rank 为例（log₂(4)=2，log₂(32)=5）：

### 方案全览

| # | 方案 | 描述 | 步数 | 线程 | channel | 框内利用快链路 | 兼容性 | 复杂度 | 延迟 | 评估 |
|---|------|------|------|------|---------|-------------|--------|--------|------|------|
| ① | AllReduce(8) 回退 | 复用 AllReduce(SUM, 8 bytes) 实现 barrier，旧 hcomm 方案 | ~3-5 | ~7 | 10 | ✓ | 全场景 | 极低 | 中 | 浪费带宽+计算，待替代 |
| ② | 串行 Mesh1D+NHR DPU | 框内 Mesh1D(AICPU) + 框间 NHR(DPU)，串行执行，已有 | 1+log(M)=3 | N_intra-1=7 | 10 | ✓ | 仅HostDPU | 已实现 | 低 | 现有方案，不改 |
| ③ | 拍平 Mesh1D（全并行） | 全 rank 扁平，Mesh1D 一步全并行，所有 rank 同时互相 SendRecvWrite | 1 | N_total-1=31 | 31 | ✓ | 全AICPU | 低 | 低 | 一次性线程数多，见排除分析 |
| ③' | 拍平 Mesh1D（分轮并发） | 照搬 AlltoAll 模式，限制并发数为 16，分轮执行 | ceil((N-1)/16)=2 | min(16,N-1)=16 | 31 | ✓ | 全AICPU | 低 | 低 | channel 数多，但线程可控 |
| ④ | 拍平 NHR | 全 rank 扁平，NHR 串行 log(N) 步，每步 partner 距离翻倍 | log(N)=5 | 1 | 31 | ✓ | 全AICPU | 低 | 中高 | 步数多，不分级编排 |
| ⑤ | 串行两级 Mesh1D+NHR AICPU | 框内 Mesh1D(1步并行) + 框间 NHR(log(M)步串行)，先框间后框内，全 AICPU | 1+log(M)=3 | N_intra-1+1=8 | 10 | ✓ | 全AICPU | 中 | 中 | **综合最优** |
| ⑥ | 并行两级 Mesh1D+NHR AICPU | 框内 Mesh1D 和框间 NHR 同时执行，前同步后汇合，全 AICPU | max(1,log(M))=2 | N_intra+1=9 | 10 | ✓ | 全AICPU | 中高 | 中低 | 收益有限，复杂度高 |

### 排除分析

- **① AllReduce(8)**：传 32 字节无用数据 + 做 SUM 计算，就是要被替代的方案
- **③ 拍平 Mesh1D（全并行）**：一次性 N-1 个线程全并行，32 rank 需 31 个线程；虽然未超 AICPU 上限（200），但资源开销大
- **③' 拍平 Mesh1D（分轮并发）**：照搬 AlltoAll 的 `ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE=16` 分轮模式，线程可控（16），但 channel 数仍为 N-1（31），远多于两级方案的 10 条；且不分级编排，无法按拓扑层级优化执行顺序

### 候选方案对比

以 32 rank、4 pod、每 pod 8 rank 为例（log₂(4)=2，log₂(32)=5）：

| 维度 | ② DPU（已有） | ③' 拍平Mesh1D（分轮） | ④ 拍平 NHR | ⑤ 串行两级 AICPU | ⑥ 并行两级 AICPU |
|------|-------------|---------------------|-----------|----------------|----------------|
| 步数/轮数 | 3 | 2 | 5 | 3 | 2 |
| 线程 | 7 | 16 | 1 | 8 | 9 |
| channel | 10 | 31 | 31 | 10 | 10 |
| 框内利用快链路 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 延迟 | 低（DPU网络快） | 低 | 中高 | 中 | 中低 |
| 与DPU版结构对称 | — | ✗ | ✗ | ✓ | ✓ |
| 新增代码量 | 0 | ~150行 | ~150行 | ~250行 | ~350行 |
| 复用现有executor | — | ✓ SoleExecutor | 需新建 | ✓ SequenceExecutor | 需新建Concurrent |
| 风险 | 无 | 低 | 低 | 低 | 中（并行同步） |

以 64 rank、8 pod、每 pod 8 rank 为例（log₂(8)=3，log₂(64)=6）：

| 维度 | ② DPU（已有） | ③' 拍平Mesh1D（分轮） | ④ 拍平 NHR | ⑤ 串行两级 AICPU | ⑥ 并行两级 AICPU |
|------|-------------|---------------------|-----------|----------------|----------------|
| 步数/轮数 | 1+log(8)=4 | ceil(63/16)=4 | log(64)=6 | 1+log(8)=4 | max(1,log(8))=3 |
| 线程 | 7 | 16 | 1 | 8 | 9 |
| channel | 7+7=14 | 63 | 63 | 7+7=14 | 14 |
| 框内利用快链路 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 延迟 | 低（DPU网络快） | 低 | 高 | 中 | 中低 |
| 与DPU版结构对称 | — | ✗ | ✗ | ✓ | ✓ |
| 风险 | 无 | 低 | 低 | 低 | 中（并行同步） |

以 128 rank、16 pod、每 pod 8 rank 为例（log₂(16)=4，log₂(128)=7）：

| 维度 | ② DPU（已有） | ③' 拍平Mesh1D（分轮） | ④ 拍平 NHR | ⑤ 串行两级 AICPU | ⑥ 并行两级 AICPU |
|------|-------------|---------------------|-----------|----------------|----------------|
| 步数/轮数 | 1+log(16)=5 | ceil(127/16)=8 | log(128)=7 | 1+log(16)=5 | max(1,log(16))=4 |
| 线程 | 7 | 16 | 1 | 8 | 9 |
| channel | 7+15=22 | 127 | 127 | 7+15=22 | 22 |
| 框内利用快链路 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 延迟 | 低（DPU网络快） | 高 | 高 | 中 | 中低 |
| 与DPU版结构对称 | — | ✗ | ✗ | ✓ | ✓ |
| 风险 | 无 | 低 | 低 | 低 | 中（并行同步） |

### 规模趋势总结

| 维度 | 32卡 | 64卡 | 128卡 | 趋势 |
|------|------|------|-------|------|
| ③' 拍平轮数 | 2 | 4 | 8 | 线性增长 |
| ⑤ 两级步数 | 3 | 4 | 5 | 对数增长 |
| ③' channel | 31 | 63 | 127 | 线性增长 |
| ⑤ channel | 10 | 14 | 22 | 增长但斜率低 |
| ③'/⑤ 步数比 | 0.67 | 1.0 | 1.6 | 两级优势随规模放大 |
| ③'/⑤ channel 比 | 3.1 | 4.5 | 5.8 | 两级优势随规模放大 |

**结论**：规模越大，两级方案优势越大——拍平方案的轮数和 channel 随 rank 数线性增长，两级方案对数增长。串行两级在代码复杂度和资源之间取得最佳平衡。

## 选定方案：串行两级 Mesh1D(框内) + NHR(框间)，全 AICPU

### 执行流程

```
每个 rank:
  1. 框间 NHR AICPU（log(M)步串行，1线程，网络 channel）
  2. 框内 Mesh1D（1步全并行，N_intra-1线程，HCCS channel）

先框间后框内，与现有 DPU 版 SequenceExecutor 顺序一致。
```

### 选型理由

1. **步数少**：3 步 vs 拍平 NHR 的 5 步，框内 Mesh1D 一步并行完成
2. **框内利用快链路**：Mesh1D 走 HCCS（框内高速互联），不退化到网络
3. **线程可控**：8 个线程（N_intra-1+1），低于拍平 Mesh1D 分轮的 16 个
4. **channel 少**：10 条 channel，远低于拍平方案的 31 条
5. **与 DPU 版结构对称**：复用 `SequenceExecutor`，无需新写 executor
6. **代码量最小**：仅新增 1 个模板文件（NHR AICPU），executor 和 selector 复用/微调
7. **风险最低**：不引入并行同步复杂度

## 范围边界

### In Scope

- 新增 `InsTempBarrierNhrAicpu` 模板（框间 AICPU NHR）
- 新增 executor 注册 `InsBarrierMesh1DNhrAicpu`（两级：Mesh1D + NHR AICPU）
- 新增 executor 注册 `InsBarrierMesh1D`（单级：仅 Mesh1D，复用现有模板）
- 新增 executor 注册 `InsBarrierNhrAicpu`（单级：仅 NHR，CLOS/每server出1卡）
- 修改 selector 增加 `SelectAicpuAlgo` 路径，按 `topoLevelNums` 分流
- 修改 `barrier_op.cc` 增加引擎白名单，HostDPU 和 AICPU 引擎走新流程，其余回退

### Out of Scope

- CCU 引擎 barrier
- AIV 引擎 barrier
- 图模式（`HcclBarrierGraphMode`）——当前 barrier 无图模式入口，`opMode` 硬编码为 `OPBASE`，本次不涉及
- 并行执行优化（方案⑥）
- 拍平算法（方案③③'④）
- 现有 HostDPU 路径的任何改动

## 场景路由

```
950 + HostDPU 引擎 + 多级拓扑    → selector 选 InsBarrierMeshNhrDPU（串行 DPU，已有，不改）
950 + AICPU 引擎 + 多级正常拓扑  → selector 选 InsBarrierMesh1DNhrAicpu（两级 AICPU，新增）
950 + AICPU 引擎 + 多级每server1卡 → selector 选 InsBarrierNhrAicpu（单级 NHR，新增）
950 + AICPU 引擎 + 单级 Mesh1D   → selector 选 InsBarrierMesh1D（单级 Mesh1D，新增）
950 + AICPU 引擎 + 单级 CLOS     → selector 选 InsBarrierNhrAicpu（单级 NHR，新增）
950 + CCU/AIV 引擎               → 本次不涉及，回退旧 HcclBarrier
非950                            → 回退旧 HcclBarrier（不变）
```

## 涉及文件

| 文件 | 操作 |
|------|------|
| `src/ops/barrier/template/aicpu/ins_temp_barrier_nhr_aicpu.cc` | 新增 |
| `src/ops/barrier/template/aicpu/ins_temp_barrier_nhr_aicpu.h` | 新增 |
| `src/ops/barrier/template/aicpu/CMakeLists.txt` | 加新文件 |
| `src/ops/barrier/executor/ins_v2_barrier_sole_executor.h` | 新增 |
| `src/ops/barrier/executor/ins_v2_barrier_sole_executor.cc` | 新增（含单级注册） |
| `src/ops/barrier/executor/CMakeLists.txt` | 加新文件 |
| `src/ops/barrier/executor/ins_v2_barrier_sequence_executor.cc` | 修改（加两级注册） |
| `src/ops/barrier/selector/barrier_auto_selector.cc` | 加 SelectAicpuAlgo（按 topoLevelNums/Level1Nhr/localNetInsSize 分流） |
| `src/ops/barrier/selector/barrier_auto_selector.h` | 加 SelectAicpuAlgo 声明 |
| `src/ops/barrier/barrier_op.cc` | 引擎白名单判断 |

## 关键设计决策

### NHR AICPU 模板复用 DPU 版的 RunNHRBarrier 逻辑

DPU 版 `InsTempBarrierNHRDPU::RunNHRBarrier` 已实现完整的 NHR step 通信逻辑（空 slice SendRecvWrite/SendWrite/RecvWrite）。AICPU 版直接复用这段逻辑，区别仅在于：

- DPU 版：通过 `HcommSendRequest`/`HcommWaitResponse` 发给 DPU 执行，`RunNHRBarrier` 在 `#ifndef AICPU_COMPILE` 下
- AICPU 版：直接在 AICPU 上执行 `RunNHRBarrier`，去掉 DPU 的请求-响应封装

### Executor 复用 SequenceExecutor

现有 `InsV2BarrierSequenceExecutor` 是模板化的，通过 `REGISTER_EXECUTOR_BY_TWO_TEMPS` 注册不同模板组合。新增注册即可：

```cpp
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_BARRIER,
                               InsBarrierMesh1DNhrAicpu,        // 新 algName
                               InsV2BarrierSequenceExecutor,    // 复用
                               TopoMatchMultilevel,             // 复用
                               InsTempBarrierMesh1D,            // 复用框内模板
                               InsTempBarrierNhrAicpu);         // 新框间模板
```

### barrier_op.cc 引擎白名单

在 `barrier_op.cc` 中用白名单判断是否走新流程，仅 HostDPU 和 AICPU 引擎放行，其余直接回退：

```cpp
CHK_RET(HcclGetOpExpansionMode(comm, param));

// 仅支持 HostDPU 和 AICPU 引擎，其余回退旧 HcclBarrier
bool isHostDpu = IsBarrierHostDpu(comm);
bool isAicpu = (param.opExecuteConfig == OpExecuteConfig::AICPU_TS);
if (!isHostDpu && !isAicpu) {
    return BarrierFallbackToOldFlow(comm, stream);
}

// Selector 按 opExecuteConfig 自动选 DPU 或 AICPU 算法
SelectorStatus selRet = Selector(comm, param, topoInfo, algName);
if (selRet != SelectorStatus::MATCH) {
    return BarrierFallbackToOldFlow(comm, stream);
}
```

**选择白名单而非 Selector 全分发的原因**：

- CCU 引擎在 Selector 内部有级联 fallthrough（CCU_MS → CCU_SCHED → CCU_FAIL → SelectAicpuAlgo），若不拦截会误匹配 AICPU 算法
- 白名单方式显式、可读，CCU/AIV 在进入 Selector 前就回退，避免不必要的分发开销
- 无需在 `SelectAicpuAlgo` 中加 CCU_FAIL 守卫等 hack
- 将来新增 CCU/AIV 支持时，只需在白名单加对应引擎 + selector 加对应 `Select*Algo`

## 成功标准

1. 950 芯片 AICPU 引擎场景，barrier 走新 AICPU 流程，不再回退到旧 AllReduce
2. HostDPU 引擎场景行为不变，仍走 DPU 路径
3. CCU/AIV 引擎场景及非 950 场景仍回退旧 HcclBarrier
4. AICPU barrier 不搬运任何数据，仅做 signal 同步
5. 代码编译通过（AICPU 编译模式）
6. Selector 返回 NOT_MATCH 时回退旧 HcclBarrier，不崩溃
7. AICPU 引擎未开启时（`ShouldUseInnerOp` 返回 true），回退旧 HcclBarrier（预期行为）

## 风险分析

### [已解决] 通信原语签名差异

**问题**：DPU 版 `dpu_alg_data_trans_wrapper.h` 的 `SendRecvWrite(info)` 不需要 thread 参数；AICPU 版 `alg_data_trans_wrapper.h` 的 `SendRecvWrite(info, thread)` 必须传 thread 参数。

**解决**：`RunNHRBarrier` 签名增加 `const ThreadHandle &thread` 参数，`KernelRun` 传入 `templateResource.threads[0]`（主线程）。所有 `SendRecvWrite`/`SendWrite`/`RecvWrite` 调用都加 thread 参数。

### [已解决] CalcChannelRequestNhr 的 AICPU_COMPILE 保护

**问题**：`CalcChannelRequestNhr` 有 `#ifndef AICPU_COMPILE` 保护，AICPU 编译时是空函数。

**分析**：不是问题。`CalcRes` 在 Host 侧执行（hccl 库），实际创建 channel。AICPU 侧（scatter_aicpu_kernel）`CalcRes` 是空跳过，channel 通过 `GetAlgResWithEngine` 序列化传入 `resCtx`，`Orchestrate` 反序列化后从 `remoteRankToChannelInfo_` 取出。

### [低风险] 单级 NHR 的 TopoMatch1D 兼容性

**问题**：`InsTempBarrierNhrAicpu` 用于单级 CLOS/每server出1卡时，配合 `TopoMatch1D` + `SoleExecutor`。`TopoMatch1D::MatchTopo` 产出扁平的 `algHierarchyInfo.infos[0]`（全 rank 列表），`SoleExecutor::CalcRes` 把它传给 `InsTempBarrierNhrAicpu::CalcRes`，后者调 `CalcChannelRequestNhr`。

**分析**：AllGather 的纯 NHR（`InsAllGatherNHR`）也是 `TopoMatch1D` + `SoleExecutor` + `CalcChannelRequestNhr`，已验证可行。Barrier 复用相同模式，风险低。但 AllGather NHR 的 `CalcRes` 有 CLOS 特殊处理（`CalcChannelRequestNHRWithPriorityTopo`），Barrier 直接用 `CalcChannelRequestNhr`，CLOS 场景可能需要类似处理。

**缓解**：首期可以先不支持 CLOS 单级场景（selector 中 CLOS 返回 NOT_MATCH 回退），验证后再开启。

### [低风险] SoleExecutor 线程分配

**问题**：`InsTempBarrierNhrAicpu::CalcRes` 设 `slaveThreadNum = 0`，SoleExecutor 的 `Orchestrate` 中 `templateResource.threads = resCtx.threads`。需确认 `resCtx.threads` 至少包含主线程。

**分析**：`HcclExecOp` 中 AICPU 引擎会通过 `HcclThreadAcquireWithStream` 获取主线程，`resCtx.threads[0]` 是主线程。`slaveThreadNum = 0` 时只有主线程，`threads.size() >= 1` 满足 `KernelRun` 的检查。

## 测试方案

### 测试文件

新增 `test/st/algorithm/testcase/barrier_aicpu_testcase.cc`，参考 `all_gather_aicpu_testcase.cc` 结构。

Barrier 无数据搬运，不需要校验数据正确性。当前 checker 不支持 barrier（没有 `CheckBarrier` 函数，且 checker 的内存/语义校验都是基于数据 buffer 的，barrier 无数据可校验）。只需验证：
1. 所有 rank 执行完成不 hang（同步语义正确）
2. 不回退到旧 AllReduce 流程（日志中无 `BarrierFallbackToOldFlow`）
3. 从流检查不报错（`CheckSlaveTaskQueue` 的首尾 task 类型检查）

### 测试框架

```cpp
#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "acl/acl_rt.h"
#include "alg_env_config.h"
#include <thread>

using namespace HcclSim;

class ST_BARRIER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

void RunBarrierAicpuA5(const TopoMeta &topoInfo)
{
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_ENABLE_OPEN_AICPU", "1", 1);

    auto rankSize = AnalyseRankSize(topoInfo);
    std::vector<std::thread> threads;
    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));
            CHK_RET(HcclBarrier(comm, stream));
            CHK_RET(HcclCommDestroy(comm));
        });
    }
    for (auto &thread : threads) { thread.join(); }

    // barrier 无数据校验，只需确认不 hang + 从流检查通过
    SimWorld::Global()->Deinit();
}
```

### checker 兼容性说明

当前 checker 不支持 barrier：
- 没有 `CheckBarrier` 函数
- `CheckSlaveTaskQueue` 检查从流首 task 必须是 `LOCAL_WAIT_FROM`、尾 task 必须是 `LOCAL_POST_TO`
- barrier 的空 slice `SendRecvWrite` 产生的 task 类型可能是 `WRITE` 而非 `LOCAL_POST_TO`，会触发从流检查报错

**处理方式**：barrier ST 测试不调用 `CheckSlaveTaskQueue` / `GenAndCheckGraph`，仅验证不 hang + 日志确认走新流程。如果从流检查报错，需分析是否需要调整 checker 或跳过从流检查。

### 测试用例

#### 单级 Mesh1D（`InsBarrierMesh1D`）

| 用例名 | 拓扑 | rank 数 | 说明 |
|--------|------|---------|------|
| `st_barrier_a5_aicpu_mesh_1d_1rank` | `{{{0}}}` | 1 | 单卡，直接返回 |
| `st_barrier_a5_aicpu_mesh_1d_4rank` | `{{{0, 1, 2, 3}}}` | 4 | 常规 Mesh1D |
| `st_barrier_a5_aicpu_mesh_1d_8rank` | `{{{0, 1, 2, 3, 4, 5, 6, 7}}}` | 8 | 满配 Mesh1D |

#### 单级 NHR（`InsBarrierNhrAicpu`，每server出1卡）

| 用例名 | 拓扑 | rank 数 | 说明 |
|--------|------|---------|------|
| `st_barrier_a5_aicpu_nhr_4rank` | `{{{0}, {0}, {0}, {0}}}` | 4 | 4 server 各 1 卡 |

#### 两级 Mesh1D+NHR AICPU（`InsBarrierMesh1DNhrAicpu`，多server多卡）

| 用例名 | 拓扑 | rank 数 | 说明 |
|--------|------|---------|------|
| `st_barrier_a5_aicpu_meshnhr_2x4rank` | `{{{0, 1, 2, 3}, {0, 1, 2, 3}}}` | 8 | 2 server 各 4 卡 |
| `st_barrier_a5_aicpu_meshnhr_3x3rank` | `{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}}` | 9 | 3 server 各 3 卡（非2幂） |

#### 回退场景验证

| 用例名 | 拓扑 | 环境变量 | 预期 |
|--------|------|---------|------|
| `st_barrier_a5_aicpu_not_enabled_fallback` | `{{{0, 1}}}` | 不设 `HCCL_ENABLE_OPEN_AICPU` | 回退旧 HcclBarrier，不报错 |

### CMakeLists 修改

`test/st/algorithm/testcase/CMakeLists.txt` 加：

```cmake
set(src_list
    ...
    barrier_aicpu_testcase.cc   # 新增
)
```

### 验证点

| 验证项 | 方法 |
|--------|------|
| 不 hang | 所有 rank 线程 join 完成 |
| 走新 AICPU 流程 | 日志含 `SelectAicpuAlgo` + `InsBarrierMesh1DNhrAicpu` 或 `InsBarrierMesh1D` 或 `InsBarrierNhrAicpu` |
| 不回退旧流程 | 日志不含 `BarrierFallbackToOldFlow`（除未开启 AICPU 的回退用例） |
| 纯 signal 同步 | 日志含 `RunNHRBarrier` / `RunBarrierMesh`，不含数据搬运 |
| 从流检查 | 如调用 checker，关注 `CheckSlaveTaskQueue` 是否报错；如报错则跳过 checker 仅验证不 hang |

## 实现方案

### 1. 新增 `ins_temp_barrier_nhr_aicpu.h`

```cpp
#ifndef INS_TEMP_BARRIER_NHR_AICPU_H
#define INS_TEMP_BARRIER_NHR_AICPU_H

#pragma once

#include "alg_v2_template_base.h"

namespace ops_hccl {

// Barrier 框间 AICPU 模板：按 NHR 步数完成全员同步，不搬运数据。
// 与 DPU 版的区别：直接在 AICPU 上执行 RunNHRBarrier，不通过 DPU 请求-响应封装。
class InsTempBarrierNhrAicpu : public InsAlgTemplateBase {
public:
    InsTempBarrierNhrAicpu() = default;
    InsTempBarrierNhrAicpu(const OpParam &param, const u32 rankId,
                           const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempBarrierNhrAicpu() override = default;

    std::string Describe() const override
    {
        return "Template of Barrier NHR AICPU with tempRankSize " + std::to_string(templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;

protected:
    u32 GetRankFromMap(const uint32_t rankIdx) const;
    HcclResult RunNHRBarrier(const std::map<u32, std::vector<ChannelInfo>> &channels,
                             const ThreadHandle &thread);
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override {}
};

}  // namespace ops_hccl

#endif  // INS_TEMP_BARRIER_NHR_AICPU_H
```

与 DPU 版头文件对比：
- 去掉 `DPUKernelRun` 虚函数重写（不走 DPU）
- `RunNHRBarrier` 去掉 `const` 修饰（AICPU 版可能需要修改成员变量）
- 其余接口完全对称

### 2. 新增 `ins_temp_barrier_nhr_aicpu.cc`

```cpp
#include "ins_temp_barrier_nhr_aicpu.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "template_utils.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {

InsTempBarrierNhrAicpu::InsTempBarrierNhrAicpu(const OpParam &param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks) {}

HcclResult InsTempBarrierNhrAicpu::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    // 与 DPU 版一致：0 线程，0 notify，仅申请框间 NHR channel
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    return HCCL_SUCCESS;
}

u64 InsTempBarrierNhrAicpu::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    return 0;  // barrier 无数据搬运
}

HcclResult InsTempBarrierNhrAicpu::KernelRun(const OpParam &param,
    const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    (void)tempAlgParams;
    HCCL_INFO("[InsTempBarrierNhrAicpu] Run Start, rank[%u] rankSize[%u]", myRank_, templateRankSize_);

    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempBarrierNhrAicpu] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 直接在 AICPU 上执行 NHR barrier，不走 DPU 请求-响应
    // AICPU 通信原语需要传 thread 参数（与 DPU 版不同）
    CHK_RET(RunNHRBarrier(templateResource.channels, templateResource.threads[0]));

    HCCL_INFO("[InsTempBarrierNhrAicpu] Run End");
    return HCCL_SUCCESS;
}

u32 InsTempBarrierNhrAicpu::GetRankFromMap(const uint32_t rankIdx) const
{
    return subCommRanks_[0].at(rankIdx);
}

HcclResult InsTempBarrierNhrAicpu::RunNHRBarrier(
    const std::map<u32, std::vector<ChannelInfo>> &channels, const ThreadHandle &thread)
{
    // 复用 DPU 版 RunNHRBarrier 的 NHR step 逻辑，但使用 AICPU 通信原语（需传 thread 参数）
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));

    for (uint32_t step = 0; step < nSteps; ++step) {
        uint32_t deltaRank = 1u << (nSteps - 1 - step);
        uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
        uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

        auto rxChannel = channels.at(GetRankFromMap(recvFrom));
        auto txChannel = channels.at(GetRankFromMap(sendTo));

        std::vector<DataSlice> emptySlices;
        if (txChannel[0].remoteRank == rxChannel[0].remoteRank) {
            TxRxChannels sendRecvChannels(txChannel[0], rxChannel[0]);
            TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {emptySlices, emptySlices});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] SendRecvWrite failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else if (txChannel[0].remoteRank < rxChannel[0].remoteRank) {
            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] Send failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] Recv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] Recv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] Send failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempBarrierNhrAicpu", InsTempBarrierNhrAicpu);

}  // namespace ops_hccl
```

与 DPU 版实现对比：

| 维度 | DPU 版 | AICPU 版 |
|------|--------|---------|
| KernelRun | `HcommBatchModeEnd` → `HcommSendRequest` → `HcommWaitResponse` → `HcommBatchModeStart` | 直接调 `RunNHRBarrier(channels, thread)` |
| RunNHRBarrier | `#ifndef AICPU_COMPILE` 保护（仅 DPU 侧编译） | 无保护，AICPU 直接编译执行 |
| 通信原语 | `dpu_alg_data_trans_wrapper.h`：`SendRecvWrite(info)` 无需 thread | `alg_data_trans_wrapper.h`：`SendRecvWrite(info, thread)` **必须传 thread** |
| CalcRes | 相同（`CalcChannelRequestNhr` 在 Host 侧执行，AICPU 侧 `#ifndef AICPU_COMPILE` 空跳过，channel 通过序列化传入） | 相同 |
| DPUKernelRun | 有（DPU 侧回调） | 无（不需要） |

**与 AllGather NHR 的关键区别**：

AllGather 的 `InsTempAllGatherNHR` 是独立的 AICPU 模板，有自己的 `GetRes` 计算线程（`slaveThreadNum = threadNum - 1`），`KernelRun` 中用 `threads[channelIdx]` 传给 `SendRecvRead`。

Barrier 的 `InsTempBarrierNhrAicpu` 复用 DPU 版结构，`CalcRes` 设 `slaveThreadNum = 0`（无子线程），`KernelRun` 用 `templateResource.threads[0]`（主线程）传给 `SendRecvWrite`。NHR 的 log(M) 步在主线程上串行执行，每步调一次 `SendRecvWrite`/`SendWrite`/`RecvWrite`。

### 3. 修改 `template/aicpu/CMakeLists.txt`

```cmake
set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_barrier_mesh_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_barrier_nhr_dpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_barrier_nhr_aicpu.cc   # 新增
)
```

### 4. 新增 `executor/ins_v2_barrier_sole_executor.h` 和 `.cc`

单级拓扑的算法注册放在单独的 sole executor 文件中（与 AllGather 的 `ins_v2_all_gather_sole_executor.cc` 结构对称）。

**`ins_v2_barrier_sole_executor.h`**：

```cpp
#ifndef INS_V2_BARRIER_SOLE_EXECUTOR_H
#define INS_V2_BARRIER_SOLE_EXECUTOR_H

#include "executor_common_ops.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2BarrierSoleExecutor : public InsCollAlgBase {
public:
    InsV2BarrierSoleExecutor() {}
    ~InsV2BarrierSoleExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

private:
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};

}  // namespace ops_hccl

#endif  // INS_V2_BARRIER_SOLE_EXECUTOR_H
```

**`ins_v2_barrier_sole_executor.cc`**：

```cpp
#include "ins_v2_barrier_sole_executor.h"
#include "topo_match_1d.h"
#include "ins_temp_barrier_mesh_1D.h"
#include "ins_temp_barrier_nhr_aicpu.h"
#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    InsAlgTemplate tempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    CHK_RET(tempAlg.CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    TemplateDataParams tempDataParams{};
    tempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempDataParams.repeatNum = 1;

    InsAlgTemplate tempAlg(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    TemplateResource templateResource;
    templateResource.channels = remoteRankToChannelInfo_[0];
    templateResource.threads = resCtx.threads;

    CHK_RET(tempAlg.KernelRun(param, tempDataParams, templateResource));
    return HCCL_SUCCESS;
}

// 单级 Mesh1D 注册
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BARRIER,
                 InsBarrierMesh1D,
                 InsV2BarrierSoleExecutor,
                 TopoMatch1D,
                 InsTempBarrierMesh1D);

// 单级 NHR 注册（CLOS拓扑/每server出1卡）
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BARRIER,
                 InsBarrierNhrAicpu,
                 InsV2BarrierSoleExecutor,
                 TopoMatch1D,
                 InsTempBarrierNhrAicpu);

}  // namespace ops_hccl
```

### 5. 修改 `executor/ins_v2_barrier_sequence_executor.cc`

仅新增两级 AICPU 注册（单级注册在 sole executor 中）：

```cpp
// 已有：DPU 版（两级）
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_BARRIER,
                               InsBarrierMeshNhrDPU,
                               InsV2BarrierSequenceExecutor,
                               TopoMatchMultilevel,
                               InsTempBarrierMesh1D,
                               InsTempBarrierNHRDPU);

// 新增：AICPU 版（两级，多级拓扑）
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_BARRIER,
                               InsBarrierMesh1DNhrAicpu,
                               InsV2BarrierSequenceExecutor,
                               TopoMatchMultilevel,
                               InsTempBarrierMesh1D,
                               InsTempBarrierNhrAicpu);
```

同时在文件头部新增 include：

```cpp
#include "ins_temp_barrier_nhr_aicpu.h"
```

### 6. 修改 `selector/barrier_auto_selector.h`

新增 `SelectAicpuAlgo` 声明：

```cpp
class BarrierAutoSelector : public AutoSelectorBase {
private:
    SelectorStatus SelectDPUAlgo(...) const override;      // 已有
    SelectorStatus SelectAicpuAlgo(...) const override;    // 新增
};
```

### 7. 修改 `selector/barrier_auto_selector.cc`

新增 `SelectAicpuAlgo` 实现，参考 AllGather selector 的分流逻辑：

```cpp
SelectorStatus BarrierAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap;
    HCCL_INFO("[BarrierAutoSelector][SelectAicpuAlgo] start, topoLevelNums[%u], level0Topo[%u], "
        "Level1Nhr[%d], localNetInsSizeOfLayer[0][%u]",
        topoInfo->topoLevelNums, topoInfo->level0Topo,
        topoInfo->Level1Nhr, topoInfo->netLayerDetails.localNetInsSizeOfLayer[0]);

    if (topoInfo->topoLevelNums > 1) {
        // 多级拓扑
        if (topoInfo->Level1Nhr || topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            // 每个server出1卡或Layer1为NHR拓扑：无需框内Mesh，纯NHR
            selectAlgName = "InsBarrierNhrAicpu";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            // 正常多级：框内Mesh1D + 框间NHR
            selectAlgName = "InsBarrierMesh1DNhrAicpu";
        } else {
            HCCL_ERROR("[BarrierAutoSelector][SelectAicpuAlgo] multi-level topo not match, "
                "level0Topo[%u]", topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        // 单级拓扑
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsBarrierMesh1D";
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            // CLOS拓扑走NHR
            selectAlgName = "InsBarrierNhrAicpu";
        } else {
            HCCL_ERROR("[BarrierAutoSelector][SelectAicpuAlgo] topo not match, level0Topo[%u]",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_INFO("[BarrierAutoSelector][SelectAicpuAlgo] Algo match[%s]", selectAlgName.c_str());
    return SelectorStatus::MATCH;
}
```

### 算法选择路由

| 场景 | 条件 | 算法 | executor |
|------|------|------|---------|
| 单级 Mesh1D | `topoLevelNums==1 && level0Topo==MESH_1D` | `InsBarrierMesh1D` | SoleExecutor + TopoMatch1D |
| 单级 CLOS | `topoLevelNums==1 && level0Topo==CLOS` | `InsBarrierNhrAicpu` | SoleExecutor + TopoMatch1D |
| 多级，每server出1卡 | `localNetInsSizeOfLayer[0]==1` | `InsBarrierNhrAicpu` | SoleExecutor + TopoMatch1D |
| 多级，Layer1为NHR | `Level1Nhr==true` | `InsBarrierNhrAicpu` | SoleExecutor + TopoMatch1D |
| 多级，正常 | 其他 | `InsBarrierMesh1DNhrAicpu` | SequenceExecutor + TopoMatchMultilevel |

**纯 NHR 场景说明**：每个 server 出 1 卡时，layer0 只有 1 个 rank，框内无需同步（Mesh1D 直接返回），只需框间 NHR。此时用单级 NHR（`SoleExecutor` + `TopoMatch1D`）比两级（`SequenceExecutor` + `TopoMatchMultilevel`）更简洁，避免不必要的两级编排开销。

### 8. 修改 `barrier_op.cc`

将 `IsBarrierHostDpu` 拦截改为引擎白名单：

```cpp
HcclResult BarrierOutPlace(HcclComm comm, aclrtStream stream, const std::string &tag)
{
    // ... 前面不变 ...

    CHK_RET(HcclGetOpExpansionMode(comm, param));

    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, barrier returns directly", __func__);
        return HCCL_SUCCESS;
    }

    // 改前：仅 HostDPU 走新流程
    // if (!IsBarrierHostDpu(comm)) {
    //     return BarrierFallbackToOldFlow(comm, stream);
    // }

    // 改后：引擎白名单，HostDPU 和 AICPU 走新流程，其余回退
    bool isHostDpu = IsBarrierHostDpu(comm);
    bool isAicpu = (param.opExecuteConfig == OpExecuteConfig::AICPU_TS);
    if (!isHostDpu && !isAicpu) {
        HCCL_INFO("[BarrierOutPlace] engine not supported, fallback to legacy HcclBarrier");
        return BarrierFallbackToOldFlow(comm, stream);
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    SelectorStatus selRet = Selector(comm, param, topoInfo, algName);
    if (selRet != SelectorStatus::MATCH) {
        HCCL_INFO("[BarrierOutPlace] selector not matched, fallback to legacy HcclBarrier");
        return BarrierFallbackToOldFlow(comm, stream);
    }
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return BarrierFallbackToOldFlow(comm, stream);
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, ResPackGraphMode()));
    HCCL_INFO("Execute BarrierOutPlace success.");
    return HCCL_SUCCESS;
}
```

### 改动量汇总

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `template/aicpu/ins_temp_barrier_nhr_aicpu.h` | 新增 | ~40 行 |
| `template/aicpu/ins_temp_barrier_nhr_aicpu.cc` | 新增 | ~100 行 |
| `template/aicpu/CMakeLists.txt` | 修改 | +1 行（加 ins_temp_barrier_nhr_aicpu.cc） |
| `executor/ins_v2_barrier_sole_executor.h` | 新增 | ~35 行 |
| `executor/ins_v2_barrier_sole_executor.cc` | 新增 | ~60 行（含 2 个单级注册） |
| `executor/CMakeLists.txt` | 修改 | +1 行（加 ins_v2_barrier_sole_executor.cc） |
| `executor/ins_v2_barrier_sequence_executor.cc` | 修改 | +8 行（include + 两级注册） |
| `selector/barrier_auto_selector.h` | 修改 | +3 行 |
| `selector/barrier_auto_selector.cc` | 修改 | ~25 行（按 topoLevelNums/Level1Nhr/localNetInsSize 分流） |
| `barrier_op.cc` | 修改 | ~8 行（白名单替换） |
| `src/scatter_aicpu_kernel.cmake` | 修改 | +2 行（加 nhr_aicpu.cc + sole_executor.cc 到 NOT COMP_850 块） |
| **总计** | | **~282 行** |

## Barrier AICPU 全流程调用链

### 入口层：`barrier_op.cc`

```
HcclBarrier(comm, stream)                          // 用户调用入口
  ├─ 版本/芯片检查（< 9.0 或非 950 → 回退旧 HcclBarrier）
  ├─ BarrierInitAndCheck(comm, stream, opTag)       // 参数校验 + 生成 tag
  ├─ BarrierEntryLog(stream, opTag, "HcclBarrier")  // 入口日志
  ├─ BarrierOutPlace(comm, stream, opTag)           // 核心执行 ← 见下
  └─ LogHcclExit("HcclBarrier", opTag, startut)     // 退出日志
```

### 核心层：`BarrierOutPlace()`

```
BarrierOutPlace(comm, stream, tag)
  ├─ 构造 OpParam（opType=HCCL_CMD_BARRIER, inputPtr=nullptr, count=0）
  ├─ HcclGetOpExpansionMode(comm, param)            // 设置 param.opExecuteConfig
  ├─ rankSize==1 → 直接返回
  │
  ├─ 【引擎白名单判断】（新增改动点）
  │   isHostDpu = IsBarrierHostDpu(comm)            // 拓扑检查：是否 HostDPU-only
  │   isAicpu = (param.opExecuteConfig == AICPU_TS) // 引擎检查：是否 AICPU
  │   if (!isHostDpu && !isAicpu)
  │       └─ BarrierFallbackToOldFlow()             // CCU/AIV/其他 → 回退
  │
  ├─ Selector(comm, param, topoInfo, algName)       // 算法选择 ← 见下
  │   if (selRet != MATCH)
  │       └─ BarrierFallbackToOldFlow()             // 未匹配 → 回退
  │
  ├─ ShouldUseInnerOp() && OPBASE → 回退
  │
  └─ HcclExecOp(comm, param, topoInfo, algName, ResPackGraphMode())  // 执行 ← 见下
```

### 算法选择层：`Selector()` → `AutoSelectorBase::Select()`

```
Selector(comm, param, topoInfo, algName)            // op_common.cc
  ├─ HcclCalcTopoInfo(comm, param, topoInfo)        // 计算拓扑信息
  ├─ CheckAsymmetricTopoSupport(param.opType, topoInfo)  // 非对称检查
  ├─ ExecuteSelector::Run(param, topoInfo, algName)
  │   └─ AutoSelectorBase::Select(param, topoInfo, algName)
  │       │
  │       ├─ CheckHostDPUOnly() == true?
  │       │   └─ YES → SelectDPUAlgo() → "InsBarrierMeshNhrDPU"   // HostDPU 路径
  │       │
  │       ├─ CCU_MS → SelectCcuMsAlgo() → NOT_MATCH（barrier 未实现）
  │       ├─ CCU_SCHED → SelectCcuScheduleAlgo() → NOT_MATCH
  │       ├─ AIV → ProcessAivConfig() → NOT_MATCH
  │       │
  │       └─ IsStarsState(AICPU_TS)?
  │           └─ YES → SelectAicpuAlgo()                            // AICPU 路径（新增）
  │                     └─ selectAlgName = "InsBarrierMesh1DNhrAicpu"
  │                     └─ return MATCH
  │
  ├─ SetCommEngine(param)                            // 设置 param.engine = COMM_ENGINE_AICPU_TS
  ├─ LoadAICPUKernel()                               // 加载 AICPU kernel
  ├─ SetOpParamAlgTag(param, algName)
  └─ SetExecTimeout(param)
```

### 执行层：`HcclExecOp()` → executor → template

```
HcclExecOp(comm, param, topoInfo, algName, resPack)  // op_common.cc
  │
  ├─ CollAlgExecRegistryV2::GetAlgExec(BARRIER, "InsBarrierMesh1DNhrAicpu")
  │   └─ 返回 InsV2BarrierSequenceExecutor<TopoMatchMultilevel,
  │         InsTempBarrierMesh1D, InsTempBarrierNhrAicpu> 实例
  │
  ├─ HcclGetAlgRes(comm, param, executor, topoInfo, ...)  // 资源计算
  │   ├─ executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo)
  │   │   └─ TopoMatchMultilevel::MatchTopo(comm, topoInfo, algHierarchyInfo)
  │   │       ├─ 获取 layerNum、instSizeList
  │   │       ├─ isSymmetric = CheckVecElementAllSame(instSizeList)
  │   │       ├─ TopoForLayer0() → algHierarchyInfo.infos[0]  // 框内 rank 列表
  │   │       ├─ TopoForLayer1() → algHierarchyInfo.infos[1]  // 框间 rank 列表
  │   │       └─ (3级时) TopoForLayer2() → algHierarchyInfo.infos[2]
  │   │
  │   ├─ executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest)
  │   │   ├─ InsTempBarrierMesh1D::CalcRes()   → 框内: N_intra-1 线程, N_intra-1 channel
  │   │   └─ InsTempBarrierNhrAicpu::CalcRes() → 框间: 0 线程, log(M) channel
  │   │   └─ 合并: slaveThreadNum = max(N_intra-1, 0), channels = {框内, 框间}
  │   │
  │   └─ GetAlgResWithEngine()  → 分配线程/channel/notify，序列化 resCtx
  │
  ├─ HcclAicpuKernelEntranceLaunch(comm, param, ...)  // AICPU kernel 下发
  │   ├─ HcommThreadNotifyRecordOnThread()  // Host 通知 Device 主线程
  │   ├─ AicpuKernelLaunch(comm, param, unfoldThread)  // 下发到 AICPU
  │   │   └─ executor->Orchestrate(param, resCtx)      // ← 编排执行
  │   └─ HcclThreadNotifyWaitOnThreadDefault()  // Host 等待 Device 完成
  │
  └─ (完成)
```

### 编排层：`InsV2BarrierSequenceExecutor::Orchestrate()`

```
Orchestrate(param, resCtx)
  │
  ├─ 构造框间模板: InsTempBarrierNhrAicpu(param, myRank, algHierarchyInfo.infos[1])
  ├─ 构造框内模板: InsTempBarrierMesh1D(param, myRank, algHierarchyInfo.infos[0])
  │
  ├─ 分配资源:
  │   templateResourceInter.channels = remoteRankToChannelInfo_[1]  // 框间 channel
  │   templateResourceInter.threads = resCtx.threads
  │   templateResourceIntra.channels = remoteRankToChannelInfo_[0]  // 框内 channel
  │   templateResourceIntra.threads = resCtx.threads
  │
  ├─ 执行框间（先）:
  │   interTempAlg.KernelRun(param, interTempDataParams, templateResourceInter)
  │   └─ InsTempBarrierNhrAicpu::KernelRun()
  │       └─ RunNHRBarrier(channels)
  │           └─ for step in 0..log(M):
  │               ├─ 计算 deltaRank, recvFrom, sendTo
  │               └─ 空slice SendRecvWrite / SendWrite + RecvWrite  // 纯 signal 同步
  │
  └─ 执行框内（后）:
      intraTempAlg.KernelRun(param, intraTempDataParams, templateResourceIntra)
      └─ InsTempBarrierMesh1D::KernelRun()
          ├─ PreSyncInterThreads(主线程 → N_intra-1 子线程)
          ├─ RunBarrierMesh(threads, channels)
          │   └─ for each connectedRank:
          │       └─ 空slice SendRecvWrite(threads[i])  // 分发到子线程，并行执行
          └─ PostSyncInterThreads(子线程 → 主线程)  // 等待全部完成
```

### 完整调用链一览

```
用户调用
  HcclBarrier
    └─ BarrierOutPlace
        ├─ 引擎白名单 (HostDPU / AICPU)
        ├─ Selector
        │   └─ AutoSelectorBase::Select
        │       └─ SelectAicpuAlgo → "InsBarrierMesh1DNhrAicpu"
        └─ HcclExecOp
            ├─ HcclGetAlgRes
            │   ├─ CalcAlgHierarchyInfo (TopoMatchMultilevel::MatchTopo)
            │   └─ CalcRes (Mesh1D + NhrAicpu)
            └─ HcclAicpuKernelEntranceLaunch
                └─ AicpuKernelLaunch
                    └─ executor->Orchestrate
                        ├─ NhrAicpu::KernelRun (框间, log(M)步)
                        │   └─ RunNHRBarrier (空slice signal同步)
                        └─ Mesh1D::KernelRun (框内, 1步并行)
                            ├─ PreSync → RunBarrierMesh → PostSync
                            └─ (空slice signal同步)
```

## 架构分析

### 整体架构图

```
┌─────────────────────────────────────────────────────┐
│                   barrier_op.cc                      │
│  HcclBarrier → BarrierOutPlace                      │
│  ├─ 引擎白名单: isHostDpu || isAicpu → 放行          │
│  └─ 其余 → BarrierFallbackToOldFlow                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              AutoSelectorBase::Select()              │
│  ├─ CheckHostDPUOnly=true → SelectDPUAlgo           │  HostDPU 路径(已有)
│  ├─ CCU/AIV → NOT_MATCH                             │  不支持
│  └─ IsStarsState(AICPU_TS) → SelectAicpuAlgo        │  AICPU 路径(新增)
│      ├─ 单级 MESH_1D  → "InsBarrierMesh1D"          │
│      ├─ 单级 CLOS     → "InsBarrierNhrAicpu"        │
│      ├─ 多级每server1卡 → "InsBarrierNhrAicpu"      │
│      └─ 多级正常       → "InsBarrierMesh1DNhrAicpu" │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    HcclExecOp                        │
│  ├─ GetAlgExec → 获取 executor 实例                  │
│  ├─ HcclGetAlgRes → CalcAlgHierarchyInfo + CalcRes  │
│  └─ HcclAicpuKernelEntranceLaunch → Orchestrate     │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌─────────────┐ ┌────────────┐ ┌──────────────┐
│ SoleExecutor │ │ Sequence   │ │ Sequence     │
│ +TopoMatch1D │ │ Executor   │ │ Executor     │
│              │ │ +Multilevel│ │ +Multilevel  │
│ 单级Mesh1D   │ │ 两级AICPU  │ │ 两级DPU(已有)│
│ 或单级NHR    │ │ Mesh1D+NHR │ │ Mesh1D+NHRDPU│
└──────┬───────┘ └─────┬──────┘ └──────┬───────┘
       │                 │               │
       ▼                 ▼               ▼
┌─────────────┐ ┌────────────┐ ┌──────────────┐
│ Mesh1D      │ │ NHR AICPU  │ │ NHR DPU      │
│ 或 NHR AICPU│ │ → Mesh1D   │ │ → Mesh1D     │
│ (空slice)   │ │ (空slice)  │ │ (空slice)    │
└─────────────┘ └────────────┘ └──────────────┘
```

### 三层正交分离

| 维度 | 职责 | 决策依据 |
|------|------|---------|
| 引擎白名单（barrier_op.cc） | 放行/回退 | `IsBarrierHostDpu`（拓扑属性）+ `opExecuteConfig`（引擎配置） |
| 算法选择（Selector） | 选哪个算法名 | `topoLevelNums` + `level0Topo` + `Level1Nhr` + `localNetInsSize` |
| 执行编排（Executor） | 怎么执行 | 算法名决定 executor 类型（Sole/Sequence） |

三层各管各的，互不耦合：
- 白名单不关心具体选哪个算法，只决定是否进入新流程
- Selector 不关心 executor 怎么编排，只返回算法名
- Executor 不关心引擎怎么选的，只按算法名执行

### 两条引擎路径共存

```
HostDPU 路径:  Select() 内部 CheckHostDPUOnly → SelectDPUAlgo → DPU executor
AICPU 路径:    Select() 内部 IsStarsState    → SelectAicpuAlgo → AICPU executor
```

两条路径在 `Select()` 内部分叉，互不干扰。HostDPU 先判断（拓扑覆盖引擎），AICPU 后判断（引擎未覆盖）。

HostDPU 和 AICPU 是两个正交维度：
- AICPU/CCU/AIV 是**设备侧执行引擎**（用户配置选择），负责框内通信
- HostDPU 是**拓扑属性**（物理链路决定），负责框间通信走 DPU 还是设备侧

HostDPU=true 时，框间走 DPU，框内仍由设备引擎执行——所以 `opExecuteConfig` 可以是任意值，`Select()` 内部才覆盖为 `HOSTCPU`。

### 三种算法形态

| 形态 | executor | topo match | 模板 | 适用场景 |
|------|---------|-----------|------|---------|
| 单级 | SoleExecutor | TopoMatch1D | 1个 | 单级拓扑 / 每server1卡 |
| 两级AICPU | SequenceExecutor | TopoMatchMultilevel | 2个(Mesh1D+NHR AICPU) | 多级正常 |
| 两级DPU | SequenceExecutor | TopoMatchMultilevel | 2个(Mesh1D+NHR DPU) | HostDPU |

### 架构优点

1. **与现有 DPU 路径完全对称**：AICPU 两级和 DPU 两级共用 `SequenceExecutor`，只是模板参数不同
2. **渐进式扩展**：新增 AICPU 不改 DPU 路径，新增 CCU/AIV 只需加白名单 + selector
3. **白名单防 CCU fallthrough**：在 `barrier_op.cc` 提前拦截，不依赖 `Select()` 内部行为
4. **selector 按拓扑分流**：单级/多级/每server1卡 自动选不同算法，用户无感

### 架构风险

#### [中] 单级 NHR 的 CalcChannelRequestNhr 兼容性

`InsTempBarrierNhrAicpu` 的 `CalcRes` 调 `CalcChannelRequestNhr`，该函数内部 `if (netLayerNum > 1 && netLayer == 0) continue` 会跳过 layer0 链路。单级场景用 `TopoMatch1D`，`netLayerNum == 1`，不会跳过，行为正确。AllGather 的纯 NHR（`InsAllGatherNHR`）也是同样调用，已验证可行。但 AllGather NHR 的 `CalcRes` 对 CLOS 有特殊处理（`CalcChannelRequestNHRWithPriorityTopo`），Barrier 直接用 `CalcChannelRequestNhr`，CLOS 场景可能需要类似处理。

**缓解**：首期 CLOS 单级场景可以先返回 NOT_MATCH 回退，验证后再开启。

#### [低] Selector 未处理 MESH_1D_CLOS

AllGather selector 对 `MESH_1D_CLOS`（UBX 机型）有专门处理。Barrier selector 只处理 `MESH_1D` 和 `CLOS`，未处理 `MESH_1D_CLOS`，会返回 `NOT_MATCH` 回退。这是预期行为（首期不支持 UBX）。

#### [低] ShouldUseInnerOp 二次回退

白名单放行后，`ShouldUseInnerOp(param.opExecuteConfig) && OPBASE` 还会检查 AICPU 是否开启。如果用户未设 `HCCL_ENABLE_OPEN_AICPU`，会二次回退。这与白名单的 `isAicpu` 判断不冲突——`isAicpu` 检查引擎类型（`AICPU_TS`），`ShouldUseInnerOp` 检查引擎开关（`HcclCheckAicpuEnableOpen`），是不同维度。

#### [信息] 图模式不支持

当前 barrier 不支持图模式：
- 没有 `HcclBarrierGraphMode` 函数（AllGather/AllReduce/Broadcast/ReduceScatter 都有）
- `barrier_op.cc` 中 `param.opMode` 硬编码为 `OpMode::OPBASE`
- 本次 AICPU 改动只涉及 `OPBASE` 路径，不新增图模式入口
- 图模式支持需要单独的 `HcclBarrierGraphMode` 函数 + `OpMode::OFFLOAD` 路径，属于后续工作
