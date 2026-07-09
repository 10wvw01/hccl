---
date: 2026-07-01
topic: broadcast-3-level-sequence
---

# Broadcast 3 级 Sequence Executor (Scatter + AllGather)

## Summary

为 broadcast 操作新增 3 级 sequence executor，支持 Scatter(level0) → Scatter(level1) → Scatter(level2) → AllGather(level2) → AllGather(level1) → AllGather(level0) 的线性串行数据流，使 3 层拓扑场景下跨层级链路带宽得到充分利用，避免 Squeeze2D 压缩后框内 mesh 组的高带宽链路被合并浪费。

---

## 1. 问题背景

### 1.1 现状分析

当前 broadcast 在 3 级拓扑下使用 parallel executor（Scatter+AllGather 并行），通过 `TopoMatchSqueeze2D` 将 3 级压缩为 2 级并行执行。压缩后 layer0 被合并为整个 server 的扁平组，框内 mesh 组的高带宽链路（level0 有 7+8 端口）未充分利用。同时 parallel executor 需要 4 个模板参数 + 复杂的数据切分和线程分配，维护成本高。UBOE 场景下 parallel executor 是合理的，但非 UBOE 的三级拓扑缺少串行 3 级选项。

当前 `BroadcastAutoSelector::SelectAicpuAlgo` 三级拓扑分支：

```cpp
if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
    if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[1] == 1) {
        selectAlgName = "InsBroadcastNHR";              // 每 server 出 1 卡 → 扁平 NHR
    } else {
        selectAlgName = "InsBroadcastParallelNHRNHRUboe"; // 每 server 出多卡 → Squeeze2D 压缩
    }
}
```

| 三级拓扑条件 | 选择的算法 | TopoMatch | 是否三级 | 问题 |
|-------------|-----------|-----------|---------|------|
| 每 server 1 卡 | `InsBroadcastNHR` | `TopoMatch1D` | ❌ 扁平 | 不分三级 |
| 每 server 多卡（不区分 UBOE） | `InsBroadcastParallelNHRNHRUboe` | `TopoMatchSqueeze2D` | ❌ 压缩为两级 | 框内 mesh 带宽浪费，且不区分 UBOE |

**核心问题**：每 server 多卡时，代码不区分 UBOE 和非 UBOE，全部走 `ParallelNHRNHRUboe` + `TopoMatchSqueeze2D`。非 UBOE 的三级拓扑被压缩为两级，框内 mesh 高带宽链路浪费。

### 1.2 修改后的 Selector 逻辑

```cpp
// BroadcastAutoSelector::SelectAicpuAlgo（对标 AllReduce L355/L371 的 level0Topo 门控）
if (topoInfo->topoLevelNums > 1) {
    if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3 && topoInfo->level2Uboe) {
        if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[1] == 1) {
            selectAlgName = "InsBroadcastNHR";
        } else {
            selectAlgName = "InsBroadcastParallelNHRNHRUboe";   // UBOE 三级 → 现有 parallel
        }
    } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
        selectAlgName = "InsBroadcastNHR";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
            selectAlgName = "InsBroadcastSequenceMesh1DNHRNHR"; // MESH_1D 三级 → 三级 sequence（新增）
        } else {
            selectAlgName = "InsBroadcastParallelMesh1DNHR";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        selectAlgName = "InsBroadcastNHR";                      // CLOS → NHR（不走 sequence）
    } else {
        return SelectorStatus::NOT_MATCH;
    }
} else {
    return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
}
```

---

## 2. 方案概览：Scatter 3 级 + AllGather 3 级（6 步串行）

### 2.1 执行流程

```
每个 rank, 每轮 loop:
  Phase 1 (Scatter, 3步串行, 数据分片递减):
    Step 1: Level0 Scatter Mesh1D — root 框内分发数据分片 (repeatNum = 1)
    Step 2: Level1 Scatter NHR    — root 超节点内跨框分发 (repeatNum = 1)
    Step 3: Level2 Scatter NHR    — root 跨超节点分发 (repeatNum = 1)

  Phase 2 (AllGather, 3步串行, 数据分片递增, 反序):
    Step 4: Level2 AllGather NHR  — 跨超节点收集
    Step 5: Level1 AllGather NHR  — 超节点内跨框收集
    Step 6: Level0 AllGather Mesh1D — 框内收集到 OUTPUT
```

### 2.2 选型理由

1. **完全复用现有模板**——Scatter Mesh1D/NHR + AllGather Mesh1D/NHR 都已有，无需新模板
2. **与 AllReduce 3 级对称**——AllReduce = RS(3级) + AG(3级)，Broadcast = Scatter(3级) + AG(3级)
3. **注册宏已有**——`REGISTER_EXEC_V2_MULTI` 支持 6+ 模板参数
4. **TopoMatchMultilevel 已有**——直接复用
5. **参考实现已有**——AllReduce 3 级 executor 可直接参考

### 2.3 性能权衡

6 步串行比 parallel executor 的 4 步（每步并行）步数多，但：
- 每步数据量递减（Scatter）/递增（AllGather），实际延迟不是 6×单步
- 3 级串行充分利用每级链路带宽，不压缩层级
- 对大数据量场景，带宽利用率优势弥补步数劣势
- 对小数据量场景，由 selector 选择 parallel executor

---

## 3. Actors

- A1. **Selector**: 根据拓扑和数据量选择算法，决定何时使用 3 级 sequence executor
- A2. **TopoMatchMultilevel**: 拓扑匹配层，生成 3 层 algHierarchyInfo
- A3. **BroadcastSequenceExecutor3Level**: 3 级序列执行器，编排 Scatter→AllGather 数据流
- A4. **AlgTemplate**: 各级算法模板（Scatter Mesh1D/NHR + AllGather Mesh1D/NHR）

---

## 4. Key Flows

- F1. **3 级 Broadcast 执行流**
  - **Trigger:** Selector 检测到 3 层拓扑且非 UBOE，选择 3 级 sequence algorithm
  - **Actors:** A1, A2, A3, A4
  - **Steps:**
    1. TopoMatchMultilevel 生成 `infos.size() == 3` 的 algHierarchyInfo
    2. CalcRes 计算 6 级资源需求（Scatter×3 + AllGather×3），合并线程/notify/channel
    3. OrchestrateLoop 循环分片：Phase1 Scatter 3 级 → Phase2 AllGather 3 级
    4. Scatter 阶段：INPUT → Mesh(L0) → CCL → NHR(L1) → CCL → NHR(L2) → CCL
    5. AllGather 阶段：CCL → NHR(L2) → CCL → NHR(L1) → CCL → Mesh(L0) → OUTPUT
  - **Outcome:** 所有 rank 的 outputPtr 包含 root 的完整数据

---

## 5. Requirements

### 5.1 Executor 框架

- R1. 新建 `InsV2BroadcastSequenceExecutor3Level` 类，接受 6 个算法模板参数 `<AlgTopoMatch, ScatterT0, ScatterT1, ScatterT2, AllGatherT0, AllGatherT1, AllGatherT2>`
- R2. 数据流为线性串行 6 步：INPUT → Scatter(L0) → CCL → Scatter(L1) → CCL → Scatter(L2) → CCL → AllGather(L2) → CCL → AllGather(L1) → CCL → AllGather(L0) → OUTPUT，全部 `repeatNum = 1`
- R3. 支持 loop 分片循环处理大数据量，每轮循环依次执行 6 步，与现有 3 级 loop 机制一致
- R4. 注册使用 `REGISTER_EXEC_V2_MULTI`，算法名 `InsBroadcastSequenceMesh1DNHRNHR`

### 5.2 资源计算

- R5. CalcRes 为 6 级分别计算 AlgResourceRequest，合并 slaveThreadNum 为 max(所有级)，notifyNumPerThread 为各级 max，channels 拆为 3 层 `channels[0/1/2]`
- R6. CCL Buffer scratchMultiple 计算：取第一步（ScatterL0）的 `CalcScratchMultiple`（=1，`ins_temp_scatter_mesh_1D.cc:86` 无条件返回，不检查 opMode），与 AllReduce 3 级只取 RSL0 一致。`totalRankAlign = N0*N1*N2`，`maxCountPerLoop = cclMem.size / totalMult / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / totalRankAlign * totalRankAlign`（两级对齐与 AllReduce 3 级一致）。不取 6 步 max——单段复用下各步串行执行，最大 CCL 需求为 AGL0 的 `N0*sliceSizeL0 = cclMem.size`，`totalMult=1` 即可保证不溢出。AGL0 用 ZAxisDetour（继承基类 `CalcScratchMultiple`，非 OPBASE 返回 0），但 3 级 executor 从不调 AGL0 的 `CalcScratchMultiple`，opMode 依赖不触发（RSK3 无风险）

### 5.3 模板参数生成

- R7. 全部 6 步 `repeatNum = 1`，与 AllReduce 3 级一致——多级协调由 executor 通过 CCL buffer offset 管理，不依赖模板内部 repeatNum
- R8. 各步的 sliceSize / tailSize / stride 由 executor 根据层级 rank 数动态计算，tailSize 跨级传播与 AllReduce 3 级完全一致（`ins_v2_all_reduce_sequence_executor_aicpu_3level.cc:249-444`）：每级 `sliceSize = cnt / N * dts`，`tailSize = sliceSize + cnt % N * dts`，下级根据 `rankIdxLevelX_ == N-1` 选择用上级的 tailSize（末尾 rank）还是 sliceSize（正常 rank）作为输入。Scatter 阶段递减（÷N0 → ÷N1 → ÷N2），AllGather 阶段用对应 Scatter 步的 sliceSize/tailSize 恢复
- R9. skipLevel1 场景（rankSizeLevel1==1）跳过 L1，sliceSize 不变直接传递

### 5.4 Root 处理

- R10. Broadcast 有 root 概念，Scatter 阶段只有包含 root 的层级组执行，其他组跳过（与 parallel executor 的 `intraLocalRoot_ == root_` 判断一致）
- R11. AllGather 阶段所有组都执行（收集完整数据）
- R12. Scatter 模板已有 `SetRoot` 接口，executor 在构造 Scatter 模板时调用 `SetRoot(localRoot)` 设置每级的本地 root

### 5.5 CCL Buffer 管理

- R13. 采用单段复用模式——所有 6 步共用同一块 CCL Buffer 区域，每步通过 offset 参数指定读写位置，不分区

### 5.6 拓扑匹配

- R14. 复用现有 TopoMatchMultilevel（与 RS/AG/AR 3 级 sequence 一致），生成 `infos.size() == 3`

### 5.7 算法选择

- R15. 在 BroadcastAutoSelector::SelectAicpuAlgo 中新增分支：3 层拓扑且 `level0Topo == MESH_1D` 且非 UBOE（`level0Topo == Level0Shape::MESH_1D && !level2Uboe`）时选择 `InsBroadcastSequenceMesh1DNHRNHR`；UBOE 或 CLOS/MESH_1D_CLOS 拓扑走 parallel executor
- R16. 不设数据量阈值——三级非 UBOE 即选 sequence

---

## 6. Acceptance Examples

- AE1. **Covers R1, R2, R7-R9.** Given 8P×8Pod×2Cluster（128卡 3层拓扑），root=rank0，执行 broadcast，executor 依次执行 Scatter(L0, repeatNum=1) → Scatter(L1, repeatNum=1) → Scatter(L2, repeatNum=1) → AllGather(L2, repeatNum=1) → AllGather(L1, repeatNum=1) → AllGather(L0, repeatNum=1)，所有 rank 的 outputPtr 包含 rank0 的完整数据。

- AE2. **Covers R5.** 6 级资源请求合并后，slaveThreadNum = max(6 级)，channels 包含 3 层。

- AE3. **Covers R10, R11.** root=rank0 在 server0，Scatter Level0 只有 server0 的组执行（包含 root），其他 server 跳过；AllGather 阶段所有组执行。

- AE4. **Covers R15, R16.** Given topoLevelNums==3 且 level0Topo==MESH_1D 且 level2Uboe==false，SelectAicpuAlgo 返回 "InsBroadcastSequenceMesh1DNHRNHR"。Given level2Uboe==true 或 level0Topo==CLOS/MESH_1D_CLOS，选择 parallel executor。

---

## 7. Success Criteria

- 3 级 sequence executor 在 3 层拓扑场景下正确完成 broadcast，所有 rank 输出与 root 输入一致
- 跨超节点链路上传输的数据量仅为 Scatter 后的子集，而非全量数据
- Selector 能正确识别 3 层拓扑且非 UBOE 场景并选择 3 级算法
- 现有 parallel executor 行为不受影响

---

## 8. Scope Boundaries

### In Scope

- 新建 `InsV2BroadcastSequenceExecutor3Level` executor
- 注册 `InsBroadcastSequenceMesh1DNHRNHR`
- Selector 新增 3 级分支（三级非 UBOE 选 sequence）
- 复用现有 Scatter Mesh1D/NHR + AllGather Mesh1D/NHR 模板
- 复用现有 TopoMatchMultilevel（与 RS/AG/AR 3 级 sequence 一致）
- 新增 `broadcast_3level_testcase.cc` ST 测试文件

### Out of Scope

- 不修改现有 parallel executor
- 不新增 Scatter 或 AllGather 模板
- 不支持 4 级及以上拓扑
- 不实现流水线（pipeline）模式
- CCU/AIV 引擎不涉及
- UBOE 场景继续走 parallel executor
- 图模式 `enableRemoteMemAccess` 优化（直接远程写入 `remoteOutputGraphMode`）——3 级 executor 功能正确但未优化，AllReduce 3 级同样未实现

---

## 9. Key Decisions

- **Scatter+AllGather vs 树形广播**: 选择 Scatter+AllGather，因为完全复用现有模板，与 AllReduce 3 级结构对称
- **6 步串行 vs 4 步并行**: 3 级串行作为非 UBOE 场景的默认选择，UBOE 场景继续用 parallel executor
- **TopoMatchMultilevel vs TopoMatch3Level**: 使用 TopoMatchMultilevel，与 RS/AG/AR 3 级 sequence 一致；支持 HostDPU；Layer2 分组用 `baseModSize = L0*L1`（TopoMatch3Level 用 `L1`，分组规则不同）
- **CCL Buffer 单段 vs 双段**: 选择单段复用，简化管理，每步通过 offset 指定位置
- **数据量阈值**: 不设阈值，三级非 UBOE 即选 sequence
- **AGL0 模板选择**: 使用 `InsTempAllGatherMesh1D1DZAxisDetour`（Z 轴绕路子类），与 AllReduce 3 级一致。ZAxisDetour override 了 `KernelRun`/`RunAllGatherMesh`/`LocalDataCopy`，自带正确的 tail 处理（`CalcSliceSizeForChannel` 按 `dmaRead` 分离 tx/rx）且无 CCL 拷贝重叠问题（`LocalDataCopy` 无 CCL 拷贝步骤，tx 直接从 OUTPUT 读取）。基类 `InsTempAllGatherMesh1D` 的 tail bug 修复保留用于其他 executor。

---

## 10. 技术设计细节

### 10.1 完整执行流程：入口 → Selector → Executor

```
用户调用 HcclBroadcast(buf, count, dataType, root, comm, stream)
  └─ BroadcastOutPlace
      └─ Selector
          └─ SelectAicpuAlgo
              └─ topoLevelNums==3 && !level2Uboe
                  → "InsBroadcastSequenceMesh1DNHRNHR"  (新增)
      └─ HcclExecOp
          └─ GetAlgExec → InsV2BroadcastSequenceExecutor3Level
          └─ HcclGetAlgRes
          │   ├─ CalcAlgHierarchyInfo → TopoMatchMultilevel::MatchTopo
          │   │   → infos[0] = 框内 rank 列表
          │   │   → infos[1] = 超节点内同索引 rank
          │   │   → infos[2] = 跨超节点同索引 rank
          │   └─ CalcRes → 6 级资源计算合并
          └─ HcclAicpuKernelEntranceLaunch
              └─ Orchestrate
                  └─ OrchestrateLoop
```

### 10.2 CalcRes：6 级资源计算

```
6 个模板各调 CalcRes:
  ScatterL0 (Mesh1D)  → resReqScatterL0  (框内 channel[0])
  ScatterL1 (NHR)     → resReqScatterL1  (框间 channel[1])
  ScatterL2 (NHR)     → resReqScatterL2  (跨超节点 channel[2])
  AllGatherL2 (NHR)   → resReqAGL2       (跨超节点 channel[2])
  AllGatherL1 (NHR)   → resReqAGL1       (框间 channel[1])
  AllGatherL0 (Mesh1D1D ZAxisDetour) → resReqAGL0      (框内 channel[0])

合并:
  slaveThreadNum = max(6级)
  notifyNumPerThread = max(6级)
  notifyNumOnMainThread = max(6级)
  channels[0] = 框内 channel
  channels[1] = 框间 channel
  channels[2] = 跨超节点 channel
```

### 10.3 OrchestrateLoop：每轮 6 步串行

```
初始化:
  构造 6 个模板实例 (ScatterL0/L1/L2 + AGL0/L1/L2)
  设置 SetRoot(localRoot) 到 Scatter 模板
  分配 TemplateResource (channel + thread)

while (processedDataCount < dataCount_):
    currDataCount = min(remaining, maxCountPerLoop)

    // ====== Phase 1: Scatter 3 级 (数据分片递减) ======

    // Step 1: ScatterL0 — 框内 Mesh1D 分发
    //   input: INPUT buffer, output: CCL buffer (offset=0)
    //   repeatNum = 1
    //   只有 root 所在框执行, 其他框跳过
    algTemplateScatterL0->SetRoot(localRootL0)
    algTemplateScatterL0->KernelRun(...)

    // Step 2: ScatterL1 — 超节点内跨框 NHR 分发
    //   input: CCL buffer (ScatterL0 输出), output: CCL buffer
    //   repeatNum = 1
    //   只有 root 所在超节点执行
    algTemplateScatterL1->SetRoot(localRootL1)
    algTemplateScatterL1->KernelRun(...)

    // Step 3: ScatterL2 — 跨超节点 NHR 分发
    //   input: CCL buffer (ScatterL1 输出), output: CCL buffer
    //   repeatNum = 1
    //   只有 root 所在跨超节点组执行
    algTemplateScatterL2->SetRoot(localRootL2)
    algTemplateScatterL2->KernelRun(...)

    // ====== Phase 2: AllGather 3 级 (数据递增, 反序) ======

    // Step 4: AGL2 — 跨超节点 NHR 收集
    //   repeatNum = 1
    //   所有组执行 (无 root 限制)
    algTemplateAGL2->KernelRun(...)

    // Step 5: AGL1 — 超节点内跨框 NHR 收集
    //   repeatNum = 1
    //   所有组执行
    algTemplateAGL1->KernelRun(...)

    // Step 6: AGL0 — 框内 Mesh1D1D ZAxisDetour 收集到 OUTPUT
    //   input: CCL buffer, output: OUTPUT buffer (param.outputPtr)
    //   repeatNum = 1
    //   所有组执行
    algTemplateAGL0->KernelRun(...)

    processedDataCount += currDataCount
    loop++
```

### 10.4 CCL Buffer 单段复用

所有 6 步共用同一块 CCL Buffer，每步通过 `hcclBuffBaseOff` / `inBuffBaseOff` / `outBuffBaseOff` 指定读写位置，不分区。

> **关键约束**：NHR 模板（ScatterNHR / AllGatherNHR）内部通过 `hcclBuffBaseOff + sliceIdx * sliceSize` 访问 CCL Buffer，需要 `templateRankSize_ * sliceSize` 连续空间。Mesh1D 模板同样通过 `hcclBuffBaseOff` 定位 scratch 区。每步必须正确设置 `hcclBuffBaseOff`，使其指向当前步数据在 CCL Buffer 中的起始位置，与 AllReduce 3 级模式一致。

### 10.5 Root 处理与跳过逻辑

Broadcast 有 root 概念，与 AllReduce 3 级（无 root）关键不同。Broadcast 中只有 root 持有源数据，Scatter 阶段逐级分发数据，每级只有"数据路径"上的组才有数据可 scatter。非数据路径上的组没有数据源，无法执行 Scatter。

> **跳过根本原因**：某级组内没有持有数据的 rank → 该组无有效 root → 跳过 `KernelRun`。若不跳过，ScatterNHR 的 `GetStepInfo` 调用 `GetAlgRank(root_, subCommRanks_[0], rootAlgRank)`（`ins_temp_scatter_nhr.cc:89`），root 不在组内时返回 `HCCL_E_PARA`（`template_utils.cc:17`）但返回值未检查，导致 NHR 向错误 rank 发送或 hang——这是不跳过时的技术后果，而非根本原因。即便修复 `GetAlgRank` 检查，无数据可 scatter 仍然产生错误结果。

```
root = param.root (全局 rank)

root 索引:
  rootIdx0 = root % rankSizeLevel0_
  rootIdx1 = (root / rankSizeLevel0_) % rankSizeLevel1_
  rootIdx2 = root / (rankSizeLevel0_ * rankSizeLevel1_)

本 rank 索引:
  rankIdxLevel0_ = myRank_ % rankSizeLevel0_
  rankIdxLevel1_ = (myRank_ / rankSizeLevel0_) % rankSizeLevel1_
  rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_)

各级 Scatter 的 root（全局 rank）与跳过条件:

  ScatterL0: root = root_ (全局 root)
    跳过条件: rankIdxLevel2_ != rootIdx2 || rankIdxLevel1_ != rootIdx1  (不在 root 的框)
    原理: 只有 root 所在框的 INPUT 有有效数据

  ScatterL1: root = root_ - rootIdx0 + rankIdxLevel0_  (root 框内与本 rank 同 rankIdx0 的 rank)
    跳过条件: rankIdxLevel2_ != rootIdx2  (不在 root 的超节点) 或 skipLevel1_
    原理: 只有 root 超节点内的跨框组才有 root 框的 rank 作为数据源;
          root 公式保证 root 在本 rank 的 level1 组内(同 rankIdx0, root 的 rankIdx1)

  ScatterL2: root = rootIdx2 * (N0*N1) + rankIdxLevel1_ * N0 + rankIdxLevel0_
             (root 超节点内与本 rank 同 rankIdx0/1 的 rank)
    跳过条件: 无 (所有 rank 参与)
    原理: root 超节点的 rank 有数据, 分发到所有超节点;
          root 公式保证 root 始终在本 rank 的 level2 组内(同 rankIdx0/1, root 的 rankIdx2)

AllGather 阶段: 所有组都执行 (无 root, 无跳过)
```

> **root 在组内验证**：ScatterL1 的 root = `root_ - rootIdx0 + rankIdxLevel0_`。root 的 rankIdx0 = rootIdx0，所以 root 的 rankIdx0 = rootIdx0。本 rank 的 level1 组包含 rankIdx0 相同的 rank。root 的 rankIdx0 = rootIdx0，但 root 公式算出的是 `root_ - rootIdx0 + rankIdxLevel0_`，其 rankIdx0 = `(root_ - rootIdx0 + rankIdxLevel0_) % N0 = rankIdxLevel0_`（因为 root_ - rootIdx0 是 N0 的倍数）。所以 root 的 rankIdx0 = rankIdxLevel0_ = 本 rank 的 rankIdx0，root 确实在本 rank 的 level1 组内。✓

> **数据流完整性**：跳过 ScatterL0 的 rank（不在 root 框）→ CCL[0] 为空 → ScatterL1 中 NHR PreCopy 跳过（inBuffType==HCCL_BUFFER）→ NHR 通信中仅接收 → CCL 填入有效数据。跳过 ScatterL0+L1 的 rank（不在 root 超节点）→ CCL 全空 → ScatterL2 中 NHR 仅接收 → CCL 填入有效数据。AllGather 阶段所有 rank 参与，数据通过层级收集完整。

### 10.6 数据切分与 Offset 计算

符号：`dts = dataTypeSize_`，`rankIdx0 = rankIdxLevel0_`，`rankIdx1 = rankIdxLevel1_`

```
ScatterL0 (Mesh1D, 框内 N0 张卡):
  sliceSize = currDataCount / N0 * dts                        (sliceSizeL0)
  inBuffType = INPUT, outBuffType = HCCL_BUFFER
  inBuffBaseOff = processedDataCount * dts, outBuffBaseOff = 0, hcclBuffBaseOff = 0
  inputSliceStride = sliceSizeL0, outputSliceStride = 0
  repeatNum = 1
  # 结果: 每 rank 在 CCL[0] 收到 sliceSizeL0 数据

ScatterL1 (NHR, 超节点内 N1 个框):
  sliceSize = currDataCount / N0 / N1 * dts                   (sliceSizeL1 = sliceSizeL0 / N1)
  inBuffType = HCCL_BUFFER, outBuffType = HCCL_BUFFER
  inBuffBaseOff = 0, outBuffBaseOff = 0, hcclBuffBaseOff = 0
  inputSliceStride = sliceSizeL1, outputSliceStride = sliceSizeL1
  repeatNum = 1
  # PreCopy 跳过(inBuffType==HCCL_BUFFER), PostCopy 跳过(srcOff==dstOff)
  # 结果: rank k 数据在 CCL[k * sliceSizeL1]

ScatterL2 (NHR, 跨超节点 N2 个组):
  sliceSize = currDataCount / N0 / N1 / N2 * dts              (sliceSizeL2 = sliceSizeL1 / N2)
  inBuffType = HCCL_BUFFER, outBuffType = HCCL_BUFFER
  inBuffBaseOff = rankIdx1 * sliceSizeL1, outBuffBaseOff = rankIdx1 * sliceSizeL1, hcclBuffBaseOff = rankIdx1 * sliceSizeL1
  inputSliceStride = sliceSizeL2, outputSliceStride = sliceSizeL2
  repeatNum = 1
  # NHR 访问 N2 * sliceSizeL2 = sliceSizeL1 连续空间, PostCopy 跳过

AGL2 (NHR, 跨超节点):
  sliceSize = sliceSizeL2                                     (输入切片大小)
  inBuffType = HCCL_BUFFER, outBuffType = HCCL_BUFFER
  inBuffBaseOff = rankIdx1 * sliceSizeL1, outBuffBaseOff = 0, hcclBuffBaseOff = rankIdx1 * sliceSizeL1
  inputSliceStride = sliceSizeL2, outputSliceStride = sliceSizeL1
  repeatNum = 1
  # LocalDataCopy 跳过(inOff==scOff), PostLocalCopy 跳过(outputPtr==hcclBuff.addr)
  # 结果: CCL[rankIdx1 * sliceSizeL1] 有 sliceSizeL1 数据

AGL1 (NHR, 超节点内):
  sliceSize = sliceSizeL1
  inBuffType = HCCL_BUFFER, outBuffType = HCCL_BUFFER
  inBuffBaseOff = 0, outBuffBaseOff = 0, hcclBuffBaseOff = 0
  inputSliceStride = sliceSizeL1, outputSliceStride = sliceSizeL1
  repeatNum = 1
  # LocalDataCopy 跳过, PostLocalCopy 跳过
  # 结果: CCL[0] 有 sliceSizeL0 数据

AGL0 (Mesh1D1D ZAxisDetour, 框内):
  sliceSize = sliceSizeL0
  inBuffType = HCCL_BUFFER, outBuffType = OUTPUT
  inBuffBaseOff = 0, outBuffBaseOff = processedDataCount * dts, hcclBuffBaseOff = 0
  inputSliceStride = sliceSizeL0, outputSliceStride = sliceSizeL0
  repeatNum = 1
  # LocalDataCopy: CCL→OUTPUT 拷贝本 rank 份, CCL 内拷贝跳过(inOff==cclOff)
  # RunAllGatherMesh: SendRecvRead 直接写 OUTPUT, PostLocalCopy 跳过(inBuffType==HCCL_BUFFER)
  # 结果: OUTPUT 有完整 currDataCount * dts 数据
```

> **skipLevel1 兼容**：当 N1==1 时 rankIdx1=0, sliceSizeL1=sliceSizeL0, rankIdx1*sliceSizeL1=0，offset 自然退化。

### 10.7 与 AllReduce 3 级的关键差异

| 维度 | AllReduce 3 级 | Broadcast 3 级 |
|------|---------------|----------------|
| Phase 1 | ReduceScatter（带 reduce） | Scatter（无 reduce） |
| Root | 无 root 概念 | 有 root，Scatter 阶段需 SetRoot + 条件跳过（L0: 仅 root 框, L1: 仅 root 超节点, L2: 全参与）；AllGather 全参与 |
| repeatNum | 全部 = 1 | 全部 = 1（相同） |
| 模板0 | ReduceScatterMesh1DZAxisDetour | ScatterMesh1D |
| 模板1-2 | ReduceScatterNHR | ScatterNHR |
| 模板3-5 | AllGatherNHR/NHR/Mesh1D1DZAxisDetour | AllGatherNHR/NHR/Mesh1D1DZAxisDetour |
| CCL Buffer | 双段分区（rsResult + meshComm） | 单段复用 |
| hcclBuffBaseOff | 每步显式设置（RSL0=meshCommBuffOffset_, 其余按 rankIdx 计算） | 每步显式设置（L0/L1=0, L2=rankIdx1*sliceSizeL1），同 AllReduce 模式 |
| offset 设计原则 | hcclBuffBaseOff = 数据所在位置，使 LocalDataCopy/PreCopy 跳过冗余拷贝 | 同左（完全一致） |

### 10.8 注册

```cpp
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_BROADCAST,
    InsBroadcastSequenceMesh1DNHRNHR,
    InsV2BroadcastSequenceExecutor3Level,
    TopoMatchMultilevel,
    InsTempScatterMesh1D,      // Scatter L0 (框内)
    InsTempScatterNHR,          // Scatter L1 (框间)
    InsTempScatterNHR,          // Scatter L2 (跨超节点)
    InsTempAllGatherNHR,        // AllGather L2 (跨超节点)
    InsTempAllGatherNHR,        // AllGather L1 (框间)
    InsTempAllGatherMesh1D1DZAxisDetour);  // AllGather L0 (框内, Z 轴绕路)
```

---

## 11. 风险分析：L0 Scatter 不支持 Z 轴绕路

### 11.1 现状

| Level | AllReduce 3 级 | Broadcast 3 级 |
|-------|---------------|----------------|
| L0 down | `ReduceScatterMesh1DZAxisDetour`（Z 轴绕路） | `ScatterMesh1D`（plain，**无 Z 轴绕路变体**） |
| L0 up | `AllGatherMesh1D1DZAxisDetour`（Z 轴绕路） | `AllGatherMesh1D1DZAxisDetour`（Z 轴绕路） |

**`ScatterMesh1DZAxisDetour` 不存在**——Scatter 模板没有 Z 轴绕路变体，而 ReduceScatter 有。

### 11.2 根因

**Z 轴绕路模板的 `CalcRes` 计算 level0 + level1 两级 channel**，支持 CLOS 拓扑（多 channel/rank）：

- `ReduceScatterMesh1DZAxisDetour::CalcRes`（`ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.cc:26-50`）：
  ```cpp
  CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels);  // 框内 mesh
  CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels);  // 框间 CLOS
  mergedChannels = level0Channels + level1Channels;
  level0ChannelNumPerRank_ = CalcChannelsPerRank(level0Channels);
  level1ChannelNumPerRank_ = CalcChannelsPerRank(level1Channels);
  ```

- `ScatterMesh1D::CalcRes`（`ins_temp_scatter_mesh_1D.cc:54-71`）：
  ```cpp
  CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels);  // 仅 level0
  // 无 level1 channel，不支持 CLOS 多端口
  ```

### 11.3 风险链

1. **Selector 不检查 level0Topo**（对应 R15）— 原 selector 仅检查 `topoLevelNums==3 && !level2Uboe`，不检查 `level0Topo == MESH_1D`，CLOS 拓扑会路由到 sequence executor。**已缓解**：selector 条件改为 `level0Topo == MESH_1D && !level2Uboe`（见 §1.2 修改后的 selector 逻辑）。

2. **ScatterMesh1D 在 CLOS 下 CalcRes 失败或 channel 不完整** — `CalcChannelRequestMesh1D` 仅支持 MESH_1D。CLOS 拓扑下可能无法正确计算框间 channel。

3. **ZAxisDetour AG 退化到单 channel 模式** — executor 的 `CalcRes` 中 channel 仅从 Scatter 模板分配（`channels[0] = resReqScatterL0.channels[0]`）。ZAxisDetour 的 `SetchannelsPerRank` 收到单 channel/rank 数据时：
   - `channelsPerRank_ = 1` → 不进入多 channel 分支
   - `level0ChannelNumPerRank_ = 1`（默认），`level1ChannelNumPerRank_ = 0`（默认），`level0DataRatio_ = 1.0`（默认）
   - 所有数据走 level0 channel，无法利用 CLOS 框间带宽

4. **AllReduce 不受影响** — AllReduce L0 两端（RS + AG）都用 ZAxisDetour，channel 由 RS 的 `CalcRes` 计算（含 level1），AG 收到完整的多 channel 结构。

### 11.4 影响矩阵

| level0 拓扑 | ScatterMesh1D CalcRes | ZAxisDetour AG | 结果 |
|-------------|----------------------|----------------|------|
| MESH_1D | 正确（单 channel） | 单 channel 模式（`channelsPerRank_=1`） | **正常工作** |
| CLOS / MESH_1D_CLOS | 可能失败或 channel 不完整 | 退化到单 channel，无法利用 CLOS 带宽 | **功能可能正常但性能退化，或 CalcRes 失败** |

### 11.5 依赖与缓解

1. **selector 层（对应 R15）**：selector 条件纳入 `level0Topo == MESH_1D` 检查（见 §1.2 修改后的 selector 逻辑），CLOS/MESH_1D_CLOS 拓扑回退 parallel executor。**该缓解已落地到 selector 条件。**

2. **新建 `ScatterMesh1DZAxisDetour`（未来）**：参照 `ReduceScatterMesh1DZAxisDetour`，override `CalcRes` 调用 `CalcChannelRequestMesh1DLevel0` + `CalcChannelRequestMesh1DLevel1`，override `KernelRun` 使用 merged channel 分发数据。此后 Broadcast 3 级可支持 CLOS 拓扑。

3. **Channel 分配方式（现有隐患）**：executor 的 `CalcRes` 中 `resourceRequest.channels[X]` 仅取 Scatter 模板的 channel 结果，AllGather 的 channel 请求被丢弃。即使新建 `ScatterMesh1DZAxisDetour`，也需确保 Scatter 和 AllGather 的 ZAxisDetour 产生一致的 channel 结构。

---

## 12. Dependencies / Assumptions

- TopoMatchMultilevel 已存在，可直接复用
- Scatter Mesh1D/NHR 和 AllGather Mesh1D/NHR 模板已存在，可直接复用
- `REGISTER_EXEC_V2_MULTI` 宏已存在，支持 6 模板参数
- AllReduce 3 级 executor（`InsV2AllReduceSequenceExecutorAicpu3Level`）可作为参考实现
- 假设 CCL Buffer 足够大——若不足，loop 次数增加
- **AGL0 使用 `InsTempAllGatherMesh1D1DZAxisDetour`（Z 轴绕路），依赖 ZAxisDetour 的 `SetchannelsPerRank` 能处理单 channel 场景（MESH_1D 退化模式）**
- **ScatterL0 使用 `InsTempScatterMesh1D`（plain），仅支持 MESH_1D 拓扑。CLOS 拓扑需 selector 层拦截或新建 `ScatterMesh1DZAxisDetour`**
- **Channel 从 Scatter 模板分配，假设 Scatter 和 AllGather 在同一 level 产生兼容的 channel 结构**
- **图模式（OFFLOAD）：3 级 executor 不设 `enableRemoteMemAccess`，图模式下不使用 `remoteOutputGraphMode` 直接远程写入，功能正确但性能未优化。AllReduce 3 级同样未实现，两者对称**
- **opMode 依赖（RSK3 无风险）：`totalMult` 取 ScatterL0 的 `CalcScratchMultiple`（=1，无条件返回），AGL0 的 `CalcScratchMultiple`（ZAxisDetour 继承基类，非 OPBASE 返回 0）从未被调用。ZAxisDetour 的 `PostLocalCopy` opMode 门控对 AGL0 无效（`inBuffType==HCCL_BUFFER` 已早返回）**

---

## 13. 参考文件

- `src/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.cc` — AllReduce 3 级参考实现（6 模板参数）
- `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_3level.cc` — RS 3 级参考实现（3 模板参数）
- `src/ops/all_gather/executor/ins_v2_all_gather_sequence_executor_3level.cc` — AG 3 级参考实现（3 模板参数）
- `src/ops/broadcast/executor/ins_v2_broadcast_parallel_executor.cc` — 现有 parallel executor（4 模板参数，Scatter+AllGather）
- `src/ops/op_common/topo/topo_match_multilevel.cc` — 多级拓扑匹配器（三级 sequence executor 统一使用）
- `src/ops/op_common/topo/topo_match_3_level.cc` — 三级拓扑匹配器（OmniPipe 用，本方案不用）
- `src/ops/scatter/template/aicpu/ins_temp_scatter_mesh_1D.h` — Scatter Mesh1D 模板
- `src/ops/scatter/template/aicpu/ins_temp_scatter_nhr.h` — Scatter NHR 模板
- `src/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.h` — AllGather Mesh1D 模板
- `src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr.h` — AllGather NHR 模板
