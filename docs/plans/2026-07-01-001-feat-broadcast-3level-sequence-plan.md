---
title: "feat: Broadcast 3 级 Sequence Executor（Scatter + AllGather）"
type: feat
status: active
created: 2026-07-01
depth: standard
target_repo: hccl_2039
origin: docs/brainstorms/broadcast-3-level-sequence-requirements.md
---

## Problem Frame

Broadcast 在三级非 UBOE 拓扑下无三级 sequence 算法，全部走 `ParallelNHRNHRUboe` + `Squeeze2D` 压缩为两级并行，框内 mesh 高带宽链路浪费。新增 `InsV2BroadcastSequenceExecutor3Level`，通过 Scatter(3级) + AllGather(3级) 的 6 步串行实现三级 broadcast。

(see origin: docs/brainstorms/broadcast-3-level-sequence-requirements.md)

---

## Scope

### In

- 新建 `InsV2BroadcastSequenceExecutor3Level` executor（6 模板参数）
- 注册 `InsBroadcastSequenceMesh1DNHRNHR`（`REGISTER_EXEC_V2_MULTI` + `TopoMatchMultilevel`）
- Selector 新增三级非 UBOE 分支
- 新增 `broadcast_3level_testcase.cc` ST 测试
- 复用现有 Scatter Mesh1D/NHR + AllGather Mesh1D/NHR 模板

### Out

- 不修改现有 parallel executor
- 不新增 Scatter 或 AllGather 模板
- CCU/AIV 引擎
- Reduce 三级 sequence（另做）
- Barrier/Scatter 三级

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 执行模式 | Scatter(3级) + AllGather(3级) 串行 | 与 AllReduce 3 级对称，复用现有模板 |
| TopoMatch | TopoMatchMultilevel | 与 RS/AG/AR 3 级一致，支持 HostDPU，Layer2 baseModSize=L0*L1 |
| CCL Buffer | 单段复用 | 简化管理，每步通过 offset 指定位置 |
| Root 处理 | Scatter 阶段含 root 的组执行，AllGather 全组执行 | 与 parallel executor 一致 |
| 算法选择 | 三级非 UBOE → sequence；UBOE → parallel | 不设数据量阈值 |
| 模板参数 | 6 个（Scatter L0/L1/L2 + AG L2/L1/L0） | REGISTER_EXEC_V2_MULTI 支持 |

---

## Implementation Units

### U1. 新建 Broadcast 3 级 Sequence Executor

**Goal:** 创建 `InsV2BroadcastSequenceExecutor3Level`，参考 AllReduce 3 级 executor 结构。

**Files:**
- `src/ops/broadcast/executor/ins_v2_broadcast_sequence_executor_3level.h` — 新增
- `src/ops/broadcast/executor/ins_v2_broadcast_sequence_executor_3level.cc` — 新增
- `src/ops/broadcast/executor/CMakeLists.txt` — 加新文件

**Dependencies:** 无

**Approach:**
- 头文件参考 `ins_v2_all_reduce_sequence_executor_aicpu_3level.h`，6 模板参数
- 新增成员变量 `u32 rootIdx0_/rootIdx1_/rootIdx2_` 和 `u32 rankIdxLevel2_`（AllReduce 3 级无 root 概念，Broadcast 需要）
- `CalcAlgHierarchyInfo`：调 `TopoMatchMultilevel::MatchTopo`
- `CalcRes`：6 个模板各调 `CalcRes`，合并 slaveThreadNum/notify/channels[0/1/2]
- `Orchestrate`：设 myRank_/rankSize/dataType/root_ 等，计算 root 索引和 rank 索引，调 OrchestrateLoop
- `OrchestrateLoop`：
  - 构造 6 个模板实例，Scatter 模板调 `SetRoot`（详见下方"Root 处理与 Scatter 跳过"章节）
  - CCL Buffer 单段复用
  - loop 循环：Step1-3 Scatter L0→L1→L2（条件跳过），Step4-6 AG L2→L1→L0（全参与）
  - 每步通过 `GenTempAlgParamsXxx` 生成参数（sliceSize 递减/递增）
- `GenBaseTempAlgParams`：设 buffer type（INPUT→CCL→CCL→CCL→CCL→CCL→OUTPUT）
- `GenTempResource`：从 remoteRankToChannelInfo_ 取对应层级 channel + thread
- skipLevel1_ 支持（rankSizeLevel1==1 时跳过 L1）
- 注册：`REGISTER_EXEC_V2_MULTI(BROADCAST, InsBroadcastSequenceMesh1DNHRNHR, ..., TopoMatchMultilevel, ScatterMesh1D, ScatterNHR, ScatterNHR, AGNHR, AGNHR, AGMesh1D1DZAxisDetour)`

**Root 处理与 Scatter 跳过（解决 R2）：**

> **根本原因**：Broadcast 中只有 root 持有源数据。Scatter 阶段逐级分发数据，每级只有"数据路径"上的组才有数据可 scatter。非数据路径上的组没有数据源，无法执行 Scatter。
>
> **各级数据可用性**：
> - ScatterL0 前：只有 root 所在框的 root rank 的 INPUT 有有效数据
> - ScatterL0 后：root 所在框的所有 rank 有各自分片；其他框无数据
> - ScatterL1 后：root 所在超节点的所有 rank 有各自分片；其他超节点无数据
> - ScatterL2 后：所有 rank 有各自分片（root 超节点分发到所有超节点）
>
> **跳过决策**：某级组内没有持有数据的 rank → 该组无有效 root → 跳过 `KernelRun`。若不跳过，ScatterNHR 的 `GetStepInfo` 调用 `GetAlgRank(root_, subCommRanks_[0], rootAlgRank)`（`ins_temp_scatter_nhr.cc:89`），root 不在组内时返回 `HCCL_E_PARA`（`template_utils.cc:17`）但返回值未检查，`rootAlgRank` 未定义，导致 NHR 向错误 rank 发送或 hang——这是不跳过时的技术后果，而非跳过的根本原因。即便修复 `GetAlgRank` 检查，无数据可 scatter 仍然产生错误结果。

在 `Orchestrate` 中计算 root 索引和 rank 索引：

```cpp
root_ = param.root;
rootIdx0_ = root_ % rankSizeLevel0_;
rootIdx1_ = (root_ / rankSizeLevel0_) % rankSizeLevel1_;
rootIdx2_ = root_ / (rankSizeLevel0_ * rankSizeLevel1_);

rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
rankIdxLevel1_ = (myRank_ / rankSizeLevel0_) % rankSizeLevel1_;
rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_);
```

每步 Scatter 的 root 计算与跳过条件：

| 步骤 | root（全局 rank） | 跳过条件 | 原理 |
|------|-------------------|---------|------|
| ScatterL0 | `root_`（全局 root） | `rankIdxLevel2_ != rootIdx2_ \|\| rankIdxLevel1_ != rootIdx1_`（不在 root 的框） | 只有 root 所在框有 INPUT 数据 |
| ScatterL1 | `root_ - rootIdx0_ + rankIdxLevel0_`（root 框内与本 rank 同 rankIdx0 的 rank） | `rankIdxLevel2_ != rootIdx2_`（不在 root 的超节点）或 `skipLevel1_` | 只有 root 所在超节点的框间组才有 root 框的 rank 作为数据源 |
| ScatterL2 | `rootIdx2_ * (N0*N1) + rankIdxLevel1_ * N0 + rankIdxLevel0_`（root 超节点内与本 rank 同 rankIdx0/1 的 rank） | **不跳过**（所有 rank 参与） | root 超节点的 rank 有数据，分发到所有超节点；root 始终在本 rank 的 level2 组内 |

OrchestrateLoop 中的执行逻辑：

> **为什么需要 `SetRoot`**：模板构造函数从 `param.root` 初始化 `root_`（`alg_v2_template_base.cc:18`），即全局 root。这对 ScatterL0 正确（全局 root 在 root 框的 level0 组内），但对 ScatterL1/L2 错误——各级组内的有效 root 是不同 rank。AICPU `InsTempScatterNHR` 的 `GetStepInfo`/`PreCopy`/`PostCopy` 均使用 `root_` 成员变量（不读 `tempAlgParams.root`，与 DPU 版本 `InsTempScatterNHRDPUInter` 在 `KernelRun` 内调 `SetRoot(tempAlgParams.root)` 不同），因此必须在 `KernelRun` 前显式调 `SetRoot` 覆盖。

```cpp
// ====== 模板初始化（loop 前，SetRoot 只需调一次）======

// 构造 6 个模板 + SetchannelsPerRank（与 AllReduce 3 级一致）
algTemplateScatterL0 = make_shared<...>(param, myRank_, infos[0]);
algTemplateScatterL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]);
// ... 其余 5 个模板同理 ...

// Scatter 模板调 SetRoot（AllGather 模板无 root，不需要）
// ScatterL0: root = 全局 root（构造函数已设，SetRoot 冗余但显式）
algTemplateScatterL0->SetRoot(root_);
// ScatterL1: root = root 框内与本 rank 同 rankIdx0 的 rank（覆盖构造函数的全局 root）
if (!skipLevel1_) {
    algTemplateScatterL1->SetRoot(root_ - rootIdx0_ + rankIdxLevel0_);
}
// ScatterL2: root = root 超节点内与本 rank 同 rankIdx0/1 的 rank
algTemplateScatterL2->SetRoot(rootIdx2_ * (rankSizeLevel0_ * rankSizeLevel1_)
    + rankIdxLevel1_ * rankSizeLevel0_ + rankIdxLevel0_);

// ====== loop 循环 ======
while (processedDataCount < dataCount_) {
    // ... 计算 currDataCount（末轮可能非对齐，由 tailSize 处理）...

    // ====== Phase 1: Scatter 3 级（条件跳过 + tailSize 传播）======

    // Step 1: ScatterL0 — 只有 root 所在框执行
    if (rankIdxLevel2_ == rootIdx2_ && rankIdxLevel1_ == rootIdx1_) {
        GenTempAlgParamsScatterL0(loop, currDataCount, processedDataCount, tempAlgParamsScatterL0);
        CHK_RET(algTemplateScatterL0->KernelRun(param, tempAlgParamsScatterL0, templateResourceScatterL0));
    } else {
        // 跳过的 rank 仍需计算 sliceSize/tailSize 供后续步骤使用
        GenTempAlgParamsScatterL0(loop, currDataCount, processedDataCount, tempAlgParamsScatterL0);
    }

    // Step 2: ScatterL1 — 只有 root 所在超节点执行
    u64 sliceSizeL1 = tempAlgParamsScatterL0.sliceSize;
    u64 tailSizeL1 = tempAlgParamsScatterL0.tailSize;
    if (!skipLevel1_) {
        GenTempAlgParamsScatterL1(loop, currDataCount, tempAlgParamsScatterL0.sliceSize,
            tempAlgParamsScatterL0.tailSize, tempAlgParamsScatterL1);
        if (rankIdxLevel2_ == rootIdx2_) {
            CHK_RET(algTemplateScatterL1->KernelRun(param, tempAlgParamsScatterL1, templateResourceScatterL1));
        }
        sliceSizeL1 = tempAlgParamsScatterL1.sliceSize;
        tailSizeL1 = tempAlgParamsScatterL1.tailSize;
    } else {
        sliceSizeL1 = tempAlgParamsScatterL0.sliceSize;
        tailSizeL1 = (rankIdxLevel0_ == rankSizeLevel0_ - 1) ?
            tempAlgParamsScatterL0.tailSize : tempAlgParamsScatterL0.sliceSize;
    }

    // Step 3: ScatterL2 — 所有 rank 执行
    GenTempAlgParamsScatterL2(loop, currDataCount, sliceSizeL1, tailSizeL1, tempAlgParamsScatterL2);
    CHK_RET(algTemplateScatterL2->KernelRun(param, tempAlgParamsScatterL2, templateResourceScatterL2));

    // ====== Phase 2: AllGather 3 级（所有 rank 执行，无跳过）======

    // Step 4: AGL2 — 用 ScatterL2 的 sliceSize/tailSize，outputSliceStride = sliceSizeL1
    GenTempAlgParamsAGL2(loop, currDataCount, tempAlgParamsScatterL2.sliceSize,
        tempAlgParamsScatterL2.tailSize, sliceSizeL1, tempAlgParamsAGL2);
    CHK_RET(algTemplateAGL2->KernelRun(param, tempAlgParamsAGL2, templateResourceAGL2));

    // Step 5: AGL1 — 用 ScatterL1 的 sliceSize/tailSize
    if (!skipLevel1_) {
        GenTempAlgParamsAGL1(loop, currDataCount, sliceSizeL1, tailSizeL1, tempAlgParamsAGL1);
        CHK_RET(algTemplateAGL1->KernelRun(param, tempAlgParamsAGL1, templateResourceAGL1));
    }

    // Step 6: AGL0 — 用 ScatterL0 的 sliceSize/tailSize
    GenTempAlgParamsAGL0(loop, currDataCount, processedDataCount,
        tempAlgParamsScatterL0.sliceSize, tempAlgParamsScatterL0.tailSize, tempAlgParamsAGL0);
    CHK_RET(algTemplateAGL0->KernelRun(param, tempAlgParamsAGL0, templateResourceAGL0));

    processedDataCount += currDataCount;
    loop++;
}
```

> **数据完整性**：跳过 ScatterL0/L1 的 rank 的 CCL Buffer 对应位置为空，但 NHR 的 PreCopy 因 `inBuffType==HCCL_BUFFER` 跳过（不读空数据），NHR 通信中这些 rank 仅接收（非 root 不发送），接收后 CCL 填入有效数据。AllGather 阶段所有 rank 参与，数据通过层级收集完整。

**tailSize 跨级传播（解决 R6，与 AllReduce 3 级模式完全一致）：**

> Scatter 与 ReduceScatter 的 tail 计算相同——按 rank 数均分数据，末尾 rank 吃余数。各级 `GenTempAlgParamsXxx` 的签名和 tail 传播逻辑直接复用 AllReduce 3 级（`ins_v2_all_reduce_sequence_executor_aicpu_3level.cc:249-444`）。

```
ScatterL0 (从 currDataCount 计算):
  sliceSizeL0 = currDataCount / N0 * dts                          // 正常 rank 的分片
  tailSizeL0  = (currDataCount / N0 + currDataCount % N0) * dts   // 末尾 rank 的分片（含余数）

ScatterL1 (接收 sliceSizeL0, tailSizeL0):
  if rankIdxLevel0_ == N0 - 1:                    // 本 rank 是 L0 末尾 → 用 tailSizeL0
    cnt = tailSizeL0 / dts
  else:                                            // 正常 rank → 用 sliceSizeL0
    cnt = sliceSizeL0 / dts
  sliceSizeL1 = cnt / N1 * dts
  tailSizeL1  = sliceSizeL1 + cnt % N1 * dts

ScatterL2 (接收 sliceSizeL1, tailSizeL1):
  if rankIdxLevel1_ == N1 - 1:                    // 本 rank 是 L1 末尾 → 用 tailSizeL1
    cnt = tailSizeL1 / dts
  else:                                            // 正常 rank → 用 sliceSizeL1
    cnt = sliceSizeL1 / dts
  sliceSizeL2 = cnt / N2 * dts
  tailSizeL2  = sliceSizeL2 + cnt % N2 * dts

AGL2 (接收 ScatterL2 的 sliceSizeL2/tailSizeL2, 以及 sliceSizeL1):
  sliceSize = sliceSizeL2,  tailSize = tailSizeL2
  outputSliceStride = sliceSizeL1                              // 恢复到 L1 粒度

AGL1 (接收 sliceSizeL1, tailSizeL1):
  sliceSize = sliceSizeL1,  tailSize = tailSizeL1

AGL0 (接收 ScatterL0 的 sliceSizeL0, tailSizeL0):
  sliceSize = sliceSizeL0,  tailSize = tailSizeL0
```

> **skipLevel1 tail 传播**（与 AllReduce `line 582-586` 一致）：当 `skipLevel1_` 时，ScatterL1 不执行，L0 的 slice/tail 直接传给 L2：
> ```
> sliceSizeL1 = sliceSizeL0
> tailSizeL1 = (rankIdxLevel0_ == N0 - 1) ? tailSizeL0 : sliceSizeL0
> ```

> **与 AllReduce 的差异**：AllReduce 有 `tailSize > rsResultBuffSize_` 的末轮 buffer 适配（`line 548-564`），Broadcast 单段复用 `totalMult=1` 时全 buffer 可用，无需此适配——总数据量 `currDataCount * dts ≤ cclMem.size`，AGL0 需 `N0 * sliceSizeL0 = currDataCount * dts`，恰好不溢出。

> **root 公式验证**（2×2×2，root=0，rankIdx0=0,1, rankIdx1=0,1, rankIdx2=0,1）：
> - rank 0 (0,0,0)：L0 执行(root=0)，L1 执行(root=0)，L2 执行(root=0)
> - rank 1 (1,0,0)：L0 执行(root=0)，L1 执行(root=1)，L2 执行(root=1)
> - rank 2 (0,1,0)：L0 跳过(idx1≠0)，L1 执行(root=0)，L2 执行(root=2)
> - rank 3 (1,1,0)：L0 跳过(idx1≠0)，L1 执行(root=1)，L2 执行(root=3)
> - rank 4 (0,0,1)：L0 跳过(idx2≠0)，L1 跳过(idx2≠0)，L2 执行(root=0)
> - rank 5 (1,0,1)：L0 跳过，L1 跳过，L2 执行(root=1)
> - rank 6 (0,1,1)：L0 跳过，L1 跳过，L2 执行(root=2)
> - rank 7 (1,1,1)：L0 跳过，L1 跳过，L2 执行(root=3)

**CCL Buffer 分配（单段复用 + 每步动态 offset）：**

各模板 CalcScratchMultiple 返回值：

| 步骤 | 模板 | 返回值 |
|------|------|--------|
| Scatter L0 | ScatterMesh1D | 1 |
| Scatter L1 | ScatterNHR | rankSizeLevel1_ |
| Scatter L2 | ScatterNHR | rankSizeLevel2_ |
| AG L2 | AllGatherNHR | rankSizeLevel2_ |
| AG L1 | AllGatherNHR | rankSizeLevel1_ |
| AG L0 | AllGatherMesh1D1DZAxisDetour | rankSizeLevel0_（OPBASE）/ 0（非 OPBASE） |

> **totalMult 取第一步（ScatterL0）的 CalcScratchMultiple**（与 AllReduce 3 级只取 RSL0 一致），不取 6 步 max。原因：单段复用模式下各步串行执行，最大 CCL 需求为 AGL0 的 `N0 * sliceSizeL0 = maxCountPerLoop * dts`，恰好等于 CCL Buffer 大小，`totalMult = 1` 即可保证不溢出。取 max 反而过度保守（buffer 除以 N0，maxCountPerLoop 缩小 N0 倍，loop 次数增多）。同时绕过 AGL0 的 `opMode_` 依赖（RSK3 消除）。

```
totalMult = algTemplateScatterL0->CalcScratchMultiple(INPUT, HCCL_BUFFER)  // = 1
totalRankAlign = rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_       // 与 AllReduce 3 级一致
maxCountPerLoop = cclMem.size / totalMult / HCCL_MIN_SLICE_ALIGN *
                  HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / totalRankAlign * totalRankAlign
```

> **两级对齐**（与 AllReduce 3 级 `ins_v2_all_reduce_sequence_executor_aicpu_3level.cc:535-537` 一致）：① `HCCL_MIN_SLICE_ALIGN` 硬件最小切片对齐；② `totalRankAlign = N0*N1*N2` 确保 `currDataCount` 可被各级 rank 数整除，使 `sliceSizeL0/L1/L2` 无余数，简化 tail 处理。末轮 loop 的非对齐余数由 tail 逻辑处理（R6）。

每步 offset（含 `hcclBuffBaseOff`，与 AllReduce 3 级模式一致）：

> **关键约束**：NHR 模板（ScatterNHR `ins_temp_scatter_nhr.cc:297,339` / AllGatherNHR `ins_temp_all_gather_nhr.cc:175-177`）内部通过 `hcclBuffBaseOff + sliceIdx * sliceSize` 访问 CCL Buffer，需要 `templateRankSize_ * sliceSize` 连续空间。Mesh1D 模板（ScatterMesh1D `ins_temp_scatter_mesh_1D.cc:159,240` / AllGatherMesh1D `ins_temp_all_gather_mesh_1D.cc:153,169,171`）同样通过 `hcclBuffBaseOff` 定位 scratch 区。**每步必须正确设置 `hcclBuffBaseOff`**，否则模板在错误位置读写，导致数据损坏或 hang。

> **设计原则**（与 AllReduce 3 级一致）：各级 Scatter 输出位置 = 下级输入位置 = `hcclBuffBaseOff`，使 NHR 模板的 `LocalDataCopy`/`PreCopy` 检测到 `inputPtr == hcclBuff.addr && inOff == scOff` 而跳过冗余拷贝。

符号定义：`dts = dataTypeSize_`，`N0/N1/N2 = rankSizeLevel0_/1_/2_`，`rankIdx0/1 = rankIdxLevel0_/1_`

- `sliceSizeL0 = currDataCount / N0 * dts`
- `sliceSizeL1 = currDataCount / N0 / N1 * dts`（= `sliceSizeL0 / N1`）
- `sliceSizeL2 = currDataCount / N0 / N1 / N2 * dts`（= `sliceSizeL1 / N2`）

| 步骤 | inBuffType | outBuffType | inBuffBaseOff | outBuffBaseOff | hcclBuffBaseOff | inputSliceStride | outputSliceStride | sliceSize |
|------|------------|-------------|---------------|----------------|-----------------|-------------------|-------------------|-----------|
| ScatterL0 (Mesh1D) | INPUT | HCCL_BUFFER | processedDataCount * dts | 0 | 0 | sliceSizeL0 | 0 | sliceSizeL0 |
| ScatterL1 (NHR) | HCCL_BUFFER | HCCL_BUFFER | 0 | 0 | 0 | sliceSizeL1 | sliceSizeL1 | sliceSizeL1 |
| ScatterL2 (NHR) | HCCL_BUFFER | HCCL_BUFFER | rankIdx1 * sliceSizeL1 | rankIdx1 * sliceSizeL1 | rankIdx1 * sliceSizeL1 | sliceSizeL2 | sliceSizeL2 | sliceSizeL2 |
| AGL2 (NHR) | HCCL_BUFFER | HCCL_BUFFER | rankIdx1 * sliceSizeL1 | 0 | rankIdx1 * sliceSizeL1 | sliceSizeL2 | sliceSizeL1 | sliceSizeL2 |
| AGL1 (NHR) | HCCL_BUFFER | HCCL_BUFFER | 0 | 0 | 0 | sliceSizeL1 | sliceSizeL1 | sliceSizeL1 |
| AGL0 (Mesh1D1D ZAxisDetour) | HCCL_BUFFER | OUTPUT | 0 | processedDataCount * dts | 0 | sliceSizeL0 | sliceSizeL0 | sliceSizeL0 |

> **skipLevel1 兼容**：当 `N1==1` 时 `rankIdx1=0`、`sliceSizeL1=sliceSizeL0`，`rankIdx1 * sliceSizeL1 = 0`，offset 表自然退化，无需特殊分支。

每步数据流与 `hcclBuffBaseOff` 推导（对照模板源码验证）：

1. **ScatterL0 (Mesh1D)**：root 从 INPUT（`inBuffBaseOff = processedDataCount * dts`）读取，`outputSliceStride=0` 使每个 rank 在自己 CCL 的 `hcclBuffBaseOff=0` 处收到 `sliceSizeL0` 数据。`PreCopy` 仅 root 执行（`ins_temp_scatter_mesh_1D.cc:137`），`PostCopy` 因 `outBuffType==HCCL_BUFFER` 跳过（`ins_temp_scatter_mesh_1D.cc:174`）。**结果**：CCL[0..sliceSizeL0]。

2. **ScatterL1 (NHR)**：输入为 CCL[0]（ScatterL0 输出）。`hcclBuffBaseOff=0`，NHR 通信访问 `N1 * sliceSizeL1 = sliceSizeL0` 连续空间。`PreCopy` 因 `inBuffType==HCCL_BUFFER` 跳过（`ins_temp_scatter_nhr.cc:202`）。`PostCopy` 中 `srcOffset = 0 + myAlgRank * sliceSizeL1`，`dstOffset = 0 + myAlgRank * sliceSizeL1`，`srcOffset == dstOffset` 跳过（`ins_temp_scatter_nhr.cc:243`）。**结果**：rank `k` 数据在 CCL[`k * sliceSizeL1`]。

3. **ScatterL2 (NHR)**：输入为 CCL[`rankIdx1 * sliceSizeL1`]（ScatterL1 输出）。`hcclBuffBaseOff = rankIdx1 * sliceSizeL1`，NHR 访问 `N2 * sliceSizeL2 = sliceSizeL1` 连续空间。`PreCopy` 跳过，`PostCopy` 中 `srcOffset == dstOffset` 跳过。**结果**：数据在 CCL[`rankIdx1 * sliceSizeL1 + myAlgRankL2 * sliceSizeL2`]。

4. **AGL2 (NHR)**：`hcclBuffBaseOff = rankIdx1 * sliceSizeL1`（与 ScatterL2 相同），数据已在正确位置。`LocalDataCopy` 中 `inOff = sliceSizeL2 * myAlgRank + rankIdx1 * sliceSizeL1 = scOff`，且 `inputPtr == hcclBuff.addr`，跳过拷贝（`ins_temp_all_gather_nhr.cc:395`）。NHR 通信收集 `N2` 份 `sliceSizeL2`，产出 `sliceSizeL1`。`PostLocalCopy` 因 `outputPtr == hcclBuff.addr` 跳过（`ins_temp_all_gather_nhr.cc:408`）。**结果**：CCL[`rankIdx1 * sliceSizeL1`] 有 `sliceSizeL1` 数据。

5. **AGL1 (NHR)**：`hcclBuffBaseOff = 0`，N1 份输入各在 CCL[`k * sliceSizeL1`]（AGL2 输出）。`LocalDataCopy` 中 `inOff = sliceSizeL1 * myAlgRank + 0 = scOff` 跳过拷贝。NHR 通信收集 `N1` 份 `sliceSizeL1`，产出 `sliceSizeL0`。`PostLocalCopy` 跳过。**结果**：CCL[0] 有 `sliceSizeL0` 数据。

6. **AGL0 (Mesh1D1D ZAxisDetour)**：`hcclBuffBaseOff = 0`，输入在 CCL[0]（AGL1 输出）。ZAxisDetour override 了 `KernelRun`/`RunAllGatherMesh`/`LocalDataCopy`（`Z_axis_detour.cc`）：`LocalDataCopy` 仅 input→OUTPUT 一步、无 CCL 重分布拷贝（tx 直接从 OUTPUT 读取），`RunAllGatherMesh` 经 `SendRecvRead` 将远端 CCL 数据写入 OUTPUT 且自带正确 tail 处理（`CalcSliceSizeForChannel` 按 `dmaRead` 分离 tx/rx），`PostLocalCopy` 因 `inBuffType==HCCL_BUFFER` 早返回。offset 与基类一致（`outBuffBaseOff = processedDataCount * dts`）。**结果**：OUTPUT 有完整 `currDataCount * dts` 数据。详见 Key Decision（需求 L206）/ RSK3 / RSK13。

> **注意**：`AGL1.outputSliceStride` 在表中设为 `sliceSizeL1`（语义正确值），但因 `PostLocalCopy` 跳过而不影响行为。AllReduce 3 级 AGL1 设为 `0`（同样因跳过而不影响），两者等效。

与 AllReduce 3 级对比：

| | AllReduce 3 级 | Broadcast 3 级 |
|---|---|---|
| 分区方式 | 双段（rsResult + meshComm） | 单段复用 |
| scratchMultiplier | 只取 RSL0 的值 | 只取 ScatterL0 的值（=1），与 AllReduce 只取第一步一致 |
| maxCountPerLoop | cclMem / (mult+1) | cclMem / totalMult（totalMult=1，即全 buffer） |
| hcclBuffBaseOff | 每步显式设置（RSL0=meshCommBuffOffset_，RSL1/AGL1=0，RSL2/AGL2=rankIdx1*sliceSizeRSL1） | 每步显式设置（ScatterL0/ScatterL1/AGL1/AGL0=0，ScatterL2/AGL2=rankIdx1*sliceSizeL1） |
| offset 设计原则 | NHR 步骤 hcclBuffBaseOff = 数据所在位置，使 LocalDataCopy/PreCopy 跳过冗余拷贝 | 同左（完全一致） |
| GenTempAlgParams 签名 | 6 个 `GenTempAlgParamsXxx` 方法，每个接收上一步的 sliceSize/tailSize | 同左 |

**Patterns to follow:** `src/ops/all_reduce/executor/ins_v2_all_reduce_sequence_executor_aicpu_3level.cc`

**Test scenarios:**

| 用例名 | 拓扑 | root | 数据类型 | 数据量 | 说明 |
|--------|------|------|---------|--------|------|
| `st_broadcast_3level_2x2x2_int8_send1` | 2×2×2 | 0 | INT8 | 1 | 最小数据量 |
| `st_broadcast_3level_2x2x2_fp32_root0` | 2×2×2 | 0 | FP32 | 200 | 最小三级 |
| `st_broadcast_3level_4x2x2_int32_root0` | 4×2×2 | 0 | INT32 | 200 | 中等规模 |
| `st_broadcast_3level_2x4x4_fp16_root0` | 2×4×4 | 0 | FP16 | 300 | 多框多卡 |
| `st_broadcast_3level_2x2x2_int8_root3` | 2×2×2 | 3 | INT8 | 200 | root 非0框 + INT8 |
| `st_broadcast_3level_2x2x2_bfp16_root4` | 2×2×2 | 4 | BFP16 | 200 | root 非0超节点 + BFP16 |
| `st_broadcast_3level_2x2x2_int64_root7` | 2×2×2 | 7 | INT64 | 200 | root 末尾 rank + INT64 |
| `st_broadcast_3level_2x1x4_fp32_skip_l1` | 2×1×4 | 0 | FP32 | 200 | L1 退化 |
| `st_broadcast_3level_2x4x1_fp32_one_card_per_server` | 2×4×1 | 0 | FP32 | 200 | 每 server 1 卡 |
| `st_broadcast_3level_2x2x2_fp32_send501m_plus_1` | 2×2×2 | 0 | FP32 | 501MB+1 | 大数据量 loop |
| `st_broadcast_3level_2x2x2_fp32_send201` | 2×2×2 | 0 | FP32 | 201 | 非对齐 |

**Verification:** 编译通过 + ST 用例通过

---

### U2. 修改 Selector

**Goal:** 新增三级非 UBOE 分支，选 `InsBroadcastSequenceMesh1DNHRNHR`。

**Files:**
- `src/ops/broadcast/selector/broadcast_auto_selector.cc` — 修改

**Dependencies:** U1

**Approach:**
- 在 `SelectAicpuAlgo` 三级拓扑分支中，`else`（每 server 多卡）按顺序拆分（对标 AllReduce L355/L371）：
  - `level2Uboe == true` → `InsBroadcastParallelNHRNHRUboe`（UBOE 走现有 parallel，不变）
  - `level0Topo == MESH_1D`（非 UBOE）→ `InsBroadcastSequenceMesh1DNHRNHR`（新增；CLOS 缓解：仅 MESH_1D 走 sequence）
  - 否则（CLOS/MESH_1D_CLOS）→ `InsBroadcastParallelNHRNHRUboe`

**Patterns to follow:** `src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc`（RS 三级 selector 分支）

**Test scenarios:**
- 三级 MESH_1D 且非 UBOE → 选 sequence
- 三级 UBOE → 选 parallel
- 三级 CLOS/MESH_1D_CLOS 且非 UBOE → 选 parallel（CLOS 缓解）
- 三级每 server 1 卡 → 选 NHR

**Verification:** grep 确认 selector 返回正确算法名

---

### U3. 修改 CMake 和 scatter_aicpu_kernel.cmake

**Goal:** 注册新文件到构建系统。

**Files:**
- `src/ops/broadcast/executor/CMakeLists.txt` — 加 .cc（U1 已含）
- `src/scatter_aicpu_kernel.cmake` — NOT COMP_850 块加 executor .cc

**Dependencies:** U1

**Approach:**
- executor/CMakeLists.txt：加 `ins_v2_broadcast_sequence_executor_3level.cc`
- scatter_aicpu_kernel.cmake：在 barrier 相关条目附近加新 executor

**Verification:** grep 确认 cmake 中有新文件

---

### U4. ST 测试

**Goal:** 新增 `broadcast_3level_testcase.cc`，验证三级 sequence broadcast。

**Files:**
- `test/st/algorithm/testcase/broadcast_3level_testcase.cc` — 新增
- `test/st/algorithm/testcase/CMakeLists.txt` — 加新文件

**Dependencies:** U1, U2, U3

**Approach:**
- 参考 `all_gather_3level_testcase.cc` 结构
- 用例：对称三级（2x2x2、4x2x2）、root 不同位置、不同数据量
- 调用 `CheckBroadcast(taskQueues, rankSize, dataType, count, root)` 验证数据正确性
- 验证不 hang + 走新流程（日志含 `InsBroadcastSequenceMesh1DNHRNHR`）

**Test scenarios:**
- 8P×8Pod×2Cluster 对称三级，root=0
- 4P×2Pod×2Cluster 对称三级，root 在非 0 server
- 2P×2Pod×2Cluster 小规模三级
- 大数据量触发 loop 分片

**Verification:** `bash build.sh --st` 用例通过

---

## Risks

### 高风险

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| RSK1 | CCL Buffer 单段复用 `hcclBuffBaseOff` 缺失导致 NHR 模板读写错位 | Low（已修复） | 已补充每步完整 offset 表（含 `hcclBuffBaseOff`/`inBuffBaseOff`/`outBuffBaseOff`/stride），对照 4 个模板源码验证 NHR/Mesh1D 内部 buffer 布局兼容性，与 AllReduce 3 级 offset 模式一致。详见 U1 CCL Buffer 分配章节 |
| RSK2 | R10 "非 root 组跳过 Scatter" 缺乏参考实现，可能导致 hang | Low（已修复） | 已补充完整的 root 跳过方案。**根本原因**：broadcast 中非数据路径上的组没有数据可 scatter。**方案**：① 各级 root 计算公式（ScatterL0=root_, ScatterL1=root_-rootIdx0+rankIdx0, ScatterL2=rootIdx2*(N0*N1)+rankIdx1*N0+rankIdx0）；② 跳过条件（L0: 不在 root 框, L1: 不在 root 超节点, L2: 不跳过——root 超节点的 rank 在所有 level2 组内且有数据）；③ `GetAlgRank` 未检查返回值是不跳过时的技术后果（`template_utils.cc:17` + `ins_temp_scatter_nhr.cc:89`），非根本原因；④ 2×2×2 全 8 rank 验证表。详见 U1 "Root 处理与 Scatter 跳过" 章节 |

### 中风险

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| RSK3 | AllGatherMesh1D `CalcScratchMultiple` 依赖 `opMode_`，非 OPBASE 时返回 0 | **无风险** | 原始分析前提"Broadcast opMode 硬编码 OPBASE"不准确——Broadcast 有图模式入口 `HcclBroadcastGraphMode`（`broadcast_op.cc:55`）设 `OFFLOAD`。但即使非 OPBASE，风险也不触发：①`totalMult` 只取 ScatterL0 的 `CalcScratchMultiple`（=1，`ins_temp_scatter_mesh_1D.cc:86`，无条件返回，不检查 opMode）；②AGL0（ZAxisDetour）的 `CalcScratchMultiple` 从未被 3 级 executor 调用；③ZAxisDetour 的 `PostLocalCopy` opMode 门控（`Z_axis_detour.cc:280`）对 AGL0 无效——`inBuffType==HCCL_BUFFER` 已早返回（`ins_temp_all_gather_mesh_1D.cc:286`）。详见"图模式处理分析"章节 |
| RSK4 | `maxCountPerLoop` 缺少 `totalRankAlign` 对齐，R6 与 U1 公式不一致 | Low（已修复） | 已统一公式为 `maxCountPerLoop = cclMem.size / totalMult / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / totalRankAlign * totalRankAlign`（`totalRankAlign = N0*N1*N2`），与 AllReduce 3 级 `ins_v2_all_reduce_sequence_executor_aicpu_3level.cc:535-537` 一致。确保 `currDataCount` 可被各级 rank 数整除，`sliceSizeL0/L1/L2` 无余数。R6 与 U1 公式已统一 |
| RSK5 | Selector 缺少 level0 拓扑 fallback（CLOS 误入 sequence） | Low（已修复） | Selector 三级分支条件改为 `level0Topo == Level0Shape::MESH_1D && !level2Uboe` 才选 sequence；CLOS/MESH_1D_CLOS/UBOE 均回退 parallel executor。与需求 R15 / L262 缓解措施一致 |
| RSK6 | tailSize 跨级传播逻辑未明确 | Low（已修复） | 已补充完整的 tailSize 计算与传播方案，与 AllReduce 3 级模式完全一致：① ScatterL0 从 `currDataCount` 计算 `sliceSize/tailSize`；② ScatterL1/L2 根据 `rankIdxLevelX_ == N-1` 选上级 tail 或 slice 作为输入；③ AGL2/L1/L0 用对应 Scatter 步的 slice/tail；④ skipLevel1 传播与 AllReduce `line 582-586` 一致；⑤ 无需 AllReduce 的末轮 buffer 适配（单段复用 totalMult=1，全 buffer 可用）。详见 U1 "tailSize 跨级传播" 章节 |

### 低风险

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| RSK7 | 需求编号不连续（原 R1-R9 后跳到 R13） | Low（已修复） | 需求已重编号为 R1-R16 连续，R10-R12 已补齐（原 R13-R19 → R10-R16）。本风险消除 |
| RSK8 | 与现有 broadcast 2 级 sequence executor 模式 diverge | Low | 现有 2 级用 `REGISTER_EXECUTOR_BY_FOUR_TEMPS` + DPU 模板 + `SplitData`，新 3 级用 `REGISTER_EXEC_V2_MULTI` + AICPU 模板 + `sliceSize/stride`。两种模式共存增加维护成本。Plan Out of Scope 已隐含，建议显式注明 |
| RSK9 | Scatter 模板 SetRoot 接口与 executor 集成问题 | Low | 已确认 ScatterMesh1D（`ins_temp_scatter_mesh_1D.cc:22`）/ ScatterNHR（`ins_temp_scatter_nhr.cc:28`）有 SetRoot |
| RSK10 | 三级非对称拓扑不支持 | Low | TopoMatchMultilevel 三级非对称已拦截 |
| RSK11 | checker 不支持 broadcast 三级 | Low | CheckBroadcast 已支持 root 语义（`hccl_verifier.cc:67`），可调用验证 |
| RSK12 | ScatterNHR 缺 `sliceSize==0` 早返回，0 大小 NHR 传输 | Medium | `ScatterNHR::KernelRun`（`ins_temp_scatter_nhr.cc:156`）无 `sliceSize==0 && tailSize==0` 早返回（AllGatherNHR/AllGatherMesh1D 有）。`send1` 测试通过（N=2），但未测大 N。建议增加早返回保护 |
| RSK13 | AllGatherMesh1D 基类修复未被 AllGather 3 级测试覆盖 | Low（已验证） | AllGather 3 级用 ZAxisDetour 子类（override 了被修复的方法），不走基类路径。已运行全量 9 套件 162 测试回归，全部通过 |
| RSK14 | N0=1 未被 Broadcast 3 级测试覆盖 | Low | selector 通常在 `localNetInsSizeOfLayer[1]==1` 时选 NHR，N0=1 走不到 sequence executor。但理论上存在 N0=1 且不走 NHR 的拓扑 |
| RSK15 | Channel 仅从 Scatter 模板分配，AllGather channel 请求被丢弃 | Low-Medium | executor `CalcRes` 中 `channels[X]` 仅取 Scatter 的 `resReqScatterLX.channels[0]`。隐含假设 Scatter 和 AllGather 在同一 level 产生兼容的 channel 结构。当前安全（同 NHR 基础设施），但缺乏显式验证 |
| RSK16 | N2=1 时 3 级 executor 的 level2 channel 为空 | Low | `TopoMatchMultilevel` 拒绝非对称但不拒绝 N2=1。N2=1 时 `SetchannelsPerRank` 在空 channel 上失败。selector 应路由到 2 级 executor，但未显式保证 |
| RSK17 | 图模式未设 `enableRemoteMemAccess`，性能未优化 | Low | 3 级 executor 不设 `enableRemoteMemAccess`（始终 false），图模式下不使用 `remoteOutputGraphMode` 直接远程写入。功能正确但多一次 CCL 中转。AllReduce 3 级同样未实现，两者对称 |

### 风险处理优先级

1. **RSK1-RSK2, RSK4, RSK6（已修复）**——CCL Buffer offset、root 跳过、totalRankAlign、tailSize 传播均已补充完整设计
2. **RSK3（无风险）**——opMode 依赖被 totalMult 取 ScatterL0 + AGL0 用 ZAxisDetour 完全绕过
3. **RSK5（已修复）**——Selector 三级条件纳入 `level0Topo == MESH_1D` 检查，CLOS 回退 parallel
4. **RSK12（中风险）**——ScatterNHR 缺 sliceSize==0 早返回，建议增加保护
5. **RSK13（已验证）**——AllGatherMesh1D 基类修复全量回归 162 测试通过
6. **RSK7-RSK11, RSK14-RSK17（低风险）**——已验证或影响有限，可在实现过程中处理

### 图模式处理分析

> Broadcast 与 AllReduce 3 级在图模式处理上**完全对称**，无差异风险。

**图模式入口**：两者均有图模式 API（`HcclBroadcastGraphMode` / `HcclAllReduceGraphMode`），设 `opMode = OFFLOAD`。Selector 不检查 opMode，拓扑匹配即选 3 级 sequence。

**3 级 executor 的 opMode 处理**：

| 维度 | 说明 |
|------|------|
| Executor 检查 opMode | 否——AllReduce 和 Broadcast 3 级 executor 均不检查 |
| `enableRemoteMemAccess` | **两者均未设**——始终 `false`。Sole executor 在图模式下设 `enableRemoteMemAccess = (opMode==OFFLOAD)` 以使用 `remoteOutputGraphMode` 直接远程写入，3 级 executor 未实现此优化 |
| Buffer sizing | opMode 无关——`totalMult` 取 ScatterL0/ReduceScatterL0 的 `CalcScratchMultiple`（均无条件返回，不检查 opMode） |
| ZAxisDetour PostLocalCopy | `if (opMode==OPBASE)` 门控（`Z_axis_detour.cc:280`），图模式跳过。但对 AGL0 无效——`inBuffType==HCCL_BUFFER` 已早返回 |
| 资源复用 | 图模式不复用（`op_common.cc:1018`：非 OPBASE + 非 CCU → 不复用），每次全量 CalcRes。两者一致 |

**图模式功能正确性**：数据流正确（通过 CCL scratch 中转），只是路径非最优（未用 `remoteOutputGraphMode` 直接远程写入）。AllReduce 3 级也同样未实现此优化，两者对称。

**图模式与 RSK3 的关系**：RSK3 原始分析的前提"Broadcast opMode 硬编码 OPBASE"不准确。Broadcast 可为 OFFLOAD（图模式），但 RSK3 描述的风险（`CalcScratchMultiple` 返回 0 致 buffer 溢出）不触发——`totalMult` 不依赖 AGL0 的 `CalcScratchMultiple`。

## Deferred

- Reduce 三级 sequence（结构相同，另做）
- 非对称三级拓扑支持
- 图模式优化（`enableRemoteMemAccess` 直接远程写入）——AllReduce 3 级同样未实现，两者对称
- ScatterMesh1DZAxisDetour 新建（支持 CLOS 拓扑，当前 ScatterL0 仅支持 MESH_1D）
