# Broadcast 3 级 vs AllReduce 3 级全流程调用差异与风险分析

## 日期
2026-07-02

## 分析方法
逐行对比 Broadcast 与 AllReduce 3 级 Sequence Executor 的完整调用链（从用户 API 到 OrchestrateLoop），识别所有差异点并评估风险。

---

## 调用链总览

```
用户 API
  │
  ├─ Broadcast: HcclBroadcast(buf, count, type, root, comm, stream)
  │   └─ BroadcastInitAndCheck → BroadcastOutPlace
  │       └─ OpParam: inputPtr=outputPtr=buf, root=root, opMode=OPBASE(硬编码)
  │
  └─ AllReduce: HcclAllReduce(sendBuf, recvBuf, count, type, op, comm, stream)
      └─ AllReduceInitAndCheck → AllReduceOutPlace → FillAllReduceOpParam
          └─ OpParam: inputPtr=sendBuf, outputPtr=recvBuf, reduceType=op, opMode=参数化
              │
              ▼
      Selector (按 opType 分发)
      ├─ BroadcastAutoSelector::SelectAicpuAlgo — 仅拓扑判断
      └─ AllReduceAutoSelector::SelectAicpuAlgo — 5 重门控 + level0Topo==MESH_1D
              │
              ▼
      HcclExecOp → GetAlgExec(opType, algName) → executor
              │
              ▼
      CalcAlgHierarchyInfo (TopoMatchMultilevel, 相同)
      → CalcRes (模板不同)
      → Orchestrate → OrchestrateLoop (核心差异)
```

---

## 9 个差异点

| # | 差异点 | AllReduce | Broadcast | 风险 |
|---|--------|-----------|-----------|------|
| D1 | **API/OpParam** | `inputPtr≠outputPtr`（独立 buffer） | `inputPtr==outputPtr`（in-place） | R1（低） |
| D2 | **root 字段** | 不设（`INVALID_VALUE_RANKID`） | 设为 `param.root` | 无 |
| D3 | **reduceType** | 设为 `param.reduceType` | 不设（`HCCL_REDUCE_RESERVED`） | R2（低） |
| D4 | **opMode** | 参数化（OPBASE 或 OFFLOAD） | 硬编码 `OPBASE` | R3（低） |
| D5 | **Selector 门控** | 5 重门控 + `level0Topo==MESH_1D` | 仅拓扑判断 | **R4（中）** |
| D6 | **L0 模板** | `ReduceScatterMesh1DZAxisDetour` / `AllGatherMesh1D1DZAxisDetour` | `ScatterMesh1D` / `AllGatherMesh1D` | **R4（中）** |
| D7 | **Buffer 模型** | 双段（rsResult + meshComm） | 单段复用 | R5（低） |
| D8 | **OrchestrateLoop 参与** | 全 rank 无条件执行 | Scatter 条件跳过（root 框/超节点） | R6（低） |
| D9 | **末轮 tail 适配** | 有（`tailSize > rsResultBuffSize_` 缩减 currDataCount） | 无 | R7（低） |

---

## 风险详细分析

### R1：in-place buffer 影响 AllGatherMesh1D LocalDataCopy（低风险）

**差异**：Broadcast `inputPtr==outputPtr`（同一 buffer），AllReduce `inputPtr≠outputPtr`。

**影响**：AllGatherMesh1D::LocalDataCopy step 1 从 `inputPtr` 拷贝到 `outputPtr`。当两者相同且 offset 相同时，`skipOutCopy=true` 跳过拷贝。当 offset 不同时（如 AGL0 的 `inOff=0, outOff=rankIdx0*sliceSize`），仍执行拷贝，但源和目标是同一 buffer 的不同区域——这是合法的（不重叠时）。

**评估**：AllReduce 也可能有 in-place 场景（`sendBuf==recvBuf`），模板已处理。低风险。

### R2：reduceType 未设（低风险）

**差异**：Broadcast 不设 `reduceType`（`HCCL_REDUCE_RESERVED`），AllReduce 设为用户 `op`。

**影响**：Scatter 模板（`ScatterMesh1D`/`ScatterNHR`）不使用 `reduceOp_`，只有 `ReduceScatterNHR` 使用。Broadcast 的模板构造函数 `InsAlgTemplateBase(param, rankId, subCommRanks)` 会设 `reduceOp_ = param.reduceType`（`alg_v2_template_base.cc:18`），但 Scatter 模板的 `KernelRun` 不读 `reduceOp_`。

**评估**：Scatter 模板不依赖 `reduceType`，无风险。

### R3：opMode 硬编码 OPBASE（低风险）

**差异**：Broadcast `opMode` 硬编码 `OPBASE`，AllReduce 参数化。

**影响**：`AllGatherMesh1D::CalcScratchMultiple` 在非 OPBASE 时返回 0。Broadcast 的 `totalMult = ScatterL0->CalcScratchMultiple()` = 1（ScatterMesh1D 无 opMode 依赖），绕过了此问题。

**评估**：Broadcast 的 `opMode` 始终是 OPBASE，且 `totalMult` 取 ScatterL0（不依赖 opMode），无风险。

### R4：Selector 无 level0 拓扑检查 + plain 模板不支持 CLOS（中风险）

**差异**：
- AllReduce selector 要求 `level0Topo == MESH_1D` 才选 3 级 sequence（`all_reduce_auto_selector.cc:371`）
- Broadcast selector 仅检查 `topoLevelNums==3 && !level2Uboe`（`broadcast_auto_selector.cc:128-134`），不检查 `level0Topo`
- AllReduce L0 用 `ZAxisDetour` 变体（override `CalcRes`，支持 CLOS），Broadcast L0 用 plain 变体（仅 MESH_1D）

**影响**：当 level0 是 CLOS/MESH_1D_CLOS 拓扑时：
- `ScatterMesh1D::CalcRes` 调 `CalcChannelRequestMesh1D`，可能无法正确计算 CLOS channel
- `AllGatherMesh1D::CalcRes` 同样仅支持 MESH_1D
- AllReduce 的 `ZAxisDetour` 变体 override 了 `CalcRes`，支持 CLOS

**评估**：**中风险**。CLOS/MESH_1D_CLOS level0 拓扑下可能 channel 计算失败。需在 selector 增加 `level0Topo == MESH_1D` 检查。

### R5：单段 vs 双段 buffer（低风险）

**差异**：AllReduce 双段（`rsResultBuffOffset_=0, meshCommBuffOffset_=rsResultBuffSize_`），Broadcast 单段（全 buffer）。

**影响**：AllReduce 的 `maxCountPerLoop = meshCommBuffSize / ...`（仅 meshComm 区），Broadcast 的 `maxCountPerLoop = cclMem.size / 1 / ...`（全 buffer）。Broadcast 的 maxCountPerLoop 更大，loop 次数更少。

**评估**：已验证 AGL0 最大需求 `N0 * sliceSizeL0 = currDataCount * dts ≤ cclMem.size`，不溢出。低风险。

### R6：Scatter 条件跳过（低风险，已验证）

**差异**：AllReduce 全 rank 无条件执行所有步骤；Broadcast 的 ScatterL0/L1 条件跳过。

**影响**：跳过的 rank 不执行 `KernelRun`，但仍调 `GenTempAlgParamsXxx` 计算 sliceSize/tailSize 供后续步骤使用。跳过的 rank CCL buffer 为空，但 NHR 通信中仅接收（非 root 不发送）。

**评估**：已通过 2×2×2 全 8 rank 验证、数据流完整性分析确认正确。低风险。

### R7：无末轮 tail 适配（低风险）

**差异**：AllReduce 有末轮 `currDataCount` 缩减（`tailSize > rsResultBuffSize_` 时），Broadcast 无。

**影响**：AllReduce 的缩减是为了防止 tail 超出 rsResultBuffSize_（双段限制）。Broadcast 单段全 buffer 可用，最大需求 = buffer 大小，无需缩减。

**评估**：低风险。但需注意 `currDataCount < N0` 的极端 case（如 `send1` 用例），此时 `sliceSizeL0=0, tailSizeL0=dts`。已通过 `send1` 测试验证。

---

## 风险汇总

| 风险 | 级别 | 差异点 | 状态 |
|------|------|--------|------|
| **R4：CLOS 拓扑不支持** | **中** | D5+D6 | **待修复**（selector 加 `level0Topo==MESH_1D` 检查） |
| R1：in-place buffer | 低 | D1 | 已验证正确 |
| R2：reduceType 未设 | 低 | D3 | Scatter 模板不依赖 |
| R3：opMode 硬编码 | 低 | D4 | totalMult 绕过 opMode 依赖 |
| R5：单段 buffer | 低 | D7 | 已验证不溢出 |
| R6：Scatter 条件跳过 | 低 | D8 | 已验证正确 |
| R7：无末轮 tail 适配 | 低 | D9 | 单段 buffer 无需适配 |

---

## 结论

全流程调用链中，从 API 入口到 Selector 到 Executor 的路径是共享代码参数化分发，差异集中在 Selector 门控逻辑（D5）和 Executor 实现（D6-D9）。唯一中风险是 R4——Broadcast selector 缺少 `level0Topo==MESH_1D` 检查，且 plain 模板不支持 CLOS 拓扑。其余差异均为 Broadcast 语义所需，已通过 ST 测试验证正确。

---

## 补充分析：遗漏风险（第二轮）

对模板内部行为、边界 case、共享模板修改影响范围进行深入分析，发现以下遗漏风险：

### R8：ScatterNHR 缺少 sliceSize==0 早返回（中风险）

**发现**：`ScatterNHR::KernelRun`（`ins_temp_scatter_nhr.cc:156-198`）没有 `sliceSize==0 && tailSize==0` 的早返回，而 `AllGatherNHR::KernelRun:95-98` 和 `AllGatherMesh1D::KernelRun:66-69` 都有。

**影响**：当 `currDataCount < N0`（如 count=1, N0=2）时，非末尾 rank 的 `sliceSize=0, tailSize=0`。ScatterNHR 仍执行完整 NHR 步骤（`GetNHRStepNum(N)` 步），产生 0 大小的 SendWrite/SendRead 操作。

**评估**：`send1` 测试（count=1, N0=2）通过，确认 0 大小传输在 N=2 时是 no-op。但未测试大 N（如 N0=4, N1=4, count=1）场景。如果传输层拒绝 0 大小操作，会失败。建议为 ScatterNHR 增加早返回保护。

### R9：AllGatherMesh1D 修复未被 AllGather 3 级测试覆盖（中风险）

**发现**：AllGather 3 级使用 `InsTempAllGatherMesh1D1DZAxisDetour`（`ins_v2_all_gather_sequence_executor_3level.cc:328`），该子类 **override 了 `KernelRun`、`RunAllGatherMesh`、`LocalDataCopy`**（`Z_axis_detour.h:30-42`）。因此 AllGather 3 级测试不走基类的修复路径。

**影响**：基类 `InsTempAllGatherMesh1D` 的修复（tx/rx 分离 sliceSize + LocalDataCopy 避免重叠）仅被 Broadcast 3 级测试覆盖。其他使用基类的 executor（Broadcast parallel、AllGather parallel/sole/2-level、AllReduce parallel、Reduce parallel）的测试未验证此修复。

**评估**：修复是向后兼容的（tailSize==0 时行为不变）。但建议运行全量 ST 回归确认无副作用。

### R10：N0=1 未被 Broadcast 3 级测试覆盖（低风险，测试覆盖缺口）

**发现**：所有 Broadcast 3 级测试的 N0≥2。N0=1 时：
- `ScatterMesh1D` 无 `templateRankSize_==1` 早返回，但 `RunMesh` 循环跳过自身 rank（`:224-226`），PreCopy 正确拷贝 INPUT→CCL
- `AllGatherMesh1D` 有 `templateRankSize_==1` 早返回（`:80-82`），LocalDataCopy 已先执行，数据正确到 OUTPUT

**评估**：数据流正确，但缺少测试覆盖。selector 在 `localNetInsSizeOfLayer[1]==1` 时选 NHR 而非 sequence executor，所以 N0=1 通常走不到我们的 executor。但理论上存在 N0=1 且 `localNetInsSizeOfLayer[1]!=1` 的拓扑。

### R11：Channel 仅从 Scatter 模板分配，AllGather channel 请求被丢弃（低-中风险）

**发现**：`CalcRes` 中 `resourceRequest.channels[X]` 仅取 Scatter 模板的 `resReqScatterLX.channels[0]`（`:138,144,150`），AllGather 模板的 channel 请求被丢弃。

**影响**：隐含假设 Scatter 和 AllGather 在同一 level 产生相同 channel 结构（都调 `CalcChannelRequestNhr`）。`SetchannelsPerRank` 在 OrchestrateLoop 中用同一 channel map 设置所有模板。如果两者 channel 结构不同，`GetRes()` 返回的 `slaveThreadNum` 可能与 `CalcRes` 时不一致。

**评估**：当前两者使用相同 NHR channel 基础设施，实际安全。但缺乏显式验证，脆弱。

### R12：N2=1 时 3 级 executor 的 level2 channel 为空（低风险）

**发现**：`TopoMatchMultilevel` 对 3 级拓扑拒绝非对称（`topo_match_multilevel.cc:273-278`），但未拒绝 N2=1。N2=1 时 level2 组只有 1 个 rank，`SetchannelsPerRank` 在空 channel 上失败。

**评估**：selector 应在 `topoLevelNums==3` 但实际 N2=1 时路由到 2 级 executor。但未显式保证。无测试覆盖。

### R13：LocalDataCopy 修复的潜在边界 case（低风险，latent）

**发现**：修复在 `inputPtr==hcclBuff.addr` 时从 OUTPUT 拷贝到 CCL。如果 `outputPtr==hcclBuff.addr`（即 inputPtr==outputPtr==hcclBuff.addr），且 `skipOutCopy=true`（inOff==outOff），则 OUTPUT 也未写入，从 OUTPUT 拷贝到 CCL 会拷贝 stale 数据。

**评估**：当前 Broadcast AGL0 中 `outputPtr=param.outputPtr`（用户 buffer）≠ `hcclBuff.addr`（CCL），不触发此 case。但其他 executor 若配置 in-place CCL（inputPtr==outputPtr==hcclBuff.addr），可能触发。latent 风险。

---

## 更新后的风险汇总

| 风险 | 级别 | 差异点/来源 | 状态 |
|------|------|------------|------|
| **R4：CLOS 拓扑不支持** | **中** | D5+D6 | 待修复（selector 加 level0Topo 检查） |
| **R8：ScatterNHR 缺 sliceSize==0 早返回** | **中** | 模板内部 | 建议增加早返回保护 |
| **R9：AllGatherMesh1D 修复未被全量回归** | **中** | 共享模板修改 | 建议运行全量 ST 回归 |
| R1：in-place buffer | 低 | D1 | 已验证正确 |
| R2：reduceType 未设 | 低 | D3 | Scatter 模板不依赖 |
| R3：opMode 硬编码 | 低 | D4 | totalMult 绕过 |
| R5：单段 buffer | 低 | D7 | 已验证不溢出 |
| R6：Scatter 条件跳过 | 低 | D8 | 已验证正确 |
| R7：无末轮 tail 适配 | 低 | D9 | 单段 buffer 无需 |
| R10：N0=1 未测试 | 低 | 测试覆盖缺口 | selector 通常路由到 NHR |
| R11：Channel 仅从 Scatter 分配 | 低-中 | CalcRes | 当前安全，缺乏显式验证 |
| R12：N2=1 level2 channel 为空 | 低 | 边界 case | selector 应路由到 2 级 |
| R13：LocalDataCopy 修复 latent case | 低 | 共享模板修改 | 当前不触发 |

**可执行建议**：
1. **R4**：selector 增加 `level0Topo == MESH_1D` 检查
2. **R8**：ScatterNHR::KernelRun 增加 `sliceSize==0 && tailSize==0` 早返回
3. **R9**：运行 AllGather/AllReduce/Broadcast parallel + sole 的全量 ST 回归
