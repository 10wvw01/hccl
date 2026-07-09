# Broadcast 3 级 Sequence Executor 实现修复记录

## 日期
2026-07-01

## 概述
为 Broadcast 操作新增 3 级 Sequence Executor（Scatter + AllGather 6 步串行），实现过程中遇到并修复了 5 个问题。

---

## 修复 1：预存 build 错误 — ThreadType/ThreadConfig 未定义

**文件**：`src/common/hcomm_dlsym/hccl_res_dl.h`

**现象**：编译 `hccl_compat` 目标时报错 `ThreadType has not been declared` / `ThreadConfig does not name a type`。

**根因**：`ThreadType` 和 `ThreadConfig` 在 `#if CANN_VERSION_NUM < CANN_VERSION(9, 1, 0)` 内定义（line 47-65），但 `HcclThreadAcquireWithConfig` 声明（line 84-85）在 `#if` 块外使用它们。安装的 toolkit 版本为 9.1.0（`CANN_VERSION_NUM = 91000000`），条件 `91000000 < 91000000` 为 false，定义被跳过。

**修复**：将版本 guard 从 `CANN_VERSION(9, 1, 0)` 改为 `CANN_VERSION(9, 2, 0)`，使 9.1.0 toolkit 进入 fallback 定义分支。

```diff
- #if CANN_VERSION_NUM < CANN_VERSION(9, 1, 0)
+ #if CANN_VERSION_NUM < CANN_VERSION(9, 2, 0)
```

---

## 修复 2：ST 链接错误 — 缺少 HcommIsSupport stub

**文件**：`test/st/algorithm/utils/src/hccl_proxy/hccl_stub.cc`

**现象**：ST 测试链接时报 `undefined reference to HcommIsSupportHcclAicpuKernelLaunch` 和 `HcommIsSupportHcclGroupStatusGet`。

**根因**：新构建的 `libhccl.so` 引用了这两个符号（由 `DECL_SUPPORT_FLAG` 宏生成，在 `hccl_host_comm_dl.h:51,56` 和 `hccl_device_comm_dl.h:25,30` 声明），但 ST 测试的 stub 库未提供对应实现。

**修复**：在 `hccl_stub.cc` 末尾补充两个 stub 函数，返回 `false`（表示不支持）：

```cpp
bool HcommIsSupportHcclAicpuKernelLaunch()
{
    return false;
}

bool HcommIsSupportHcclGroupStatusGet()
{
    return false;
}
```

---

## 修复 3：AGL0 inputSliceStride 错误 — rank 1 读 CCL 越界

**文件**：`src/ops/broadcast/executor/ins_v2_broadcast_sequence_executor_3level.cc`

**现象**：`st_broadcast_3level_2x2x2_fp32_root0`（count=200, FP32）失败，verifier 报 `Missing buffer semantics in tail: already checked total size is 400, which should be 800`。

**根因**：AGL0（AllGatherMesh1D）的 `inputSliceStride` 设为 `sliceSizeL0`（400），导致 `LocalDataCopy` 中 `inOff = inputSliceStride * myAlgRank + inBuffBaseOff = 400 * 1 + 0 = 400`。但 AGL1 输出后数据在 CCL[0..400]，rank 1 读 CCL[400] 无有效数据，只写入了一半数据到 OUTPUT。

**修复**：将 AGL0 的 `inputSliceStride` 从 `sliceSizeL0` 改为 `0`，与 AllReduce 3 级一致。AGL1 后所有 rank 的数据都在 CCL[0]，`inputSliceStride=0` 使所有 rank 从 CCL[0] 读取自己的分片。

```diff
- tempAlgParamsAGL0.inputSliceStride = tempAlgParamsAGL0.sliceSize;
+ tempAlgParamsAGL0.inputSliceStride = 0;
  tempAlgParamsAGL0.outputSliceStride = tempAlgParamsAGL0.sliceSize;
```

---

## 修复 4：AllGatherMesh1D 尾块处理 bug — LocalDataCopy CCL 拷贝重叠（+ AGL0 切 ZAxisDetour）

**文件**：`src/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.cc`（基类修复）+ `src/ops/broadcast/executor/ins_v2_broadcast_sequence_executor_3level.cc`（切换 ZAxisDetour）

**现象**：`st_broadcast_3level_2x2x2_fp32_send201`（count=201, FP32, N0=2）失败，verifier 报 `Slice is conflict in CCL[0, size=404] vs CCL[400, size=404]`。同样影响 `send501m+1`。

**根因：基类 `LocalDataCopy` 的 CCL 重分布拷贝自交叠（Bug B）**

基类 `InsTempAllGatherMesh1D` 是**纯 read 语义**：`RunAllGatherMesh` 只调 `SendRecvRead`（`ins_temp_all_gather_mesh_1D.cc:213`），每个 rank 从对端 CCL 的 strided 槽位 `CCL[sliceSize*connectedAlgRank]` 读数据（`rxScratchOffset = sliceSize*connectedAlgRank`）。因此每个 rank 必须先把自己的数据搬到自己 CCL 的 `CCL[sliceSize*myAlgRank]` 槽位、供对端来读——这就是 `LocalDataCopy` 的 step 2（`ins_temp_all_gather_mesh_1D.cc:258-270`）。

broadcast AGL0 场景 `inBuffType==HCCL_BUFFER`，`inputPtr == hcclBuff.addr`，每个 rank 自己的输入就在 `CCL[0]`（`inputSliceStride=0`，`inOff=0`）。末尾 rank（`myAlgRank=1`，`sliceSize=400`，`tailSize=404`）的 step 2：

```
srcSlice    = inputPtr@inOff   = CCL[0],    404B  →  CCL[0,   404)
cclDstSlice = hcclBuff@cclOff  = CCL[400],  404B  →  CCL[400, 804)   (cclOff = sliceSize*myAlgRank = 400)
                                                     ↑ 同一个 CCL buffer
重叠 [400, 404) → DMA 自交叠 → 数据损坏
```

verifier 报的 `CCL[0,size=404] vs CCL[400,size=404]` 即此两段。rank 0（`myAlgRank=0`）`inOff==cclOff==0` 命中 `skipCclCopy` 跳过，不出错——故只有尾块炸。注意这是**本地 step 2 自交叠**（同一 rank 在同一 CCL buffer 内 src/dst 重叠），并非与远端写入的冲突。

> **关于"tx/rx 共用 sliceSize"（原 Bug A 判断）**：经核实这**不是 active bug**。基类 `RunAllGatherMesh` 只调 `SendRecvRead`，而 `SendRecvRead`（`alg_data_trans_wrapper.cc:587-621`）**只读 `rxSlicesList_`、只下 `HcommReadOnThread`**，tx 切片整个被忽略。因此 `sliceSize` 对 tx 对不对无影响；对真正执行的 rx，`connectedAlgRank==last → tailSize` 本就正确。基类修复中分离 `txSliceSize`/`rxSliceSize` 在 read-only 路径上是**空操作**（tx 切片构造后即被丢弃），并非本用例失败的原因。本用例失败仅由 Bug B 造成。

**修复方式（两步）**：

**第一步：基类修复**（`ins_temp_all_gather_mesh_1D.cc`，兜底其他使用基类的 executor）：
- Bug B — 当 `inputPtr == hcclBuff.addr` 时，step 2 的源从 `input(CCL)` 改为 `OUTPUT`。因 step 1 刚把数据从 `input(CCL[0])` 拷到 `OUTPUT[outOff]`，OUTPUT 已持有一份干净副本，且 `outBuffType != HCCL_BUFFER` 保证 OUTPUT 与 CCL 是不同 buffer、永不重叠：
  ```cpp
  if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr) {
      DataSlice outSrcSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
      LocalCopy(threads[0], outSrcSlice, cclDstSlice);   // OUTPUT → CCL
  } else {
      LocalCopy(threads[0], srcSlice, cclDstSlice);       // input → CCL（原逻辑，input 与 CCL 不同 buffer 时无重叠）
  }
  ```
  数据路径：`CCL[0] →(step1)→ OUTPUT[outOff] →(step2)→ CCL[sliceSize*myAlgRank]`。
- Bug A — 分离 `txSliceSize`/`rxSliceSize`（分别按 `myAlgRank`/`connectedAlgRank` 判定）。保留为代码整洁/防御性修改，read-only 路径上无功能影响。

**第二步：Broadcast 3 级 AGL0 切换 ZAxisDetour 模板**：

AGL0（`InsAlgTemplate5`）经 `REGISTER_EXEC_V2_MULTI` 绑定为 `InsTempAllGatherMesh1D1DZAxisDetour`（`ins_v2_broadcast_sequence_executor_3level.cc:618`），与 AllReduce 3 级一致。ZAxisDetour override 了 `KernelRun`、`RunAllGatherMesh`、`LocalDataCopy` 三个方法，自带正确的 tail 处理，**不依赖基类修复**——broadcast 3 级实际生效的是这一步。

### 为什么 ZAxisDetour 不触发 Bug B（结构性消除）

ZAxisDetour 的 `LocalDataCopy`（`Z_axis_detour.cc:213-248`）**只有 input→OUTPUT 一步，没有 input→CCL 的重分布拷贝**。它能省掉这次拷贝，是因为其 rx 源偏移用 `inputSliceStride * connectedAlgRank`（`Z_axis_detour.cc:140`），与 `LocalDataCopy` 中 src 的 `inputSliceStride * myAlgRank`（`:231`）一致——直接从对端 input 所在位置读，不要求数据被搬到 `sliceSize*algRank` 槽位：

- broadcast AGL0 `inputSliceStride=0` → 对端数据就在对端 `CCL[0]`（即对端 input）→ rx 直接读 `CCL[0]`，无需任何重分布。
- 故 step 2 整个不存在，Bug B 从结构上不可能发生。

tail 处理上，`CalcSliceSizeForChannel`（`Z_axis_detour.cc:107-121`）按 `dmaRead` 区分：read 看 `connectedAlgRank==last`、write 看 `myAlgRank==last`。（注：broadcast AGL0 `dmaRead=true` → `SendRecvRead`，tx 同样是死代码，分离在此场景不生效；该分离仅对 ZAxisDetour 的 write 路径 `dmaRead=false` 有意义。）

**本质差异**

两者在 broadcast AGL0 下都是 read 语义（`dmaRead=true`，均走 `SendRecvRead`，无 tx/写远端）。差异在于"对端数据从哪里读"：

| | 基类 `AllGatherMesh1D` | ZAxisDetour |
|---|---|---|
| **rx 源偏移** | `sliceSize * connectedAlgRank`（要求数据在 strided 槽位） | `inputSliceStride * connectedAlgRank`（数据在 input 原位） |
| **是否需重分布到 CCL** | 需要（LocalDataCopy step 2：input→CCL[strided]） | 不需要（直接读 input 原位） |
| **LocalDataCopy** | 2 步：input→OUTPUT + input→CCL（step 2 在 input==CCL 时自交叠） | 1 步：input→OUTPUT（无 CCL 拷贝） |
| **CCL 冲突** | 尾块 step 2 src/dst 同 buffer 重叠（**本地自交叠**） | 无 CCL 拷贝，无冲突 |

数据路径（broadcast AGL0，read-only）：
- 基类：`input(CCL[0]) →(step1)→ OUTPUT[own槽]` + `input(CCL[0]) →(step2)→ CCL[strided]`（供对端 rx 读）；对端 `CCL[strided] →(rx)→ OUTPUT[peer槽]`
- ZAxisDetour：`input(CCL[0]) →(LocalDataCopy)→ OUTPUT[own槽]`；对端 `CCL[0] →(rx)→ OUTPUT[peer槽]`（直接读对端 input 原位）

> 基类无 tx/写远端、`KernelRun` 也不调 `PostLocalCopy`（`aa43a055` 已移除该调用，rx 直接写入 OUTPUT）。原记录中"→(tx)→ remote CCL →(rx)→ CCL scratch →(PostLocalCopy)→ OUTPUT"的描述不适用于当前 read-only 基类，已更正。

**基类修复保留**：基类的两处修复仍然保留，用于兜底其他仍使用基类 `InsTempAllGatherMesh1D` 的 executor（AllGather parallel/sole/2-level、AllReduce parallel、Reduce parallel、Broadcast parallel）。其中 **Bug B 修复（input==CCL 时从 OUTPUT 拷）对这些 caller 是真实功能修复**；Bug A 修复（分离 tx/rx sliceSize）在 read-only 基类上为空操作，仅作防御性整理。R9 回归验证确认 162 个测试全部通过。

---

## 修复 5：one_card_per_server 测试 — selector 路由到 NHR 而非 sequence

**文件**：`test/st/algorithm/testcase/broadcast_3level_testcase.cc`

**现象**：`st_broadcast_3level_2x4x1_fp32_one_card_per_server`（2×4×1 拓扑）失败，trace 显示仅 2 个 task/rank（executor 几乎未执行）。

**根因**：2×4×1 拓扑（1 card/server）中 `localNetInsSizeOfLayer[1] == 1`，selector 的第一个条件命中，选择 `InsBroadcastNHR`（扁平 NHR），而非我们的 `InsBroadcastSequenceMesh1DNHRNHR`。测试未覆盖目标 executor。

**修复**：将拓扑从 2×4×1 改为 2×4×2（2 cards/server），确保 `localNetInsSizeOfLayer[1] != 1`，selector 路由到 sequence executor：

```diff
- TEST_F(..., st_broadcast_3level_2x4x1_fp32_one_card_per_server) {
-     GenTopoMeta(topoMeta, 2, 4, 1);
+ TEST_F(..., st_broadcast_3level_2x4x2_fp32_multi_server) {
+     GenTopoMeta(topoMeta, 2, 4, 2);
```

---

## 最终测试结果

```
[==========] 11 tests from 1 test suite ran. (6322 ms total)
[  PASSED  ] 11 tests.
```

| # | 用例名 | 拓扑 | root | 类型 | 数据量 | 说明 |
|---|--------|------|------|------|--------|------|
| 1 | `st_broadcast_3level_2x2x2_int8_send1` | 2×2×2 | 0 | INT8 | 1 | 最小数据量 |
| 2 | `st_broadcast_3level_2x2x2_fp32_root0` | 2×2×2 | 0 | FP32 | 200 | 最小三级 |
| 3 | `st_broadcast_3level_4x2x2_int32_root0` | 4×2×2 | 0 | INT32 | 200 | 中等规模 |
| 4 | `st_broadcast_3level_2x4x4_fp16_root0` | 2×4×4 | 0 | FP16 | 300 | 多框多卡 |
| 5 | `st_broadcast_3level_2x2x2_int8_root3` | 2×2×2 | 3 | INT8 | 200 | root 非0框 |
| 6 | `st_broadcast_3level_2x2x2_bfp16_root4` | 2×2×2 | 4 | BFP16 | 200 | root 非0超节点 |
| 7 | `st_broadcast_3level_2x2x2_int64_root7` | 2×2×2 | 7 | INT64 | 200 | root 末尾 rank |
| 8 | `st_broadcast_3level_2x1x4_fp32_skip_l1` | 2×1×4 | 0 | FP32 | 200 | L1 退化 |
| 9 | `st_broadcast_3level_2x4x2_fp32_multi_server` | 2×4×2 | 0 | FP32 | 200 | 多 server |
| 10 | `st_broadcast_3level_2x2x2_fp32_send501m_plus_1` | 2×2×2 | 0 | FP32 | 501MB+1 | 大数据量 loop |
| 11 | `st_broadcast_3level_2x2x2_fp32_send201` | 2×2×2 | 0 | FP32 | 201 | 非对齐 |

---

## 修改文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/ops/broadcast/executor/ins_v2_broadcast_sequence_executor_3level.h` | 新增 | Executor 头文件 |
| `src/ops/broadcast/executor/ins_v2_broadcast_sequence_executor_3level.cc` | 新增 | Executor 实现 |
| `src/ops/broadcast/selector/broadcast_auto_selector.cc` | 修改 | 新增 3 级非 UBOE 分支 |
| `src/ops/broadcast/executor/CMakeLists.txt` | 修改 | 加新 .cc |
| `src/scatter_aicpu_kernel.cmake` | 修改 | 加新 .cc |
| `src/common/hcomm_dlsym/hccl_res_dl.h` | 修改 | 修复预存 build 错误（版本 guard） |
| `src/ops/all_gather/template/aicpu/ins_temp_all_gather_mesh_1D.cc` | 修改 | 修复 tail 处理 bug（CCL 拷贝避免重叠 + tx/rx 分离 sliceSize 防御性整理） |
| `test/st/algorithm/testcase/broadcast_3level_testcase.cc` | 新增 | 11 个 ST 测试 |
| `test/st/algorithm/testcase/CMakeLists.txt` | 修改 | 加新测试文件 |
| `test/st/algorithm/utils/src/hccl_proxy/hccl_stub.cc` | 修改 | 补充 2 个 stub 函数 |
