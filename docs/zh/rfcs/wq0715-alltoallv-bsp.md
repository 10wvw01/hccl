# wq0715 AllToAllV BSP 方案说明

本文说明 `wq0715` 分支当前实现的 AllToAllV BSP 优化方案。重点包括整体代码路径、每个模块的职责、BSP 调度公式、no-memcpy 远端 `rdispls` 处理方式，以及从 0622/0625 分支复用或对齐的内容。

## 1. 方案目标

`wq0715` 的目标是在 AllToAllV 场景下实现一条新的 BSP 跨 POD 调度路径：

1. 将通信域抽象为 `C x R` 的二维结构：
   - `C` 表示 POD 数量。
   - `R` 表示每个 POD 内的 rank 数量。
   - 全局 rank 编号满足 `rank = c * R + r`。
2. 对跨 POD 通信任务做轮次化编排：
   - `deltaC in {1, 2, ..., C - 1}` 表示目标 POD 偏移。
   - `deltaR in {0, 1, ..., R - 1}` 表示目标 POD 内的 rank 行偏移。
   - 每个 `deltaC` 是一个大轮次。
   - 每个大轮次中包含 `R` 个 slot，每个 slot 映射到一个 plane。
3. 当前 plane 选择规则为：
   - `plane = deltaR`
   - 因此要求每个 peer 至少有 `R` 条 channel/plane。
   - 这个约束和 0622 AllToAll no-memcpy CLOS V3 的思路一致：0622 中按 `rowNum` 做跨列 slot 编排，选择 channel 时也检查 `remoteChannels.size() >= rowNum`。
4. 数据传输采用 AllToAllV no-memcpy 方式：
   - 本 rank 直接写远端 rank 的 `output/recvBuf`。
   - 通过 0625 分支的 AllToAllV host exchange 机制拿到远端 `rdispls[myRank]` 和 `recvCounts[myRank]`。

这条路径不是 0625 的 AB/relay/stage1 方案，也不是 0622 的 AllToAll Mesh2D+CLOS V3 原方案。它复用了这些分支的部分基础能力，但核心调度是 `wq0715` 新增的 BSP 公式。

## 2. 入口和算法选择

入口文件：

- `src/ops/all_to_all_v/selector/alltoallv_auto_selector.cc`

新增环境变量开关：

```bash
HCCL_ENABLE_A2AV_BSP=1
```

该开关打开后，AllToAllV selector 会优先尝试选择 BSP 路径。

拓扑模式由已有环境变量控制：

```bash
HCCL_A2A_OPT_TOPO=pod_direct
HCCL_A2A_OPT_TOPO=pod_ubx_v2
```

当前选择关系如下：

| 环境变量 | 选择的算法名 | 拓扑匹配器 | 模板 |
| --- | --- | --- | --- |
| `HCCL_ENABLE_A2AV_BSP=1`, `HCCL_A2A_OPT_TOPO=pod_direct` | `InsAlltoAllVBspPodDirect` | `TopoMatchAlltoAllPodDirect` | `InsTempAlltoAllVBsp` |
| `HCCL_ENABLE_A2AV_BSP=1`, `HCCL_A2A_OPT_TOPO=pod_ubx_v2` | `InsAlltoAllVBsp` | `TopoMatchUBX_V2` | `InsTempAlltoAllVBsp` |
| `HCCL_ENABLE_A2AV_BSP=1`, 未设置 `HCCL_A2A_OPT_TOPO` | `InsAlltoAllVBsp` | `TopoMatchUBX_V2` | `InsTempAlltoAllVBsp` |

支持的 topo 条件：

- 当 `topoMode` 为 `pod_ubx_v2` 或 `pod_direct` 时，对齐 0625 的 pod 模式判断，允许：
  - `MESH_1D`
  - `CLOS`
  - `MESH_1D_CLOS && !level0PcieMix`
- 未设置 pod 模式时，仅允许：
  - `MESH_1D_CLOS && !level0PcieMix`

如果拓扑不满足上述条件，selector 会 fallback 到原有 AllToAllV 默认路径。

## 3. executor 注册关系

入口文件：

- `src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.cc`
- `src/ops/all_to_all_v/executor/ins_v2_all_to_all_v_sole_executor.h`

当前注册了两条 BSP AllToAllV 算法：

```cpp
REGISTER_EXEC_V2(HCCL_CMD_ALLTOALLV,
    InsAlltoAllVBsp,
    InsV2AlltoAllVSoleExecutor,
    TopoMatchUBX_V2,
    InsTempAlltoAllVBsp);

REGISTER_EXEC_V2(HCCL_CMD_ALLTOALLV,
    InsAlltoAllVBspPodDirect,
    InsV2AlltoAllVSoleExecutor,
    TopoMatchAlltoAllPodDirect,
    InsTempAlltoAllVBsp);
```

含义是：

- 算法名只决定走哪条注册路径。
- `InsV2AlltoAllVSoleExecutor` 仍然复用主线 AllToAllV executor 框架。
- `TopoMatchUBX_V2` 和 `TopoMatchAlltoAllPodDirect` 只负责生成拓扑层次信息。
- 最终执行模板都是 `InsTempAlltoAllVBsp`。

因此，BSP 调度模板和具体拓扑匹配器已经做了基本解耦。但这种解耦有前提：拓扑匹配器必须输出 BSP 所需的本 POD 分组信息，并且 rank 编号必须满足 `rank = c * R + r`。

## 4. 拓扑信息如何变成 BSP 的 C x R

核心函数：

- `BuildBspHierarchyInfo`

输入：

- `topoInfo`
- `algHierarchyInfo`

其中 `algHierarchyInfo` 是拓扑匹配器输出的层次信息。

当前实现逻辑：

1. 从 `algHierarchyInfo.infos[0]` 中查找包含当前 rank 的 group。
2. 如果有多个 group 包含当前 rank，选择 size 最小的 group。
3. 将这个 group 作为当前 rank 所在 POD：
   - `localPodRanks`
   - `R = localPodRanks.size()`
4. 要求：
   - `userRankSize % R == 0`
5. 构造完整全局 rank 列表：
   - `fullRanks = [0, 1, ..., userRankSize - 1]`
   - 这个列表不是从 AllToAllVC 拿的，也不是从用户的 `sendCounts/recvCounts` 推出来的，而是 executor 根据 `topoInfo->userRankSize` 自己顺序构造的。
   - 0625 的 AllToAllV AB 路径中也有同样风格的 `fullRanks` 构造：遍历 `0..userRankSize-1`，得到完整全局通信矩阵。
6. 传给模板的层次信息变成：
   - `{localPodRanks, fullRanks}`

模板侧在 `NormalizeSubCommRanks` 中解释这两个 group：

```cpp
rowNum = localPodRanks.size();  // R
rankNum = fullRanks.size();     // C * R
colNum = rankNum / rowNum;      // C
```

因此，BSP 模板并不直接依赖 `TopoMatchUBX_V2` 或 `TopoMatchAlltoAllPodDirect` 的内部细节。它只要求 executor 传入：

```text
group0 = 本 POD 内 rank 集合
group1 = 全局完整 rank 集合
```

这就是目前 pod_direct 和 UBX 共用同一套 BSP 模板的关键。

## 5. 资源申请逻辑

资源申请由模板 `InsTempAlltoAllVBsp::CalcRes` 完成。

入口文件：

- `src/ops/all_to_all_v/template/aicpu/ins_temp_all_to_all_v_bsp.cc`

主要逻辑：

1. 调用 `NormalizeSubCommRanks` 得到：
   - `rowNum_ = R`
   - `colNum_ = C`
   - `rankNum_ = C * R`
2. 申请 channel：
   - `MESH_1D_CLOS && !level0PcieMix` 时，使用 `CalcChannelRequestMesh1DWithPriorityTopo`，并筛选 `COMM_PROTOCOL_UBC_CTP`。
   - 其他情况使用 `CalcChannelRequestMesh1D`。
3. 申请线程和 notify：
   - BSP 需要 `2 * R` 个 slave thread。
   - 前 `R` 个线程用于 send。
   - 后 `R` 个线程用于 recv。
   - 每个线程当前使用 1 个 notify。

为什么是 `2 * R`：

- 每个 `deltaC` 轮次里有 `R` 个 `deltaR` slot。
- 每个 slot 绑定一个 plane。
- 当前实现把每个 plane 对应成一组 send thread + recv thread。

## 6. AllToAllV 变长数据和 loop 分片

这里的“切分”需要分清两类：

1. 算法层面的 A/B 数据切分：
   - `wq0715` 不需要复用 0625 的 A/B 切分、AB relay、AB inline 或 V2 stage1 方案。
   - BSP 方案只需要按 `deltaC/deltaR/plane` 对跨 POD 任务做编排。
2. executor 为控制单次传输大小做的 loop 分片：
   - 当前 `wq0715` 仍然保留了主线/0625 sole executor 的 per-loop 分片框架。
   - 这个分片不是 A/B 算法切分，只是把超大 `sendCounts/recvCounts` 按 `maxDataCountPerLoop` 分多轮处理。
   - 每个 loop 内仍然执行同一套 BSP 调度，只是本轮处理的是当前 chunk。

输入数组来自 AllToAllV 参数：

- `sendCounts`
- `recvCounts`
- `sdispls`
- `rdispls`

每轮循环处理的数据量由最大收发 count 决定：

```text
maxSendOrRecvDataCount = max(max(sendCounts), max(recvCounts))
maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize
loopTimes = ceil(maxSendOrRecvDataCount / maxDataCountPerLoop)
```

每个 loop 中，会为每个 rank 计算本轮实际处理的：

- `tempAlgParams.sendCounts[i]`
- `tempAlgParams.recvCounts[i]`
- `tempAlgParams.sdispls[i]`
- `tempAlgParams.rdispls[i]`

如果某个 rank 的数据已经在之前 loop 处理完，本轮 count 置 0，displ 固定到尾部。

BSP no-memcpy 路径还会额外填充：

- `tempAlgParams.remoteRdispls`
- `tempAlgParams.remoteRecvCounts`
- `tempAlgParams.enableRemoteMemAccess = true`

这些信息用于直接写远端 output。

如果后续确认目标场景的数据量永远不需要按 `UB_MAX_DATA_SIZE` 等传输上限分片，可以再简化成单 loop；但当前实现保留 loop 分片更稳，不影响 BSP 的任务编排逻辑。

## 7. 远端 rdispls 问题如何解决

AllToAllV 和 AllToAll 的核心区别是：

- AllToAll 每个 rank 给对端的数据大小和布局通常是规则的。
- AllToAllV 每个 rank 给每个对端的数据大小和位置都可能不同。

当 rank0 想直接写 rank1 的 `recvBuf/output` 时，rank0 必须知道：

```text
rank1 的 rdispls[rank0]
rank1 的 recvCounts[rank0]
```

否则 rank0 不知道自己这段数据应该写到 rank1 output 的哪个区间。

这个问题在 `wq0715` 中复用了 0625 分支的 AllToAllV no-memcpy exchange 机制。

相关文件：

- `src/ops/op_common/inc/alg_param.h`
- `src/ops/op_common/op_common.cc`
- `src/ops/op_common/template/template_utils.h`

### 7.1 Host exchange 数据结构

新增/复用结构：

```cpp
struct A2AVNoMemcpyExchangeInfo {
    OpExchangeInfo base;
    u32 magic;
    u32 version;
    u32 rankSize;
    u32 userRank;
    u64 totalSendCountWithoutSelf;
    u64 maxSendCountWithoutSelf;
    u64 sendCounts[64];
    u64 recvCounts[64];
    u64 sdispls[64];
    u64 rdispls[64];
};
```

当前限制：

```text
A2AV_EXCHANGE_MAX_RANK_SIZE = 64
```

也就是说，这套 no-memcpy exchange 当前最多支持 64 rank 的 AllToAllV 元信息交换。

### 7.2 什么时候触发 exchange

`op_common.cc` 中通过算法名判断：

```cpp
NeedAlltoAllVNoMemcpyExchange(param)
```

当前会触发的 BSP 算法名包括：

- `InsAlltoAllVBsp`
- `InsAlltoAllVBspPodDirect`

同时这两个算法名也加入了：

- `IsRemoteUserMemExperimentalAlg`
- `IsAlltoAllNoMemcpyAlg`

作用是：

1. 让 HCCL 注册/获取远端 user input/output 地址。
2. 让 HCCL 在建链前后交换 AllToAllV host 元信息。

### 7.3 exchange 的处理流程

在 `HcclGetChannelImpl` 中：

1. 如果是 BSP no-memcpy AllToAllV 算法：
   - 调用 `AddAlltoAllVNoMemcpyExchangeInfo`
   - 将本 rank 的 `sendCounts/recvCounts/sdispls/rdispls` 放入通信域 exchange 信息。
2. 调用 `HcclChannelAcquire` 建链。
3. 再次调用 `AddAlltoAllVNoMemcpyExchangeInfo`。
4. 对每个 remote rank 调用 `GetRemoteAlltoAllVInfo`。
5. 将对端信息填入 `ChannelInfo`：
   - `remoteAlltoAllVRdisplForLocalRank`
   - `remoteAlltoAllVRecvCountForLocalRank`
   - `remoteAlltoAllVRecvCounts`
   - `remoteAlltoAllVRdispls`

executor 的 `FillBspAlltoAllVRemoteInfo` 再把这些信息转成模板参数：

```text
tempAlgParams.remoteRdispls[remoteRank]
tempAlgParams.remoteRecvCounts[remoteRank]
```

模板真正发送时使用：

```text
remote output address + remoteRdispls[txRank] * dataTypeSize
```

这就解决了“发送端不知道远端 rdispls[myRank]”的问题。

## 8. BSP 调度公式

模板执行入口：

- `InsTempAlltoAllVBsp::KernelRun`

先做本 rank 自拷贝：

```text
input[sdispls[myRank]] -> output[rdispls[myRank]]
```

然后执行跨 POD 调度：

```cpp
for deltaC in [1, C - 1]:
    CalcBspRoundPlan(deltaC)
    for each deltaR in [0, R - 1]:
        RunBspSlot(...)
```

对当前 rank：

```text
myCol = myRank / R
myRow = myRank % R
```

发送目标：

```text
txCol = (myCol + deltaC) % C
txRow = (myRow + deltaR) % R
txRank = txCol * R + txRow
```

接收来源：

```text
rxCol = (myCol + C - deltaC) % C
rxRow = (myRow + R - deltaR) % R
rxRank = rxCol * R + rxRow
```

plane 选择：

```text
plane = SelectPlane(deltaC, deltaR) = deltaR
```

因此每个 rank 的跨 POD任务数是：

```text
(C - 1) * R
```

每个 `deltaC` 大轮次有 `R` 个任务，对应 `R` 个 plane。

## 9. 收发不对称如何实现

BSP slot 里同时计算：

- `txRank`：当前 rank 本 slot 要发送到的 rank。
- `rxRank`：当前 rank 本 slot 要接收来自的 rank。

这两个 rank 不要求相同。

代码逻辑：

1. 如果 `txRank == rxRank`，并且收发数据都非空：
   - 走 `SendRecvWrite`
   - 这相当于复用同一个 peer/channel 的收发 batch。
2. 否则：
   - 如果 `rxSize > 0`，走 `RecvWrite`
   - 如果 `txSize > 0`，走 `SendBatchWrite`

因此，当前 BSP 路径已经支持收发不对称：

```text
本 rank 给 rank2 发数据的同时，可以接收来自 rank3 的数据。
```

这部分实现思路对齐了 0622 AllToAll Mesh2D+CLOS V3 中“同 peer/channel 可 batch，否则拆 send/recv thread”的模式，但具体的 AllToAllV BSP 排程公式是 `wq0715` 自己实现的。

## 10. no-memcpy 写远端 output

发送侧在 `RunBspSlot` 中构造：

```text
src = local input + sdispls[txRank] * dataTypeSize
dst = txChannel.remoteOutputGraphMode.addr + remoteRdispls[txRank] * dataTypeSize
size = sendCounts[txRank] * dataTypeSize
```

其中：

- `local input` 是当前 rank 的 sendBuf。
- `remoteOutputGraphMode.addr` 是远端 rank 的 output/recvBuf 地址。
- `remoteRdispls[txRank]` 实际表示远端 rank 中给当前 rank 预留的接收偏移。

代码中还做了保护：

1. 必须开启 `enableRemoteMemAccess`。
2. 远端 output 地址不能为空。
3. `remoteRecvCounts[txRank]` 与本地 `sendCounts[txRank]` 在语义上应一致。
   - 这表示：如果当前 rank 给 `txRank` 发送 100 MB，那么 `txRank` 的 `recvCounts[当前rank]` 应该也是 100 MB。
   - 它不是要求 `txRank` 同时也给当前 rank 发送 100 MB；反向发送大小由 `txRank` 自己的 `sendCounts[当前rank]` 决定。
   - 当前代码在 per-loop 参数里会把 `remoteRecvCounts[i]` 调整成本轮 chunk 的发送大小，所以这个字段更多用于本轮长度一致性表达；真正解决远端写入位置的是 `remoteRdispls[i]`。
4. 本地 input 范围不能越界。
5. 远端 output 范围不能越界。
6. 本地 output 接收范围不能越界。

## 11. pod_direct 变体

pod_direct 变体新增的是算法入口和拓扑匹配器组合：

```text
InsAlltoAllVBspPodDirect
    -> TopoMatchAlltoAllPodDirect
    -> InsTempAlltoAllVBsp
```

`TopoMatchAlltoAllPodDirect` 的职责是：

1. 在 layer0 找当前 rank 所属的本 POD。
2. 在 layer1 找当前 rank 到非本 POD 的直连 peer。
3. 输出 `algHierarchyInfo` 给 executor。

executor 不直接使用 layer1 peer 列表来构造完整矩阵，而是：

- 用 `infos[0]` 识别本 POD 的 `R`。
- 自己构造 `[0..rankSize-1]` 作为完整 rank 矩阵。

因此 pod_direct 和 UBX 的差别主要在 topology matcher：

- UBX 路径：`TopoMatchUBX_V2`
- pod_direct 路径：`TopoMatchAlltoAllPodDirect`

后续执行模板完全一致。

## 12. 和 UBX 是否完全解耦

不能说完全解耦。

更准确的说法是：

```text
BSP 调度模板已经不绑定某一个具体 topo matcher，
但仍然依赖 topo matcher 输出满足 BSP 契约。
```

这个契约包括：

1. `algHierarchyInfo.infos[0]` 中必须能找到当前 rank 所属的本 POD group。
2. 本 POD group 的大小就是 `R`。
3. `userRankSize % R == 0`，这样才能得到 `C = userRankSize / R`。
4. 全局 rank 编号必须满足 `rank = c * R + r`。
5. 每个跨 POD peer 必须能申请到 channel。
6. 每个 peer 的 channel 数量必须至少为 `R`。
7. channel vector 的下标要和 plane 编号一致，因为当前直接使用 `channels[remoteRank][plane]`。

如果未来要接入新的 UBX 识别方式，通常只需要新增或替换 topology matcher，但它必须满足上述契约。

## 13. 复用了哪些分支能力

### 13.1 复用 master 的基础框架

复用内容：

- AllToAllV selector 框架。
- `InsV2AlltoAllVSoleExecutor` 的基础执行框架。
- AllToAllV 参数读取和 loop 切分方式。
- channel 申请和 `TemplateResource` 传递机制。
- AICPU template 的 `KernelRun/CalcRes` 框架。
- `SendBatchWrite/RecvWrite/SendRecvWrite/LocalCopy` 等数据传输封装。

`wq0715` 没有重写整个 AllToAllV 执行系统，而是在现有 executor/template 体系中增加了一条新算法。

### 13.2 复用 0625 的 AllToAllV no-memcpy exchange

0625 里解决了 AllToAllV no-memcpy 的关键问题：

```text
发送端如何知道远端 rdispls[myRank]
```

`wq0715` 复用了这套机制，包括：

- `A2AVNoMemcpyExchangeInfo`
- `A2AV_EXCHANGE_MAX_RANK_SIZE`
- `NeedAlltoAllVNoMemcpyExchange`
- `FillA2AVNoMemcpyExchangeInfo`
- `AddAlltoAllVNoMemcpyExchangeInfo`
- `GetRemoteAlltoAllVInfo`
- `ChannelInfo` 中的远端 AllToAllV 元信息字段
- `TemplateDataParams` 中的 `remoteRdispls/remoteRecvCounts`

这部分是 `wq0715` 能做 AllToAllV 直接写远端 output 的基础。

### 13.3 对齐 0625 的 pod topo 选择方式

`wq0715` 对齐了 0625 中 `HCCL_A2A_OPT_TOPO` 的使用方式：

- `pod_ubx_v2`
- `pod_direct`

并且 pod 模式下支持 `MESH_1D/CLOS/MESH_1D_CLOS` 的判断方式也参考了 0625。

但需要注意：

- 0625 的 AllToAllV 主要是 AB/no-memcpy/relay/stage1 等路径。
- `wq0715` 没有照搬这些调度策略。
- `wq0715` 只复用了它的 AllToAllV 元信息交换能力和拓扑模式入口风格。

### 13.4 复用/对齐 0622 的 pod_direct 和收发拆分思想

0622 中有 AllToAll Mesh2D+CLOS V3、pod_direct、no-memcpy 等优化路径。

`wq0715` 对齐或复用了其中两个方面：

1. pod_direct 拓扑匹配器：
   - `TopoMatchAlltoAllPodDirect`
2. 收发不对称执行思想：
   - 同 peer/channel 可走 `SendRecvWrite`。
   - 否则拆成 recv thread 和 send thread。

但 `wq0715` 的通信任务排序不是 0622 的 Mesh2D+CLOS V3 排程，而是 BSP 的：

```text
deltaC round + deltaR plane mapping
```

## 14. 当前与 0622/0625 的主要区别

### 14.1 与 0622 的区别

0622 主要针对 AllToAll：

- Mesh2D + CLOS 并行优化。
- V3 no-memcpy。
- pod_ubx_v2/pod_direct。

`wq0715` 针对 AllToAllV：

- 变长 `sendCounts/recvCounts/sdispls/rdispls`。
- 需要解决远端 `rdispls[myRank]`。
- 调度公式是 BSP，而不是 0622 原本的 Mesh2D+CLOS V3 映射。

### 14.2 与 0625 的区别

0625 主要是 AllToAllV 的多种 no-memcpy/AB/relay/stage1 方案。

`wq0715`：

- 没走 AB 分流。
- 没走 relay 两阶段 CCL buffer。
- 没走 V2 Stage1 no-memcpy。
- 直接使用 BSP 跨 POD 轮次化调度。
- 直接写远端 output。
- 仅复用 0625 的 AllToAllV 元信息 exchange 能力。

## 15. 当前限制和风险点

### 15.1 rank 编号假设

BSP 公式要求：

```text
rank = c * R + r
```

如果真实 rank 编号不是按 POD 连续排列，当前 `txRank/rxRank` 计算会错。

这一点不是 `wq0715` 独有的假设。0622/0625 的相关 pod/UBX 代码中也能看到同类假设：

- `TopoMatchUBX_V2` 用 `rankId / layer0Size == myRank / layer0Size` 判断是否属于同一 POD。
- `TopoMatchAlltoAllPodDirect` 用 `rankId / localPodSize == myRank / localPodSize` 判断是否属于同一 POD。
- 0622 AllToAll no-memcpy CLOS V3 用 `myRow = myRank % rowNum`、`myCol = myRank / rowNum`、`txRank = txCol * rowNum + peerRow` 做二维映射。

因此从代码层面看，0622/0625 也默认目标环境的 rank 编号按 POD 连续排列。真实环境是否一定如此，取决于 rank table 和 topo graph 的生成方式；这些分支的实现是按“POD 内 rank 连续编号”这个前提写的。

### 15.2 R 的识别依赖 layer0 group

当前通过 `algHierarchyInfo.infos[0]` 找本 POD group。

如果 topo matcher 的 layer0 不是严格 POD 分组，或者存在异常重叠分组，`R` 可能识别错误。

### 15.3 要求 P >= R

当前：

```text
plane = deltaR
```

并且 channel 选择为：

```text
channels[remoteRank][plane]
```

因此每个 peer 至少要有 `R` 条 channel/plane。

### 15.4 exchange rank 数上限

当前：

```text
A2AV_EXCHANGE_MAX_RANK_SIZE = 64
```

超过 64 rank 的 AllToAllV no-memcpy 元信息交换暂不支持。

### 15.5 不支持 pcie/read protocol

`KernelRun` 中检查：

```text
isDmaRead_ == true -> HCCL_E_NOT_SUPPORT
```

也就是说当前 BSP 模板只面向写远端 output 的路径。

### 15.6 plane 映射还没有树形寻优

当前 `SelectPlane(deltaC, deltaR)` 很简单：

```text
return deltaR
```

后续如果要加入“plane 与行偏移的树形寻优编排策略”，最自然的位置就是扩展 `SelectPlane`：

```text
plane = f(deltaC, deltaR)
```

或者在每个 `deltaC` 轮次前生成一个 `deltaR -> plane` 的映射表。

需要保证每个轮次内映射无冲突，并且 plane 下标不能超过实际 channel 数。

## 16. 当前验证情况

当前已做的检查：

- `git diff --check` 已通过。
- 新增 BSP 文件做了 no-index whitespace check，没有空白错误输出。

尚未完成：

- 未跑完整编译。
- 未在真实多 POD 环境验证 channel/plane 数量是否满足 `P >= R`。
- 未验证真实 rank 编号是否满足 `rank = c * R + r`。
- 未验证 pod_direct 和 UBX 两种 matcher 在目标环境下输出的 layer0 POD group 是否完全符合 BSP 契约。

## 17. 总结

`wq0715` 当前形成了一条新的 AllToAllV BSP no-memcpy 路径：

```text
selector
  -> InsAlltoAllVBsp / InsAlltoAllVBspPodDirect
  -> topology matcher
  -> BuildBspHierarchyInfo
  -> InsTempAlltoAllVBsp
  -> deltaC/deltaR/plane BSP 调度
  -> 直接写远端 output
```

它的核心新增点是：

1. AllToAllV BSP 调度模板。
2. 通用 `C x R` POD 建模，不再固定 4 x 4。
3. `deltaC` 轮次和 `deltaR -> plane` 映射。
4. 收发不对称 slot 执行。
5. pod_direct 变体入口。
6. 复用 0625 的远端 AllToAllV 元信息 exchange，解决 no-memcpy 写远端 output 的 offset 问题。

后续如果要加入树形寻优，建议优先改造 `SelectPlane(deltaC, deltaR)`，保持其余 BSP 数据流不变。
