# topo_host.cc 与 topo_match_multilevel.cc 分析

## topo_host.cc 的作用

`topo_host.cc` 是**拓扑信息采集与计算层**，负责从底层 HCOMM/RankGraph API 获取原始拓扑数据，解析并填充到 `TopoInfo` / `TopoInfoWithNetLayerDetails` 结构体中。它回答的是"**这个集群长什么样**"。

### 1. 基础拓扑信息采集（填充 TopoInfo）

| 函数 | 作用 |
|------|------|
| `InitRankInfo(comm, TopoInfo*)` | 入口函数，计算本 rank 的身份信息 |
| `CalcMyRankInfo` | 获取 rankId、rankSize、deviceType、serverIdx、superPodIdx |
| `SetServerModuleInfo` | 解析 module（模组）信息：每模组卡数、模组数、是否 A+X 异构 |
| `SetSuperPodInfo` | 解析超节点信息：超节点数、每超节点 server 数；区分 2 层/3 层拓扑 |
| `CalcGroupIdx` | 根据 rank 在哪一层的位置，计算 serverIdx 或 superPodIdx |
| `GetPairLinkCounter` | 统计 server 内所有 rank 对之间的链路协议类型计数（HCCS/ROCE/PCIe/SIO） |
| `CalcLinkInfo` | 根据 pairLinkCounter 判断 HCCS_SW 和 SIO 链路关系 |
| `IsDiffDeviceModule` | 判断是否存在 A+X 异构模组 |

### 2. 细粒度拓扑形状分析（填充 TopoInfoWithNetLayerDetails）

| 函数 | 作用 |
|------|------|
| `CalcTopoShape` | 入口函数，串联以下所有分析步骤 |
| `ExtractNetLayerDetails` | 提取每层网络的实例数、实例大小列表、本 rank 所属实例大小 |
| `CalcLevel1Nhr` | 当 L0 的 GCD=1 时标记 `Level1Nhr=true`（Mesh 退化为 NHR） |
| `ExtractTopoDetails` | 提取每层的 topoInst 类型、大小、包含的 rank 列表 |
| `CalcLevel0TopoShape` | 判断 L0 拓扑形状（Mesh1D / Mesh2D / Ring 等） |
| `Is2DieFullMesh` | 判断 L0 是否为 2-Die 全互连 |
| `CalcLevel0MeshType` | 判断 L0 Mesh 是单 Die / 2Die 规则 / 2Die 不规则 |
| `IsLevel0PcieMix` | 判断 L0 是否为 PCIe 混合拓扑 |
| `CalAllLevelEndpointAttrBwCoeff` | 收集所有层级的端点带宽系数（目前未被调用） |

---

## topo_match_multilevel.cc 的作用

`topo_match_multilevel.cc` 是**拓扑匹配与算法分层编排层**，继承自 `TopoMatchBase`，负责根据拓扑信息**决定每一层用哪些 rank 组成通信子组**，输出 `AlgHierarchyInfoForAllLevel`。它回答的是"**每一层该怎么组网通信**"。

核心流程（`MatchTopo`）：

```
1. 校验 topoLevelNums（1~3层）
2. 检查 L0 实例大小是否对称
3. TopoForLayer0 → 构建 L0 通信组（Mesh1D / Mesh2D / 非对称 GCD 子组）
4. TopoForLayer1 → 构建 L1 通信组（同序号卡跨 server 组网，NHR 模式）
5. TopoForLayer2 → 构建 L2 通信组（同序号卡跨 pod 组网，仅 3 层拓扑时）
```

关键逻辑：
- **L1 组网**：取 `rankId % layer0Size == myRank % layer0Size` 的 rank，即"同位置卡跨 server 组网"
- **L2 组网**：取 `rankId % (layer0Size * layer1Size) == myRank % (layer0Size * layer1Size)` 的 rank，即"同位置卡跨 pod 组网"

---

## 两者的关系

```
┌─────────────────────────────────────────────────────────┐
│                    调用链                                 │
│                                                          │
│  Selector / Executor                                     │
│       │                                                  │
│       ▼                                                  │
│  topo_host.cc :: InitRankInfo(comm, TopoInfo*)           │
│       │  采集原始拓扑 → 填充 TopoInfo                     │
│       ▼                                                  │
│  topo_host.cc :: InitRankInfo(comm, TopoInfoWithNetLayerDetails*)  │
│       │  在 TopoInfo 基础上 + CalcTopoShape()            │
│       │  分析拓扑形状 → 填充 TopoInfoWithNetLayerDetails  │
│       ▼                                                  │
│  topo_match_multilevel.cc :: MatchTopo(comm, topoInfo, algHierarchyInfo) │
│       │  读取 topoInfo 中的拓扑信息                       │
│       │  决定每层通信子组 → 填充 AlgHierarchyInfoForAllLevel │
│       ▼                                                  │
│  Executor 使用 algHierarchyInfo 进行实际通信编排           │
└─────────────────────────────────────────────────────────┘
```

**简单说：`topo_host.cc` 是"感知"（知道拓扑长什么样），`topo_match_multilevel.cc` 是"决策"（决定怎么分组通信）。**

| 维度 | `topo_host.cc` | `topo_match_multilevel.cc` |
|------|----------------|---------------------------|
| 职责 | 采集拓扑信息、分析拓扑形状 | 根据拓扑信息构建通信子组 |
| 输入 | HCOMM RankGraph API | `TopoInfoWithNetLayerDetails` |
| 输出 | `TopoInfo` / `TopoInfoWithNetLayerDetails` | `AlgHierarchyInfoForAllLevel` |
| 层级 | 底层（数据采集） | 上层（策略决策） |
| 是否感知 3 层 | 是（`SetSuperPodInfo` 区分 2/3 层） | 是（`TopoForLayer2` 仅 3 层时调用） |
| 类结构 | 自由函数（无类） | `TopoMatchBase` 子类 |

`topo_match_multilevel.cc` 依赖 `topo_host.cc` 提供的 `topoInfo` 中的 `topoLevelNums`、`netLayerDetails`、`level0PcieMix`、`deviceType` 等字段来做决策。两者是**生产者-消费者**关系。

---

## SetSuperPodInfo 区分 2/3 层的机制

`SetSuperPodInfo` 通过 `HcclRankGraphGetLayers` 返回的 **`netLayersNum`（网络层数）** 来区分 2 层和 3 层拓扑：

```cpp
uint32_t *netlayers = nullptr;
uint32_t netLayersNum = 0;
CHK_RET(HcclRankGraphGetLayers(comm, &netlayers, &netLayersNum));
```

`netLayersNum` 来自 rank table 配置，表示 rank table 中声明了几层网络。三种情况：

### 情况 1：netLayersNum == 3（三层拓扑）

```
netLayers = {0, 1, 2}  或  {0, 1, 3}
```

- L0 = Server 内（HCCS 链路）
- L1 = Server 间同 Pod（RoCE 链路）
- L2 = Pod 间（RoCE 链路）

处理逻辑：
1. 获取 L0 层的 `level0SizeList`（每个 L0 实例的 rank 数，即每个 server 的卡数）
2. 获取 L1 层的 `level1SizeList`（每个 L1 实例的 rank 数，即每个超节点的总 rank 数）
3. `superPodNum = level1RankListNum`（L1 实例数 = 超节点数）
4. 通过 `CalculateServersPerSuperPod(level0SizeList, level1SizeList, superPodToServerNum)` 计算每个超节点包含多少 server
5. `serverNumPerSuperPod = superPodToServerNum[superPodIdx]`（取本 rank 所在超节点的 server 数）

```
示例：24 rank，L0=4, L1=3, L2=2

level0SizeList = [4, 4, 4, 4, 4, 4]   ← 6 个 server，每个 4 卡
level1SizeList = [12, 12]              ← 2 个超节点，每个 12 rank
superPodNum = 2
superPodToServerNum = [3, 3]           ← 每个超节点 3 个 server
```

### 情况 2：netLayersNum == 2（二层拓扑）

```
netLayers = {0, 1}
```

- L0 = Server 内
- L1 = Server 间（无 Pod 层级概念）

处理逻辑：
1. 只获取 L0 层的 `level0SizeList`
2. `superPodNum = 1`（只有一个超节点，即整个集群）
3. `serverNumPerSuperPod = level0RankListNum`（L0 实例数 = server 数）

```
示例：8 rank，L0=4, L1=2

level0SizeList = [4, 4]   ← 2 个 server，每个 4 卡
superPodNum = 1
serverNumPerSuperPod = 2   ← 唯一超节点包含 2 个 server
```

### 情况 3：netLayersNum < 2（单层拓扑）

```
netLayers = {0}
```

- L0 = Server 内（单机场景）

处理逻辑：
1. `superPodNum = 1`
2. `serverNumPerSuperPod = 1`（只有 1 个 server）

### 判断逻辑汇总

```
netLayersNum == 3  →  三层拓扑，需要计算 superPodNum 和 serverNumPerSuperPod
                      从 L0/L1 实例大小推算每个超节点的 server 数
netLayersNum == 2  →  二层拓扑，superPodNum=1，serverNumPerSuperPod=L0实例数
netLayersNum <  2  →  单层拓扑，superPodNum=1，serverNumPerSuperPod=1
```

### 非对称处理

三层拓扑中，如果不同超节点包含的 server 数不一致（`multiSuperPodDiffServerNumMode=true`），会取 GCD 作为统一的 `serverNumPerSuperPod`，并重新计算 `superPodNum = serverNum / serverNumPerSuperPod`，将非对称拓扑转化为对称拓扑以适配 NHR-HCF 算法。

```cpp
if (!topoInfo->multiModuleDiffDeviceNumMode && topoInfo->multiSuperPodDiffServerNumMode) {
    topoInfo->serverNumPerSuperPod = CalGCD(superPodToServerNum);
    topoInfo->multiSuperPodDiffServerNumMode = false;
    topoInfo->superPodNum = topoInfo->serverNum / topoInfo->serverNumPerSuperPod;
}
```
