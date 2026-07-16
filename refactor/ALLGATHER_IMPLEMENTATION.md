# Refactor AllGather 实现详解

## 目录

1. [整体架构](#1-整体架构)
2. [API 入口层](#2-api-入口层)
3. [算法选择层 (Selector)](#3-算法选择层-selector)
4. [算法构造层 (Algorithm)](#4-算法构造层-algorithm)
5. [执行引擎层 (Executor)](#5-执行引擎层-executor)
6. [模板层 (Template)](#6-模板层-template)
7. [完整数据流](#7-完整数据流)

---

## 1. 整体架构

```
api/          → C API 入口 (HcclAllGather)
selector/     → 根据设备、数据量、拓扑选择最优算法
algorithm/    → 构造算法描述树 (HcclAlgorithm + AlgoExecDesc)
executor/     → 编排执行 (CalcAlgHierarchyInfo → CalcRes → Orchestrate)
template/     → 具体通信算法实现 (Mesh1D / NHR)
```

核心数据结构 `HcclAlgorithm` 包含四个成员：

```cpp
class HcclAlgorithm {
    HcclCMDType       hcclCmdType;    // 算子类型: ALLGATHER
    HcclAlgEngineType  engineType;     // 引擎: AICPU / CCU_MS / AIV
    shared_ptr<TopoMatchBase> topoMatch; // 拓扑匹配器
    AlgoExecDesc      algoExecDesc;   // 算法执行树
};
```

`AlgoExecDesc` 是一棵**递归的算法描述树**：

```cpp
struct AlgoExecDesc {
    HcclAlgExecPolicy execPolicy;      // SOLE | SEQUENCE | PARALLEL | CONCURRENT
    vector<VariantType> children;      // TemplateExecDesc(叶子) | shared_ptr<AlgoExecDesc>(子树)
    vector<u32> dataSplitRatio;        // PARALLEL 时的数据切分比例
};
```

---

## 2. API 入口层

**文件**：`refactor/ops/api/all_gather_op.cc`、`all_gather_op.h`

### 2.1 调用链

```
HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream)
  │
  ├── AllGatherInitAndCheck()   // 参数校验 (comm/sendBuf/recvBuf 非空, sendCount>0, dataType有效)
  │
  └── AllGatherOutPlace()
        └── AllGatherOutPlaceCommon()
```

### 2.2 AllGatherOutPlaceCommon 核心逻辑

```cpp
HcclResult AllGatherOutPlaceCommon(sendBuf, recvBuf, sendCount, dataType, comm, stream, tag, opMode, resPack)
{
    // 1. 获取 rankSize
    HcclGetRankSize(comm, &userRankSize);

    // 2. 计算 buffer 大小 (AllGather 特性: output = input × rankSize)
    u64 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize  = sendCount * perDataSize;
    u64 outputSize = inputSize * userRankSize;

    // 3. 构造 OpParam
    OpParam param;
    param.inputPtr  = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = sendCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // 4. 算法选择
    HcclAlgorithm alg;
    Selector(comm, param, topoInfo, alg);

    // 5. 边界处理
    if (ShouldUseInnerOp(...)) return HcclAllGatherInner(...);
    if (userRankSize == 1) return SingleRankProc(comm, param);

    // 6. 执行
    HcclExecOp(comm, param, topoInfo, alg, resPack);
}
```

---

## 3. 算法选择层 (Selector)

**文件**：`refactor/ops/selector/execute_selector.cc`

### 3.1 选择逻辑

根据以下条件选出最优算法：
- 设备类型 (DevType)
- 数据量大小 (决定用 Mesh 还是 NHR)
- 拓扑结构 (单 server、多 server、clos 拓扑)
- 环境变量 (`HCCL_OP_EXPANSION_MODE`)

### 3.2 选择结果

返回填充好 `engineType` 和 `algoExecDesc` 的 `HcclAlgorithm`：

```
AICPU AllGather 算法表 (11 种):
├── OMNIPIPE_UBOE          (OmniPipe UBOE)
├── NHR                    (单 NHR)
├── PARALLEL_MESH1D_NHR_UBOE (Mesh||NHR UBOE)
├── SEQUENCE_NHR_MESH1D     (NHR→Mesh 串行)
├── PARALLEL_MESH1D_NHR     (Mesh||NHR 并行)
├── MESH1D1D_ZAXIS_DETOUR   (Mesh 1D Z轴)
├── MESH1D                  (单 Mesh 1D)
├── PARALLEL_MESH1D_NHR_PCIE (PCIe)
├── OMNIPIPE_PCIE           (OmniPipe PCIe)
├── CONCURRENT_MESH1D_NHR   (Mesh+NHR 并发)
└── PARALLEL_MESH1D_NHR_MULTIJETTY (多 Jetty)
```

---

## 4. 算法构造层 (Algorithm)

**文件**：`refactor/ops/algorithm/all_gather/algorithm_all_gather_aicpu.cc`

### 4.1 全局算法表

```cpp
const HcclAlgorithm g_aicpuAllGatherAlgoMap[11] = {
    MakeAicpuAllGatherAlgo(shared_ptr<TopoMatch3Level>()),                    // OmniPipe
    MakeAicpuAllGatherAlgo(shared_ptr<TopoMatch1D>(), nhrDesc),              // NHR
    MakeAicpuAllGatherAlgo(shared_ptr<TopoMatchSqueeze2D>(), uboeDesc),      // UBOE
    // ... 其余 8 种
};
```

### 4.2 算法树示例

#### SOLE 类型（单模板）

```
execPolicy: SOLE
children: [TemplateExecDesc{NHR, subCommIndex=0}]
dataSplitRatio: [1]
```

#### PARALLEL 类型（并行两模板）

```
execPolicy: PARALLEL
children: [TemplateExecDesc{Mesh, subCommIndex=0},
           TemplateExecDesc{NHR,  subCommIndex=1}]
dataSplitRatio: [1, 1]   // 各一半数据
```

#### SEQUENCE 嵌套类型（OmniPipe）

```
d1: PARALLEL [Mesh(sub0), NHR(sub1)]             1:1
d2: PARALLEL [NHR(sub1), Mesh(sub0)]             1:1
d3: SEQUENCE [d1, d2]                            2:2
d4: PARALLEL [d3, NHR(sub2)]                     2:2
d5: PARALLEL [NHR(sub2), d3]                     2:2
d6: SEQUENCE [d4, d5]                            4:4  ← 根节点
```

### 4.3 模板描述符表

```cpp
// algorithm/all_gather/all_gather_template_desc.h
enum class HcclAllGatherTemplateDescType {
    ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY,          // NHR, ONE_SHOT
    ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY,     // FULLMESH, ONE_SHOT
    ALLGATHER_TEMPLATE_FULLMESH_MULTIPLE_JETTY,   // FULLMESH, MULTI_JETTY
    ALLGATHER_TEMPLATE_NHR_MULTIPLE_JETTY,        // NHR, MULTI_JETTY
    ALLGATHER_TEMPLATE_DESC_COUNT,
};

extern const TemplateDesc g_allGatherTemplateDescMap[4];
```

---

## 5. 执行引擎层 (Executor)

**文件**：`refactor/ops/executor/ops_executor.cc`

### 5.1 构造函数

从 `OpParam` 提取关键参数，存入 `DataInfo` 和成员变量：

```cpp
OpsExecutor::OpsExecutor(HcclAlgorithm &algo, OpParam &param)
    : algo_(algo)
{
    dataInfo_.inputPtr  = param.inputPtr;
    dataInfo_.inputSize = param.inputSize;   // AllGather: sendCount × dataTypeSize
    dataInfo_.outputPtr = param.outputPtr;
    dataInfo_.outputSize = param.outputSize; // AllGather: inputSize × rankSize
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType]; // e.g. FP32=4
}
```

### 5.2 执行三阶段

```
1. CalcAlgHierarchyInfo(comm, topoInfo)
     → topoMatch->MatchTopo() 匹配拓扑 → algHierarchyInfo_
     → rankSize_ = Π(每层第一个rank组大小)

2. CalcRes(resReq)
     → CalcResRecursion(algoExecDesc) 递归遍历算法树
       ├── TemplateExecDesc → CalcTemplateRes
       │     ├── GetTemplate() 实例化 BaseTemplate
       │     └── baseTemplate->CalcRes() 计算 thread/notify/channel
       └── AlgoExecDesc → 递归进入子树
     → 汇总: slaveThreadNum, notifyNumOnMainThread, notifyNumPerThread, channels

3. Orchestrate(resCtx)
     → InitRes(resCtx) 初始化 buffer/thread/channel
     → dataCount = dataInfo_.inputSize / dataTypeSize_
     → maxProcCntPerLoop = GetMaxProcCntPerLoop(dataCount)
     → for loopIdx in 0..loopTimes:
          processCount = (last?) remainder : maxProcCntPerLoop
          tailCount = (last?) processCount % rankSize_ : 0
          InitAlgoExecDataDesc(offset, processCount, tailCount)
          OrchestrateLoop(algoExecDesc) → 递归执行
```

### 5.3 CalcResRecursion 详细流程

```cpp
CalcResRecursion(AlgoExecDesc &node, u32 &subCommMask)
{
    for each child in node.children:
        if child is TemplateExecDesc:
            CalcTemplateRes(child)
                baseTemplate = GetTemplate(engineType, tmplDesc, ranks, myRank)
                baseTemplate->CalcRes(comm, engineType, tempRequest)
                // 取 max 更新 maxSlaveThreadNum_[subComm], maxNotifyOnMain_[subComm], ...
                subCommMask |= (1U << child.subCommIndex)
        else if child is AlgoExecDesc:
            CalcResRecursion(*child) → 递归

    UpdateSubCommMaskMap(node, subCommMask)  // 记录节点的同步掩码
}
```

### 5.4 OrchestrateLoop 详细流程

```cpp
OrchestrateLoop(AlgoExecDesc &node, AlgoExecDataDesc &parentData)
{
    for each child:
        if node.execPolicy == SEQUENCE:
            PreSyncBySubCommMask(node)  // 等待上一个子节点完成
        if node.execPolicy == PARALLEL:
            UpdateDataSplitParallel(node, parentData, childIdx, children)  // 按 ratio 分数据
        else:
            UpdateDataSplitSequence(node, parentData, childIdx, children)  // 串行传递数据

        if child is TemplateExecDesc:
            RunTemplateDesc(child, children[i])
                GenTemplateRes(subComm, res)         // 从成员表取 channel/thread
                GenTemplateDataParams(dataDesc, params) // 映射数据参数
                baseTemplate->KernelRun(engine, params, res) // 实际通信!
        else:
            OrchestrateLoop(*child, children[i])  // 递归

        if node.execPolicy == SEQUENCE:
            PostSyncBySubCommMask(node)  // 回到主线程等待
}
```

### 5.5 GetMaxProcCntPerLoop 数据切分

```cpp
u64 GetMaxProcCntPerLoop(u64 dataCount) {
    // 1. CCL buffer scratch 约束
    maxByCcl = cclBufferSize / (scratchMultiple_ × dataTypeSize_)
    // 2. UB 传输约束
    maxByUb = UB_MAX_DATA_SIZE / dataTypeSize_
    // 3. 取最小值
    resCount = min(dataCount, maxByCcl, maxByUb)
    // 4. AllGather 不需要 rankSize 对齐
    if (algo_.hcclCmdType != ALLGATHER) {
        resCount = (resCount / rankSize_) × rankSize_
    }
    return max(resCount, 1ULL)
}
```

### 5.6 InitAlgoExecDataDesc AllGather 特殊处理

```cpp
if (algo_.hcclCmdType == ALLGATHER) {
    desc.sliceCount = dataCount;              // 不除以 rankSize
    desc.ranksForInputData = {myRank_};       // AllGather 每个rank输入独立
} else {
    desc.sliceCount = dataCount / rankSize_;
    desc.ranksForInputData = {0,1,...,rankSize-1};
}
```

---

## 6. 模板层 (Template)

**文件**：`refactor/ops/template/base_template.h`

### 6.1 BaseTemplate 接口

```cpp
class BaseTemplate {
public:
    // 资源计算: 构建 channel 描述符, 计算线程和 notify 数
    virtual HcclResult CalcRes(HcclComm comm, HcclAlgEngineType engineType, AlgResourceRequest &res);

    // scratch buffer 倍率
    virtual float CalcScratchMultiple(BufferType in, BufferType out);

    // 核心: 执行通信 (PreCopy → 通信原语 → PostCopy)
    virtual HcclResult KernelRun(BaseEngine &engine, const TemplateDataParams &params,
                                 TemplateResource &res, vector<u32> &ranksForOutputData);

protected:
    bool IsNhr() const;   // algType == NHR 或 NHR_V1
};
```

### 6.2 CalcRes 实现

```
BaseTemplate::CalcRes(comm, engineType, res)
  ├── rankSize <= 1 → 空通道, 直接返回
  ├── 构造 subcommInfo = {ranks_}
  ├── IsNhr() ?
  │     ├── true  → CalcChannelRequestNhr(comm, engineType, ...)   // clos 拓扑
  │     └── false → CalcChannelRequestMesh1D(comm, engineType, ...) // mesh 1D
  ├── res.channels.push_back(levelChannels)
  └── GetRes(res)
        ├── NHR:   threadNum = channelsPerRank,      notifyPerThread = 1
        └── Mesh:  threadNum = rankSize - 1,          notifyPerThread = 1
           res.slaveThreadNum = threadNum - 1
           res.notifyNumOnMainThread = threadNum - 1
           res.notifyNumPerThread.assign(slaveThreadNum, notifyPerThread)
```

### 6.3 Mesh vs NHR 资源差异

| 指标 | Mesh 1D (FULLMESH) | NHR |
|------|-------------------|-----|
| 通道构造 | `CalcChannelRequestMesh1D` (本 server 内) | `CalcChannelRequestNhr` (跨 clos) |
| 线程数 | rankSize - 1 | channelsPerRank |
| notify/thread | 1 | 1 |
| slaveThread | rankSize - 2 | channelsPerRank - 1 |
| ranksSz=8 时 | slave=6, notify=6, notifyPerThread=1×6 | channels=7, slave=6, notify=6, notifyPerThread=1×6 |

---

## 7. 完整数据流

### 7.1 单算子模式 (OPBASE)

```
HcclAllGather(SendBuf, RecvBuf, sendCount=1024, FP32, comm, stream)
│
├── outputSize = 1024×4×rankSize = 4096×128 = 524288 bytes
├── OpParam: inputPtr=SendBuf, inputSize=4096, outputPtr=RecvBuf, outputSize=524288
│
├── Selector → HcclAlgorithm { engineType=AICPU, algoExecDesc=<OmniPipe树> }
│
├── HcclExecOp:
│   │
│   ├── CalcAlgHierarchyInfo
│   │     topoMatch->MatchTopo(comm, topoInfo, algHierarchyInfo_)
│   │     infos: [[8 ranks], [8 ranks], [2 ranks]] → rankSize_ = 128
│   │
│   ├── CalcRes
│   │     CalcResRecursion(d6) → 遍历所有 TemplateExecDesc
│   │     level0: maxSlave=6(Mesh), notifyOnMain=6, notifyPerThread={1×6}
│   │     level1: maxSlave=6(NHR),  notifyOnMain=6, notifyPerThread={1×6}
│   │     level2: maxSlave=1(NHR),  notifyOnMain=1, notifyPerThread={1×1}
│   │     → slaveThreadNum=23, notifyPerThread=[7,1×6, 7,1×6, 2,1]
│   │     scratchMultiple_ = 128 (ALLGATHER → rankSize)
│   │
│   └── Orchestrate(resCtx)
│         InitRes: cclBuffer, threads, channelTable
│         dataCount = 4096/4 = 1024
│         maxPerLoop = GetMaxProcCntPerLoop(1024) = min(1024, cclBuf/scratch, UB/4)
│         loopTimes = ceil(1024 / maxPerLoop)
│
│         for loop 0..N:
│           processCount & tailCount
│           OrchestrateLoop(algoExecDesc)
│             ├── d4(PAR): UpdateDataSplit(d3, NHR)
│             │     ├── d3(SEQ): PreSync→d1→d2→PostSync
│             │     │     ├── d1(PAR): RunTemplateDesc(Mesh) + RunTemplateDesc(NHR)
│             │     │     └── d2(PAR): RunTemplateDesc(NHR) + RunTemplateDesc(Mesh)
│             │     └── RunTemplateDesc(NHR)
│             └── d5(PAR): 同上, NHR + d3
│
│         每个 RunTemplateDesc:
│           GenTemplateRes → channel/thread
│           GenTemplateDataParams → dataOffset/sliceCount/tail/stride/ranks
│           baseTemplate->KernelRun(engine, params, res)
│             → PreCopy(从 userIn→cclBuffer)
│             → 通信原语(AllGather on cclBuffer)
│             → PostCopy(从 cclBuffer→userOut)
```

### 7.2 关键 AllGather 特性

1. **outputSize = inputSize × rankSize**：每个 rank 收集所有 rank 的数据
2. **sliceCount 不除以 rankSize**：与 AllReduce 不同，AllGather 的 input 来自单个 rank
3. **ranksForInputData = {myRank_}**：只有自己的 rank
4. **scratchMultiple_ = rankSize**：AllGather 的 CCL buffer 需要容纳 rankSize 份数据
5. **尾块 tailCount**：`processCount % rankSize_`，处理不整除的边界情况

### 7.3 资源汇总规则

| 策略 | slaveThread | scratch | 数据关系 |
|------|------------|---------|---------|
| PARALLEL | 求和 | 求和 | 各子节点分数据 |
| SEQUENCE | 取 max | 取 max | 子节点数据 = 前一个输出 |

### 7.4 目录结构

```
refactor/ops/
├── api/
│   ├── all_gather_op.h/cc          ← HcclAllGather C API
├── selector/
│   └── execute_selector.cc         ← 算法选择
├── algorithm/
│   └── all_gather/
│       ├── all_gather_template_desc.h/cc ← 模板描述符表
│       └── algorithm_all_gather_aicpu.cc ← 11种AICPU算法树构造
├── executor/
│   └── ops_executor.h/cc           ← 执行引擎 (三阶段)
├── template/
│   ├── base_template.h             ← 模板基类 (CalcRes/KernelRun)
│   ├── template_factory.h          ← GetTemplate 工厂
│   └── aicpu/                      ← AICPU 子类实现
└── op_common/
    └── hccl_algorithm.h            ← HcclAlgorithm + AlgoExecDesc 数据结构
```
