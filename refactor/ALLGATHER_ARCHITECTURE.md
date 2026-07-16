# Refactor 目录 AllGather 实现架构

## 总览

```
API (all_gather_op) → Selector → HcclAlgorithm → OpsExecutor → BaseTemplate → Kernel
```

6 层调用链，每层职责单一：

| 层 | 模块 | 职责 |
|----|------|------|
| 1. API | `api/all_gather_op.cc` | C 接口入口，参数校验，组装 OpParam |
| 2. Selector | `selector/execute_selector.cc` | 根据设备/数据量/拓扑选择最优算法 |
| 3. Algorithm | `algorithm/all_gather/` | 构造 HcclAlgorithm（含 AlgoExecDesc 树） |
| 4. Executor | `executor/ops_executor.cc` | 三阶段编排：拓扑匹配→资源计算→循环执行 |
| 5. Template | `template/base_template.h` | 单次通信原语：CalcRes + KernelRun |
| 6. Engine | `engine/` | 具体引擎驱动（AICPU/CCU/AIV） |

---

## 1. API 入口

```cpp
// api/all_gather_op.cc
HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream)
  → AllGatherOutPlace()
    → AllGatherOutPlaceCommon()
```

核心逻辑：

```cpp
OpParam param;
param.inputPtr      = sendBuf;
param.inputSize     = sendCount * sizeof(type);
param.outputPtr     = recvBuf;
param.outputSize    = inputSize * rankSize;  // AllGather 特性：输出 = rankSize 倍
param.DataDes.count = sendCount;
param.DataDes.dataType = dataType;
param.opType        = HCCL_CMD_ALLGATHER;

HcclAlgorithm alg;
TopoInfoWithNetLayerDetails topoInfo;

Selector(comm, param, topoInfo, &alg);       // 选算法
HcclExecOp(comm, param, topoInfo, alg, ...);  // 统一执行
```

关键差异：`outputSize = inputSize × rankSize`，这是 AllGather 与其他算子的本质区别。

---

## 2. 算法选择

```cpp
Selector(comm, param, topoInfo, &alg)
```

根据以下条件选出 `HcclAlgorithm`：
- 设备类型（A5/A4）
- 通信域规模
- 物理拓扑（server/superPod）
- 数据量
- 用户配置（环境变量 `HCCL_OP_EXPANSION_MODE`）

选出的算法填充到 `HcclAlgorithm`：
```cpp
struct HcclAlgorithm {
    HcclCMDType       hcclCmdType;     // HCCL_CMD_ALLGATHER
    HcclAlgEngineType  engineType;     // AICPU / CCU_MS / AIV
    shared_ptr<TopoMatchBase> topoMatch;  // 拓扑匹配器
    AlgoExecDesc      algoExecDesc;   // 算法执行树
};
```

---

## 3. 算法树 (AlgoExecDesc)

### 3.1 数据结构

```cpp
struct AlgoExecDesc {
    HcclAlgExecPolicy execPolicy;       // SOLE | SEQUENCE | PARALLEL | CONCURRENT
    vector<VariantType> children;       // 叶子=TemplateExecDesc, 子树=shared_ptr<AlgoExecDesc>
    vector<u32> dataSplitRatio;        // PARALLEL 时数据切分比例
};

using VariantType = variant<TemplateExecDesc, shared_ptr<AlgoExecDesc>>;

struct TemplateExecDesc {
    TemplateDesc templateDesc;          // 模板类型（Mesh/NHR/FullMesh）
    int subCommIndex;                   // 通信子域索引（INTRA=0, INTER=1, POD=2）
};
```

### 3.2 算法表

`algorithm/all_gather/algorithm_all_gather_aicpu.cc` 定义了 11 种 AICPU AllGather：

| 算法 | execPolicy | children |
|------|-----------|----------|
| OmniPipeUboe | SOLE | Mesh |
| NHR | SOLE | NHR(intra) |
| ParallelMesh1DNHRUboe | PARALLEL | NHR(intra) + NHR(inter) |
| SequenceNHRMesh1D | SEQUENCE | Mesh(intra) → NHR(inter) |
| ParallelMesh1DNHR | PARALLEL | Mesh(intra) + NHR(inter) |
| Mesh1D1DZAxisDetour | SOLE | Mesh(intra) |
| Mesh1D | SOLE | Mesh(intra) |
| ParallelMesh1DNHRPcie | PARALLEL | Mesh(intra) + NHR(inter) |
| OmniPipePcie | SOLE | Mesh |
| ConcurrentMesh1DNHR | CONCURRENT | Mesh(intra) + NHR(inter) |
| ParallelMesh1DNHRMultiJetty | PARALLEL | Mesh(intra) + NHR(inter) |

### 3.3 OmniPipe 示例（最复杂）

```
algoExecDesc6 (SEQ, 4:4)                    ← 根
├── algoExecDesc4 (PAR, 2:2)
│   ├── algoExecDesc3 (SEQ, 2:2)
│   │   ├── algoExecDesc1 (PAR, 1:1)
│   │   │   ├── Mesh(sub0)            ← 叶子模板
│   │   │   └── NHR(sub1)
│   │   └── algoExecDesc2 (PAR, 1:1)
│   │       ├── NHR(sub1)
│   │       └── Mesh(sub0)
│   └── NHR(sub2)
└── algoExecDesc5 (PAR, 2:2)
    ├── NHR(sub2)
    └── algoExecDesc3 (同上, shared_ptr 共享)
```

- `algoExecDesc3` 被 `d4` 和 `d5` 共享（`shared_ptr`），CalcRes 中同一组模板只算一次资源
- SEQ 子节点串行需要 PreSync/PostSync 同步
- PAR 子节点并行切分数据

---

## 4. 执行器 (OpsExecutor)

### 4.1 三阶段

```
Phase 1: CalcAlgHierarchyInfo
  输出: algHierarchyInfo_, rankSize_

Phase 2: CalcRes
  输出: notify/thread/channel 资源请求

Phase 3: Orchestrate
  输入: resCtx (CCL buffer/线程/通道)
  流程: InitRes → GetMaxProcCntPerLoop → for loop → OrchestrateLoop
```

### 4.2 Phase 1: 拓扑匹配

```cpp
CalcAlgHierarchyInfo(comm, topoInfo)
  algo_.topoMatch->MatchTopo(comm, topoInfo, algHierarchyInfo_);
  // 算 rankSize: 每层取第一个 rank 组大小累乘
  for each level: rankSize_ *= infos[level][0].size();
```

### 4.3 Phase 2: 资源计算

```cpp
CalcRes(req)
  CalcResRecursion(algo_.algoExecDesc)
    for each child:
      if TemplateExecDesc → CalcTemplateRes(tmpl)
        → GetTemplate → BaseTemplate::CalcRes (NHR: threadNum=channels×2, Mesh: threadNum=rankSize-1)
      if AlgoExecDesc → CalcResRecursion(subTree)  // 递归
  
  // 汇总: SEQ取max, PAR求和
  for each level:
    slaveThreadNum += maxSlave + 1
    notifyPerThread = [notifyOnMain+1, (maxSlave copies of maxNotifyPerThread)]
    channels.push_back(requestChannels)
  
  scratchMultiple_ = (ALLGATHER) ? rankSize_ : 1
```

### 4.4 Phase 3: 编排执行

```cpp
Orchestrate(resCtx)
  InitRes(resCtx)  // 初始化 CCL buffer/thread/channel
  
  dataCount = inputSize / dataTypeSize
  maxProcCntPerLoop = GetMaxProcCntPerLoop(dataCount)
    // 约束: CCL buffer / (scratchMultiple × dataTypeSize)
    // 约束: UB_MAX_DATA_SIZE / dataTypeSize
    // AllGather 不做 rankSize 对齐
  
  loopTimes = ceil(dataCount / maxProcCntPerLoop)
  for loop 0..N-1:
    processCount = (last) ? remainder : maxProcCntPerLoop
    tailCount    = (last) ? processCount % rankSize : 0
    InitAlgoExecDataDesc(desc, offset, processCount, tailCount)
    OrchestrateLoop(algoExecDesc, desc)
    offset += processCount
```

### 4.5 递归编排

```cpp
OrchestrateLoop(algoExecDesc, dataDesc)
  if SEQUENCE && children > 1: PreSyncBySubCommMask()
  
  for each child:
    if PARALLEL: UpdateDataSplitParallel()     // 按 ratio 切分 sliceCount/sliceOffset
    if SEQUENCE: UpdateDataSplitSequence()      // 下游 input = 上游 output
    
    if TemplateExecDesc → RunTemplateDesc()
      → GenTemplateRes(subCommIndex, templateResource)
      → GenTemplateDataParams(dataDesc, templateDataParams)
      → BaseTemplate::KernelRun(engine, params, resource)
    
    if AlgoExecDesc → OrchestrateLoop(subTree)  // 递归
    
    if SEQUENCE: PostSyncBySubCommMask()
  
  MergeChildrenOutput()  // PAR: scratch 求和, SEQ: scratch 取max
```

---

## 5. 模板层

### 5.1 CalcRes

```cpp
BaseTemplate::CalcRes(comm, engineType, res)
  // AllGather 输出=rankSize×输入 → 需要 scratch buffer
  CalcChannelRequestNhr/Mesh1D(comm, engineType, ranks, channels)
  GetRes(res)
    NHR: threadNum = channelsPerRank × 2, notifyPerThread = 2
    Mesh: threadNum = rankSize - 1,        notifyPerThread = 1
    slaveThreadNum = threadNum - 1
```

### 5.2 KernelRun

```cpp
BaseTemplate::KernelRun(engine, params, resource, ranksForOutput)
  // PreCopy: input → CCL buffer
  // 通信原语: engine.Launch(mesh/NHR logic)
  // PostCopy: CCL buffer → output
```

---

## 6. 数据流示例

以 `128 ranks, 1024 elem/rank, FP32` 为例：

```
输入: sendBuf (1024 × 4 = 4096 bytes per rank)
输出: recvBuf (4096 × 128 = 524288 bytes per rank)

OpParam:
  inputSize  = 4096
  outputSize = 524288  (= rankSize × inputSize)

Orchestrate:
  dataCount = 4096 / 4 = 1024
  maxProcCntPerLoop = min(CCL_bound, UB_bound, dataCount)
  loops = ceil(1024 / maxProcCntPerLoop)

  每轮:
    sliceCount = 1024 / rankSize (非AG) 或 1024 (AG)
    stride = sliceCount
    tailCount = 0 (整除时)
    
    TemplateDataParams:
      sliceCount = 1024
      stride = 1024
      tailCount = 0
```

---

## 7. 关键设计决策

| 决策 | 说明 |
|------|------|
| `variant` 嵌套树 | 用 `std::variant<TemplateExecDesc, shared_ptr<AlgoExecDesc>>` 统一叶子和子树 |
| `shared_ptr` 共享子树 | d3 被 d4 和 d5 共用，CalcRes 不重复计算 |
| ratio 传递 | `dataSplitRatio` 控制 PARALLEL 节点的数据切分比例 |
| SEQ 取 max, PAR 求和 | 资源聚合策略，SEEQ 复用 scratch，PAR 独立 scratch |
| AllGather 不 rankSize 对齐 | GetMaxProcCntPerLoop 中 AllGather 跳过 rankSize 整除检查 |
| `scratchMultiple_ = rankSize` | AllGather 每 rank 收集 rankSize 份数据，需要 rankSize 倍 scratch |
