# AICPU 模板策略架构 — 软件设计说明书

> 本文档描述 `src/experimental/aicpu/` 目录下的 AICPU 模板策略架构，采用工厂 + 策略模式，
> 将算法模板沿算法 / 通信设备 / 数据切片 / 拓扑四个维度正交分解，通过基类模板组装各策略。

---

## 1. 系统概述

### 1.1 设计目标

将 AllGather 等算子的 AICPU 模板按四个正交维度拆分为独立策略，实现跨算子复用：

| 维度 | 策略接口 | 职责 | 变体 |
|------|---------|------|------|
| **算法** | 直接继承 `AicpuTemplate` | 数据交换逻辑（LocalCopy / RunAlgorithm / PostLocalCopy） | Mesh1D, NHR |
| **通信设备** | `ICommunicationPolicy` | 数据传输与同步（Pre/PostSync, TransferData, Dispatch） | AICPU, DPU |
| **数据切片** | `ISlicePolicy` | 数据编排与切片描述（RepeatNum, SliceDescs, 缓冲区定位） | NoSlice, OmniSlice |
| **拓扑** | `ITopoPolicy` | 计算资源分配（CalcRes, 线程/Scratch/通道） | Mesh1D, CLOS |

### 1.2 核心变化（相比旧架构）

- **删除 `IAlgPolicy` 接口层**：算法类直接继承 `AicpuTemplate`，通过虚函数多态替代委托
- **取消默认构造函数**：所有模板必须通过 Factory 以 `(OpParam, rankId, subCommRanks)` 参数化构建
- **策略成员 `protected`**：`slicePolicy_`、`topoPolicy_`、`communicationPolicy_` 对算法子类可见

---

## 2. 类继承体系

### 2.1 整体结构

```
InsAlgTemplateBase                             (ops/op_common/template/)
  |
  +-- AicpuTemplate                            (common/aicpu_template.cc)
  |     |  持有: communicationPolicy_, slicePolicy_, topoPolicy_
  |     |  虚方法: LocalDataCopy, RunAlgorithm, PostLocalCopy, GetAlgName
  |     |  实现:   KernelRun, DPUKernelRun, CalcRes, GetThreadNum, ...
  |     |
  |     +-- AllGatherMesh1DAlg                 (alg/all_gather/)
  |     |      重写: LocalDataCopy, RunAlgorithm, PostLocalCopy
  |     |      用途: 环状 Mesh 1D 全收集
  |     |
  |     +-- AllGatherNHRAlg                    (alg/all_gather/)
  |            重写: LocalDataCopy, RunAlgorithm, PostLocalCopy
  |            新增: PrepareDataSplitForMultiChannel (多通道数据切分)
  |            用途: NHR 递归倍增全收集
  |
  +-- ICommunicationPolicy                     (policies/communication/)
  |     +-- AicpuCommunication                 AICPU 设备通信 (SendRecvBatch)
  |     +-- DpuCommunication                   DPU 设备通信 (Send/Recv + DispatchToRemote)
  |
  +-- ISlicePolicy                             (policies/slice/)
  |     +-- NoSlicePolicy                      scratch 中转模式 (repeatNum 来自参数)
  |     +-- OmniSlicePolicy                    OmniPipe 零拷贝模式 (repeatNum 来自 stepSliceInfo)
  |
  +-- ITopoPolicy                              (policies/topo/)
        +-- Mesh1dTopo                         Mesh1D 拓扑: CalcChannelRequestMesh1D
        +-- ClosTopo                           CLOS 拓扑: 多通道 + CalcChannelRequestNHR
```

### 2.2 策略组合矩阵

| 模板名称 | 算法 | 设备 | 切片 | 拓扑 |
|---------|------|------|------|------|
| `AllGatherMesh1D` | `AllGatherMesh1DAlg` | `AicpuCommunication` | `NoSlicePolicy` | `Mesh1dTopo` |
| `AllGatherOmniPipeMesh1D` | `AllGatherMesh1DAlg` | `AicpuCommunication` | `OmniSlicePolicy` | `Mesh1dTopo` |
| `AllGatherNHR` | `AllGatherNHRAlg` | `AicpuCommunication` | `NoSlicePolicy` | `ClosTopo` |
| `AllGatherOmniPipeNHR` | `AllGatherNHRAlg` | `AicpuCommunication` | `OmniSlicePolicy` | `ClosTopo` |
| `AllGatherNHR_DPU` | `AllGatherNHRAlg` | `DpuCommunication` | `NoSlicePolicy` | `ClosTopo` |
| `AllGatherOmniPipeNHR_DPU` | `AllGatherNHRAlg` | `DpuCommunication` | `OmniSlicePolicy` | `ClosTopo` |

---

## 3. 公共数据结构

### 3.1 RunContext（运行上下文）

```cpp
struct RunContext {
    const OpParam* param;
    TemplateDataParams* tempAlgParams;
    TemplateResource* templateResource;
    const std::map<u32, std::vector<ChannelInfo>>* channels;
    const std::vector<std::vector<u32>>* subCommRanks;
    u32 myRank;
    u32 templateRankSize;
    HcclDataType dataType;
    bool enableRemoteMemAccess;
    bool isDmaRead;
    OpMode opMode;
};
```

### 3.2 TransferArgs（数据传输参数）

```cpp
struct TransferArgs {
    const ChannelInfo& channelSend;
    const ChannelInfo& channelRecv;
    const std::vector<DataSlice>& txSrc;
    const std::vector<DataSlice>& txDst;
    const std::vector<DataSlice>& rxSrc;
    const std::vector<DataSlice>& rxDst;
    HcclDataType dataType;
    bool isDmaRead;
    ThreadHandle* thread;
};
```

### 3.3 SliceRequest / SliceInfo（切片请求/结果）

```cpp
struct SliceRequest {
    u32 txSliceIdx;   // 发送切片在 scratch 中的编号
    u32 rxSliceIdx;   // 接收切片在 scratch 中的编号
    u32 rpt;          // 重复轮次
    u64 txSplitSize;  // 发送数据量 (考虑 tail 和多通道拆分)
    u64 txSplitOff;   // 发送偏移量
    u64 rxSplitSize;  // 接收数据量
    u64 rxSplitOff;   // 接收偏移量
};

struct SliceInfo {
    u64 dataSize;       // 数据字节数
    u64 count;          // 元素个数
    u64 localOffset;    // 本地缓冲区偏移
    u64 remoteOffset;   // 远端缓冲区偏移
};
```

### 3.4 TemplateConfig（工厂配置）

```cpp
enum class AllGatherAlgType { MESH_1D, NHR, NHR_DPU };
enum class DeviceType      { AICPU, DPU };
enum class PipeType        { NO_PIPE, OMNI_PIPE };

struct TemplateConfig {
    AllGatherAlgType algType;
    DeviceType      deviceType;
    PipeType        pipeType;
};
```

---

## 4. 接口详细声明

### 4.1 AicpuTemplate（核心模板基类）

```cpp
class AicpuTemplate : public InsAlgTemplateBase {
public:
    // --- 构造/析构 ---
    explicit AicpuTemplate(const OpParam& param, u32 rankId,
        const std::vector<std::vector<u32>>& subCommRanks);
    ~AicpuTemplate() override;

    // --- InsAlgTemplateBase 覆写 ---
    std::string Describe() const override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param,
        const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult KernelRun(const OpParam& param,
        const TemplateDataParams& tempAlgParams,
        TemplateResource& templateResource) override;
    HcclResult DPUKernelRun(const TemplateDataParams& tempAlgParam,
        const std::map<u32, std::vector<ChannelInfo>>& channels,
        u32 myRank,
        const std::vector<std::vector<u32>>& subCommRanks) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

    // --- 策略注入 ---
    void SetCommunicationPolicy(ICommunicationPolicy* communicationPolicy);
    void SetSlicePolicy(ISlicePolicy* slicePolicy);
    void SetTopoPolicy(ITopoPolicy* topoPolicy);
    void SetTemplateName(const std::string& name);

    // --- 虚方法 (算法子类覆写) ---
    virtual HcclResult LocalDataCopy(const RunContext& ctx);
    virtual HcclResult RunAlgorithm(const RunContext& ctx,
        ICommunicationPolicy* communicationPolicy);
    virtual HcclResult PostLocalCopy(const RunContext& ctx);
    virtual std::string GetAlgName() const;

protected:
    ICommunicationPolicy* communicationPolicy_;
    ISlicePolicy* slicePolicy_;
    ITopoPolicy* topoPolicy_;
    RunContext runCtx_;
    std::string templateName_;
};
```

### 4.2 ICommunicationPolicy（通信设备接口）

```cpp
class ICommunicationPolicy {
public:
    virtual ~ICommunicationPolicy() = default;

    virtual HcclResult PreRunSync(const RunContext& ctx) = 0;
    virtual HcclResult PostRunSync(const RunContext& ctx) = 0;
    virtual HcclResult TransferData(const TransferArgs& args) = 0;

    virtual bool NeedsRemoteExecution() const;
    virtual HcclResult DispatchToRemote(const RunContext& ctx,
        const std::string& algName) = 0;

    virtual void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdx, u32 threadNum) = 0;
    virtual void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdx, u32 threadNum) = 0;
    virtual std::string GetCommunicationName() const = 0;
    virtual bool IsSingleThreaded() const;
};
```

**AicpuCommunication**（本地 AICPU 执行）：
- `PreRunSync`: 线程间同步（multi-thread 场景）
- `PostRunSync`: 线程间同步收尾
- `TransferData`: `SendRecvBatchRead` / `SendRecvBatchWrite`（根据 PCIe 协议选择读写方式）
- `NeedsRemoteExecution()`: 返回 `false`
- `DispatchToRemote`: 空实现

**DpuCommunication**（远端 DPU 执行）：
- `PreRunSync`: `HcommBatchModeEnd` + `HcommThreadSynchronize`
- `PostRunSync`: `HcommBatchModeStart`
- `TransferData`: 拆分为 Send/Recv 或 SendRecvWrite（根据 remoteRank 关系）
- `NeedsRemoteExecution()`: 返回 `true`
- `DispatchToRemote`: 序列化 `DPURunInfo`，`HcommSendRequest` / `HcommWaitResponse`
- `IsSingleThreaded()`: 返回 `true`

### 4.3 ISlicePolicy（数据切片接口）

```cpp
class ISlicePolicy {
public:
    virtual ~ISlicePolicy() = default;

    virtual bool NeedsLocalDataCopy() const = 0;
    virtual bool NeedsPostLocalCopy() const = 0;
    virtual u32 GetRepeatNum(const RunContext& ctx) const = 0;

    virtual HcclResult GetSliceDescs(const RunContext& ctx,
        const SliceRequest& req,
        std::vector<SliceInfo>& txDescs,
        std::vector<SliceInfo>& rxDescs) = 0;

    virtual void* GetLocalTxBuffer(const RunContext& ctx) const = 0;
    virtual void* GetLocalRxBuffer(const RunContext& ctx) const = 0;

    virtual std::string GetSliceName() const = 0;
};
```

**NoSlicePolicy**（标准 scratch 中转模式）：
- `NeedsLocalDataCopy`: true — 需要 input → scratch
- `NeedsPostLocalCopy`: true — 需要 scratch → output
- `GetRepeatNum`: 返回 `tempAlgParams->repeatNum`
- `GetSliceDescs`: 基于 sliceSize * templateRankSize 排列，区分 enableRemoteMemAccess 下的 remote 地址策略
- `GetLocalTxBuffer / GetLocalRxBuffer`: `hcclBuff.addr + hcclBuffBaseOff`

**OmniSlicePolicy**（零拷贝 OmniPipe 模式）：
- `NeedsLocalDataCopy`: false — 直接操作 input/output
- `NeedsPostLocalCopy`: false
- `GetRepeatNum`: 返回 `stepSliceInfo.inputOmniPipeSliceStride[0].size()`
- `GetSliceDescs`: 使用 `stepSliceInfo` 中的预计算偏移和步长
- `GetLocalTxBuffer / GetLocalRxBuffer`: 直接返回 `hcclBuff.addr`

### 4.4 ITopoPolicy（拓扑资源接口）

```cpp
class ITopoPolicy {
public:
    virtual ~ITopoPolicy() = default;

    virtual HcclResult CalcRes(HcclComm comm, const OpParam& param,
        const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest,
        const RunContext& ctx) = 0;

    virtual u64 GetThreadNum(const RunContext& ctx) const = 0;
    virtual HcclResult AllocResource(AlgResourceRequest& resourceRequest,
        const RunContext& ctx) const = 0;
    virtual u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType,
        const RunContext& ctx) = 0;

    virtual u32 GetChannelsPerRank() const;
    virtual std::string GetTopoName() const = 0;
};
```

**Mesh1dTopo**：
- `CalcRes`: 调用 `CalcChannelRequestMesh1D`，单通道
- `GetThreadNum`: `templateRankSize - 1`（每步一个线程）
- `CalcScratchMultiple`: OPBASE 模式下返回 `templateRankSize`
- `GetChannelsPerRank`: 默认返回 0（不覆写，等效单通道）

**ClosTopo**：
- `CalcRes`: 调用 `CalcChannelRequestNHR` / `CalcChannelRequestNHRWithPriorityTopo`，多通道（Clos）
- `GetThreadNum`: 返回 `channelsPerRank_`
- `CalcScratchMultiple`: 返回 `templateRankSize`
- `GetChannelsPerRank`: 返回实际通道数

---

## 5. 核心流程

### 5.1 模板创建流程（TemplateFactory）

```
Step 0: ParseTemplateName(templateName, config)
        根据模板名称字符串解析出 algType / deviceType / pipeType

Step 1: CreateAlgInstance(config.algType, param, rankId, subCommRanks)
        ├─ MESH_1D → new AllGatherMesh1DAlg(param, rankId, subCommRanks)
        ├─ NHR     → new AllGatherNHRAlg(param, rankId, subCommRanks)
        └─ NHR_DPU → new AllGatherNHRAlg(param, rankId, subCommRanks)

Step 2: CreateSlicePolicy(config.pipeType)
        ├─ NO_PIPE   → new NoSlicePolicy()
        └─ OMNI_PIPE → new OmniSlicePolicy()

Step 3: CreateCommunicationPolicy(config.deviceType)
        ├─ AICPU → new AicpuCommunication()
        └─ DPU   → new DpuCommunication()

Step 4: CreateTopoPolicy(config.algType)
        ├─ MESH_1D → new Mesh1dTopo()
        └─ NHR/NHR_DPU → new ClosTopo()

Step 5: 将三个策略注入模板
        tmpl->SetSlicePolicy(slicePolicy)
        tmpl->SetTopoPolicy(topoPolicy)
        tmpl->SetCommunicationPolicy(communicationPolicy)

Step 6: return unique_ptr<AicpuTemplate>(tmpl)
```

### 5.2 KernelRun 执行流程

```
                  +---------------------------+
                  |    KernelRun(param,       |
                  |      tempAlgParams, res)  |
                  +---------------------------+
                              |
                  Step 1: 检查 communicationPolicy_ 非空
                  Step 2: sliceSize==0 && tailSize==0 → return (空数据)
                  Step 3: BuildRunContext (填充 ctx)
                  Step 4: communicationPolicy_->PreRunSync(ctx)
                              |
                  Step 5: templateRankSize_ <= 1 ?
                       /                    \
                     YES                    NO
                      |                      |
            slicePolicy_->         slicePolicy_->
            NeedsLocalDataCopy()?   NeedsLocalDataCopy()?
              ↓ YES                  ↓ YES
            this->LocalDataCopy     this->LocalDataCopy
                      |                      |
            slicePolicy_->         NeedsRemoteExecution()?
            NeedsPostLocalCopy()?     /              \
              ↓ YES              YES (DPU)         NO (AICPU)
            this->PostLocalCopy       |                |
                      |        DispatchToRemote   this->RunAlgorithm
            PostRunSync     (HcommSendRequest/    (本地执行算法)
                      |       HcommWaitResponse)
                  return                   |
                              slicePolicy_->
                              NeedsPostLocalCopy()?
                                ↓ YES
                              this->PostLocalCopy
                                    |
                        communicationPolicy_->PostRunSync
                                    |
                                return HCCL_SUCCESS
```

### 5.3 DPUKernelRun 执行流程

```
Step 1: 构造 RunContext ctx (从参数 channels / myRank / subCommRanks)
Step 2: this->RunAlgorithm(ctx, communicationPolicy_)
        (RunAlgorithm 中调用 communicationPolicy_->TransferData,
         而 DpuCommunication::TransferData 构造 SendRecvWrite 等 DPU 指令)
Step 3: return
```

### 5.4 CalcRes 执行流程

```
Step 1: topoPolicy_->CalcRes(comm, param, topoInfo, resourceRequest, runCtx_)
          (计算通道请求 → resourceRequest.channels)

Step 2: HandleAllocResource(resourceRequest, runCtx_)
          ├─ DPU (IsSingleThreaded): slaveThreadNum=0, notifyNum 置空
          └─ AICPU: topoPolicy_->AllocResource(...)
               ├─ Mesh1dTopo: 线程数 = templateRankSize - 1
               └─ ClosTopo: 线程数 = channelsPerRank_
```

---

## 6. 核心框架伪代码

### 6.1 KernelRun 调度器

```cpp
HcclResult AicpuTemplate::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    // 1. 空数据快速返回
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        return HCCL_SUCCESS;
    }

    // 2. 构建运行时上下文
    RunContext ctx;
    BuildRunContext(param, tempAlgParams, templateResource, ctx);

    // 3. 预同步 (线程 barrier / DPU eager-mode 切换)
    CHK_RET(communicationPolicy_->PreRunSync(ctx));

    // 4. 单 rank 特殊路径: 只做本地拷贝
    if (templateRankSize_ <= 1) {
        if (slicePolicy_->NeedsLocalDataCopy())
            CHK_RET(this->LocalDataCopy(ctx));   // input → scratch
        if (slicePolicy_->NeedsPostLocalCopy())
            CHK_RET(this->PostLocalCopy(ctx));   // scratch → output
        CHK_RET(communicationPolicy_->PostRunSync(ctx));
        return HCCL_SUCCESS;
    }

    // 5. 本地数据搬移 (input → scratch)
    if (slicePolicy_->NeedsLocalDataCopy())
        CHK_RET(this->LocalDataCopy(ctx));

    // 6. 算法执行 (本地 AICPU 或 远端 DPU)
    if (communicationPolicy_->NeedsRemoteExecution()) {
        CHK_RET(communicationPolicy_->DispatchToRemote(ctx, templateName_));
    } else {
        CHK_RET(this->RunAlgorithm(ctx, communicationPolicy_));
    }

    // 7. 本地数据搬回 (scratch → output)
    if (slicePolicy_->NeedsPostLocalCopy())
        CHK_RET(this->PostLocalCopy(ctx));

    // 8. 后同步 (线程 barrier / DPU batch-mode 切换)
    CHK_RET(communicationPolicy_->PostRunSync(ctx));

    return HCCL_SUCCESS;
}
```

### 6.2 Mesh1D 算法 — RunAlgorithm

```cpp
HcclResult AllGatherMesh1DAlg::RunAlgorithm(const RunContext& ctx,
    ICommunicationPolicy* communicationPolicy)
{
    u32 repeatNum = slicePolicy_->GetRepeatNum(ctx);
    u64 sliceSize = ctx.tempAlgParams->sliceSize;
    void* txLocal = slicePolicy_->GetLocalTxBuffer(ctx);
    void* rxLocal = slicePolicy_->GetLocalRxBuffer(ctx);

    // 环状迭代: 共 (templateRankSize - 1) 步
    for (step = 0; step < templateRankSize - 1; ++step) {
        u32 peer = (myAlgRank + 1 + step) % templateRankSize;

        // 组装 src/dst 切片列表 (每步多轮)
        for (rpt = 0; rpt < repeatNum; ++rpt) {
            SliceRequest req{myAlgRank, peer, rpt, effectiveSize, 0, effectiveSize, 0};
            slicePolicy_->GetSliceDescs(ctx, req, txDescs, rxDescs);

            txSrc ← (txLocal + localOff) → (remoteBuff + remoteOff) → txDst
            rxSrc ← (remoteBuff + remoteOff) → (rxLocal + localOff) → rxDst
        }

        // 通过设备策略执行数据传输
        communicationPolicy->TransferData({channels, txSrc, txDst, rxSrc, rxDst, ...});
    }
    return HCCL_SUCCESS;
}
```

### 6.3 NHR 算法 — RunAlgorithm

```cpp
HcclResult AllGatherNHRAlg::RunAlgorithm(const RunContext& ctx,
    ICommunicationPolicy* communicationPolicy)
{
    // 1. 按通道数拆分数据
    PrepareDataSplitForMultiChannel(ctx);

    u32 nSteps = GetNHRStepNum(templateRankSize);
    u32 repeatNum = slicePolicy_->GetRepeatNum(ctx);

    // 2. 多通道并行迭代
    for (channelIdx = 0; channelIdx < topoPolicy_->GetChannelsPerRank(); ++channelIdx) {
        // 3. NHR 倍增步骤
        for (step = 0; step < nSteps; ++step) {
            NHRGetStepInfo(step, nSteps, stepInfo, myAlgRank, templateRankSize);

            // 4. 每步多切片 + 多轮
            for (rpt = 0; rpt < repeatNum; ++rpt) {
                for (i = 0; i < stepInfo.nSlices; ++i) {
                    // 5. 通过 SlicePolicy 获取本地/远端偏移
                    slicePolicy_->GetSliceDescs(ctx, sliceReq, txDescs, rxDescs);

                    txSrc ← localBuff + localOff
                    txDst ← remoteBuff + remoteOff
                    rxSrc ← remoteBuff + remoteOff
                    rxDst ← localBuff + localOff
                }
            }

            // 6. 设备策略执行实际传输
            communicationPolicy->TransferData({channelSend, channelRecv, ...});
        }
    }
    return HCCL_SUCCESS;
}
```

### 6.4 工厂创建

```cpp
unique_ptr<AicpuTemplate> TemplateFactory::CreateTemplate(
    const TemplateConfig& config, const OpParam& param, u32 rankId,
    const vector<vector<u32>>& subCommRanks, const string& templateName)
{
    // 1. 创建算法子类实例 (它本身就是 AicpuTemplate)
    AicpuTemplate* tmpl = CreateAlgInstance(config.algType, param, rankId, subCommRanks);

    // 2. 创建三种策略
    ISlicePolicy* slicePolicy = CreateSlicePolicy(config.pipeType);
    ICommunicationPolicy* commPolicy = CreateCommunicationPolicy(config.deviceType);
    ITopoPolicy* topoPolicy = CreateTopoPolicy(config.algType);

    // 3. 注入策略到模板
    tmpl->SetSlicePolicy(slicePolicy);
    tmpl->SetTopoPolicy(topoPolicy);
    tmpl->SetCommunicationPolicy(commPolicy);
    tmpl->SetTemplateName(templateName);

    return unique_ptr<AicpuTemplate>(tmpl);
}
```

---

## 7. 目录结构

```
src/experimental/aicpu/
│
├── SOFTWARE_DESIGN.md                          <-- 本文档
│
├── common/
│   ├── aicpu_template.h/.cc                    核心模板基类 (继承 InsAlgTemplateBase)
│   ├── template_factory.h/.cc                  工厂 (创建+组装策略)
│   └── ...                                     (alg_v2_template_base.h 等公共头)
│
├── alg/
│   └── all_gather/
│       ├── all_gather_mesh_1d_alg.h/.cc        Mesh1D 环状算法
│       └── all_gather_nhr_alg.h/.cc            NHR 递归倍增算法
│
├── policies/
│   ├── policy_common.h                        RunContext / 工具函数
│   │
│   ├── communication/
│   │   ├── communication_policy.h             ICommunicationPolicy 接口 + TransferArgs
│   │   ├── aicpu_communication.h/.cc          AICPU 本地传输实现
│   │   └── dpu_communication.h/.cc           DPU 远端传输 + DispatchToRemote 实现
│   │
│   ├── slice/
│   │   ├── slice_policy.h                     ISlicePolicy 接口 + SliceRequest/SliceInfo
│   │   ├── no_slice.h/.cc                     NoSlice 标准 scratch 中转
│   │   └── omni_slice.h/.cc                   OmniSlice 零拷贝 OmniPipe
│   │
│   └── topo/
│       ├── topo_policy.h                      ITopoPolicy 接口
│       ├── mesh_1d_topo.h/.cc                 Mesh1D 拓扑资源
│       └── clos_topo.h/.cc                    CLOS 拓扑资源 (多通道)
```

---

## 8. 扩展指南

### 8.1 新增算法（以新算子 ReduceScatter 为例）

1. 在 `alg/reduce_scatter/` 下创建 `ReduceScatterNHRAlg`，继承 `AicpuTemplate`
2. 覆写 `LocalDataCopy`、`RunAlgorithm`、`PostLocalCopy`、`GetAlgName`
3. 在 `TemplateConfig` 中新增算法枚举
4. 在 Factory 的 `CreateAlgInstance` 中添加分支
5. 复用现有 `ICommunicationPolicy`、`ISlicePolicy`、`ITopoPolicy`

### 8.2 新增拓扑策略

1. 在 `policies/topo/` 下新建类，实现 `ITopoPolicy`
2. 覆写 `CalcRes`（通道计算）、`GetThreadNum`、`AllocResource`、`CalcScratchMultiple`
3. 在 Factory 的 `CreateTopoPolicy` 中添加分支

### 8.3 新增通信设备策略

1. 在 `policies/communication/` 下新建类，实现 `ICommunicationPolicy`
2. 覆写 `PreRunSync`、`PostRunSync`、`TransferData`、`DispatchToRemote`
3. 在 Factory 的 `CreateCommunicationPolicy` 中添加分支

---

*文档最后更新: 2026-06-10*