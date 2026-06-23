# Barrier AICPU 引擎支持需求文档

## 问题背景

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
- 新增 executor 注册 `InsBarrierMesh1DNhrAicpu`
- 修改 selector 增加 `SelectAicpuAlgo` 路径
- 修改 `barrier_op.cc` 增加引擎白名单，HostDPU 和 AICPU 引擎走新流程，其余回退

### Out of Scope

- CCU 引擎 barrier
- AIV 引擎 barrier
- 并行执行优化（方案⑥）
- 拍平算法（方案③③'④）
- 现有 HostDPU 路径的任何改动

## 场景路由

```
950 + HostDPU 引擎 → selector 选 InsBarrierMeshNhrDPU（串行 DPU，已有，不改）
950 + AICPU 引擎   → selector 选 InsBarrierMesh1DNhrAicpu（串行 AICPU，新增）
950 + CCU/AIV 引擎 → 本次不涉及，回退旧 HcclBarrier
非950              → 回退旧 HcclBarrier（不变）
```

## 涉及文件

| 文件 | 操作 |
|------|------|
| `src/ops/barrier/template/aicpu/ins_temp_barrier_nhr_aicpu.cc` | 新增 |
| `src/ops/barrier/template/aicpu/ins_temp_barrier_nhr_aicpu.h` | 新增 |
| `src/ops/barrier/template/aicpu/CMakeLists.txt` | 加新文件 |
| `src/ops/barrier/executor/ins_v2_barrier_sequence_executor.cc` | 新增注册 |
| `src/ops/barrier/selector/barrier_auto_selector.cc` | 加 SelectAicpuAlgo |
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

### Selector 增加 AICPU 路径

```cpp
SelectorStatus SelectAicpuAlgo(...) {
    selectAlgName = "InsBarrierMesh1DNhrAicpu";
    return SelectorStatus::MATCH;
}
```

## 成功标准

1. 950 芯片 AICPU 引擎场景，barrier 走新 AICPU 流程，不再回退到旧 AllReduce
2. HostDPU 引擎场景行为不变，仍走 DPU 路径
3. CCU/AIV 引擎场景及非 950 场景仍回退旧 HcclBarrier
4. AICPU barrier 不搬运任何数据，仅做 signal 同步
5. 代码编译通过（AICPU 编译模式）
6. Selector 返回 NOT_MATCH 时回退旧 HcclBarrier，不崩溃

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
    HcclResult RunNHRBarrier(const std::map<u32, std::vector<ChannelInfo>> &channels);
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
    CHK_RET(RunNHRBarrier(templateResource.channels));

    HCCL_INFO("[InsTempBarrierNhrAicpu] Run End");
    return HCCL_SUCCESS;
}

u32 InsTempBarrierNhrAicpu::GetRankFromMap(const uint32_t rankIdx) const
{
    return subCommRanks_[0].at(rankIdx);
}

HcclResult InsTempBarrierNhrAicpu::RunNHRBarrier(
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    // 与 DPU 版 RunNHRBarrier 逻辑完全一致，去掉 #ifndef AICPU_COMPILE 保护
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
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] SendRecvWrite failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else if (txChannel[0].remoteRank < rxChannel[0].remoteRank) {
            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] Send failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] Recv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] Recv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo),
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
| KernelRun | `HcommBatchModeEnd` → `HcommSendRequest` → `HcommWaitResponse` → `HcommBatchModeStart` | 直接调 `RunNHRBarrier` |
| RunNHRBarrier | `#ifndef AICPU_COMPILE` 保护（仅 DPU 侧编译） | 无保护，AICPU 直接编译执行 |
| CalcRes | 相同 | 相同 |
| DPUKernelRun | 有（DPU 侧回调） | 无（不需要） |

### 3. 修改 `template/aicpu/CMakeLists.txt`

```cmake
set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_barrier_mesh_1D.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_barrier_nhr_dpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_barrier_nhr_aicpu.cc   # 新增
)
```

### 4. 修改 `executor/ins_v2_barrier_sequence_executor.cc`

在现有注册之后新增一行：

```cpp
// 已有：DPU 版
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_BARRIER,
                               InsBarrierMeshNhrDPU,
                               InsV2BarrierSequenceExecutor,
                               TopoMatchMultilevel,
                               InsTempBarrierMesh1D,
                               InsTempBarrierNHRDPU);

// 新增：AICPU 版
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

executor 本身不需要改动——`InsV2BarrierSequenceExecutor` 是模板化的，两个注册共用同一个 executor 实现，只是模板参数不同。

### 5. 修改 `selector/barrier_auto_selector.h`

新增 `SelectAicpuAlgo` 声明：

```cpp
class BarrierAutoSelector : public AutoSelectorBase {
private:
    SelectorStatus SelectDPUAlgo(...) const override;      // 已有
    SelectorStatus SelectAicpuAlgo(...) const override;    // 新增
};
```

### 6. 修改 `selector/barrier_auto_selector.cc`

新增 `SelectAicpuAlgo` 实现：

```cpp
SelectorStatus BarrierAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap;
    HCCL_DEBUG("[BarrierAutoSelector][SelectAicpuAlgo] start, topoLevelNums[%u]", topoInfo->topoLevelNums);
    selectAlgName = "InsBarrierMesh1DNhrAicpu";
    HCCL_DEBUG("[BarrierAutoSelector][SelectAicpuAlgo] Algo match[%s]", selectAlgName.c_str());
    return SelectorStatus::MATCH;
}
```

### 7. 修改 `barrier_op.cc`

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
| `template/aicpu/CMakeLists.txt` | 修改 | +1 行 |
| `executor/ins_v2_barrier_sequence_executor.cc` | 修改 | +6 行（include + 注册） |
| `selector/barrier_auto_selector.h` | 修改 | +3 行 |
| `selector/barrier_auto_selector.cc` | 修改 | +12 行 |
| `barrier_op.cc` | 修改 | ~8 行（白名单替换） |
| **总计** | | **~170 行** |

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
