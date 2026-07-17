/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_ENGINE_H
#define OPS_HCCL_AICPU_ENGINE_H

#include "base_engine.h"
#include "kernel_launch.h"

namespace ops_hccl {

/**
 * AICPU 引擎 Engine
 * 职责：实现 AICPU 引擎的资源创建与 kernel 下发能力。
 * 资源创建：通过 HcommChannelCreate/HcommThreadCreate/HcommNotifyCreate 创建 channel、notify、thread；
 * Kernel 下发：通过 HcclLaunchAicpuKernel 入口完成环境准备、算法编排与 profiling 上报。
 * resCtx_ 成员：CreateRes 创建的资源句柄回填到 resCtx_，供 LaunchKernel 直接使用，无需重新反序列化。
 */
class AiCpuEngine : public BaseEngine {
public:
    AiCpuEngine() = default;
    ~AiCpuEngine() override = default;

    /**
     * 创建 AICPU 引擎所需的运行时资源。
     * 工作流程：
     *   1. 将 algHierarchyInfo 和序列化的 HcclAlgorithm 存入 resCtx_；
     *   2. 遍历每层级的 channels，调用 HcommChannelCreate 创建通信通道并回填句柄到 resCtx_；
     *   3. 根据 slaveThreadNum 调用 HcommThreadCreate 创建主线程与从线程，回填到 resCtx_.threads；
     *   4. 根据 notifyNumPerThread 与 notifyNumOnMainThread 调用 HcommNotifyCreate 创建同步通知；
     *   5. 申请跨 Rank 缓存 cclMem（HcclMalloc），回填到 resCtx_.cclMem。
     * 输入参数：
     *   - comm: 通信域句柄
     *   - alg: 算法描述对象引用，序列化后存入 resCtx_ 供 device 侧重建 executor
     *   - algHierarchyInfo: 拓扑分级信息，由 CalcAlgHierarchyInfo 生成
     *   - resReq: 资源请求，由 OpsExecutor::CalcRes 生成
     * 返回值：
     *   - HCCL_SUCCESS: 资源创建成功
     *   - HCCL_E_INTERNAL: 资源创建失败
     */
    HcclResult CreateRes(HcclComm comm, const OpParam &param, HcclAlgorithm &alg,
                         AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq,
                         TopoInfoWithNetLayerDetails &topoInfo) override;

    /**
     * 下发 AICPU kernel 到设备侧执行。
     * 工作流程：
     *   1. 加载 AICPU kernel 二进制（LoadAICPUKernel）；
     *   2. 调用 HcclLaunchAicpuKernel 完成环境准备、算法编排与 profiling 上报：
     *      a. 获取通信域句柄；
     *      b. 从 resCtx_ 反序列化 HcclAlgorithm，重建 executor；
     *      c. 根据 opType 还原变长数据；
     *      d. 设置 batch mode，注册 DFX 信息；
     *      e. 主 thread 等待 Host stream 的 notify 通知；
     *      f. 调用 executor.CalcRes + executor.Orchestrate 驱动算法编排；
     *      g. 上报 profiling，通知 Host stream 完成，结束 batch mode。
     *   3. 释放通信域句柄。
     * 输入参数：
     *   - param: 算子参数，包含 commName、tag、opType、数据描述等
     * 返回值：
     *   - HCCL_SUCCESS: kernel 下发并执行成功
     *   - HCCL_E_INTERNAL: 下发或执行失败
     */
    HcclResult LaunchKernel(const OpParam &param) override;

    /**
     * AICPU引擎数据传输接口。
     */
    HcclResult Send(const TransferContext &ctx) override;

private:
    // 已创建的资源上下文，CreateRes 回填、LaunchKernel 使用
    AlgResourceCtxSerializable resCtx_;
    // 序列化后的 resCtx_ 字节流，生命周期需覆盖 kernel launch
    std::vector<char> resCtxSequence_;

    // ───────────── AICPU 数据传输 wrapper (私有成员函数) ─────────────
    // 基于 Hcomm*OnThread 系列 AICPU 专用原语, 复制自 alg_data_trans_wrapper.cc


    // Write 系列 (非PCIe: 本端主动推送)
    HcclResult SendWrite(const DataInfo &sendInfo, const ThreadHandle &thread, HcclReduceOp reduceOp);
    HcclResult RecvWrite(const DataInfo &recvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp);
    HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp);
    // Read 系列 (PCIe: 本端主动拉取)
    HcclResult SendRead(const DataInfo &sendInfo, const ThreadHandle &thread, HcclReduceOp reduceOp);
    HcclResult RecvRead(const DataInfo &recvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp);
    HcclResult SendRecvRead(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp);

    // CreateRes 内部辅助函数（对应原始 op_common.cc 中的 HcclGetThread / HcclGetChannelImpl 等）
    HcclResult HcclGetThread(HcclComm comm, const OpParam &param, AlgResourceRequest &resReq);
    HcclResult SaveMainThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle thread, u32 notifyNum);
    HcclResult SaveUnfoldThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle unfoldThread);
    HcclResult HcclGetChannel(HcclComm comm, const OpParam &param, AlgResourceRequest &resReq);
    HcclResult HcclGetChannelImpl(u32 level, HcclComm comm, const OpParam &param,
        std::vector<HcclChannelDesc>& channelRequest, CommEngine commEngine);
    HcclResult AddExchangeInfo(HcclComm comm, const OpParam &param);

    // LaunchKernel 内部辅助函数（对应原始 op_common.cc 中的 HcclAicpuKernelEntranceLaunch / AicpuKernelLaunch）
    HcclResult HcclAicpuKernelEntranceLaunch(const OpParam &param);
    HcclResult AicpuKernelLaunch(const OpParam &param, ThreadHandle unfoldThread);

};

}  // namespace ops_hccl

#endif  // OPS_HCCL_AICPU_ENGINE_H
