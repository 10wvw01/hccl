/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_BASE_ENGINE_H
#define OPS_HCCL_BASE_ENGINE_H

#include "hccl/base.h"
#include "alg_param.h"

namespace ops_hccl {

// ★★★ 统一数据传输上下文 ★★★
// 调用者填充, 传给 BaseEngine::Send()。
struct TransferContext {
    // ──── 数据传输描述 ────
    bool enableRemoteMemAccess = true;             // 是否可直接访问对端 input/output
    BufferType buffType = BufferType::OUTPUT;       // 当前操作的 buffer 类型
    TxRxSlicesList txRxSlicesList;                  // 收发数据切片 + rank 信息
    TemplateResource templateRes;                   // 资源 (channels map + threads vector)

    // ──── Reduce 参数 (reduceOp != HCCL_REDUCE_RESERVED 时需要) ────
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;

    // ──── 扩展 (引擎专用参数透传, 不建议常规使用) ────
    void* reserved = nullptr;
};

class HcclAlgorithm;


/**
 * 引擎 Engine 基类
 * 职责：抽象不同设备引擎（AICPU/AIV/CCU）的 Kernel 下发与资源创建能力。
 * 不同引擎子类实现各自的资源创建方式与 kernel 下发流程：
 *   - AICPU: 创建 channel、notify、thread，下发 AICPU kernel
 *   - AIV:   创建 channel、共享内存，下发 AIV kernel
 *   - CCU:   创建 cclMem、notify、thread、channel，下发 CCU kernel
 */
class BaseEngine {
public:
    BaseEngine() = default;
    virtual ~BaseEngine() = default;

    /**
     * 创建引擎所需的运行时资源。
     * 工作流程：
     *   1. 根据 resReq 中的资源需求，创建对应的资源（channel/notify/thread/cclMem 等）；
     *   2. 将创建的资源句柄、algHierarchyInfo、序列化的 HcclAlgorithm 回填到引擎内部 resCtx_ 中；
     *   3. resCtx_ 供后续 LaunchKernel 在 device 侧使用。
     * 输入参数：
     *   - comm: 通信域句柄
     *   - alg: 算法描述对象引用，序列化后存入 resCtx_ 供 device 侧重建 executor
     *   - algHierarchyInfo: 拓扑分级信息，由 CalcAlgHierarchyInfo 生成
     *   - resReq: 资源请求，由 OpsExecutor::CalcRes 生成
     * 返回值：
     *   - HCCL_SUCCESS: 资源创建成功
     *   - 其他: 资源创建失败
     */
    virtual HcclResult CreateRes(HcclComm comm, const OpParam &param, HcclAlgorithm &alg,
                                 AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq) = 0;

    /**
     * 下发 kernel 到设备侧执行。
     * 工作流程：
     *   1. 准备执行环境（加载 kernel 二进制、初始化 thread）；
     *   2. 从引擎内部 resCtx_ 反序列化 HcclAlgorithm，重建 executor 并执行编排；
     *   3. 等待设备侧执行完成并上报 profiling。
     * 输入参数：
     *   - param: 算子参数，包含 commName、tag、opType、数据描述等
     * 返回值：
     *   - HCCL_SUCCESS: kernel 下发并执行成功
     *   - HCCL_E_INTERNAL: 下发或执行失败
     */
    virtual HcclResult LaunchKernel(const OpParam &param) = 0;

    /**
     * 数据传输统一接口。
     * 工作流程：
     *   1. 解析 ctx 中的channel信息和数据信息；
     *   2. 根据入参选择发送方式（write\read）\reduce；
     *   3. 将数据发送到目的地址。
     * 输入参数：
     *   - ctx: 发送数据上下文
     * 返回值：
     *   - HCCL_SUCCESS: 数据发送成功
     *   - HCCL_E_INTERNAL: 数据发送失败
     */
    virtual HcclResult Send(const TransferContext &ctx) = 0;
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_BASE_ENGINE_H
