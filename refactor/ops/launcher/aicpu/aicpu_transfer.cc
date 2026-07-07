// ============================================================
// file: src/transfer/engines/aicpu/launcher/aicpu_transfer.cc
//
// AICPU 引擎数据传输实现。
// ============================================================

#include "transfer.h"
#include "aicpu_launcher.h"
#include "alg_data_trans_wrapper.h"

// ───────────── 主 Send ─────────────
HcclResult AiCpuLauncher::Send(const TransferContext &ctx) {
    TransferDirection direction = WRITE;
    if (ctx.enableRemoteMemAccess == true && ctx.buffType == OUTPUT) {
        direction = READ;
    } else {
        direction = WRITE;
    }

    DataInfo sendInfo = BuildDataInfo(ctx);
    const ThreadHandle thread

    // 主路由矩阵
    switch (direction) {
        case TransferDirection::WRITE:
            if (ctx.reduceOp != NONE) {
                return SendReduce(&sendInfo, ctx.dataType, ctx.reduceOp, &ctx.templateRes.threads);
            }
            return SendWrite(sendInfo);

        case TransferDirection::READ:
            if (ctx.reduceOp != NONE) {
                return SendReadReduce(&sendInfo, ctx.dataType, ctx.reduceOp, &ctx.templateRes.threads);
            }
            return  SendRead(&sendInfo, &ctx.templateRes.threads)
    }

    return HCCL_ERR_INVALID_PARAM;
}

