#ifndef OPS_HCCL_EXEC_OP_H
#define OPS_HCCL_EXEC_OP_H

#include "common.h"
#include "ccu_kernel.h"

namespace ops_ccu {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
}

#endif // OPS_HCCL_EXEC_OP_H
