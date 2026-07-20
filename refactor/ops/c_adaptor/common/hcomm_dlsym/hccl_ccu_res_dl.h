#ifndef HCCL_CCU_RES_DL_H
#define HCCL_CCU_RES_DL_H
#include "dlsym_common.h"
#include "hccl_types.h"
#include "ccu_types_dl.h"

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(HcclResult, HcclCommQueryCcuIns, HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum);
DECL_SUPPORT_FLAG(HcclCommQueryCcuIns);

void HcclCcuResDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCL_CCU_RES_DL_H
