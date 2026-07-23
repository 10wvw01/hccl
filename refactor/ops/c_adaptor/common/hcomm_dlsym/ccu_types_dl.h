#ifndef CCU_TYPES_DL_H
#define CCU_TYPES_DL_H
#include <cstdint>
#include "dlsym_common.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
#include "ccu_types.h"
#else
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { CCU_SUCCESS = 0 } CcuStatus;
typedef CcuStatus CcuResult;
typedef struct { uint32_t dummy; } CcuInsHandle;
typedef uint32_t CcuKernelHandle;
#ifdef __cplusplus
}
#endif
#endif
#endif
