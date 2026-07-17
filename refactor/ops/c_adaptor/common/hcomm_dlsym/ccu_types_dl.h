#ifndef CCU_TYPES_DL_H
#define CCU_TYPES_DL_H
#include <cstdint>
#include "dlsym_common.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { CCU_SUCCESS = 0 } CcuStatus;
typedef struct { uint32_t dummy; } CcuInsHandle;
typedef uint32_t CcuKernelHandle;
#ifdef __cplusplus
}
#endif
#endif
