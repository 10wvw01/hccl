#ifndef ALL_REDUCE_H
#define ALL_REDUCE_H

#include <stdint.h>

// 枚举定义（兼容HCCL风格）
typedef enum { HCCL_SUCCESS = 0, HCCL_FAILED = 1 } Hcc1Result;

typedef enum { HCCL_DATA_TYPE_FP32 = 0, HCCL_DATA_TYPE_FP16 = 1, HCCL_DATA_TYPE_INT32 = 2 } Hcc1DataType;

typedef enum { HCCL_REDUCE_SUM = 0, HCCL_REDUCE_MAX = 1, HCCL_REDUCE_MIN = 2, HCCL_REDUCE_PROD = 3 } Hcc1ReduceOp;

// 不透明句柄（实际使用中由底层管理）
typedef void *Hcc1Comm;
typedef void *aclrtStream;

#ifdef __cplusplus
extern "C" {
#endif

// 核心接口：完全符合题目给出的函数原型
Hcc1Result Hcc1AllReduce(void *sendBuf, void *recvBuf, uint64_t count, Hcc1DataType dataType, Hcc1ReduceOp op,
    Hcc1Comm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ALL_REDUCE_H