#ifndef ALL_REDUCE_H
#define ALL_REDUCE_H

#include <stdint.h>

// 枚举定义（兼容HCCL风格）
typedef enum {
    HCCL_SUCCESS = 0,
    HCCL_FAILED = 1
} HcclResult;

typedef enum {
    HCCL_DATA_TYPE_FP32 = 0,
    HCCL_DATA_TYPE_FP16 = 1,
    HCCL_DATA_TYPE_INT32 = 2
} HcclDataType;

typedef enum {
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_MAX = 1,
    HCCL_REDUCE_MIN = 2,
    HCCL_REDUCE_PROD = 3
} HcclReduceOp;

// 不透明句柄（实际使用中由底层管理）
typedef void* HcclComm;
typedef void* aclrtStream;

#ifdef __cplusplus
extern "C" {
#endif

// 核心接口：完全符合题目给出的函数原型
HcclResult HcclAllReduce(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    HcclReduceOp op,
    HcclComm comm,
    aclrtStream stream
);

#ifdef __cplusplus
}
#endif

#endif // ALL_REDUCE_H