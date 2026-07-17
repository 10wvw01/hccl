/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * op_common UT stub functions.
 */

#include <cstring>
#include <cstdarg>
#include <cstdint>
#include <vector>

#include "test_helpers.h"
#include "hccl_algorithm.h"
#include "binary_stream.h"

// ───────────── log stub (src/common/log.cc 依赖) ─────────────
bool IsErrorToWarn() { return false; }
bool HcclCheckLogLevel(int, int) { return false; }

extern "C" errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (dest == nullptr || count > destMax) { return 1; }
    (void)memset(dest, c, count);
    return 0;
}

// snprintf_s 桩（securec 库符号, UT 环境可能缺失）
int snprintf_s(char *dest, size_t destMax, size_t count, const char *format, ...)
{
    (void)destMax; (void)count;
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(dest, destMax, format, args);
    va_end(args);
    return ret;
}

// strncpy_s 桩（securec 库符号, UT 环境可能缺失）
#ifndef EOK
#define EOK 0
#endif
#ifndef EINVAL
#define EINVAL 22
#endif
errno_t strncpy_s(char *strDest, size_t destMax, const char *strSrc, size_t count)
{
    if (strDest == nullptr || strSrc == nullptr || destMax == 0) { return EINVAL; }
    size_t copyLen = (count < destMax) ? count : destMax - 1;
    (void)memcpy(strDest, strSrc, copyLen);
    strDest[copyLen] = '\0';
    return EOK;
}

// ───────────── HcclAlgorithm 序列化桩 (C++ 链接) ─────────────
namespace ops_hccl {
void HcclAlgorithm::SerializeTo(BinaryStream &) const {}
void HcclAlgorithm::DeserializeFrom(BinaryStream &) {}
void AlgoExecDesc::Serialize(BinaryStream &, const AlgoExecDesc &) {}
AlgoExecDesc AlgoExecDesc::Deserialize(BinaryStream &) { return AlgoExecDesc{}; }
} // namespace ops_hccl
