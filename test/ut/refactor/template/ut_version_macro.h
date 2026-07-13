/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Version macro used by UT compile command before public headers are parsed.
 */

#ifndef HCCL_UT_VERSION_MACRO_H
#define HCCL_UT_VERSION_MACRO_H

#ifndef CANN_VERSION
#define CANN_VERSION(major, minor, patch) ((major) * 10000000 + (minor) * 100000 + (patch) * 1000)
#endif

#endif // HCCL_UT_VERSION_MACRO_H
