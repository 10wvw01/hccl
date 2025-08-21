/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_base.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "workflow_pub.h"
#include "param_check_pub.h"
#include "rank_consistentcy_checker.h"
#include "externalinput_pub.h"
#include "env_config.h"
#include "detect_connect_anomalies.h"
#include "../common/src/topo/topoinfo_detect.h"
#include "../common/src/topo/topoinfo_ranktable_partition.h"
#include "../common/src/state_guard.h"
#include "sal_pub.h"
#include "profiling_manager_pub.h"
#include "adapter_prof.h"
#include "adapter_qos_pub.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "mem_host_pub.h"
#include "hcom_common.h"
#include "comm_config_pub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "external/runtime/rt_error_codes.h"
#include "mmpa_api.h"

using namespace std;
using namespace hccl;

HcclResult GetCaptureInfo(aclrtStream stream, rtStreamCaptureStatus &captureStatus, uint32_t &modelId, bool &isCapture)
{
   HCCL_WARNING("[%s]Stream capture does not support!", __func__);
   return HCCL_SUCCESS;
}
