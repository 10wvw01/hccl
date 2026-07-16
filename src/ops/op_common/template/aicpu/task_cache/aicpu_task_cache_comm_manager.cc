/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_comm_manager.h"
#include "log.h"
#include "hcomm_primitives_dl.h"
#include <mutex>

namespace ops_hccl {

AicpuTaskCacheCommManager &AicpuTaskCacheCommManager::Instance()
{
    static AicpuTaskCacheCommManager instance;
    return instance;
}

void AicpuTaskCacheCommManager::AddCommTagMap(HcclComm comm, const std::string &tagName)
{
    HCCL_DEBUG("[%s] comm[%p] tagName[%s]", __func__, comm, tagName.c_str());
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    commToTagMap_[comm].push_back(tagName);
}

void AicpuTaskCacheCommManager::EvitTaskCache(HcclComm comm)
{
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = commToTagMap_.find(comm);
    if (it != commToTagMap_.end()) {
        if (HcommIsSupportHcommAicpuTsTaskCacheClear()) {
            for (const auto &tag : it->second) {
                HCCL_INFO("[EvitTaskCache] comm[%p] clear cache tag[%s]", comm, tag.c_str());
                CHK_PRT(static_cast<HcclResult>(HcommAicpuTsTaskCacheClear(tag.c_str())));
            }
        }
        commToTagMap_.erase(comm);
    }
}

void AicpuTaskCacheCommManager::EvitAllTaskCache()
{
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    if (HcommIsSupportHcommAicpuTsTaskCacheClear()) {
        for (const auto &pair : commToTagMap_) {
            for (const auto &tag : pair.second) {
                HCCL_INFO("[EvitAllTaskCache] clear cache tag[%s]", tag.c_str());
                CHK_PRT(static_cast<HcclResult>(HcommAicpuTsTaskCacheClear(tag.c_str())));
            }
        }
    }
    commToTagMap_.clear();
}

} // namespace ops_hccl
