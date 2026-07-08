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
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    commToTagMap_[comm].push_back(tagName);
}

const std::vector<std::string> &AicpuTaskCacheCommManager::GetTagsByCommName(HcclComm comm) const
{
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = commToTagMap_.find(comm);
    if (it != commToTagMap_.end()) {
        return it->second;
    }
    static const std::vector<std::string> emptyVec;
    return emptyVec;
}

std::vector<HcclComm> AicpuTaskCacheCommManager::GetAllCommNames() const
{
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    std::vector<HcclComm> commNames;
    commNames.reserve(commToTagMap_.size());
    for (const auto &pair : commToTagMap_) {
        commNames.push_back(pair.first);
    }
    return commNames;
}

void AicpuTaskCacheCommManager::RemoveCommTagMapByCommName(HcclComm comm)
{
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    commToTagMap_.erase(comm);
}

void AicpuTaskCacheCommManager::evitTaskCache(HcclComm comm)
{
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = commToTagMap_.find(comm);
    if (it != commToTagMap_.end()) {
        if (HcommIsSupportHcommAicpuTsTaskCacheClear()) {
            for (const auto &tag : it->second) {
                HCCL_INFO("[evitTaskCache] comm[%p] clear cache tag[%s]", comm, tag.c_str());
                CHK_PRT(static_cast<HcclResult>(HcommAicpuTsTaskCacheClear(tag.c_str())));
            }
        }
        commToTagMap_.erase(comm);
    }
}

void AicpuTaskCacheCommManager::evitAllTaskCache()
{
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    if (HcommIsSupportHcommAicpuTsTaskCacheClear()) {
        for (const auto &pair : commToTagMap_) {
            for (const auto &tag : pair.second) {
                HCCL_INFO("[evitAllTaskCache] clear cache tag[%s]", tag.c_str());
                CHK_PRT(static_cast<HcclResult>(HcommAicpuTsTaskCacheClear(tag.c_str())));
            }
        }
    }
    commToTagMap_.clear();
}

} // namespace ops_hccl
