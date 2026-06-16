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
#include <mutex>

namespace ops_hccl {

AicpuTaskCacheCommManager& AicpuTaskCacheCommManager::Instance()
{
    static AicpuTaskCacheCommManager instance;
    return instance;
}

void AicpuTaskCacheCommManager::AddCommTagMap(const std::string& commName, const std::string& tagName)
{
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    commToTagMap_[commName].push_back(tagName);
}

const std::vector<std::string>& AicpuTaskCacheCommManager::GetTagsByCommName(const std::string& commName) const
{
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = commToTagMap_.find(commName);
    if (it != commToTagMap_.end()) {
        return it->second;
    }
    static const std::vector<std::string> emptyVec;
    return emptyVec;
}

std::vector<std::string> AicpuTaskCacheCommManager::GetAllCommNames() const
{
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    std::vector<std::string> commNames;
    commNames.reserve(commToTagMap_.size());
    for (const auto &pair : commToTagMap_) {
        commNames.push_back(pair.first);
    }
    return commNames;
}

void AicpuTaskCacheCommManager::RemoveCommTagMapByCommName(const std::string &commName)
{
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    commToTagMap_.erase(commName);
}

} // namespace ops_hccl
