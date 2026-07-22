/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <sstream>
#include <memory>
#include <cstring>
#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <mutex>

#include "alg_param.h"
#include "ops_executor.h"
#include "hccl_algorithm.h"
#include "kernel_launch.h"
#include "hcomm_primitives.h"
#include "hcomm_primitives_dl.h"
#include "dfx/task_exception_fun.h"
#include "hcomm_diag_dl.h"
#include "hcomm_device_profiling_dl.h"
#include "hccl_device_comm_dl.h"
#include "exec_timeout_manager.h"
#include "binary_stream.h"
#include "log.h"

using namespace ops_hccl;

namespace {

// ───────────── BatchTransfer 支持状态（搬自 alg_data_trans_wrapper.cc）─────────────
enum HcommBatchTransferSupportState {
    HCOMM_BATCH_TRANSFER_UNINIT = -1,
    HCOMM_BATCH_TRANSFER_UNSUPPORTED = 0,
    HCOMM_BATCH_TRANSFER_SUPPORTED = 1,
};

std::atomic<int> g_hcommBatchTransferSupportState{HCOMM_BATCH_TRANSFER_UNINIT};

HcclResult InitHcommBatchTransferOnThreadSupported(bool isSupported)
{
    int target = isSupported ? HCOMM_BATCH_TRANSFER_SUPPORTED : HCOMM_BATCH_TRANSFER_UNSUPPORTED;
    int expected = HCOMM_BATCH_TRANSFER_UNINIT;
    if (g_hcommBatchTransferSupportState.compare_exchange_strong(expected, target)) {
        return HCCL_SUCCESS;
    }

    if (expected != target) {
        HCCL_ERROR("[AlgDataTransWrapper] HcommBatchTransferOnThread support mismatch, cached[%d], ctx[%d].",
            expected, target);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// 统计缓存信息
struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};

    double hitRate() const
    {
        uint64_t total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / total : 0.0;
    }

    void Reset()
    {
        hits = 0;
        misses = 0;
    }
};

// 通信域缓存
class CommDomainCache {
public:
    explicit CommDomainCache(const std::string &commName) : commName_(commName) {}

    const std::string &GetCommName() const { return commName_; }

    // 获得缓存项，返回共享所有权保证使用期间对象稳定存活
    std::shared_ptr<const AlgResourceCtxSerializable> Get(const std::string &algTag)
    {
        std::shared_lock<std::shared_timed_mutex> lock(mutex_);
        auto it = cache_.find(algTag);
        return it != cache_.end() ? it->second : nullptr;
    }

    // 缓存算法
    void Put(const std::string &algTag, const AlgResourceCtxSerializable &value)
    {
        std::unique_lock<std::shared_timed_mutex> lock(mutex_);
        cache_[algTag] = std::make_shared<AlgResourceCtxSerializable>(value);
    }

    // 移除特定算法
    bool Remove(const std::string &algTag)
    {
        std::unique_lock<std::shared_timed_mutex> lock(mutex_);
        return cache_.erase(algTag) > 0;
    }

    // 清空所有缓存项
    void Clear()
    {
        std::unique_lock<std::shared_timed_mutex> lock(mutex_);
        cache_.clear();
    }

    CacheStats &GetStats() { return stats_; }
    const CacheStats &GetStats() const { return stats_; }

    size_t GetCacheSize() const
    {
        std::shared_lock<std::shared_timed_mutex> lock(mutex_);
        return cache_.size();
    }

private:
    std::string commName_;
    std::unordered_map<std::string, std::shared_ptr<const AlgResourceCtxSerializable>> cache_;
    CacheStats stats_;
    mutable std::shared_timed_mutex mutex_;
};

// 通信域缓存管理器
class CommDomainCacheManager {
public:
    // 获取算法缓存
    std::shared_ptr<const AlgResourceCtxSerializable> Get(const std::string &algTag,
        const std::string &paramCommName)
    {
        std::string commName = ExtractCommName(algTag);
        if (commName.empty()) {
            commName = paramCommName;
        }

        CommDomainCache *commCache = GetOrCreateComm(commName);
        if (commCache != nullptr) {
            auto &stats = commCache->GetStats();
            auto result = commCache->Get(algTag);
            if (result) {
                stats.hits++;
                return result;
            }
            stats.misses++;
        }
        return nullptr;
    }

    // 缓存算法结果
    void Put(const std::string &algTag, const AlgResourceCtxSerializable &value,
        const std::string &paramCommName)
    {
        std::string commName = ExtractCommName(algTag);
        if (commName.empty()) {
            commName = paramCommName;
        }

        CommDomainCache *commCache = GetOrCreateComm(commName);
        if (commCache != nullptr) {
            commCache->Put(algTag, value);
        }
    }

    // 释放通信域缓存
    bool ReleaseComm(const std::string &commName)
    {
        std::unique_lock<std::shared_timed_mutex> lock(mapMutex_);
        return commCaches_.erase(commName) > 0;
    }

    // 获得通信域统计信息
    bool GetCommStats(const std::string &commName, CacheStats &outStats, size_t &outCacheSize) const
    {
        std::shared_lock<std::shared_timed_mutex> lock(mapMutex_);
        auto it = commCaches_.find(commName);
        if (it != commCaches_.end()) {
            outStats.hits = it->second.GetStats().hits.load();
            outStats.misses = it->second.GetStats().misses.load();
            outCacheSize = it->second.GetCacheSize();
            return true;
        }
        return false;
    }

    // 获得全局统计信息
    void GetGlobalStats(size_t &totalCommDomains, size_t &totalCacheEntries,
        uint64_t &totalHits, uint64_t &totalMisses) const
    {
        std::shared_lock<std::shared_timed_mutex> lock(mapMutex_);
        totalCommDomains = commCaches_.size();
        totalCacheEntries = 0;
        totalHits = 0;
        totalMisses = 0;
        for (const auto &pair : commCaches_) {
            totalCacheEntries += pair.second.GetCacheSize();
            totalHits += pair.second.GetStats().hits.load();
            totalMisses += pair.second.GetStats().misses.load();
        }
    }

    // 清空所有缓存
    void ClearAll()
    {
        std::unique_lock<std::shared_timed_mutex> lock(mapMutex_);
        commCaches_.clear();
    }

    // 从 algTag 中提取通信域名称
    std::string ExtractCommName(const std::string &algTag) const
    {
        size_t firstUnderscore = algTag.find('_');
        if (firstUnderscore == std::string::npos) {
            return "";
        }
        size_t secondUnderscore = algTag.find('_', firstUnderscore + 1);
        if (secondUnderscore == std::string::npos) {
            return "";
        }
        return algTag.substr(firstUnderscore + 1, secondUnderscore - firstUnderscore - 1);
    }

private:
    // 获取或创建通信域缓存
    CommDomainCache *GetOrCreateComm(const std::string &commName)
    {
        // 先尝试读锁快速寻找
        {
            std::shared_lock<std::shared_timed_mutex> lock(mapMutex_);
            auto it = commCaches_.find(commName);
            if (it != commCaches_.end()) {
                return &it->second;
            }
        }
        // 未找到，获取写锁创建
        {
            std::unique_lock<std::shared_timed_mutex> lock(mapMutex_);
            // 双重检查
            auto it = commCaches_.find(commName);
            if (it != commCaches_.end()) {
                return &it->second;
            }
            auto result = commCaches_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(commName),
                std::forward_as_tuple(commName));
            return &result.first->second;
        }
    }

    mutable std::shared_timed_mutex mapMutex_;
    std::unordered_map<std::string, CommDomainCache> commCaches_;
};

// 全局缓存管理器实例
thread_local CommDomainCacheManager g_cacheManager;

std::unique_ptr<AlgResourceCtxSerializable> DeserializeResCtx(const OpParam *param)
{
    HCCL_INFO("[DeserializeResCtx] enter, param=%p, resCtx=%p, ctxSize=%u", param, param->resCtx, param->ctxSize);
    if (param->resCtx == nullptr || param->ctxSize == 0) {
        HCCL_ERROR("[DeserializeResCtx] invalid resCtx=%p or ctxSize=%u", param->resCtx, param->ctxSize);
        return nullptr;
    }
    auto resCtx = std::make_unique<AlgResourceCtxSerializable>();
    char *ctx = static_cast<char *>(param->resCtx);
    HCCL_INFO("[DeserializeResCtx] start copy seq, ctx=%p, ctxSize=%u", ctx, param->ctxSize);
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    HCCL_INFO("[DeserializeResCtx] seq copy done, seq.size=%zu", seq.size());
    resCtx->DeSerialize(seq);
    HCCL_INFO("[DeserializeResCtx] DeSerialize done");
    return resCtx;
}
} // anonymous namespace

namespace ops_hccl {

// 选择走新（CollAlgExecRegistryV2）/老（CollAlgExecRegistry）算子流程
// A5芯片或者 template 名称前缀为 "opv2_" 走新流程，其他芯片走老流程
bool IsOpsV2(const char *algName, DevType deviceType)
{
    if (algName != nullptr) {
        const char *prefix = "opv2_";
        if (strncmp(algName, prefix, strlen(prefix)) == 0) {
            return true;
        }
    }
#ifdef MACRO_DEV_TYPE_NEW
    if (deviceType == DevType::DEV_TYPE_950) {
#else
    if (deviceType == DevType::DEV_TYPE_910_95) {
#endif
        return true;
    }
    return false;
}

HcclResult RestoreVarDataBatchSendRecv(OpParam &param)
{
    u64 sendRecvItemSize = static_cast<u64>(sizeof(HcclSendRecvItem));
    u64 itemNum = static_cast<u64>(param.batchSendRecvDataDes.itemNum);
    if (param.varMemSize != itemNum * sendRecvItemSize) {
        HCCL_ERROR("param.varMemSize[%lu] is not equal to itemNum[%lu] multiply [HcclSendRecvItem] size[%lu]."
                   "Failed to restore end recv info for BatchSendRecv!",
            param.varMemSize,
            itemNum,
            sendRecvItemSize);
        return HCCL_E_PARA;
    }
    param.batchSendRecvDataDes.sendRecvItemsPtr = reinterpret_cast<HcclSendRecvItem *>(param.varData);
    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataAlltoAllV(OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataAlltoAllV] param.varMemSize [%llu] is invalid,"
                   " ALL_TO_ALL_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize,
            ALL_TO_ALL_V_VECTOR_NUM,
            rankSize,
            sizeof(u64)),
        HCCL_E_PARA);

    constexpr u32 ALL_TO_ALL_V_OFFSET_SCOUNTS = 0;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RECV_COUNTS = 1;
    constexpr u32 ALL_TO_ALL_V_OFFSET_SDISPLS = 2;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RDISPLS = 3;

    u64 *data = reinterpret_cast<u64 *>(param.varData);
    param.all2AllVDataDes.sendCounts = data;
    param.all2AllVDataDes.recvCounts = data + ALL_TO_ALL_V_OFFSET_RECV_COUNTS * rankSize;
    param.all2AllVDataDes.sdispls = data + ALL_TO_ALL_V_OFFSET_SDISPLS * rankSize;
    param.all2AllVDataDes.rdispls = data + ALL_TO_ALL_V_OFFSET_RDISPLS * rankSize;

    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataReduceScatterV(OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(param.varMemSize != REDUCE_SCATTER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataReduceScatterV] param.varMemSize [%llu] is invalid,"
                   "REDUCE_SCATTER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize,
            REDUCE_SCATTER_V_VECTOR_NUM,
            rankSize,
            sizeof(u64)),
        HCCL_E_PARA);

    u64 *data = reinterpret_cast<u64 *>(param.varData);
    param.vDataDes.counts = data;
    param.vDataDes.displs = data + rankSize;
    return HCCL_SUCCESS;
}

HcclResult RestoreVarDataAllGatherV(OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(param.varMemSize != ALL_GATHER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR("[RestoreVarDataAllGatherV] param.varMemSize [%llu] is invalid,"
                   "ALL_GATHER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize,
            ALL_GATHER_V_VECTOR_NUM,
            rankSize,
            sizeof(u64)),
        HCCL_E_PARA);

    u64 *data = reinterpret_cast<u64 *>(param.varData);
    param.vDataDes.counts = data;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.counts[%u]:%u", i, reinterpret_cast<u64 *>(param.vDataDes.counts)[i]);
    }
    param.vDataDes.displs = data + rankSize;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.displs[%u]:%u", i, reinterpret_cast<u64 *>(param.vDataDes.displs)[i]);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

/**
 * AICPU kernel 下发入口（C 接口，与 src 侧保持一致）。
 * 在 device 侧执行，通过 aclrt kernel launch 下发。
 */
extern "C" unsigned int HcclLaunchAicpuKernel(OpParam *param)
{
    if (param == nullptr) {
        HCCL_ERROR("%s param is nullptr", __func__);
        return 1;
    }
    HCCL_INFO("%s Entry, commName[%s], tag[%s], algTag[%s], algName[%s], opType[%d]",
        __func__, param->commName, param->tag, param->algTag, param->algName, static_cast<int>(param->opType));

    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s HcommAcquireComm success, commName[%s]", __func__, param->commName);

    std::string algName = std::string(param->algName);
    bool isOpsV2 = ops_hccl::IsOpsV2(param->algName, param->deviceType);
    HCCL_INFO("%s IsOpsV2[%d], algName[%s], deviceType[%d]", __func__, isOpsV2, algName.c_str(),
        static_cast<int>(param->deviceType));

    if (!isOpsV2) {
        HCCL_INFO("%s Entering legacy ops path (CreateScatter)", __func__);
        ops_hccl::ScatterOpInfo opInfo;
        if (ops_hccl::CreateScatter(param, &opInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s CreateScatter fail", __func__);
            return 1;
        }
        HCCL_INFO("%s CreateScatter success", __func__);

        if (HcommIsSupportHcommRegOpInfo()) {
            HCCL_INFO("%s HcommRegOpInfo is supported", __func__);
            if (HcommRegOpInfo(param->commName, reinterpret_cast<void *>(&opInfo), sizeof(ops_hccl::ScatterOpInfo))
                != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%u]", __func__, param->commName,
                    opInfo.algTag, sizeof(ops_hccl::ScatterOpInfo));
                return 1;
            }
            HCCL_INFO("%s HcommRegOpInfo success, commName[%s], algTag[%s]", __func__, param->commName, opInfo.algTag);
        } else {
            HCCL_INFO("%s HcommRegOpInfo is NOT supported", __func__);
        }

        if (HcommIsSupportHcommRegOpTaskException()) {
            HCCL_INFO("%s HcommRegOpTaskException is supported", __func__);
            if (HcommRegOpTaskException(param->commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName,
                    param->algTag);
                return 1;
            }
            HCCL_INFO("%s HcommRegOpTaskException success, commName[%s], algTag[%s]", __func__, param->commName,
                param->algTag);
        } else {
            HCCL_INFO("%s HcommRegOpTaskException is NOT supported", __func__);
        }
    }

    if (isOpsV2) {
        HCCL_INFO("%s Entering OpsV2 path", __func__);

        // 判断通信域状态
        HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
        if (HcommIsSupportHcclCommGetStatus()) {
            HCCL_INFO("%s HcclCommGetStatus is supported", __func__);
            auto statusRet = HcclCommGetStatus(param->commName, &commStatus);
            if (statusRet != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcclCommGetStatus fail, commName[%s], ret = %d", __func__, param->commName, statusRet);
                return 1;
            }
            HCCL_INFO("%s HcclCommGetStatus success, commName[%s], commStatus[%d]", __func__, param->commName,
                static_cast<int>(commStatus));
            if (commStatus != HCCL_COMM_STATUS_READY) {
                HCCL_ERROR("%s commStatus is not ready!, commStatus = %d", __func__, static_cast<int>(commStatus));
                return 1;
            }
        } else {
            HCCL_INFO("%s HcclCommGetStatus is NOT supported", __func__);
        }

        // 通过缓存实现反序列化优化
        std::shared_ptr<const AlgResourceCtxSerializable> cachedResCtxHolder;
        std::unique_ptr<AlgResourceCtxSerializable> resCtx;
        const AlgResourceCtxSerializable *resCtxPtr{nullptr};

        if (param->opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            HCCL_INFO("%s Start cache lookup for algTag[%s], commName[%s]", __func__, param->algTag, param->commName);
            cachedResCtxHolder = g_cacheManager.Get(param->algTag, param->commName);
            if (cachedResCtxHolder != nullptr && ops_hccl::IsResCtxCacheReusable(*cachedResCtxHolder, *param)) {
                HCCL_INFO("%s Cache HIT for algTag[%s], cachedComm[%p], currentComm[%p]", __func__, param->algTag,
                    cachedResCtxHolder->commInfoPtr, param->hcclComm);
                std::string commName = g_cacheManager.ExtractCommName(param->algTag);
                if (commName.empty()) {
                    commName = param->commName;
                }
                CacheStats stats;
                size_t cacheSize;
                if (g_cacheManager.GetCommStats(commName, stats, cacheSize)) {
                    constexpr u32 hitRateNum = 100;
                    HCCL_INFO("%s comm[%s] hitRate=%.2f%%, cacheSize=%zu",
                        __func__, commName.c_str(), stats.hitRate() * hitRateNum, cacheSize);
                }
                resCtxPtr = cachedResCtxHolder.get();
            } else {
                bool isStaleCache = (cachedResCtxHolder != nullptr);
                HCCL_INFO("%s Cache MISS/STALE for algTag[%s], isStaleCache[%d]", __func__, param->algTag, isStaleCache);
                // 未命中或者通信域恢复后缓存失效，进行反序列化并存入缓存
                resCtx = DeserializeResCtx(param);
                g_cacheManager.Put(param->algTag, *resCtx, param->commName);
                resCtxPtr = resCtx.get();
                if (isStaleCache) {
                    HCCL_INFO("%s Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]",
                        __func__, param->algTag, cachedResCtxHolder->commInfoPtr, param->hcclComm);
                } else {
                    HCCL_INFO("%s Cache MISS and stored for algTag[%s]", __func__, param->algTag);
                }
            }
        } else {
            HCCL_INFO("%s BATCH_SEND_RECV, skip cache, start deserialize", __func__);
            resCtx = DeserializeResCtx(param);
            resCtxPtr = resCtx.get();
        }

        // 还原变长数据指针
        HCCL_INFO("%s Start RestoreVarData, opType[%d]", __func__, static_cast<int>(param->opType));
        HcclResult ret = HCCL_SUCCESS;
        if (param->opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            HCCL_INFO("%s RestoreVarDataBatchSendRecv", __func__);
            ret = ops_hccl::RestoreVarDataBatchSendRecv(*param);
        } else if (param->opType == HCCL_CMD_ALLTOALLV || param->opType == HCCL_CMD_ALLTOALLVC
                   || param->opType == HCCL_CMD_ALLTOALL) {
            HCCL_INFO("%s RestoreVarDataAlltoAllV", __func__);
            ret = ops_hccl::RestoreVarDataAlltoAllV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_REDUCE_SCATTER_V) {
            HCCL_INFO("%s RestoreVarDataReduceScatterV", __func__);
            ret = ops_hccl::RestoreVarDataReduceScatterV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_ALLGATHER_V) {
            HCCL_INFO("%s RestoreVarDataAllGatherV", __func__);
            ret = ops_hccl::RestoreVarDataAllGatherV(*param, *resCtxPtr);
        } else {
            HCCL_INFO("%s No RestoreVarData needed for opType[%d]", __func__, static_cast<int>(param->opType));
        }
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("%s RestoreVarData failed for opType[%d], ret[%d]", __func__, static_cast<int>(param->opType),
                static_cast<int>(ret));
            return 1;
        }
        HCCL_INFO("%s RestoreVarData success for opType[%d]", __func__, static_cast<int>(param->opType));

        // 获取 Device 侧主 thread，设置 batch mode
        ThreadHandle thread = resCtxPtr->threads[0];
        HCCL_INFO("%s Main thread handle[%llu]", __func__, thread);

        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommBatchModeStart failed, algTag[%s]", __func__, param->algTag);
            return 1;
        }
        HCCL_INFO("%s HcommBatchModeStart success, algTag[%s]", __func__, param->algTag);

        // 注册 DFX op 信息（需在第一个 task 之前上报）
        HCCL_INFO("%s Start DFX op info registration", __func__);
        HcclDfxOpInfoCompat dfxOpInfo{};
        if (ops_hccl::ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s ConvertToHcclDfxOpInfo fail, commName[%s], algTag[%s]", __func__, param->commName,
                param->algTag);
            return 1;
        }
        HCCL_INFO("%s ConvertToHcclDfxOpInfo success", __func__);

        if (HcclDfxRegOpInfoByCommId(param->commName, reinterpret_cast<void *>(&dfxOpInfo)) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcclDfxRegOpInfoByCommId fail, commName[%s], algTag[%s]", __func__, param->commName,
                param->algTag);
            return 1;
        }
        HCCL_INFO("%s HcclDfxRegOpInfoByCommId success", __func__);

        // 上报主流和第一个 task（wait 之前）
        HCCL_INFO("%s Start HcommProfilingReportKernelStartTask", __func__);
        if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportKernelStartTask failed, thread[%llu], commName[%s]", __func__, thread,
                param->commName);
            return 1;
        }
        HCCL_INFO("%s HcommProfilingReportKernelStartTask success", __func__);

        // 主 thread 等待 Host stream 的 notify 通知
        HCCL_INFO("%s Start NotifyWait setup", __func__);
        ThreadHandle exportedAicpuTsThread = param->opThread;
        u32 maxNotifyNum = resCtxPtr->notifyNumOnMainThread;
        for (u32 i = 0; i < resCtxPtr->notifyNumPerThread.size(); i++) {
            if (resCtxPtr->notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resCtxPtr->notifyNumPerThread[i];
            }
        }
        HCCL_INFO("%s NotifyWait setup: mainThread[%llu], exportedAicpuTsThread[%llu], maxNotifyNum[%u], "
                  "fullTimeout[%u], waitTimeout[%u]",
            __func__, thread, exportedAicpuTsThread, maxNotifyNum, resCtxPtr->fullTimeout, resCtxPtr->waitTimeout);

        if (HcommIsSupportHcommThreadResAcquireTimeOut()) {
            HCCL_INFO("%s HcclThreadResAcquireTimeOut is supported, timeout[%u]", __func__, resCtxPtr->fullTimeout);
            CHK_RET(HcclThreadResAcquireTimeOut(resCtxPtr->fullTimeout));
            HCCL_INFO("%s HcclThreadResAcquireTimeOut success", __func__);
        } else {
            HCCL_INFO("%s HcclThreadResAcquireTimeOut is NOT supported", __func__);
        }

        if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {
            HCCL_INFO("%s HcclSetNotifyWaitTimeOut is supported, timeout[%u]", __func__, resCtxPtr->waitTimeout);
            CHK_RET(HcclSetNotifyWaitTimeOut(resCtxPtr->waitTimeout));
            HCCL_INFO("%s HcclSetNotifyWaitTimeOut success", __func__);
        } else {
            HCCL_INFO("%s HcclSetNotifyWaitTimeOut is NOT supported", __func__);
        }

        HCCL_INFO("%s HcclThreadNotifyWaitOnThreadDefault, thread[%llu], maxNotifyNum[%u], timeout[%u]", __func__,
            thread, maxNotifyNum, resCtxPtr->waitTimeout);
        CHK_RET(static_cast<HcclResult>(
            HcclThreadNotifyWaitOnThreadDefault(thread, maxNotifyNum, resCtxPtr->waitTimeout)));
        HCCL_INFO("%s HcclThreadNotifyWaitOnThreadDefault success", __func__);

        // 从 resCtx->algoSerialData 反序列化 HcclAlgorithm，重建 executor
        HCCL_INFO("%s Start algorithm deserialization, algoSerialData size[%zu]", __func__,
            resCtxPtr->algoSerialData.size());
        HcclAlgorithm alg;
        std::vector<char> algoData = resCtxPtr->algoSerialData;
        BinaryStream algoBs(algoData);
        alg.DeserializeFrom(algoBs);
        HCCL_INFO("%s HcclAlgorithm::DeserializeFrom success", __func__);

        auto executor = alg.GetExecutor(*param);
        HCCL_INFO("%s alg.GetExecutor success, executor[%p]", __func__, executor.get());

        // 设置执行超时时间
        ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout);
        HCCL_INFO("%s SetExecTimeout[%u]", __func__, param->opConfig.execTimeout);

        // 设置 BatchTransfer 是否可行
        HCCL_INFO("%s InitHcommBatchTransferOnThreadSupported, isSupported[%d]", __func__,
            resCtxPtr->isHcommBatchTransferOnThreadSupported);
        CHK_RET(InitHcommBatchTransferOnThreadSupported(resCtxPtr->isHcommBatchTransferOnThreadSupported));
        HCCL_INFO("%s InitHcommBatchTransferOnThreadSupported success", __func__);

        // 执行算法编排
        HCCL_INFO("%s Start Orchestrate for alg[%s]", __func__, algName.c_str());
        if (executor->Orchestrate(const_cast<AlgResourceCtxSerializable &>(*resCtxPtr)) != HCCL_SUCCESS) {
            HCCL_ERROR("%s executor->Orchestrate failed for alg[%s]", __func__, algName.c_str());
            return 1;
        }
        HCCL_INFO("%s executor->Orchestrate success", __func__);

        // 上报主流和最后一个 task（notify 之后）
        HCCL_INFO("%s Start HcommProfilingReportKernelEndTask", __func__);
        if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportKernelEndTask failed, thread[%llu], commName[%s]", __func__, thread,
                param->commName);
            return 1;
        }
        HCCL_INFO("%s HcommProfilingReportKernelEndTask success", __func__);

        // 主 thread 通知 Host stream 完成
        constexpr u32 DEFAULT_NOTIFY_IDX = 0;
        HCCL_INFO("%s HcommThreadNotifyRecordOnThread, srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__,
            thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));
        HCCL_INFO("%s HcommThreadNotifyRecordOnThread success", __func__);

        // 上报 device op profiling
        HCCL_INFO("%s Start HcommProfilingReportDeviceOp", __func__);
        if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportDeviceOp failed, commName[%s]", __func__, param->commName);
            return 1;
        }
        HCCL_INFO("%s HcommProfilingReportDeviceOp success", __func__);

        // 结束 batch mode
        HCCL_INFO("%s Start HcommBatchModeEnd", __func__);
        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommBatchModeEnd failed, algTag[%s]", __func__, param->algTag);
            return 1;
        }
        HCCL_INFO("%s HcommBatchModeEnd success", __func__);
    } else {
        // 老流程（CollAlgExecRegistry）在重构中不再支持
        HCCL_ERROR("%s legacy executor path (CollAlgExecRegistry) is not supported in refactor, algName[%s]",
            __func__, algName.c_str());
        return 1;
    }

    // 释放通信域
    HCCL_INFO("%s Start HcommReleaseComm", __func__);
    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s HcommReleaseComm success, commName[%s]", __func__, param->commName);

    HCCL_INFO("%s Success, tag[%s], algTag[%s], commName[%s], algName[%s], opType[%d]",
        __func__, param->tag, param->algTag, param->commName, algName.c_str(), static_cast<int>(param->opType));
    return 0;
}
