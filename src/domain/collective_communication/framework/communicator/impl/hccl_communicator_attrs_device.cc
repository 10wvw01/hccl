/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator_attrs.h"
#include "device_capacity.h"
#include "config.h"
#include "externalinput_pub.h"
#include "env_config.h"

using namespace std;

namespace hccl
{
    HcclResult HcclCommunicatorAttrs::Init(HcclCommParams &params, const RankTable_t &rankTable)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
                                           WorldGroupInfo &groupCommonData)
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicatorAttrs::IsStandardCard()
    {
        return false;
    }

    bool HcclCommunicatorAttrs::Is310PDuoCard()
    {
        return false;
    }

    bool HcclCommunicatorAttrs::IsCommon310P3DUO(const std::vector<RankInfo_t> &rankList)
    {
        return false;
    }

    bool HcclCommunicatorAttrs::CompareWithUserRank(const RankInfo &left, const RankInfo &right)
    {
        return false;
    }

    HcclResult HcclCommunicatorAttrs::CheckDeviceType(const DevType deviceType) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::GetNicInfo(const NICDeployment &nicDeploy, const u32 curRankIndex,
                                                 const std::vector<RankInfo_t> &servRankList, RankInfo &rankInfo) const
    {
        return HCCL_SUCCESS;
    }

    // private
    HcclResult HcclCommunicatorAttrs::InitCommParams(HcclCommParams &params)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SetServerId(const RankTable_t &rankTable)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SetServerNum(const std::vector<RankInfo_t> &ranks)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice(const RankTable_t &rankTable)
    {
        return HCCL_SUCCESS;
    }

    // sub group适配获取server内设配数
    HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice(const std::vector<RankInfo> &rankList)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::TransformRankInfoByServerId(
        const std::vector<RankInfo_t> &rankList, ServRankInfo &servRankInfo) const
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicatorAttrs::CompareWithDevicePhyId(const RankInfo_t &left, const RankInfo_t &right)
    {
        return false;
    }

    HcclResult HcclCommunicatorAttrs::SetModuleInfo(const std::vector<RankInfo_t> &rankList)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SetSuperPodInfo(const std::vector<RankInfo_t> &rankList)
    {
        return HCCL_SUCCESS;
    }

    // 集群中存在910B A+X时，0-7卡: moduleIdx = 2 * serverIdx; 8-15卡: moduleIdx = 2 * serverIdx + 1
    // 集群中不存在910B A+X时，moduleIdx = serverIdx
    HcclResult HcclCommunicatorAttrs::GetModuleIdx(const RankInfo_t &rankInfo, u32 &moduleIdx)
    {
        return HCCL_SUCCESS;
    }

    // 用于标识集群中是否存在 910B A+X形态
    bool HcclCommunicatorAttrs::IsDiffDeviceModule(const std::vector<RankInfo_t> &rankList) const
    {
        return false;
    }

    HcclResult HcclCommunicatorAttrs::InitHccsPortNum()
    {
        return HCCL_SUCCESS;
    }

    // 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
    HcclResult HcclCommunicatorAttrs::CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SetRankInfoList(const RankTable_t &rankTable)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::CheckRankTable(const RankTable_t &rankTable, const ServRankInfo &servRankInfo)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::CheckDevPhyId(const s32 &devicePhyId) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SortRankInfoList()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::CheckNicDeploy(NICDeployment nicDeploy, DevType deviceType) const
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::CheckDevCount(const u32 devNum)
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicatorAttrs::Check2N(u32 num) const
    {
        return false;
    }

    HcclResult HcclCommunicatorAttrs::SetLocalRankInfo()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SetLocalRankInfoSubGroup(const std::vector<RankInfo> &rankList)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::CheckLocalRankInfo()
    {
        return HCCL_SUCCESS;
    }

    u32 HcclCommunicatorAttrs::CalMeshAggRankSize(int halfDevNum) const
    {
        return 0;
    }

    HcclResult HcclCommunicatorAttrs::SetMeshAggregationRankSize(u32 size)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::CalAndSetMeshAggRankSize()
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::SetWorldGroupInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &phyIdNicInfoMap,
        std::vector<RankInfo> &worldRankInfoList, std::vector<u32> &nicRanksPort, std::vector<u32> &vnicRanksPort)
    {
        return HCCL_SUCCESS;
    }

    HcclResult HcclCommunicatorAttrs::TransformRankList(
        const std::vector<RankInfo> &rankListIn, std::vector<RankInfo_t> &rankListOut)
    {
        return HCCL_SUCCESS;
    }

    bool HcclCommunicatorAttrs::IsEnableRoce()
    {
        return false;
    }

    // a+x mesh间需要同时保证ip有效和roce开关打开才能走rdma
    bool HcclCommunicatorAttrs::IsUsedRdmaLevel0AndIpInvalid()
    {
        return false;
    }

    bool HcclCommunicatorAttrs::IsSupportEnableRoce()
    {
        return false;
    }

    void HcclCommunicatorAttrs::GetTopoAttr(HcclTopoAttr &topoAttr)
    {
    }

    void HcclCommunicatorAttrs::GetAlgoAttr(HcclAlgoAttr &algoAttr)
    {
    }

    u32 HcclCommunicatorAttrs::GetLocalNicPort(NicType nicType)
    {
        return 0;
    }
}
