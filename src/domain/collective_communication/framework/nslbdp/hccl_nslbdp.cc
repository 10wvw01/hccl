/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <hccl/hccl_types.h>
#include "device_capacity.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "hccl_nslbdp.h"
#include "task_abort_handler_pub.h"
#include "coll_alg_utils.h"
#include "topoinfo_struct.h"
#include "coll_alg_utils.h"
#include "sal_pub.h"
#include "hcom_common.h"
#include "coll_alg_param.h"
#include "adapter_rts_common.h"
#include "adapter_hccp_common.h"
#include "hccl_nslb_md5.h"

namespace hccl {

const u32 COMM_SLICING_LENGTH = 1024; // group name max length

hcclNslbDp &hcclNslbDp::GetInstance()
{
    static hcclNslbDp pNslbDp;
    return pNslbDp;
}


hcclNslbDp::hcclNslbDp()
{
}

hcclNslbDp::~hcclNslbDp()
{
}


void hcclNslbDp::InitCmmDesc(std::string &identifier_nslb)
{
    nslbdp_identifier_ = identifier_nslb;
    HCCL_INFO("NSLBDP-SWK InitCmmDesc nslbdp_identifier_[%s] .", nslbdp_identifier_.c_str());
    return;
}

std::string hcclNslbDp::GetCmmDesc()
{
    HCCL_INFO("NSLBDP-SWK GetCmmDesc nslbdp_identifier_[%s] .", nslbdp_identifier_.c_str());
    return nslbdp_identifier_;
}

void hcclNslbDp::SetDeviceType()
{
    check910_93_ = true;
    return;
}

bool hcclNslbDp::GetDeviceType()
{
    bool is91093 = check910_93_;
    return is91093;
}

HcclResult hcclNslbDp::setHccpInfo(u32 buffer_size, void* tlv_handle)
{
    HCCL_INFO("HCCL nslbdp try setHccpInfo.");
    nslbdp_buffsize_ = buffer_size;
    nslbdp_handle_ = tlv_handle;
    nslbdpIsInitHccp_ = true;
    HCCL_INFO("HCCL setHccpInfo buffer_size:[%u].", buffer_size);
    return HCCL_SUCCESS;
}

bool hcclNslbDp::getHccpInitFlag()
{
    bool nslbdp_IsInitHccp = nslbdpIsInitHccp_;
    return nslbdp_IsInitHccp;
}

HcclResult hcclNslbDp::clearnHccpInitFlag()
{
    nslbdpIsInitHccp_ = false;
    return HCCL_SUCCESS;
}

u32 hcclNslbDp::getBufferSize()
{
    u32 nslbdpBuffsize = nslbdp_buffsize_;
    return nslbdpBuffsize;
}

HcclResult hcclNslbDp::HcclSetGlobalRankTotalNum(u32 nRanks)
{
    //填充表4
    hcclNslbDpGlobalRankVal_.rankTotalNum = nRanks;
    return HCCL_SUCCESS;
}

void hcclNslbDp::SetGlobalCommTaskId(u64 taskId)
{
    hcclNslbDpGlobalCommInfo_.taskId = taskId;
    return;
}

void hcclNslbDp::SetGlobalCommNodeId(u32 nodeId)
{
    hcclNslbDpGlobalCommInfo_.nodeId= nodeId;
    return;
}

void hcclNslbDp::SetGlobalCommLocalRankNum(u32 localRankNum)
{
    hcclNslbDpGlobalCommInfo_.localRankNum= localRankNum;
    return;
}

void hcclNslbDp::SetGlobalCommRankTotalNum(u32 rankTotalNum)
{
    hcclNslbDpGlobalCommInfo_.rankTotalNum= rankTotalNum;
    return;
}

u64 hcclNslbDp::GetGlobalCommTaskId()
{
    return hcclNslbDpGlobalCommInfo_.taskId;
}

u32 hcclNslbDp::GetGlobalCommNodeId()
{
    return hcclNslbDpGlobalCommInfo_.nodeId;
}

u8 hcclNslbDp::GetGlobalCommLocalRankNum()
{
    return hcclNslbDpGlobalCommInfo_.localRankNum;
}

u32 hcclNslbDp::GetGlobalCommRankTotalNum()
{
    return hcclNslbDpGlobalCommInfo_.rankTotalNum;
}

//获取 l4SPortId
u32 hcclNslbDp::Getl4SPortId()
{
    u32 NslbDpL4SPortId = hcclNslbDpL4SPortId_;
    return NslbDpL4SPortId;
}

//切分identifier字段
void hcclNslbDp::SplitString(const std::string& identifier, std::vector<std::string>& splitInfo, const std::string& frag)
{
    std::string::size_type pos2 = identifier.find(frag);
    std::string::size_type pos1 = 0;
    while(std::string::npos != pos2)
    {
        splitInfo.push_back(identifier.substr(pos1, pos2-pos1));
        pos1 = pos2 + frag.size();
        pos2 = identifier.find(frag, pos1);
    }
    if(pos1 != identifier.length()) {
        splitInfo.push_back(identifier.substr(pos1));
    }
    return;
}

// 将IP转换成U32值
u32 hcclNslbDp::ipToUint32(const std::string& ipAddress)
{
    struct sockaddr_in sa;
    inet_pton(AF_INET, ipAddress.c_str(), &(sa.sin_addr));
    return ntohl(sa.sin_addr.s_addr); // Convert to host byte order
}

/*执行send 流程*/
HcclResult hcclNslbDp::SendCommRankTable(uint32_t rank, NslbDpCommConfigVal globalCommInfo)
{
    HCCL_INFO("hcclNslbDp entry SendTableFir");
    u32 rankTotalNum = globalCommInfo.rankTotalNum;

    HCCL_INFO("HCCL SendTableFir rankTotalNum:[%u].", rankTotalNum);
    if (rankTotalNum > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
        return HCCL_SUCCESS;
    }
    u32 packetNum = NSLBDP_PKTNUM_FIR;
    if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_FIR) {
        packetNum = NSLBDP_PKTNUM_FIR;
    } else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_SEC) {
        packetNum = NSLBDP_PKTNUM_SEC;
    } else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_THR) {
        packetNum = NSLBDP_PKTNUM_THR;
    } else {
        packetNum = NSLBDP_PKTNUM_FOU;
    }

    if (packetNum - 1 != rank) {
        HCCL_INFO("hcclNslbDp only rank [%u] send TableFir.");
        return HCCL_SUCCESS;
    }
    SendTableProc(packetNum, globalCommInfo);

    HCCL_INFO("hcclNslbDp entry SendTableFir end");
    return HCCL_SUCCESS;
}


// 读配置文件的场景与正常创建通信域场景下填充通信域信息表（表一）
void hcclNslbDp::SetGlobalCommRankTable_RootInfo(const RankTable_t &rankTable, const HcclBasicRankInfo &localRankInfo,
                                                 const std::string& identifier, u32 nRanks, u32 rank)
{
    HCCL_INFO("NSLBDP-WEN Entry SetGlobalCommRankTable_RootInfo commDesc[%s] - size = [%u].", identifier.c_str(), rankTable.rankList.size());
    u64 checkTaskId = GetGlobalCommTaskId();
    if (checkTaskId == 0 || nRanks == 1) {
        return;
    }

    //判断是否跨机
    if (CheckMultiMachine(rankTable) == false) {
        HCCL_INFO("HCCL nslb-dp CheckMultiMachine is false");
        return;
    }

    NslbDpCommConfigVal globalCommInfo{};

    u32 sRet = memset_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(globalCommInfo.commDesc));
    if (sRet != EOK) {
        HCCL_INFO("memset_s commDesc fail");
        return;
    }
    // 获取通信域唯一标识
    s32 ret = strncpy_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH,  identifier.c_str(), identifier.size());
    if (ret != EOK) {
        HCCL_INFO("strncpy_s globalCommInfo.commDesc fail");
    }
    u64 utime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    globalCommInfo.commInitTime = utime;

    std::vector<std::string> splitInfo;
    SplitString(identifier, splitInfo, "_");
    if (splitInfo.size() > NSLBDP_SPLIT_SIZE) {
        std::string times = splitInfo[NSLBDP_SPLIT_SIZE];
        globalCommInfo.commInitTime = std::stoull(times);
        globalCommInfo.commInitTime = globalCommInfo.commInitTime / NSLBDP_INITIME_MILLISENDS;
    }

    globalCommInfo.taskId = checkTaskId;
    globalCommInfo.rankTotalNum = nRanks;

    u32 sendRankIp[NSLBDP_PKTNUM_FOU] = {0, 0, 0, 0};
    HCCL_RUN_INFO("HCCL SetGlobalCommRankTable_RootInfo rankList.size:[%zu] success.", rankTable.rankList.size());

    for(u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
        NslbDpRankInfo dpRankInfo;
        HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
        std::string deviceIp = tmpIp.GetReadableAddress();
        dpRankInfo.deviceIp = ipToUint32(deviceIp);
        u32 sendNum = nRanks / NSLBDP_RANKTOTALNUM_BLOCK_FIR;
        if(rankIndex == (sendNum * NSLBDP_RANKTOTALNUM_BLOCK_FIR)) {
            sendRankIp[sendNum] = ipToUint32(deviceIp);
        }
        HCCL_INFO("HCCL SetGlobalCommRankTable_RootInfo deviceIp:[%u] success.",
                  dpRankInfo.deviceIp);

        std::string serverIp = rankTable.rankList[rankIndex].serverId;
        dpRankInfo.serverIp = ipToUint32(serverIp);
        HCCL_INFO("HCCL SetGlobalCommRankTable_RootInfo serverIp:[%u] success.", dpRankInfo.serverIp);
        if (rankTable.rankList[rankIndex].superPodIdx == INVALID_UINT) {
            dpRankInfo.podId = 0;
        } else {
            dpRankInfo.podId = rankTable.rankList[rankIndex].superPodIdx;
        }
        dpRankInfo.rev = 0;
        globalCommInfo.rankInfo.push_back(dpRankInfo);
    }
    NSLBMD5::calculateRankInfoMd5(globalCommInfo.rankInfo, globalCommInfo.commMd5Sum);

    std::string nslbdpmd5 = NSLBMD5::md5ToString(globalCommInfo.commMd5Sum);
    HCCL_INFO("HCCL NSLBDP-MD5 nslbdpmd5:[%s] success.", nslbdpmd5.c_str());

    std::string npuIp = localRankInfo.deviceIP[0].GetReadableIP();
    for(u32 sendInx = 0; sendInx < NSLBDP_PKTNUM_FOU; sendInx++) {
        if (ipToUint32(npuIp) != 0 && ipToUint32(npuIp) == sendRankIp[sendInx]) {
            SendCommRankTable(rank, globalCommInfo);
            break;
        }
    }
    hcclNslbDpCommConfig_.push_back(globalCommInfo);
    HCCL_INFO("HCCL nslbdp Entry SetGlobalCommRankTable_RootInfo end size = [%zu]", hcclNslbDpCommConfig_.size());
}

//填充表5
void hcclNslbDp::SetGlobalDisRankTable(const HcclBasicRankInfo &rankTable)
{
    HCCL_INFO("HCCL Entry SetGlobalDisRankTable");
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return;
    }
    std::string npuIp = rankTable.deviceIP[0].GetReadableIP();
    if (ipToUint32(npuIp) == 0) {
        return;
    }
    hcclNslbDpGlobalDisRankVal_.taskId = taskId;
    hcclNslbDpGlobalDisRankVal_.nodeId = GetGlobalCommNodeId();
    hcclNslbDpGlobalDisRankVal_.localRankNum = GetGlobalCommLocalRankNum();
    hcclNslbDpGlobalDisRankVal_.rankTotalNum = GetGlobalCommRankTotalNum();

    std::string serverIp = rankTable.hostIP.GetReadableIP();
    hcclNslbDpGlobalDisRankVal_.serverIp = ipToUint32(serverIp);
    HCCL_INFO("HCCL SetGlobalDisRankTable serverIp:[%u].", hcclNslbDpGlobalDisRankVal_.serverIp);

    hcclNslbDpGlobalDisRankVal_.npuIp = ipToUint32(npuIp);
    HCCL_INFO("HCCL SetGlobalDisRankTable npuIp:[%u].", hcclNslbDpGlobalDisRankVal_.npuIp);
}

//check 是否是多机场景
bool hcclNslbDp::CheckMultiMachine(const RankTable_t rankTable)
{
    u32 serverIpFir = 0;
    u16 podIdFir = 0;
    bool bIsMultiMachine = false;
    if (GetDeviceType() == true) {
        u16 podIdIndex = 0;
        for(u32 index = 0; index < rankTable.rankList.size(); index++) {
            if (rankTable.rankList[index].superPodIdx == INVALID_UINT) {
                podIdIndex = 0;
            } else {
                podIdIndex = rankTable.rankList[index].superPodIdx;
            }
            if (index == 0) {
                podIdFir = podIdIndex;
                continue;
            } 

            if (podIdFir != podIdIndex) {
                return true;
            }
        }
        return false;
    } 
    for(u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
        std::string serverIpInfo = rankTable.rankList[rankIndex].serverId;
        u32 serverIp = ipToUint32(serverIpInfo);
        u16 podIdInfo = 0;
        if (rankTable.rankList[rankIndex].superPodIdx == INVALID_UINT) {
            podIdInfo = 0;
        } else {
            podIdInfo = rankTable.rankList[rankIndex].superPodIdx;
        }
        if (rankIndex == 0) {
            serverIpFir = serverIp;
            podIdFir = podIdInfo;
            continue;
        } 

        if (serverIp != serverIpFir || podIdFir != podIdInfo) {
            bIsMultiMachine = true;
            break;
        }
    }
    return bIsMultiMachine;
}

//无ranktable场景， 子通信域场景表1 赋值
HcclResult hcclNslbDp::SetCommInfo_NoRankTable(const hccl::RankTable_t rankTable, std::string identifier)
{
    HCCL_INFO("HCCL Entry SetCommInfo_NoRankTable");
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    //判断是否跨机
    if (CheckMultiMachine(rankTable) == false) {
        HCCL_INFO("HCCL nslb-dp CheckMultiMachine is false");
        return HCCL_SUCCESS;
    }

    NslbDpCommConfigVal globalCommInfo;

    s32 sRet = memset_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(globalCommInfo.commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memset_s commDesc fail");
        return HCCL_SUCCESS;
    }

    // 获取通信域唯一标识
    s32 ret = strncpy_s(globalCommInfo.commDesc, COMM_DESC_MAX_LENGTH,  identifier.c_str(), identifier.size());
    if (ret != EOK) {
        HCCL_INFO("strncpy_s globalCommInfo.commDesc fail");
    }
    globalCommInfo.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

    std::string descstr = globalCommInfo.commDesc;

    CHK_PRT_RET(ret != EOK, HCCL_ERROR("GetIdentifier str copy fail. return[%d]", ret), HCCL_E_INTERNAL);

    std::vector<std::string> splitInfo;
    SplitString(identifier, splitInfo, "_");
    
    if (splitInfo.size() > NSLBDP_SPLIT_SIZE) {
        std::string trst = splitInfo[NSLBDP_SPLIT_SIZE];
        globalCommInfo.commInitTime = std::stoull(trst);
        globalCommInfo.commInitTime = globalCommInfo.commInitTime / NSLBDP_INITIME_MILLISENDS;
    } else {
        // commInitTime在有ranktable的赋值
        u64 utime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        globalCommInfo.commInitTime = utime;
    }
    u16 nRanks = rankTable.rankNum;

    globalCommInfo.taskId = taskId;
    globalCommInfo.rankTotalNum = nRanks;

    for(u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
        NslbDpRankInfo dpRankInfo;
        HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
        std::string deviceIp = tmpIp.GetReadableAddress();
        dpRankInfo.deviceIp = ipToUint32(deviceIp);
        HCCL_INFO("HCCL SetCommInfo_RankTableExit deviceIp:[%u] success.", dpRankInfo.deviceIp);
        if (rankTable.rankList[rankIndex].superPodIdx == INVALID_UINT) {
            dpRankInfo.podId = 0;
        } else {
            dpRankInfo.podId = rankTable.rankList[rankIndex].superPodIdx;
        }
        dpRankInfo.rev = 0;
        globalCommInfo.rankInfo.push_back(dpRankInfo);
    }
    hcclNslbDpCommConfig_.push_back(globalCommInfo);

    return HCCL_SUCCESS;
}

//有ranktable场景， 表1 赋值
HcclResult hcclNslbDp::SetCommInfo_RankTableExit(RankTable_t rankTable)
{
    HCCL_INFO("hcclNslbDp entry SetCommInfo_RankTableExit");
    u16 nRanks = rankTable.rankNum;
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0 || nRanks == 1) {
        return HCCL_SUCCESS;;
    }

    NslbDpCommConfigVal globalCommInfo = {};
    // 获取通信域唯一标识
    s32 sRet = memset_s(hcclNslbDpGlobalRankVal_.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(hcclNslbDpGlobalRankVal_.commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memset_s set null fail");
        return HCCL_SUCCESS;;
    }
    char commDesc[COMM_DESC_MAX_LENGTH] = "HCCL_WORLD_GROUP";
    sRet = memcpy_s(globalCommInfo.commDesc, sizeof(globalCommInfo.commDesc), commDesc, COMM_DESC_MAX_LENGTH);
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commDesc fail");
        return HCCL_SUCCESS;;
    }

    globalCommInfo.taskId = taskId;
    globalCommInfo.rankTotalNum = nRanks;

    u64 utime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    globalCommInfo.commInitTime = utime;

    u32 size = rankTable.rankList.size();
    HCCL_INFO("HCCL SetCommInfo_RankTableExit size:[%u] success.", size);

    for(size_t ranksize = 0; ranksize < rankTable.rankList.size(); ranksize++) {
        for(size_t rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
            u16 podId = 0;
            if (rankTable.rankList[rankIndex].superPodIdx != INVALID_UINT) {
                podId = rankTable.rankList[rankIndex].superPodIdx; 
            }
            //按照rankid 排序
            if (ranksize == rankTable.rankList[rankIndex].rankId) {
                NslbDpRankInfo dpRankInfo;
                 HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
                std::string deviceIp = tmpIp.GetReadableAddress();
                dpRankInfo.deviceIp = ipToUint32(deviceIp);
                HCCL_INFO("HCCL SetCommInfo_RankTableExit deviceIp:[%s] success.", deviceIp.c_str());
                std::string serverIp = rankTable.rankList[rankIndex].serverId;
                dpRankInfo.serverIp = ipToUint32(serverIp);
                HCCL_INFO("HCCL SetCommInfo_RankTableExit serverIp:[%s] success.", serverIp.c_str());
                dpRankInfo.podId = podId;
                dpRankInfo.rev = 0;
                globalCommInfo.rankInfo.push_back(dpRankInfo);
            }
        }
    }
    NSLBMD5::calculateRankInfoMd5(globalCommInfo.rankInfo, globalCommInfo.commMd5Sum);
    std::string nslbdpmd5 = NSLBMD5::md5ToString(globalCommInfo.commMd5Sum);
    HCCL_INFO("HCCL NSLBDP-MD5 nslbdpmd5:[%s] success.", nslbdpmd5.c_str());

    hcclNslbDpCommConfig_.push_back(globalCommInfo);
    HCCL_INFO("hcclNslbDp entry SetCommInfo_RankTableExit end");

    return HCCL_SUCCESS;
}

//有ranktable场景， 表4 赋值
HcclResult hcclNslbDp::SetGlobalRank_RankTableExit(const hccl::RankTable_t  rankTable)
{
    u64 taskId = GetGlobalCommTaskId();
    HCCL_INFO("HCCL SetGlobalRank_RankTableExit[%ull] success.", taskId);

    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    u16 nRanks = rankTable.rankNum;
    hcclNslbDpGlobalRankVal_.taskId = taskId;
    s32 sRet = memset_s(hcclNslbDpGlobalRankVal_.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(hcclNslbDpGlobalRankVal_.commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memset_s commDesc fail");
        return HCCL_SUCCESS;
    }
    char commDesc[COMM_DESC_MAX_LENGTH] = "HCCL_WORLD_GROUP";
    sRet = memcpy_s(hcclNslbDpGlobalRankVal_.commDesc, sizeof(hcclNslbDpGlobalRankVal_.commDesc), commDesc, COMM_DESC_MAX_LENGTH);
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commDesc fail");
        return HCCL_SUCCESS;
    }

    hcclNslbDpGlobalRankVal_.commInitTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    hcclNslbDpGlobalRankVal_.rankTotalNum = nRanks;

    for (size_t ranksize = 0; ranksize < rankTable.rankList.size(); ranksize++) {
        for (size_t rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
            if(ranksize == rankTable.rankList[rankIndex].rankId) {
                TableFourRankInfo dpGloRankInfo;
                HcclIpAddress tmpIp = rankTable.rankList[rankIndex].deviceInfo.deviceIp[0];
                std::string deviceIp = tmpIp.GetReadableAddress();
                dpGloRankInfo.deviceIp = ipToUint32(deviceIp);

                std::string serverIp = rankTable.rankList[rankIndex].serverId;
                dpGloRankInfo.serverIp = ipToUint32(serverIp);

                hcclNslbDpGlobalRankVal_.rankInfo.push_back(dpGloRankInfo);
            }
        }
    }
    NSLBMD5::calculateTableFourRankInfoMd5(hcclNslbDpGlobalRankVal_.rankInfo, hcclNslbDpGlobalRankVal_.commMd5Sum);
    std::string nslbdpmd5 = NSLBMD5::md5ToString(hcclNslbDpGlobalRankVal_.commMd5Sum);
    HCCL_INFO("HCCL NSLBDP-MD5 nslbdpmd5:[%s] success.", nslbdpmd5.c_str());
    return HCCL_SUCCESS;
}

//拼接 l4SPortId  ---待对齐
HcclResult hcclNslbDp::GetNslbDpl4SPortId(u32 rankSize, u8 algType, u16 *l4SPortId)
{
    u16 priFlag = NSLBDP_PRIVATE_PORT;
    u16 CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_BEGIN;
    if (rankSize > NSLBDP_COMMINTERVAL_FLAGSIX) {
        CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_SEV;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGFIV) {
         CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_SIX;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGFOU) {
         CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_FIV;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGTHR) {
         CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_FOR;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAGSEC) {
         CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_THR;
    } else if (rankSize > NSLBDP_COMMINTERVAL_FLAG) {
         CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_SEC;
    } else {
         CommIntervalFlag = NSLB_COMM_INTERVAL_FLAG_FIR;
    }
    u16 CommPrecisely = rankSize % NSLBDP_COMMINTERVAL_FLAG;
    u8 CommalgType = algType;

    *l4SPortId = (priFlag << NSLBDP_RANGE_ID) + (CommIntervalFlag << NSLBDP_COMMON_RANGE) + (CommPrecisely << NSLBDP_ALGO_RANGE) + CommalgType;
    hcclNslbDpL4SPortId_ = *l4SPortId;

    HCCL_INFO("NslbDp-GetNslbDpl4SPortId  hcclNslbDpL4SPortId_[%u]", hcclNslbDpL4SPortId_);
    return HCCL_SUCCESS;
}

// 表6赋值
HcclResult hcclNslbDp::SetNslbDpRootRank(HcclCMDType opType, u32 rootRank, std::string identifier, u8 algType)
{
    HCCL_INFO("hcclNslbDp entry SetNslbDpRootRank");
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    if (hcclNslbDpRootRankVal_.taskId == 0) {
        HCCL_INFO("nslbdpRootRank first entry");
        // 获取task id 
        hcclNslbDpRootRankVal_.taskId = taskId;
        s32 sRet = memset_s(hcclNslbDpRootRankVal_.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(hcclNslbDpRootRankVal_.commDesc));
        if (sRet != EOK) {
            HCCL_ERROR("memset_s commDesc fail");
            return HCCL_SUCCESS;
        }
        // 获取通信域唯一标识
        s32 ret = strncpy_s(hcclNslbDpRootRankVal_.commDesc, COMM_DESC_MAX_LENGTH, identifier.c_str(), identifier.size());
        if (ret != EOK) {
            HCCL_INFO("strncpy_s hcclNslbDpRootRankVal_.commDesc fail");
        }
        hcclNslbDpRootRankVal_.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

        std::vector<std::string> splitInfo;
        SplitString(identifier, splitInfo, "_");
        if (splitInfo.size() > NSLBDP_SPLIT_SIZE) {
            std::string trst = splitInfo[NSLBDP_SPLIT_SIZE];
            hcclNslbDpRootRankVal_.commInitTime = std::stoull(trst);
            hcclNslbDpRootRankVal_.commInitTime = hcclNslbDpRootRankVal_.commInitTime / NSLBDP_INITIME_MILLISENDS;
        } else {
            // commInitTime在有ranktable的赋值
            hcclNslbDpRootRankVal_.commInitTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        }

        // 获取operator、algorithm
        hcclNslbDpRootRankVal_.oper = GetNslbOpType(opType);;
    
        hcclNslbDpRootRankVal_.algorithm = algType;
    
        hcclNslbDpRootRankVal_.rootRankNum = hcclNslbDpRootRankVal_.rootRankNum + 1;

        NslbDpRankId rootRankId;
        rootRankId.rankID = rootRank;
        hcclNslbDpRootRankVal_.rankId.push_back(rootRankId);

        HCCL_INFO("HCCL NslbDp-Rootable rootRankNum[%u]-rootRank[%u]-algorithm[%u].", 
                   hcclNslbDpRootRankVal_.rootRankNum, rootRank, hcclNslbDpRootRankVal_.algorithm);

        HCCL_INFO("NslbDp-Rootable entry SetNslbDpRootRank end");
        SendRootRankTable();
        return HCCL_SUCCESS;
    }
    HCCL_INFO("HCCL NslbDp-Rootable rankId.size[%zu]-rootRankNum[%u]-rootRank[%u].",
               hcclNslbDpRootRankVal_.rankId.size(), hcclNslbDpRootRankVal_.rootRankNum, rootRank);

    for (const auto& rank : hcclNslbDpRootRankVal_.rankId) {
        HCCL_INFO("NslbDp-Rootable rootRank exit rankID[%u]-rootRank[%u]",rank.rankID, rootRank);
        if (rank.rankID == rootRank) {
            return HCCL_SUCCESS;
        }
    }

    NslbDpRankId rootRankId;
    rootRankId.rankID =  static_cast<u16>(rootRank);
    hcclNslbDpRootRankVal_.rankId.push_back(rootRankId);
    hcclNslbDpRootRankVal_.rootRankNum = hcclNslbDpRootRankVal_.rootRankNum + 1;

    // 新增场景下走send 流程
    SendRootRankTable();

    return HCCL_SUCCESS;
}

// AlgType 的转换
u8 hcclNslbDp::GetNslbLevel1AlgType(AlgTypeLevel1 algValue)
{
    switch (algValue) {
        case AlgTypeLevel1::ALG_LEVEL1_RING: {
            return NSLB_ALGO_TYPE_RING;
        }
        case AlgTypeLevel1::ALG_LEVEL1_PIPELINE: {
            return NSLB_ALGO_TYPE_PIPELINE;
        }
        case AlgTypeLevel1::ALG_LEVEL1_HD: {
            return NSLB_ALGO_TYPE_HDR;
        }
        case AlgTypeLevel1::ALG_LEVEL1_NHR: {
            return NSLB_ALGO_TYPE_NHR;
        }
        case AlgTypeLevel1::ALG_LEVEL1_NHR_V1: {
            return NSLB_ALGO_TYPE_NHR_V1;
        }
        case AlgTypeLevel1::ALG_LEVEL1_NB: {
            return NSLB_ALGO_TYPE_NB;
        }
        default:
            return NSLB_ALGO_TYPE_NA;
    }
}

// Level2 的 AlgType 转换
u8 hcclNslbDp::GetNslbLevel2AlgType(AlgTypeLevel2 algValue)
{
    switch (algValue) {
        case AlgTypeLevel2::ALG_LEVEL2_RING: {
            return NSLB_ALGO_TYPE_RING;
        }
        case AlgTypeLevel2::ALG_LEVEL2_HD: {
            return NSLB_ALGO_TYPE_HDR;
        }
        case AlgTypeLevel2::ALG_LEVEL2_NHR: {
            return NSLB_ALGO_TYPE_NHR;
        }
        case AlgTypeLevel2::ALG_LEVEL2_NB: {
            return NSLB_ALGO_TYPE_NB;
        }
        default:
            return NSLB_ALGO_TYPE_NA;
    }
}

// opType 的转换
u8 hcclNslbDp::GetNslbOpType(HcclCMDType opType)
{
    switch (opType) {
        case HcclCMDType::HCCL_CMD_BROADCAST: {
            return NSLBDP_CMD_BROADCAST;
        }
        case HcclCMDType::HCCL_CMD_ALLREDUCE: {
            return NSLBDP_CMD_ALLREDUCE;
        }
        case HcclCMDType::HCCL_CMD_REDUCE: {
            return NSLBDP_CMD_REDUCE;
        }
        case HcclCMDType::HCCL_CMD_SEND: {
            return NSLBDP_CMD_SEND;
        }
        case HcclCMDType::HCCL_CMD_RECEIVE: {
            return NSLBDP_CMD_RECEIVE;
        }
        case HcclCMDType::HCCL_CMD_ALLGATHER: {
            return NSLBDP_CMD_ALLGATHER;
        }
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER: {
            return NSLBDP_CMD_REDUCE_SCATTER;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALLV: {
            return NSLBDP_CMD_ALLTOALLV;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALLVC: {
            return NSLBDP_CMD_ALLTOALLVC;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALL: {
            return NSLBDP_CMD_ALLTOALL;
        }
        case HcclCMDType::HCCL_CMD_GATHER: {
            return NSLBDP_CMD_GATHER;
        }
        case HcclCMDType::HCCL_CMD_SCATTER: {
            return NSLBDP_CMD_SCATTER;
        }
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV: {
            return NSLBDP_CMD_BATCH_SEND_RECV;
        }
        default:
            return 0;
    }
}

//获取通信量的前4bit
u64 hcclNslbDp::GetNslbDpFirstFourBit(u8 opType, u8 algType)
{
    u64 firstFourBit = 0;
    if (opType == NSLBDP_CMD_ALLREDUCE) {
        firstFourBit = 1 << NSLBDP_BEGINFOURBIT;
        firstFourBit = firstFourBit + 1;
    }

    if (opType == NSLBDP_CMD_ALLGATHER) {
        firstFourBit = 1 << 1;
    }

    HCCL_INFO("NslbDp-firstFourBit  firstFourBit[%llu]-algType[%u]", firstFourBit, algType);
    return firstFourBit;
}

//校验算法的一致性
bool hcclNslbDp::CheckAlgoConsistency(HcclCMDType opType, std::string& algName)
{
    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        if (algName.find("AllReduce") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        if (algName.find("AllGather") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        if (algName.find("ReduceScatter") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        if (algName != "RunAlltoAllVFullMesh") {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        if (algName.find("BroadCast") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_SCATTER) {
        if (algName.find("Scatter") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    }
    return true;
}


// 写算法算子表--表3
HcclResult hcclNslbDp::GetAlgAdjacencyTable(HcclCMDType opType, u32 srcLocalRankId, u32 rootRank, 
            u8 algType, std::string identifier, AdjInfo nslbAdjInfo)
{
    HCCL_INFO("hcclNslbDp entry GetAlgAdjacencyTable");
    HCCL_INFO("HCCL hcclNslbDpAlgorithmInfo_ size:[%zu].", hcclNslbDpAlgorithmInfo_.size());

    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }
    HCCL_RUN_INFO("NSLBDP-TABLE3 opType:[%u],srcLocalRankId[%u],rootRank[%u]-commDesc[%s],dstRankNum:[%u].",
               opType, srcLocalRankId, rootRank, identifier.c_str(), nslbAdjInfo.dstRankNum);

    if (GetNslbOpType(opType) != NSLBDP_CMD_ALLREDUCE && GetNslbOpType(opType) != NSLBDP_CMD_ALLGATHER
        && GetNslbOpType(opType) != NSLBDP_CMD_REDUCE_SCATTER && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALL
        && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALLV && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALLVC
        && GetNslbOpType(opType) != NSLBDP_CMD_BROADCAST  && GetNslbOpType(opType) != NSLBDP_CMD_SCATTER
        && GetNslbOpType(opType) != NSLBDP_CMD_SEND  && GetNslbOpType(opType) != NSLBDP_CMD_BATCH_SEND_RECV 
        && GetNslbOpType(opType) != NSLBDP_CMD_REDUCE) {
        return HCCL_SUCCESS;
    }

    NslbDpAlgorithmInfo algorithmInfo;
    algorithmInfo.taskId = taskId;
    s32 sRet = memset_s(algorithmInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(algorithmInfo.commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commDesc fail");
        return HCCL_SUCCESS;
    }
    s32 ret = strncpy_s(algorithmInfo.commDesc, COMM_DESC_MAX_LENGTH, identifier.c_str(), identifier.size());
    if (ret != EOK) {
        HCCL_INFO("strncpy_s algorithmInfo.commDesc fail");
    }
    algorithmInfo.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

    HCCL_INFO("NSLBDP-SWK CheckCommDescExit algorithmInfo.commDesc[%s] .", algorithmInfo.commDesc);

    // 去除不存在的通信域信息
    bool commDescExit = false;
    for (const auto& info : hcclNslbDpCommConfig_) {
        if (strcmp(algorithmInfo.commDesc, info.commDesc) == 0) {
            commDescExit = true;
            break;
        }
    }
    if (commDescExit == false) {
        return HCCL_SUCCESS;
    }

    // 根据表一信息填充MD5
    for (size_t comsize = 0; comsize < hcclNslbDpCommConfig_.size(); comsize++) {
        if(strcmp(algorithmInfo.commDesc, hcclNslbDpCommConfig_[comsize].commDesc) == 0) {
            sRet = memcpy_s(algorithmInfo.commMd5Sum, sizeof(algorithmInfo.commMd5Sum),
                        hcclNslbDpCommConfig_[comsize].commMd5Sum, sizeof(hcclNslbDpCommConfig_[comsize].commMd5Sum));
            if (sRet != EOK) {
                HCCL_ERROR("memcpy_s commMd5Sum fail");
                return HCCL_SUCCESS;
            }
            break;
        }
    }
    HCCL_RUN_INFO("NSLBDP-SWK add adjINfo:***[%llu]***[%u]***[%u]***[%u]***[%u]-[%zu] success.",
                   taskId, srcLocalRankId, rootRank, GetNslbOpType(opType),algType, nslbAdjInfo.nsAdjInfo.size());

    for (const auto& info : hcclNslbDpAlgorithmInfo_) {
        HCCL_INFO("NSLBDP_TYPE_TBL_ADJ info: *[%llu]-[%u]-[%u]-[%u]-[%u]* success.",
                  taskId, info.srcLocalRankId, info.rootRank, info.oper, info.algorithm);
        if (info.taskId == taskId && info.srcLocalRankId == srcLocalRankId &&
            info.rootRank ==  static_cast<u16>(rootRank) && info.oper == GetNslbOpType(opType) && info.algorithm == algType &&
            strcmp(algorithmInfo.commDesc, info.commDesc) == 0) {
            HCCL_INFO("hcclNslbDp Deduplication hcclNslbDpAlgorithmInfo_");
            return HCCL_SUCCESS;
        }
    }

    HCCL_INFO("NSLBDP-ADJ add adjINfo:***[%llu]***[%u]***[%u]***[%u]***[%u] success.",
                   taskId, srcLocalRankId, rootRank, GetNslbOpType(opType), algType);

    algorithmInfo.srcLocalRankId = srcLocalRankId;
    algorithmInfo.rootRank =  static_cast<u16>(rootRank);
    algorithmInfo.oper = GetNslbOpType(opType);
    algorithmInfo.algorithm = algType;

    algorithmInfo.dstRankNum = nslbAdjInfo.dstRankNum;
    HCCL_INFO("NSLBDP-ADJ nslbAdjInfo.dstRankNum:[%u].", nslbAdjInfo.dstRankNum);

    if (nslbAdjInfo.nsAdjInfo.size() == 0) {
        algorithmInfo.dstRankNum = 0;
        HCCL_INFO("hcclNslbDp get nsAdjInfo fail dstRankNum:[%u]", algorithmInfo.dstRankNum);
        return HCCL_SUCCESS;
    } else {
        for (size_t rankIndex = 0; rankIndex < nslbAdjInfo.nsAdjInfo.size(); rankIndex++) {
            NslbDpAdjInfo adjInfo;
            adjInfo.dstLocalRankId = nslbAdjInfo.nsAdjInfo[rankIndex].dstLocalRankId;
            adjInfo.phaseId = nslbAdjInfo.nsAdjInfo[rankIndex].phaseId;
            adjInfo.rev = nslbAdjInfo.nsAdjInfo[rankIndex].rev;
            HCCL_INFO("NSLBDP-ADJ adjINfo:[%u]-[%u]-[%u] success.",
                        srcLocalRankId, nslbAdjInfo.nsAdjInfo[rankIndex].dstLocalRankId, nslbAdjInfo.nsAdjInfo[rankIndex].phaseId);
            algorithmInfo.AdjInfo.push_back(adjInfo);
        }
    }

    algorithmInfo.sedFlag = 0;
    hcclNslbDpAlgorithmInfo_.push_back(algorithmInfo);
    HCCL_INFO("hcclNslbDp entry GetAlgAdjacencyTable end");
    return HCCL_SUCCESS;
}

bool hcclNslbDp::CheckSupportOptype(HcclCMDType opType)
{
    if (GetNslbOpType(opType) != NSLBDP_CMD_ALLREDUCE && GetNslbOpType(opType) != NSLBDP_CMD_ALLGATHER
        && GetNslbOpType(opType) != NSLBDP_CMD_REDUCE_SCATTER && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALL
        && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALLV && GetNslbOpType(opType) != NSLBDP_CMD_ALLTOALLVC
        && GetNslbOpType(opType) != NSLBDP_CMD_BROADCAST && GetNslbOpType(opType) != NSLBDP_CMD_SCATTER
        && GetNslbOpType(opType) != NSLBDP_CMD_BATCH_SEND_RECV && GetNslbOpType(opType) != NSLBDP_CMD_REDUCE
        && GetNslbOpType(opType) != NSLBDP_CMD_SEND) {
        return false;
    }
    return true;
}

bool hcclNslbDp::CheckCommDescExit(NslbDpOperatorInfo &OperatorInfo)
{
    HCCL_INFO("CheckCommDescExit hcclNslbDpCommConfig_ size:[%zu].", hcclNslbDpCommConfig_.size());
    for (const auto& info : hcclNslbDpCommConfig_) {
        HCCL_INFO("NSLBDP-OPER CheckCommDescExit info.commDesc[%s] .", info.commDesc);
        HCCL_INFO("NSLBDP-OPER CheckCommDescExit OperatorInfo.commDesc[%s] .", OperatorInfo.commDesc);
        if (strcmp(OperatorInfo.commDesc, info.commDesc) == 0) {
            return true;
        }
    }
    return false;
}

void hcclNslbDp::fullcommDescInitTime(std::string identifier, NslbDpOperatorInfo &OperatorInfo)
{
    // 获取通信域唯一标识
    s32 ret = strncpy_s(OperatorInfo.commDesc, COMM_DESC_MAX_LENGTH,  identifier.c_str(), identifier.size());
    if (ret != EOK) {
        HCCL_INFO("GetIdentifier str copy fail. return[%d]", ret);
        return;
    }
    HCCL_INFO("NSLBDP-OPER fullcommDescInitTime commDesc[%s] .", identifier.c_str());

    OperatorInfo.commDesc[COMM_DESC_MAX_LENGTH - 1] = '\0';

    std::vector<std::string> splitInfo;
    SplitString(identifier, splitInfo, "_");
    if (splitInfo.size() > NSLBDP_SPLIT_SIZE) {
        std::string trst = splitInfo[NSLBDP_SPLIT_SIZE];
        OperatorInfo.commInitTime = std::stoull(trst);
        OperatorInfo.commInitTime = OperatorInfo.commInitTime / NSLBDP_INITIME_MILLISENDS;
    } else {
        // commInitTime在有ranktable的赋值
        OperatorInfo.commInitTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    return;
}

bool hcclNslbDp::CheckSameOperatorVal(size_t operSize, NslbDpOperatorInfo &OperatorInfo, u32 rootRank)
{
    size_t num = hcclNslbDpOperatorVal_.size();
    if (operSize >= num) {
        return false;
    }
    // 保留         hcclNslbDpOperatorVal_[operSize].commInitTime == OperatorInfo.commInitTime &&
    if (hcclNslbDpOperatorVal_[operSize].taskId == OperatorInfo.taskId &&
        hcclNslbDpOperatorVal_[operSize].rootRank == rootRank &&
        hcclNslbDpOperatorVal_[operSize].oper == OperatorInfo.oper &&
        hcclNslbDpOperatorVal_[operSize].algorithm == OperatorInfo.algorithm &&
        strcmp(OperatorInfo.commDesc, hcclNslbDpOperatorVal_[operSize].commDesc) == 0) {
        return true;
    }
    return false;
}

// 写算法算子表--表2
HcclResult hcclNslbDp::GenerateOpAndAdjTable(HcclCMDType opType, u32 rootRank, u32 srcLocalRankId, u8 algType,
                                             std::string identifier, u64 count, u32 rankSize)
{
    HCCL_INFO("NSLB-HCCL count=[%llu], hcclNslbDpOperatorVal_ size:[%zu].", count, hcclNslbDpOperatorVal_.size());
    u64 taskId = GetGlobalCommTaskId();
    if (taskId == 0) {
        return HCCL_SUCCESS;
    }

    // 获取，0号
    if (rootRank != 0) {
        return HCCL_SUCCESS;
    }
    u64 trafficNum = count;
    if (CheckSupportOptype(opType) == false) {
        trafficNum = 0;
        HCCL_INFO("NSLB-HCCL CheckSupportOptype false trafficNum == 0.");
    }

    NslbDpOperatorInfo OperatorInfo = {};
    // 获取task id 
    OperatorInfo.taskId = taskId;
    s32 sRet = memset_s(OperatorInfo.commDesc, COMM_DESC_MAX_LENGTH, 0, sizeof(OperatorInfo.commDesc));
    if (sRet != EOK) {
        HCCL_ERROR("memset_s commDesc fail");
        return HCCL_SUCCESS;
    }

    fullcommDescInitTime(identifier, OperatorInfo);

    // 去除不存在的通信域信息
    if (CheckCommDescExit(OperatorInfo) == false) {
        HCCL_INFO("[nslbdp-oper] CheckCommDesc not exit ");
        return HCCL_SUCCESS;
    }

    // 获取operator、algorithm
    OperatorInfo.oper = GetNslbOpType(opType);
    OperatorInfo.algorithm = algType;
    u64 trafficCount = GetNslbDpFirstFourBit(OperatorInfo.oper, OperatorInfo.algorithm);
    trafficCount = (trafficCount << NSLBDP_TRAFFICCONUT) + trafficNum;

    for (size_t operSize = 0; operSize < hcclNslbDpOperatorVal_.size(); operSize++) {
        if (CheckSameOperatorVal(operSize, OperatorInfo, rootRank) == true) {
            if (hcclNslbDpOperatorVal_[operSize].trafficCnt < trafficCount && srcLocalRankId == 0) {
                hcclNslbDpOperatorVal_[operSize].trafficCnt = trafficCount;
                hcclNslbDpOperatorVal_[operSize].sedFlag = 0;
                SendRankTableOpAndAdj(hcclNslbDpOperatorVal_[operSize]);
                HCCL_RUN_INFO("NSLBDP-ADJ commDesc[%s] try to update trafficCnt[%llu] success.", OperatorInfo.commDesc, trafficCount);
            }
            return HCCL_SUCCESS;
        }
    }

    OperatorInfo.trafficCnt = trafficCount; // 判断变大
    OperatorInfo.rootRank = rootRank;
    HCCL_INFO("NSLBDP-OPER add operInfo:***[%llu]***[%llu]***[%u]***[%u]***[%u] success.",
                   taskId, OperatorInfo.commInitTime, rootRank, OperatorInfo.oper, OperatorInfo.algorithm);

    GetNslbDpl4SPortId(rankSize, algType, &OperatorInfo.l4SPortId);
    if (srcLocalRankId == 0) {
        SendRankTableOpAndAdj(OperatorInfo);
    }
    hcclNslbDpOperatorVal_.push_back(OperatorInfo);

    return HCCL_SUCCESS;
}

/*根将表1 序列化处理*/
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableFir(NslbDpCommConfigInfo cominfo) 
{
    HCCL_INFO("hcclNslbDp entry serializeTLV_TableFir beg");

    std::vector<uint8_t> tlvData;

    // Task ID
    uint64_t netTaskId = htobe64(cominfo.taskId); 
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // CommDesc
    tlvData.insert(tlvData.end(), cominfo.commDesc, cominfo.commDesc + COMM_DESC_MAX_LENGTH);

    // CommInitTime
    uint64_t netCommInitTime = htobe64(cominfo.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime, sizeof(netCommInitTime));

    // packetId
    uint16_t netPacketId = htons(cominfo.packetId);
    tlvData.resize(tlvData.size() + sizeof(netPacketId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netPacketId), sizeof(netPacketId), &netPacketId, sizeof(netPacketId));

    // rev
    uint16_t netRev = htons(cominfo.rev);
    tlvData.resize(tlvData.size() + sizeof(netRev));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev), sizeof(netRev), &netRev, sizeof(netRev));

    // packetNum
    uint16_t netPacketNum = htons(cominfo.packetNum);
    tlvData.resize(tlvData.size() + sizeof(netPacketNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netPacketNum), sizeof(netPacketNum), &netPacketNum, sizeof(netPacketNum));

    // revSecond
    uint16_t netRevSecond = htons(cominfo.revSecond);
    tlvData.resize(tlvData.size() + sizeof(netRevSecond));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRevSecond), sizeof(netRevSecond), &netRevSecond, sizeof(netRevSecond));

    // sendRankInfo
    for (const auto& rankInfo : cominfo.sendRankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp, sizeof(netDeviceIp));
    }

    // CommMd5Sum 
    tlvData.insert(tlvData.end(), cominfo.commMd5Sum, cominfo.commMd5Sum + sizeof(cominfo.commMd5Sum));

    // RankTotalNum
    uint16_t netRankTotalNum = htons(cominfo.rankTotalNum);
    tlvData.resize(tlvData.size() + sizeof(netRankTotalNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRankTotalNum), sizeof(netRankTotalNum), &netRankTotalNum, sizeof(netRankTotalNum));

    // Rank Number 
    uint16_t netRankNum = htons(cominfo.rankNum);
    tlvData.resize(tlvData.size() + sizeof(netRankNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRankNum), sizeof(netRankNum), &netRankNum, sizeof(netRankNum));

    // Rank Info
    for (const auto& rankInfo : cominfo.rankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        uint32_t netServerIp = htonl(rankInfo.serverIp);
        uint16_t netPodId = htons(rankInfo.podId);
        uint16_t netRev3 = htons(rankInfo.rev);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp, sizeof(netDeviceIp));
        tlvData.resize(tlvData.size() + sizeof(netServerIp));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netServerIp), sizeof(netServerIp), &netServerIp, sizeof(netServerIp));
        tlvData.resize(tlvData.size() + sizeof(netPodId));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netPodId), sizeof(netPodId), &netPodId, sizeof(netPodId));
        tlvData.resize(tlvData.size() + sizeof(netRev3));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev3), sizeof(netRev3), &netRev3, sizeof(netRev3));
    }
    HCCL_INFO("hcclNslbDp entry serializeTLV_TableFir end");

    return tlvData;
}

void hcclNslbDp::fullCommConfigInfo(NslbDpCommConfigInfo &tab_f, NslbDpCommConfigVal cominfo, u32 packetNum)
{
    tab_f.taskId = cominfo.taskId;
    s32 sRet = memcpy_s(tab_f.commDesc, sizeof(tab_f.commDesc), cominfo.commDesc, COMM_DESC_MAX_LENGTH);
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s commDesc fail");
    }
    tab_f.commInitTime = cominfo.commInitTime;

    tab_f.rev = 0;
    tab_f.packetNum = packetNum;
    tab_f.revSecond = 0;
    for (u32 dip = 0; dip < packetNum; dip++) {
        if(cominfo.rankInfo.size() == 0) {
            break;
        }
        tab_f.sendRankInfo[dip].deviceIp = cominfo.rankInfo[dip * COMM_SLICING_LENGTH].deviceIp;
    }
    sRet = memcpy_s(tab_f.commMd5Sum, sizeof(tab_f.commMd5Sum), cominfo.commMd5Sum, sizeof(cominfo.commMd5Sum));
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s commmd5 fail");
    }
}

/*根据表1 中的rank数量进行分片处理*/
HcclResult hcclNslbDp::SendTableProc(u32 packetNum, NslbDpCommConfigVal cominfo)
{
    HCCL_INFO("hcclNslbDp entry SendTableProc packetNum:[%u]", packetNum);
    for (u32 i = 1; i <= packetNum; i++) {
        NslbDpCommConfigInfo tab_f = {};
        fullCommConfigInfo(tab_f, cominfo, packetNum);

        if (i != packetNum) {
            tab_f.packetId = i - 1;
            if(cominfo.rankInfo.size() < i * NSLBDP_RANKTOTALNUM_BLOCK_FIR) {
                HCCL_INFO("hcclNslbDp COmm RankInfo not as expected");
                return HCCL_SUCCESS;
            }
            for (u32 rankindex = (i-1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR; rankindex < i * NSLBDP_RANKTOTALNUM_BLOCK_FIR; rankindex++) {
                NslbDpRankInfo rankInfoTemp;
                rankInfoTemp.deviceIp = cominfo.rankInfo[rankindex].deviceIp;
                rankInfoTemp.serverIp = cominfo.rankInfo[rankindex].serverIp;
                rankInfoTemp.podId = cominfo.rankInfo[rankindex].podId;
                rankInfoTemp.rev = cominfo.rankInfo[rankindex].rev;
                tab_f.rankInfo.push_back(rankInfoTemp);
            }
            tab_f.rankTotalNum = NSLBDP_RANKTOTALNUM_BLOCK_FIR;
            tab_f.rankNum = NSLBDP_RANKTOTALNUM_BLOCK_FIR;
        } else {
            tab_f.packetId = packetNum - 1;
            u16 rankNum = cominfo.rankInfo.size();
            HCCL_INFO("HCCL SendTableFir rankNum:[%u] success.", rankNum);

            for (u32 j = (i-1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR; j < cominfo.rankInfo.size(); j++) {
                NslbDpRankInfo rankInfoTemp;
                rankInfoTemp.deviceIp = cominfo.rankInfo[j].deviceIp;
                rankInfoTemp.serverIp = cominfo.rankInfo[j].serverIp;
                rankInfoTemp.podId = cominfo.rankInfo[j].podId;
                rankInfoTemp.rev = cominfo.rankInfo[j].rev;
                tab_f.rankInfo.push_back(rankInfoTemp);
            }
            tab_f.rankTotalNum = cominfo.rankTotalNum;
            HCCL_INFO("HCCL SendTableFir rankTotalNum:[%u] success.", tab_f.rankTotalNum);
            tab_f.rankNum = rankNum;
        }
        SendRankTable(tab_f);
    }
    return HCCL_SUCCESS;
}

/*遍历表1 执行send 流程*/
HcclResult hcclNslbDp::SendTableFir(uint32_t rank)
{
    HCCL_INFO("hcclNslbDp entry SendTableFir");
    size_t size = hcclNslbDpCommConfig_.size();
    HCCL_INFO("HCCL SendTableFir size:[%u] success.", size);
    for (size_t i = 0; i < hcclNslbDpCommConfig_.size(); i++) {
        u32 rankTotalNum = hcclNslbDpCommConfig_[i].rankTotalNum;
        if (rankTotalNum > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
            return HCCL_SUCCESS;
        }
        u32 packetNum = NSLBDP_PKTNUM_FIR;
        if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_FIR) {
            packetNum = NSLBDP_PKTNUM_FIR;
        } else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_SEC) {
            packetNum = NSLBDP_PKTNUM_SEC;
        } else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_THR) {
            packetNum = NSLBDP_PKTNUM_THR;
        } else {
            packetNum = NSLBDP_PKTNUM_FOU;
        }
        if (packetNum - 1 != rank) {
            HCCL_INFO("hcclNslbDp only rank [%u] send TableFir.");
            return HCCL_SUCCESS;
        }
        SendTableProc(packetNum, hcclNslbDpCommConfig_[i]); 
    }
    HCCL_INFO("hcclNslbDp entry SendTableFir end");
    return HCCL_SUCCESS;
}


HcclResult hcclNslbDp::SendRankTable(NslbDpCommConfigInfo tab_f)
{
    if (getHccpInitFlag() == false) {
        return HCCL_SUCCESS;
    }

    u32 tablen = sizeof(tab_f);
    HCCL_INFO("HCCL SendRankTable NslbDpCommConfigInfo len:[%u].", tablen);

    std::vector<uint8_t> tlvData = serializeTLV_TableFir(tab_f);
    u32 datlen = tlvData.size();
    HCCL_INFO("HCCL SendRankTable tlvData.len:[%u] success.", datlen);

    if(nslbdp_handle_ == nullptr) {
        HCCL_INFO("HCCL ndlbdp  nslbdp_handle_ error SendRankTable.");
        return HCCL_SUCCESS;
    }

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_COMM_INFO;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(nslbdp_handle_, reinterpret_cast<tlv_msg*>(&sendMsg), reinterpret_cast<tlv_msg*>(&recvMsg));

	HCCL_INFO("!!!!!!!!!!hccl send Table 1 to hccp. ret(%d)!!!!!!!!!!\n", ret);

    return HCCL_SUCCESS;
}


/*根将表2 序列化处理*/
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableOpAndAdj(NslbDpOperatorInfo &info) 
{
    HCCL_INFO("HCCL ndlbdp entry serializeTLV_TableOpAndAdj.");
    std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), info.commDesc, info.commDesc + COMM_DESC_MAX_LENGTH);

    // 处理 commInitTime
    uint64_t netCommInitTime = htobe64(info.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime, sizeof(netCommInitTime));

    // 处理 oper
    tlvData.push_back(info.oper);

    // 处理 algorithm
    tlvData.push_back(info.algorithm);

    // 处理 rootRank
    uint16_t netRootRank = htons(info.rootRank);
    tlvData.resize(tlvData.size() + sizeof(netRootRank));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRootRank), sizeof(netRootRank), &netRootRank, sizeof(netRootRank));

    // 处理 trafficCnt
    uint64_t netTrafficCnt = htobe64(info.trafficCnt);
    tlvData.resize(tlvData.size() + sizeof(netTrafficCnt));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netTrafficCnt), sizeof(netTrafficCnt), &netTrafficCnt, sizeof(netTrafficCnt));

    // 处理 l4SPortId
    uint16_t netL4SPortId = htons(info.l4SPortId);
    tlvData.resize(tlvData.size() + sizeof(netL4SPortId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netL4SPortId), sizeof(netL4SPortId), &netL4SPortId, sizeof(netL4SPortId));

    // 处理 maskLen
    uint16_t netMaskLen = htons(info.maskLen);
    tlvData.resize(tlvData.size() + sizeof(netMaskLen));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netMaskLen), sizeof(netMaskLen), &netMaskLen, sizeof(netMaskLen));
    HCCL_INFO("HCCL ndlbdp entry serializeTLV_TableOpAndAdj end.");
    return tlvData;
}


/*执行send 流程*/
HcclResult hcclNslbDp::SendRankTableOpAndAdj(NslbDpOperatorInfo &tab_f)
{
    if (getHccpInitFlag() == false) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("HCCL ndlbdp entry SendRankTableOpAndAdj.");
    std::vector<uint8_t> tlvData = serializeTLV_TableOpAndAdj(tab_f);

    u32 datlen = tlvData.size();
    HCCL_INFO("HCCL SendRankTableOpAndAdj tlvData.len:[%u] success.", datlen);
	
    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_OPER;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(nslbdp_handle_, reinterpret_cast<tlv_msg*>(&sendMsg), reinterpret_cast<tlv_msg*>(&recvMsg));

	HCCL_INFO("!!!!!!!!!!hccl send Table 2 to hccp. ret(%d)!!!!!!!!!!\n", ret);

    return HCCL_SUCCESS;
}


/*遍历表2 执行send 流程*/
HcclResult hcclNslbDp::SendOpAndAdjTable()
{
    HCCL_INFO("HCCL ndlbdp entry SendOpAndAdjTable.");
    for (size_t i = 0; i < hcclNslbDpOperatorVal_.size(); i++) {
        if (hcclNslbDpOperatorVal_[i].sedFlag == 1) {
            continue;
        }
        NslbDpOperatorInfo tab_f = {};
        tab_f.taskId = hcclNslbDpOperatorVal_[i].taskId;
        s32 sRet = memcpy_s(tab_f.commDesc, sizeof(tab_f.commDesc), hcclNslbDpOperatorVal_[i].commDesc, COMM_DESC_MAX_LENGTH);
        if (sRet != EOK) {
            HCCL_ERROR("memcpy_s commDesc fail");
        }
        tab_f.commInitTime = hcclNslbDpOperatorVal_[i].commInitTime;
		tab_f.rootRank = hcclNslbDpOperatorVal_[i].rootRank;
		tab_f.oper = hcclNslbDpOperatorVal_[i].oper;
		tab_f.algorithm = hcclNslbDpOperatorVal_[i].algorithm;
		tab_f.trafficCnt = hcclNslbDpOperatorVal_[i].trafficCnt;
		tab_f.l4SPortId = hcclNslbDpOperatorVal_[i].l4SPortId;
		tab_f.maskLen = hcclNslbDpOperatorVal_[i].maskLen;
        SendRankTableOpAndAdj(tab_f);
        hcclNslbDpOperatorVal_[i].sedFlag = 1;
        HCCL_INFO("HCCL try to sen RankTableOpAndAdj times:[%u].", i);
    }
    HCCL_INFO("HCCL ndlbdp entry SendOpAndAdjTable end.");
    return HCCL_SUCCESS;
}


/*根将表3 序列化处理*/
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableAlgorithmInfo(NslbDpAlgorithmTlv &info) 
{
    HCCL_INFO("HCCL ndlbdp entry serializeTLV_TableAlgorithmInfo.");
	std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), info.commDesc, info.commDesc + COMM_DESC_MAX_LENGTH);

    // CommMd5Sum 
    tlvData.insert(tlvData.end(), info.commMd5Sum, info.commMd5Sum + sizeof(info.commMd5Sum));

    // 处理 srcLocalRankId
    uint16_t netSrcLocalRankId = htons(info.srcLocalRankId);
    tlvData.resize(tlvData.size() + sizeof(netSrcLocalRankId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netSrcLocalRankId), sizeof(netSrcLocalRankId), &netSrcLocalRankId, sizeof(netSrcLocalRankId));

    // 处理 oper
    tlvData.push_back(info.oper);

    // 处理 algorithm
    tlvData.push_back(info.algorithm);

    // 处理 rootRank
    uint16_t netRootRank = htons(info.rootRank);
    tlvData.resize(tlvData.size() + sizeof(netRootRank));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRootRank), sizeof(netRootRank), &netRootRank, sizeof(netRootRank));

    // 处理 rev
    uint16_t netRev = htons(info.rev);
    tlvData.resize(tlvData.size() + sizeof(netRev));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev), sizeof(netRev), &netRev, sizeof(netRev));

    // 处理 dstRankNum
    uint16_t netDstRankNum = htons(info.dstRankNum);
    tlvData.resize(tlvData.size() + sizeof(netDstRankNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netDstRankNum), sizeof(netDstRankNum), &netDstRankNum, sizeof(netDstRankNum));

    // 处理 revSecond
    uint16_t netRevSecond = htons(info.revsecond);
    tlvData.resize(tlvData.size() + sizeof(netRevSecond));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRevSecond), sizeof(netRevSecond), &netRevSecond, sizeof(netRevSecond));

    HCCL_INFO("HCCL ndlbdp serializeTLV_TableAlgorithmInfo proc adjInfo.");
    // 处理 adjInfo
    for (const auto& adj : info.AdjInfo) {
        uint16_t netDstLocalRankId = htons(adj.dstLocalRankId);
        tlvData.resize(tlvData.size() + sizeof(netDstLocalRankId));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netDstLocalRankId), sizeof(netDstLocalRankId), &netDstLocalRankId, sizeof(netDstLocalRankId));
        tlvData.push_back(adj.phaseId);
        tlvData.push_back(adj.rev);
        HCCL_INFO("serializeTLV_TableAlgorithmInfo adjInfo:***[%u]***[%c]***.", adj.dstLocalRankId, adj.phaseId);
    }
    HCCL_INFO("HCCL ndlbdp serializeTLV_TableAlgorithmInfo proc adjInfo end.");

	return tlvData;
}


HcclResult hcclNslbDp::SendRankTableAlgorithmInfo(NslbDpAlgorithmTlv &tab_f)
{
    if (getHccpInitFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableAlgorithmInfo(tab_f);

    u32 datlen = tlvData.size();
    HCCL_INFO("HCCL SendRankTableAlgorithmInfo tlvData.len:[%u] success.", datlen);
	
    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_ADJ;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(nslbdp_handle_, reinterpret_cast<tlv_msg*>(&sendMsg), reinterpret_cast<tlv_msg*>(&recvMsg));

	HCCL_INFO("!!!!!!!!!!hccl send Table 3 to hccp. ret(%d)!!!!!!!!!!\n", ret);
    return HCCL_SUCCESS;
}

/*遍历表3 执行send 流程*/
HcclResult hcclNslbDp::SendAlgorithmInfoTable()
{
    u32 size = hcclNslbDpAlgorithmInfo_.size();
    HCCL_INFO("HCCL ndlbdp entry SendAlgorithmInfoTable size=[%u].", size);

    for (size_t i = 0; i < hcclNslbDpAlgorithmInfo_.size(); i++) {
        if (hcclNslbDpAlgorithmInfo_[i].sedFlag == 1) {
            continue;
        }
        NslbDpAlgorithmTlv tab_f = {};
        tab_f.taskId = hcclNslbDpAlgorithmInfo_[i].taskId;
        s32 sRet = memcpy_s(tab_f.commDesc, sizeof(tab_f.commDesc), hcclNslbDpAlgorithmInfo_[i].commDesc, COMM_DESC_MAX_LENGTH);
        sRet = memcpy_s(tab_f.commMd5Sum, sizeof(tab_f.commMd5Sum), 
                        hcclNslbDpAlgorithmInfo_[i].commMd5Sum, sizeof(hcclNslbDpAlgorithmInfo_[i].commMd5Sum));
        if (sRet != EOK) {
            HCCL_INFO("memcpy_s commDesc fail");
        }
		tab_f.srcLocalRankId = hcclNslbDpAlgorithmInfo_[i].srcLocalRankId;
		tab_f.rootRank = hcclNslbDpAlgorithmInfo_[i].rootRank;
		tab_f.oper = hcclNslbDpAlgorithmInfo_[i].oper;
		tab_f.algorithm = hcclNslbDpAlgorithmInfo_[i].algorithm;

        tab_f.rev = 0;

		tab_f.dstRankNum = hcclNslbDpAlgorithmInfo_[i].dstRankNum;
        HCCL_INFO("[nslbdp-ring]dstRankNum:[%u]success.", tab_f.dstRankNum);
		tab_f.revsecond = 0;
		tab_f.AdjInfo = hcclNslbDpAlgorithmInfo_[i].AdjInfo;
		for (const auto& adj : hcclNslbDpAlgorithmInfo_[i].AdjInfo) { 
			tab_f.AdjInfo.push_back(adj); 
		}
        SendRankTableAlgorithmInfo(tab_f);
        HCCL_INFO("HCCL try to sen AlgorithmInfoTable times:[%u].", i);

        hcclNslbDpAlgorithmInfo_[i].sedFlag = 1;
    }
    HCCL_INFO("HCCL ndlbdp entry SendAlgorithmInfoTable end.");

    return HCCL_SUCCESS;
}


// 序列化表4
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableGlobalRankInfo(NslbDpGlobalRankInfo &info)
{
    HCCL_INFO("HCCL ndlbdp entry serializeTLV_TableGlobalRankInfo.");
    std::vector<uint8_t> tlvData;
    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), info.commDesc, info.commDesc + COMM_DESC_MAX_LENGTH);

    // 处理 commInitTime
    uint64_t netCommInitTime = htobe64(info.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime, sizeof(netCommInitTime));

    // 处理 packetId
    uint16_t netPacketId = htons(info.packetId);
    tlvData.resize(tlvData.size() + sizeof(netPacketId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netPacketId), sizeof(netPacketId), &netPacketId, sizeof(netPacketId));

    // 处理 rev
    uint16_t netRev = htons(info.rev);
    tlvData.resize(tlvData.size() + sizeof(netRev));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev), sizeof(netRev), &netRev, sizeof(netRev));

    HCCL_INFO("HCCL hcclNslbDpGlobalRankVal_ info.packetNum:[%u]", info.packetNum);
    // 处理 packetNum
    uint16_t netPacketNum = htons(info.packetNum);
    tlvData.resize(tlvData.size() + sizeof(netPacketNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netPacketNum), sizeof(netPacketNum), &netPacketNum, sizeof(netPacketNum));
    HCCL_INFO("HCCL hcclNslbDpGlobalRankVal_ info.packetNum:[%u]", netPacketNum);

    // 处理 rev2
    uint16_t netRev2 = htons(info.rev2);
    tlvData.resize(tlvData.size() + sizeof(netRev2));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev2), sizeof(netRev2), &netRev2, sizeof(netRev2));

    // 处理 sendRankInfo
    for (const auto& rankInfo : info.sendRankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp, sizeof(netDeviceIp));
    }

    // 处理 commMd5Sum
    tlvData.insert(tlvData.end(), info.commMd5Sum, info.commMd5Sum + sizeof(info.commMd5Sum));

    // 处理 rankTotalNum
    uint32_t netRankTotalNum = htonl(info.rankTotalNum);
    tlvData.resize(tlvData.size() + sizeof(netRankTotalNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRankTotalNum), sizeof(netRankTotalNum), &netRankTotalNum, sizeof(netRankTotalNum));

    // 处理 rankNum
    uint16_t netRankNum = htons(info.rankNum);
    tlvData.resize(tlvData.size() + sizeof(netRankNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRankNum), sizeof(netRankNum), &netRankNum, sizeof(netRankNum));

    // 处理 rev3
    uint16_t netRev3 = htons(info.rev3);
    tlvData.resize(tlvData.size() + sizeof(netRev3));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRev3), sizeof(netRev3), &netRev3, sizeof(netRev3));

    // 处理 rankInfo
    for (const auto& rankInfo : info.rankInfo) {
        uint32_t netDeviceIp = htonl(rankInfo.deviceIp);
        uint32_t netServerIp = htonl(rankInfo.serverIp);
        tlvData.resize(tlvData.size() + sizeof(netDeviceIp));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netDeviceIp), sizeof(netDeviceIp), &netDeviceIp, sizeof(netDeviceIp));
        tlvData.resize(tlvData.size() + sizeof(netServerIp));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netServerIp), sizeof(netServerIp), &netServerIp, sizeof(netServerIp));
    }
    return tlvData;
}


// 发送Rank表信息
HcclResult hcclNslbDp::SendRankTableGlobalRank(NslbDpGlobalRankInfo &tab_f)
{
    if (getHccpInitFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableGlobalRankInfo(tab_f);
    u32 datlen = tlvData.size();
    HCCL_INFO("HCCL SendRankTableGlobalRank tlvData.len:[%u] success.", datlen);

    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_RANK;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(nslbdp_handle_, reinterpret_cast<tlv_msg*>(&sendMsg), reinterpret_cast<tlv_msg*>(&recvMsg));

    HCCL_INFO("!!!!!!!!!!hccl send Table 4 to hccp. ret(%d)!!!!!!!!!!\n", ret);

    return HCCL_SUCCESS;
}

void hcclNslbDp::fullCommonGlobalRankInfo(NslbDpGlobalRankInfo tab_f, NslbDpGlobalRankVal &cominfo)
{
    tab_f.rev = 0;
    tab_f.rev2 = 0;
    tab_f.rev3 = 0;
    tab_f.taskId = cominfo.taskId;
    s32 sRet = memcpy_s(tab_f.commDesc, sizeof(tab_f.commDesc), cominfo.commDesc, sizeof(cominfo.commDesc));
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s commDesc fail");
    }
    tab_f.commInitTime = cominfo.commInitTime;
    sRet = memcpy_s(tab_f.commMd5Sum, sizeof(tab_f.commMd5Sum), cominfo.commMd5Sum, sizeof(cominfo.commMd5Sum));
    if (sRet != EOK) {
        HCCL_ERROR("memcpy_s commMD5 fail");
    }

    u16 rankTotalNum = cominfo.rankTotalNum;
    size_t dipsize = rankTotalNum < 256 ? rankTotalNum : 256;

    for (size_t dip = 0; dip < dipsize; dip++) {
        size_t dpsize = cominfo.rankInfo.size();
        if(dpsize < dip * COMM_SLICING_LENGTH) {
            HCCL_ERROR("HCCL ndlbdp SendTableGlobalRankProc dip:[%u] dpsize:[%u]", dip, dpsize);
            break;
        }
        u32 tempdeviceIp = cominfo.rankInfo[dip * COMM_SLICING_LENGTH].deviceIp;
        tab_f.sendRankInfo[dip].deviceIp = tempdeviceIp;
    }
}

/*根据表4 中的rank数量进行分片处理*/
HcclResult hcclNslbDp::SendTableGlobalRankProc(uint32_t packetNum, NslbDpGlobalRankVal &cominfo)
{
    HCCL_INFO("HCCL ndlbdp entry SendTableGlobalRankProc packetNum:[%u]", packetNum);
    for (uint32_t i = 1; i <= packetNum; i++) {
        NslbDpGlobalRankInfo tab_f = {};
        fullCommonGlobalRankInfo(tab_f, cominfo);
        if (i != packetNum) {
            tab_f.packetId = i;
            tab_f.packetNum = packetNum;
            uint32_t start = (i - 1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
            uint32_t end = i * NSLBDP_RANKTOTALNUM_BLOCK_FIR;
            end = (end > cominfo.rankInfo.size()) ? cominfo.rankInfo.size() : end; // Ensure end does not exceed rankInfo size
            for (uint32_t j = start; j < end; ++j) {
                TableFourRankInfo rankInfoTemp;
                rankInfoTemp.deviceIp = cominfo.rankInfo[j].deviceIp;
                rankInfoTemp.serverIp = cominfo.rankInfo[j].serverIp;
                tab_f.rankInfo.push_back(rankInfoTemp);
            }
            tab_f.rankTotalNum = NSLBDP_RANKTOTALNUM_BLOCK_FIR;
            tab_f.rankNum = NSLBDP_RANKTOTALNUM_BLOCK_FIR;
            SendRankTableGlobalRank(tab_f);
        } else {
            HCCL_INFO("HCCL ndlbdp entry SendTableGlobalRankProc second.");
            tab_f.packetId = i;
            tab_f.packetNum = packetNum;
            uint32_t rankNum = cominfo.rankInfo.size();
            uint32_t start = (i - 1) * NSLBDP_RANKTOTALNUM_BLOCK_FIR;

            for (uint32_t j = start; j < rankNum; j++) {
                TableFourRankInfo rankInfoTemp;
                rankInfoTemp.deviceIp = cominfo.rankInfo[j].deviceIp;
                rankInfoTemp.serverIp = cominfo.rankInfo[j].serverIp;
                tab_f.rankInfo.push_back(rankInfoTemp);
            }
            tab_f.rankTotalNum = cominfo.rankTotalNum;
            tab_f.rankNum = rankNum;
            SendRankTableGlobalRank(tab_f);
        }
    }
    return HCCL_SUCCESS;
}

// 根据rankNum的总数做分片发送流程
HcclResult hcclNslbDp::SendGlobalRankTable()
{
    HCCL_INFO("HCCL ndlbdp try to send GlobalRankTable.");
	u32 rankTotalNum  = hcclNslbDpGlobalRankVal_.rankTotalNum;
    if (rankTotalNum > NSLBDP_RANKTOTALNUM_BLOCK_FOU) {
        return HCCL_SUCCESS;
    }
    u32 packetNum;
	if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_FIR) {
        packetNum = NSLBDP_PKTNUM_FIR;
    }else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_SEC) {
        packetNum = NSLBDP_PKTNUM_SEC;
    } else if (rankTotalNum <= NSLBDP_RANKTOTALNUM_BLOCK_THR) {
        packetNum = NSLBDP_PKTNUM_THR;
    } else {
        packetNum = NSLBDP_PKTNUM_FOU;
    }
    SendTableGlobalRankProc(packetNum, hcclNslbDpGlobalRankVal_);

    return HCCL_SUCCESS;
}

// 序列化表5
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableGlobalDisRankVal(NslbDpGlobalDisRankVal &info)
{
    HCCL_INFO("HCCL ndlbdp entry serializeTLV_TableGlobalDisRankVal.");
    std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(info.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 保留 tlvData.insert(tlvData.end(), reinterpret_cast<const uint8_t*>(&netTaskId), reinterpret_cast<const uint8_t*>(&netTaskId) + sizeof(netTaskId));
    // 处理 npuIp
    uint32_t netNpuIp = htonl(info.npuIp);
    tlvData.resize(tlvData.size() + sizeof(netNpuIp));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netNpuIp), sizeof(netNpuIp), &netNpuIp, sizeof(netNpuIp));

    // 保留 tlvData.insert(tlvData.end(), reinterpret_cast<const uint8_t*>(&netNpuIp), reinterpret_cast<const uint8_t*>(&netNpuIp) + sizeof(netNpuIp));
    // 处理 serverIp
    uint32_t netServerIp = htonl(info.serverIp);
    tlvData.resize(tlvData.size() + sizeof(netServerIp));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netServerIp), sizeof(netServerIp), &netServerIp, sizeof(netServerIp));

    // 保留 tlvData.insert(tlvData.end(), reinterpret_cast<const uint8_t*>(&netServerIp), reinterpret_cast<const uint8_t*>(&netServerIp) + sizeof(netServerIp));
    // 处理 nodeId
    uint32_t netNodeId = htonl(info.nodeId);
    tlvData.resize(tlvData.size() + sizeof(netNodeId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netNodeId), sizeof(netNodeId), &netNodeId, sizeof(netNodeId));

    // 保留 tlvData.insert(tlvData.end(), reinterpret_cast<const uint8_t*>(&netNodeId), reinterpret_cast<const uint8_t*>(&netNodeId) + sizeof(netNodeId));
    // 处理 localRankNum
    tlvData.push_back(info.localRankNum);

    // 处理 rev
    tlvData.insert(tlvData.end(), info.rev, info.rev + sizeof(info.rev));

    // 处理 rankTotalNum
    uint32_t netRankTotalNum = htonl(info.rankTotalNum);
    tlvData.resize(tlvData.size() + sizeof(netRankTotalNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRankTotalNum), sizeof(netRankTotalNum), &netRankTotalNum, sizeof(netRankTotalNum));

    // 保留 tlvData.insert(tlvData.end(), reinterpret_cast<const uint8_t*>(&netRankTotalNum), reinterpret_cast<const uint8_t*>(&netRankTotalNum) + sizeof(netRankTotalNum));
    return tlvData;
}


// 发送Rank表信息
HcclResult hcclNslbDp::SendRankTableGlobalDisRankVal(NslbDpGlobalDisRankVal &tab_f)
{
    if (getHccpInitFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableGlobalDisRankVal(tab_f);
    u32 datlen = tlvData.size();
    HCCL_INFO("HCCL SendRankTableGlobalDisRankVal tlvData.len:[%u] success.", datlen);
	
    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_RANK_DIST;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(nslbdp_handle_, reinterpret_cast<tlv_msg*>(&sendMsg), reinterpret_cast<tlv_msg*>(&recvMsg));

    HCCL_INFO("!!!!!!!!!!hccl send Table NSLBDP_TYPE_TBL_RANK_DIST to hccp. ret(%d)!!!!!!!!!!\n", ret);
    return HCCL_SUCCESS;
}


// 遍历表5 执行send流程
HcclResult hcclNslbDp::SendGlobalDisRankTable()
{
    HCCL_INFO("HCCL ndlbdp entry serializeTLV_TableGlobalDisRankVal.");
    NslbDpGlobalDisRankVal tab_f = {};
    tab_f.taskId = hcclNslbDpGlobalDisRankVal_.taskId;
    tab_f.npuIp = hcclNslbDpGlobalDisRankVal_.npuIp;
    tab_f.serverIp = hcclNslbDpGlobalDisRankVal_.serverIp;
    tab_f.nodeId = hcclNslbDpGlobalDisRankVal_.nodeId;
    tab_f.localRankNum = hcclNslbDpGlobalDisRankVal_.localRankNum;
    s32 sRet = memcpy_s(tab_f.rev, sizeof(tab_f.rev), hcclNslbDpGlobalDisRankVal_.rev, sizeof(hcclNslbDpGlobalDisRankVal_.rev));
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s rev info fail");
    }
    tab_f.rankTotalNum = hcclNslbDpGlobalDisRankVal_.rankTotalNum;
        
    SendRankTableGlobalDisRankVal(tab_f);
    return HCCL_SUCCESS;
}


// 序列化表6
std::vector<uint8_t> hcclNslbDp::serializeTLV_TableRootRank(NslbDpRootRank &config)
{
    HCCL_INFO("HCCL ndlbdp entry serializeTLV_TableRootRank.");
    std::vector<uint8_t> tlvData;

    // 处理 taskId
    uint64_t netTaskId = htobe64(config.taskId);
    tlvData.resize(tlvData.size() + sizeof(netTaskId));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netTaskId), sizeof(netTaskId), &netTaskId, sizeof(netTaskId));

    // 处理 commDesc
    tlvData.insert(tlvData.end(), config.commDesc, config.commDesc + COMM_DESC_MAX_LENGTH);

    // 处理 commInitTime
    uint64_t netCommInitTime = htobe64(config.commInitTime);
    tlvData.resize(tlvData.size() + sizeof(netCommInitTime));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netCommInitTime), sizeof(netCommInitTime), &netCommInitTime, sizeof(netCommInitTime));

    // 处理 oper
    tlvData.push_back(config.oper);

    // 处理 algorithm
    tlvData.push_back(config.algorithm);

    // 处理 revfir
    uint16_t netRevfir = htons(config.revfir);
    tlvData.resize(tlvData.size() + sizeof(netRevfir));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRevfir), sizeof(netRevfir), &netRevfir, sizeof(netRevfir));

    // 处理 rootRankNum
    uint16_t netRootRankNum = htons(config.rootRankNum);
    tlvData.resize(tlvData.size() + sizeof(netRootRankNum));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRootRankNum), sizeof(netRootRankNum), &netRootRankNum, sizeof(netRootRankNum));

    // 处理 revsec
    uint16_t netRevsec = htons(config.revsec);
    tlvData.resize(tlvData.size() + sizeof(netRevsec));
    (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRevsec), sizeof(netRevsec), &netRevsec, sizeof(netRevsec));

    // 处理 rankId
    for (const auto& id : config.rankId) {
        uint32_t netRankId = htonl(id.rankID);
        tlvData.resize(tlvData.size() + sizeof(netRankId));
        (void)memcpy_s(tlvData.data() + tlvData.size() - sizeof(netRankId), sizeof(netRankId), &netRankId, sizeof(netRankId));
    }
    return tlvData;
}


// 发送Rank表信息
HcclResult hcclNslbDp::SendRankTableRootRank(NslbDpRootRank &tab_f)
{
    if (getHccpInitFlag() == false) {
        return HCCL_SUCCESS;
    }
    std::vector<uint8_t> tlvData = serializeTLV_TableRootRank(tab_f);

    u32 datlen = tlvData.size();
    HCCL_INFO("HCCL SendRankTableRootRank tlvData.len:[%u] success.", datlen);
	
    nslb_msg sendMsg;
    nslb_msg recvMsg;
    sendMsg.type = NSLBDP_TYPE_TBL_ROOT_RANK;
    sendMsg.length = datlen;
    sendMsg.data.assign(tlvData.begin(), tlvData.end());
    s32 ret = H2DTlvRequest(nslbdp_handle_, reinterpret_cast<tlv_msg*>(&sendMsg), reinterpret_cast<tlv_msg*>(&recvMsg));

    HCCL_INFO("!!!!!!!!!!hccl send Table 6 to hccp. ret(%d)!!!!!!!!!!\n", ret);
    return HCCL_SUCCESS;
}


// 遍历表6 执行send流程
HcclResult hcclNslbDp::SendRootRankTable()
{
    HCCL_INFO("ndlbdp-rootRank entry SendRootRankTable.");
    NslbDpRootRank tab_f = {};
    tab_f.taskId = hcclNslbDpRootRankVal_.taskId;
    s32 sRet = memcpy_s(tab_f.commDesc, sizeof(tab_f.commDesc), hcclNslbDpRootRankVal_.commDesc, sizeof(hcclNslbDpRootRankVal_.commDesc));
    if (sRet != EOK) {
        HCCL_INFO("memcpy_s commDesc info fail");
    }
    tab_f.commInitTime = hcclNslbDpRootRankVal_.commInitTime;
    tab_f.oper = hcclNslbDpRootRankVal_.oper;
    tab_f.algorithm = hcclNslbDpRootRankVal_.algorithm;
    tab_f.revfir = hcclNslbDpRootRankVal_.revfir;
    tab_f.rootRankNum = hcclNslbDpRootRankVal_.rootRankNum;
    tab_f.revsec = hcclNslbDpRootRankVal_.revsec;
    tab_f.rankId = hcclNslbDpRootRankVal_.rankId;
    
    for (const auto& rank : hcclNslbDpRootRankVal_.rankId) {
        tab_f.rankId.push_back(rank); 
    }
    
    SendRankTableRootRank(tab_f);

    return HCCL_SUCCESS;
}
}