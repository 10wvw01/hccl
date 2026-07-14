/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include <vector>
#include <set>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "alg_type.h"
#include "channel_request.h"
#include "topo.h"
#include "topo_host.h"

namespace ops_hccl {

HcclResult GetProtocolByEngine(HcclAlgEngineType engineType, std::vector<CommProtocol> &protocols)
{
    protocols.clear();
    switch (engineType) {
        case HcclAlgEngineType::AICPU:
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_CTP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_TP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_PCIE);
            break;
        case HcclAlgEngineType::CCU_MS:
        case HcclAlgEngineType::CCU_SCHED:
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_CTP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_TP);
            break;
        case HcclAlgEngineType::AIV:
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UB_MEM);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_PCIE);
            break;
        default:
            HCCL_WARNING("[GetProtocolByEngine] Unknown engine[%d], set protocol to RESERVED",
                         static_cast<int>(engineType));
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CreateChannelFromLink(HcclComm comm, u32 myRank, u32 rank, uint32_t netLayer, u32 idx,
    const CommLink& link, const std::string& funcName, std::vector<HcclChannelDesc>& channels)
{
    (void) comm;
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = rank;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    HCCL_INFO("[CreateChannelFromLink]%s Add channel request between %zu and %zu, netLayerIdx %u, "
              "linkListIdx %u, protocol %zu",
              funcName.c_str(), myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
    channels.push_back(channelDesc);
    return HCCL_SUCCESS;
}

HcclResult ProcessLinkForProtocol(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound, const std::string& funcName)
{
    protocolFound = false;
    for (auto expectedProtocol : expectedProtocols) {
        for (u32 idx = 0; idx < linkList.size(); idx++) {
            if (linkList[idx].linkAttr.linkProtocol == expectedProtocol) {
                CHK_RET(CreateChannelFromLink(comm, myRank, remoteRank, netLayer, idx, linkList[idx],
                    funcName, channels));
                protocolFound = true;
            }
        }
        if (protocolFound) {
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProcessLinkForProtocolNhr(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound)
{
    return ProcessLinkForProtocol(comm, expectedProtocols, linkList, myRank, remoteRank,
        netLayer, channels, protocolFound, std::string("[CalcLevel1ChannelRequestNhr]"));
}

HcclResult CalcChannelRequestMesh1D(HcclComm comm, HcclAlgEngineType engineType, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
                HcclResult::HCCL_E_PARA);
    u32 myRank = topoInfo->userRank;
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(engineType, expectedProtocols));
    for (u32 rank: subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        size_t channelCountBefore = channels.size();
        uint32_t *netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
        for (auto netLayer : netLayersVector) {
            CommLink *linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
            if (listSize == 0) {
                continue;
            }
            std::vector<CommLink> links(linkList, linkList + listSize);
            bool protocolFound = false;
            CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links, myRank, rank, netLayer, channels, protocolFound,
                std::string("[CalcChannelRequestMesh1D]")));
            if (channels.size() > channelCountBefore) {
                break;
            }
        }
        CHK_PRT_RET(channels.size() == channelCountBefore,
            HCCL_ERROR("[CalcChannelRequestMesh1D] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                myRank, rank), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhr(HcclComm comm, HcclAlgEngineType engineType, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
    channels.clear();
    std::set<u32> connectRanks;
    u32 myRank = topoInfo->userRank;
    auto it = std::find(subcommInfo[0].begin(), subcommInfo[0].end(), myRank);
    CHK_PRT_RET((it == subcommInfo[0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", myRank),
                HcclResult::HCCL_E_PARA);

    u32 localRank = std::distance(subcommInfo[0].begin(), it);
    u32 localRankSize = subcommInfo[0].size();
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));

    // 根据engine获取期望的协议类型列表
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(engineType, expectedProtocols));

    for (u32 rankIdx: connectRanks) {
        size_t channelCountBefore = channels.size();
        uint32_t *netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

        for (auto netLayer : netLayersVector) {
            if (netLayerNum > 1 && netLayer == 0) {
                continue; // 跨框场景，nhr算法只取layer1的的链路
            }
            CommLink *linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, subcommInfo[0][rankIdx], &linkList, &listSize));

            if (listSize == 0) {
                continue;
            }

            std::vector<CommLink> links(linkList, linkList + listSize);
            bool protocolFound = false;
            CHK_RET(ProcessLinkForProtocolNhr(comm, expectedProtocols, links, myRank, subcommInfo[0][rankIdx], netLayer, channels, protocolFound));

            if (channels.size() > channelCountBefore) {
                break;
            }
        }

        CHK_PRT_RET(channels.size() == channelCountBefore,
            HCCL_ERROR("[CalcChannelRequestNhr] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                myRank, subcommInfo[0][rankIdx]), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

}
