/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_
#define HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_

#include <array>
#include <set>
#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;
using RankGroup = std::vector<RankId>;

struct Mesh2DieCacheCtx {
    uint32_t dieNum;
    std::vector<RankId> rankGroup[2];
    std::set<RankId> closPeers;
    uint32_t closBwCoeff[2];
    uint32_t totalBwCoeff;
    uint32_t closMinorDieId;
    uint32_t closMajorDieId;

    std::vector<char> Serialize() const
    {
        std::vector<char> buf;
        auto append = [&buf](const void *data, size_t len) {
            const char *p = static_cast<const char *>(data);
            buf.insert(buf.end(), p, p + len);
        };
        auto appendVec = [&buf](const std::vector<RankId> &v) {
            uint32_t sz = static_cast<uint32_t>(v.size());
            buf.insert(buf.end(), reinterpret_cast<const char *>(&sz), reinterpret_cast<const char *>(&sz) + sizeof(uint32_t));
            buf.insert(buf.end(), reinterpret_cast<const char *>(v.data()), reinterpret_cast<const char *>(v.data()) + sz * sizeof(RankId));
        };
        auto appendSet = [&buf](const std::set<RankId> &s) {
            uint32_t sz = static_cast<uint32_t>(s.size());
            buf.insert(buf.end(), reinterpret_cast<const char *>(&sz), reinterpret_cast<const char *>(&sz) + sizeof(uint32_t));
            for (auto &v : s) {
                buf.insert(buf.end(), reinterpret_cast<const char *>(&v), reinterpret_cast<const char *>(&v) + sizeof(RankId));
            }
        };
        append(&dieNum, sizeof(uint32_t));
        appendVec(rankGroup[0]);
        appendVec(rankGroup[1]);
        appendSet(closPeers);
        append(closBwCoeff, sizeof(uint32_t) * 2);
        append(&totalBwCoeff, sizeof(uint32_t));
        append(&closMinorDieId, sizeof(uint32_t));
        append(&closMajorDieId, sizeof(uint32_t));
        return buf;
    }

    void Deserialize(const char *buf, size_t len)
    {
        size_t off = 0;
        auto read = [&off, buf, len](void *dst, size_t n) {
            memcpy(dst, buf + off, n);
            off += n;
        };
        auto readVec = [&off, buf, len](std::vector<RankId> &v) {
            uint32_t sz;
            memcpy(&sz, buf + off, sizeof(uint32_t));
            off += sizeof(uint32_t);
            v.resize(sz);
            memcpy(v.data(), buf + off, sz * sizeof(RankId));
            off += sz * sizeof(RankId);
        };
        auto readSet = [&off, buf, len](std::set<RankId> &s) {
            uint32_t sz;
            memcpy(&sz, buf + off, sizeof(uint32_t));
            off += sizeof(uint32_t);
            s.clear();
            for (uint32_t i = 0; i < sz; i++) {
                RankId v;
                memcpy(&v, buf + off, sizeof(RankId));
                off += sizeof(RankId);
                s.insert(v);
            }
        };
        read(&dieNum, sizeof(uint32_t));
        readVec(rankGroup[0]);
        readVec(rankGroup[1]);
        readSet(closPeers);
        read(closBwCoeff, sizeof(uint32_t) * 2);
        read(&totalBwCoeff, sizeof(uint32_t));
        read(&closMinorDieId, sizeof(uint32_t));
        read(&closMajorDieId, sizeof(uint32_t));
    }
};

class CcuTempAlltoAllVMesh1D2Die : public CcuAlgTemplateBase {
public:
    CcuTempAlltoAllVMesh1D2Die() = default;
    explicit CcuTempAlltoAllVMesh1D2Die(const OpParam &param, RankId rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAlltoAllVMesh1D2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of alltoallv ccu mesh 1D 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
        TemplateResource& templateResource) override;

    void SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo);

private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs,
                                std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);
    void FillRankGroupTaskArgs(uint32_t dieId, const Mesh2DieCacheCtx &cacheCtx,
        const LoopGroupConfig &config, std::vector<uint64_t> &taskArgs);
    HcclResult SaveCacheCtx(HcclComm comm, const OpParam &param);
    HcclResult LoadCacheCtx(const OpParam &param, Mesh2DieCacheCtx &cacheCtx);

    const uint32_t DIE_NUM = 2;

    std::vector<std::vector<HcclChannelDesc>> channels_{2};
    std::array<RankGroup, 2> rankGroup_;
    std::map<uint32_t, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
    std::set<RankId> closPeers_;
    uint32_t closMinorDieId_ = 0;
    uint32_t closMajorDieId_ = 1;
    uint32_t closBwCoeff_[2] = {0, 0};
    uint32_t totalBwCoeff_ = 0;

    A2ASendRecvInfo localSendRecvInfo_;
};

} // namespace ops_hccl
#endif // HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_
