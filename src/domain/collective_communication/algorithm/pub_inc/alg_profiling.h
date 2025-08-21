/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once 
#include <string>
#include <map>
#include <array>
#include "hccl_types.h"
#include "hccl_common.h"
#include "common.h"

namespace hccl{

typedef void (*TaskCallBack)(void *userPtr, void *param, u32 length);

struct TaskParaAiv{
    HcclCMDType cmdType;
    u32 tag;
    u64 size;
    u32 blockDim;
    u32 rankSize;
    s32 aivRdmaStep;
    void* flagMem;
    u32 rank;
    TaskParaAiv()
        : cmdType(HcclCMDType::HCCL_CMD_INVALID), tag(0), size(0), blockDim(0), rankSize(0), aivRdmaStep(0), flagMem(nullptr),
          rank(0)
    {}
    TaskParaAiv(
        HcclCMDType cmdType, u32 tag, u64 size, u32 blockDim, u32 rankSize, s32 aivRdmaStep, void *flagMem, u32 rank)
        : cmdType(cmdType), tag(tag), size(size), blockDim(blockDim), rankSize(rankSize), aivRdmaStep(aivRdmaStep),
          flagMem(flagMem), rank(rank)
    {}
};

struct TaskParaGeneral{
    void* stream{nullptr};
    bool isMainStream{false};
    u64 beginTime{0};
    struct TaskParaAiv aiv;

    TaskParaGeneral() : stream(nullptr), isMainStream(false), beginTime(0)
    {}

    ~TaskParaGeneral() {}
};

class AlgWrap{
public:
    static AlgWrap& GetInstance();
    HcclResult RegisterAlgCallBack(const std::string& comm, void* userPtr, TaskCallBack callback, s32 deviceLogicID);
    void UnregisterAlgCallBack(const std::string& comm);
    HcclResult TaskAivProfiler(const std::string& comm, struct TaskParaGeneral& taskParaGeneral);

private:
    AlgWrap(){ initialized_ = true; };
    ~AlgWrap(){ initialized_ = false; };
    
    // initialized 是否已初始化，避免析构后访问类成员
    bool initialized_ = false;
    std::mutex aivCallBackMutex_;
    std::map<std::string, std::array<TaskCallBack, MAX_MODULE_DEVICE_NUM>> aivCallBackMap_;
    std::map<std::string, std::array<void*, MAX_MODULE_DEVICE_NUM>> aivCallBackUserPtrMap_;
};

}