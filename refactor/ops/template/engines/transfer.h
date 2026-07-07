// ============================================================
// file: src/transfer/transfer_context.h
//
// 统一数据传输上下文定义。
//
// 设计原则:
//   P1 — 统一接口: 所有字段引擎无关 / 算子无关 / 拓扑无关。
//   P4 — 调用明确: engine 由调用者指定,接口内部不做引擎选择。
//   P5 — Rank 上移: slices 的构建反映调用者已完成 rank 角色判断。
//
// Channel 信息由调用者维护管理,作为独立入参传入 Send(),
// 不包含在 TransferContext 中。
//
// 本文件不依赖任何引擎专用头文件, 仅使用标准 C++ 类型。
// ============================================================

#ifndef HCCL_TRANSFER_H
#define HCCL_TRANSFER_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <sstream> 
#include  <string>
#include "template_utils.h"

// ───────────── 数据传输方向 (纯数据流方向) ─────────────
// 仅描述 "本地 → 远端" 还是 "远端 → 本地",
// 不携带算子语义, 不携带拓扑语义。
enum class TransferDirection : uint8_t {
    WRITE = 0,   // 本地 → 远端 (Push)
    READ  = 1,   // 远端 → 本地 (Pull)
};


// ───────────── 融合操作类型 ─────────────
// 标识数据搬运过程中是否融合 Reduce。
// NONE   : 纯数据搬运。
// REDUCE : 传输 + Reduce 融合 。
enum class HcclReduceOp : uint8_t {
    NONE   = 0,
    REDUCE_ADD = 1,
    REDUCE_MAX = 2,
    REDUCE_MIN = 3,
    REDUCE_RSV
};


// ───────────── 数据类型 / ReduceOp 透传 ─────────────
// 为避免直接依赖 hccl 公共头, 这里用强类型别名。
// 实际接入时可在 transfer_context.cc 或集成层做类型映射。
using HcclDataType = uint32_t;
using HcclResult   = int32_t;

// 标准 HcclResult 取值 (与主仓 hccl_common.h 对齐)
constexpr HcclResult HCCL_SUCCESS            = 0;
constexpr HcclResult HCCL_ERR_INVALID_PARAM  = -1;
constexpr HcclResult HCCL_ERR_INVALID_ENGINE = -2;
constexpr HcclResult HCCL_ERR_INTERNAL       = -3;
constexpr HcclResult HCCL_ERR_UNSUPPORTED    = -4;

// ───────────── 单个数据切片 ─────────────
struct DataSlice {
    void* addr_ = nullptr;
    u64 offset_{0}; // Slice相对于input/output的偏移字节数
    u64 size_{0};    // Slice的数据大小，单位：字节
    u64 count_{0};   // 数据元素个数

    DataSlice(void* addr, u64 offset, u64 size, u64 count)
    : addr_(addr), offset_(offset), size_(size), count_(count)
    {
    }

    DataSlice(void* addr, u64 offset, u64 size)
    : addr_(addr), offset_(offset), size_(size)
    {
        count_ = 0;
    }

    std::string Describe() const {
        std::ostringstream oss;
        oss << "DataSlice: addr=" << addr_ // 指针地址会自动格式化为十六进制
            << ", offset=" << offset_
            << ", size=" << size_
            << ", count=" << count_;
        return oss.str();
    }
};

struct SlicesList {
    std::vector<DataSlice> srcSlices_;
    std::vector<DataSlice> dstSlices_;

    SlicesList(const std::vector<DataSlice> &srcSlices, const std::vector<DataSlice> &dstSlices)
        : srcSlices_(srcSlices), dstSlices_(dstSlices)
    {
    }
};
struct TxRxSlicesList {
    SlicesList txSlicesList_;
    SlicesList rxSlicesList_;
    U32 srcRankId_;
    U32 dstRankId_;
    TxRxSlicesList(const SlicesList &txSlicesList, const SlicesList &rxSlicesList,  
                    U32 &srcRankId, U32 &dstRankId)
        : txSlicesList_(txSlicesList), rxSlicesList_(rxSlicesList), srcRankId_(srcRankId), dstRankId_(dstRankId) 
    {
    }
};

// ★★★ 统一数据传输上下文 ★★★
// 调用者填充, 传给 HcclDataTransfer::Send()。
struct TransferContext {
    // ──── 数据传输描述 ────
    bool enableRemoteMemAccess = true;
    BufferType buffType = OUTPUT;
    TxRxSlicesList txRxSlicesList;
    TemplateResource templateRes;

    // ──── Reduce 参数 (reduceOp != NONE 时需要) ────
    HcclDataType dataType = 0;
    HcclReduceOp reduceOp = HcclReduceOp::NONE;

    // ──── 扩展 (引擎专用参数透传, 不建议常规使用) ────
    void* reserved = nullptr;
};


#endif  // HCCL_TRANSFER_H
