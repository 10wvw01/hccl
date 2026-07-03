#ifndef OMNI_PARSER_H
#define OMNI_PARSER_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "hccl/base.h"
#include "hccl_common.h"
#include "log.h"
#include "binary_stream.h"

namespace ops_hccl {
namespace omni {

// ============== Bit Field Constants ==============

// Instruction header field widths and offsets
namespace field {
    // Common
    constexpr uint32_t OPCODE_WIDTH = 5;
    constexpr uint32_t OPCODE_OFFSET = 0;

    // ResRequest fields
    constexpr uint32_t SLAVE_THREAD_NUM_WIDTH = 5;
    constexpr uint32_t SLAVE_THREAD_NUM_OFFSET = 5;
    constexpr uint32_t NOTIFY_NUM_MAIN_WIDTH = 5;
    constexpr uint32_t NOTIFY_NUM_MAIN_OFFSET = 10;
    constexpr uint32_t NOTIFY_NUM_PER_THREAD_WIDTH = 5;
    constexpr uint32_t NOTIFY_NUM_PER_THREAD_OFFSET = 15;
    constexpr uint32_t NET_LAYER_NUM_WIDTH = 2;
    constexpr uint32_t NET_LAYER_NUM_OFFSET = 20;
    constexpr uint32_t CHAN_COUNT_WIDTH = 8;
    constexpr uint32_t CHAN_COUNT_OFFSET = 22;

    // Sync fields
    constexpr uint32_t MAIN_THREAD_IDX_WIDTH = 5;
    constexpr uint32_t MAIN_THREAD_IDX_OFFSET = 5;
    constexpr uint32_t SUB_THREAD_NUM_WIDTH = 5;
    constexpr uint32_t SUB_THREAD_NUM_OFFSET = 10;

    // Control fields
    constexpr uint32_t NET_LAYER_ID_WIDTH = 2;
    constexpr uint32_t NET_LAYER_ID_OFFSET = 5;
    constexpr uint32_t LINK_PROTO_WIDTH = 3;
    constexpr uint32_t LINK_PROTO_OFFSET = 7;
    constexpr uint32_t SLICE_NUM_WIDTH = 10;
    constexpr uint32_t SLICE_NUM_OFFSET = 10;
    constexpr uint32_t SRC_SLICE_NUM_WIDTH = 4;
    constexpr uint32_t SRC_SLICE_NUM_OFFSET = 20;
    constexpr uint32_t DST_SLICE_NUM_WIDTH = 4;
    constexpr uint32_t DST_SLICE_NUM_OFFSET = 24;
    constexpr uint32_t NOTIFY_FLAG_WIDTH = 1;
    constexpr uint32_t NOTIFY_FLAG_OFFSET = 28;
    constexpr uint32_t NOTIFY_THREAD_WIDTH = 4;
    constexpr uint32_t NOTIFY_THREAD_OFFSET = 29;
    constexpr uint32_t WAIT_FLAG_WIDTH = 1;
    constexpr uint32_t WAIT_FLAG_OFFSET = 33;
    constexpr uint32_t WAIT_THREAD_WIDTH = 4;
    constexpr uint32_t WAIT_THREAD_OFFSET = 34;
    constexpr uint32_t THREAD_IDX_WIDTH = 5;
    constexpr uint32_t THREAD_IDX_OFFSET = 38;
    constexpr uint32_t REDUCE_TYPE_WIDTH = 2;
    constexpr uint32_t REDUCE_TYPE_OFFSET = 43;
    constexpr uint32_t INPUT_DATA_TYPE_WIDTH = 4;
    constexpr uint32_t INPUT_DATA_TYPE_OFFSET = 45;
    constexpr uint32_t OUTPUT_DATA_TYPE_WIDTH = 4;
    constexpr uint32_t OUTPUT_DATA_TYPE_OFFSET = 49;
    constexpr uint32_t INSTRUCTION_ID_WIDTH = 10;
    constexpr uint32_t INSTRUCTION_ID_OFFSET = 53;

    // Channel fields
    constexpr uint32_t CHAN_NET_LAYER_ID_WIDTH = 5;
    constexpr uint32_t CHAN_NET_LAYER_ID_OFFSET = 0;
    constexpr uint32_t CHAN_LOCAL_RANK_WIDTH = 10;
    constexpr uint32_t CHAN_LOCAL_RANK_OFFSET = 5;
    constexpr uint32_t CHAN_REMOTE_RANK_WIDTH = 10;
    constexpr uint32_t CHAN_REMOTE_RANK_OFFSET = 15;
    constexpr uint32_t CHAN_LINK_PROTO_WIDTH = 3;
    constexpr uint32_t CHAN_LINK_PROTO_OFFSET = 25;

    // Slice fields
    constexpr uint32_t SLICE_BUFFER_TYPE_WIDTH = 2;
    constexpr uint32_t SLICE_BUFFER_TYPE_OFFSET = 0;
    constexpr uint32_t SLICE_IDX_WIDTH = 10;
    constexpr uint32_t SLICE_IDX_OFFSET = 2;
    constexpr uint32_t SLICE_RANK_ID_WIDTH = 10;
    constexpr uint32_t SLICE_RANK_ID_OFFSET = 12;
    constexpr uint32_t SLICE_RECV_RANK_ID_WIDTH = 10;
    constexpr uint32_t SLICE_RECV_RANK_ID_OFFSET = 22;
    constexpr uint32_t SLICE_CNT_WIDTH = 10;
    constexpr uint32_t SLICE_CNT_OFFSET = 32;
    constexpr uint32_t SLICE_GAP_WIDTH = 10;
    constexpr uint32_t SLICE_GAP_OFFSET = 42;
} // namespace field

// Binary sizes
constexpr size_t INSTRUCTION_HEADER_SIZE = 8;
constexpr size_t CHANNEL_ENTRY_SIZE = 4;
constexpr size_t SLICE_ENTRY_SIZE = 8;
constexpr size_t SUBTHREAD_ENTRY_SIZE = 1;

// ============== Enums (compatible with old_parser) ==============

enum OpType {
    OP_RES_REQUEST = 0,
    OP_PRE_SYNC_INTER_THREADS = 1,
    OP_POST_SYNC_INTER_THREADS = 2,
    OP_LOCAL_COPY = 3,
    OP_LOCAL_REDUCE = 4,
    OP_SEND_RECV_WRITE = 5,
    OP_SEND_WRITE = 6,
    OP_RECV_WRITE = 7,
    OP_SEND_RECV_WRITE_REDUCE = 8,
    OP_SEND_WRITE_REDUCE = 9,
    OP_RECV_WRITE_REDUCE = 10,
    OP_SEND_RECV_READ = 11,
    OP_SEND_READ = 12,
    OP_RECV_READ = 13,
    OP_SEND_RECV_READ_REDUCE = 14,
    OP_SEND_READ_REDUCE = 15,
    OP_RECV_READ_REDUCE = 16,
    OP_SEND_RECV_WRITE_DPU = 17,
    OP_SEND_WRITE_DPU = 18,
    OP_RECV_WRITE_DPU = 19,
    OP_GROUP_BROAD_CAST = 20,
    OP_GROUP_REDUCE = 21,
    OP_WAIT_EVENT = 22
};

enum BufferTypeTmp {
    HCCL_BUFFER = 0,
    INPUT = 1,
    OUTPUT = 2,
    DEFAULT
};

enum class InstructionType : uint8_t {
    RES_REQUEST = 0,
    SYNC = 1,
    CONTROL = 2
};

// ============== Inline Helpers ==============

inline InstructionType GetInstructionType(OpType opType) {
    if (opType == OP_RES_REQUEST) {
        return InstructionType::RES_REQUEST;
    } else if (opType == OP_PRE_SYNC_INTER_THREADS || opType == OP_POST_SYNC_INTER_THREADS) {
        return InstructionType::SYNC;
    } else {
        return InstructionType::CONTROL;
    }
}

// ============== Enum to String Converters ==============

std::string OpTypeToString(OpType op);
std::string HcclDataTypeToString(HcclDataType dt);
std::string HcclReduceOpToString(HcclReduceOp op);
std::string BufferTypeTmpToString(BufferTypeTmp bt);
std::string CommProtocolToString(CommProtocol cp);

// ============== Data Structures (compatible with old_parser) ==============

struct OmniSliceInfo {
    BufferTypeTmp sliceType;
    uint64_t sliceIdx;
    uint64_t remoteRank;
    uint64_t remoteRecvRank;
    uint64_t cnt;
    uint64_t gap;

    OmniSliceInfo() : sliceType(DEFAULT), sliceIdx(0), remoteRank(0), remoteRecvRank(0), cnt(1), gap(1) {}
    std::string toString() const;

    void Serialize(BinaryStream& binaryStream) const;
    void DeSerialize(BinaryStream& binaryStream);
};

struct OmniChannelInfo {
    CommProtocol channelProtocol;
    uint64_t remoteRank;
    uint64_t netlayerId;

    OmniChannelInfo() : channelProtocol(COMM_PROTOCOL_RESERVED), remoteRank(0), netlayerId(0) {}
    std::string toString() const;

    void Serialize(BinaryStream& binaryStream) const;
    void DeSerialize(BinaryStream& binaryStream);
};

struct OmniSendRecvInfo {
    HcclDataType inputDataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceType;
    uint64_t sliceNum;
    uint64_t threadIdx;
    uint64_t netlayerId;
    std::vector<OmniSliceInfo> srcSliceInfo;
    std::vector<OmniSliceInfo> dstSliceInfo;

    OmniSendRecvInfo()
        : inputDataType(HCCL_DATA_TYPE_INT8), outputDataType(HCCL_DATA_TYPE_INT8),
          reduceType(HCCL_REDUCE_SUM), sliceNum(0), threadIdx(0), netlayerId(0) {}
    std::string toString() const;

    void Serialize(BinaryStream& binaryStream) const;
    void DeSerialize(BinaryStream& binaryStream);
};

struct OmniSyncInfo {
    uint64_t mainThreadIdx;
    uint64_t subThreadNum;
    std::vector<uint8_t> subThreadIds;

    OmniSyncInfo() : mainThreadIdx(0), subThreadNum(0) {}
    std::string toString() const;

    void Serialize(BinaryStream& binaryStream) const;
    void DeSerialize(BinaryStream& binaryStream);
};

struct OmniNormalInstruction {
    OpType opType;
    OmniSendRecvInfo sendRecvInfo;  // Control instruction data
    OmniSyncInfo syncInfo;          // Sync instruction data

    OmniNormalInstruction() : opType(OP_LOCAL_COPY) {}
    std::string toString() const;

    void Serialize(BinaryStream& binaryStream) const;
    void DeSerialize(BinaryStream& binaryStream);
};

struct ResInfo {
    uint32_t slaveThreadNum;
    uint32_t notifyNumOnMainThread;
    uint32_t notifyNumPerThread;
    uint32_t netLayerNum;
    std::vector<std::map<uint32_t, OmniChannelInfo>> mapchannelInfo;
    std::vector<OmniChannelInfo> vecChannelInfo;

    ResInfo() : slaveThreadNum(0), notifyNumOnMainThread(0),
                notifyNumPerThread(0), netLayerNum(0) {}
    std::string toString() const;

    void Serialize(BinaryStream& binaryStream) const;
    void DeSerialize(BinaryStream& binaryStream);
};

struct XmlInfo {
    ResInfo resInfo;
    std::vector<OmniNormalInstruction> vecNormalInstruction;
    uint32_t syncNum; // ccu 使用
    std::string toString() const;

    void Serialize(BinaryStream& binaryStream) const;
    void DeSerialize(BinaryStream& binaryStream);
};

// ============== Parser ==============

class BinaryParser {
public:
    explicit BinaryParser(bool showHex = false) : showHex_(showHex), rankId_(0), fileSize_(0) {}

    // Step 1: Set file metadata (filename, rankId, fileSize)
    // Returns HCCL_SUCCESS if file exists and is accessible
    HcclResult SetFile(const std::string& filepath, uint32_t rankId = 0);

    // Step 2: Parse file content into XmlInfo
    // Caller owns the XmlInfo, parser does not retain it
    HcclResult Parse(XmlInfo& xmlInfo);

    // Getters for file metadata
    const std::string& GetFilename() const { return filename_; }
    uint32_t GetRankId() const { return rankId_; }
    size_t GetFileSize() const { return fileSize_; }

private:
    bool showHex_;
    std::string filename_;
    std::string filepath_;
    uint32_t rankId_;
    size_t fileSize_;

    // Bit extraction helpers
    static uint32_t ExtractBits(uint64_t value, uint32_t width, uint32_t offset);
    static uint32_t ExtractBits32(uint32_t value, uint32_t width, uint32_t offset);

    // Header parsing helpers
    void ParseResRequest(XmlInfo& xmlInfo, const std::vector<uint8_t>& data,
                         size_t& offset, uint64_t headerValue, uint32_t rankId);
    void ParseSyncInstruction(XmlInfo& xmlInfo, const std::vector<uint8_t>& data,
                              size_t& offset, uint64_t headerValue, OpType opType);
    void ParseControlInstruction(XmlInfo& xmlInfo, const std::vector<uint8_t>& data,
                                 size_t& offset, uint64_t headerValue, OpType opType);

    // Slice/Channel parsing
    OmniSliceInfo ParseSlice(uint64_t sliceValue);
    OmniChannelInfo ParseChannel(uint32_t channelValue);

    // Debug output
    void PrintHexDump(const std::vector<uint8_t>& data, const std::string& filepath);
};

} // namespace omni
} // namespace ops_hccl

#endif // OMNI_PARSER_H
