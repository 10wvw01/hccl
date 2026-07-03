#include "omni_parser.h"
#include <fstream>

namespace ops_hccl {
namespace omni {

// ============== Enum to String Converters ==============

std::string OpTypeToString(OpType op) {
    switch (op) {
        case OP_RES_REQUEST: return "RES_REQUEST";
        case OP_PRE_SYNC_INTER_THREADS: return "PRE_SYNC_INTER_THREADS";
        case OP_POST_SYNC_INTER_THREADS: return "POST_SYNC_INTER_THREADS";
        case OP_LOCAL_COPY: return "LOCAL_COPY";
        case OP_LOCAL_REDUCE: return "LOCAL_REDUCE";
        case OP_SEND_RECV_WRITE: return "SEND_RECV_WRITE";
        case OP_SEND_WRITE: return "SEND_WRITE";
        case OP_RECV_WRITE: return "RECV_WRITE";
        case OP_SEND_RECV_WRITE_REDUCE: return "SEND_RECV_WRITE_REDUCE";
        case OP_SEND_WRITE_REDUCE: return "SEND_WRITE_REDUCE";
        case OP_RECV_WRITE_REDUCE: return "RECV_WRITE_REDUCE";
        case OP_SEND_RECV_READ: return "SEND_RECV_READ";
        case OP_SEND_READ: return "SEND_READ";
        case OP_RECV_READ: return "RECV_READ";
        case OP_SEND_RECV_READ_REDUCE: return "SEND_RECV_READ_REDUCE";
        case OP_SEND_READ_REDUCE: return "SEND_READ_REDUCE";
        case OP_RECV_READ_REDUCE: return "RECV_READ_REDUCE";
        case OP_SEND_RECV_WRITE_DPU: return "SEND_RECV_WRITE_DPU";
        case OP_SEND_WRITE_DPU: return "SEND_WRITE_DPU";
        case OP_RECV_WRITE_DPU: return "RECV_WRITE_DPU";
        case OP_GROUP_BROAD_CAST: return "GROUP_BROAD_CAST";
        case OP_GROUP_REDUCE: return "GROUP_REDUCE";
        case OP_WAIT_EVENT: return "WAIT_EVENT";
        default: return "UNKNOWN(" + std::to_string(static_cast<int>(op)) + ")";
    }
}

std::string HcclDataTypeToString(HcclDataType dt) {
    switch (dt) {
        case HCCL_DATA_TYPE_INT8: return "INT8";
        case HCCL_DATA_TYPE_INT16: return "INT16";
        case HCCL_DATA_TYPE_INT32: return "INT32";
        case HCCL_DATA_TYPE_FP16: return "FP16";
        case HCCL_DATA_TYPE_FP32: return "FP32";
        case HCCL_DATA_TYPE_INT64: return "INT64";
        case HCCL_DATA_TYPE_UINT64: return "UINT64";
        case HCCL_DATA_TYPE_UINT8: return "UINT8";
        case HCCL_DATA_TYPE_UINT16: return "UINT16";
        case HCCL_DATA_TYPE_UINT32: return "UINT32";
        case HCCL_DATA_TYPE_FP64: return "FP64";
        case HCCL_DATA_TYPE_BFP16: return "BFP16";
        case HCCL_DATA_TYPE_INT128: return "INT128";
        case HCCL_DATA_TYPE_HIF8: return "HIF8";
        case HCCL_DATA_TYPE_FP8E4M3: return "FP8E4M3";
        case HCCL_DATA_TYPE_FP8E5M2: return "FP8E5M2";
        case HCCL_DATA_TYPE_RESERVED: return "RESERVED";
        default: return "UNKNOWN(" + std::to_string(static_cast<int>(dt)) + ")";
    }
}

std::string HcclReduceOpToString(HcclReduceOp op) {
    switch (op) {
        case HCCL_REDUCE_SUM: return "SUM";
        case HCCL_REDUCE_PROD: return "PROD";
        case HCCL_REDUCE_MAX: return "MAX";
        case HCCL_REDUCE_MIN: return "MIN";
        case HCCL_REDUCE_RESERVED: return "RESERVED";
        default: return "UNKNOWN(" + std::to_string(static_cast<int>(op)) + ")";
    }
}

std::string BufferTypeTmpToString(BufferTypeTmp bt) {
    switch (bt) {
        case HCCL_BUFFER: return "HCCL_BUFFER";
        case INPUT: return "INPUT";
        case OUTPUT: return "OUTPUT";
        case DEFAULT: return "DEFAULT";
        default: return "UNKNOWN(" + std::to_string(static_cast<int>(bt)) + ")";
    }
}

std::string CommProtocolToString(CommProtocol cp) {
    switch (cp) {
        case COMM_PROTOCOL_HCCS: return "HCCS";
        case COMM_PROTOCOL_ROCE: return "ROCE";
        case COMM_PROTOCOL_PCIE: return "PCIE";
        case COMM_PROTOCOL_SIO: return "SIO";
        case COMM_PROTOCOL_RESERVED: return "RESERVED";
        default: return "UNKNOWN(" + std::to_string(static_cast<int>(cp)) + ")";
    }
}

// ============== Data Structure toString() ==============

std::string OmniSliceInfo::toString() const {
    std::ostringstream oss;
    oss << "OmniSliceInfo(sliceType=" << BufferTypeTmpToString(sliceType)
        << ", sliceIdx=" << sliceIdx
        << ", remoteRank=" << remoteRank
        << ", remoteRecvRank=" << remoteRecvRank
        << ", cnt=" << cnt
        << ", gap=" << gap << ")";
    return oss.str();
}

std::string OmniChannelInfo::toString() const {
    std::ostringstream oss;
    oss << "OmniChannelInfo(channelProtocol=" << CommProtocolToString(channelProtocol)
        << ", remoteRank=" << remoteRank
        << ", netlayerId=" << netlayerId << ")";
    return oss.str();
}

std::string OmniSendRecvInfo::toString() const {
    std::ostringstream oss;
    oss << "OmniSendRecvInfo:\n";
    oss << "  inputDataType: " << HcclDataTypeToString(inputDataType) << "\n";
    oss << "  outputDataType: " << HcclDataTypeToString(outputDataType) << "\n";
    oss << "  reduceType: " << HcclReduceOpToString(reduceType) << "\n";
    oss << "  sliceNum: " << sliceNum << "\n";
    oss << "  threadIdx: " << threadIdx << "\n";
    oss << "  netlayerId: " << netlayerId << "\n";

    if (!srcSliceInfo.empty()) {
        oss << "  srcSliceInfo (" << srcSliceInfo.size() << "):\n";
        for (size_t i = 0; i < srcSliceInfo.size(); ++i) {
            oss << "    [" << i << "] " << srcSliceInfo[i].toString() << "\n";
        }
    }
    if (!dstSliceInfo.empty()) {
        oss << "  dstSliceInfo (" << dstSliceInfo.size() << "):\n";
        for (size_t i = 0; i < dstSliceInfo.size(); ++i) {
            oss << "    [" << i << "] " << dstSliceInfo[i].toString() << "\n";
        }
    }
    return oss.str();
}

std::string OmniSyncInfo::toString() const {
    std::ostringstream oss;
    oss << "OmniSyncInfo:\n";
    oss << "  mainThreadIdx: " << mainThreadIdx << "\n";
    oss << "  subThreadNum: " << subThreadNum << "\n";
    if (!subThreadIds.empty()) {
        oss << "  subThreadIds (" << subThreadIds.size() << "): [";
        for (size_t i = 0; i < subThreadIds.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << static_cast<int>(subThreadIds[i]);
        }
        oss << "]\n";
    }
    return oss.str();
}

std::string OmniNormalInstruction::toString() const {
    std::ostringstream oss;
    oss << "OmniNormalInstruction:\n";
    oss << "  opType: " << OpTypeToString(opType) << "\n";
    if (GetInstructionType(opType) == InstructionType::SYNC) {
        oss << syncInfo.toString();
    } else {
        oss << sendRecvInfo.toString();
    }
    return oss.str();
}

std::string ResInfo::toString() const {
    std::ostringstream oss;
    oss << "ResInfo:\n";
    oss << "  slaveThreadNum: " << slaveThreadNum << "\n";
    oss << "  notifyNumOnMainThread: " << notifyNumOnMainThread << "\n";
    oss << "  notifyNumPerThread: " << notifyNumPerThread << "\n";
    oss << "  netLayerNum: " << netLayerNum << "\n";
    if (!mapchannelInfo.empty()) {
        oss << "  mapchannelInfo (" << mapchannelInfo.size() << " netlayers):\n";
        for (size_t i = 0; i < mapchannelInfo.size(); ++i) {
            oss << "    NetLayer[" << i << "]:\n";
            for (const auto& pair : mapchannelInfo[i]) {
                oss << "      remoteRank[" << pair.first << "]: " << pair.second.toString() << "\n";
            }
        }
    }
    if (!vecChannelInfo.empty()) {
        oss << "  vecChannelInfo (" << vecChannelInfo.size() << "):\n";
        for (size_t i = 0; i < vecChannelInfo.size(); ++i) {
            oss << "    [" << i << "] " << vecChannelInfo[i].toString() << "\n";
        }
    }
    return oss.str();
}

std::string XmlInfo::toString() const {
    std::ostringstream oss;
    oss << "\n" << std::string(60, '=') << "\n";
    oss << "XmlInfo\n";
    oss << std::string(60, '=') << "\n";
    oss << resInfo.toString();
    oss << "  vecNormalInstruction (" << vecNormalInstruction.size() << " instructions):\n";
    for (size_t i = 0; i < vecNormalInstruction.size(); ++i) {
        oss << "  [" << i << "] " << vecNormalInstruction[i].toString();
    }
    return oss.str();
}

// ============== Serialization Implementation ==============

void OmniSliceInfo::Serialize(BinaryStream& binaryStream) const {
    binaryStream << static_cast<uint32_t>(sliceType);
    binaryStream << sliceIdx;
    binaryStream << remoteRank;
    binaryStream << remoteRecvRank;
    binaryStream << cnt;
    binaryStream << gap;
}

void OmniSliceInfo::DeSerialize(BinaryStream& binaryStream) {
    uint32_t typeVal;
    binaryStream >> typeVal;
    sliceType = static_cast<BufferTypeTmp>(typeVal);
    binaryStream >> sliceIdx;
    binaryStream >> remoteRank;
    binaryStream >> remoteRecvRank;
    binaryStream >> cnt;
    binaryStream >> gap;
}

void OmniChannelInfo::Serialize(BinaryStream& binaryStream) const {
    binaryStream << static_cast<uint32_t>(channelProtocol);
    binaryStream << remoteRank;
    binaryStream << netlayerId;
}

void OmniChannelInfo::DeSerialize(BinaryStream& binaryStream) {
    uint32_t protoVal;
    binaryStream >> protoVal;
    channelProtocol = static_cast<CommProtocol>(protoVal);
    binaryStream >> remoteRank;
    binaryStream >> netlayerId;
}

void OmniSendRecvInfo::Serialize(BinaryStream& binaryStream) const {
    binaryStream << static_cast<uint32_t>(inputDataType);
    binaryStream << static_cast<uint32_t>(outputDataType);
    binaryStream << static_cast<uint32_t>(reduceType);
    binaryStream << sliceNum;
    binaryStream << threadIdx;
    binaryStream << netlayerId;

    // Serialize srcSliceInfo
    binaryStream << static_cast<uint64_t>(srcSliceInfo.size());
    for (const auto& slice : srcSliceInfo) {
        slice.Serialize(binaryStream);
    }

    // Serialize dstSliceInfo
    binaryStream << static_cast<uint64_t>(dstSliceInfo.size());
    for (const auto& slice : dstSliceInfo) {
        slice.Serialize(binaryStream);
    }
}

void OmniSendRecvInfo::DeSerialize(BinaryStream& binaryStream) {
    uint32_t inputVal, outputVal, reduceVal;
    binaryStream >> inputVal;
    binaryStream >> outputVal;
    binaryStream >> reduceVal;
    inputDataType = static_cast<HcclDataType>(inputVal);
    outputDataType = static_cast<HcclDataType>(outputVal);
    reduceType = static_cast<HcclReduceOp>(reduceVal);

    binaryStream >> sliceNum;
    binaryStream >> threadIdx;
    binaryStream >> netlayerId;

    // Deserialize srcSliceInfo
    uint64_t srcSize;
    binaryStream >> srcSize;
    srcSliceInfo.resize(srcSize);
    for (auto& slice : srcSliceInfo) {
        slice.DeSerialize(binaryStream);
    }

    // Deserialize dstSliceInfo
    uint64_t dstSize;
    binaryStream >> dstSize;
    dstSliceInfo.resize(dstSize);
    for (auto& slice : dstSliceInfo) {
        slice.DeSerialize(binaryStream);
    }
}

void OmniSyncInfo::Serialize(BinaryStream& binaryStream) const {
    binaryStream << mainThreadIdx;
    binaryStream << subThreadNum;
    binaryStream << subThreadIds;
}

void OmniSyncInfo::DeSerialize(BinaryStream& binaryStream) {
    binaryStream >> mainThreadIdx;
    binaryStream >> subThreadNum;
    binaryStream >> subThreadIds;
}

void OmniNormalInstruction::Serialize(BinaryStream& binaryStream) const {
    binaryStream << static_cast<uint32_t>(opType);

    // Serialize sendRecvInfo (always serialize both for simplicity)
    sendRecvInfo.Serialize(binaryStream);

    // Serialize syncInfo
    syncInfo.Serialize(binaryStream);
}

void OmniNormalInstruction::DeSerialize(BinaryStream& binaryStream) {
    uint32_t opVal;
    binaryStream >> opVal;
    opType = static_cast<OpType>(opVal);

    // Deserialize both (one may be empty depending on instruction type)
    sendRecvInfo.DeSerialize(binaryStream);
    syncInfo.DeSerialize(binaryStream);
}

void ResInfo::Serialize(BinaryStream& binaryStream) const {
    binaryStream << slaveThreadNum;
    binaryStream << notifyNumOnMainThread;
    binaryStream << notifyNumPerThread;
    binaryStream << netLayerNum;

    // Serialize vecChannelInfo
    binaryStream << static_cast<uint64_t>(vecChannelInfo.size());
    for (const auto& channel : vecChannelInfo) {
        channel.Serialize(binaryStream);
    }
}

void ResInfo::DeSerialize(BinaryStream& binaryStream) {
    binaryStream >> slaveThreadNum;
    binaryStream >> notifyNumOnMainThread;
    binaryStream >> notifyNumPerThread;
    binaryStream >> netLayerNum;

    // Deserialize vecChannelInfo
    uint64_t vecSize;
    binaryStream >> vecSize;
    vecChannelInfo.resize(vecSize);
    for (auto& channel : vecChannelInfo) {
        channel.DeSerialize(binaryStream);
    }
}

void XmlInfo::Serialize(BinaryStream& binaryStream) const {
    // Serialize resInfo
    resInfo.Serialize(binaryStream);

    // Serialize vecNormalInstruction
    binaryStream << static_cast<uint64_t>(vecNormalInstruction.size());
    for (const auto& instr : vecNormalInstruction) {
        instr.Serialize(binaryStream);
    }
    binaryStream << syncNum;
}

void XmlInfo::DeSerialize(BinaryStream& binaryStream) {
    // Deserialize resInfo
    resInfo.DeSerialize(binaryStream);

    // Deserialize vecNormalInstruction
    uint64_t instrSize;
    binaryStream >> instrSize;
    vecNormalInstruction.resize(instrSize);
    for (auto& instr : vecNormalInstruction) {
        instr.DeSerialize(binaryStream);
    }
    binaryStream >> syncNum;
}

// ============== Parser Implementation ==============

uint32_t BinaryParser::ExtractBits(uint64_t value, uint32_t width, uint32_t offset) {
    uint64_t mask = ((1ULL << width) - 1ULL) << offset;
    return static_cast<uint32_t>((value & mask) >> offset);
}

uint32_t BinaryParser::ExtractBits32(uint32_t value, uint32_t width, uint32_t offset) {
    uint32_t mask = ((1U << width) - 1U) << offset;
    return (value & mask) >> offset;
}

OmniSliceInfo BinaryParser::ParseSlice(uint64_t sliceValue) {
    OmniSliceInfo slice;
    slice.sliceType = static_cast<BufferTypeTmp>(
        ExtractBits(sliceValue, field::SLICE_BUFFER_TYPE_WIDTH, field::SLICE_BUFFER_TYPE_OFFSET));
    slice.sliceIdx = ExtractBits(sliceValue, field::SLICE_IDX_WIDTH, field::SLICE_IDX_OFFSET);
    slice.remoteRank = ExtractBits(sliceValue, field::SLICE_RANK_ID_WIDTH, field::SLICE_RANK_ID_OFFSET);
    slice.remoteRecvRank = ExtractBits(sliceValue, field::SLICE_RECV_RANK_ID_WIDTH, field::SLICE_RECV_RANK_ID_OFFSET);
    slice.cnt = ExtractBits(sliceValue, field::SLICE_CNT_WIDTH, field::SLICE_CNT_OFFSET);
    slice.gap = ExtractBits(sliceValue, field::SLICE_GAP_WIDTH, field::SLICE_GAP_OFFSET);
    if (slice.remoteRecvRank == 1023) {
        slice.remoteRecvRank = slice.remoteRank;
    }
    return slice;
}

OmniChannelInfo BinaryParser::ParseChannel(uint32_t channelValue) {
    OmniChannelInfo channel;

    channel.netlayerId = ExtractBits32(channelValue, field::CHAN_NET_LAYER_ID_WIDTH, field::CHAN_NET_LAYER_ID_OFFSET);
    channel.remoteRank = ExtractBits32(channelValue, field::CHAN_REMOTE_RANK_WIDTH, field::CHAN_REMOTE_RANK_OFFSET);
    uint32_t linkProtoValue = ExtractBits32(channelValue, field::CHAN_LINK_PROTO_WIDTH, field::CHAN_LINK_PROTO_OFFSET);
    channel.channelProtocol = static_cast<CommProtocol>(linkProtoValue);
    return channel;
}

void BinaryParser::ParseResRequest(XmlInfo& xmlInfo, const std::vector<uint8_t>& data,
                                    size_t& offset, uint64_t headerValue, uint32_t rankId) {
    auto& resInfo = xmlInfo.resInfo;

    resInfo.slaveThreadNum = ExtractBits(headerValue, field::SLAVE_THREAD_NUM_WIDTH, field::SLAVE_THREAD_NUM_OFFSET);
    resInfo.notifyNumOnMainThread = ExtractBits(headerValue, field::NOTIFY_NUM_MAIN_WIDTH, field::NOTIFY_NUM_MAIN_OFFSET);
    resInfo.notifyNumPerThread = ExtractBits(headerValue, field::NOTIFY_NUM_PER_THREAD_WIDTH, field::NOTIFY_NUM_PER_THREAD_OFFSET);
    resInfo.netLayerNum = ExtractBits(headerValue, field::NET_LAYER_NUM_WIDTH, field::NET_LAYER_NUM_OFFSET);
    uint32_t chanCount = ExtractBits(headerValue, field::CHAN_COUNT_WIDTH, field::CHAN_COUNT_OFFSET);

    offset += INSTRUCTION_HEADER_SIZE;

    HCCL_DEBUG("rank[%u] ResRequest: slaveThreads=%u, notifyMain=%u, notifyPerThread=%u, netLayers=%u, chanCount=%u",
              rankId, resInfo.slaveThreadNum, resInfo.notifyNumOnMainThread,
              resInfo.notifyNumPerThread, resInfo.netLayerNum, chanCount);

    // Parse vecChannelInfo
    for (uint32_t i = 0; i < chanCount; ++i) {
        if (offset + CHANNEL_ENTRY_SIZE > data.size()) {
            HCCL_WARNING("Incomplete channel data at offset %zu (need %zu, have %zu)",
                        offset, CHANNEL_ENTRY_SIZE, data.size() - offset);
            return;
        }

        uint32_t channelValue = static_cast<uint32_t>(data[offset]) |
                               (static_cast<uint32_t>(data[offset+1]) << 8) |
                               (static_cast<uint32_t>(data[offset+2]) << 16) |
                               (static_cast<uint32_t>(data[offset+3]) << 24);
        offset += CHANNEL_ENTRY_SIZE;

        uint32_t localRank = ExtractBits32(channelValue, field::CHAN_LOCAL_RANK_WIDTH, field::CHAN_LOCAL_RANK_OFFSET);

        if (localRank == rankId) {
            resInfo.vecChannelInfo.push_back(ParseChannel(channelValue));
        }
    }
}

void BinaryParser::ParseSyncInstruction(XmlInfo& xmlInfo, const std::vector<uint8_t>& data,
                                         size_t& offset, uint64_t headerValue, OpType opType) {
    uint32_t mainThreadIdx = ExtractBits(headerValue, field::MAIN_THREAD_IDX_WIDTH, field::MAIN_THREAD_IDX_OFFSET);
    uint32_t subThreadNum = ExtractBits(headerValue, field::SUB_THREAD_NUM_WIDTH, field::SUB_THREAD_NUM_OFFSET);

    offset += INSTRUCTION_HEADER_SIZE;

    OmniNormalInstruction instr;
    instr.opType = opType;
    instr.syncInfo.mainThreadIdx = mainThreadIdx;
    instr.syncInfo.subThreadNum = subThreadNum;

    HCCL_DEBUG("SyncInstruction: op=%s, mainThread=%u, subThreadNum=%u",
             OpTypeToString(opType).c_str(), mainThreadIdx, subThreadNum);

    // Parse subthread IDs
    for (uint32_t i = 0; i < subThreadNum; ++i) {
        if (offset + SUBTHREAD_ENTRY_SIZE > data.size()) {
            HCCL_WARNING("Incomplete subthread data at offset %zu", offset);
            return;
        }
        instr.syncInfo.subThreadIds.push_back(data[offset]);
        offset += SUBTHREAD_ENTRY_SIZE;
    }

    xmlInfo.vecNormalInstruction.push_back(instr);
}

void BinaryParser::ParseControlInstruction(XmlInfo& xmlInfo, const std::vector<uint8_t>& data,
                                            size_t& offset, uint64_t headerValue, OpType opType) {
    uint32_t netLayerId = ExtractBits(headerValue, field::NET_LAYER_ID_WIDTH, field::NET_LAYER_ID_OFFSET);
    uint32_t sliceNum = ExtractBits(headerValue, field::SLICE_NUM_WIDTH, field::SLICE_NUM_OFFSET);
    uint32_t srcSliceNum = ExtractBits(headerValue, field::SRC_SLICE_NUM_WIDTH, field::SRC_SLICE_NUM_OFFSET);
    uint32_t dstSliceNum = ExtractBits(headerValue, field::DST_SLICE_NUM_WIDTH, field::DST_SLICE_NUM_OFFSET);
    uint32_t threadIdx = ExtractBits(headerValue, field::THREAD_IDX_WIDTH, field::THREAD_IDX_OFFSET);
    uint32_t reduceType = ExtractBits(headerValue, field::REDUCE_TYPE_WIDTH, field::REDUCE_TYPE_OFFSET);
    uint32_t inputDataType = ExtractBits(headerValue, field::INPUT_DATA_TYPE_WIDTH, field::INPUT_DATA_TYPE_OFFSET);
    uint32_t outputDataType = ExtractBits(headerValue, field::OUTPUT_DATA_TYPE_WIDTH, field::OUTPUT_DATA_TYPE_OFFSET);

    offset += INSTRUCTION_HEADER_SIZE;

    OmniNormalInstruction instr;
    instr.opType = opType;
    instr.sendRecvInfo.inputDataType = static_cast<HcclDataType>(inputDataType);
    instr.sendRecvInfo.outputDataType = static_cast<HcclDataType>(outputDataType);
    instr.sendRecvInfo.reduceType = static_cast<HcclReduceOp>(reduceType);
    instr.sendRecvInfo.sliceNum = sliceNum;
    instr.sendRecvInfo.threadIdx = threadIdx;
    instr.sendRecvInfo.netlayerId = netLayerId;

    HCCL_DEBUG("ControlInstruction: op=%s, sliceNum=%u, netlayer=%u, thread=%u, srcSlices=%u, dstSlices=%u",
             OpTypeToString(opType).c_str(), sliceNum, netLayerId, threadIdx, srcSliceNum, dstSliceNum);

    // Parse source slices
    for (uint32_t i = 0; i < srcSliceNum; ++i) {
        if (offset + SLICE_ENTRY_SIZE > data.size()) {
            HCCL_WARNING("Incomplete src slice data at offset %zu", offset);
            return;
        }

        uint64_t sliceValue = static_cast<uint64_t>(data[offset]) |
                             (static_cast<uint64_t>(data[offset+1]) << 8) |
                             (static_cast<uint64_t>(data[offset+2]) << 16) |
                             (static_cast<uint64_t>(data[offset+3]) << 24) |
                             (static_cast<uint64_t>(data[offset+4]) << 32) |
                             (static_cast<uint64_t>(data[offset+5]) << 40) |
                             (static_cast<uint64_t>(data[offset+6]) << 48) |
                             (static_cast<uint64_t>(data[offset+7]) << 56);

        offset += SLICE_ENTRY_SIZE;

        OmniSliceInfo omniSliceInfo = ParseSlice(sliceValue);
        instr.sendRecvInfo.srcSliceInfo.push_back(omniSliceInfo);

        for (u32 i = 1; i < omniSliceInfo.cnt; i++) {
            omniSliceInfo.sliceIdx += omniSliceInfo.gap;
            instr.sendRecvInfo.srcSliceInfo.push_back(omniSliceInfo);
        }
    }

    // Parse destination slices
    for (uint32_t i = 0; i < dstSliceNum; ++i) {
        if (offset + SLICE_ENTRY_SIZE > data.size()) {
            HCCL_WARNING("Incomplete dst slice data at offset %zu", offset);
            return;
        }

        uint64_t sliceValue = static_cast<uint64_t>(data[offset]) |
                             (static_cast<uint64_t>(data[offset+1]) << 8) |
                             (static_cast<uint64_t>(data[offset+2]) << 16) |
                             (static_cast<uint64_t>(data[offset+3]) << 24) |
                             (static_cast<uint64_t>(data[offset+4]) << 32) |
                             (static_cast<uint64_t>(data[offset+5]) << 40) |
                             (static_cast<uint64_t>(data[offset+6]) << 48) |
                             (static_cast<uint64_t>(data[offset+7]) << 56);
        offset += SLICE_ENTRY_SIZE;

        OmniSliceInfo omniSliceInfo = ParseSlice(sliceValue);
        instr.sendRecvInfo.dstSliceInfo.push_back(omniSliceInfo);

        for (u32 i = 1; i < omniSliceInfo.cnt; i++) {
            omniSliceInfo.sliceIdx += omniSliceInfo.gap;
            instr.sendRecvInfo.dstSliceInfo.push_back(omniSliceInfo);
        }
    }

    xmlInfo.vecNormalInstruction.push_back(instr);
}

void BinaryParser::PrintHexDump(const std::vector<uint8_t>& data, const std::string& filepath) {
    std::cout << "\n" << std::string(60, '$') << "\n";
    std::cout << "Hex Dump: " << filepath << "\n";
    std::cout << std::string(60, '$') << "\n";

    for (size_t i = 0; i < data.size(); i += 16) {
        std::cout << std::hex << std::setfill('0') << std::setw(4) << i << ": ";

        for (size_t j = 0; j < 16; ++j) {
            if (i + j < data.size()) {
                std::cout << std::hex << std::setfill('0') << std::setw(2)
                          << static_cast<int>(data[i + j]) << " ";
            } else {
                std::cout << "   ";
            }
        }

        std::cout << " ";
        for (size_t j = 0; j < 16 && i + j < data.size(); ++j) {
            uint8_t c = data[i + j];
            std::cout << (32 <= c && c < 127 ? static_cast<char>(c) : '.');
        }
        std::cout << "\n";
    }
    std::cout << std::dec;
}

HcclResult BinaryParser::SetFile(const std::string& filepath, uint32_t rankId) {
    // Check if file exists and get size
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        HCCL_ERROR("Failed to open file: %s", filepath.c_str());
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    fileSize_ = file.tellg();
    file.close();

    // Store file metadata
    filepath_ = filepath;
    size_t lastSlash = filepath.find_last_of("/\\");
    filename_ = (lastSlash != std::string::npos) ? filepath.substr(lastSlash + 1) : filepath;
    rankId_ = rankId;

    HCCL_INFO("SetFile: %s, size=%zu bytes, rank=%u", filename_.c_str(), fileSize_, rankId_);
    return HCCL_SUCCESS;
}

HcclResult BinaryParser::Parse(XmlInfo& xmlInfo) {
    // Open file
    std::ifstream file(filepath_, std::ios::binary);
    if (!file.is_open()) {
        HCCL_ERROR("Failed to open file: %s", filepath_.c_str());
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    // Read all data
    std::vector<uint8_t> data(fileSize_);
    file.read(reinterpret_cast<char*>(data.data()), fileSize_);
    file.close();

    // Show hex dump if requested
    if (showHex_) {
        PrintHexDump(data, filepath_);
    }

    HCCL_INFO("Parsing file: %s, size=%zu bytes, rank=%u", filename_.c_str(), fileSize_, rankId_);

    // Parse instructions
    size_t offset = 0;
    while (offset < data.size()) {
        if (offset + INSTRUCTION_HEADER_SIZE > data.size()) {
            HCCL_WARNING("Incomplete header at offset %zu", offset);
            return HCCL_E_PARA;
        }

        // Read header (little-endian)
        uint64_t headerValue = static_cast<uint64_t>(data[offset]) |
                              (static_cast<uint64_t>(data[offset+1]) << 8) |
                              (static_cast<uint64_t>(data[offset+2]) << 16) |
                              (static_cast<uint64_t>(data[offset+3]) << 24) |
                              (static_cast<uint64_t>(data[offset+4]) << 32) |
                              (static_cast<uint64_t>(data[offset+5]) << 40) |
                              (static_cast<uint64_t>(data[offset+6]) << 48) |
                              (static_cast<uint64_t>(data[offset+7]) << 56);

        OpType opType = static_cast<OpType>(ExtractBits(headerValue, field::OPCODE_WIDTH, field::OPCODE_OFFSET));
        InstructionType instrType = GetInstructionType(opType);

        HCCL_DEBUG("offset=%zu, opCode=%u (%s)", offset, static_cast<uint32_t>(opType), OpTypeToString(opType).c_str());

        switch (instrType) {
            case InstructionType::RES_REQUEST:
                ParseResRequest(xmlInfo, data, offset, headerValue, rankId_);
                break;
            case InstructionType::SYNC:
                ParseSyncInstruction(xmlInfo, data, offset, headerValue, opType);
                break;
            case InstructionType::CONTROL:
                ParseControlInstruction(xmlInfo, data, offset, headerValue, opType);
                break;
        }
    }

    HCCL_INFO("Parsing complete: %zu instructions", xmlInfo.vecNormalInstruction.size());
    return HCCL_SUCCESS;
}

} // namespace omni
} // namespace ops_hccl
