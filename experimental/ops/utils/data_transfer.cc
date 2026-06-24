HcclResult SendWrite(const DataInfo &sendInfo, const ThreadHandle &thread);
HcclResult RecvWrite(const DataInfo &recvInfo, const ThreadHandle &thread);
HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread);

HcclResult SendWriteReduce(const DataReduceInfo &sendInfo, const ThreadHandle &thread);
HcclResult RecvWriteReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread);
HcclResult SendRecvWriteReduce(const SendRecvReduceInfo &sendRecvInfo, const ThreadHandle &thread);

HcclResult SendRead(const DataInfo &sendInfo, const ThreadHandle &thread);
HcclResult RecvRead(const DataInfo &recvInfo, const ThreadHandle &thread);
HcclResult SendRecvRead(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread);

HcclResult SendReadReduce(const DataReduceInfo &sendInfo, const ThreadHandle &thread);
HcclResult RecvReadReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread);
HcclResult SendRecvReadReduce(const SendRecvReduceInfo &sendRecvInfo, const ThreadHandle &thread);