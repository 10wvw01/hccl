# HCCL Code Example

This directory provides sample code demonstrating how to use HCCL APIs to implement collective communication in different scenarios.

## Point-to-Point Communication

- [HcclSend/HcclRecv (basic receive/send functions)](./01_point_to_point/01_send_recv)
- [HcclBatchSendRecv (ring communication)](./01_point_to_point/02_batch_send_recv_ring)

## Collective Communication

- [AllReduce](./02_collectives/01_allreduce)
- [Broadcast](./02_collectives/02_broadcast)
- [AllGather](./02_collectives/03_allgather)
- [ReduceScatter](./02_collectives/04_reduce_scatter)
- [Reduce](./02_collectives/05_reduce)
- [AlltoAll](./02_collectives/06_alltoall)
- [AlltoAllV](./02_collectives/07_alltoallv)
- [AlltoAllVC](./02_collectives/08_alltoallvc)
- [Scatter](./02_collectives/09_scatter )

## AI Frameworks

- [PyTorch](./03_ai_framework/01_pytorch)
- [Tensorflow](./03_ai_framework/02_tensorflow)

## Custom Point-to-Point Communication Operators

- [Custom Send/Recv Operators (Based on AICPU Communication Engine)](./04_custom_ops_p2p)
