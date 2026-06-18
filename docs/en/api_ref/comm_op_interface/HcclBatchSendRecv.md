# HcclBatchSendRecv<a name="ZH-CN_TOPIC_0000002486993298"></a>

## Supported Products<a name="zh-cn_topic_0000001811681609_section10594071513"></a>

<a name="zh-cn_topic_0000001811681609_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001811681609_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001811681609_p1883113061818"><a name="zh-cn_topic_0000001811681609_p1883113061818"></a><a name="zh-cn_topic_0000001811681609_p1883113061818"></a><span id="zh-cn_topic_0000001811681609_ph20833205312295"><a name="zh-cn_topic_0000001811681609_ph20833205312295"></a><a name="zh-cn_topic_0000001811681609_ph20833205312295"></a> Product </span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001811681609_p783113012187"><a name="zh-cn_topic_0000001811681609_p783113012187"></a><a name="zh-cn_topic_0000001811681609_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001811681609_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001811681609_p48327011813"><a name="zh-cn_topic_0000001811681609_p48327011813"></a><a name="zh-cn_topic_0000001811681609_p48327011813"></a><span id="zh-cn_topic_0000001811681609_ph583230201815"><a name="zh-cn_topic_0000001811681609_ph583230201815"></a><a name="zh-cn_topic_0000001811681609_ph583230201815"></a><term id="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001811681609_p7948163910184"><a name="zh-cn_topic_0000001811681609_p7948163910184"></a><a name="zh-cn_topic_0000001811681609_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001811681609_p14832120181815"><a name="zh-cn_topic_0000001811681609_p14832120181815"></a><a name="zh-cn_topic_0000001811681609_p14832120181815"></a><span id="zh-cn_topic_0000001811681609_ph1292674871116"><a name="zh-cn_topic_0000001811681609_ph1292674871116"></a><a name="zh-cn_topic_0000001811681609_ph1292674871116"></a><term id="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001811681609_p19948143911820"><a name="zh-cn_topic_0000001811681609_p19948143911820"></a><a name="zh-cn_topic_0000001811681609_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training/inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000001811681609_section212645315215"></a>

Complete send and receive tasks in batches on the current rank. The send and receive tasks of the current rank are asynchronous and do not block each other.

## Prototype<a name="zh-cn_topic_0000001811681609_section13125135314218"></a>

```
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001811681609_section1812717539212"></a>

<a name="zh-cn_topic_0000001811681609_table18137135310213"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001811681609_row1417285311217"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001811681609_p131726530216"><a name="zh-cn_topic_0000001811681609_p131726530216"></a><a name="zh-cn_topic_0000001811681609_p131726530216"></a> Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001811681609_p01721653524"><a name="zh-cn_topic_0000001811681609_p01721653524"></a><a name="zh-cn_topic_0000001811681609_p01721653524"></a> Input/Output </p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001811681609_p7172195319214"><a name="zh-cn_topic_0000001811681609_p7172195319214"></a><a name="zh-cn_topic_0000001811681609_p7172195319214"></a> Description </p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001811681609_row1117295311215"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p195466529172"><a name="zh-cn_topic_0000001811681609_p195466529172"></a><a name="zh-cn_topic_0000001811681609_p195466529172"></a>sendRecvInfo</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p161721753129"><a name="zh-cn_topic_0000001811681609_p161721753129"></a><a name="zh-cn_topic_0000001811681609_p161721753129"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p5172353028"><a name="zh-cn_topic_0000001811681609_p5172353028"></a><a name="zh-cn_topic_0000001811681609_p5172353028"></a> Start address of the list of send and receive tasks to be distributed in the rank.</p>
The value is of the <p id="zh-cn_topic_0000001811681609_p14928105415418"><a name="zh-cn_topic_0000001811681609_p14928105415418"></a><a name="zh-cn_topic_0000001811681609_p14928105415418"></a>`HcclSendRecvItem type`. For details, see <a href="https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclSendRecvItem.md#ZH-CN_TOPIC_0000002519072197">HcclSendRecvItem</a>.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row41722531724"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p38341698166"><a name="zh-cn_topic_0000001811681609_p38341698166"></a><a name="zh-cn_topic_0000001811681609_p38341698166"></a>itemNum</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p617275320217"><a name="zh-cn_topic_0000001811681609_p617275320217"></a><a name="zh-cn_topic_0000001811681609_p617275320217"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p41729531229"><a name="zh-cn_topic_0000001811681609_p41729531229"></a><a name="zh-cn_topic_0000001811681609_p41729531229"></a> Number of tasks to be received and sent by the rank.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row1117220531028"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p15172653628"><a name="zh-cn_topic_0000001811681609_p15172653628"></a><a name="zh-cn_topic_0000001811681609_p15172653628"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p1517219536214"><a name="zh-cn_topic_0000001811681609_p1517219536214"></a><a name="zh-cn_topic_0000001811681609_p1517219536214"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p834310091714"><a name="zh-cn_topic_0000001811681609_p834310091714"></a><a name="zh-cn_topic_0000001811681609_p834310091714"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row18172165315213"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p101724536217"><a name="zh-cn_topic_0000001811681609_p101724536217"></a><a name="zh-cn_topic_0000001811681609_p101724536217"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p917215320212"><a name="zh-cn_topic_0000001811681609_p917215320212"></a><a name="zh-cn_topic_0000001811681609_p917215320212"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p88321247141619"><a name="zh-cn_topic_0000001811681609_p88321247141619"></a><a name="zh-cn_topic_0000001811681609_p88321247141619"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001811681609_section1513715531221"></a>

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001811681609_section86843302218"></a>

-   "Asynchronous" means that the send and receive tasks on the same device are asynchronous and do not block each other. However, the send and receive tasks between devices are still synchronous. Therefore, the send and receive tasks between devices must be in one-to-one mapping, which is the same as `HcclSend` and `HcclRecv`.
-   The task list must not contain duplicate send or receive tasks pointed to the same rank.
-   In the current version, this API does not support the scenario where Virtual Pipeline (VPP) is enabled.
-   For Atlas A2 training/inference products, when this API is used in a large-scale cluster (ranksize\>500), the number of concurrent tasks cannot exceed 3.
-   For [the Atlas 200T A2 Box16 heterogeneous subrack](https://support.huawei.com/enterprise/zh/doc/EDOC1100318274/287e0458), if the link setup between NPUs on the server fails (error code: EI0010), set the environment variable `HCCL_INTRA_ROCE_ENABLE` to `1` and `HCCL_INTRA_Pcie_ENABLE` to `0` to enable the RoCE loop for communication between multiple NPUs on the server. (Ensure that the server has RoCE NICs and the RDMA links between devices that have send/recv relationships are reachable.) The following is an example of configuring environment variables:

    ```
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_INTRA_PCIE_ENABLE=0
    ```

## Call Example<a name="zh-cn_topic_0000001811681609_section204039211474"></a>

```c
// Allocate device memory for collective communication.
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);
aclrtMalloc((void **)&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc((void **)&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task flow.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform the Send/Recv operation to send data to the next rank and receive data from the previous rank.
// HcclBatchSendRecv can deliver multiple send and receive tasks on the local rank at the same time.
uint32_t next = (deviceId + 1) % count;
uint32_t prev = (deviceId - 1 + count) % count;
HcclSendRecvItem sendRecvInfo[2];
sendRecvInfo[0] = HcclSendRecvItem{HCCL_SEND, sendBuf, count, HCCL_DATA_TYPE_FP32, next};
sendRecvInfo[1] = HcclSendRecvItem{HCCL_RECV, recvBuf, count, HCCL_DATA_TYPE_FP32, prev};
HcclBatchSendRecv(sendRecvInfo, 2, hcclComm, stream);

// Wait until the collective communication task in the task flow is complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
