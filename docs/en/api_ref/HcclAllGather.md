# HcclAllGather<a name="ZH-CN_TOPIC_0000002486832340"></a>

## Supported Products<a name="zh-cn_topic_0000001265081270_section10594071513"></a>

<a name="zh-cn_topic_0000001265081270_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265081270_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265081270_p1883113061818"><a name="zh-cn_topic_0000001265081270_p1883113061818"></a><a name="zh-cn_topic_0000001265081270_p1883113061818"></a><span id="zh-cn_topic_0000001265081270_ph20833205312295"><a name="zh-cn_topic_0000001265081270_ph20833205312295"></a><a name="zh-cn_topic_0000001265081270_ph20833205312295"></a> Product </span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265081270_p783113012187"><a name="zh-cn_topic_0000001265081270_p783113012187"></a><a name="zh-cn_topic_0000001265081270_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265081270_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265081270_p48327011813"><a name="zh-cn_topic_0000001265081270_p48327011813"></a><a name="zh-cn_topic_0000001265081270_p48327011813"></a><span id="zh-cn_topic_0000001265081270_ph583230201815"><a name="zh-cn_topic_0000001265081270_ph583230201815"></a><a name="zh-cn_topic_0000001265081270_ph583230201815"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training /inference series products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265081270_p7948163910184"><a name="zh-cn_topic_0000001265081270_p7948163910184"></a><a name="zh-cn_topic_0000001265081270_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265081270_p14832120181815"><a name="zh-cn_topic_0000001265081270_p14832120181815"></a><a name="zh-cn_topic_0000001265081270_p14832120181815"></a><span id="zh-cn_topic_0000001265081270_ph1292674871116"><a name="zh-cn_topic_0000001265081270_ph1292674871116"></a><a name="zh-cn_topic_0000001265081270_ph1292674871116"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265081270_p19948143911820"><a name="zh-cn_topic_0000001265081270_p19948143911820"></a><a name="zh-cn_topic_0000001265081270_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training /inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000001265081270_section59721402"></a>

`AllGather` is a collective communication operator that re-sorts the inputs of all ranks in the communicator by rank ID, combines the inputs, and sends the results to the outputs of all ranks.

![](figures/allgather-0.png)

> [!NOTE]NOTE
> For the `AllGather` operation, each rank receives a set of data that is resorted by rank ID, that is, `AllGather` outputs of all ranks are the same.

## Prototype<a name="zh-cn_topic_0000001265081270_section66288034"></a>

```
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001265081270_section621706"></a>

<a name="zh-cn_topic_0000001265081270_table51170717"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265081270_row27848947"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265081270_p41172271"><a name="zh-cn_topic_0000001265081270_p41172271"></a><a name="zh-cn_topic_0000001265081270_p41172271"></a> Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265081270_p46619622"><a name="zh-cn_topic_0000001265081270_p46619622"></a><a name="zh-cn_topic_0000001265081270_p46619622"></a> Input/Output </p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265081270_p18093058"><a name="zh-cn_topic_0000001265081270_p18093058"></a><a name="zh-cn_topic_0000001265081270_p18093058"></a> Description </p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265081270_row56251627"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p60087944"><a name="zh-cn_topic_0000001265081270_p60087944"></a><a name="zh-cn_topic_0000001265081270_p60087944"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p35285314"><a name="zh-cn_topic_0000001265081270_p35285314"></a><a name="zh-cn_topic_0000001265081270_p35285314"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p39538176"><a name="zh-cn_topic_0000001265081270_p39538176"></a><a name="zh-cn_topic_0000001265081270_p39538176"></a> Source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row20299268"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p33628036"><a name="zh-cn_topic_0000001265081270_p33628036"></a><a name="zh-cn_topic_0000001265081270_p33628036"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p39516413"><a name="zh-cn_topic_0000001265081270_p39516413"></a><a name="zh-cn_topic_0000001265081270_p39516413"></a> Output </p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p46712854"><a name="zh-cn_topic_0000001265081270_p46712854"></a><a name="zh-cn_topic_0000001265081270_p46712854"></a>Destination buffer address where the collective communication result is stored.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row17762502"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p29476530"><a name="zh-cn_topic_0000001265081270_p29476530"></a><a name="zh-cn_topic_0000001265081270_p29476530"></a>sendCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p38788755"><a name="zh-cn_topic_0000001265081270_p38788755"></a><a name="zh-cn_topic_0000001265081270_p38788755"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p54881489"><a name="zh-cn_topic_0000001265081270_p54881489"></a><a name="zh-cn_topic_0000001265081270_p54881489"></a> `sendBuf` data size for the `AllGather` operation. The `recvBuf` data size is equal to `count * rank_size`.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row24171358"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p11722994"><a name="zh-cn_topic_0000001265081270_p11722994"></a><a name="zh-cn_topic_0000001265081270_p11722994"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p10038421"><a name="zh-cn_topic_0000001265081270_p10038421"></a><a name="zh-cn_topic_0000001265081270_p10038421"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p51201049151217"><a name="zh-cn_topic_0000001265081270_p51201049151217"></a><a name="zh-cn_topic_0000001265081270_p51201049151217"></a>Data type of the `Allgather` operation, defined in <a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000001265081270_p578154617218"><a name="zh-cn_topic_0000001265081270_p578154617218"></a><a name="zh-cn_topic_0000001265081270_p578154617218"></a><span id="zh-cn_topic_0000001265081270_ph13754548217"><a name="zh-cn_topic_0000001265081270_ph13754548217"></a><a name="zh-cn_topic_0000001265081270_ph13754548217"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data types supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000001265081270_p94179211177"><a name="zh-cn_topic_0000001265081270_p94179211177"></a><a name="zh-cn_topic_0000001265081270_p94179211177"></a><span id="zh-cn_topic_0000001265081270_ph14880920154918"><a name="zh-cn_topic_0000001265081270_ph14880920154918"></a><a name="zh-cn_topic_0000001265081270_ph14880920154918"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term16184138172215"></a>Data types supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row3143429"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p53291177"><a name="zh-cn_topic_0000001265081270_p53291177"></a><a name="zh-cn_topic_0000001265081270_p53291177"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p21618074"><a name="zh-cn_topic_0000001265081270_p21618074"></a><a name="zh-cn_topic_0000001265081270_p21618074"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p6233535"><a name="zh-cn_topic_0000001265081270_p6233535"></a><a name="zh-cn_topic_0000001265081270_p6233535"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row56101816"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p47953287"><a name="zh-cn_topic_0000001265081270_p47953287"></a><a name="zh-cn_topic_0000001265081270_p47953287"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p59011071"><a name="zh-cn_topic_0000001265081270_p59011071"></a><a name="zh-cn_topic_0000001265081270_p59011071"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p15167431"><a name="zh-cn_topic_0000001265081270_p15167431"></a><a name="zh-cn_topic_0000001265081270_p15167431"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001265081270_section5595356"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001265081270_section50358210"></a>

The ranks must have the same `sendCount` and `dataType`.

## Call Example<a name="zh-cn_topic_0000001265081270_section204039211474"></a>

```c
// Allocate device memory for collective communication.
void *sendBuf = nullptr, *recvBuf = nullptr;
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // Number of data elements sent by each rank.
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and streams.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, devId, &hcclComm);

// Create a task flow.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute AllGather to concatenate sendBuf of all ranks in the communicator in rank ID order, and send the result to recvBuf of all ranks.
HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
