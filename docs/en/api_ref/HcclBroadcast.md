# HcclBroadcast<a name="ZH-CN_TOPIC_0000002519072187"></a>

## Supported Products<a name="zh-cn_topic_0000001312481237_section10594071513"></a>

<a name="zh-cn_topic_0000001312481237_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312481237_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312481237_p1883113061818"><a name="zh-cn_topic_0000001312481237_p1883113061818"></a><a name="zh-cn_topic_0000001312481237_p1883113061818"></a><span id="zh-cn_topic_0000001312481237_ph20833205312295"><a name="zh-cn_topic_0000001312481237_ph20833205312295"></a><a name="zh-cn_topic_0000001312481237_ph20833205312295"></a>Product</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312481237_p783113012187"><a name="zh-cn_topic_0000001312481237_p783113012187"></a><a name="zh-cn_topic_0000001312481237_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312481237_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312481237_p48327011813"><a name="zh-cn_topic_0000001312481237_p48327011813"></a><a name="zh-cn_topic_0000001312481237_p48327011813"></a><span id="zh-cn_topic_0000001312481237_ph583230201815"><a name="zh-cn_topic_0000001312481237_ph583230201815"></a><a name="zh-cn_topic_0000001312481237_ph583230201815"></a><term id="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312481237_p7948163910184"><a name="zh-cn_topic_0000001312481237_p7948163910184"></a><a name="zh-cn_topic_0000001312481237_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481237_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312481237_p14832120181815"><a name="zh-cn_topic_0000001312481237_p14832120181815"></a><a name="zh-cn_topic_0000001312481237_p14832120181815"></a><span id="zh-cn_topic_0000001312481237_ph1292674871116"><a name="zh-cn_topic_0000001312481237_ph1292674871116"></a><a name="zh-cn_topic_0000001312481237_ph1292674871116"></a><term id="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312481237_p19948143911820"><a name="zh-cn_topic_0000001312481237_p19948143911820"></a><a name="zh-cn_topic_0000001312481237_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training /inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000001312481237_section10896307"></a>

`Broadcast` is a collective communication operator that broadcasts the data of the root rank to other ranks in the communicator.

![](figures/allgather.png)

## Prototype<a name="zh-cn_topic_0000001312481237_section1210700"></a>

```
HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001312481237_section30957904"></a>

<a name="zh-cn_topic_0000001312481237_table59291480"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312481237_row30364973"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312481237_p43643771"><a name="zh-cn_topic_0000001312481237_p43643771"></a><a name="zh-cn_topic_0000001312481237_p43643771"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312481237_p45484537"><a name="zh-cn_topic_0000001312481237_p45484537"></a><a name="zh-cn_topic_0000001312481237_p45484537"></a>Input/Output</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312481237_p60368847"><a name="zh-cn_topic_0000001312481237_p60368847"></a><a name="zh-cn_topic_0000001312481237_p60368847"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312481237_row58038438"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481237_p3493029"><a name="zh-cn_topic_0000001312481237_p3493029"></a><a name="zh-cn_topic_0000001312481237_p3493029"></a>buf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481237_p14499957"><a name="zh-cn_topic_0000001312481237_p14499957"></a><a name="zh-cn_topic_0000001312481237_p14499957"></a>input/output</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481237_p33645886"><a name="zh-cn_topic_0000001312481237_p33645886"></a><a name="zh-cn_topic_0000001312481237_p33645886"></a>Data buffer, used as a send buffer for root ranks, while a receive buffer for non-root ranks.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481237_row34377519"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481237_p33115615"><a name="zh-cn_topic_0000001312481237_p33115615"></a><a name="zh-cn_topic_0000001312481237_p33115615"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481237_p65119131"><a name="zh-cn_topic_0000001312481237_p65119131"></a><a name="zh-cn_topic_0000001312481237_p65119131"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481237_p40158284"><a name="zh-cn_topic_0000001312481237_p40158284"></a><a name="zh-cn_topic_0000001312481237_p40158284"></a>Number of data elements involved in the `Broadcast` operation. For example, `count = 1` indicates that one int32 data element is involved.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481237_row25880244"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481237_p15925039"><a name="zh-cn_topic_0000001312481237_p15925039"></a><a name="zh-cn_topic_0000001312481237_p15925039"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481237_p14859795"><a name="zh-cn_topic_0000001312481237_p14859795"></a><a name="zh-cn_topic_0000001312481237_p14859795"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481237_p7991229191118"><a name="zh-cn_topic_0000001312481237_p7991229191118"></a><a name="zh-cn_topic_0000001312481237_p7991229191118"></a>Data type of the `Broadcast` operation, defined in <a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000001312481237_p114661218162118"><a name="zh-cn_topic_0000001312481237_p114661218162118"></a><a name="zh-cn_topic_0000001312481237_p114661218162118"></a><span id="zh-cn_topic_0000001312481237_ph13754548217"><a name="zh-cn_topic_0000001312481237_ph13754548217"></a><a name="zh-cn_topic_0000001312481237_ph13754548217"></a><term id="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data type supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000001312481237_p94179211177"><a name="zh-cn_topic_0000001312481237_p94179211177"></a><a name="zh-cn_topic_0000001312481237_p94179211177"></a><span id="zh-cn_topic_0000001312481237_ph14880920154918"><a name="zh-cn_topic_0000001312481237_ph14880920154918"></a><a name="zh-cn_topic_0000001312481237_ph14880920154918"></a><term id="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001312481237_zh-cn_topic_0000001312391781_term16184138172215"></a>Data type supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481237_row28263486"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481237_p7641038"><a name="zh-cn_topic_0000001312481237_p7641038"></a><a name="zh-cn_topic_0000001312481237_p7641038"></a>root</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481237_p14944322"><a name="zh-cn_topic_0000001312481237_p14944322"></a><a name="zh-cn_topic_0000001312481237_p14944322"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481237_p2530563"><a name="zh-cn_topic_0000001312481237_p2530563"></a><a name="zh-cn_topic_0000001312481237_p2530563"></a>Rank ID of the the broadcast root.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481237_row22775067"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481237_p32841145"><a name="zh-cn_topic_0000001312481237_p32841145"></a><a name="zh-cn_topic_0000001312481237_p32841145"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481237_p42887128"><a name="zh-cn_topic_0000001312481237_p42887128"></a><a name="zh-cn_topic_0000001312481237_p42887128"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481237_p51305376"><a name="zh-cn_topic_0000001312481237_p51305376"></a><a name="zh-cn_topic_0000001312481237_p51305376"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481237_row59095202"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481237_p21982073"><a name="zh-cn_topic_0000001312481237_p21982073"></a><a name="zh-cn_topic_0000001312481237_p21982073"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481237_p35717517"><a name="zh-cn_topic_0000001312481237_p35717517"></a><a name="zh-cn_topic_0000001312481237_p35717517"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481237_p7437765"><a name="zh-cn_topic_0000001312481237_p7437765"></a><a name="zh-cn_topic_0000001312481237_p7437765"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001312481237_section10185684"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001312481237_section24562296"></a>

-   The ranks must have the same `count`, `dataType`, and `root`.
-   There can be only one root rank globally.

## Call Example<a name="zh-cn_topic_0000001312481237_section204039211474"></a>

```c
// Allocate device memory for collective communication.
void *buf = nullptr;    // Data buffer, used as a send buffer for root ranks, while a receive buffer for non-root ranks.
uint64_t count = 8;     // Number of data elements involved in the broadcast operation.
size_t mallocSize = count * sizeof(float);
aclrtMalloc(&buf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Construct the input data on the root rank.
if (deviceId == rootRank) {    
    aclrtMemcpy(buf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task flow.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform the broadcast operation to broadcast the data of the root rank in the communicator to other ranks.
HcclBroadcast(buf, count, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
