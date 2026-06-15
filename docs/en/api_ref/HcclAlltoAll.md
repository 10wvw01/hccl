# HcclAlltoAll<a name="ZH-CN_TOPIC_0000002486992306"></a>

## Supported Products<a name="zh-cn_topic_0000001690107441_section10594071513"></a>

<a name="zh-cn_topic_0000001690107441_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001690107441_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001690107441_p1883113061818"><a name="zh-cn_topic_0000001690107441_p1883113061818"></a><a name="zh-cn_topic_0000001690107441_p1883113061818"></a><span id="zh-cn_topic_0000001690107441_ph20833205312295"><a name="zh-cn_topic_0000001690107441_ph20833205312295"></a><a name="zh-cn_topic_0000001690107441_ph20833205312295"></a>Product</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001690107441_p783113012187"><a name="zh-cn_topic_0000001690107441_p783113012187"></a><a name="zh-cn_topic_0000001690107441_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001690107441_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001690107441_p48327011813"><a name="zh-cn_topic_0000001690107441_p48327011813"></a><a name="zh-cn_topic_0000001690107441_p48327011813"></a><span id="zh-cn_topic_0000001690107441_ph583230201815"><a name="zh-cn_topic_0000001690107441_ph583230201815"></a><a name="zh-cn_topic_0000001690107441_ph583230201815"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001690107441_p7948163910184"><a name="zh-cn_topic_0000001690107441_p7948163910184"></a><a name="zh-cn_topic_0000001690107441_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001690107441_p14832120181815"><a name="zh-cn_topic_0000001690107441_p14832120181815"></a><a name="zh-cn_topic_0000001690107441_p14832120181815"></a><span id="zh-cn_topic_0000001690107441_ph1292674871116"><a name="zh-cn_topic_0000001690107441_ph1292674871116"></a><a name="zh-cn_topic_0000001690107441_ph1292674871116"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001690107441_p19948143911820"><a name="zh-cn_topic_0000001690107441_p19948143911820"></a><a name="zh-cn_topic_0000001690107441_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training /inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000001690107441_section37208511199"></a>

`AlltoAll` is a collective communication operator that sends the same-sized data to all ranks and receives the same-sized data from all ranks.

![](figures/allreduce-5.png)

The `AlltoAll` operation splits the input data into a specific number of blocks along a give dimension, sends the blocks sequentially to other ranks, receives data from other ranks, and concatenates the received data along the same dimension in order.

## Prototype<a name="zh-cn_topic_0000001690107441_section35919731916"></a>

```
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001690107441_section2586134311199"></a>

<a name="zh-cn_topic_0000001690107441_table0576473316"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001690107441_row1060511716320"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001690107441_p146051071139"><a name="zh-cn_topic_0000001690107441_p146051071139"></a><a name="zh-cn_topic_0000001690107441_p146051071139"></a> Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001690107441_p1160527939"><a name="zh-cn_topic_0000001690107441_p1160527939"></a><a name="zh-cn_topic_0000001690107441_p1160527939"></a>Input/Output</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001690107441_p86058714320"><a name="zh-cn_topic_0000001690107441_p86058714320"></a><a name="zh-cn_topic_0000001690107441_p86058714320"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001690107441_row166054719318"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p111231019101719"><a name="zh-cn_topic_0000001690107441_p111231019101719"></a><a name="zh-cn_topic_0000001690107441_p111231019101719"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p51231519111711"><a name="zh-cn_topic_0000001690107441_p51231519111711"></a><a name="zh-cn_topic_0000001690107441_p51231519111711"></a>Input</p>
</td>
Address of the <td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p612301916172"><a name="zh-cn_topic_0000001690107441_p612301916172"></a><a name="zh-cn_topic_0000001690107441_p612301916172"></a> source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row460577337"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p412311914178"><a name="zh-cn_topic_0000001690107441_p412311914178"></a><a name="zh-cn_topic_0000001690107441_p412311914178"></a>sendCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p01231219171717"><a name="zh-cn_topic_0000001690107441_p01231219171717"></a><a name="zh-cn_topic_0000001690107441_p01231219171717"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p5123171961713"><a name="zh-cn_topic_0000001690107441_p5123171961713"></a><a name="zh-cn_topic_0000001690107441_p5123171961713"></a>Amount of data sent to each rank.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row206057717312"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p81241419101717"><a name="zh-cn_topic_0000001690107441_p81241419101717"></a><a name="zh-cn_topic_0000001690107441_p81241419101717"></a>sendType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p151241419151715"><a name="zh-cn_topic_0000001690107441_p151241419151715"></a><a name="zh-cn_topic_0000001690107441_p151241419151715"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p142831647144118"><a name="zh-cn_topic_0000001690107441_p142831647144118"></a><a name="zh-cn_topic_0000001690107441_p142831647144118"></a>Sent data type, defined in<a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000001690107441_p28501558152815"><a name="zh-cn_topic_0000001690107441_p28501558152815"></a><a name="zh-cn_topic_0000001690107441_p28501558152815"></a><span id="zh-cn_topic_0000001690107441_ph11201161819282"><a name="zh-cn_topic_0000001690107441_ph11201161819282"></a><a name="zh-cn_topic_0000001690107441_ph11201161819282"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data type supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000001690107441_p12283142716193"><a name="zh-cn_topic_0000001690107441_p12283142716193"></a><a name="zh-cn_topic_0000001690107441_p12283142716193"></a><span id="zh-cn_topic_0000001690107441_ph161710422273"><a name="zh-cn_topic_0000001690107441_ph161710422273"></a><a name="zh-cn_topic_0000001690107441_ph161710422273"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215"></a>Data type supported by Atlas A2 training /inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row146051372315"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p512431910174"><a name="zh-cn_topic_0000001690107441_p512431910174"></a><a name="zh-cn_topic_0000001690107441_p512431910174"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p151241419131716"><a name="zh-cn_topic_0000001690107441_p151241419131716"></a><a name="zh-cn_topic_0000001690107441_p151241419131716"></a>Output</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p161241219201719"><a name="zh-cn_topic_0000001690107441_p161241219201719"></a><a name="zh-cn_topic_0000001690107441_p161241219201719"></a>Destination buffer address where the collective communication result is stored.</p>
<p id="zh-cn_topic_0000001690107441_p57711733112312"><a name="zh-cn_topic_0000001690107441_p57711733112312"></a><a name="zh-cn_topic_0000001690107441_p57711733112312"></a>The address configured for `recvBuf` and `sendBuf` must not be the same.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row14605137334"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p121254195173"><a name="zh-cn_topic_0000001690107441_p121254195173"></a><a name="zh-cn_topic_0000001690107441_p121254195173"></a>recvCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p4125181921717"><a name="zh-cn_topic_0000001690107441_p4125181921717"></a><a name="zh-cn_topic_0000001690107441_p4125181921717"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p612571911172"><a name="zh-cn_topic_0000001690107441_p612571911172"></a><a name="zh-cn_topic_0000001690107441_p612571911172"></a>Amount of data received from each rank.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row6173151115173"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p13125319181717"><a name="zh-cn_topic_0000001690107441_p13125319181717"></a><a name="zh-cn_topic_0000001690107441_p13125319181717"></a>recvType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p11125141921712"><a name="zh-cn_topic_0000001690107441_p11125141921712"></a><a name="zh-cn_topic_0000001690107441_p11125141921712"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p1712581910177"><a name="zh-cn_topic_0000001690107441_p1712581910177"></a><a name="zh-cn_topic_0000001690107441_p1712581910177"></a>Received data type, defined in <a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000001690107441_p14891165052917"><a name="zh-cn_topic_0000001690107441_p14891165052917"></a><a name="zh-cn_topic_0000001690107441_p14891165052917"></a><span id="zh-cn_topic_0000001690107441_ph10615564290"><a name="zh-cn_topic_0000001690107441_ph10615564290"></a><a name="zh-cn_topic_0000001690107441_ph10615564290"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_2"></a>Data type supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000001690107441_p12838161092110"><a name="zh-cn_topic_0000001690107441_p12838161092110"></a><a name="zh-cn_topic_0000001690107441_p12838161092110"></a><span id="zh-cn_topic_0000001690107441_ph17071145182910"><a name="zh-cn_topic_0000001690107441_ph17071145182910"></a><a name="zh-cn_topic_0000001690107441_ph17071145182910"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215_1"></a>Data type supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row9173121111174"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p6125101981716"><a name="zh-cn_topic_0000001690107441_p6125101981716"></a><a name="zh-cn_topic_0000001690107441_p6125101981716"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p1312561917172"><a name="zh-cn_topic_0000001690107441_p1312561917172"></a><a name="zh-cn_topic_0000001690107441_p1312561917172"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p512551951715"><a name="zh-cn_topic_0000001690107441_p512551951715"></a><a name="zh-cn_topic_0000001690107441_p512551951715"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row2017331101714"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p712516198177"><a name="zh-cn_topic_0000001690107441_p712516198177"></a><a name="zh-cn_topic_0000001690107441_p712516198177"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p812551921714"><a name="zh-cn_topic_0000001690107441_p812551921714"></a><a name="zh-cn_topic_0000001690107441_p812551921714"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p41253199175"><a name="zh-cn_topic_0000001690107441_p41253199175"></a><a name="zh-cn_topic_0000001690107441_p41253199175"></a> Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001690107441_section12554172517195"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001690107441_section92549325194"></a>

-   The ranks must have the same `sendCount`, `sendType`, `recvCount`, and `recvType`.
-   The performance of the `AlltoAll` operation is related to the size of the buffer for storing shared data between NPUs. When the communication data size exceeds the buffer size, the performance deteriorates significantly. If the `AlltoAll` communication data size in the service is large, you are advised to increase the buffer size appropriately by setting environment variable `HCCL\_BUFFSIZE` to improve the communication performance.

## Call Example<a name="zh-cn_topic_0000001690107441_section204039211474"></a>

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

// Perform `AlltoAll` to send data of the same size to all ranks in the communicator and receive data of the same size from all ranks.
size_t perCount = count / rankSize;
HcclAlltoAll(sendBuf, perCount, HCCL_DATA_TYPE_FP32, recvBuf, perCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
