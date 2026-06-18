# HcclAlltoAllVC<a name="ZH-CN_TOPIC_0000002519072191"></a>

## Supported Products<a name="zh-cn_topic_0000002417514657_section10594071513"></a>

<a name="zh-cn_topic_0000002417514657_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002417514657_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002417514657_p1883113061818"><a name="zh-cn_topic_0000002417514657_p1883113061818"></a><a name="zh-cn_topic_0000002417514657_p1883113061818"></a><span id="zh-cn_topic_0000002417514657_ph20833205312295"><a name="zh-cn_topic_0000002417514657_ph20833205312295"></a><a name="zh-cn_topic_0000002417514657_ph20833205312295"></a> Product </span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002417514657_p783113012187"><a name="zh-cn_topic_0000002417514657_p783113012187"></a><a name="zh-cn_topic_0000002417514657_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002417514657_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002417514657_p48327011813"><a name="zh-cn_topic_0000002417514657_p48327011813"></a><a name="zh-cn_topic_0000002417514657_p48327011813"></a><span id="zh-cn_topic_0000002417514657_ph583230201815"><a name="zh-cn_topic_0000002417514657_ph583230201815"></a><a name="zh-cn_topic_0000002417514657_ph583230201815"></a><term id="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002417514657_p7948163910184"><a name="zh-cn_topic_0000002417514657_p7948163910184"></a><a name="zh-cn_topic_0000002417514657_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002417514657_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002417514657_p14832120181815"><a name="zh-cn_topic_0000002417514657_p14832120181815"></a><a name="zh-cn_topic_0000002417514657_p14832120181815"></a><span id="zh-cn_topic_0000002417514657_ph1292674871116"><a name="zh-cn_topic_0000002417514657_ph1292674871116"></a><a name="zh-cn_topic_0000002417514657_ph1292674871116"></a><term id="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002417514657_p19948143911820"><a name="zh-cn_topic_0000002417514657_p19948143911820"></a><a name="zh-cn_topic_0000002417514657_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training/inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000002417514657_section37208511199"></a>

`AlltoAllVC` is a collective communication operator that sends data (with customizable size) to all ranks in the communicator and receives data from all ranks. Unlike `AlltoAllV`, `AlltoAllVC` uses the argument `sendCountMatrix` to pass receive and send parameters of all ranks.

![]()

## Prototype<a name="zh-cn_topic_0000002417514657_section35919731916"></a>

```
HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000002417514657_section2586134311199"></a>

<a name="zh-cn_topic_0000002417514657_table0576473316"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002417514657_row1060511716320"><th class="cellrowborder" valign="top" width="20.18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002417514657_p146051071139"><a name="zh-cn_topic_0000002417514657_p146051071139"></a><a name="zh-cn_topic_0000002417514657_p146051071139"></a> Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002417514657_p1160527939"><a name="zh-cn_topic_0000002417514657_p1160527939"></a><a name="zh-cn_topic_0000002417514657_p1160527939"></a> Input/Output </p>
</th>
<th class="cellrowborder" valign="top" width="62.64999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002417514657_p86058714320"><a name="zh-cn_topic_0000002417514657_p86058714320"></a><a name="zh-cn_topic_0000002417514657_p86058714320"></a> Description </p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002417514657_row166054719318"><td class="cellrowborder" valign="top" width="20.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002417514657_p111231019101719"><a name="zh-cn_topic_0000002417514657_p111231019101719"></a><a name="zh-cn_topic_0000002417514657_p111231019101719"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002417514657_p51231519111711"><a name="zh-cn_topic_0000002417514657_p51231519111711"></a><a name="zh-cn_topic_0000002417514657_p51231519111711"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002417514657_p612301916172"><a name="zh-cn_topic_0000002417514657_p612301916172"></a><a name="zh-cn_topic_0000002417514657_p612301916172"></a> Source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002417514657_row460577337"><td class="cellrowborder" valign="top" width="20.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002417514657_p412311914178"><a name="zh-cn_topic_0000002417514657_p412311914178"></a><a name="zh-cn_topic_0000002417514657_p412311914178"></a>sendCountMatrix</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002417514657_p01231219171717"><a name="zh-cn_topic_0000002417514657_p01231219171717"></a><a name="zh-cn_topic_0000002417514657_p01231219171717"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002417514657_p5123171961713"><a name="zh-cn_topic_0000002417514657_p5123171961713"></a><a name="zh-cn_topic_0000002417514657_p5123171961713"></a>Two-dimensional `uint64` array, representing the size of data sent. The array shape is `[rankSize][rankSize]`. `sendCountMatrix[i][j] = n` indicates that the size of data sent from rank i to rank j is n.</p>
<p id="zh-cn_topic_0000002417514657_p17301512161515"><a name="zh-cn_topic_0000002417514657_p17301512161515"></a><a name="zh-cn_topic_0000002417514657_p17301512161515"></a> For example, if `sendType` is `float32`, `sendCountMatrix[i][j] = n` indicates that rank `i` sends `n` float32 data elements to rank `j`.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002417514657_row206057717312"><td class="cellrowborder" valign="top" width="20.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002417514657_p81241419101717"><a name="zh-cn_topic_0000002417514657_p81241419101717"></a><a name="zh-cn_topic_0000002417514657_p81241419101717"></a>sendType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002417514657_p151241419151715"><a name="zh-cn_topic_0000002417514657_p151241419151715"></a><a name="zh-cn_topic_0000002417514657_p151241419151715"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002417514657_p5123171961713"><a name="zh-cn_topic_0000002417514657_p5123171961713"></a><a name="zh-cn_topic_0000002417514657_p5123171961713"></a>Sent data type, defined in<a href="https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000002417514657_p526014481278"><a name="zh-cn_topic_0000002417514657_p526014481278"></a><a name="zh-cn_topic_0000002417514657_p526014481278"></a> <span id="zh-cn_topic_0000002417514657_ph13754548217"><a name="zh-cn_topic_0000002417514657_ph13754548217"></a><a name="zh-cn_topic_0000002417514657_ph13754548217"></a><term id="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data types supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000002417514657_p12283142716193"><a name="zh-cn_topic_0000002417514657_p12283142716193"></a><a name="zh-cn_topic_0000002417514657_p12283142716193"></a> <span id="zh-cn_topic_0000002417514657_ph14880920154918"><a name="zh-cn_topic_0000002417514657_ph14880920154918"></a><a name="zh-cn_topic_0000002417514657_ph14880920154918"></a><term id="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term16184138172215"></a>>Data types supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002417514657_row146051372315"><td class="cellrowborder" valign="top" width="20.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002417514657_p512431910174"><a name="zh-cn_topic_0000002417514657_p512431910174"></a><a name="zh-cn_topic_0000002417514657_p512431910174"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002417514657_p151241419131716"><a name="zh-cn_topic_0000002417514657_p151241419131716"></a><a name="zh-cn_topic_0000002417514657_p151241419131716"></a> Output </p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002417514657_p161241219201719"><a name="zh-cn_topic_0000002417514657_p161241219201719"></a><a name="zh-cn_topic_0000002417514657_p161241219201719"></a>Destination buffer address where the collective communication result is stored.</p>
<p id="zh-cn_topic_0000002417514657_p57711733112312"><a name="zh-cn_topic_0000002417514657_p57711733112312"></a><a name="zh-cn_topic_0000002417514657_p57711733112312"></a>The addresses configured for `recvBuf` and `sendBuf` must not be the same.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002417514657_row6173151115173"><td class="cellrowborder" valign="top" width="20.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002417514657_p13125319181717"><a name="zh-cn_topic_0000002417514657_p13125319181717"></a><a name="zh-cn_topic_0000002417514657_p13125319181717"></a>recvType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002417514657_p11125141921712"><a name="zh-cn_topic_0000002417514657_p11125141921712"></a><a name="zh-cn_topic_0000002417514657_p11125141921712"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002417514657_p85711315720"><a name="zh-cn_topic_0000002417514657_p85711315720"></a><a name="zh-cn_topic_0000002417514657_p85711315720"></a>Received data type, defined in <a href="https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000002417514657_p2687413112810"><a name="zh-cn_topic_0000002417514657_p2687413112810"></a><a name="zh-cn_topic_0000002417514657_p2687413112810"></a> <span id="zh-cn_topic_0000002417514657_ph11201161819282"><a name="zh-cn_topic_0000002417514657_ph11201161819282"></a><a name="zh-cn_topic_0000002417514657_ph11201161819282"></a><term id="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term1253731311225_2"></a>Data types supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000002417514657_p1075482252214"><a name="zh-cn_topic_0000002417514657_p1075482252214"></a><a name="zh-cn_topic_0000002417514657_p1075482252214"></a> <span id="zh-cn_topic_0000002417514657_ph161710422273"><a name="zh-cn_topic_0000002417514657_ph161710422273"></a><a name="zh-cn_topic_0000002417514657_ph161710422273"></a><term id="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000002417514657_zh-cn_topic_0000001312391781_term16184138172215_1"></a>Data types supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002417514657_row9173121111174"><td class="cellrowborder" valign="top" width="20.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002417514657_p6125101981716"><a name="zh-cn_topic_0000002417514657_p6125101981716"></a><a name="zh-cn_topic_0000002417514657_p6125101981716"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002417514657_p1312561917172"><a name="zh-cn_topic_0000002417514657_p1312561917172"></a><a name="zh-cn_topic_0000002417514657_p1312561917172"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002417514657_p512551951715"><a name="zh-cn_topic_0000002417514657_p512551951715"></a><a name="zh-cn_topic_0000002417514657_p512551951715"></a>Communicator in which collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002417514657_row2017331101714"><td class="cellrowborder" valign="top" width="20.18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002417514657_p712516198177"><a name="zh-cn_topic_0000002417514657_p712516198177"></a><a name="zh-cn_topic_0000002417514657_p712516198177"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002417514657_p812551921714"><a name="zh-cn_topic_0000002417514657_p812551921714"></a><a name="zh-cn_topic_0000002417514657_p812551921714"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002417514657_p41253199175"><a name="zh-cn_topic_0000002417514657_p41253199175"></a><a name="zh-cn_topic_0000002417514657_p41253199175"></a>Stream used by the current rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000002417514657_section12554172517195"></a>

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000002417514657_section92549325194"></a>

The performance of the `AlltoAllVC` operation is related to the size of the buffer for storing shared data between NPUs. When the communication data size exceeds the buffer size, the performance deteriorates significantly. If the `AlltoAllVC` communication data size in the service is large, you are advised to increase the buffer size appropriately by setting environment variable `HCCL\_BUFFSIZE` to improve the communication performance.

## Call Example<a name="zh-cn_topic_0000002417514657_section204039211474"></a>

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

// Set the size of data to be sent and received. The size of data to be sent is the same as that of data to be received.
std::vector&lt;uint64_t> sendCountMatrix(rankSize * rankSize);
for (uint32_t i = 0; i &lt; rankSize; ++i) {
    for (uint32_t j = 0; j &lt; rankSize; ++j) {
        sendCountMatrix[i * rankSize + j] = count / rankSize;
    }
}

// Perform AlltoAllVC to send data of the same size to all ranks in the communicator and receive data of the same size from all ranks. The data size can be customized.
HcclAlltoAllVC(sendBuf, sendCountMatrix.data(), HCCL_DATA_TYPE_FP32, recvBuf, HCCL_DATA_TYPE_FP32, hcclComm, stream);

// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
