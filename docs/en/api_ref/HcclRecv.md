# HcclRecv<a name="ZH-CN_TOPIC_0000002486833322"></a>

## Supported Products<a name="zh-cn_topic_0000001264921402_section10594071513"></a>

<a name="zh-cn_topic_0000001264921402_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001264921402_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001264921402_p1883113061818"><a name="zh-cn_topic_0000001264921402_p1883113061818"></a><a name="zh-cn_topic_0000001264921402_p1883113061818"></a><span id="zh-cn_topic_0000001264921402_ph20833205312295"><a name="zh-cn_topic_0000001264921402_ph20833205312295"></a><a name="zh-cn_topic_0000001264921402_ph20833205312295"></a> Product </span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001264921402_p783113012187"><a name="zh-cn_topic_0000001264921402_p783113012187"></a><a name="zh-cn_topic_0000001264921402_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001264921402_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001264921402_p48327011813"><a name="zh-cn_topic_0000001264921402_p48327011813"></a><a name="zh-cn_topic_0000001264921402_p48327011813"></a><span id="zh-cn_topic_0000001264921402_ph583230201815"><a name="zh-cn_topic_0000001264921402_ph583230201815"></a><a name="zh-cn_topic_0000001264921402_ph583230201815"></a><term id="zh-cn_topic_0000001264921402_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training /inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001264921402_p7948163910184"><a name="zh-cn_topic_0000001264921402_p7948163910184"></a><a name="zh-cn_topic_0000001264921402_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001264921402_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001264921402_p14832120181815"><a name="zh-cn_topic_0000001264921402_p14832120181815"></a><a name="zh-cn_topic_0000001264921402_p14832120181815"></a><span id="zh-cn_topic_0000001264921402_ph1292674871116"><a name="zh-cn_topic_0000001264921402_ph1292674871116"></a><a name="zh-cn_topic_0000001264921402_ph1292674871116"></a><term id="zh-cn_topic_0000001264921402_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001264921402_p19948143911820"><a name="zh-cn_topic_0000001264921402_p19948143911820"></a><a name="zh-cn_topic_0000001264921402_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training/inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000001264921402_section115673714311"></a>

Receive data from the source rank to the specified location on the current rank.

## Prototype<a name="zh-cn_topic_0000001264921402_section165671571131"></a>

```
HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001264921402_section145681471310"></a>

<a name="zh-cn_topic_0000001264921402_table0576473316"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001264921402_row1060511716320"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001264921402_p146051071139"><a name="zh-cn_topic_0000001264921402_p146051071139"></a><a name="zh-cn_topic_0000001264921402_p146051071139"></a> Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001264921402_p1160527939"><a name="zh-cn_topic_0000001264921402_p1160527939"></a><a name="zh-cn_topic_0000001264921402_p1160527939"></a> Input/Output </p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001264921402_p86058714320"><a name="zh-cn_topic_0000001264921402_p86058714320"></a><a name="zh-cn_topic_0000001264921402_p86058714320"></a> Description </p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001264921402_row166054719318"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264921402_p106051976316"><a name="zh-cn_topic_0000001264921402_p106051976316"></a><a name="zh-cn_topic_0000001264921402_p106051976316"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264921402_p560511719312"><a name="zh-cn_topic_0000001264921402_p560511719312"></a><a name="zh-cn_topic_0000001264921402_p560511719312"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264921402_p1760577138"><a name="zh-cn_topic_0000001264921402_p1760577138"></a><a name="zh-cn_topic_0000001264921402_p1760577138"></a> Data receive buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001264921402_row460577337"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264921402_p26051971534"><a name="zh-cn_topic_0000001264921402_p26051971534"></a><a name="zh-cn_topic_0000001264921402_p26051971534"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264921402_p2605973319"><a name="zh-cn_topic_0000001264921402_p2605973319"></a><a name="zh-cn_topic_0000001264921402_p2605973319"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264921402_p1160510719318"><a name="zh-cn_topic_0000001264921402_p1160510719318"></a><a name="zh-cn_topic_0000001264921402_p1160510719318"></a>Number of data element received.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001264921402_row156051072036"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264921402_p460557831"><a name="zh-cn_topic_0000001264921402_p460557831"></a><a name="zh-cn_topic_0000001264921402_p460557831"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264921402_p2605187530"><a name="zh-cn_topic_0000001264921402_p2605187530"></a><a name="zh-cn_topic_0000001264921402_p2605187530"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264921402_p15650182941610"><a name="zh-cn_topic_0000001264921402_p15650182941610"></a><a name="zh-cn_topic_0000001264921402_p15650182941610"></a>Received data type, defined in <a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000001264921402_p39291313119"><a name="zh-cn_topic_0000001264921402_p39291313119"></a><a name="zh-cn_topic_0000001264921402_p39291313119"></a><span id="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_ph13754548217"><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_ph13754548217"></a><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_ph13754548217"></a><term id="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225"></a> Data types supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000001264921402_p12283142716193"><a name="zh-cn_topic_0000001264921402_p12283142716193"></a><a name="zh-cn_topic_0000001264921402_p12283142716193"></a> <span id="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_ph14880920154918"><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_ph14880920154918"></a><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_ph14880920154918"></a><term id="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001264921402_zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term16184138172215"></a>Data types supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001264921402_row206057717312"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264921402_p1605673319"><a name="zh-cn_topic_0000001264921402_p1605673319"></a><a name="zh-cn_topic_0000001264921402_p1605673319"></a>srcRank</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264921402_p7605177635"><a name="zh-cn_topic_0000001264921402_p7605177635"></a><a name="zh-cn_topic_0000001264921402_p7605177635"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264921402_p7605571034"><a name="zh-cn_topic_0000001264921402_p7605571034"></a><a name="zh-cn_topic_0000001264921402_p7605571034"></a>ID of the source rank in the communicator.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001264921402_row146051372315"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264921402_p126053714319"><a name="zh-cn_topic_0000001264921402_p126053714319"></a><a name="zh-cn_topic_0000001264921402_p126053714319"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264921402_p26051871637"><a name="zh-cn_topic_0000001264921402_p26051871637"></a><a name="zh-cn_topic_0000001264921402_p26051871637"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264921402_p17605774316"><a name="zh-cn_topic_0000001264921402_p17605774316"></a><a name="zh-cn_topic_0000001264921402_p17605774316"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001264921402_row14605137334"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264921402_p196051071736"><a name="zh-cn_topic_0000001264921402_p196051071736"></a><a name="zh-cn_topic_0000001264921402_p196051071736"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264921402_p1605107937"><a name="zh-cn_topic_0000001264921402_p1605107937"></a><a name="zh-cn_topic_0000001264921402_p1605107937"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264921402_p0605274311"><a name="zh-cn_topic_0000001264921402_p0605274311"></a><a name="zh-cn_topic_0000001264921402_p0605274311"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001264921402_section14576971837"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001264921402_section5471426246"></a>

The `HcclSend` and `HcclRecv` APIs must be used in pairs. That is, after a process calls the `HcclSend` API, it can call the next API only after the corresponding `HcclRecv` API receives data, as shown in the following figure.

![](figures/zh-cn_image_0000001532063748.png)

## Call Example<a name="zh-cn_topic_0000001264921402_section204039211474"></a>

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task flow.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform the Send/Recv operation. NPUs 0, 2, 4, and 6 send data, and NPUs 1, 3, 5, and 7 receive data.
// The HcclSend and HcclRecv APIs must be used in pairs.
if (deviceId % 2 == 0) {
    // Allocate the device memory for storing the input data.
    aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Initialize host input buffers
    aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // Perform the Send operation.
    HcclSend(sendBuf, count, HCCL_DATA_TYPE_FP32, deviceId + 1, hcclComm, stream);
} else {
    // Allocate the device memory for receiving data.
    aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Perform the Recv operation.
    HcclRecv(recvBuf, count, HCCL_DATA_TYPE_FP32, deviceId - 1, hcclComm, stream);
}

// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
