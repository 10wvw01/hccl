# HcclAllReduce<a name="ZH-CN_TOPIC_0000002518992195"></a>

## Supported Products<a name="zh-cn_topic_0000001312641237_section161778316247"></a>

<a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312641237_p1883113061818"><a name="zh-cn_topic_0000001312641237_p1883113061818"></a><a name="zh-cn_topic_0000001312641237_p1883113061818"></a><span id="zh-cn_topic_0000001312641237_ph20833205312295"><a name="zh-cn_topic_0000001312641237_ph20833205312295"></a><a name="zh-cn_topic_0000001312641237_ph20833205312295"></a>Product</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p783113012187"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p783113012187"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312641237_p48327011813"><a name="zh-cn_topic_0000001312641237_p48327011813"></a><a name="zh-cn_topic_0000001312641237_p48327011813"></a><span id="zh-cn_topic_0000001312641237_ph583230201815"><a name="zh-cn_topic_0000001312641237_ph583230201815"></a><a name="zh-cn_topic_0000001312641237_ph583230201815"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p7948163910184"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p7948163910184"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312641237_p14832120181815"><a name="zh-cn_topic_0000001312641237_p14832120181815"></a><a name="zh-cn_topic_0000001312641237_p14832120181815"></a><span id="zh-cn_topic_0000001312641237_ph1292674871116"><a name="zh-cn_topic_0000001312641237_ph1292674871116"></a><a name="zh-cn_topic_0000001312641237_ph1292674871116"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p19948143911820"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p19948143911820"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training /inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000001312641237_section48254661"></a>

`AllReduce` is a collective communication operator that performs a reduction operation (such as sum, max, or min) across all ranks in the communicator and then scatters the results to each rank's output buffer. The reduction operation type is specified by the `op` parameter.

![](figures/allreduce.png)

## Prototype<a name="zh-cn_topic_0000001312641237_section57557412"></a>

```
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001312641237_section31638772"></a>

<a name="zh-cn_topic_0000001312641237_table66592127"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312641237_row61502840"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312641237_p15674164"><a name="zh-cn_topic_0000001312641237_p15674164"></a><a name="zh-cn_topic_0000001312641237_p15674164"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.03%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312641237_p61647805"><a name="zh-cn_topic_0000001312641237_p61647805"></a><a name="zh-cn_topic_0000001312641237_p61647805"></a>Input/Output</p>
</th>
<th class="cellrowborder" valign="top" width="62.77%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312641237_p27416314"><a name="zh-cn_topic_0000001312641237_p27416314"></a><a name="zh-cn_topic_0000001312641237_p27416314"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312641237_row6128980"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p26685362"><a name="zh-cn_topic_0000001312641237_p26685362"></a><a name="zh-cn_topic_0000001312641237_p26685362"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p14030717"><a name="zh-cn_topic_0000001312641237_p14030717"></a><a name="zh-cn_topic_0000001312641237_p14030717"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p62746268"><a name="zh-cn_topic_0000001312641237_p62746268"></a><a name="zh-cn_topic_0000001312641237_p62746268"></a>Source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row27845503"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p40893240"><a name="zh-cn_topic_0000001312641237_p40893240"></a><a name="zh-cn_topic_0000001312641237_p40893240"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p24018105"><a name="zh-cn_topic_0000001312641237_p24018105"></a><a name="zh-cn_topic_0000001312641237_p24018105"></a>Output</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p66418347"><a name="zh-cn_topic_0000001312641237_p66418347"></a><a name="zh-cn_topic_0000001312641237_p66418347"></a>Destination buffer address where the collective communication result is stored.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row60894213"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p33484259"><a name="zh-cn_topic_0000001312641237_p33484259"></a><a name="zh-cn_topic_0000001312641237_p33484259"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p27870469"><a name="zh-cn_topic_0000001312641237_p27870469"></a><a name="zh-cn_topic_0000001312641237_p27870469"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p42915540"><a name="zh-cn_topic_0000001312641237_p42915540"></a><a name="zh-cn_topic_0000001312641237_p42915540"></a>Number of data elements participating in the `AllReduce` operation. For example, if only one `int32` element is involved, then `count = 1`.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row50695543"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p12698356"><a name="zh-cn_topic_0000001312641237_p12698356"></a><a name="zh-cn_topic_0000001312641237_p12698356"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p21933905"><a name="zh-cn_topic_0000001312641237_p21933905"></a><a name="zh-cn_topic_0000001312641237_p21933905"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p1532218361093"><a name="zh-cn_topic_0000001312641237_p1532218361093"></a><a name="zh-cn_topic_0000001312641237_p1532218361093"></a>Data type of the `allreduce` operation, in <a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000001312641237_p1516617265207"><a name="zh-cn_topic_0000001312641237_p1516617265207"></a><a name="zh-cn_topic_0000001312641237_p1516617265207"></a><span id="zh-cn_topic_0000001312641237_ph13754548217"><a name="zh-cn_topic_0000001312641237_ph13754548217"></a><a name="zh-cn_topic_0000001312641237_ph13754548217"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data type supported by Atlas A3 training/inference products</term></span>: int8, int16, int32, int64, float16, float32, bfp16</p>
<p id="zh-cn_topic_0000001312641237_p195289510913"><a name="zh-cn_topic_0000001312641237_p195289510913"></a><a name="zh-cn_topic_0000001312641237_p195289510913"></a><span id="zh-cn_topic_0000001312641237_ph14880920154918"><a name="zh-cn_topic_0000001312641237_ph14880920154918"></a><a name="zh-cn_topic_0000001312641237_ph14880920154918"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215"></a>Data type supported by Atlas A2 training/inference products</term></span>: int8, int16, int32, int64, float16, float32, bfp16 Note that the performance will deteriorate for the `int64` data type.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row17907308"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p41205809"><a name="zh-cn_topic_0000001312641237_p41205809"></a><a name="zh-cn_topic_0000001312641237_p41205809"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p49336210"><a name="zh-cn_topic_0000001312641237_p49336210"></a><a name="zh-cn_topic_0000001312641237_p49336210"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p36810105"><a name="zh-cn_topic_0000001312641237_p36810105"></a><a name="zh-cn_topic_0000001312641237_p36810105"></a>`Reduce` operation type. Currently, `sum`, `prod`, `max`, and `min` are supported.</p>
<div class="note" id="zh-cn_topic_0000001312641237_note5731134315341"><a name="zh-cn_topic_0000001312641237_note5731134315341"></a><a name="zh-cn_topic_0000001312641237_note5731134315341"></a><span class="notetitle">Note: </span><div class="notebody"><p id="zh-cn_topic_0000001312641237_p9984151202012"><a name="zh-cn_topic_0000001312641237_p9984151202012"></a><a name="zh-cn_topic_0000001312641237_p9984151202012"></a>For <span id="zh-cn_topic_0000001312641237_ph79242619219"><a name="zh-cn_topic_0000001312641237_ph79242619219"></a><a name="zh-cn_topic_0000001312641237_ph79242619219"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 training/inference products, </term></span>, the `prod` operation does not support data types `int16` and `bfp16`.</p>
<p id="zh-cn_topic_0000001312641237_p10731124313342"><a name="zh-cn_topic_0000001312641237_p10731124313342"></a><a name="zh-cn_topic_0000001312641237_p10731124313342"></a>For <span id="zh-cn_topic_0000001312641237_ph49172713419"><a name="zh-cn_topic_0000001312641237_ph49172713419"></a><a name="zh-cn_topic_0000001312641237_ph49172713419"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215_1"></a>Atlas A2 training/inference products</term></span>, the `prod` operation does not support data types `int16` and `bfp16`.</p>
</div></div>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row62855489"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p58129833"><a name="zh-cn_topic_0000001312641237_p58129833"></a><a name="zh-cn_topic_0000001312641237_p58129833"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p10896009"><a name="zh-cn_topic_0000001312641237_p10896009"></a><a name="zh-cn_topic_0000001312641237_p10896009"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p10161546"><a name="zh-cn_topic_0000001312641237_p10161546"></a><a name="zh-cn_topic_0000001312641237_p10161546"></a>Communicator in which collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row24345054"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p25792371"><a name="zh-cn_topic_0000001312641237_p25792371"></a><a name="zh-cn_topic_0000001312641237_p25792371"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p8807292"><a name="zh-cn_topic_0000001312641237_p8807292"></a><a name="zh-cn_topic_0000001312641237_p8807292"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p42302050"><a name="zh-cn_topic_0000001312641237_p42302050"></a><a name="zh-cn_topic_0000001312641237_p42302050"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001312641237_section16313497"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001312641237_section12603749"></a>

-   All ranks must have the same `count`, `dataType`, and `op`.
-   Each rank has only one input.

## Call Example<a name="zh-cn_topic_0000001312641237_section204039211474"></a>

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

// Execute AllReduce to sum input data of all ranks in the communicator and send the result to the output buffer of all ranks.
HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
