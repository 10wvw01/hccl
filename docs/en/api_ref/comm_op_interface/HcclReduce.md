# HcclReduce<a name="ZH-CN_TOPIC_0000002486832342"></a>

## Supported Products<a name="zh-cn_topic_0000001316510814_section10594071513"></a>

<a name="zh-cn_topic_0000001316510814_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001316510814_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001316510814_p1883113061818"><a name="zh-cn_topic_0000001316510814_p1883113061818"></a><a name="zh-cn_topic_0000001316510814_p1883113061818"></a><span id="zh-cn_topic_0000001316510814_ph20833205312295"><a name="zh-cn_topic_0000001316510814_ph20833205312295"></a><a name="zh-cn_topic_0000001316510814_ph20833205312295"></a>Product</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001316510814_p783113012187"><a name="zh-cn_topic_0000001316510814_p783113012187"></a><a name="zh-cn_topic_0000001316510814_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001316510814_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001316510814_p48327011813"><a name="zh-cn_topic_0000001316510814_p48327011813"></a><a name="zh-cn_topic_0000001316510814_p48327011813"></a><span id="zh-cn_topic_0000001316510814_ph583230201815"><a name="zh-cn_topic_0000001316510814_ph583230201815"></a><a name="zh-cn_topic_0000001316510814_ph583230201815"></a><term id="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001316510814_p7948163910184"><a name="zh-cn_topic_0000001316510814_p7948163910184"></a><a name="zh-cn_topic_0000001316510814_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001316510814_p14832120181815"><a name="zh-cn_topic_0000001316510814_p14832120181815"></a><a name="zh-cn_topic_0000001316510814_p14832120181815"></a><span id="zh-cn_topic_0000001316510814_ph1292674871116"><a name="zh-cn_topic_0000001316510814_ph1292674871116"></a><a name="zh-cn_topic_0000001316510814_ph1292674871116"></a><term id="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001316510814_p19948143911820"><a name="zh-cn_topic_0000001316510814_p19948143911820"></a><a name="zh-cn_topic_0000001316510814_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## Function Description<a name="zh-cn_topic_0000001316510814_section48254661"></a>

`Reduce` is a collective communication operator that performs reduction operations (such as sum, max, and min) on the data of all ranks and sends the result to the specified position on the root rank.

![]

## Prototype<a name="zh-cn_topic_0000001316510814_section57557412"></a>

```
HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001316510814_section31638772"></a>

<a name="zh-cn_topic_0000001316510814_table66592127"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001316510814_row61502840"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001316510814_p15674164"><a name="zh-cn_topic_0000001316510814_p15674164"></a><a name="zh-cn_topic_0000001316510814_p15674164"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001316510814_p61647805"><a name="zh-cn_topic_0000001316510814_p61647805"></a><a name="zh-cn_topic_0000001316510814_p61647805"></a>Input/Output</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001316510814_p27416314"><a name="zh-cn_topic_0000001316510814_p27416314"></a><a name="zh-cn_topic_0000001316510814_p27416314"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001316510814_row6128980"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p26685362"><a name="zh-cn_topic_0000001316510814_p26685362"></a><a name="zh-cn_topic_0000001316510814_p26685362"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p14030717"><a name="zh-cn_topic_0000001316510814_p14030717"></a><a name="zh-cn_topic_0000001316510814_p14030717"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p62746268"><a name="zh-cn_topic_0000001316510814_p62746268"></a><a name="zh-cn_topic_0000001316510814_p62746268"></a> source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row27845503"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p40893240"><a name="zh-cn_topic_0000001316510814_p40893240"></a><a name="zh-cn_topic_0000001316510814_p40893240"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p24018105"><a name="zh-cn_topic_0000001316510814_p24018105"></a><a name="zh-cn_topic_0000001316510814_p24018105"></a>Output</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p66418347"><a name="zh-cn_topic_0000001316510814_p66418347"></a><a name="zh-cn_topic_0000001316510814_p66418347"></a>Destination buffer address where the collective communication result is stored.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row60894213"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p33484259"><a name="zh-cn_topic_0000001316510814_p33484259"></a><a name="zh-cn_topic_0000001316510814_p33484259"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p27870469"><a name="zh-cn_topic_0000001316510814_p27870469"></a><a name="zh-cn_topic_0000001316510814_p27870469"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p42915540"><a name="zh-cn_topic_0000001316510814_p42915540"></a><a name="zh-cn_topic_0000001316510814_p42915540"></a>Number of data elements involved in the `Broadcast` operation. For example, `count = 1` indicates that one int32 data element is involved.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row50695543"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p12698356"><a name="zh-cn_topic_0000001316510814_p12698356"></a><a name="zh-cn_topic_0000001316510814_p12698356"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p21933905"><a name="zh-cn_topic_0000001316510814_p21933905"></a><a name="zh-cn_topic_0000001316510814_p21933905"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p97875916336"><a name="zh-cn_topic_0000001316510814_p97875916336"></a><a name="zh-cn_topic_0000001316510814_p97875916336"></a>Data type of the `Reduce` operation, defined in <a href="https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000001316510814_p143239448228"><a name="zh-cn_topic_0000001316510814_p143239448228"></a><a name="zh-cn_topic_0000001316510814_p143239448228"></a><span id="zh-cn_topic_0000001316510814_ph13754548217"><a name="zh-cn_topic_0000001316510814_ph13754548217"></a><a name="zh-cn_topic_0000001316510814_ph13754548217"></a><term id="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data type of Atlas A3 training/inference products</term></span>: int8, int16, int32, int64, float16, float32, bfp16</p>
<p id="zh-cn_topic_0000001316510814_p6332172033414"><a name="zh-cn_topic_0000001316510814_p6332172033414"></a><a name="zh-cn_topic_0000001316510814_p6332172033414"></a><span id="zh-cn_topic_0000001316510814_ph14880920154918"><a name="zh-cn_topic_0000001316510814_ph14880920154918"></a><a name="zh-cn_topic_0000001316510814_ph14880920154918"></a><term id="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term16184138172215"></a>Data type supported by Atlas A2 training/inference products</term></span>: int8, int16, int32, int64, float16, float32, bfp16 Note that the performance will deteriorate for the `int64` data type.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row17907308"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p41205809"><a name="zh-cn_topic_0000001316510814_p41205809"></a><a name="zh-cn_topic_0000001316510814_p41205809"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p49336210"><a name="zh-cn_topic_0000001316510814_p49336210"></a><a name="zh-cn_topic_0000001316510814_p49336210"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p36810105"><a name="zh-cn_topic_0000001316510814_p36810105"></a><a name="zh-cn_topic_0000001316510814_p36810105"></a>`Reduce` operation type. Currently, `sum`, `prod`, `max`, and `min` are supported.</p>
<div class="note" id="zh-cn_topic_0000001316510814_note4427520124313"><a name="zh-cn_topic_0000001316510814_note4427520124313"></a><a name="zh-cn_topic_0000001316510814_note4427520124313"></a><span class="notetitle">Note: </span><div class="notebody"><p id="zh-cn_topic_0000001316510814_p9984151202012"><a name="zh-cn_topic_0000001316510814_p9984151202012"></a><a name="zh-cn_topic_0000001316510814_p9984151202012"></a>For <span id="zh-cn_topic_0000001316510814_ph79242619219"><a name="zh-cn_topic_0000001316510814_ph79242619219"></a><a name="zh-cn_topic_0000001316510814_ph79242619219"></a><term id="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 training/inference products</term></span>, the `prod` operation does not support data type `int16` or `bfp16` in the current version.</p>
<p id="zh-cn_topic_0000001316510814_p10731124313342"><a name="zh-cn_topic_0000001316510814_p10731124313342"></a><a name="zh-cn_topic_0000001316510814_p10731124313342"></a>For <span id="zh-cn_topic_0000001316510814_ph49172713419"><a name="zh-cn_topic_0000001316510814_ph49172713419"></a><a name="zh-cn_topic_0000001316510814_ph49172713419"></a><term id="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000001316510814_zh-cn_topic_0000001312391781_term16184138172215_1"></a>Atlas A2 training/inference products</term></span>, the `prod` operation does not support data type `int16` or `bfp16`.</p>
</div></div>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row532453124917"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p3324183174914"><a name="zh-cn_topic_0000001316510814_p3324183174914"></a><a name="zh-cn_topic_0000001316510814_p3324183174914"></a>root</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p7324133134913"><a name="zh-cn_topic_0000001316510814_p7324133134913"></a><a name="zh-cn_topic_0000001316510814_p7324133134913"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p932418344915"><a name="zh-cn_topic_0000001316510814_p932418344915"></a><a name="zh-cn_topic_0000001316510814_p932418344915"></a>Rank ID of the reduce root.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row62855489"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p58129833"><a name="zh-cn_topic_0000001316510814_p58129833"></a><a name="zh-cn_topic_0000001316510814_p58129833"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p10896009"><a name="zh-cn_topic_0000001316510814_p10896009"></a><a name="zh-cn_topic_0000001316510814_p10896009"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p10161546"><a name="zh-cn_topic_0000001316510814_p10161546"></a><a name="zh-cn_topic_0000001316510814_p10161546"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001316510814_row24345054"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001316510814_p25792371"><a name="zh-cn_topic_0000001316510814_p25792371"></a><a name="zh-cn_topic_0000001316510814_p25792371"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001316510814_p8807292"><a name="zh-cn_topic_0000001316510814_p8807292"></a><a name="zh-cn_topic_0000001316510814_p8807292"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001316510814_p42302050"><a name="zh-cn_topic_0000001316510814_p42302050"></a><a name="zh-cn_topic_0000001316510814_p42302050"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001316510814_section16313497"></a>

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001316510814_section12603749"></a>

All ranks must have the same `count`, `dataType`, and `op`.

## Call Example<a name="zh-cn_topic_0000001316510814_section204039211474"></a>

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

// Perform Reduce to add sendBuf of all ranks at the corresponding location and then send the result to recvBuf of the root rank.
HcclReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, rootRank, hcclComm, stream);
// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
