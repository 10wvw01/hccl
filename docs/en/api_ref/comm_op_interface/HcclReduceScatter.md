# HcclReduceScatter<a name="ZH-CN_TOPIC_0000002518992197"></a>

## Supported Products<a name="zh-cn_topic_0000001265400114_section10594071513"></a>

<a name="zh-cn_topic_0000001265400114_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265400114_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265400114_p1883113061818"><a name="zh-cn_topic_0000001265400114_p1883113061818"></a><a name="zh-cn_topic_0000001265400114_p1883113061818"></a><span id="zh-cn_topic_0000001265400114_ph20833205312295"><a name="zh-cn_topic_0000001265400114_ph20833205312295"></a><a name="zh-cn_topic_0000001265400114_ph20833205312295"></a> Product </span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265400114_p783113012187"><a name="zh-cn_topic_0000001265400114_p783113012187"></a><a name="zh-cn_topic_0000001265400114_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265400114_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265400114_p48327011813"><a name="zh-cn_topic_0000001265400114_p48327011813"></a><a name="zh-cn_topic_0000001265400114_p48327011813"></a><span id="zh-cn_topic_0000001265400114_ph583230201815"><a name="zh-cn_topic_0000001265400114_ph583230201815"></a><a name="zh-cn_topic_0000001265400114_ph583230201815"></a><term id="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265400114_p7948163910184"><a name="zh-cn_topic_0000001265400114_p7948163910184"></a><a name="zh-cn_topic_0000001265400114_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265400114_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265400114_p14832120181815"><a name="zh-cn_topic_0000001265400114_p14832120181815"></a><a name="zh-cn_topic_0000001265400114_p14832120181815"></a><span id="zh-cn_topic_0000001265400114_ph1292674871116"><a name="zh-cn_topic_0000001265400114_ph1292674871116"></a><a name="zh-cn_topic_0000001265400114_ph1292674871116"></a><term id="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265400114_p19948143911820"><a name="zh-cn_topic_0000001265400114_p19948143911820"></a><a name="zh-cn_topic_0000001265400114_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training/inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000001265400114_section31291646"></a>

Perform the sum operation (or other reduction operations) on the inputs of all ranks, and then distribute the result evenly to the output buffers of ranks according to the rank IDs. Each process receives `1/rank\_size` portion of data from other processes for reduction.

As shown in the following figure, there are four ranks: rank0, rank1, rank2, and rank3. The input data of each rank is divided into four parts. Each process obtains 1/4 of the data of each rank to perform the sum operation (or other operations) and sends the result to the output buffer.

![]()

## Prototype<a name="zh-cn_topic_0000001265400114_section18389930"></a>

```
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001265400114_section13189358"></a>

<a name="zh-cn_topic_0000001265400114_table24749807"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265400114_row60665573"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265400114_p14964341"><a name="zh-cn_topic_0000001265400114_p14964341"></a><a name="zh-cn_topic_0000001265400114_p14964341"></a> Parameter </p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265400114_p4152081"><a name="zh-cn_topic_0000001265400114_p4152081"></a><a name="zh-cn_topic_0000001265400114_p4152081"></a> Input/Output </p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265400114_p774306"><a name="zh-cn_topic_0000001265400114_p774306"></a><a name="zh-cn_topic_0000001265400114_p774306"></a> Description </p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265400114_row62718864"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265400114_p47063234"><a name="zh-cn_topic_0000001265400114_p47063234"></a><a name="zh-cn_topic_0000001265400114_p47063234"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265400114_p54025633"><a name="zh-cn_topic_0000001265400114_p54025633"></a><a name="zh-cn_topic_0000001265400114_p54025633"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265400114_p14000148"><a name="zh-cn_topic_0000001265400114_p14000148"></a><a name="zh-cn_topic_0000001265400114_p14000148"></a>Address of the source data buffer.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265400114_row58892473"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265400114_p5560998"><a name="zh-cn_topic_0000001265400114_p5560998"></a><a name="zh-cn_topic_0000001265400114_p5560998"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265400114_p47787675"><a name="zh-cn_topic_0000001265400114_p47787675"></a><a name="zh-cn_topic_0000001265400114_p47787675"></a> Output </p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265400114_p45596481"><a name="zh-cn_topic_0000001265400114_p45596481"></a><a name="zh-cn_topic_0000001265400114_p45596481"></a>Destination buffer address where the collective communication result is stored.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265400114_row7715150"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265400114_p20947391"><a name="zh-cn_topic_0000001265400114_p20947391"></a><a name="zh-cn_topic_0000001265400114_p20947391"></a>recvCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265400114_p19017142"><a name="zh-cn_topic_0000001265400114_p19017142"></a><a name="zh-cn_topic_0000001265400114_p19017142"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265400114_p63993496"><a name="zh-cn_topic_0000001265400114_p63993496"></a><a name="zh-cn_topic_0000001265400114_p63993496"></a>`recvBuf` data size involved in the `ReduceScatter` operation. The size of `sendBuf` data is `recvCount` × rank size.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265400114_row39070558"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265400114_p10598606"><a name="zh-cn_topic_0000001265400114_p10598606"></a><a name="zh-cn_topic_0000001265400114_p10598606"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265400114_p53180767"><a name="zh-cn_topic_0000001265400114_p53180767"></a><a name="zh-cn_topic_0000001265400114_p53180767"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265400114_p1114114215138"><a name="zh-cn_topic_0000001265400114_p1114114215138"></a><a name="zh-cn_topic_0000001265400114_p1114114215138"></a>Data type of the `ReduceScatter` operation, defined in HcclDataType.</p>
<p id="zh-cn_topic_0000001265400114_p261131302211"><a name="zh-cn_topic_0000001265400114_p261131302211"></a><a name="zh-cn_topic_0000001265400114_p261131302211"></a> <span id="zh-cn_topic_0000001265400114_ph13754548217"><a name="zh-cn_topic_0000001265400114_ph13754548217"></a><a name="zh-cn_topic_0000001265400114_ph13754548217"></a><term id="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data types supported by Atlas A3 training/inference products</term></span>: int8, int16, int32, int64, float16, float32, bfp16</p>
<p id="zh-cn_topic_0000001265400114_p16921716151418"><a name="zh-cn_topic_0000001265400114_p16921716151418"></a><a name="zh-cn_topic_0000001265400114_p16921716151418"></a> <span id="zh-cn_topic_0000001265400114_ph14880920154918"><a name="zh-cn_topic_0000001265400114_ph14880920154918"></a><a name="zh-cn_topic_0000001265400114_ph14880920154918"></a><term id="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term16184138172215"></a>Data types supported by Atlas A2 training/inference products</term></span>: int8, int16, int32, int64, float16, float32, bfp16 Note that the performance will deteriorate for the `int64` data type.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265400114_row46964992"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265400114_p46067993"><a name="zh-cn_topic_0000001265400114_p46067993"></a><a name="zh-cn_topic_0000001265400114_p46067993"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265400114_p40519951"><a name="zh-cn_topic_0000001265400114_p40519951"></a><a name="zh-cn_topic_0000001265400114_p40519951"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265400114_p60890569"><a name="zh-cn_topic_0000001265400114_p60890569"></a><a name="zh-cn_topic_0000001265400114_p60890569"></a>`Reduce` operation type. Currently, `sum`, `prod`, `max`, and `min` are supported.</p>
<div class="note" id="zh-cn_topic_0000001265400114_note4427520124313"><a name="zh-cn_topic_0000001265400114_note4427520124313"></a><a name="zh-cn_topic_0000001265400114_note4427520124313"></a><span class="notetitle"> Note: </span><div class="notebody"><p id="zh-cn_topic_0000001265400114_p9984151202012"><a name="zh-cn_topic_0000001265400114_p9984151202012"></a><a name="zh-cn_topic_0000001265400114_p9984151202012"></a><span id="zh-cn_topic_0000001265400114_ph79242619219"><a name="zh-cn_topic_0000001265400114_ph79242619219"></a><a name="zh-cn_topic_0000001265400114_ph79242619219"></a><term id="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term1253731311225_2"></a>For Atlas A3 training/inference products</term></span>, the `prod` operation in the current version does not support data type `int16` or `bfp16`.</p>
<p id="zh-cn_topic_0000001265400114_p10731124313342"><a name="zh-cn_topic_0000001265400114_p10731124313342"></a><a name="zh-cn_topic_0000001265400114_p10731124313342"></a> <span id="zh-cn_topic_0000001265400114_ph49172713419"><a name="zh-cn_topic_0000001265400114_ph49172713419"></a><a name="zh-cn_topic_0000001265400114_ph49172713419"></a><term id="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000001265400114_zh-cn_topic_0000001312391781_term16184138172215_1"></a>For Atlas A2 training/inference products</term></span>, the `prod` operation in the current version does not support data type `int16` or `bfp16`.</p>
</div></div>
</td>
</tr>
<tr id="zh-cn_topic_0000001265400114_row11144211"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265400114_p30265903"><a name="zh-cn_topic_0000001265400114_p30265903"></a><a name="zh-cn_topic_0000001265400114_p30265903"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265400114_p35619075"><a name="zh-cn_topic_0000001265400114_p35619075"></a><a name="zh-cn_topic_0000001265400114_p35619075"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265400114_p66572856"><a name="zh-cn_topic_0000001265400114_p66572856"></a><a name="zh-cn_topic_0000001265400114_p66572856"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265400114_row62284798"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265400114_p11903911"><a name="zh-cn_topic_0000001265400114_p11903911"></a><a name="zh-cn_topic_0000001265400114_p11903911"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265400114_p24692740"><a name="zh-cn_topic_0000001265400114_p24692740"></a><a name="zh-cn_topic_0000001265400114_p24692740"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265400114_p53954942"><a name="zh-cn_topic_0000001265400114_p53954942"></a><a name="zh-cn_topic_0000001265400114_p53954942"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001265400114_section51595365"></a>

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001265400114_section61705107"></a>

-   The ranks must have the same `recvCount`, `dataType`, and `op`.

## Call Example<a name="zh-cn_topic_0000001265400114_section204039211474"></a>

```c
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // Number of data elements received by each rank.
uint64_t sendSize = rankSize * recvCount * sizeof(float);
uint64_t recvSize = recvCount * sizeof(float);

// Allocate device memory for collective communication.
void *sendBuf = nullptr, *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and streams.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Perform ReduceScatter to sum sendBuf of all ranks and then evenly distribute the result to the recvBuf of each rank in the rank ID order.
HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
