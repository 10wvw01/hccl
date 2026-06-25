# HcclScatter<a name="ZH-CN_TOPIC_0000002486992308"></a>

## Supported Products<a name="zh-cn_topic_0000001779917185_section10594071513"></a>

<a name="zh-cn_topic_0000001779917185_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001779917185_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001779917185_p1883113061818"><a name="zh-cn_topic_0000001779917185_p1883113061818"></a><a name="zh-cn_topic_0000001779917185_p1883113061818"></a><span id="zh-cn_topic_0000001779917185_ph20833205312295"><a name="zh-cn_topic_0000001779917185_ph20833205312295"></a><a name="zh-cn_topic_0000001779917185_ph20833205312295"></a>Product</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001779917185_p783113012187"><a name="zh-cn_topic_0000001779917185_p783113012187"></a><a name="zh-cn_topic_0000001779917185_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001779917185_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001779917185_p48327011813"><a name="zh-cn_topic_0000001779917185_p48327011813"></a><a name="zh-cn_topic_0000001779917185_p48327011813"></a><span id="zh-cn_topic_0000001779917185_ph583230201815"><a name="zh-cn_topic_0000001779917185_ph583230201815"></a><a name="zh-cn_topic_0000001779917185_ph583230201815"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001779917185_p7948163910184"><a name="zh-cn_topic_0000001779917185_p7948163910184"></a><a name="zh-cn_topic_0000001779917185_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001779917185_p14832120181815"><a name="zh-cn_topic_0000001779917185_p14832120181815"></a><a name="zh-cn_topic_0000001779917185_p14832120181815"></a><span id="zh-cn_topic_0000001779917185_ph1292674871116"><a name="zh-cn_topic_0000001779917185_ph1292674871116"></a><a name="zh-cn_topic_0000001779917185_ph1292674871116"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001779917185_p19948143911820"><a name="zh-cn_topic_0000001779917185_p19948143911820"></a><a name="zh-cn_topic_0000001779917185_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training /inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Description<a name="zh-cn_topic_0000001779917185_section662317284469"></a>

`Scatter` is a collective communication operator that scatters data of the root rank to other ranks.

## Prototype<a name="zh-cn_topic_0000001779917185_section13621172844611"></a>

```
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000001779917185_section1462482884619"></a>

<a name="zh-cn_topic_0000001779917185_table24749807"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001779917185_row60665573"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001779917185_p14964341"><a name="zh-cn_topic_0000001779917185_p14964341"></a><a name="zh-cn_topic_0000001779917185_p14964341"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001779917185_p4152081"><a name="zh-cn_topic_0000001779917185_p4152081"></a><a name="zh-cn_topic_0000001779917185_p4152081"></a>Input/Output</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001779917185_p774306"><a name="zh-cn_topic_0000001779917185_p774306"></a><a name="zh-cn_topic_0000001779917185_p774306"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001779917185_row62718864"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p47063234"><a name="zh-cn_topic_0000001779917185_p47063234"></a><a name="zh-cn_topic_0000001779917185_p47063234"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p54025633"><a name="zh-cn_topic_0000001779917185_p54025633"></a><a name="zh-cn_topic_0000001779917185_p54025633"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p14000148"><a name="zh-cn_topic_0000001779917185_p14000148"></a><a name="zh-cn_topic_0000001779917185_p14000148"></a>Source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row58892473"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p5560998"><a name="zh-cn_topic_0000001779917185_p5560998"></a><a name="zh-cn_topic_0000001779917185_p5560998"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p47787675"><a name="zh-cn_topic_0000001779917185_p47787675"></a><a name="zh-cn_topic_0000001779917185_p47787675"></a>Output</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p45596481"><a name="zh-cn_topic_0000001779917185_p45596481"></a><a name="zh-cn_topic_0000001779917185_p45596481"></a>Destination buffer address where the collective communication result is stored.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row7715150"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p20947391"><a name="zh-cn_topic_0000001779917185_p20947391"></a><a name="zh-cn_topic_0000001779917185_p20947391"></a>recvCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p19017142"><a name="zh-cn_topic_0000001779917185_p19017142"></a><a name="zh-cn_topic_0000001779917185_p19017142"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p63993496"><a name="zh-cn_topic_0000001779917185_p63993496"></a><a name="zh-cn_topic_0000001779917185_p63993496"></a>Number of `recvBuf` data elements involved in the `scatter` operation. For example, `count = 1` indicates that one int32 data element is involved.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row39070558"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p10598606"><a name="zh-cn_topic_0000001779917185_p10598606"></a><a name="zh-cn_topic_0000001779917185_p10598606"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p53180767"><a name="zh-cn_topic_0000001779917185_p53180767"></a><a name="zh-cn_topic_0000001779917185_p53180767"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p7991229191118"><a name="zh-cn_topic_0000001779917185_p7991229191118"></a><a name="zh-cn_topic_0000001779917185_p7991229191118"></a>Data type of the `Scatter` operation, defined in HcclDataType.</p>
<p id="zh-cn_topic_0000001779917185_p16850153843018"><a name="zh-cn_topic_0000001779917185_p16850153843018"></a><a name="zh-cn_topic_0000001779917185_p16850153843018"></a><span id="zh-cn_topic_0000001779917185_ph13754548217"><a name="zh-cn_topic_0000001779917185_ph13754548217"></a><a name="zh-cn_topic_0000001779917185_ph13754548217"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data type supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000001779917185_p94179211177"><a name="zh-cn_topic_0000001779917185_p94179211177"></a><a name="zh-cn_topic_0000001779917185_p94179211177"></a><span id="zh-cn_topic_0000001779917185_ph841715341959"><a name="zh-cn_topic_0000001779917185_ph841715341959"></a><a name="zh-cn_topic_0000001779917185_ph841715341959"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term16184138172215"></a>Data type supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row46964992"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p7641038"><a name="zh-cn_topic_0000001779917185_p7641038"></a><a name="zh-cn_topic_0000001779917185_p7641038"></a>root</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p14944322"><a name="zh-cn_topic_0000001779917185_p14944322"></a><a name="zh-cn_topic_0000001779917185_p14944322"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p2530563"><a name="zh-cn_topic_0000001779917185_p2530563"></a><a name="zh-cn_topic_0000001779917185_p2530563"></a>Rank ID of the scatter root.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row11144211"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p30265903"><a name="zh-cn_topic_0000001779917185_p30265903"></a><a name="zh-cn_topic_0000001779917185_p30265903"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p35619075"><a name="zh-cn_topic_0000001779917185_p35619075"></a><a name="zh-cn_topic_0000001779917185_p35619075"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p66572856"><a name="zh-cn_topic_0000001779917185_p66572856"></a><a name="zh-cn_topic_0000001779917185_p66572856"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row62284798"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p11903911"><a name="zh-cn_topic_0000001779917185_p11903911"></a><a name="zh-cn_topic_0000001779917185_p11903911"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p24692740"><a name="zh-cn_topic_0000001779917185_p24692740"></a><a name="zh-cn_topic_0000001779917185_p24692740"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p53954942"><a name="zh-cn_topic_0000001779917185_p53954942"></a><a name="zh-cn_topic_0000001779917185_p53954942"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000001779917185_section663212844612"></a>

[HcclResult]: `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000001779917185_section16632182812468"></a>

-   The ranks must have the same `recvCount`, `dataType`, and `root`.
-   There can be only one root rank globally.
-   `sendBuf` of a non-root rank can be empty. `sendBuf` of the root rank cannot be empty.

## Call Example<a name="zh-cn_topic_0000001779917185_section204039211474"></a>

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t sendCount = 8;
uint64_t recvCount = 1;
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = recvCount * sizeof(float);

// Allocate the device memory for receiving the Scatter result.
ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
// On the root rank, allocate the device memory for storing the sent data.
if (device == rootRank) {
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
}

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, device, &hcclComm);

// Create a task flow.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform the Scatter operation is to evenly distribute data of the root rank in the communicator to other ranks.
HcclScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// Wait until the collective communication task in the task flow is complete.
aclrtSynchronizeStream(stream);

// Free resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task flow.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
