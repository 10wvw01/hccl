# HcclAllGatherV<a name="ZH-CN_TOPIC_0000002486992304"></a>

## Supported Products<a name="zh-cn_topic_0000002139400230_section10594071513"></a>

<a name="zh-cn_topic_0000002139400230_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002139400230_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002139400230_p1883113061818"><a name="zh-cn_topic_0000002139400230_p1883113061818"></a><a name="zh-cn_topic_0000002139400230_p1883113061818"></a><span id="zh-cn_topic_0000002139400230_ph20833205312295"><a name="zh-cn_topic_0000002139400230_ph20833205312295"></a><a name="zh-cn_topic_0000002139400230_ph20833205312295"></a> Product </span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002139400230_p783113012187"><a name="zh-cn_topic_0000002139400230_p783113012187"></a><a name="zh-cn_topic_0000002139400230_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002139400230_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002139400230_p48327011813"><a name="zh-cn_topic_0000002139400230_p48327011813"></a><a name="zh-cn_topic_0000002139400230_p48327011813"></a><span id="zh-cn_topic_0000002139400230_ph583230201815"><a name="zh-cn_topic_0000002139400230_ph583230201815"></a><a name="zh-cn_topic_0000002139400230_ph583230201815"></a><term id="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002139400230_p7948163910184"><a name="zh-cn_topic_0000002139400230_p7948163910184"></a><a name="zh-cn_topic_0000002139400230_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002139400230_p14832120181815"><a name="zh-cn_topic_0000002139400230_p14832120181815"></a><a name="zh-cn_topic_0000002139400230_p14832120181815"></a><span id="zh-cn_topic_0000002139400230_ph1292674871116"><a name="zh-cn_topic_0000002139400230_ph1292674871116"></a><a name="zh-cn_topic_0000002139400230_ph1292674871116"></a><term id="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training/inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002139400230_p19948143911820"><a name="zh-cn_topic_0000002139400230_p19948143911820"></a><a name="zh-cn_topic_0000002139400230_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training/inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000002139400230_section59721402"></a>

`AllGatherV` is a collective communication operator that re-sorts the inputs of all ranks in the communicator by rank ID, combines the inputs, and sends the results to the outputs of all ranks.

Unlike the `AllGather` operator, the `AllGatherV` operator allows different nodes in the communicator to have different input sizes.

![]()

> [!NOTE]NOTE
> For the `AllGatherV` operation, each rank receives a set of data that is resorted by rank ID, that is, `AllGatherV` outputs of all ranks are the same.

## Prototype<a name="zh-cn_topic_0000002139400230_section66288034"></a>

```
HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000002139400230_section621706"></a>

<a name="zh-cn_topic_0000002139400230_table51170717"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002139400230_row27848947"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002139400230_p41172271"><a name="zh-cn_topic_0000002139400230_p41172271"></a><a name="zh-cn_topic_0000002139400230_p41172271"></a> Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002139400230_p46619622"><a name="zh-cn_topic_0000002139400230_p46619622"></a><a name="zh-cn_topic_0000002139400230_p46619622"></a> Input/Output </p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002139400230_p18093058"><a name="zh-cn_topic_0000002139400230_p18093058"></a><a name="zh-cn_topic_0000002139400230_p18093058"></a> Description </p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002139400230_row56251627"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p60087944"><a name="zh-cn_topic_0000002139400230_p60087944"></a><a name="zh-cn_topic_0000002139400230_p60087944"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p35285314"><a name="zh-cn_topic_0000002139400230_p35285314"></a><a name="zh-cn_topic_0000002139400230_p35285314"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p39538176"><a name="zh-cn_topic_0000002139400230_p39538176"></a><a name="zh-cn_topic_0000002139400230_p39538176"></a> Source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row17762502"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p29476530"><a name="zh-cn_topic_0000002139400230_p29476530"></a><a name="zh-cn_topic_0000002139400230_p29476530"></a>sendCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p38788755"><a name="zh-cn_topic_0000002139400230_p38788755"></a><a name="zh-cn_topic_0000002139400230_p38788755"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p4564121932310"><a name="zh-cn_topic_0000002139400230_p4564121932310"></a><a name="zh-cn_topic_0000002139400230_p4564121932310"></a>Size of `sendBuf` data involved in the `AllGatherV` operation.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row104115316273"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p33628036"><a name="zh-cn_topic_0000002139400230_p33628036"></a><a name="zh-cn_topic_0000002139400230_p33628036"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p39516413"><a name="zh-cn_topic_0000002139400230_p39516413"></a><a name="zh-cn_topic_0000002139400230_p39516413"></a> Output </p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p46712854"><a name="zh-cn_topic_0000002139400230_p46712854"></a><a name="zh-cn_topic_0000002139400230_p46712854"></a>Destination buffer address where the collective communication result is stored.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row191691533142715"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p91694339272"><a name="zh-cn_topic_0000002139400230_p91694339272"></a><a name="zh-cn_topic_0000002139400230_p91694339272"></a>recvCounts</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p151692333273"><a name="zh-cn_topic_0000002139400230_p151692333273"></a><a name="zh-cn_topic_0000002139400230_p151692333273"></a> Output </p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p834451310267"><a name="zh-cn_topic_0000002139400230_p834451310267"></a><a name="zh-cn_topic_0000002139400230_p834451310267"></a> A `uint64_t` arry representing the size of data each rank receives into `recvBuf` in the `AllGatherV` operation.</p>
<p id="zh-cn_topic_0000002139400230_p201691033122720"><a name="zh-cn_topic_0000002139400230_p201691033122720"></a><a name="zh-cn_topic_0000002139400230_p201691033122720"></a> The i-th element indicates the data size received from rank `i`. The data size must be the same as the value of `sendCount` of rank `i`.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row1510102214287"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p1951014222284"><a name="zh-cn_topic_0000002139400230_p1951014222284"></a><a name="zh-cn_topic_0000002139400230_p1951014222284"></a>recvDispls</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p12510162232820"><a name="zh-cn_topic_0000002139400230_p12510162232820"></a><a name="zh-cn_topic_0000002139400230_p12510162232820"></a> Output </p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p1631164584317"><a name="zh-cn_topic_0000002139400230_p1631164584317"></a><a name="zh-cn_topic_0000002139400230_p1631164584317"></a> A `uint64_t` array representing the offset of each rank's data in `recvBuf` in the `AllGatherV` operation, in units of `dataType`.</p>
<p id="zh-cn_topic_0000002139400230_p136693624615"><a name="zh-cn_topic_0000002139400230_p136693624615"></a><a name="zh-cn_topic_0000002139400230_p136693624615"></a> The i-th element indicates the start offset of the data received from rank `i` in `recvBuf`. </p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row24171358"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p11722994"><a name="zh-cn_topic_0000002139400230_p11722994"></a><a name="zh-cn_topic_0000002139400230_p11722994"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p10038421"><a name="zh-cn_topic_0000002139400230_p10038421"></a><a name="zh-cn_topic_0000002139400230_p10038421"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p51201049151217"><a name="zh-cn_topic_0000002139400230_p51201049151217"></a><a name="zh-cn_topic_0000002139400230_p51201049151217"></a>Data type of the `AllGatherV` operation, defined in <a href="https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000002139400230_p578154617218"><a name="zh-cn_topic_0000002139400230_p578154617218"></a><a name="zh-cn_topic_0000002139400230_p578154617218"></a><span id="zh-cn_topic_0000002139400230_ph15590543154217"><a name="zh-cn_topic_0000002139400230_ph15590543154217"></a><a name="zh-cn_topic_0000002139400230_ph15590543154217"></a><term id="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Data types supported by Atlas A3 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
<p id="zh-cn_topic_0000002139400230_p94179211177"><a name="zh-cn_topic_0000002139400230_p94179211177"></a><a name="zh-cn_topic_0000002139400230_p94179211177"></a> <span id="zh-cn_topic_0000002139400230_ph14880920154918"><a name="zh-cn_topic_0000002139400230_ph14880920154918"></a><a name="zh-cn_topic_0000002139400230_ph14880920154918"></a><term id="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000002139400230_zh-cn_topic_0000001312391781_term16184138172215"></a>Data type supported by Atlas A2 training/inference products</term></span>: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row3143429"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p53291177"><a name="zh-cn_topic_0000002139400230_p53291177"></a><a name="zh-cn_topic_0000002139400230_p53291177"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p21618074"><a name="zh-cn_topic_0000002139400230_p21618074"></a><a name="zh-cn_topic_0000002139400230_p21618074"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p6233535"><a name="zh-cn_topic_0000002139400230_p6233535"></a><a name="zh-cn_topic_0000002139400230_p6233535"></a>Communicator where collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002139400230_row56101816"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002139400230_p47953287"><a name="zh-cn_topic_0000002139400230_p47953287"></a><a name="zh-cn_topic_0000002139400230_p47953287"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002139400230_p59011071"><a name="zh-cn_topic_0000002139400230_p59011071"></a><a name="zh-cn_topic_0000002139400230_p59011071"></a> Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002139400230_p15167431"><a name="zh-cn_topic_0000002139400230_p15167431"></a><a name="zh-cn_topic_0000002139400230_p15167431"></a>Stream used by this rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000002139400230_section5595356"></a>

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000002139400230_section50358210"></a>

-   The ranks must have the same `recvCounts`, `recvDispls`, and `dataType`.
-   Atlas A3 training/inference products support only the single-server scenario.
-   Atlas A2 training/inference products support only symmetric multi-server distribution. Asymmetric distribution (unequal number of NPUs per server) is not supported.
