# HcclReduceScatterV<a name="ZH-CN_TOPIC_0000002519072189"></a>

## Supported Products<a name="zh-cn_topic_0000002174837853_section10594071513"></a>

<a name="zh-cn_topic_0000002174837853_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002174837853_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002174837853_p1883113061818"><a name="zh-cn_topic_0000002174837853_p1883113061818"></a><a name="zh-cn_topic_0000002174837853_p1883113061818"></a><span id="zh-cn_topic_0000002174837853_ph20833205312295"><a name="zh-cn_topic_0000002174837853_ph20833205312295"></a><a name="zh-cn_topic_0000002174837853_ph20833205312295"></a>Product</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002174837853_p783113012187"><a name="zh-cn_topic_0000002174837853_p783113012187"></a><a name="zh-cn_topic_0000002174837853_p783113012187"></a>Supported</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002174837853_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002174837853_p48327011813"><a name="zh-cn_topic_0000002174837853_p48327011813"></a><a name="zh-cn_topic_0000002174837853_p48327011813"></a><span id="zh-cn_topic_0000002174837853_ph583230201815"><a name="zh-cn_topic_0000002174837853_ph583230201815"></a><a name="zh-cn_topic_0000002174837853_ph583230201815"></a><term id="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 training products and Atlas A3 inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002174837853_p1827411542492"><a name="zh-cn_topic_0000002174837853_p1827411542492"></a><a name="zh-cn_topic_0000002174837853_p1827411542492"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002174837853_p14832120181815"><a name="zh-cn_topic_0000002174837853_p14832120181815"></a><a name="zh-cn_topic_0000002174837853_p14832120181815"></a><span id="zh-cn_topic_0000002174837853_ph1292674871116"><a name="zh-cn_topic_0000002174837853_ph1292674871116"></a><a name="zh-cn_topic_0000002174837853_ph1292674871116"></a><term id="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 training products and Atlas A2 inference products</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002174837853_p19948143911820"><a name="zh-cn_topic_0000002174837853_p19948143911820"></a><a name="zh-cn_topic_0000002174837853_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]NOTE
> For Atlas A2 training products/Atlas A2 inference products, support is limited to the Atlas 800T A2 training server, Atlas 900 A2 PoD cluster basic unit, and Atlas 200T A2 Box16 heterogeneous subrack.

## Function Description<a name="zh-cn_topic_0000002174837853_section31291646"></a>

`HcclReduceScatterV` is a collective communication operator that performs a reduction operation (such as sum, max, or min) across all ranks and then scatters the results to each rank's output buffer. Each process receives the reduced result corresponding to its own rank ID.

Unlike the standard `ReduceScatter` operator, `ReduceScatterV` allows different ranks within the communicator to contribute varying amounts of data.

As shown in the figure below, ranks 0–3 each perform a reduction operation (for example, `sum`) on the data corresponding to their own rank ID from all other ranks, and send the result to their respective output buffers.

![](figures/allgather-3.png)

## Prototype<a name="zh-cn_topic_0000002174837853_section18389930"></a>

```
HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameter Description<a name="zh-cn_topic_0000002174837853_section13189358"></a>

<a name="zh-cn_topic_0000002174837853_table24749807"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002174837853_row60665573"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002174837853_p14964341"><a name="zh-cn_topic_0000002174837853_p14964341"></a><a name="zh-cn_topic_0000002174837853_p14964341"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="17.150000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002174837853_p4152081"><a name="zh-cn_topic_0000002174837853_p4152081"></a><a name="zh-cn_topic_0000002174837853_p4152081"></a>Input/Output</p>
</th>
<th class="cellrowborder" valign="top" width="62.64999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002174837853_p774306"><a name="zh-cn_topic_0000002174837853_p774306"></a><a name="zh-cn_topic_0000002174837853_p774306"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002174837853_row62718864"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p47063234"><a name="zh-cn_topic_0000002174837853_p47063234"></a><a name="zh-cn_topic_0000002174837853_p47063234"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p54025633"><a name="zh-cn_topic_0000002174837853_p54025633"></a><a name="zh-cn_topic_0000002174837853_p54025633"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p14000148"><a name="zh-cn_topic_0000002174837853_p14000148"></a><a name="zh-cn_topic_0000002174837853_p14000148"></a>Source data buffer address.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row885675512302"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p3857115511308"><a name="zh-cn_topic_0000002174837853_p3857115511308"></a><a name="zh-cn_topic_0000002174837853_p3857115511308"></a>sendCounts</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p885716551303"><a name="zh-cn_topic_0000002174837853_p885716551303"></a><a name="zh-cn_topic_0000002174837853_p885716551303"></a>Input</p>
</td>
Array of `uint64_t` values specifying the amount of data contributed by each rank in `sendBuf`.</p>
<p id="zh-cn_topic_0000002174837853_p201691033122720"><a name="zh-cn_topic_0000002174837853_p201691033122720"></a><a name="zh-cn_topic_0000002174837853_p201691033122720"></a>The i-th element indicates the data size sent to rank `i`. </p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row3606563311"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p136063673112"><a name="zh-cn_topic_0000002174837853_p136063673112"></a><a name="zh-cn_topic_0000002174837853_p136063673112"></a>sendDispls</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p260614610316"><a name="zh-cn_topic_0000002174837853_p260614610316"></a><a name="zh-cn_topic_0000002174837853_p260614610316"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p123091053114318"><a name="zh-cn_topic_0000002174837853_p123091053114318"></a><a name="zh-cn_topic_0000002174837853_p123091053114318"></a>Array of `uint64_t` values specifying the offset (in units of `dataType`) of each rank's data within `sendBuf`. </p>
<p id="zh-cn_topic_0000002174837853_p136693624615"><a name="zh-cn_topic_0000002174837853_p136693624615"></a><a name="zh-cn_topic_0000002174837853_p136693624615"></a> The i-th element indicates the offset of the data sent to rank `i`. |</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row58892473"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p5560998"><a name="zh-cn_topic_0000002174837853_p5560998"></a><a name="zh-cn_topic_0000002174837853_p5560998"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p47787675"><a name="zh-cn_topic_0000002174837853_p47787675"></a><a name="zh-cn_topic_0000002174837853_p47787675"></a>Output</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p45596481"><a name="zh-cn_topic_0000002174837853_p45596481"></a><a name="zh-cn_topic_0000002174837853_p45596481"></a>Destination buffer address where the collective communication result is stored.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row7715150"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p20947391"><a name="zh-cn_topic_0000002174837853_p20947391"></a><a name="zh-cn_topic_0000002174837853_p20947391"></a>recvCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p19017142"><a name="zh-cn_topic_0000002174837853_p19017142"></a><a name="zh-cn_topic_0000002174837853_p19017142"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p2198183114317"><a name="zh-cn_topic_0000002174837853_p2198183114317"></a><a name="zh-cn_topic_0000002174837853_p2198183114317"></a>Data size received by the current rank into `recvBuf`.</p>
<p id="zh-cn_topic_0000002174837853_p20599122510436"><a name="zh-cn_topic_0000002174837853_p20599122510436"></a><a name="zh-cn_topic_0000002174837853_p20599122510436"></a>For rank `i`, this value must match the i-th element of `sendCounts`.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row39070558"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p10598606"><a name="zh-cn_topic_0000002174837853_p10598606"></a><a name="zh-cn_topic_0000002174837853_p10598606"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p53180767"><a name="zh-cn_topic_0000002174837853_p53180767"></a><a name="zh-cn_topic_0000002174837853_p53180767"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p1114114215138"><a name="zh-cn_topic_0000002174837853_p1114114215138"></a><a name="zh-cn_topic_0000002174837853_p1114114215138"></a>Data type of the `ReduceScatterV` operation, specified in <a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>.</p>
<p id="zh-cn_topic_0000002174837853_p261131302211"><a name="zh-cn_topic_0000002174837853_p261131302211"></a><a name="zh-cn_topic_0000002174837853_p261131302211"></a>For <span id="zh-cn_topic_0000002174837853_ph13754548217"><a name="zh-cn_topic_0000002174837853_ph13754548217"></a><a name="zh-cn_topic_0000002174837853_ph13754548217"></a><term id="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 training/inference products</term></span>: int8, int16, int32, float16, float32, bfp16</p>
<p id="zh-cn_topic_0000002174837853_p16921716151418"><a name="zh-cn_topic_0000002174837853_p16921716151418"></a><a name="zh-cn_topic_0000002174837853_p16921716151418"></a>For <span id="zh-cn_topic_0000002174837853_ph14880920154918"><a name="zh-cn_topic_0000002174837853_ph14880920154918"></a><a name="zh-cn_topic_0000002174837853_ph14880920154918"></a><term id="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 training/inference products</term></span>: int8, int16, int32, float16, float32, bfp16</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row46964992"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p46067993"><a name="zh-cn_topic_0000002174837853_p46067993"></a><a name="zh-cn_topic_0000002174837853_p46067993"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p40519951"><a name="zh-cn_topic_0000002174837853_p40519951"></a><a name="zh-cn_topic_0000002174837853_p40519951"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p135519418474"><a name="zh-cn_topic_0000002174837853_p135519418474"></a><a name="zh-cn_topic_0000002174837853_p135519418474"></a>`Reduce` operation type.</p>
<p id="zh-cn_topic_0000002174837853_p2077910161390"><a name="zh-cn_topic_0000002174837853_p2077910161390"></a><a name="zh-cn_topic_0000002174837853_p2077910161390"></a>For <span id="zh-cn_topic_0000002174837853_ph164912228399"><a name="zh-cn_topic_0000002174837853_ph164912228399"></a><a name="zh-cn_topic_0000002174837853_ph164912228399"></a><term id="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 training/inference products</term></span>: sum, max, min</p>
<p id="zh-cn_topic_0000002174837853_p60890569"><a name="zh-cn_topic_0000002174837853_p60890569"></a><a name="zh-cn_topic_0000002174837853_p60890569"></a>For <span id="zh-cn_topic_0000002174837853_ph149551118476"><a name="zh-cn_topic_0000002174837853_ph149551118476"></a><a name="zh-cn_topic_0000002174837853_ph149551118476"></a><term id="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000002174837853_zh-cn_topic_0000001312391781_term16184138172215_1"></a>Atlas A2 training/inference products</term></span>: sum, max, min</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row11144211"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p30265903"><a name="zh-cn_topic_0000002174837853_p30265903"></a><a name="zh-cn_topic_0000002174837853_p30265903"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p35619075"><a name="zh-cn_topic_0000002174837853_p35619075"></a><a name="zh-cn_topic_0000002174837853_p35619075"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p66572856"><a name="zh-cn_topic_0000002174837853_p66572856"></a><a name="zh-cn_topic_0000002174837853_p66572856"></a>Communicator in which collective communication is performed.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002174837853_row62284798"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002174837853_p11903911"><a name="zh-cn_topic_0000002174837853_p11903911"></a><a name="zh-cn_topic_0000002174837853_p11903911"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.150000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002174837853_p24692740"><a name="zh-cn_topic_0000002174837853_p24692740"></a><a name="zh-cn_topic_0000002174837853_p24692740"></a>Input</p>
</td>
<td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002174837853_p53954942"><a name="zh-cn_topic_0000002174837853_p53954942"></a><a name="zh-cn_topic_0000002174837853_p53954942"></a>Stream used by the current rank.</p>
</td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0000002174837853_section51595365"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193): `HCCL_SUCCESS` on success, or others on failure.

## Constraints<a name="zh-cn_topic_0000002174837853_section61705107"></a>

-   All ranks must have the same `sendCounts`, `sendDispls`, `dataType`, and `op`.
-   Atlas A3 training/inference products support only the single-server scenario.
-   Atlas A2 training/inference products support only symmetric multi-server distribution. Asymmetric distribution (unequal number of NPUs per server) is not supported.
