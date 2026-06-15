# Communication Operator API List

HCCL provides communication operator APIs in C language. Framework developers can call these APIs to implement framework adaptation in single-operator mode to implement distributed capabilities.

<a name="zh-cn_topic_0000001312721317_table1554562693420"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312721317_row125461626123420"><th class="cellrowborder" valign="top" width="25.46%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001312721317_p8546826113410"><a name="zh-cn_topic_0000001312721317_p8546826113410"></a><a name="zh-cn_topic_0000001312721317_p8546826113410"></a><strong id="zh-cn_topic_0000001312721317_b225214248340"><a name="zh-cn_topic_0000001312721317_b225214248340"></a><a name="zh-cn_topic_0000001312721317_b225214248340"></a> API</strong></p>
</th>
<th class="cellrowborder" valign="top" width="74.53999999999999%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001312721317_p10546122613347"><a name="zh-cn_topic_0000001312721317_p10546122613347"></a><a name="zh-cn_topic_0000001312721317_p10546122613347"></a><strong id="zh-cn_topic_0000001312721317_b132536244347"><a name="zh-cn_topic_0000001312721317_b132536244347"></a><a name="zh-cn_topic_0000001312721317_b132536244347"></a> Description</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312721317_row967319459273"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p116731455279"><a name="zh-cn_topic_0000001312721317_p116731455279"></a><a name="zh-cn_topic_0000001312721317_p116731455279"></a><a href="./context/HcclAllReduce.md">HcclAllReduce</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p11287144214307"><a name="zh-cn_topic_0000001312721317_p11287144214307"></a><a name="zh-cn_topic_0000001312721317_p11287144214307"></a>API for the `AllReduce` collective communication operator. It performs a reduction operation (such as sum, max, or min) across all ranks in the communicator and then scatters the results to each rank's output buffer. The reduction operation type is specified by the `op` parameter.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row48931643182715"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p17893204382717"><a name="zh-cn_topic_0000001312721317_p17893204382717"></a><a name="zh-cn_topic_0000001312721317_p17893204382717"></a><a href="./context/HcclBroadcast.md">HcclBroadcast</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p10286104283019"><a name="zh-cn_topic_0000001312721317_p10286104283019"></a><a name="zh-cn_topic_0000001312721317_p10286104283019"></a>API for the `Broadcast` collective communication operator. It broadcasts the data of the root rank to other ranks in the communicator.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row498016411272"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p098016412276"><a name="zh-cn_topic_0000001312721317_p098016412276"></a><a name="zh-cn_topic_0000001312721317_p098016412276"></a><a href="./context/HcclAllGather.md">HcclAllGather</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p52852426305"><a name="zh-cn_topic_0000001312721317_p52852426305"></a><a name="zh-cn_topic_0000001312721317_p52852426305"></a>API for the `AllGather` collective communication operator. It re-sorts the inputs of all ranks in the communicator by rank ID, combines the inputs, and sends the results to the outputs of all ranks.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row1030115031119"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p23013071117"><a name="zh-cn_topic_0000001312721317_p23013071117"></a><a name="zh-cn_topic_0000001312721317_p23013071117"></a><a href="./context/HcclAllGatherV.md">HcclAllGatherV</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p680714122118"><a name="zh-cn_topic_0000001312721317_p680714122118"></a><a name="zh-cn_topic_0000001312721317_p680714122118"></a>API for the `AllGatherV` collective communication operator. It re-sorts the inputs of all ranks in the communicator by rank ID, combines the inputs, and sends the results to the outputs of all ranks.</p>
<p id="zh-cn_topic_0000001312721317_p5604103112110"><a name="zh-cn_topic_0000001312721317_p5604103112110"></a><a name="zh-cn_topic_0000001312721317_p5604103112110"></a>Unlike `AllGather`, the `AllGatherV` operator allows different nodes in the communicator to have different input sizes.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row1569917547284"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p10178125714617"><a name="zh-cn_topic_0000001312721317_p10178125714617"></a><a name="zh-cn_topic_0000001312721317_p10178125714617"></a><a href="./context/HcclReduceScatter.md">HcclReduceScatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p22858424300"><a name="zh-cn_topic_0000001312721317_p22858424300"></a><a name="zh-cn_topic_0000001312721317_p22858424300"></a>API for the `ReduceScatter` collective communication operator. It performs the sum operation (or other reduction operations) on the inputs of all ranks, and then distributes the result evenly to the output buffers of ranks according to the rank IDs. Each process receives `1/rank\_size` portion of data from other processes for reduction.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row198415527106"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p484125214102"><a name="zh-cn_topic_0000001312721317_p484125214102"></a><a name="zh-cn_topic_0000001312721317_p484125214102"></a><a href="./context/HcclReduceScatterV.md">HcclReduceScatterV</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p40902273"><a name="zh-cn_topic_0000001312721317_p40902273"></a><a name="zh-cn_topic_0000001312721317_p40902273"></a>API for the `HcclReduceScatterV` collective communication operator. It performs a reduction operation (such as sum, max, or min) across all ranks and then scatters the results to each rank's output buffer. Each process receives the reduced result corresponding to its own rank ID.</p>
<p id="zh-cn_topic_0000001312721317_p147680373120"><a name="zh-cn_topic_0000001312721317_p147680373120"></a><a name="zh-cn_topic_0000001312721317_p147680373120"></a>Unlike the standard `ReduceScatter` operator, `ReduceScatterV` allows different ranks within the communicator to contribute varying amounts of data.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row075325292818"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p616718244304"><a name="zh-cn_topic_0000001312721317_p616718244304"></a><a name="zh-cn_topic_0000001312721317_p616718244304"></a><a href="./context/HcclReduce.md">HcclReduce</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p2284124216302"><a name="zh-cn_topic_0000001312721317_p2284124216302"></a><a name="zh-cn_topic_0000001312721317_p2284124216302"></a>API for the `Reduce` collective communication operator. It performs reduction operations (such as sum, max, and min) on the data of all ranks and sends the result to the specified position on the root rank.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row159481335268"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p775018350304"><a name="zh-cn_topic_0000001312721317_p775018350304"></a><a name="zh-cn_topic_0000001312721317_p775018350304"></a><a href="./context/HcclAlltoAll.md">HcclAlltoAll</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p1283154273017"><a name="zh-cn_topic_0000001312721317_p1283154273017"></a><a name="zh-cn_topic_0000001312721317_p1283154273017"></a>API for the `AlltoAll` collective communication operator. It sends the same-sized data to all ranks and receives the same-sized data from all ranks.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row1387718321301"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p647573913015"><a name="zh-cn_topic_0000001312721317_p647573913015"></a><a name="zh-cn_topic_0000001312721317_p647573913015"></a><a href="./context/HcclAlltoAllV.md">HcclAlltoAllV</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p12284742143010"><a name="zh-cn_topic_0000001312721317_p12284742143010"></a><a name="zh-cn_topic_0000001312721317_p12284742143010"></a>API for the `AlltoAllV` collective communication operator. It sends data (with customizable sizes) to all ranks in the communicator and receives data from all ranks.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row178091240103313"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p1580912402331"><a name="zh-cn_topic_0000001312721317_p1580912402331"></a><a name="zh-cn_topic_0000001312721317_p1580912402331"></a><a href="./context/HcclAlltoAllVC.md">HcclAlltoAllVC</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p88091440193317"><a name="zh-cn_topic_0000001312721317_p88091440193317"></a><a name="zh-cn_topic_0000001312721317_p88091440193317"></a><span id="zh-cn_topic_0000002417514657_ph20748105914110"><a name="zh-cn_topic_0000002417514657_ph20748105914110"></a><a name="zh-cn_topic_0000002417514657_ph20748105914110"></a>C</span>API for the `AlltoAllV` collective communication operator. It sends data (with customizable sizes) to all ranks in the communicator and receives data from all ranks. <span id="zh-cn_topic_0000002417514657_ph1509131064715"><a name="zh-cn_topic_0000002417514657_ph1509131064715"></a><a name="zh-cn_topic_0000002417514657_ph1509131064715"></a>C</span>Unlike `AlltoAllV`, `AlltoAllVC` uses the argument `sendCountMatrix` to pass receive and send parameters of all ranks.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row384384553515"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p11538195022820"><a name="zh-cn_topic_0000001312721317_p11538195022820"></a><a name="zh-cn_topic_0000001312721317_p11538195022820"></a><a href="./context/HcclScatter.md">HcclScatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p1584416459354"><a name="zh-cn_topic_0000001312721317_p1584416459354"></a><a name="zh-cn_topic_0000001312721317_p1584416459354"></a>API for the `Scatter` collective communication operator. It evenly divides and scatters data of the root rank to other ranks.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row14675121613328"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p167651663216"><a name="zh-cn_topic_0000001312721317_p167651663216"></a><a name="zh-cn_topic_0000001312721317_p167651663216"></a><a href="./context/HcclSend.md">HcclSend</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p1367731611321"><a name="zh-cn_topic_0000001312721317_p1367731611321"></a><a name="zh-cn_topic_0000001312721317_p1367731611321"></a>Send API for point-to-point communication. It sends data from a specified location on the current node to a specified location on the destination node.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row119221338194819"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p3999134233116"><a name="zh-cn_topic_0000001312721317_p3999134233116"></a><a name="zh-cn_topic_0000001312721317_p3999134233116"></a><a href="./context/HcclRecv.md">HcclRecv</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p9923133818487"><a name="zh-cn_topic_0000001312721317_p9923133818487"></a><a name="zh-cn_topic_0000001312721317_p9923133818487"></a>Receive API for point-to-point communication. It receives data from the source rank to the specified location on the current rank.</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row6327155483515"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p7327165413359"><a name="zh-cn_topic_0000001312721317_p7327165413359"></a><a name="zh-cn_topic_0000001312721317_p7327165413359"></a><a href="./context/HcclBatchSendRecv.md">HcclBatchSendRecv</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p732712548358"><a name="zh-cn_topic_0000001312721317_p732712548358"></a><a name="zh-cn_topic_0000001312721317_p732712548358"></a>API for Asynchronous batch point-to-point communication. It completes send and receive tasks in batches on the current rank. The send and receive tasks of the current rank are asynchronous and do not block each other.</p>
</td>
</tr>
</tbody>
</table>
