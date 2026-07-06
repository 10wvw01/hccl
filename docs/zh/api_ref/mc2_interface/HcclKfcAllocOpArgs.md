# HcclKfcAllocOpArgs

## 功能说明

申请并初始化MC2自定义算子资源创建所需的OpArgs内存。申请成功后，可通过HcclKfcOpArgsSetSrcDataType、HcclKfcOpArgsSetDstDataType、HcclKfcOpArgsSetReduceType、HcclKfcOpArgsSetCount、HcclKfcOpArgsSetAlgConfig和HcclKfcOpArgsSetCommEngine设置OpArgs参数。

## 函数原型

```c
HcclResult HcclKfcAllocOpArgs(void **opArgs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opArgs | 输出 | 指向OpArgs内存指针的地址。接口成功返回后，`*opArgs`指向已初始化的OpArgs内存。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- opArgs不能为空。
- 接口内部会为OpArgs申请内存，使用完成后需要调用[HcclKfcFreeOpArgs](HcclKfcFreeOpArgs.md)释放。
- 申请成功后，OpArgs中的源数据类型和目的数据类型默认初始化为HCCL_DATA_TYPE_FP16，归约类型默认初始化为HCCL_REDUCE_SUM，count默认初始化为0。
