# HcclKfcOpArgsSetSrcDataType

## 功能说明

设置OpArgs中的源数据类型参数。

## 函数原型

```c
HcclResult HcclKfcOpArgsSetSrcDataType(void *opArgs, uint8_t srcDataType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opArgs | 输入 | 通过[HcclKfcAllocOpArgs](HcclKfcAllocOpArgs.md)申请的OpArgs内存地址。 |
| srcDataType | 输入 | 源数据类型，按[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)取值传入。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- opArgs不能为空。
- srcDataType需要是有效的HcclDataType取值，HCCL_DATA_TYPE_RESERVED不支持。
