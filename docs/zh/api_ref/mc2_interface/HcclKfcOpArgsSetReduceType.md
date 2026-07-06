# HcclKfcOpArgsSetReduceType

## 功能说明

设置OpArgs中的归约类型参数。

## 函数原型

```c
HcclResult HcclKfcOpArgsSetReduceType(void *opArgs, uint32_t reduceType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opArgs | 输入 | 通过[HcclKfcAllocOpArgs](HcclKfcAllocOpArgs.md)申请的OpArgs内存地址。 |
| reduceType | 输入 | 归约类型，按[HcclReduceOp](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclReduceOp.md)取值传入。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- opArgs不能为空。
- reduceType需要是有效的HcclReduceOp取值，HCCL_REDUCE_RESERVED不支持。
