# HcclKfcOpArgsSetCount

## 功能说明

设置OpArgs中的数据量参数。

## 函数原型

```c
HcclResult HcclKfcOpArgsSetCount(void *opArgs, uint64_t count)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opArgs | 输入 | 通过[HcclKfcAllocOpArgs](HcclKfcAllocOpArgs.md)申请的OpArgs内存地址。 |
| count | 输入 | 参与通信的数据个数。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- opArgs不能为空。
- count不能超过系统支持的最大count值。
