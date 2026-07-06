# HcclKfcFreeOpArgs

## 功能说明

释放通过[HcclKfcAllocOpArgs](HcclKfcAllocOpArgs.md)申请的OpArgs内存。

## 函数原型

```c
HcclResult HcclKfcFreeOpArgs(void *opArgs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opArgs | 输入 | 通过HcclKfcAllocOpArgs申请的OpArgs内存地址。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- opArgs不能为空。
- opArgs应为HcclKfcAllocOpArgs返回的有效内存地址，避免重复释放或释放非本接口族申请的内存。
