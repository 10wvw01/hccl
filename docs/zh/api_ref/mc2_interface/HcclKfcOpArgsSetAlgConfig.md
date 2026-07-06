# HcclKfcOpArgsSetAlgConfig

## 功能说明

设置OpArgs中的算法配置字符串。

## 函数原型

```c
HcclResult HcclKfcOpArgsSetAlgConfig(void *opArgs, char *algConfig)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opArgs | 输入 | 通过[HcclKfcAllocOpArgs](HcclKfcAllocOpArgs.md)申请的OpArgs内存地址。 |
| algConfig | 输入 | 算法配置字符串。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- opArgs不能为空。
- algConfig不能为空。
- algConfig字符串长度需要小于内部算法配置缓存长度，否则接口返回失败。
