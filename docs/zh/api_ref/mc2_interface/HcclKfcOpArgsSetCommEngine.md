# HcclKfcOpArgsSetCommEngine

## 功能说明

设置OpArgs中的通信引擎类型参数。

## 函数原型

```c
HcclResult HcclKfcOpArgsSetCommEngine(void *opArgs, uint8_t commEngine)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opArgs | 输入 | 通过[HcclKfcAllocOpArgs](HcclKfcAllocOpArgs.md)申请的OpArgs内存地址。 |
| commEngine | 输入 | 通信引擎类型。当前接口支持COMM_ENGINE_AICPU和COMM_ENGINE_AIV。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- opArgs不能为空。
- commEngine仅支持COMM_ENGINE_AICPU和COMM_ENGINE_AIV，其他取值返回不支持。
