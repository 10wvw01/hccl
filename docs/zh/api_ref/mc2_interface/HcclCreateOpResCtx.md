# HcclCreateOpResCtx

## 功能说明

基于通信域、算子类型和OpArgs参数创建MC2自定义算子通信资源上下文。

## 函数原型

```c
HcclResult HcclCreateOpResCtx(HcclComm comm, uint8_t opType, void *opArgs, void **opResCtx)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 通信域。 |
| opType | 输入 | 算子类型编号，需要小于内部支持的最大算子类型值。 |
| opArgs | 输入 | 通过[HcclKfcAllocOpArgs](HcclKfcAllocOpArgs.md)申请并设置完成的OpArgs内存地址。 |
| opResCtx | 输出 | 指向创建出的通信资源上下文指针的地址。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- comm、opArgs和opResCtx不能为空。
- opType需要是有效的算子类型编号。
- 调用该接口前，应先通过HcclKfcAllocOpArgs申请OpArgs，并按业务需要设置数据类型、归约类型、count、算法配置和通信引擎参数。
