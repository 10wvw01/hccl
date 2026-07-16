#include "all_reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 模拟comm和stream（仿真用）
Hcc1Comm g_comm = (Hcc1Comm)0x1;
aclrtStream g_stream = (aclrtStream)0x2;

int main()
{
    // 假设运行在rank 0（仅作演示）
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};

    // 实际多进程测试需启动16个进程，此处仅验证接口逻辑
    Hcc1Result ret = Hcc1AllReduce(input, output, 4, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, g_comm, g_stream);

    if (ret == HCCL_SUCCESS) {
        printf("AllReduce SUCCESS\n");
        // 预期在16卡全归约下，结果为原值*16（若全部参与）
        // 此处仿真仅单卡，实际结果需多卡验证
        for (int i = 0; i < 4; i++)
            printf("%f ", output[i]);
        printf("\n");
    } else {
        printf("AllReduce FAILED\n");
    }
    return 0;
}