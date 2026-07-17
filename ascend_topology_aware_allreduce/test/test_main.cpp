#include "../include/all_reduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

HcclComm g_comm = (HcclComm)0x1;
aclrtStream g_stream = (aclrtStream)0x2;

int main() {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};

    HcclResult ret = HcclAllReduce(
        input, output, 4,
        HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM,
        g_comm,
        g_stream
    );

    if (ret == HCCL_SUCCESS) {
        printf("AllReduce SUCCESS\n");
        for(int i=0; i<4; i++) printf("%f ", output[i]);
        printf("\n");
    } else {
        printf("AllReduce FAILED\n");
    }

    return 0;
}