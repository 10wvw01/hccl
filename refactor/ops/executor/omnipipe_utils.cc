/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <omnipipe_utils.h>

/*
xB为x轴(慢轴)等效带宽，yB为y轴(快轴)等效带，需要先将xy作为一个整体计算xy的等效带宽
已知需要传输的斜对角数据总量为(ranksize_x - 1)*(ranksize_y -1)
已经通过y轴传输的斜对角数据为(sumY - 1)*(ranksize_y -1)，
sumY的计算方法见上一页的ydata[i] = xdata[i-1] * (ranksize(x) - 1)递推公式
那么通过x轴传输的斜对角数据为（SumY为不含最后一步的累计传输数）
xdataRatio[最后一步斜对角数据] =（(ranksize_x - 1)*(ranksize_y -1) - （sumY - 1)*(ranksize_y -1))/(1+ bandwithRatio)
=(ranksize_y -1)(ranksize_x - sumY)/(1+ bandwithRatio)
等效带宽的计算xyB = xB/( 1 +xdataRatio[最后一步斜对角数据] )
          xyB = xB(1+bandwithRatio)/(1+bandwithRatio +(ranksize_y -1)(ranksize_x - sumY))
*/

double CalcBandwidth2D(double xB, double yB, u32 xRankSize, u32 yRankSize, u32 maxStepNum, u32 &steps,
    double *xDataSize, double *yDataSize)
{
    HCCL_INFO("[CalcBandwidth2D] start");
    if (yRankSize == 1) {
        HCCL_INFO("[CalcBandwidth2D] xB=[%f]", xB);
        steps = 1;
        xDataSize[0] = 1;
        yDataSize[0] = 1;
        return xB;
    } else if (xRankSize == 1) {
        HCCL_INFO("[CalcBandwidth2D] yB=[%f]", yB);
        steps = 1;
        xDataSize[0] = 1;
        yDataSize[0] = 1;
        return yB;
    } else {
        // 根据数据量为1计算每步数据比例
        double bandwidthRatio = yB / xB; // 带宽比例
        double scale = 1.0;
        steps = CalcOmniPipeSteps(bandwidthRatio, xRankSize, maxStepNum, scale);
        CalcOmniPipeData(bandwidthRatio, xRankSize, yRankSize, steps, scale, xDataSize, yDataSize);
        double xds = 0;
        // 根据慢轴总时间计算等效带宽
        for (int i = 0; i < steps; i++) {
            xds = xds + xDataSize[i];
        }
        HCCL_INFO("[CalcBandwidth2D] Bandwidth2D=[%f]", xB / xds);
        return xB / xds;
    }
}

/* 数据递推公式
growth = (ranksize(x) - 1)/bandwithRatio
  - growth > 1：扇出压力大（rank 数多）→ 慢轴每步数据量递增
1+growth+growth^2+.......growth^(steps-2)  = bandwithRatio
steps = ceil(loggrowth((growth -1)*bandwithRatio + 1)) + 1  or max(5)
  - growth = 1：扇出和带宽比抵消 → 慢轴每步数据量不变
steps = bandwithRatio  + 1 or max(5);
  - growth < 1：带宽优势大 → 慢轴每步数据量递减
steps = max(5)
ydata[0] = slicecount
ydata[i] = xdata[i-1] * (ranksize(x) - 1)
ydata[steps - 1] = slicecount * bandwithRatio /(bandwithRatio + 1)
xdata[i] = (slicecount * growth^i)/bandwithRatio
xdata[steps -2] = slicecout -sum(xdata[i])
xdata[steps - 1] = slicecount /(bandwithRatio + 1)
*/
u32 CalcOmniPipeSteps(double bandwidthRatio, u32 xRankSize, u32 maxStep, double &scale)
{
    HCCL_INFO("[CalcOmniPipeSteps] start");
    u32 steps = 1;
    double growth = (xRankSize - 1) / bandwidthRatio;
    // 默认 scale：当 steps 被 maxStep 截断或 xRankSize <= bandwidthRatio 时启用，
    // 防止 sumYDataSzie 失控超过 xRankSize，导致斜对角步算出负值
    double geomSum = 0;
    for (u64 t = 0; t < maxStep - 1; t++) {
        geomSum += std::pow(growth, t);
    }
    scale = bandwidthRatio / geomSum;
    if (xRankSize > bandwidthRatio) {
        // X > R: 计算自然步数
        if (std::fabs(growth - 1.0) < 1e-4) {
            steps = bandwidthRatio + 1;
        } else {
            steps = ceil(std::log((growth - 1) * bandwidthRatio + 1) / std::log(growth)) + 1;
        }
        if (steps <= maxStep) {
            scale = 1.0; // 自然步数可容纳，无需缩放
        } else {
            steps = maxStep; // 截断到 maxStep，保持 scale
        }
    } else {
        // X <= R: growth < 1，走满 maxStep，保持 scale
        steps = maxStep;
    }
    HCCL_INFO("[CalcOmniPipeSteps] bandwidthRatio=[%f],growth=[%f],step=[%llu],scale=[%f]", bandwidthRatio, growth,
        steps, scale);
    return steps;
}

void CalcOmniPipeData(double bandwidthRatio, u32 xRankSize, u32 yRankSize, u32 steps, double scale,
    double *xStepP2pDataSize, double *yStepP2pDataSize)
{
    HCCL_INFO("[CalcOmniPipeData] start");
    // 1. 计算第一步的通信数据
    yStepP2pDataSize[0] = 1.0;
    xStepP2pDataSize[0] = scale / bandwidthRatio;
    double sumXDataSzie = xStepP2pDataSize[0];
    double sumYDataSzie = yStepP2pDataSize[0];
    // 2. 计算后续的通信数据
    for (u64 index = 1; index < steps - 1; index++) {
        if (index == steps - 2) {
            // 循环最后一轮特殊处理
            xStepP2pDataSize[index] = 1.0 - sumXDataSzie;
            yStepP2pDataSize[index] = bandwidthRatio * xStepP2pDataSize[index];
            sumXDataSzie += xStepP2pDataSize[index];
            sumYDataSzie += yStepP2pDataSize[index];
            continue;
        }
        yStepP2pDataSize[index] = xStepP2pDataSize[index - 1] * (xRankSize - 1);
        xStepP2pDataSize[index] = yStepP2pDataSize[index] / bandwidthRatio;
        sumXDataSzie += xStepP2pDataSize[index];
        sumYDataSzie += yStepP2pDataSize[index];
    }
    // 3. 剩余数据切分转发（斜对角步）
    xStepP2pDataSize[steps - 1] = (yRankSize - 1) * (xRankSize - sumYDataSzie) / (1 + bandwidthRatio);
    yStepP2pDataSize[steps - 1] = (yRankSize - 1) * (xRankSize - sumYDataSzie) * bandwidthRatio / (1 + bandwidthRatio);
    HCCL_INFO("[CalcOmniPipeData] end");
    return;
}