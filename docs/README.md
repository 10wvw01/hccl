# HCCL参考资料

## 产品文档

| 文档                                                         | 面向对象                                                     | 内容介绍                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [集合通信用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg) | 使用集合通信功能，基于CANN-toolkit包进行通信功能开发的用户。 | HCCL基本架构、集合通信原语、分级通信原理介绍，以及如何基于hccl软件接口进行通信域管理、点对点通信与集合通信功能开发。 |
| [集合通信源码定制开发指南（chm）](https://cann-doc.obs.cn-north-4.myhuaweicloud.com/hccl/%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E6%BA%90%E7%A0%81%E5%AE%9A%E5%88%B6%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97.chm) <br> [集合通信源码定制开发指南（pdf）](https://cann-doc.obs.cn-north-4.myhuaweicloud.com/hccl/%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E6%BA%90%E7%A0%81%E5%AE%9A%E5%88%B6%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97.pdf) | 基于cann-hccl源码仓进行通信算法定制开发、通信算子定制开发的用户。 | HCCL软件、算法、算子执行流程，源码定制开发背景知识，通信算法与通信算子的开发流程介绍，通信平台接口参考。 |
| [环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef) | 调测集合通信功能的用户。                                     | HCCL对外提供的环境变量，辅助进行集合通信功能的配置与调测。   |
| [HCCL性能测试工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest) | 进行HCCL算子性能测试的用户。                                 | 介绍如何基于HCCL Test工具上板进行集合通信算子的性能测试。    |

> 说明：若您无法正常打开chm文件，可能是本地计算机对文件做了限制锁定，以下方法可供参考：
>
> ​           右键单击“集合通信源码定制开发指南.chm”文档，选择“属性”，勾选右下角的“解除锁定”。
>
> ![chm设置](https://foruda.gitee.com/images/1752568389110465380/9c06728e_5474059.png "屏幕截图")

## 算法介绍
参见cann-hccl源码仓[README中的“算法介绍”](../README.md#算法介绍)。  

## 技术文章
- [HCCL—昇腾高性能集合通信库简介](https://www.hiascend.com/zh/developer/techArticles/20240809-1)
- [HCCL集合通信算法开发Hello World示例](https://www.hiascend.com/zh/developer/techArticles/20240903-1)
- [HCCL集合通信常见问题定位思路](https://www.hiascend.com/zh/developer/techArticles/20240930-1)
- [深度学习的分布式训练与集合通信（一）](https://www.hiascend.com/zh/developer/techArticles/20241111-1)
- [深度学习的分布式训练与集合通信（二）](https://www.hiascend.com/zh/developer/techArticles/20241122-1)

## 培训视频

- [昇腾集合通信系列教程——什么是HCCL](https://www.bilibili.com/video/BV1MMmWYkEfS/?spm_id_from=333.999.0.0)
- [昇腾集合通信系列教程——常见集合通信原语](https://www.bilibili.com/video/BV1YtUWYQEq2/?spm_id_from=333.999.0.0)
- [昇腾集合通信系列教程——集合通信典型算法](https://www.bilibili.com/video/BV1XEz5YeE3A/?spm_id_from=333.999.0.0)
- [昇腾集合通信系列教程——集合通信业务开发](https://www.bilibili.com/video/BV1rfzoYTE3W/?spm_id_from=333.999.0.0)

- [HCCL设计原理和实现系列（一）通信算子调用、HCCL软件架构和整体流程](https://www.bilibili.com/video/BV1w4sVeFEsB/?spm_id_from=333.1387.search.video_card.click&vd_source=3c01cf8ec9742857cd7979d5b1afa0bf)
- [HCCL设计原理和实现系列（二）以AllGather算子为例，通信框架实现代码精讲](https://www.bilibili.com/video/BV1HZD7YREtd/?spm_id_from=333.1387.search.video_card.click&vd_source=3c01cf8ec9742857cd7979d5b1afa0bf)
- [HCCL设计原理和实现系列（三）以AllGather算子为例，Mesh算法实现代码精讲](https://www.bilibili.com/video/BV1ccDEYXEAS/?spm_id_from=333.1387.search.video_card.click&vd_source=3c01cf8ec9742857cd7979d5b1afa0bf)
- [HCCL设计原理和实现系列（四)以AllGather算子为例介绍RHD算法实现代码](https://www.bilibili.com/video/BV1kKy6YTEYM/?spm_id_from=333.1387.search.video_card.click&vd_source=3c01cf8ec9742857cd7979d5b1afa0bf)
- [HCCL设计原理和实现系列（五）以AllGather&AllToAll算子为例介绍Ring&Pairwise算法实现代码](https://www.bilibili.com/video/BV1kzDEYdEJo/?spm_id_from=333.1387.search.video_card.click&vd_source=3c01cf8ec9742857cd7979d5b1afa0bf)
- [HCCL设计原理和实现系列（六）-AI大集群功能性能问题定位分析实操讲解](https://www.bilibili.com/video/BV1ouBMYpEJC/?spm_id_from=333.1387.search.video_card.click&vd_source=3c01cf8ec9742857cd7979d5b1afa0bf)