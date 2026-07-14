/**
 * execute_selector UT 桩函数
 * 为 Selector 函数的外部依赖及 TopoMatch 子类提供桩，避免引入大量生产代码依赖。
 */
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include <hccl/hccl_types.h>
#include <hccl/hccl_rank_graph.h>

#include "hccl_algorithm.h"
#include "alg_param.h"
#include "auto_selector_base.h"
#include "topo_match_base.h"
#include "topo_match_1d.h"
#include "topo_match_3_level.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"
#include "topo_match_squeeze_2d.h"

// ============================================================
// 通用桩：memset_s
// ============================================================
extern "C" errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (dest == nullptr || count > destMax) {
        return 1;
    }
    (void)memset(dest, c, count);
    return 0;
}

// ============================================================
// 日志桩
// ============================================================
bool IsErrorToWarn()
{
    return false;
}

// DLOG_DEBUG=0, DLOG_INFO=1, DLOG_WARN=2, DLOG_ERROR=3
// 返回 true 打开对应级别日志；这里 ERROR 及以上都打开
bool HcclCheckLogLevel(int logType, int)
{
    return logType >= 3;
}

// dlog_pub.h 声明的 weak 符号，UT 没链接真实实现，这里提供桩：vfprintf 输出到 stderr
extern "C" void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...)
{
    (void)moduleId;
    fprintf(stderr, "[HCCL][lvl=%d] ", level);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

namespace ops_hccl {

// ============================================================
// GetExternalInputHcclAlgoConfigAllType 桩
// 被 AutoSelectorBase::Select 调用，返回空 map 即可
// ============================================================
const std::map<HcclCMDType, std::vector<HcclAlgoType>> GetExternalInputHcclAlgoConfigAllType()
{
    return {};
}

// ============================================================
// Selector 函数外部依赖桩（全部返回 SUCCESS）
// ============================================================
HcclResult HcclCalcTopoInfo(HcclComm, OpParam &, std::unique_ptr<TopoInfoWithNetLayerDetails> &)
{
    return HCCL_SUCCESS;
}

HcclResult CheckAsymmetricTopoSupport(HcclCMDType, const TopoInfoWithNetLayerDetails *)
{
    return HCCL_SUCCESS;
}

HcclResult SetExecTimeout(const OpParam &)
{
    return HCCL_SUCCESS;
}

HcclResult SetMultipleDimensionSplitRatio(const OpParam &)
{
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel()
{
    return HCCL_SUCCESS;
}

HcclResult LoadAICPUKernel()
{
    return HCCL_SUCCESS;
}

// 通信域状态检查：返回 false 跳过状态检查分支
// 声明见 hccl_host_comm_dl.h (DECL_SUPPORT_FLAG 宏)
extern "C" bool HcommIsSupportHcclCommGetStatus(void)
{
    return false;
}

// HcclCommGetStatus 桩（HcommIsSupportHcclCommGetStatus 返回 false 时不会被调用，
// 但编译器仍生成符号引用，需提供定义以通过链接）
extern "C" HcclResult HcclCommGetStatus(const char *, HcclCommStatus *)
{
    return HCCL_SUCCESS;
}

// ============================================================
// load_kernel.h 中的全局变量定义（execute_selector.cc 引用 extern 声明）
// ============================================================
aclrtBinHandle g_binKernelHandle = nullptr;

// ============================================================
// TopoMatchBase 桩 (替代 topo_match_base.cc)
// ============================================================
TopoMatchBase::TopoMatchBase() = default;
TopoMatchBase::~TopoMatchBase() = default;
HcclResult TopoMatchBase::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}

// ============================================================
// TopoMatch 子类桩
// g_aicpuAllGatherAlgoMap 静态初始化构造 std::make_shared<TopoMatchXxx>()，
// 仅需链接构造/析构/MatchTopo 符号；UT 中不会调用 MatchTopo。
// ============================================================
TopoMatch1D::TopoMatch1D() : TopoMatchBase() {}
TopoMatch1D::~TopoMatch1D() {}
HcclResult TopoMatch1D::MatchTopo(HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}

TopoMatch3Level::TopoMatch3Level() : TopoMatchBase() {}
TopoMatch3Level::~TopoMatch3Level() {}
HcclResult TopoMatch3Level::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}

TopoMatchMultilevel::TopoMatchMultilevel() : TopoMatchBase() {}
TopoMatchMultilevel::~TopoMatchMultilevel() {}
HcclResult TopoMatchMultilevel::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}

TopoMatchUBX::TopoMatchUBX() : TopoMatchBase() {}
TopoMatchUBX::~TopoMatchUBX() {}
HcclResult TopoMatchUBX::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}
// topo_match_ubx.h 中声明的 virtual 函数，进入 vtable，必须提供定义
HcclResult TopoMatchUBX::TopoForLayer1(const HcclComm, uint32_t, const uint32_t,
    AlgHierarchyInfoForAllLevel &) const
{
    return HCCL_SUCCESS;
}

TopoMatchPcieMix::TopoMatchPcieMix() : TopoMatchBase() {}
TopoMatchPcieMix::~TopoMatchPcieMix() {}
HcclResult TopoMatchPcieMix::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}

TopoMatchSqueeze2D::TopoMatchSqueeze2D() : TopoMatchBase() {}
TopoMatchSqueeze2D::~TopoMatchSqueeze2D() {}
HcclResult TopoMatchSqueeze2D::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
