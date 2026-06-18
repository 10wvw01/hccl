# Reduce Scatter 3-Level Sequence Executor ST Test Case Design

## 1. Test Framework Overview

### 1.1 TopoMeta Structure

`TopoMeta = vector<SuperPodMeta>`, where `SuperPodMeta = vector<ServerMeta>`, `ServerMeta = vector<PhyDeviceId>`

3-level topology mapping:
- Outer vector (SuperPod): Level2 / cross-Pod
- Middle vector (Server): Level1 / intra-Pod
- Inner vector (PhyDeviceId): Level0 / intra-Server

`Build3LevelTopo(L0, L1, L2)` constructs TopoMeta as:
```
{cluster0: {pod0: [dev0..devL0-1], pod1: ..., podL1-1: ...},
 cluster1: {pod0: ..., ...},
 ...,
 clusterL2-1: {...}}
```

### 1.2 Sim Framework

- Device type: `DEV_TYPE_950` (910D/A5), supports L0/L1/L2 3-layer links
- NetLayer assignment: `podNum > 1` -> `netLayerList_ = {0, 1, 2}` (3 layers)
- Level0 links: same-server, UBC_CTP mesh
- Level1 links: same-Pod cross-server, UBC_CTP (or ROCE for hostdpu)
- Level2 links: cross-Pod, ROCE
- Environment: `HCCL_OP_EXPANSION_MODE=AI_CPU`, `HCCL_INDEPENDENT_OP=1`

### 1.3 Verification

Uses `CheckReduceScatter(taskQueues, rankSize, dataType, dataCount, reduceOp)` to verify semantics correctness.

---

## 2. Test Case List

### 2.1 P0 Priority (Must Run)

| # | Test Name | Topo (L0×L1×L2) | Data | DataType | ReduceOp | Coverage |
|---|-----------|------------------|------|----------|----------|----------|
| 1 | `st_reduce_scatter_3level_8x8x2_fp32_sum_basic` | 8×8×2=128 | recvCount=200 | FP32 | SUM | AE1: 3-level basic correctness on 128-card topology |
| 2 | `st_reduce_scatter_3level_8x8x2_fp32_sum_large_multi_loop` | 8×8×2=128 | recvCount=128K | FP32 | SUM | FR-03: multi-loop correctness (loopTimes > 1) |
| 3 | `st_reduce_scatter_3level_8x8x3_fp32_sum_repeatnum_gt1` | 8×8×3=192 | recvCount=200 | FP32 | SUM | P0 fix: outputRepeatStride>0, repeatNum=3 verification |
| 9 | `st_reduce_scatter_3level_8x8x2_int8_sum_multi_loop_extreme` | 8×8×2=128 | recvCount=400MB | INT8 | SUM | FR-03: extreme data multi-loop, loopTimes>>1 |
| 16 | `st_reduce_scatter_2level_backward_compat_meshnhr` | 8×8=16 (2-level) | recvCount=200 | INT32 | SUM | AC-04: backward compatibility, 2-level behavior unchanged |

### 2.2 P1 Priority (Important)

| # | Test Name | Topo (L0×L1×L2) | Data | DataType | ReduceOp | Coverage |
|---|-----------|------------------|------|----------|----------|----------|
| 5 | `st_reduce_scatter_3level_4x4x2_int32_max_different_scale` | 4×4×2=32 | recvCount=500 | INT32 | MAX | Different topology scale correctness |
| 6 | `st_reduce_scatter_3level_8x4x2_int16_min_asymmetric_mid` | 8×4×2=64 | recvCount=1000 | INT16 | MIN | Asymmetric middle layer (Level1) |
| 10 | `st_reduce_scatter_3level_4x4x2_fp32_sum_multi_loop` | 4×4×2=32 | recvCount=400MB | FP32 | SUM | Small-scale large-data loop segmentation |
| 11 | `st_reduce_scatter_3level_8x2x2_fp32_sum_small_cluster` | 8×2×2=32 | recvCount=200 | FP32 | SUM | Small cluster, boundary rank verification |

### 2.3 P2 Priority (Supplementary)

| # | Test Name | Topo (L0×L1×L2) | Data | DataType | ReduceOp | Coverage |
|---|-----------|------------------|------|----------|----------|----------|
| 4 | `st_reduce_scatter_3level_8x8x4_int32_sum_repeatnum4` | 8×8×4=256 | recvCount=200 | INT32 | SUM | Higher repeatNum (repeatNum=4) |
| 7 | `st_reduce_scatter_3level_8x8x2_fp16_sum_dtype` | 8×8×2=128 | recvCount=500K | FP16 | SUM | FP16 data type |
| 8 | `st_reduce_scatter_3level_8x8x2_bfp16_max_dtype` | 8×8×2=128 | recvCount=300 | BFP16 | MAX | BFP16 data type |
| 12 | `st_reduce_scatter_3level_4x2x2_int8_sum_corner` | 4×2×2=16 | recvCount=100 | INT8 | SUM | Extremely small topology |
| 13 | `st_reduce_scatter_3level_2x2x2_int32_prod_minimal` | 2×2×2=8 | recvCount=100 | INT32 | PROD | Minimal 3-level topology + PROD op |
| 14 | `st_reduce_scatter_3level_8x2x3_fp32_sum_level2_3cluster` | 8×2×3=48 | recvCount=200 | FP32 | SUM | Level2 has 3 clusters (repeatNum=3) |
| 15 | `st_reduce_scatter_3level_4x3x2_int32_sum_asymmetric_all` | 4×3×2=24 | recvCount=200 | INT32 | SUM | Fully asymmetric dimensions |

---

## 3. Coverage Matrix

### 3.1 Requirement Coverage

| Requirement | Test Case(s) |
|-------------|--------------|
| R1 (3-level executor class) | #1, #3, #4, #14 |
| R2 (linear serial 3-step data flow) | #1, #2, #9 |
| R3 (loop slicing for large data) | #2, #9, #10 |
| R4 (REGISTER_EXEC_V2_MULTI registration) | All 3-level tests (implicit) |
| R5 (3-level resource merge) | #1, #3 (implicit via successful execution) |
| R6 (Level0 template params, repeatNum=L1*L2) | #1, #3, #4, #14 |
| R7 (Level1 template params, repeatNum=L2) | #1, #3, #4 |
| R8 (Level2 template params, repeatNum=1) | All 3-level tests |
| R9 (Level2 template params, repeatNum=1) | All 3-level tests |
| R10 (TopoMatchMultilevel 3-layer extension) | All 3-level tests |
| R11 (Selector 3-level routing) | All 3-level tests (implicit) |

### 3.2 Acceptance Criteria Coverage

| AC | Test Case(s) |
|----|--------------|
| AC-01 (functional correctness) | #1, #3, #9 |
| AC-02 (resource merge) | #1 (implicit) |
| AC-03 (selector routing) | All 3-level tests (implicit, requires topoLevelNums>=3 triggers SequenceMesh1DNHRNHR) |
| AC-04 (backward compat) | #16 |

### 3.3 Code Review Gap Coverage

| Gap # | Description | Test Case(s) |
|-------|-------------|--------------|
| #1 | No 3-level e2e correctness test | #1 |
| #2 | No outputRepeatStride fix verification (repeatNum>1) | #3, #4, #14 |
| #3 | No multi-loop scenario verification | #2, #9, #10 |
| #4 | No Level1→Level2 buffer layout alignment verification | #1, #11 (implicit via e2e) |
| #5 | No mesh2d topology misrouting negative test | Not covered in ST (needs UT) |
| #6 | No TopoForLayer2 single-rank degradation test | Not covered (needs topology where linkNum=0 filters all but self; requires special stub) |

---

## 4. Topology Configuration Details

### 4.1 Build3LevelTopo Parameter Mapping

```
TopoMeta = {
  cluster0: [{dev0..7}, {dev0..7}, ..., {dev0..7}],  // L1 servers in cluster0
  cluster1: [{dev0..7}, {dev0..7}, ..., {dev0..7}],  // L1 servers in cluster1
  ...
}
```

For `Build3LevelTopo(8, 8, 2)`:
- 2 SuperPods (cluster0, cluster1)
- Each SuperPod has 8 Servers (pod0..pod7)
- Each Server has 8 Devices (dev0..dev7)
- Total: 2×8×8 = 128 ranks
- `topoLevelNums = 3` (netLayerList_ = {0, 1, 2})
- Rank mapping: `rankId = cluster*L0*L1 + pod*L0 + dev`

### 4.2 TopoModel NetLayer Logic

| Condition | netLayerList_ |
|-----------|---------------|
| serverNum == 1 | {0} (1 layer) |
| serverNum > 1 && podNum == 1 | {0, 1} (2 layers) |
| podNum > 1 | {0, 1, 2} (3 layers) |

3-level tests require `podNum > 1`, i.e., at least 2 SuperPods in TopoMeta.

### 4.3 Link Model (910D)

| Layer | Link Scope | Protocol |
|-------|-----------|----------|
| L0 | Same Server | UBC_CTP (mesh row/col) |
| L1 | Same Pod, cross-Server | UBC_CTP or ROCE |
| L2 | Cross-Pod | ROCE |

---

## 5. Execution Instructions

### 5.1 Build

```bash
cd test/st/algorithm
bash build.sh
```

### 5.2 Run All 3-Level Tests

```bash
export HCCL_ST_TEST_FILTER="ST_REDUCE_SCATTER_3LEVEL_TEST.*"
./hccl_checker_ops_stest
```

### 5.3 Run P0 Tests Only

```bash
export HCCL_ST_TEST_FILTER="ST_REDUCE_SCATTER_3LEVEL_TEST.st_reduce_scatter_3level_8x8x2_fp32_sum_basic:ST_REDUCE_SCATTER_3LEVEL_TEST.st_reduce_scatter_3level_8x8x2_fp32_sum_large_multi_loop:ST_REDUCE_SCATTER_3LEVEL_TEST.st_reduce_scatter_3level_8x8x3_fp32_sum_repeatnum_gt1:ST_REDUCE_SCATTER_3LEVEL_TEST.st_reduce_scatter_3level_8x8x2_int8_sum_multi_loop_extreme:ST_REDUCE_SCATTER_3LEVEL_TEST.st_reduce_scatter_2level_backward_compat_meshnhr"
./hccl_checker_ops_stest
```

### 5.4 Run Specific Test

```bash
export HCCL_ST_TEST_FILTER="ST_REDUCE_SCATTER_3LEVEL_TEST.st_reduce_scatter_3level_8x8x2_fp32_sum_basic"
./hccl_checker_ops_stest
```

---

## 6. Files Changed

| File | Action | Description |
|------|--------|-------------|
| `test/st/algorithm/testcase/reduce_scatter_3level_testcase.cc` | **NEW** | 16 ST test cases for 3-level sequence executor |
| `test/st/algorithm/testcase/CMakeLists.txt` | **MODIFIED** | Added `reduce_scatter_3level_testcase.cc` to src_list |
| `test/st/algorithm/testcase/main.cc` | **MODIFIED** | Added `ST_REDUCE_SCATTER_3LEVEL_TEST.*` to default gtest filter; added `HCCL_ST_TEST_FILTER` env var support |

---

## 7. Outstanding Test Gaps (Requires UT, Not ST)

| Gap | Description | Recommendation |
|-----|-------------|----------------|
| mesh2d misrouting | mesh2d topology should not be routed to 3-level executor | UT: verify `infos[i].size() != 1` returns HCCL_E_INTERNAL |
| TopoForLayer2 single-rank degradation | When linkNum=0 filters all peers, infos[2] has only myRank | UT: verify single-rank Level2 doesn't crash |
| Selector routing verification | Verify selector returns "InsReduceScatterSequenceMesh1DNHRNHR" | UT: unit test for ReduceScatterAutoSelector::SelectAicpuAlgo |
| CalcRes defensive checks | channels.empty(), infos.size()!=3 | UT: verify error return paths |