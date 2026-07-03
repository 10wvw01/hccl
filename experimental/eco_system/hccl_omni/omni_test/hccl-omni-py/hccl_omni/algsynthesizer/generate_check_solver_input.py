#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import json
import os
import subprocess
import sys
import argparse
from enum import Enum
from typing import DefaultDict, Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from .hcclgen.hcclgen.api import generate_hccl_xml

# TODO:多节点,拓扑种类，带宽
@dataclass
class solverParam:
    Name:str
    chunksize:int
    collective:str
    reduce_op: str = ""
    size: int = 0  #ranksize
    ranks: int = 0
@dataclass
class clusterConfig:
    name:str
    rootinfo_path: str
    ranktable_path:str # 输入ranktable 路径，文件路径
    topo_path:str # 输入topo路径，文件路径
    topo_output_path:str = "" #求解器输入topo json路径，文件路径 = 求解器json input path
    json2xml_path:str = ""       # 求解器输出xml所在目录(不是xml文件路径)
    xml_path: str = "" # 最终xml的输出路径
    in_bandwidth:int = 50
    switch_bandwidth:int = 50
    auto_generate:bool = True

def load_json(filepath: str) -> Dict[str, Any]:
    """加载json 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_topo_type(rootinfo: Dict[str, Any]) -> Optional[str]:
    """从 rootinfo 中获取拓扑类型"""
    rank_list = rootinfo.get('rank_list', [])
    if not rank_list:
        return None
    for rank in rank_list:
        level_list = rank.get('level_list', [])
        if level_list:
            return level_list[0].get('net_type')
    return None


from typing import List, Dict, Any
from collections import defaultdict

def get_ranktable_ranks(ranktable):
    server_count=ranktable.get("server_count")
    assert int(server_count) == 1,"server_count current not suppurt count > 1"
    server_list=ranktable.get("server_list")
    devices=server_list[0].get("device")
    rank_count=0
    ranks=[]
    for device in devices:
        ranks.append(int(device.get("rank_id")))
        rank_count=+1
    return rank_count,ranks

def build_adjacency_matrix(edge_list: List[Dict[str, Any]], ranktable: dict, net_layer: int) -> List[List[int]]:
    # 1. 构建拓扑实例字典 (topo_inst[topo_type][topo_inst_id] = [rank_list])
    # 使用 defaultdict 简化初始化逻辑
    rank_count,ranks=get_ranktable_ranks(ranktable)
    topo_inst = defaultdict(lambda: defaultdict(list))

    for link in edge_list:
        link_net_layer = link.get("net_layer")
        topo_type = link.get("topo_type")
        src = link.get("local_a") # src
        dst =link.get("local_b",ranks[0])
        topo_inst_id = link.get("topo_instance_id")
        topo_type=topo_type.upper()
        if (src not in ranks) or (dst not in ranks) or topo_type =="CLOS":
            continue
        # 只处理 net_layer == 0
        if link_net_layer == 0:
            # 如果该类型下没有这个实例ID，初始化空列表
            if topo_inst_id not in topo_inst[topo_type]:
                topo_inst[topo_type][topo_inst_id] = [] #(type,inst_id)=>ranks
            
            # 避免重复添加同一个 rank 
            if src not in topo_inst[topo_type][topo_inst_id]:
                topo_inst[topo_type][topo_inst_id].append(src)

            if dst not in topo_inst[topo_type][topo_inst_id]:
                topo_inst[topo_type][topo_inst_id].append(dst)
    # 2. 初始化返回结果和统计变量
    # 假设返回结构为：(所有实例的邻接矩阵列表, 统计信息字典)
    all_adjacency_matrices = [] # 包含clos

    # 3. 遍历所有实例构建矩阵并统计
    for topo_type, instances in topo_inst.items():
        for inst_id, ranks in instances.items():
            matrix = [[0 for _ in range(rank_count)] for _ in range(rank_count)]
            for r in ranks:
                # 边界检查：防止 rank 超出 rank_count
                if 0 <= r < rank_count and topo_type=="1DMESH":
                    matrix[r][r] = 1  # 示例：对角线置 1 表示该 rank 存在
            
            all_adjacency_matrices.append(matrix)

    return all_adjacency_matrices, topo_inst

def build_clos_edges_and_matrix(
    edge_list: List[Dict[str, Any]], 
    ranktable: dict ,
    net_layer: int = 0
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """
    构建 Clos 类型实例的邻接矩阵并提取其所有可用边。
    
    参数:
    - edge_list: 原始边列表
    - rank_count: 每个 Rank 节点的总数
    - net_layer: 仅处理该层级的边 (默认 0)
    
    返回:
    - (all_clos_matrices, all_clos_edges)
      - all_clos_matrices: Clos 实例的邻接矩阵列表
      - all_clos_edges: Clos 类型的所有可用边字典列表
    """
    
    # 1. 构建拓扑实例字典 (仅关注 Clos 类型)
    rank_count,ranks=get_ranktable_ranks(ranktable)
    topo_clos_inst = defaultdict(list)

    for link in edge_list:
        link_net_layer = link.get("net_layer")
        topo_type = link.get("topo_type")
        rank = link.get("local_a")
        inst_id = link.get("topo_instance_id")
        net_layer=link.get("net_layer")

        if rank not in ranks:
            continue

        # 核心过滤条件：必须是 Clos 类型 且 在指定网络层
        if topo_type.upper() == "CLOS":
            # 确保实例 ID 对应的列表存在
            if (inst_id,net_layer) not in topo_clos_inst:
                topo_clos_inst[(inst_id,net_layer)] = []
            
            # 避免重复添加同一个 rank
            if rank not in topo_clos_inst[(inst_id,net_layer)]:
                topo_clos_inst[(inst_id,net_layer)].append(rank)
    # 返回结果
    return topo_clos_inst

def build_topo_json_from_topo(edge_list,rootinfo,ranktable,config,param):
    rank_count,all_ranks=get_ranktable_ranks(ranktable)
    num_servers=ranktable.get("server_count")
    if int(num_servers) > 1:
        raise ValueError("current generator input json not support multi node")
    server_list=ranktable.get("server_list")
    num_ranks=0
    node_ranks = []
    for server in server_list:
        num_ranks=+len(server.get("device"))
        node_ranks.append(len(server.get("device")))
    # 构建该层的邻接图
    all_adjacency_matrix,topo_insts = build_adjacency_matrix(edge_list, ranktable, net_layer=0)
    clos=build_clos_edges_and_matrix(edge_list,ranktable)
    base_name = f"{config.name}"
    level0 = {
        "groups": [
            {
                "group_id": idx,
                "ranks": ranks,
                "type": topo_type,
                "bandwidth": config.in_bandwidth,
                "netlayer": 0
            }
            for topo_type, instances in topo_insts.items() 
            for idx, (inst_id, ranks) in enumerate(instances.items())
        ]
    }
    level1 = {
        "groups": [
            {
                "group_id": id,
                "ranks": ranks,
                "type": "Clos",
                "bandwidth": config.switch_bandwidth,
                "netlayer": int(netlayer)
            }
            for id, ((ins_id, netlayer), ranks) in enumerate(clos.items())
        ]
    }
    cluster={
        "level0":level0,
        "level1":level1
    }
    
    topo = {
            "name": base_name,
            "rank_base":all_ranks[0],
            "num_node":int(num_servers),
            "total_rank_num":num_ranks,
            "num_rank_node":node_ranks,
            "cluster":cluster,
            "collective":{
                "type":param.collective,
                "chunksize":param.chunksize
            },
            "output":{
                "hccl_xml":config.xml_path
            }
        }
    return topo

def generate_input_config_topo_json(rootinfo,ranktable,topo,config,Param):
    # 自动生成:只支持生成同构简单拓扑，异构拓扑不予支持，大于等于三层的拓扑暂时不支持
    rank_count = rootinfo.get('rank_count', 0)
    rank_list = rootinfo.get('rank_list',[])

    topo_type = get_topo_type(rootinfo)
    if topo_type == "TOPO_FILE_DESC" and topo:
        edge_list = topo.get('edge_list', [])
        #output_config = build_config_json_from_topo(edge_list, rootinfo ,ranktable,config, Param)
        output_topo = build_topo_json_from_topo(edge_list, rootinfo ,ranktable,config, Param)
    else:
        raise ValueError("topo type in rootinfo only support TOPO_FILE_DESC")
    return output_topo
def check_input_jsons(config,Param):
    # 检验用户提供的config和topo的json,只管关键值写没写不管写什么，值正确性交由求解器检验
    #------------------------------第一步:检查config----------------------------------------------
    topo=load_json(config.topo_output_path)
    num_node=topo.get("num_node")
    total_rank_num=topo.get("total_rank_num")
    num_rank_node=topo.get("num_rank_node")
    assert total_rank_num == sum(num_rank_node),"pls check your topo.json:ranks num is not right"
    cluster=topo.get("cluster")
def generate_solver_xml(param, config):
    """
    Generates solver input files (config.json and topo.json) if auto_generate is enabled.
    """
    # Set output paths with defaults if empty
    config.topo_output_path = config.topo_output_path if config.topo_output_path else os.path.join(config.json2xml_path, f"{config.name}{param.collective}_topo.json")
    print(f"Loading rootinfo: {config.rootinfo_path}")
    rootinfo = load_json(config.rootinfo_path)
    if config.ranktable_path != "":
        ranktable=load_json(config.ranktable_path)
    else:
        raise ValueError("lacks ranktable")
    topo_type = get_topo_type(rootinfo)
    topo = None

    if topo_type == "TOPO_FILE_DESC":
        if os.path.exists(config.topo_path):
            print(f"Loading topo: {config.topo_path}")
            topo = load_json(config.topo_path)
        else:
            print(f"Warning: Topo file not found: {config.topo_path}. Using default topology.")
    else:
        print(f"Topo type: {topo_type}. Using default topology rules.")

    # Generate fabric_groups/v1 format
    print("Generating solver topo...")
    output_topo = generate_input_config_topo_json(rootinfo,ranktable, topo, config, param)


    print(f"Writing topo.json: {config.topo_output_path}")
    with open(config.topo_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_topo, f, indent=2, ensure_ascii=False)

    print("Conversion completed!")
    #check_input_jsons(config, param)


def generate_xml(param, config):
    """
    Main workflow: generates solver input, runs collgen, and converts JSON to XML.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(config.json2xml_path):
        try:
            os.makedirs(config.json2xml_path)
            print(f"Created output directory: {config.json2xml_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory {config.json2xml_path}: {e}")

    # Step 1: Generate solver input (json)
    generate_solver_xml(param, config)

    # Step 2: Run hcclgen and return xml
    generate_hccl_xml(config.topo_output_path)
