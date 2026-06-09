#!/usr/bin/env python3
import re
from collections import defaultdict

LOG_FILE = "/home/crx/code/hccl_three_sequence_rs/hccl_2039/log202606081040.log"
OUT_DIR = "/home/crx/code/hccl_three_sequence_rs/hccl_2039"

LAYER0_SIZE = 4
LAYER1_SIZE = 2
LAYER2_SIZE = 2
TOTAL_RANKS = LAYER0_SIZE * LAYER1_SIZE * LAYER2_SIZE

def rank_to_pos(rank):
    l0 = rank % LAYER0_SIZE
    l1 = (rank // LAYER0_SIZE) % LAYER1_SIZE
    l2 = rank // (LAYER0_SIZE * LAYER1_SIZE)
    return l0, l1, l2

def fmt_size(sz):
    if sz >= 0x100000:
        return f"{sz/0x100000:.1f}MB"
    elif sz >= 0x400:
        return f"{sz/0x400:.1f}KB"
    else:
        return f"{sz}B"

def fmt_off(off):
    return f"0x{off:X}"

def parse_log(filepath):
    ranks = {}
    current_rank = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'rankId is : (\d+)', line)
            if m:
                current_rank = int(m.group(1))
                ranks[current_rank] = {'level0': [], 'level1': [], 'level2': [], 'output': []}
                continue
            if current_rank is None:
                continue
            m2 = re.match(r'rankIdx:(\d+),\s*threadIdx:(\d+),\s*\[(.+)\]', line)
            if not m2:
                continue
            rank = int(m2.group(1))
            action = m2.group(3)
            if rank != current_rank:
                continue

            if 'LocalCopy' in action and 'INPUT' in action and 'CCL' in action:
                m3 = re.search(r'srcSlice=DataSlice\[BufferType::INPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*dstSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    ranks[current_rank]['level0'].append({
                        'type': 'LocalCopy', 'src_offset': int(m3.group(1),16), 'src_size': int(m3.group(2),16),
                        'dst_offset': int(m3.group(3),16), 'dst_size': int(m3.group(4),16),
                    })
            elif 'LocalReduce' in action:
                m3 = re.search(r'srcSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*dstSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    ranks[current_rank]['level0'].append({
                        'type': 'LocalReduce', 'src_offset': int(m3.group(1),16), 'src_size': int(m3.group(2),16),
                        'dst_offset': int(m3.group(3),16), 'dst_size': int(m3.group(4),16),
                    })
            elif 'WriteReduce' in action:
                m3 = re.search(r'remoteRank=(\d+).*localSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*remoteSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    remote = int(m3.group(1))
                    entry = {'type':'WriteReduce','remote_rank':remote,
                             'local_offset':int(m3.group(2),16),'local_size':int(m3.group(3),16),
                             'remote_offset':int(m3.group(4),16),'remote_size':int(m3.group(5),16)}
                    _, l1s, l2s = rank_to_pos(current_rank)
                    _, l1d, l2d = rank_to_pos(remote)
                    if l2s != l2d: ranks[current_rank]['level2'].append(entry)
                    elif l1s != l1d: ranks[current_rank]['level1'].append(entry)
                    else: ranks[current_rank]['level0'].append(entry)
            elif re.search(r'\[Write\]:', action) and 'INPUT' in action:
                m3 = re.search(r'remoteRank=(\d+).*localSlice=DataSlice\[BufferType::INPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*remoteSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    remote = int(m3.group(1))
                    entry = {'type':'Write','remote_rank':remote,
                             'local_offset':int(m3.group(2),16),'local_size':int(m3.group(3),16),
                             'remote_offset':int(m3.group(4),16),'remote_size':int(m3.group(5),16)}
                    _, l1s, l2s = rank_to_pos(current_rank)
                    _, l1d, l2d = rank_to_pos(remote)
                    if l2s != l2d: ranks[current_rank]['level2'].append(entry)
                    elif l1s != l1d: ranks[current_rank]['level1'].append(entry)
                    else: ranks[current_rank]['level0'].append(entry)
            elif 'LocalCopy' in action and 'CCL' in action and 'OUTPUT' in action:
                m3 = re.search(r'srcSlice=DataSlice\[BufferType::CCL, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\].*dstSlice=DataSlice\[BufferType::OUTPUT, offset=(0x[0-9a-fA-F]+), size=(0x[0-9a-fA-F]+)\]', action)
                if m3:
                    ranks[current_rank]['output'].append({
                        'type':'Output','src_offset':int(m3.group(1),16),'src_size':int(m3.group(2),16),
                        'dst_offset':int(m3.group(3),16),'dst_size':int(m3.group(4),16),
                    })
    return ranks

def gen_level0_dot(ranks):
    d = []
    d.append('digraph Level0_IntraServer {')
    d.append('  rankdir=LR;')
    d.append('  node [fontsize=10];')
    d.append('  edge [fontsize=8];')
    d.append('  compound=true;')
    d.append('')

    for l2 in range(LAYER2_SIZE):
        for l1 in range(LAYER1_SIZE):
            d.append(f'  subgraph cluster_s{l2}_{l1} {{')
            d.append(f'    label="Server {l2}-{l1} (Pod{l2})"; style=filled; fillcolor="#E8F5E9";')
            for l0 in range(LAYER0_SIZE):
                rank = l2*LAYER1_SIZE*LAYER0_SIZE + l1*LAYER0_SIZE + l0
                d.append(f'    R{rank} [label="R{rank}\\nD{l0}", shape=box, style=filled, fillcolor="#C8E6C9"];')
            d.append('  }')
    d.append('')

    for rank in range(TOTAL_RANKS):
        if rank not in ranks: continue
        l0, l1, l2 = rank_to_pos(rank)
        for i, e in enumerate(ranks[rank]['level0']):
            if e['type'] == 'LocalCopy':
                d.append(f'  R{rank}_IN{i} [label="INPUT\\n{fmt_off(e["src_offset"])} {fmt_size(e["src_size"])}", shape=ellipse, style=filled, fillcolor="#FFF9C4"];')
                d.append(f'  R{rank}_CCL{i} [label="CCL\\n{fmt_off(e["dst_offset"])} {fmt_size(e["dst_size"])}", shape=ellipse, style=filled, fillcolor="#E1BEE7"];')
                d.append(f'  R{rank}_IN{i} -> R{rank}_CCL{i} [label="Copy", color="#4CAF50"];')
            elif e['type'] == 'LocalReduce':
                d.append(f'  R{rank}_RS{i} [label="CCL\\n{fmt_off(e["src_offset"])} {fmt_size(e["src_size"])}", shape=ellipse, style=filled, fillcolor="#E1BEE7"];')
                d.append(f'  R{rank}_RD{i} [label="CCL\\n{fmt_off(e["dst_offset"])} {fmt_size(e["dst_size"])}", shape=ellipse, style=filled, fillcolor="#CE93D8"];')
                d.append(f'  R{rank}_RS{i} -> R{rank}_RD{i} [label="Reduce", color="#FF9800", style=dashed];')
            elif e['type'] == 'Write':
                remote = e['remote_rank']
                d.append(f'  R{rank} -> R{remote} [label="Write\\nINPUT {fmt_off(e["local_offset"])} {fmt_size(e["local_size"])}→CCL {fmt_off(e["remote_offset"])}", color="#4CAF50", style=dotted];')

        for e in ranks[rank]['output']:
            d.append(f'  R{rank}_OUT [label="OUTPUT\\n{fmt_size(e["src_size"])}", shape=doubleoctagon, style=filled, fillcolor="#BBDEFB"];')
            d.append(f'  R{rank} -> R{rank}_OUT [label="Output", color="#1565C0"];')

    d.append('}')
    return '\n'.join(d)

def gen_level1_dot(ranks):
    d = []
    d.append('digraph Level1_InterServer {')
    d.append('  rankdir=TB;')
    d.append('  node [fontsize=10];')
    d.append('  edge [fontsize=8];')
    d.append('')

    for l2 in range(LAYER2_SIZE):
        d.append(f'  subgraph cluster_p{l2} {{')
        d.append(f'    label="Pod {l2}"; style=filled; fillcolor="#E3F2FD";')
        for l1 in range(LAYER1_SIZE):
            d.append(f'    subgraph cluster_s{l2}_{l1} {{')
            d.append(f'      label="Server {l2}-{l1}"; style=filled; fillcolor="#BBDEFB";')
            for l0 in range(LAYER0_SIZE):
                rank = l2*LAYER1_SIZE*LAYER0_SIZE + l1*LAYER0_SIZE + l0
                d.append(f'      R{rank} [label="R{rank}\\n(S{l2}-P{l1}-D{l0})", shape=box, style=filled, fillcolor="#90CAF9"];')
            d.append('    }')
        d.append('  }')
    d.append('')

    for rank in range(TOTAL_RANKS):
        if rank not in ranks: continue
        for e in ranks[rank]['level1']:
            remote = e['remote_rank']
            if e['type'] == 'WriteReduce':
                d.append(f'  R{rank} -> R{remote} [label="WriteReduce\\nCCL {fmt_off(e["local_offset"])} {fmt_size(e["local_size"])}→CCL {fmt_off(e["remote_offset"])}", color="#2196F3", penwidth=2];')

    d.append('}')
    return '\n'.join(d)

def gen_level2_dot(ranks):
    d = []
    d.append('digraph Level2_InterPod {')
    d.append('  rankdir=TB;')
    d.append('  node [fontsize=10];')
    d.append('  edge [fontsize=8];')
    d.append('')

    for l2 in range(LAYER2_SIZE):
        d.append(f'  subgraph cluster_p{l2} {{')
        d.append(f'    label="Pod {l2}"; style=filled; fillcolor="#FBE9E7";')
        for l1 in range(LAYER1_SIZE):
            d.append(f'    subgraph cluster_s{l2}_{l1} {{')
            d.append(f'      label="Server {l2}-{l1}"; style=filled; fillcolor="#FFCCBC";')
            for l0 in range(LAYER0_SIZE):
                rank = l2*LAYER1_SIZE*LAYER0_SIZE + l1*LAYER0_SIZE + l0
                d.append(f'      R{rank} [label="R{rank}\\n(S{l2}-P{l1}-D{l0})", shape=box, style=filled, fillcolor="#FFAB91"];')
            d.append('    }')
        d.append('  }')
    d.append('')

    for rank in range(TOTAL_RANKS):
        if rank not in ranks: continue
        for e in ranks[rank]['level2']:
            remote = e['remote_rank']
            if e['type'] == 'WriteReduce':
                d.append(f'  R{rank} -> R{remote} [label="WriteReduce\\nCCL {fmt_off(e["local_offset"])} {fmt_size(e["local_size"])}→CCL {fmt_off(e["remote_offset"])}", color="#FF5722", penwidth=3];')

    d.append('}')
    return '\n'.join(d)

def gen_combined_dot(ranks):
    d = []
    d.append('digraph Combined_3Level {')
    d.append('  rankdir=TB;')
    d.append('  node [fontsize=9];')
    d.append('  edge [fontsize=7];')
    d.append('  newrank=true;')
    d.append('')

    for l2 in range(LAYER2_SIZE):
        d.append(f'  subgraph cluster_p{l2} {{')
        d.append(f'    label="Pod {l2}"; style=filled; fillcolor="#F5F5F5";')
        for l1 in range(LAYER1_SIZE):
            d.append(f'    subgraph cluster_s{l2}_{l1} {{')
            d.append(f'      label="Server {l2}-{l1}"; style=filled; fillcolor="#E8F5E9";')
            for l0 in range(LAYER0_SIZE):
                rank = l2*LAYER1_SIZE*LAYER0_SIZE + l1*LAYER0_SIZE + l0
                d.append(f'      R{rank} [label="R{rank}\\nD{l0}", shape=box, style=filled, fillcolor="#C8E6C9"];')
            d.append('    }')
        d.append('  }')
    d.append('')

    d.append('  // Level0: Intra-Server Write (dotted green)')
    for rank in range(TOTAL_RANKS):
        if rank not in ranks: continue
        for e in ranks[rank]['level0']:
            if e['type'] == 'Write':
                remote = e['remote_rank']
                d.append(f'  R{rank} -> R{remote} [label="L0-Write {fmt_size(e["local_size"])}", color="#4CAF50", style=dotted, penwidth=1];')

    d.append('')
    d.append('  // Level1: Inter-Server WriteReduce (blue)')
    for rank in range(TOTAL_RANKS):
        if rank not in ranks: continue
        for e in ranks[rank]['level1']:
            remote = e['remote_rank']
            if e['type'] == 'WriteReduce':
                d.append(f'  R{rank} -> R{remote} [label="L1-WriteReduce {fmt_size(e["local_size"])}", color="#2196F3", penwidth=2];')

    d.append('')
    d.append('  // Level2: Inter-Pod WriteReduce (red, bold)')
    for rank in range(TOTAL_RANKS):
        if rank not in ranks: continue
        for e in ranks[rank]['level2']:
            remote = e['remote_rank']
            if e['type'] == 'WriteReduce':
                d.append(f'  R{rank} -> R{remote} [label="L2-WriteReduce {fmt_size(e["local_size"])}", color="#FF5722", penwidth=3];')

    d.append('')
    d.append('  // Output')
    for rank in range(TOTAL_RANKS):
        if rank not in ranks: continue
        for e in ranks[rank]['output']:
            d.append(f'  R{rank}_OUT [label="R{rank} OUT\\n{fmt_size(e["src_size"])}", shape=doubleoctagon, style=filled, fillcolor="#BBDEFB"];')
            d.append(f'  R{rank} -> R{rank}_OUT [label="Output", color="#1565C0", style=dashed];')

    d.append('')
    d.append('  // Legend')
    d.append('  subgraph cluster_legend {')
    d.append('    label="Legend"; style=filled; fillcolor="#FAFAFA";')
    d.append('    leg0 [label="Level0: Intra-Server\\n(ZAxisDetour)", shape=box, fillcolor="#C8E6C9", style=filled];')
    d.append('    leg1 [label="Level1: Inter-Server\\n(NHR)", shape=box, fillcolor="#BBDEFB", style=filled];')
    d.append('    leg2 [label="Level2: Inter-Pod\\n(NHR)", shape=box, fillcolor="#FFCCBC", style=filled];')
    d.append('    leg0 -> leg1 [label="L1-WriteReduce", color="#2196F3", penwidth=2];')
    d.append('    leg1 -> leg2 [label="L2-WriteReduce", color="#FF5722", penwidth=3];')
    d.append('  }')

    d.append('}')
    return '\n'.join(d)

def main():
    ranks = parse_log(LOG_FILE)
    print(f"Parsed {len(ranks)} ranks")

    for name, gen_func in [("level0", gen_level0_dot), ("level1", gen_level1_dot), ("level2", gen_level2_dot), ("combined", gen_combined_dot)]:
        content = gen_func(ranks)
        path = f"{OUT_DIR}/data_flow_{name}.dot"
        with open(path, 'w') as f:
            f.write(content)
        print(f"Wrote {path}")

    print("\nGenerate SVG with: dot -Tsvg data_flow_level0.dot -o data_flow_level0.svg")
    print("Or install graphviz: apt-get install graphviz")

if __name__ == "__main__":
    main()
