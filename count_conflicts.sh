#!/bin/bash
cd /home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039
git diff --name-only --diff-filter=U | while IFS= read -r f; do
    n=$(grep -c '<<<<<<' "$f" 2>/dev/null || echo 0)
    echo "$f: $n"
done
