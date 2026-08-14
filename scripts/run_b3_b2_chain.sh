#!/bin/bash
# Rebuilt chain after B3 OOM fix: B3 (bs4) -> B3 eval -> B2 (DCN sweep).
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/run_b3_road_ablation.sh > outputs/b3_ablation.log 2>&1
echo "$(date) B3 done (incl. eval), starting B2"
bash scripts/run_b2_dcn_sweep.sh > outputs/b2_sweep.log 2>&1
echo "$(date) B2 done"
