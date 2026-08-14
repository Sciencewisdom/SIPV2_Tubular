#!/bin/bash
# Chain: wait for B1 sweep to finish, then run B2 (DCN) and B3 (road ablation).
cd "$(dirname "$0")/.."
while ! grep -aq "B1 sweep complete" outputs/b1_sweep.log 2>/dev/null; do
  sleep 120
done
echo "$(date) B1 done, starting B2"
bash scripts/run_b2_dcn_sweep.sh > outputs/b2_sweep.log 2>&1
echo "$(date) B2 done, starting B3"
bash scripts/run_b3_road_ablation.sh > outputs/b3_ablation.log 2>&1
echo "$(date) B3 done"
