#!/bin/bash
# B2 rerun: the first attempt crashed on KeyError 'E1D' (train.py inline exp map
# missing E1D — now fixed). Wait for B3 (incl. eval) to finish, then run B2.
cd "$(dirname "$0")/.."
while ! grep -aq "B3 eval complete" outputs/b3_ablation.log 2>/dev/null; do
  sleep 120
done
echo "$(date) B3 done, restarting B2"
bash scripts/run_b2_dcn_sweep.sh > outputs/b2_sweep.log 2>&1
echo "$(date) B2 done"
