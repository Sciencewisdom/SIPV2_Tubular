#!/bin/bash
# B3-control: bs4 Scharr reference run. B3 ablations use bs4 (bs8 OOMs on 12GB),
# while the R1 reference used bs8. This control isolates batch size from the
# component effects: if bs4+Scharr still matches R1, the B3a Sobel collapse is
# attributable to the gradient operator, not the batch size.
set -e
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== B3-control: Scharr bs4 (default road config) ==="
$PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42 \
  --batch_size 4 --directions 16 \
  --output_dir outputs/b3_road_scharr_bs4_seed42

CKPT=$(ls outputs/b3_road_scharr_bs4_seed42/*/checkpoints/checkpoint_final.pth 2>/dev/null | head -1)
echo "=== eval scharr_bs4: $CKPT ==="
$PY scripts/eval_road_deterministic.py --checkpoint "$CKPT" --block_type sipv2_road \
  --output outputs/b3_eval_scharr_bs4.json
echo "B3 control complete"
