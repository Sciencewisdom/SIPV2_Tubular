"""Recompute per-case DRIVE break counts for the λc=0 vs λc=0.1 comparison.

The paper reports p=0.078 for the E1+clDice vs E4+clDice break-count comparison
(4 validation cases). A two-sided Wilcoxon on the cached per-case values gives
p=0.375, so this script recomputes everything from the cached prediction npy
files to find the source of the discrepancy.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import wilcoxon, ttest_rel
from skimage.morphology import skeletonize
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "data/raw/DRIVE/training/1st_manual"
VAL_IDS = ["37_training", "38_training", "39_training", "40_training"]

RUNS = {
    "E1 λ=0":   "outputs/E1_size512_bs2_seed42",
    "E4 λ=0":   "outputs/E4_size512_bs2_seed42",
    "E1 λ=0.1": "outputs/p01_e1_lambda0.1_seed42/E1_size512_bs2_seed42",
    "E4 λ=0.1": "outputs/p01_e4_lambda0.1_seed42/E4_size512_bs2_seed42",
}


def n_components(binary):
    skel = skeletonize(binary > 0)
    _, n = ndimage.label(skel)
    return n


def main():
    gts = {}
    for vid in VAL_IDS:
        gif = sorted(GT_DIR.glob(f"{vid.split('_')[0]}*.gif"))
        gt = np.array(Image.open(gif[0]).convert("L")) > 127
        gts[vid] = gt

    breaks = {}
    for name, rel in RUNS.items():
        s = json.loads((ROOT / rel / "summary.json").read_text())
        th = s["final_metrics"].get("best_dice_threshold", 0.5)
        vals = []
        for vid in VAL_IDS:
            pred = np.load(ROOT / rel / "predictions" / f"{vid}_prob.npy")
            gt = gts[vid]
            # predictions are 512x512 center-crops of the 565x584 originals? check shape
            if pred.shape != gt.shape:
                # center-crop gt to pred size
                h, w = pred.shape
                y0 = (gt.shape[0] - h) // 2
                x0 = (gt.shape[1] - w) // 2
                gt = gt[y0:y0 + h, x0:x0 + w]
            n_p = n_components(pred > th)
            n_t = n_components(gt)
            vals.append(abs(n_p - n_t))
        breaks[name] = np.array(vals, dtype=float)
        cached = s.get("per_case", {}).get("skel_break_count_per_case")
        print(f"{name}: recomputed={vals}  cached={cached}  (th={th})")

    e1d = breaks["E1 λ=0.1"] - breaks["E1 λ=0"]
    e4d = breaks["E4 λ=0.1"] - breaks["E4 λ=0"]
    print("\nper-case Δ(λ0.1−λ0): E1", e1d, " E4", e4d)
    for lbl, a, b in [("raw @λ0.1", breaks["E1 λ=0.1"], breaks["E4 λ=0.1"]),
                      ("deltas", e1d, e4d)]:
        try:
            w = wilcoxon(a, b)
        except Exception as ex:
            w = ex
        t = ttest_rel(a, b)
        print(f"{lbl}: wilcoxon={w}  ttest p={t.pvalue:.4f} (one-sided {t.pvalue/2:.4f})")


if __name__ == "__main__":
    main()
