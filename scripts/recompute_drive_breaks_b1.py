"""B1 companion to recompute_drive_breaks.py: per-case DRIVE break counts for the
skeleton-recall sweep (E1/E4 x lambda {0.1, 0.3}) against the matched lambda=0
baselines, on the same 4 validation cases and with the same protocol.

Skips runs whose predictions are not yet on disk. Run after the B1 sweep
(scripts/run_b1_skelrec_sweep.sh) completes.
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
    "E1 λ=0":        "outputs/E1_size512_bs2_seed42",
    "E4 λ=0":        "outputs/E4_size512_bs2_seed42",
    "E1 srλ=0.1":    "outputs/b1_E1_skelrec0.1_seed42/E1_size512_bs2_seed42",
    "E1 srλ=0.3":    "outputs/b1_E1_skelrec0.3_seed42/E1_size512_bs2_seed42",
    "E4 srλ=0.1":    "outputs/b1_E4_skelrec0.1_seed42/E4_size512_bs2_seed42",
    "E4 srλ=0.3":    "outputs/b1_E4_skelrec0.3_seed42/E4_size512_bs2_seed42",
}


def n_components(binary):
    skel = skeletonize(binary > 0)
    _, n = ndimage.label(skel)
    return n


def case_breaks(rel, gts):
    sp = ROOT / rel / "summary.json"
    if not sp.exists():
        return None
    s = json.loads(sp.read_text())
    th = s["final_metrics"].get("best_dice_threshold", 0.5)
    vals = []
    for vid in VAL_IDS:
        p = ROOT / rel / "predictions" / f"{vid}_prob.npy"
        if not p.exists():
            return None
        pred = np.load(p)
        gt = gts[vid]
        if pred.shape != gt.shape:
            h, w = pred.shape
            y0 = (gt.shape[0] - h) // 2
            x0 = (gt.shape[1] - w) // 2
            gt = gt[y0:y0 + h, x0:x0 + w]
        vals.append(abs(n_components(pred > th) - n_components(gt)))
    return np.array(vals, dtype=float)


def main():
    gts = {}
    for vid in VAL_IDS:
        gif = sorted(GT_DIR.glob(f"{vid.split('_')[0]}*.gif"))
        gts[vid] = np.array(Image.open(gif[0]).convert("L")) > 127

    breaks = {}
    for name, rel in RUNS.items():
        b = case_breaks(rel, gts)
        if b is None:
            print(f"SKIP {name}: predictions missing in {rel}")
        else:
            breaks[name] = b
            print(f"{name}: per-case breaks={b.tolist()}  total={b.sum():.0f}")

    for lam in ("0.1", "0.3"):
        e1, e4 = breaks.get(f"E1 srλ={lam}"), breaks.get(f"E4 srλ={lam}")
        b1, b4 = breaks.get("E1 λ=0"), breaks.get("E4 λ=0")
        if e1 is None or e4 is None or b1 is None or b4 is None:
            continue
        d1, d4 = e1 - b1, e4 - b4
        print(f"\nλ={lam}: per-case Δbreaks vs λ=0:  E1 {d1.tolist()} (Σ{d1.sum():+.0f})  "
              f"E4 {d4.tolist()} (Σ{d4.sum():+.0f})")
        t = ttest_rel(d1, d4)
        try:
            w = wilcoxon(d1, d4)
            wp = w.pvalue
        except Exception as ex:
            wp = str(ex)
        print(f"  E1-vs-E4 delta contrast: paired t p={t.pvalue:.4f}, wilcoxon p={wp}")

    out = ROOT / "outputs" / "b1_breaks_percase.json"
    out.write_text(json.dumps({k: v.tolist() for k, v in breaks.items()}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
