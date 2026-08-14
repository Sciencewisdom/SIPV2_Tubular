"""Rebuild Fig 8 (p13_failure_taxonomy_hrf.png) without the jet colormap.

The original figure was rendered with jet probability maps, which is a common
reviewer complaint (perceptually non-uniform, not colorblind-safe). This script
rebuilds the identical 3x5 grid from the cached prediction arrays in
outputs/hrf*/predictions/ using viridis (probabilities), RdBu_r (signed diff)
and a colorblind-safe red for the false-negative panel. No torch needed.

Layout (unchanged from the original figure):
  rows: HRF test cases 11_dr, 12_dr, 13_dr
  cols: DW (E1) | SIP-v2 Min (E4) | SIP-v2 Full (E5) | E5 - E1 (diff) | Failure: E5 misses
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures" / "p13_failure_taxonomy_hrf.png"

CASES = ["11_dr", "12_dr", "13_dr"]
MODELS = {
    "E1": ROOT / "outputs/hrf/E1_size512_bs2_seed42/predictions",
    "E4": ROOT / "outputs/hrf/E4_size512_bs2_seed42/predictions",
    "E5": ROOT / "outputs/hrf_E5/E5_size512_bs2_seed42/predictions",
}
COL_TITLES = ["DW (E1)", "SIP-v2 Min", "SIP-v2 Full", "E5 - E1 (diff)", "Failure: E5 misses"]

OUTSIDE = np.array([0.88, 0.88, 0.88])  # neutral gray outside the FOV


def load(d, case, kind):
    return np.load(d / f"{case}_{kind}.npy")


def main():
    fig, axes = plt.subplots(3, 5, figsize=(19.5, 11.8))
    fig.suptitle("P1-3: Failure Case Taxonomy — Pathological Images (HRF)",
                 fontsize=16, fontweight="bold", y=0.985)

    for r, case in enumerate(CASES):
        fov = load(MODELS["E5"], case, "fov").astype(bool)
        p1 = load(MODELS["E1"], case, "prob")
        p4 = load(MODELS["E4"], case, "prob")
        p5 = load(MODELS["E5"], case, "prob")
        fn5 = load(MODELS["E5"], case, "fn_map").astype(bool)

        # cols 0-2: probability maps, viridis, FOV-masked
        for c, (prob, name) in enumerate([(p1, "DW (E1)"), (p4, "SIP-v2 Min"), (p5, "SIP-v2 Full")]):
            rgb = plt.get_cmap("viridis")(prob)[..., :3]
            rgb[~fov] = OUTSIDE
            axes[r, c].imshow(rgb)
            axes[r, c].set_title(f"{case}: {name}", fontsize=10, fontweight="bold")

        # col 3: signed difference E5 - E1, diverging map centered at 0
        diff = (p5 - p1)
        rgb = plt.get_cmap("RdBu_r")((diff + 1.0) / 2.0)[..., :3]
        rgb[~fov] = OUTSIDE
        axes[r, 3].imshow(rgb)
        axes[r, 3].set_title("E5 - E1 (diff)", fontsize=10, fontweight="bold")

        # col 4: false negatives of E5 in colorblind-safe red on white
        rgb = np.ones((*fn5.shape, 3))
        rgb[fn5] = [0.84, 0.15, 0.16]  # vermillion (Okabe-Ito palette)
        rgb[~fov] = OUTSIDE
        axes[r, 4].imshow(rgb)
        axes[r, 4].set_title("Failure: E5 misses", fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
