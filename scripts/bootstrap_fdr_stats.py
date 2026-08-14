"""Round-1 audit statistics: bootstrap 95% CIs and BH-FDR correction.

Recomputes every headline paired comparison from cached per-case metrics and
adds two things the reviewers asked for:
  A5. Bootstrap 95% CI (10k resamples) for the mean paired difference.
  A6. Benjamini-Hochberg FDR-adjusted p-values across the reported test family.

Data sources (all cached, no torch needed):
  - DRIVE clDice sweep @lambda=0.1: outputs/p01_e{1,4}_lambda0.1_seed42/**/summary.json (4 val cases)
  - CHASE_DB1 E4 vs E1: outputs/chasedb1/E{1,4}_size512_bs2_seed42/summary.json (8 test cases)
  - Roads 3-seed: outputs/road_multiseed_deterministic/R{0,1,2}_seed{0,1,42}_test.json (49 test cases,
    per-case metrics averaged across seeds before pairing, per the paper's Statistical Protocol)
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)
N_BOOT = 10_000


def load_percase(summary_path, key):
    s = json.loads(Path(summary_path).read_text())
    pc = s.get("per_case") or s["final_metrics"]
    return np.asarray(pc[key], dtype=float)


def load_road_cases(path):
    d = json.loads(Path(path).read_text())
    return d["cases"] if isinstance(d, dict) else d


def road_filtered(metric, run):
    """3-seed-averaged per-case metric on the 40 foreground-bearing cases
    (pre-filtered analysis file; reproduces the paper's filtered p-values)."""
    d = json.loads((ROOT / "outputs/road_empty_crop_filtered_analysis_3seed.json").read_text())
    cases = {c["image_id"]: c[metric] for c in d[run]["cases"]}
    return cases


def road_seedavg(metric, run, seeds=(0, 1, 42), filtered=False):
    """Per-case metric averaged across seeds; optionally drop <0.5% foreground cases."""
    if filtered:
        cases = road_filtered(metric, run)
        return np.asarray([cases[k] for k in sorted(cases)])
    per_seed = {}
    for s in seeds:
        cases = load_road_cases(ROOT / f"outputs/road_multiseed_deterministic/{run}_seed{s}_test.json")
        per_seed[s] = {c["image_id"]: c for c in cases}
    ids = sorted(per_seed[seeds[0]])
    vals = [np.mean([per_seed[s][i][metric] for s in seeds]) for i in ids]
    return np.asarray(vals)


def boot_ci(a, b):
    """Bootstrap 95% CI for mean(b - a), paired resampling."""
    diff = b - a
    n = len(diff)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def paired(a, b, label, store):
    stat, p = wilcoxon(a, b)
    m, lo, hi = boot_ci(a, b)
    store.append({"label": label, "n": len(a), "mean_diff": m,
                  "ci95": [lo, hi], "p_raw": float(p)})
    return store


def bh_fdr(rows):
    ps = np.array([r["p_raw"] for r in rows])
    order = np.argsort(ps)
    ranked = ps[order]
    m = len(ps)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    for r, q in zip(rows, out):
        r["p_fdr"] = float(q)
    return rows


def main():
    rows = []

    # --- DRIVE sweep @lambda=0.1 (4 validation cases) ---
    e1b = load_percase(ROOT / "outputs/p01_e1_lambda0.1_seed42/E1_size512_bs2_seed42/summary.json", "skel_break_count_per_case")
    e4b = load_percase(ROOT / "outputs/p01_e4_lambda0.1_seed42/E4_size512_bs2_seed42/summary.json", "skel_break_count_per_case")
    paired(e1b, e4b, "DRIVE breaks E4+clDice vs E1+clDice @λ0.1", rows)

    e1s = load_percase(ROOT / "outputs/p01_e1_lambda0.1_seed42/E1_size512_bs2_seed42/summary.json", "skel_skeleton_recall_per_case")
    e4s = load_percase(ROOT / "outputs/p01_e4_lambda0.1_seed42/E4_size512_bs2_seed42/summary.json", "skel_skeleton_recall_per_case")
    paired(e1s, e4s, "DRIVE SkelRec E4+clDice vs E1+clDice @λ0.1", rows)

    # --- CHASE_DB1 E4 vs E1 (8 test cases) ---
    c1 = load_percase(ROOT / "outputs/chasedb1/E1_size512_bs2_seed42/summary.json", "skel_skeleton_recall_per_case")
    c4 = load_percase(ROOT / "outputs/chasedb1/E4_size512_bs2_seed42/summary.json", "skel_skeleton_recall_per_case")
    paired(c1, c4, "CHASE SkelRec E4 vs E1", rows)

    # --- Roads, 3-seed averaged (n=49) ---
    for metric, label in [("skel_recall", "Roads SkelRec R2 vs R1"),
                          ("apls", "Roads APLS R2 vs R1"),
                          ("gap_recovery", "Roads GapRec R2 vs R1"),
                          ("cldice", "Roads clDice R2 vs R1"),
                          ("apls", "Roads APLS R2 vs R0"),
                          ("dice", "Roads Dice R1 vs R0")]:
        run_a, run_b = ("R1", "R2") if "R2 vs R1" in label else (("R0", "R2") if "R2 vs R0" in label else ("R0", "R1"))
        a = road_seedavg(metric, run_a)
        b = road_seedavg(metric, run_b)
        paired(a, b, label + " (n=49)", rows)

    # --- Roads filtered (n=40, drop <0.5% foreground) ---
    a = road_seedavg("skel_recall", "R1", filtered=True)
    b = road_seedavg("skel_recall", "R2", filtered=True)
    paired(a, b, "Roads SkelRec R2 vs R1 filtered (n=40)", rows)
    a = road_seedavg("gap_recovery", "R1", filtered=True)
    b = road_seedavg("gap_recovery", "R2", filtered=True)
    paired(a, b, "Roads GapRec R2 vs R1 filtered (n=40)", rows)

    rows = bh_fdr(rows)

    print(f"{'comparison':52s} {'n':>3s} {'meanΔ':>9s} {'95% CI':>20s} {'p_raw':>8s} {'p_fdr':>8s}")
    for r in rows:
        print(f"{r['label']:52s} {r['n']:3d} {r['mean_diff']:+9.4f} "
              f"[{r['ci95'][0]:+8.4f},{r['ci95'][1]:+8.4f}] {r['p_raw']:8.4f} {r['p_fdr']:8.4f}")

    out = ROOT / "outputs/bootstrap_fdr_stats.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
