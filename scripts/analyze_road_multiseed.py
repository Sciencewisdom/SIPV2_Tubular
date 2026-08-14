#!/usr/bin/env python3
"""
Multi-seed statistical summary for real Massachusetts Roads experiments.
"""
import os
import json
import glob
import argparse
import numpy as np
from scipy.stats import wilcoxon


def load_all_results(base_dir):
    results = {}
    for path in glob.glob(os.path.join(base_dir, "*.json")):
        fname = os.path.basename(path)
        # e.g., R0_seed0_valid.json -> method=R0, seed=0, split=valid
        parts = fname.replace(".json", "").split("_")
        method = parts[0]
        seed = int(parts[1].replace("seed", ""))
        split = parts[2]
        key = f"{method}_{split}"
        with open(path) as f:
            data = json.load(f)
        if key not in results:
            results[key] = {}
        results[key][seed] = data
    return results


def summarize(results, split="test"):
    print("=" * 70)
    print(f"Real Massachusetts Roads: Multi-seed Summary ({split})")
    print("=" * 70)
    metrics = ["dice", "iou", "cldice", "skel_recall", "apls", "connectivity", "gap_recovery"]
    
    method_data = {}
    for method in ["R0", "R1", "R2"]:
        key = f"{method}_{split}"
        if key not in results:
            continue
        seeds = sorted(results[key].keys())
        agg = {}
        for metric in metrics:
            vals = [results[key][s]["aggregate"][metric] for s in seeds]
            agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        method_data[method] = {"seeds": seeds, "agg": agg}
    
    # Print table
    print(f"\n{'Metric':<15} {'R0 (DW)':<20} {'R1 (SIP-v2)':<20} {'R2 (+clDice)':<20}")
    print("-" * 80)
    for metric in metrics:
        row = f"{metric:<15}"
        for method in ["R0", "R1", "R2"]:
            if method in method_data:
                m = method_data[method]["agg"][metric]
                row += f" {m['mean']:.4f} ± {m['std']:.4f}  "
            else:
                row += " " * 20
        print(row)
    
    # Pairwise Wilcoxon on per-case means
    print("\n" + "=" * 70)
    print("Pairwise Wilcoxon signed-rank (per-case means)")
    print("=" * 70)
    
    cases = {}
    for method in ["R0", "R1", "R2"]:
        key = f"{method}_{split}"
        if key not in results:
            continue
        # Average per-case metrics across seeds
        all_cases = [results[key][s]["cases"] for s in sorted(results[key].keys())]
        n_cases = len(all_cases[0])
        avg_cases = []
        for i in range(n_cases):
            case = {"image_id": all_cases[0][i]["image_id"]}
            for metric in metrics:
                vals = [all_cases[s][i][metric] for s in range(len(all_cases))]
                case[metric] = float(np.mean(vals))
            avg_cases.append(case)
        cases[method] = avg_cases
    
    for a_name, b_name in [("R0", "R1"), ("R0", "R2"), ("R1", "R2")]:
        if a_name not in cases or b_name not in cases:
            continue
        print(f"\n{a_name} vs {b_name}:")
        for metric in metrics:
            a_vals = [c[metric] for c in cases[a_name]]
            b_vals = [c[metric] for c in cases[b_name]]
            try:
                stat, p = wilcoxon(a_vals, b_vals)
                diff = np.mean(b_vals) - np.mean(a_vals)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {metric:<15}: diff={diff:+.4f}, p={p:.4f} {sig}")
            except Exception as e:
                print(f"  {metric:<15}: error - {e}")
    
    return method_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/road_multiseed_deterministic")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    
    results = load_all_results(args.input_dir)
    if not results:
        print(f"No results found in {args.input_dir}")
        return
    summarize(results, split=args.split)


if __name__ == "__main__":
    main()
