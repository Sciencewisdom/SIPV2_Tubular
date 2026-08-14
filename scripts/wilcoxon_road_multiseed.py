#!/usr/bin/env python3
"""
Wilcoxon signed-rank test across multi-seed road experiments.
Requires per-case deterministic evaluation JSONs.
"""
import os
import sys
import argparse
import json
import glob
import numpy as np
from scipy.stats import wilcoxon


def load_cases(path):
    with open(path) as f:
        data = json.load(f)
    cases = data['cases']
    # Sort by image_id to align
    cases = sorted(cases, key=lambda x: x['image_id'])
    return cases


def compare_groups(path_a, path_b, name_a, name_b):
    cases_a = load_cases(path_a)
    cases_b = load_cases(path_b)
    assert len(cases_a) == len(cases_b), f"Mismatch: {len(cases_a)} vs {len(cases_b)}"
    
    metrics = ['dice', 'iou', 'cldice', 'skel_recall', 'apls', 'connectivity', 'gap_recovery']
    results = {}
    
    for key in metrics:
        vals_a = np.array([c[key] for c in cases_a])
        vals_b = np.array([c[key] for c in cases_b])
        diff = vals_b - vals_a
        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        delta = float(mean_b - mean_a)
        
        try:
            stat, p = wilcoxon(diff, alternative='two-sided')
        except ValueError:
            stat, p = 0.0, 1.0
        
        results[key] = {
            'mean_a': mean_a,
            'mean_b': mean_b,
            'delta': delta,
            'wilcoxon_stat': float(stat),
            'p_value': float(p),
            'n': len(vals_a),
        }
    
    return results


def print_table(results, title):
    print(f"\n{title}")
    print(f"{'Metric':<12} {'Mean A':>10} {'Mean B':>10} {'Delta':>10} {'p-value':>10}")
    print("-" * 60)
    for key, r in results.items():
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        print(f"{key:<12} {r['mean_a']:10.4f} {r['mean_b']:10.4f} {r['delta']:+10.4f} {r['p_value']:10.4f} {sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='wilcoxon_results.json')
    args = parser.parse_args()
    
    # Find files
    r0_files = sorted(glob.glob(os.path.join(args.input_dir, 'R0_seed*.json')))
    r1_files = sorted(glob.glob(os.path.join(args.input_dir, 'R1_seed*.json')))
    
    if len(r0_files) < 2 or len(r1_files) < 2:
        print(f"Need at least 2 seeds per group. Found R0={len(r0_files)}, R1={len(r1_files)}")
        sys.exit(1)
    
    # R1 vs R0 (pooled across seeds? No, compare same seed)
    print(f"Found R0: {[os.path.basename(f) for f in r0_files]}")
    print(f"Found R1: {[os.path.basename(f) for f in r1_files]}")
    
    all_results = {}
    
    for r0_path, r1_path in zip(r0_files, r1_files):
        seed = os.path.basename(r0_path).replace('R0_seed', '').replace('.json', '')
        res = compare_groups(r0_path, r1_path, f"R0_seed{seed}", f"R1_seed{seed}")
        all_results[f"R1_vs_R0_seed{seed}"] = res
        print_table(res, f"R1 vs R0 (seed={seed})")
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == '__main__':
    main()
