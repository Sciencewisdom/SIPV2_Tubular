#!/usr/bin/env python3
"""
Wilcoxon signed-rank test for real road experiments.
Compares R0 (DW), R1 (SIP-v2 Road), R2 (SIP-v2 + clDice).
"""
import os
import sys
import json
import argparse
from scipy.stats import wilcoxon
import numpy as np


def load_cases(path):
    with open(path) as f:
        data = json.load(f)
    return data['cases']


def extract_metric(cases, key):
    return [c[key] for c in cases]


def compare(a_cases, b_cases, name_a, name_b):
    print(f"\n=== {name_a} vs {name_b} ===")
    metrics = ['dice', 'iou', 'cldice', 'skel_recall', 'apls', 'connectivity', 'gap_recovery']
    results = {}
    for key in metrics:
        a_vals = extract_metric(a_cases, key)
        b_vals = extract_metric(b_cases, key)
        try:
            stat, p = wilcoxon(a_vals, b_vals)
            mean_diff = np.mean(b_vals) - np.mean(a_vals)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {key:15s}: diff={mean_diff:+.4f}, p={p:.4f} {sig}")
            results[key] = {'diff': float(mean_diff), 'p': float(p), 'sig': sig}
        except Exception as e:
            print(f"  {key:15s}: error - {e}")
            results[key] = {'error': str(e)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='outputs/road_real_deterministic')
    parser.add_argument('--split', type=str, default='valid', help='valid or test')
    args = parser.parse_args()

    r0_path = os.path.join(args.input_dir, f'R0_seed42_{args.split}.json')
    r1_path = os.path.join(args.input_dir, f'R1_seed42_{args.split}.json')
    r2_path = os.path.join(args.input_dir, f'R2_seed42_{args.split}.json')

    if not os.path.exists(r0_path):
        print(f"Missing {r0_path}")
        sys.exit(1)
    if not os.path.exists(r1_path):
        print(f"Missing {r1_path}")
        sys.exit(1)
    if not os.path.exists(r2_path):
        print(f"Missing {r2_path}")
        sys.exit(1)

    r0_cases = load_cases(r0_path)
    r1_cases = load_cases(r1_path)
    r2_cases = load_cases(r2_path)

    print(f"Split: {args.split}")
    print(f"R0 cases: {len(r0_cases)}")
    print(f"R1 cases: {len(r1_cases)}")
    print(f"R2 cases: {len(r2_cases)}")

    results = {}
    results['R0_vs_R1'] = compare(r0_cases, r1_cases, 'R0 (DW)', 'R1 (SIP-v2)')
    results['R0_vs_R2'] = compare(r0_cases, r2_cases, 'R0 (DW)', 'R2 (SIP-v2+clDice)')
    results['R1_vs_R2'] = compare(r1_cases, r2_cases, 'R1 (SIP-v2)', 'R2 (SIP-v2+clDice)')

    # Save results
    out = os.path.join(args.input_dir, f'wilcoxon_results_{args.split}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {out}")


if __name__ == '__main__':
    main()
