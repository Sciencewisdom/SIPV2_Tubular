#!/usr/bin/env python3
"""
Update paper Q6 section with real Massachusetts Roads results.
Reads deterministic evaluation JSONs and updates results_draft.tex.
"""
import os
import json
import argparse


def load_json(path):
    with open(path) as f:
        return json.load(f)


def format_val(val, std=None, bold=False):
    if std is not None:
        s = f"{val:.4f} $\\pm$ {std:.4f}"
    else:
        s = f"{val:.4f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='outputs/road_real_deterministic')
    parser.add_argument('--paper_dir', type=str, default='paper')
    args = parser.parse_args()

    # Load results (prefer test split, fallback to valid)
    methods = {}
    for m, name in [('R0', 'R0 (DW)'), ('R1', 'R1 (SIP-v2 Road)'), ('R2', 'R2 (SIP-v2 + clDice)')]:
        for split in ['test', 'valid']:
            path = os.path.join(args.input_dir, f'{m}_seed42_{split}.json')
            if os.path.exists(path):
                data = load_json(path)
                agg = data['aggregate']
                methods[name] = {
                    'dice': agg['dice'],
                    'dice_std': agg['dice_std'],
                    'iou': agg['iou'],
                    'iou_std': agg['iou_std'],
                    'cldice': agg['cldice'],
                    'cldice_std': agg['cldice_std'],
                    'skel_recall': agg['skel_recall'],
                    'skel_recall_std': agg['skel_recall_std'],
                    'apls': agg['apls'],
                    'apls_std': agg['apls_std'],
                    'connectivity': agg['connectivity'],
                    'connectivity_std': agg['connectivity_std'],
                    'gap_recovery': agg['gap_recovery'],
                    'gap_recovery_std': agg['gap_recovery_std'],
                    'n_cases': agg['n_cases'],
                    'split': split,
                }
                break

    if len(methods) < 3:
        print(f"Missing results. Found: {list(methods.keys())}")
        return

    # Determine best per metric for bold formatting
    metrics = ['dice', 'iou', 'cldice', 'skel_recall', 'apls', 'connectivity', 'gap_recovery']
    best = {}
    for key in metrics:
        vals = {k: v[key] for k, v in methods.items()}
        best[key] = max(vals, key=vals.get)

    # Print LaTeX table rows
    print("% Q6 Real Massachusetts Roads Results Table")
    print("% Split: {}".format(methods['R0 (DW)']['split']))
    print()
    for name in ['R0 (DW)', 'R1 (SIP-v2 Road)', 'R2 (SIP-v2 + clDice)']:
        m = methods[name]
        row = f"{name} "
        for key in metrics:
            bold = (best[key] == name)
            row += "& " + format_val(m[key], m[key+'_std'], bold=bold) + " "
        row += "\\\\"
        print(row)

    # Print aggregate summary
    print("\n% Aggregate summary for text")
    r0, r1, r2 = methods['R0 (DW)'], methods['R1 (SIP-v2 Road)'], methods['R2 (SIP-v2 + clDice)']
    print(f"R1 vs R0 Dice: {r1['dice']:.4f} vs {r0['dice']:.4f} (diff={r1['dice']-r0['dice']:+.4f})")
    print(f"R1 vs R0 APLS: {r1['apls']:.4f} vs {r0['apls']:.4f} (diff={r1['apls']-r0['apls']:+.4f})")
    print(f"R2 vs R1 Dice: {r2['dice']:.4f} vs {r1['dice']:.4f} (diff={r2['dice']-r1['dice']:+.4f})")
    print(f"R2 vs R1 APLS: {r2['apls']:.4f} vs {r1['apls']:.4f} (diff={r2['apls']-r1['apls']:+.4f})")


if __name__ == '__main__':
    main()
