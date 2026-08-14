#!/usr/bin/env python3
"""
Compare results across all experiments.
"""
import os
import sys
import json
import glob
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_results(output_dir='outputs'):
    """Load all experiment summaries."""
    results = []

    for summary_path in glob.glob(os.path.join(output_dir, '*/summary.json')):
        with open(summary_path) as f:
            data = json.load(f)

        exp = data['exp']
        metrics = data.get('final_metrics', {})

        results.append({
            'exp': exp,
            'best_dice': data.get('best_dice', 0),
            'final_dice': metrics.get('dice', 0),
            'iou': metrics.get('iou', 0),
            'sensitivity': metrics.get('sensitivity', 0),
            'specificity': metrics.get('specificity', 0),
            'pr_auc': metrics.get('pr_auc', 0),
            'cldice': metrics.get('skel_cldice', 0),
            'skel_recall': metrics.get('skel_skeleton_recall', 0),
            'branch_breaks': metrics.get('skel_break_count', 0),
        })

    return pd.DataFrame(results)


def main():
    output_dir = os.path.join('..', 'outputs')
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return

    df = load_results(output_dir)

    if len(df) == 0:
        print("No results found. Train models first.")
        return

    # Sort by experiment
    df = df.sort_values('exp')

    print("\n" + "="*80)
    print("SIP-v2 Experiment Results Summary")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Key comparisons
    print("\nKey Comparisons:")
    e4 = df[df['exp'] == 'E4']
    e1 = df[df['exp'] == 'E1']
    e2 = df[df['exp'] == 'E2']

    if len(e4) > 0 and len(e1) > 0:
        e4_dice = e4['best_dice'].values[0]
        e1_dice = e1['best_dice'].values[0]
        print(f"  E4 (SIP-v2) Dice: {e4_dice:.4f}")
        print(f"  E1 (DW) Dice:     {e1_dice:.4f}")
        print(f"  Delta:            {e4_dice - e1_dice:+.4f}")

    if len(e4) > 0 and len(e2) > 0:
        e4_cldice = e4['cldice'].values[0]
        e2_cldice = e2['cldice'].values[0]
        print(f"  E4 (SIP-v2) clDice: {e4_cldice:.4f}")
        print(f"  E2 (Iso) clDice:    {e2_cldice:.4f}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    main()
