#!/usr/bin/env python3
"""
Generate multi-seed stability figure for road experiments.
Requires deterministic evaluation JSONs.
"""
import os
import json
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_agg(path):
    with open(path) as f:
        data = json.load(f)
    return data['aggregate']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/paper_figures/road_multiseed_stability.png')
    args = parser.parse_args()
    
    r0_files = sorted(glob.glob(os.path.join(args.input_dir, 'R0_seed*.json')))
    r1_files = sorted(glob.glob(os.path.join(args.input_dir, 'R1_seed*.json')))
    
    # Extract seeds and match
    r0_dict = {}
    for f in r0_files:
        m = re.search(r'R0_seed(\d+)', os.path.basename(f))
        if m:
            r0_dict[m.group(1)] = f
    r1_dict = {}
    for f in r1_files:
        m = re.search(r'R1_seed(\d+)', os.path.basename(f))
        if m:
            r1_dict[m.group(1)] = f
    
    common_seeds = sorted(set(r0_dict.keys()) & set(r1_dict.keys()))
    
    metrics = ['dice', 'apls', 'cldice', 'gap_recovery']
    metric_labels = ['Dice', 'APLS', 'clDice', 'GapRec']
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    for ax, key, label in zip(axes, metrics, metric_labels):
        r0_vals = [load_agg(r0_dict[s])[key] for s in common_seeds]
        r1_vals = [load_agg(r1_dict[s])[key] for s in common_seeds]
        
        x = np.arange(len(common_seeds))
        width = 0.35
        bars0 = ax.bar(x - width/2, r0_vals, width, label='R0 (DW)', color='coral', edgecolor='black', linewidth=0.5)
        bars1 = ax.bar(x + width/2, r1_vals, width, label='R1 (SIP-v2)', color='steelblue', edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'seed={s}' for s in common_seeds], fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        
        all_vals = r0_vals + r1_vals
        margin = (max(all_vals) - min(all_vals)) * 0.1
        ax.set_ylim(bottom=max(0, min(all_vals) - margin), top=max(all_vals) + margin)
        
        # Add mean lines with annotations
        r0_mean = np.mean(r0_vals)
        r1_mean = np.mean(r1_vals)
        ax.axhline(r0_mean, color='coral', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(r1_mean, color='steelblue', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add value labels on bars
        for bar in bars0:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=6)
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=6)
    
    n_seeds = len(common_seeds)
    plt.suptitle(f'Multi-seed Stability: Road Extraction ({n_seeds} seeds, Synthetic Massachusetts Roads)', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
