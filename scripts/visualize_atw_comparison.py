#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

methods = ['R0 (DW)', 'R1 (SIP-v2)', 'R2 (+clDice)', 'R3 (ATW\nλ=0.3)', 'R3* (ATW\nλ=0.15)']
best_dice = [0.5967, 0.6080, 0.6160, 0.6175, 0.6166]
jpr = [0.1925, 0.2148, 0.2629, 0.2332, 0.2986]
gaprec = [0.6637, 0.6419, 0.6110, 0.6096, 0.6240]

x = np.arange(len(methods))
width = 0.25

fig, ax1 = plt.subplots(figsize=(10, 5))

bars1 = ax1.bar(x - width, best_dice, width, label='Best Dice', color='steelblue')
ax1.set_ylabel('Best Dice', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(0.55, 0.65)

ax2 = ax1.twinx()
bars2 = ax2.bar(x, jpr, width, label='JPR', color='coral')
bars3 = ax2.bar(x + width, gaprec, width, label='GapRec', color='seagreen')
ax2.set_ylabel('JPR / GapRec', color='coral')
ax2.tick_params(axis='y', labelcolor='coral')
ax2.set_ylim(0, 0.75)

# Add value labels on bars
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=8, color='steelblue')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=8, color='coral')
for bar in bars3:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=8, color='seagreen')

ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_title('ATW Ablation on Massachusetts Roads (Test Set, 49 cases)')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
out_path = 'outputs/paper_figures/atw_ablation_comparison.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
