#!/usr/bin/env python3
"""Plot validation vs test gradient conflict ratios for generalization evidence."""
import json
import numpy as np
import matplotlib.pyplot as plt

# Validation stats (from chat1progress)
valid_r0 = {
    'intersection': (0.171, 0.054),
    'endpoint': (0.072, 0.032),
    'straight': (0.336, 0.087),
    'wide': (0.212, 0.088),
    'narrow': (0.456, 0.088),
    'background_near': (0.002, 0.001),
}
valid_r2 = {
    'intersection': (0.158, 0.029),
    'endpoint': (0.065, 0.024),
    'straight': (0.296, 0.062),
    'wide': (0.122, 0.057),
    'narrow': (0.455, 0.067),
    'background_near': (0.002, 0.001),
}

# Test stats (computed from JSONs)
regions = ['intersection', 'endpoint', 'straight', 'wide', 'narrow', 'background_near']

def load_test_stats(path):
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for rname in regions:
        vals = [c[rname]['conflict_ratio'] for c in data if rname in c]
        stats[rname] = (np.mean(vals), np.std(vals))
    return stats

test_r0 = load_test_stats('/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/outputs/gradient_conflict/failure_regions_test_dw.json')
test_r2 = load_test_stats('/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/outputs/gradient_conflict/failure_regions_test_sipv2_road.json')

# Plotting
x = np.arange(len(regions))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 6))

def bar_group(ax, x, offset, stats, color, label, hatch=None):
    means = [stats[r][0]*100 for r in regions]
    stds = [stats[r][1]*100 for r in regions]
    bars = ax.bar(x + offset, means, width, yerr=stds, color=color, label=label, capsize=3, hatch=hatch, edgecolor='black', linewidth=0.5)
    return bars

bar_group(ax, x, -1.5*width, valid_r0, '#E74C3C', 'R0 Valid', hatch='//')
bar_group(ax, x, -0.5*width, valid_r2, '#3498DB', 'R2 Valid', hatch='//')
bar_group(ax, x, 0.5*width, test_r0, '#C0392B', 'R0 Test')
bar_group(ax, x, 1.5*width, test_r2, '#2980B9', 'R2 Test')

ax.set_ylabel('Sign-Conflict Ratio (%)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([r.replace('_', ' ').title() for r in regions], fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 55)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add text annotation
ax.text(0.02, 0.98, 'Validation (n=14) vs. Test (n=49)\nPatterns generalize: narrow≈45%, wide↓9%, straight↓4%',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
out_path = '/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/paper_figures/gradient_conflict_valid_vs_test.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# Print comparison table
print("\n=== Validation vs Test Gradient Conflict ===")
print(f"{'Region':<20} {'R0 Valid':<12} {'R0 Test':<12} {'R2 Valid':<12} {'R2 Test':<12}")
for r in regions:
    print(f"{r:<20} {valid_r0[r][0]*100:>5.1f}±{valid_r0[r][1]*100:<4.1f} {test_r0[r][0]*100:>5.1f}±{test_r0[r][1]*100:<4.1f} {valid_r2[r][0]*100:>5.1f}±{valid_r2[r][1]*100:<4.1f} {test_r2[r][0]*100:>5.1f}±{test_r2[r][1]*100:<4.1f}")
