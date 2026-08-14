"""
Generate comprehensive paper figures for SIP-v2 manuscript.
High-quality, publication-ready figures with professional styling.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

FIG_DIR = 'outputs/paper_figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Data: Phase 1 (DRIVE, seed=42)
PHASE1_DATA = {
    'E0': {'name': 'U-Net',       'dice': 0.8062, 'pr_auc': 0.8756, 'cldice': 0.8359, 'skel_rec': 0.8046, 'breaks': 283.8, 'params': 4.1},
    'E1': {'name': 'DW U-Net',    'dice': 0.7986, 'pr_auc': 0.8701, 'cldice': 0.8018, 'skel_rec': 0.7879, 'breaks': 177.5, 'params': 1.4},
    'E2': {'name': 'IsoDiff',     'dice': 0.7877, 'pr_auc': 0.8602, 'cldice': 0.8207, 'skel_rec': 0.7551, 'breaks': 530.5, 'params': 1.0},
    'E3': {'name': 'OldSIP',      'dice': 0.7928, 'pr_auc': 0.8767, 'cldice': 0.8252, 'skel_rec': 0.7585, 'breaks': 507.0, 'params': 1.0},
    'E4': {'name': 'SIP-v2',      'dice': 0.8007, 'pr_auc': 0.8705, 'cldice': 0.8296, 'skel_rec': 0.7632, 'breaks': 453.2, 'params': 1.4},
}

# Phase 2: +clDice
PHASE2_DATA = {
    'E0_cl': {'name': 'U-Net+clD',  'dice': 0.8030, 'cldice': 0.8332, 'skel_rec': 0.8016, 'breaks': 263.6},
    'E1_cl': {'name': 'DW+clD',     'dice': 0.7986, 'cldice': 0.8206, 'skel_rec': 0.7600, 'breaks': 431.3},
    'E4_cl': {'name': 'SIP-v2+clD', 'dice': 0.7992, 'cldice': 0.8381, 'skel_rec': 0.7860, 'breaks': 395.4},
}

# Cross-dataset
CHASE_DATA = {
    'E1': {'dice': 0.7550, 'skel_rec': 0.718, 'cldice': 0.529},
    'E4': {'dice': 0.7596, 'skel_rec': 0.738, 'cldice': 0.517},
}
HRF_DATA = {
    'E1': {'dice': 0.7472, 'skel_rec': 0.676, 'cldice': 0.600},
    'E4': {'dice': 0.7301, 'skel_rec': 0.665, 'cldice': 0.558},
}

def generate_table_figure():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    headers = ['Method', 'Params\n(M)', 'Dice', 'PR-AUC', 'clDice', 'SkelRec', 'Breaks']
    rows = []
    colors = []
    for key in ['E0', 'E1', 'E2', 'E3', 'E4']:
        d = PHASE1_DATA[key]
        rows.append([d['name'], f"{d['params']:.1f}", f"{d['dice']:.4f}",
                     f"{d['pr_auc']:.4f}", f"{d['cldice']:.4f}",
                     f"{d['skel_rec']:.4f}", f"{d['breaks']:.1f}"])
        colors.append('#d4edda' if key == 'E4' else '#f8f9fa')

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#343a40'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_facecolor('#343a40')
    for i, color in enumerate(colors, 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
            if j >= 2:
                table[(i, j)].set_text_props(fontfamily='monospace')
    best_indices = {
        2: max(range(5), key=lambda i: PHASE1_DATA[['E0','E1','E2','E3','E4'][i]]['dice']),
        3: max(range(5), key=lambda i: PHASE1_DATA[['E0','E1','E2','E3','E4'][i]]['pr_auc']),
        4: max(range(5), key=lambda i: PHASE1_DATA[['E0','E1','E2','E3','E4'][i]]['cldice']),
        5: max(range(5), key=lambda i: PHASE1_DATA[['E0','E1','E2','E3','E4'][i]]['skel_rec']),
    }
    for col, best_row in best_indices.items():
        table[(best_row + 1, col)].set_text_props(fontweight='bold', color='#155724')
    plt.title('Table 1: Quantitative Comparison on DRIVE Test Set', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/table_results_phase1.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {FIG_DIR}/table_results_phase1.png")

def generate_bar_comparison():
    methods = ['U-Net', 'DW', 'IsoDiff', 'OldSIP', 'SIP-v2']
    keys = ['E0', 'E1', 'E2', 'E3', 'E4']
    colors_bar = ['#6c757d', '#adb5bd', '#dee2e6', '#ced4da', '#28a745']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = [('Dice', 'dice', axes[0]), ('clDice', 'cldice', axes[1]),
               ('Skeleton Recall', 'skel_rec', axes[2]), ('Break Count', 'breaks', axes[3])]
    for title, key, ax in metrics:
        values = [PHASE1_DATA[k][key] for k in keys]
        bars = ax.bar(methods, values, color=colors_bar, edgecolor='black', linewidth=0.5)
        bars[-1].set_edgecolor('#155724')
        bars[-1].set_linewidth(2)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.3f}' if val < 100 else f'{val:.0f}',
                   ha='center', va='bottom', fontsize=8)
    plt.suptitle('Figure: Phase 1 Method Comparison (DRIVE)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/bar_phase1_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {FIG_DIR}/bar_phase1_comparison.png")

def generate_cldice_effect():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    methods = ['U-Net', 'DW', 'SIP-v2']
    base_keys = ['E0', 'E1', 'E4']
    cl_keys = ['E0_cl', 'E1_cl', 'E4_cl']
    x = np.arange(len(methods))
    width = 0.35
    # Dice
    base_dice = [PHASE1_DATA[k]['dice'] for k in base_keys]
    cl_dice = [PHASE2_DATA[k]['dice'] for k in cl_keys]
    axes[0].bar(x - width/2, base_dice, width, label='BCE+Dice', color='#adb5bd', edgecolor='black')
    axes[0].bar(x + width/2, cl_dice, width, label='+ clDice', color='#28a745', edgecolor='black')
    axes[0].set_ylabel('Dice', fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0.78, 0.82); axes[0].legend()
    axes[0].set_title('Best Dice')
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
    # clDice
    base_cl = [PHASE1_DATA[k]['cldice'] for k in base_keys]
    cl_cl = [PHASE2_DATA[k]['cldice'] for k in cl_keys]
    axes[1].bar(x - width/2, base_cl, width, label='BCE+Dice', color='#adb5bd', edgecolor='black')
    axes[1].bar(x + width/2, cl_cl, width, label='+ clDice', color='#28a745', edgecolor='black')
    axes[1].set_ylabel('clDice', fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods)
    axes[1].set_ylim(0.78, 0.86); axes[1].legend()
    axes[1].set_title('clDice')
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    # Skeleton Recall
    base_sk = [PHASE1_DATA[k]['skel_rec'] for k in base_keys]
    cl_sk = [PHASE2_DATA[k]['skel_rec'] for k in cl_keys]
    axes[2].bar(x - width/2, base_sk, width, label='BCE+Dice', color='#adb5bd', edgecolor='black')
    axes[2].bar(x + width/2, cl_sk, width, label='+ clDice', color='#28a745', edgecolor='black')
    axes[2].set_ylabel('Skeleton Recall', fontweight='bold')
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods)
    axes[2].set_ylim(0.73, 0.82); axes[2].legend()
    axes[2].set_title('Skeleton Recall')
    axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)
    for ax, base, cl, name in [(axes[0], base_dice, cl_dice, 'Dice'),
                                (axes[1], base_cl, cl_cl, 'clDice'),
                                (axes[2], base_sk, cl_sk, 'SkelRec')]:
        delta = cl[2] - base[2]
        sign = '+' if delta >= 0 else ''
        color = '#155724' if delta >= 0 else '#721c24'
        ax.annotate(f'{sign}{delta:.4f}', xy=(2 + width/2, cl[2]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=9, color=color, fontweight='bold')
    plt.suptitle('Figure: Effect of clDice Loss (Phase 2)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/bar_cldice_effect.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {FIG_DIR}/bar_cldice_effect.png")

def generate_cross_dataset():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    datasets = ['DRIVE', 'CHASE_DB1', 'HRF']
    e1_dice = [PHASE1_DATA['E1']['dice'], CHASE_DATA['E1']['dice'], HRF_DATA['E1']['dice']]
    e4_dice = [PHASE1_DATA['E4']['dice'], CHASE_DATA['E4']['dice'], HRF_DATA['E4']['dice']]
    e1_skel = [PHASE1_DATA['E1']['skel_rec'], CHASE_DATA['E1']['skel_rec'], HRF_DATA['E1']['skel_rec']]
    e4_skel = [PHASE1_DATA['E4']['skel_rec'], CHASE_DATA['E4']['skel_rec'], HRF_DATA['E4']['skel_rec']]
    e1_cl = [PHASE1_DATA['E1']['cldice'], CHASE_DATA['E1']['cldice'], HRF_DATA['E1']['cldice']]
    e4_cl = [PHASE1_DATA['E4']['cldice'], CHASE_DATA['E4']['cldice'], HRF_DATA['E4']['cldice']]
    x = np.arange(len(datasets))
    width = 0.35
    for ax, e1_vals, e4_vals, title, ylabel in [
        (axes[0], e1_dice, e4_dice, 'Best Dice', 'Dice'),
        (axes[1], e1_skel, e4_skel, 'Skeleton Recall', 'SkelRec'),
        (axes[2], e1_cl, e4_cl, 'clDice', 'clDice')]:
        ax.bar(x - width/2, e1_vals, width, label='DW (E1)', color='#adb5bd', edgecolor='black')
        ax.bar(x + width/2, e4_vals, width, label='SIP-v2 (E4)', color='#28a745', edgecolor='black')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_title(title); ax.legend()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if title == 'Skeleton Recall':
            ax.annotate('*', xy=(1, max(e1_vals[1], e4_vals[1]) + 0.005),
                       ha='center', fontsize=16, color='red')
    plt.suptitle('Figure: Cross-dataset Generalization', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/bar_cross_dataset.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {FIG_DIR}/bar_cross_dataset.png")

def generate_architecture_schematic():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
    def draw_box(ax, x, y, w, h, text, color='#e9ecef', fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=fontsize, fontweight='bold' if 'SIP' in text else 'normal')
    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#495057', lw=1.5))
    ax.text(7, 9.5, 'SIP-v2 Block Architecture', ha='center', va='center', fontsize=14, fontweight='bold')
    draw_box(ax, 5.5, 8.2, 3, 0.6, 'Input Feature X', '#f8f9fa')
    draw_arrow(ax, 7, 8.2, 7, 7.8)
    draw_box(ax, 1, 6.5, 3.5, 1.0, 'Reaction Branch\n(1x1 -> DW3x3 -> 1x1)', '#d4edda')
    draw_arrow(ax, 5.5, 8.5, 2.75, 7.5)
    draw_box(ax, 5.2, 6.5, 3.6, 1.0, 'Diffusion Branch\nGradient-anchored T + 8-dir D', '#cce5ff')
    draw_arrow(ax, 7, 8.2, 7, 7.5)
    draw_box(ax, 9.5, 6.5, 3.5, 1.0, 'Prototype Branch\n(num_proto=8)', '#fff3cd')
    draw_arrow(ax, 8.5, 8.5, 11.25, 7.5)
    draw_box(ax, 5.5, 5.2, 3, 0.6, 'lambda_para, lambda_perp = f(X)', '#e2e3e5', fontsize=8)
    draw_arrow(ax, 7, 6.5, 7, 5.8)
    draw_box(ax, 5, 3.8, 4, 0.8, 'Y = X + Phi(X) + beta_d*D + beta_p*P', '#f8d7da')
    draw_arrow(ax, 2.75, 6.5, 5.5, 4.5)
    draw_arrow(ax, 7, 6.5, 7, 4.6)
    draw_arrow(ax, 11.25, 6.5, 8.5, 4.5)
    draw_box(ax, 5.5, 2.5, 3, 0.6, 'Output Feature Y', '#f8f9fa')
    draw_arrow(ax, 7, 3.8, 7, 3.1)
    ax.text(0.5, 1.5, 'Key Design:', fontsize=10, fontweight='bold')
    ax.text(0.5, 1.0, '  Direction anchored to image gradient (not freely predicted)', fontsize=9)
    ax.text(0.5, 0.6, '  Diffusion strength lambda learned per location', fontsize=9)
    ax.text(0.5, 0.2, '  Small residual weights: beta_d ~ 0.05, beta_p ~ 0.03', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/architecture_schematic.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {FIG_DIR}/architecture_schematic.png")

if __name__ == '__main__':
    print("Generating comprehensive paper figures...")
    generate_table_figure()
    generate_bar_comparison()
    generate_cldice_effect()
    generate_cross_dataset()
    generate_architecture_schematic()
    print(f"\nAll figures saved to: {FIG_DIR}/")
