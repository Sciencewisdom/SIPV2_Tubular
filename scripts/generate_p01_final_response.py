"""
Generate FINAL clDice response curve with ALL data points.
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# COMPLETE data from all experiments
results = {
    'E1': {
        0.0: {'dice': 0.7986, 'cldice': 0.8018, 'skelrec': 0.7879, 'breaks': 177.5},
        0.1: {'dice': 0.7976, 'cldice': 0.8208, 'skelrec': 0.7584, 'breaks': 435.8},
        0.3: {'dice': 0.7986, 'cldice': 0.8206, 'skelrec': 0.7600, 'breaks': 431.3},
        0.5: {'dice': 0.7991, 'cldice': 0.8242, 'skelrec': 0.7732, 'breaks': 391.8},
    },
    'E4': {
        0.0: {'dice': 0.8007, 'cldice': 0.8296, 'skelrec': 0.7632, 'breaks': 453.2},
        0.05: {'dice': 0.7912, 'cldice': 0.8280, 'skelrec': 0.7630, 'breaks': 0.0},  # breaks missing
        0.1: {'dice': 0.7998, 'cldice': 0.8368, 'skelrec': 0.7818, 'breaks': 425.0},
        0.3: {'dice': 0.7992, 'cldice': 0.8381, 'skelrec': 0.7860, 'breaks': 395.4},
        0.5: {'dice': 0.7998, 'cldice': 0.8428, 'skelrec': 0.8031, 'breaks': 317.5},
    }
}

lambdas_e1 = sorted(results['E1'].keys())
lambdas_e4 = sorted(results['E4'].keys())

dice_e1 = [results['E1'][lam]['dice'] for lam in lambdas_e1]
skelrec_e1 = [results['E1'][lam]['skelrec'] for lam in lambdas_e1]
breaks_e1 = [results['E1'][lam]['breaks'] for lam in lambdas_e1]
cldice_e1 = [results['E1'][lam]['cldice'] for lam in lambdas_e1]

dice_e4 = [results['E4'][lam]['dice'] for lam in lambdas_e4]
skelrec_e4 = [results['E4'][lam]['skelrec'] for lam in lambdas_e4]
breaks_e4 = [results['E4'][lam]['breaks'] for lam in lambdas_e4]
cldice_e4 = [results['E4'][lam]['cldice'] for lam in lambdas_e4]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('P0-1 Final: clDice Sensitivity Study — Architecture-Dependent Optimization Trajectories\n(DRIVE, seed=42)',
             fontsize=13, fontweight='bold', y=0.98)

color_dw = '#E74C3C'
color_sip = '#3498DB'

# Panel A: Skeleton Recall
ax = axes[0, 0]
ax.plot(lambdas_e1, skelrec_e1, color=color_dw, marker='s', linewidth=2.5, markersize=10,
        label='DW (CNN baseline)', zorder=3)
ax.plot(lambdas_e4, skelrec_e4, color=color_sip, marker='o', linewidth=2.5, markersize=10,
        label='SIP-v2 (ours)', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('Skeleton Recall', fontsize=11)
ax.set_title('(A) Skeleton Recall', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.74, 0.82)
# Add trend arrows
ax.annotate('', xy=(0.5, 0.773), xytext=(0.05, 0.788),
            arrowprops=dict(arrowstyle='->', color=color_dw, lw=2, ls='--'))
ax.annotate('', xy=(0.5, 0.803), xytext=(0.05, 0.765),
            arrowprops=dict(arrowstyle='->', color=color_sip, lw=2, ls='--'))

# Panel B: Break Count
ax = axes[0, 1]
ax.plot(lambdas_e1, breaks_e1, color=color_dw, marker='s', linewidth=2.5, markersize=10,
        label='DW (CNN baseline)', zorder=3)
# Filter out 0 breaks for E4 lambda=0.05 (missing data)
lambdas_e4_f = [l for l, b in zip(lambdas_e4, breaks_e4) if b > 0]
breaks_e4_f = [b for b in breaks_e4 if b > 0]
ax.plot(lambdas_e4_f, breaks_e4_f, color=color_sip, marker='o', linewidth=2.5, markersize=10,
        label='SIP-v2 (ours)', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('Branch Break Count', fontsize=11)
ax.set_title('(B) Topology Continuity', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Panel C: Dice
ax = axes[1, 0]
ax.plot(lambdas_e1, dice_e1, color=color_dw, marker='s', linewidth=2.5, markersize=10,
        label='DW (CNN baseline)', zorder=3)
ax.plot(lambdas_e4, dice_e4, color=color_sip, marker='o', linewidth=2.5, markersize=10,
        label='SIP-v2 (ours)', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('Best Dice', fontsize=11)
ax.set_title('(C) Regional Segmentation', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.788, 0.803)

# Panel D: clDice
ax = axes[1, 1]
ax.plot(lambdas_e1, cldice_e1, color=color_dw, marker='s', linewidth=2.5, markersize=10,
        label='DW (CNN baseline)', zorder=3)
ax.plot(lambdas_e4, cldice_e4, color=color_sip, marker='o', linewidth=2.5, markersize=10,
        label='SIP-v2 (ours)', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('clDice', fontsize=11)
ax.set_title('(D) Centerline Dice', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.79, 0.85)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = '/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/paper_figures/p01_final_response_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {output_path}")

# Also save data
import json
with open('/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/outputs/p01_final_data.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: p01_final_data.json")
