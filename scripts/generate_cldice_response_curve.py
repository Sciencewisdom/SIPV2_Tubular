"""
Generate publication-quality clDice response curve figure.
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

with open('/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/outputs/p01_collected_results.json') as f:
    data = json.load(f)

lambdas_e1 = sorted(data['E1'].keys())
lambdas_e4 = sorted(data['E4'].keys())

dice_e1 = [data['E1'][lam]['best_dice'] for lam in lambdas_e1]
cldice_e1 = [data['E1'][lam]['cldice'] for lam in lambdas_e1]
skelrec_e1 = [data['E1'][lam]['skelrec'] for lam in lambdas_e1]
breaks_e1 = [data['E1'][lam]['breaks'] for lam in lambdas_e1]

dice_e4 = [data['E4'][lam]['best_dice'] for lam in lambdas_e4]
cldice_e4 = [data['E4'][lam]['cldice'] for lam in lambdas_e4]
skelrec_e4 = [data['E4'][lam]['skelrec'] for lam in lambdas_e4]
breaks_e4 = [data['E4'][lam]['breaks'] for lam in lambdas_e4]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('clDice Sensitivity Study: Topology Loss x Architecture Interaction',
             fontsize=14, fontweight='bold', y=0.98)

color_dw = '#E74C3C'
color_sip = '#3498DB'
marker_dw = 's'
marker_sip = 'o'

# Panel A: Skeleton Recall
ax = axes[0, 0]
ax.plot(lambdas_e1, skelrec_e1, color=color_dw, marker=marker_dw, linewidth=2,
        markersize=8, label='DW (CNN baseline)', zorder=3)
ax.plot(lambdas_e4, skelrec_e4, color=color_sip, marker=marker_sip, linewidth=2,
        markersize=8, label='SIP-v2', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('Skeleton Recall', fontsize=11)
ax.set_title('(A) Skeleton Recall vs clDice Weight', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.72, 0.82)
ax.annotate('Divergence:\nCNN degrades\nSIP-v2 improves',
            xy=(0.5, 0.5), xytext=(0.15, 0.76),
            fontsize=8, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

# Panel B: Break Count
ax = axes[0, 1]
ax.plot(lambdas_e1, breaks_e1, color=color_dw, marker=marker_dw, linewidth=2,
        markersize=8, label='DW (CNN baseline)', zorder=3)
ax.plot(lambdas_e4, breaks_e4, color=color_sip, marker=marker_sip, linewidth=2,
        markersize=8, label='SIP-v2', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('Branch Break Count', fontsize=11)
ax.set_title('(B) Topology Continuity vs clDice Weight', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate('Asymmetric:\nCNN breaks increase\nSIP-v2 breaks decrease',
            xy=(0.5, 350), xytext=(0.15, 250),
            fontsize=8, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

# Panel C: Dice
ax = axes[1, 0]
ax.plot(lambdas_e1, dice_e1, color=color_dw, marker=marker_dw, linewidth=2,
        markersize=8, label='DW (CNN baseline)', zorder=3)
ax.plot(lambdas_e4, dice_e4, color=color_sip, marker=marker_sip, linewidth=2,
        markersize=8, label='SIP-v2', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('Best Dice', fontsize=11)
ax.set_title('(C) Regional Segmentation vs clDice Weight', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.795, 0.805)
ax.annotate('Both stable:\nNo Dice sacrifice\nfor topology gain',
            xy=(0.5, 0.7995), xytext=(0.15, 0.802),
            fontsize=8, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

# Panel D: clDice
ax = axes[1, 1]
ax.plot(lambdas_e1, cldice_e1, color=color_dw, marker=marker_dw, linewidth=2,
        markersize=8, label='DW (CNN baseline)', zorder=3)
ax.plot(lambdas_e4, cldice_e4, color=color_sip, marker=marker_sip, linewidth=2,
        markersize=8, label='SIP-v2', zorder=3)
ax.set_xlabel('clDice weight lambda_c', fontsize=11)
ax.set_ylabel('clDice', fontsize=11)
ax.set_title('(D) Centerline Dice vs clDice Weight', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.79, 0.85)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = '/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/paper_figures/p01_cldice_response_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved: {output_path}")
