"""
Generate project status dashboard.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

fig = plt.figure(figsize=(14, 10))
fig.suptitle('SIP-v2 Project Status Dashboard (2026-05-27)', fontsize=16, fontweight='bold')

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Experiment Completion
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Experiments', fontsize=12, fontweight='bold')
experiments = [
    ('DRIVE E0-E5', '100%', 'green'),
    ('DRIVE E5+clDice', '100%', 'green'),
    ('CHASE_DB1 E1/E4/E5', '100%', 'green'),
    ('HRF E1/E4/E5', '100%', 'green'),
    ('Multi-seed E5', '100%', 'green'),
    ('P0-1 clDice sweep', '90%', 'orange'),
]
y = 9
for name, pct, color in experiments:
    ax1.text(0.5, y, name, fontsize=9, va='center')
    ax1.text(8, y, pct, fontsize=9, va='center', color=color, fontweight='bold')
    y -= 1.5

# Panel 2: Paper Assets
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Paper Assets', fontsize=12, fontweight='bold')
assets = [
    ('LaTeX Sections', '9/9', 'green'),
    ('PDF Version', 'v21', 'green'),
    ('Figures', '148+', 'green'),
    ('References', '47', 'green'),
    ('Tables', '8', 'green'),
]
y = 9
for name, val, color in assets:
    ax2.text(0.5, y, name, fontsize=9, va='center')
    ax2.text(8, y, val, fontsize=9, va='center', color=color, fontweight='bold')
    y -= 1.8

# Panel 3: Key Metrics
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Key Results (DRIVE)', fontsize=12, fontweight='bold')
metrics = [
    ('E5 Best Dice', '0.8032'),
    ('E5 clDice', '0.8403'),
    ('E5 SkelRec', '0.7930'),
    ('E5 Params', '1.82M'),
    ('E5 vs E1 Dice', '+0.0046'),
    ('E5 vs E1 clDice', '+0.0385'),
]
y = 9
for name, val in metrics:
    ax3.text(0.5, y, name, fontsize=9, va='center')
    ax3.text(8, y, val, fontsize=9, va='center', fontweight='bold', color='#2E86AB')
    y -= 1.5

# Panel 4: Scientific Story
ax4 = fig.add_subplot(gs[1, :])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Scientific Story: Closed Loop', fontsize=12, fontweight='bold')

story_steps = [
    ('1. Tensor Degeneracy', 'E3 fails:\nFree-learned tensors\ncollapse globally', '#FF6B6B'),
    ('2. Gradient Anchoring', 'E4 fixes it:\nImage-gradient directions\n93.8% aligned', '#4ECDC4'),
    ('3. Anisotropic > Isotropic', 'E4 > E2:\nDirectional propagation\nnecessary', '#45B7D1'),
    ('4. clDice Asymmetry', 'E4+clDice improves\nE1+clDice degrades\nArchitecture interaction', '#96CEB4'),
    ('5. PDE Stability', 'rho=0.3 acts as\nCFL-like constraint\nPrevents divergence', '#FFEAA7'),
    ('6. E5 Full Best', 'All datasets\nDice best\n1.8M params', '#DDA0DD'),
]

x_positions = [1.2, 3.0, 4.8, 6.6, 8.4]
for i, (title, text, color) in enumerate(story_steps[:5]):
    x = x_positions[i]
    rect = mpatches.FancyBboxPatch((x-0.7, 3), 1.6, 5.5, boxstyle="round,pad=0.1",
                                    facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax4.add_patch(rect)
    ax4.text(x, 7.5, title, fontsize=8, fontweight='bold', ha='center', va='top')
    ax4.text(x, 5.5, text, fontsize=7, ha='center', va='center')
    if i < 4:
        ax4.annotate('', xy=(x_positions[i+1]-0.7, 5.5), xytext=(x+0.8, 5.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Panel 5: Core Claim
ax5 = fig.add_subplot(gs[2, :])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
claim = (
    'CORE CLAIM: "Topology-aware losses require directional propagation '
    'inductive bias for stable optimization."'
)
ax5.text(5, 7, claim, fontsize=11, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2))

subclaim = (
    'This is not "SIP-v2 improves segmentation."\n'
    'This is: "Loss x Architecture interaction determines topology optimization stability."'
)
ax5.text(5, 3, subclaim, fontsize=9, ha='center', va='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = '/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/paper_figures/project_dashboard_final.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {output_path}")
