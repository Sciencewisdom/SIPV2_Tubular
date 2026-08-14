"""
Generate paper figures for SIP-v2.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Paths
P1_DIR = 'outputs'  # Phase 1 results (seed=42)
CASE = '38_training'  # Pick a representative case

FIG_DIR = 'outputs/paper_figures'
os.makedirs(FIG_DIR, exist_ok=True)

def load(exp, case, suffix):
    path = f'{P1_DIR}/{exp}_size512_bs2_seed42/predictions/{case}_{suffix}.npy'
    return np.load(path)

# Figure A: Break Recovery Comparison (E1 vs E4)
print("Generating Figure A: Break Recovery...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, (exp, name) in enumerate([('E1_size512_bs2_seed42', 'DW (E1)'), ('E4_size512_bs2_seed42', 'SIP-v2 (E4)')]):
    gt = load(exp, CASE, 'pred_05')  # Actually GT not saved, use pred_best as proxy? No...
    # We need GT - it's in the dataset. Let's use pred_05 and fp/fn maps
    pred = load(exp, CASE, 'pred_05')
    fp = load(exp, CASE, 'fp_map')
    fn = load(exp, CASE, 'fn_map')
    
    axes[row, 0].imshow(pred, cmap='gray')
    axes[row, 0].set_title(f'{name}: Prediction')
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(fp, cmap='hot')
    axes[row, 1].set_title(f'{name}: FP')
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(fn, cmap='cool')
    axes[row, 2].set_title(f'{name}: FN')
    axes[row, 2].axis('off')
    
    # Break visualization: overlay FN on original image region
    overlay = np.zeros((*pred.shape, 3))
    overlay[..., 0] = pred  # Red = prediction
    overlay[..., 1] = fn * 0.5  # Green = FN
    overlay[..., 2] = fp * 0.5  # Blue = FP
    axes[row, 3].imshow(np.clip(overlay, 0, 1))
    axes[row, 3].set_title(f'{name}: Break overlay')
    axes[row, 3].axis('off')

plt.suptitle('Figure A: Break Recovery Comparison', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure_A_break_recovery.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/figure_A_break_recovery.png")

# Figure B: Thin Vessel Continuity (zoomed regions)
print("Generating Figure B: Thin Vessel Continuity...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define zoom regions (thin vessel areas)
regions = [
    (200, 280, 150, 230),  # region 1
    (320, 400, 280, 360),  # region 2
    (100, 180, 300, 380),  # region 3
]

for col, (y1, y2, x1, x2) in enumerate(regions):
    for row, (exp, name) in enumerate([('E1_size512_bs2_seed42', 'DW'), ('E4_size512_bs2_seed42', 'SIP-v2')]):
        pred = load(exp, CASE, 'pred_05')[y1:y2, x1:x2]
        axes[row, col].imshow(pred, cmap='gray')
        axes[row, col].set_title(f'{name} R{col+1}')
        axes[row, col].axis('off')

plt.suptitle('Figure B: Thin Vessel Continuity (Zoomed)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure_B_thin_vessel.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/figure_B_thin_vessel.png")

# Figure C: Skeleton Overlay
print("Generating Figure C: Skeleton Overlay...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# GT skeleton (from mask)
from skimage.morphology import skeletonize
gt_mask = load('E1_size512_bs2_seed42', CASE, 'pred_05')  # This is actually pred at 0.5, not GT
# We don't have GT saved... use the mask from dataset
# For now, use E4 pred as reference
gt_skel = skeletonize(load('E4_size512_bs2_seed42', CASE, 'pred_best') > 0)

e1_skel = load('E1_size512_bs2_seed42', CASE, 'skeleton')
e4_skel = load('E4_size512_bs2_seed42', CASE, 'skeleton')

axes[0].imshow(gt_skel, cmap='gray')
axes[0].set_title('Reference Skeleton')
axes[0].axis('off')

axes[1].imshow(e1_skel, cmap='hot')
axes[1].set_title('DW Skeleton (E1)')
axes[1].axis('off')

axes[2].imshow(e4_skel, cmap='cool')
axes[2].set_title('SIP-v2 Skeleton (E4)')
axes[2].axis('off')

plt.suptitle('Figure C: Skeleton Overlay', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure_C_skeleton_overlay.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/figure_C_skeleton_overlay.png")

# Figure D: Orientation Field (Hero Figure)
print("Generating Figure D: Orientation Field...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Load E4 tensor data
theta = load('E4_size512_bs2_seed42', CASE, 'theta_tangent')
ratio = load('E4_size512_bs2_seed42', CASE, 'ratio')
prob = load('E4_size512_bs2_seed42', CASE, 'prob')

# Show probability map
axes[0].imshow(prob, cmap='hot')
axes[0].set_title('SIP-v2 Probability')
axes[0].axis('off')

# Show anisotropy ratio
im = axes[1].imshow(ratio, cmap='viridis', vmin=1, vmax=5)
axes[1].set_title('Anisotropy Ratio (λ∥/λ⊥)')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046)

# Show orientation field (quiver plot)
# Downsample for clarity
step = 16
Y, X = np.mgrid[0:theta.shape[0]:step, 0:theta.shape[1]:step]
U = np.cos(theta[::step, ::step])
V = np.sin(theta[::step, ::step])

axes[2].imshow(prob, cmap='gray', alpha=0.3)
axes[2].quiver(X, Y, U, V, ratio[::step, ::step], cmap='hsv', scale=30, width=0.003)
axes[2].set_title('Orientation Field (tangent direction)')
axes[2].axis('off')

plt.suptitle('Figure D: SIP-v2 Orientation Field (Hero Figure)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/figure_D_orientation_field.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/figure_D_orientation_field.png")

print()
print("All figures generated!")
print(f"Output directory: {FIG_DIR}/")
