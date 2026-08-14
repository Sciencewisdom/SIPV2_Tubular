import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = 'outputs/paper_figures'
os.makedirs(FIG_DIR, exist_ok=True)

def analyze(dataset, case_id):
    base = f'outputs/E4_size512_bs2_seed42/predictions'
    if dataset == 'chasedb1':
        base = 'outputs/chasedb1/E4_size512_bs2_seed42/predictions'
    prob = np.load(f'{base}/{case_id}_prob.npy')
    theta = np.load(f'{base}/{case_id}_theta_tangent.npy')
    ratio = np.load(f'{base}/{case_id}_ratio.npy')
    dy, dx = np.gradient(prob)
    grad_angle = np.arctan2(dy, dx)
    vessel_tangent = grad_angle + np.pi / 2
    vessel_mask = prob > 0.3
    diff = theta - vessel_tangent
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    diff_deg = np.abs(np.degrees(diff))
    vessel_diff = diff_deg[vessel_mask]
    return {
        'mean_error': np.mean(vessel_diff),
        'median_error': np.median(vessel_diff),
        'align_15': np.mean(vessel_diff < 15),
        'align_30': np.mean(vessel_diff < 30),
        'align_45': np.mean(vessel_diff < 45),
        'diff_deg': diff_deg, 'vessel_mask': vessel_mask,
        'theta': theta, 'prob': prob, 'ratio': ratio,
    }

print('='*70)
print('Tensor Orientation Alignment Analysis')
print('='*70)

drive_cases = ['37_training', '38_training', '39_training', '40_training']
results = []
for case in drive_cases:
    try:
        r = analyze('drive', case)
        results.append(r)
        print(f"DRIVE {case}: mean_err={r['mean_error']:.1f} deg, align_30={r['align_30']*100:.1f}%")
    except Exception as e:
        print(f"DRIVE {case}: {e}")

if results:
    mean_err = np.mean([r['mean_error'] for r in results])
    align_30 = np.mean([r['align_30'] for r in results])
    align_15 = np.mean([r['align_15'] for r in results])
    print(f"\nDRIVE AVERAGE: mean_err={mean_err:.1f} deg, align_15={align_15*100:.1f}%, align_30={align_30*100:.1f}%")
    
    r = results[1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(r['prob'], cmap='hot'); axes[0, 0].set_title('(a) Vessel Probability'); axes[0, 0].axis('off')
    
    error_map = np.where(r['vessel_mask'], r['diff_deg'], np.nan)
    im = axes[0, 1].imshow(error_map, cmap='RdYlGn_r', vmin=0, vmax=90)
    axes[0, 1].set_title('(b) Orientation Error (deg)'); axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[1, 0].imshow(r['ratio'], cmap='viridis', vmin=1, vmax=5)
    axes[1, 0].set_title('(c) Anisotropy Ratio'); axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    step = 12
    Y, X = np.mgrid[0:r['theta'].shape[0]:step, 0:r['theta'].shape[1]:step]
    U = np.cos(r['theta'][::step, ::step])
    V = np.sin(r['theta'][::step, ::step])
    mask_down = r['vessel_mask'][::step, ::step]
    axes[1, 1].imshow(r['prob'], cmap='gray', alpha=0.4)
    axes[1, 1].quiver(X[mask_down], Y[mask_down], U[mask_down], V[mask_down], 
                      color='red', scale=25, width=0.004, alpha=0.8)
    axes[1, 1].set_title('(d) Tangent Direction Field'); axes[1, 1].axis('off')
    
    plt.suptitle(f"SIP-v2 Tensor Orientation Alignment\nMean error: {mean_err:.1f} deg, "
                 f"30-deg alignment: {align_30*100:.1f}%", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_E_tensor_alignment_hero.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR}/figure_E_tensor_alignment_hero.png")
    
    all_errors = np.concatenate([r['diff_deg'][r['vessel_mask']] for r in results])
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(all_errors, bins=50, range=(0, 90), color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(15, color='green', linestyle='--', linewidth=2, label='15-deg')
    ax.axvline(30, color='orange', linestyle='--', linewidth=2, label='30-deg')
    ax.axvline(45, color='red', linestyle='--', linewidth=2, label='45-deg')
    ax.set_xlabel('Orientation Error (degrees)', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title(f"Tensor-Vessel Orientation Error (N={len(all_errors)} pixels, 4 cases)", fontsize=12)
    ax.legend(); ax.set_xlim(0, 90)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figure_F_alignment_histogram.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR}/figure_F_alignment_histogram.png")

print("\nDone!")
