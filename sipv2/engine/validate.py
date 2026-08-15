"""
Validation loop with comprehensive metrics and prediction saving.
Saves: probability maps, binary masks, skeletons, FP/FN maps, FOV masks.
For E4 SIP-v2: also saves tensor visualizations (lambda ratio, theta, diffusion magnitude, beta).
"""
import os
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from skimage.morphology import skeletonize

from ..metrics.region import MetricsTracker, pixel_metrics, probability_metrics
from ..metrics.skeleton import compute_all_skeleton_metrics


def _save_case_analysis(
    save_dir,
    image_id,
    pred_prob,
    gt,
    fov,
    best_thresh=0.5,
    tensor_aux=None,
):
    """
    Save comprehensive analysis for a single case.

    Args:
        save_dir: output directory
        image_id: case identifier
        pred_prob: [H, W] probability map
        gt: [H, W] ground truth
        fov: [H, W] FOV mask
        best_thresh: best threshold for binary mask
        tensor_aux: dict with E4 tensor info (optional)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Probability map
    np.save(os.path.join(save_dir, f'{image_id}_prob.npy'), pred_prob.astype(np.float32))

    # 2. Binary mask at 0.5
    pred_05 = (pred_prob > 0.5).astype(np.uint8)
    np.save(os.path.join(save_dir, f'{image_id}_pred_05.npy'), pred_05)

    # 3. Binary mask at best threshold
    pred_best = (pred_prob > best_thresh).astype(np.uint8)
    np.save(os.path.join(save_dir, f'{image_id}_pred_best.npy'), pred_best)

    # 4. Skeleton prediction (at best threshold)
    if pred_best.sum() > 0:
        skel_pred = skeletonize(pred_best > 0).astype(np.uint8)
    else:
        skel_pred = np.zeros_like(pred_best)
    np.save(os.path.join(save_dir, f'{image_id}_skeleton.npy'), skel_pred)

    # 5. FP map (at 0.5)
    fp_map = (pred_05 > 0) & (gt == 0)
    np.save(os.path.join(save_dir, f'{image_id}_fp_map.npy'), fp_map.astype(np.uint8))

    # 6. FN map (at 0.5)
    fn_map = (pred_05 == 0) & (gt > 0)
    np.save(os.path.join(save_dir, f'{image_id}_fn_map.npy'), fn_map.astype(np.uint8))

    # 7. FOV mask
    np.save(os.path.join(save_dir, f'{image_id}_fov.npy'), fov.astype(np.uint8))

    # 8. E4 tensor visualizations
    if tensor_aux is not None:
        for key in ['lambda_par', 'lambda_perp', 'ratio', 'theta_tangent',
                    'diff_norm', 'scale', 'beta']:
            if key in tensor_aux and tensor_aux[key] is not None:
                arr = tensor_aux[key]
                if torch.is_tensor(arr):
                    arr = arr.cpu().numpy()
                # Squeeze if needed
                if isinstance(arr, (int, float)):
                    continue
                if arr.ndim > 2:
                    arr = arr.squeeze()
                np.save(os.path.join(save_dir, f'{image_id}_{key}.npy'), arr.astype(np.float32))


def _register_sipv2_hooks(model):
    """Register hooks on SIPV2Blocks to capture tensor info."""
    hooks = []
    hook_data = {}

    def make_hook(name):
        def hook(module, input, output):
            # module is SIPV2BlockWrapper, output is (y, aux) from SIPV2Block
            if hasattr(module, 'block') and hasattr(module.block, 'beta_logit'):
                # We need to re-run the diffusion part to get aux
                # But forward already computed it - we need to capture it
                # The issue: SIPV2BlockWrapper discards aux
                # Instead, we'll compute aux on demand during validation
                pass
        return hook

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'SIPV2BlockWrapper':
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    return hooks, hook_data


def validate(
    model,
    val_loader,
    criterion,
    device,
    epoch,
    save_dir=None,
    block_type='conv',
    thresholds=None,
):
    """
    Validate model and compute all metrics.
    """
    model.eval()

    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    total_loss = 0.0
    num_batches = 0

    all_pred_probs = []
    all_masks = []
    all_fov_masks = []
    all_image_ids = []

    tracker = MetricsTracker()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            fov_masks = batch['fov_mask'].to(device, non_blocking=True)

            with autocast('cuda'):
                if block_type in ('sipv2', 'sipv2_full', 'sipv2_road'):
                    outputs = model(images, image=images)
                else:
                    outputs = model(images)
                loss = criterion(outputs, masks, fov_masks)

            total_loss += loss.item()
            num_batches += 1

            pred_prob = torch.sigmoid(outputs).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            fov_np = fov_masks.detach().cpu().numpy()

            tracker.update(outputs, masks, fov_masks, threshold=0.5)

            for i in range(pred_prob.shape[0]):
                all_pred_probs.append(pred_prob[i, 0])
                all_masks.append(masks_np[i, 0])
                all_fov_masks.append(fov_np[i, 0])
                all_image_ids.append(batch['image_id'][i])

    # Compute metrics
    metrics = tracker.compute()
    metrics['val_loss'] = total_loss / max(num_batches, 1)

    # Threshold scan
    best_dice_th = 0.5
    best_dice = 0.0
    dice_by_thresh = {}
    for th in thresholds:
        dices = []
        for pred, mask, fov in zip(all_pred_probs, all_masks, all_fov_masks):
            pred_fov = pred * fov
            mask_fov = mask * fov
            m = pixel_metrics(pred_fov, mask_fov, threshold=th)
            dices.append(m['dice'])
        mean_dice = np.mean(dices)
        dice_by_thresh[float(th)] = mean_dice
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_th = th

    metrics['best_dice'] = best_dice
    metrics['best_dice_threshold'] = best_dice_th
    metrics['dice_by_threshold'] = dice_by_thresh
    metrics['dice_per_case'] = dices  # per-case dice at best threshold

    # Skeleton metrics at best threshold
    skel_metrics_list = []
    for pred, mask in zip(all_pred_probs, all_masks):
        skel_m = compute_all_skeleton_metrics(pred, mask, threshold=best_dice_th)
        skel_metrics_list.append(skel_m)

    for key in ['cldice', 'skeleton_recall', 'skeleton_precision', 'thin_vessel_recall', 'break_count']:
        vals = [m[key] for m in skel_metrics_list]
        metrics[f'skel_{key}'] = np.mean(vals)
        metrics[f'skel_{key}_per_case'] = vals  # per-case for stats

    # Save predictions and analysis
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, (pred, mask, fov, img_id) in enumerate(
            zip(all_pred_probs, all_masks, all_fov_masks, all_image_ids)
        ):
            _save_case_analysis(
                save_dir,
                img_id,
                pred,
                mask,
                fov,
                best_thresh=best_dice_th,
                tensor_aux=None,  # TODO: extract tensor aux for E4
            )

    return metrics


def validate_with_tensor_capture(
    model,
    val_loader,
    criterion,
    device,
    epoch,
    save_dir=None,
):
    """
    Enhanced validation for SIP-v2 (E4) that captures tensor visualizations.
    """
    from ..models.blocks_sipv2 import SIPV2Block, SIPV2BlockWrapper

    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_pred_probs = []
    all_masks = []
    all_fov_masks = []
    all_image_ids = []
    all_tensor_aux = []

    tracker = MetricsTracker()

    # Enable aux capture on all SIPV2BlockWrappers
    sipv2_wrappers = [m for m in model.modules() if isinstance(m, SIPV2BlockWrapper)]
    for w in sipv2_wrappers:
        w.return_aux = True

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating E4'):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            fov_masks = batch['fov_mask'].to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(images, image=images)
                loss = criterion(outputs, masks, fov_masks)

            total_loss += loss.item()
            num_batches += 1

            pred_prob = torch.sigmoid(outputs).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            fov_np = fov_masks.detach().cpu().numpy()

            tracker.update(outputs, masks, fov_masks, threshold=0.5)

            # Capture tensor info from first SIPV2Block with valid aux
            for i in range(pred_prob.shape[0]):
                all_pred_probs.append(pred_prob[i, 0])
                all_masks.append(masks_np[i, 0])
                all_fov_masks.append(fov_np[i, 0])
                all_image_ids.append(batch['image_id'][i])

                # Get aux from first wrapper that has it
                taux = None
                for w in sipv2_wrappers:
                    if hasattr(w, '_last_aux') and w._last_aux is not None:
                        aux = w._last_aux
                        # Extract per-sample data
                        taux = {}
                        for key in ['lambda_par', 'lambda_perp', 'ratio', 'theta_tangent', 'diff_norm', 'scale']:
                            if key in aux:
                                val = aux[key]
                                if torch.is_tensor(val):
                                    val = val[i] if val.shape[0] == images.shape[0] else val
                                    val = val.cpu().numpy()
                                taux[key] = val
                        # Beta is scalar
                        if hasattr(w.block, 'beta_logit'):
                            taux['beta'] = torch.sigmoid(w.block.beta_logit).item()
                        break
                all_tensor_aux.append(taux)

    # Disable aux capture
    for w in sipv2_wrappers:
        w.return_aux = False

    # Compute metrics (same as regular validate)
    metrics = tracker.compute()
    metrics['val_loss'] = total_loss / max(num_batches, 1)

    thresholds = np.arange(0.05, 1.0, 0.05)
    best_dice_th = 0.5
    best_dice = 0.0
    for th in thresholds:
        dices = []
        for pred, mask, fov in zip(all_pred_probs, all_masks, all_fov_masks):
            pred_fov = pred * fov
            mask_fov = mask * fov
            m = pixel_metrics(pred_fov, mask_fov, threshold=th)
            dices.append(m['dice'])
        mean_dice = np.mean(dices)
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_th = th

    metrics['best_dice'] = best_dice
    metrics['best_dice_threshold'] = best_dice_th

    skel_metrics_list = []
    for pred, mask in zip(all_pred_probs, all_masks):
        skel_m = compute_all_skeleton_metrics(pred, mask, threshold=best_dice_th)
        skel_metrics_list.append(skel_m)

    for key in ['cldice', 'skeleton_recall', 'skeleton_precision', 'thin_vessel_recall', 'break_count']:
        vals = [m[key] for m in skel_metrics_list]
        metrics[f'skel_{key}'] = np.mean(vals)
        metrics[f'skel_{key}_per_case'] = vals  # per-case for stats

    # Save predictions
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, (pred, mask, fov, img_id, taux) in enumerate(
            zip(all_pred_probs, all_masks, all_fov_masks, all_image_ids, all_tensor_aux)
        ):
            _save_case_analysis(
                save_dir, img_id, pred, mask, fov,
                best_thresh=best_dice_th,
                tensor_aux=taux,
            )

    return metrics
