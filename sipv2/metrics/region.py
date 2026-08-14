"""
Region-based metrics: Dice, IoU, Accuracy, Sensitivity, Specificity, Precision.
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def dice_score(pred, target, threshold=0.5):
    """Hard Dice score."""
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred_binary * target).sum()
    return (2.0 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)


def soft_dice_score(pred_prob, target):
    """Soft Dice using probabilities."""
    intersection = (pred_prob * target).sum()
    return (2.0 * intersection) / (pred_prob.sum() + target.sum() + 1e-8)


def iou_score(pred, target, threshold=0.5):
    """IoU / Jaccard score."""
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)


def pixel_metrics(pred, target, threshold=0.5):
    """Compute TP, FP, FN, TN and derived metrics."""
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)

    tp = ((pred_binary == 1) & (target == 1)).sum()
    fp = ((pred_binary == 1) & (target == 0)).sum()
    fn = ((pred_binary == 0) & (target == 1)).sum()
    tn = ((pred_binary == 0) & (target == 0)).sum()

    sensitivity = tp / (tp + fn + 1e-8)  # recall
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        'dice': dice_score(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def probability_metrics(pred_prob, target):
    """
    Metrics that use probabilities.
    Returns PR-AUC, ROC-AUC, Brier score, FG/BG mean probability.
    """
    pred_flat = pred_prob.flatten()
    target_flat = target.flatten()

    # Subsample for speed if too large
    n = len(pred_flat)
    if n > 500000:
        idx = np.random.choice(n, size=500000, replace=False)
        pred_flat = pred_flat[idx]
        target_flat = target_flat[idx]

    # PR-AUC
    precision, recall, _ = precision_recall_curve(target_flat, pred_flat)
    pr_auc = auc(recall, precision)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(target_flat, pred_flat)
    except:
        roc_auc = 0.5

    # Brier score
    brier = np.mean((pred_flat - target_flat) ** 2)

    # FG/BG mean probability
    fg_mask = target_flat > 0
    bg_mask = ~fg_mask
    fg_mean = pred_flat[fg_mask].mean() if fg_mask.sum() > 0 else 0.0
    bg_mean = pred_flat[bg_mask].mean() if bg_mask.sum() > 0 else 0.0

    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'brier': float(brier),
        'fg_mean_prob': float(fg_mean),
        'bg_mean_prob': float(bg_mean),
    }


class MetricsTracker:
    """Track metrics across multiple batches/epochs."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.total_acc = 0.0
        self.total_sens = 0.0
        self.total_spec = 0.0
        self.total_prec = 0.0
        self.count = 0
        self.all_pred_probs = []
        self.all_targets = []

    def update(self, pred_logits, target, mask=None, threshold=0.5):
        """
        Args:
            pred_logits: [B, 1, H, W] model output logits
            target: [B, 1, H, W] binary
            mask: [B, 1, H, W] FOV mask
        """
        pred_prob = torch.sigmoid(pred_logits).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
            pred_prob = pred_prob * mask_np
            target_np = target_np * mask_np

        B = pred_prob.shape[0]
        for i in range(B):
            metrics = pixel_metrics(pred_prob[i, 0], target_np[i, 0], threshold)
            self.total_dice += metrics['dice']
            self.total_iou += metrics['iou']
            self.total_acc += metrics['accuracy']
            self.total_sens += metrics['sensitivity']
            self.total_spec += metrics['specificity']
            self.total_prec += metrics['precision']
            self.count += 1

            # Collect for probability metrics
            self.all_pred_probs.append(pred_prob[i, 0].flatten())
            self.all_targets.append(target_np[i, 0].flatten())

    def compute(self):
        if self.count == 0:
            return {}

        result = {
            'dice': self.total_dice / self.count,
            'iou': self.total_iou / self.count,
            'accuracy': self.total_acc / self.count,
            'sensitivity': self.total_sens / self.count,
            'specificity': self.total_spec / self.count,
            'precision': self.total_prec / self.count,
        }

        # Probability metrics across all pixels
        all_pred = np.concatenate(self.all_pred_probs)
        all_tgt = np.concatenate(self.all_targets)
        prob_m = probability_metrics(all_pred, all_tgt)
        result.update(prob_m)

        return result
