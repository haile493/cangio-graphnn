import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config


def compute_binary_metrics(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> Dict[str, float]:
    """
    Computes standard semantic segmentation metrics:
    IoU, Dice (F1), Precision, Recall, Accuracy.
    """
    bin_preds = (preds >= threshold).astype(bool)
    bin_targets = (targets >= 0.5).astype(bool)

    tp = np.logical_and(bin_preds, bin_targets).sum()
    fp = np.logical_and(bin_preds, ~bin_targets).sum()
    fn = np.logical_and(~bin_preds, bin_targets).sum()
    tn = np.logical_and(~bin_preds, ~bin_targets).sum()

    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


def compute_boundary_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation_ratio: float = 0.02,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Computes Boundary IoU (BIoU) based on Cheng et al. (CVPR 2021) and SOPSeg.
    Focuses evaluation on boundary pixels to measure boundary sharpness and delineation.
    """
    import cv2
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    gt_bin = (gt_mask >= 0.5).astype(np.uint8)

    if pred_bin.ndim == 1:
        # If flattened, attempt square reshape or return standard IoU if unreshapable
        side = int(np.round(np.sqrt(pred_bin.size)))
        if side * side == pred_bin.size:
            pred_bin = pred_bin.reshape(side, side)
            gt_bin = gt_bin.reshape(side, side)
        else:
            return float(compute_binary_metrics(pred_bin, gt_bin)["iou"])

    h, w = pred_bin.shape
    diag = np.sqrt(h**2 + w**2)
    d = max(1, int(round(dilation_ratio * diag)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))

    # Boundary band = Dilated - Eroded
    b_pred = cv2.dilate(pred_bin, kernel) - cv2.erode(pred_bin, kernel)
    b_gt = cv2.dilate(gt_bin, kernel) - cv2.erode(gt_bin, kernel)

    intersection = np.logical_and(b_pred > 0, b_gt > 0).sum()
    union = np.logical_or(b_pred > 0, b_gt > 0).sum()

    if union == 0:
        return 1.0 if np.all(pred_bin == gt_bin) else 0.0
    return float(intersection / (union + eps))


def stretch_rgb(band_data: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """Applies percentile stretch to an RGB array (H, W, 3) for clean display."""
    out = np.zeros_like(band_data, dtype=np.float32)
    for c in range(band_data.shape[2]):
        channel = band_data[:, :, c]
        val_min = np.percentile(channel, lower_pct)
        val_max = np.percentile(channel, upper_pct)
        if val_max > val_min:
            scaled = (channel - val_min) / (val_max - val_min)
            out[:, :, c] = np.clip(scaled, 0.0, 1.0)
        else:
            out[:, :, c] = channel
    return out


def plot_scene_evaluation(
    image_tensor: torch.Tensor,
    gt_mask: np.ndarray,
    pred_probs: np.ndarray,
    meta: Dict,
    save_path: str,
    threshold: float = 0.5
):
    """
    Generates a 6-panel comprehensive evaluation plot for a single scene:
    1. True Color RGB
    2. False Color (NIR-Red-Green)
    3. NDVI map
    4. Ground Truth Mask
    5. Predicted Disturbance Probability
    6. Error Map (TP, FP, FN)
    """
    img_np = image_tensor.detach().cpu().numpy()  # (C, H, W)
    
    # Channels 0, 1, 2 are Blue, Green, Red
    # Channel 3 is NIR
    blue = img_np[0]
    green = img_np[1]
    red = img_np[2]
    nir = img_np[3] if img_np.shape[0] > 3 else red

    # RGB composite (Red, Green, Blue)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_disp = stretch_rgb(rgb)

    # False Color composite (NIR, Red, Green)
    false_color = np.stack([nir, red, green], axis=-1)
    false_color_disp = stretch_rgb(false_color)

    # NDVI map
    ndvi = (nir - red) / (nir + red + 1e-7)

    bin_pred = (pred_probs >= threshold).astype(np.float32)
    gt = (gt_mask >= 0.5).astype(np.float32)

    # Confusion overlay: Green = TP, Red = FP, Blue = FN
    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    tp = (bin_pred == 1) & (gt == 1)
    fp = (bin_pred == 1) & (gt == 0)
    fn = (bin_pred == 0) & (gt == 1)
    
    overlay[tp] = [0.0, 0.9, 0.0]  # Green: Correct detection
    overlay[fp] = [0.9, 0.1, 0.1]  # Red: False alarm
    overlay[fn] = [0.1, 0.4, 0.9]  # Blue: Missed

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Can Gio Disturbance Segmentation - {meta.get('scene_id', 'Scene')}", fontsize=15, fontweight='bold')

    axes[0, 0].imshow(rgb_disp)
    axes[0, 0].set_title("True Color (RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(false_color_disp)
    axes[0, 1].set_title("False Color (NIR-R-G)")
    axes[0, 1].axis("off")

    im_ndvi = axes[0, 2].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0, 2].set_title("Vegetation Index (NDVI)")
    axes[0, 2].axis("off")
    fig.colorbar(im_ndvi, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_gt = axes[1, 0].imshow(gt, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Ground Truth (Positives: {int(gt.sum())})")
    axes[1, 0].axis("off")

    im_pred = axes[1, 1].imshow(pred_probs, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Predicted Probability (Threshold: {threshold})")
    axes[1, 1].axis("off")
    fig.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(rgb_disp * 0.4 + overlay * 0.6)
    axes[1, 2].set_title("Evaluation Map (Green: TP, Red: FP, Blue: FN)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    save_plots_dir: Optional[str] = None
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluates model across a dataloader and calculates dataset-wide metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    per_scene_results = []

    with torch.no_grad():
        for batch_idx, (images, masks, metas) in enumerate(dataloader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
            targets_np = masks.numpy()  # (B, 1, H, W)

            b_size = images.shape[0]
            for i in range(b_size):
                p_scene = probs[i, 0]
                t_scene = targets_np[i, 0]
                meta_item = {k: v[i] if isinstance(v, list) else v for k, v in metas.items()}
                
                metrics = compute_binary_metrics(p_scene, t_scene, threshold=threshold)
                metrics["scene_id"] = meta_item.get("scene_id", f"scene_{batch_idx}_{i}")
                metrics["date"] = meta_item.get("date", "Unknown")
                per_scene_results.append(metrics)

                all_preds.append(p_scene.flatten())
                all_targets.append(t_scene.flatten())

                if save_plots_dir:
                    plot_filename = f"eval_{meta_item.get('scene_id', f'scene_{batch_idx}_{i}')}.png"
                    plot_path = os.path.join(save_plots_dir, plot_filename)
                    plot_scene_evaluation(
                        image_tensor=images[i],
                        gt_mask=t_scene,
                        pred_probs=p_scene,
                        meta=meta_item,
                        save_path=plot_path,
                        threshold=threshold
                    )

    concatenated_preds = np.concatenate(all_preds)
    concatenated_targets = np.concatenate(all_targets)
    global_metrics = compute_binary_metrics(concatenated_preds, concatenated_targets, threshold=threshold)

    return global_metrics, per_scene_results
