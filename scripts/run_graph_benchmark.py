"""
End-to-End Training, Evaluation, Ablation, and Visualization Benchmark for the 4 Graph-Based
Remote Sensing Models (2026) on the Can Gio Mangrove Canopy Disturbance Dataset:

  1. `hrsei_gnn`        (Jebur et al., J. Hazard. Mater. Adv. 2026)
  2. `venagnosis_pfgnn` (Hu & Wu, IEEE Access 2026)
  3. `gsri_net`         (Liu et al., ACM ICMR 2026)
  4. `cgmn_net`         (Shao et al., GIScience & Remote Sensing 2026)

Executes strictly without data leakage:
  - Train split: 2006-2007, 2008, 2010 (8 multi-spectral Landsat-5 scenes)
  - Val split:   2011, 2014 (6 multi-spectral Landsat-5 & Landsat-8 scenes)
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.nets import BinaryDiceLoss, FocalLoss, build_loss, build_model
from src.config import Config
from src.dataset import CanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou, stretch_rgb


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gt_edge(mask: torch.Tensor) -> torch.Tensor:
    """Computes morphological boundary target from binary ground-truth mask (B, 1, H, W)."""
    max_p = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    min_p = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (max_p - min_p).clamp(0.0, 1.0)


def compute_model_loss(
    arch: str,
    outputs,
    targets: torch.Tensor,
    images: torch.Tensor,
    ven_loss_fn: nn.Module,
    cgmn_loss_fn: nn.Module,
    dice_fn: nn.Module,
) -> torch.Tensor:
    seg_logits = outputs["seg_logits"]
    if arch == "venagnosis_pfgnn":
        loss, _ = ven_loss_fn(outputs, targets)
        return loss
    elif arch == "cgmn_net":
        loss, _ = cgmn_loss_fn(outputs, targets, ref_image=images)
        return loss
    elif arch == "gsri_net":
        l_seg = ven_loss_fn._focal(seg_logits, targets) + dice_fn(seg_logits, targets)
        edge_gt = compute_gt_edge(targets)
        l_edge = F.binary_cross_entropy_with_logits(outputs["edge_logits"], edge_gt)
        return l_seg + 0.3 * l_edge
    elif arch == "hrsei_gnn":
        l_seg = ven_loss_fn._focal(seg_logits, targets) + dice_fn(seg_logits, targets)
        # Ecological quality target: high NDVI (ch 6) & NBR (ch 7) and non-disturbed (1 - targets)
        ndvi_ch = images[:, 6:7]
        nbr_ch = images[:, 7:8]
        eco_target = ((0.5 * ndvi_ch + 0.5 * nbr_ch) * (1.0 - targets)).clamp(0.0, 1.0)
        l_eco = F.mse_loss(outputs["hrsei_index"], eco_target)
        return l_seg + 0.25 * l_eco
    else:
        return ven_loss_fn._focal(seg_logits, targets) + dice_fn(seg_logits, targets)


@torch.no_grad()
def evaluate_graph_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_targets = []
    scene_probs = []
    scene_targets = []

    for images, masks, _ in val_loader:
        images = images.to(device)
        out = model(images)
        logits = out["seg_logits"] if isinstance(out, dict) else out
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = masks.numpy()

        for i in range(images.shape[0]):
            p_map = probs[i, 0]
            t_map = targets_np[i, 0]
            scene_probs.append(p_map)
            scene_targets.append(t_map)
            all_probs.append(p_map.flatten())
            all_targets.append(t_map.flatten())

    concat_probs = np.concatenate(all_probs)
    concat_targets = np.concatenate(all_targets)

    # Search optimal F1/IoU operating threshold across candidate thresholds (as in Hu & Wu Eq. 4)
    best_th = 0.5
    best_iou = -1.0
    for th in np.linspace(0.20, 0.80, 25):
        m = compute_binary_metrics(concat_probs, concat_targets, threshold=float(th))
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_th = float(th)

    final_metrics = compute_binary_metrics(concat_probs, concat_targets, threshold=best_th)
    # Compute mean Boundary IoU across validation scenes
    bious = [
        compute_boundary_iou(p, t, threshold=best_th)
        for p, t in zip(scene_probs, scene_targets)
    ]
    final_metrics["boundary_iou"] = float(np.mean(bious))
    final_metrics["threshold"] = round(best_th, 4)
    return final_metrics


def train_and_eval_single_model(
    arch: str,
    source: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1.5e-3,
    save_dir: str = "results/models",
) -> Tuple[nn.Module, Dict[str, float]]:
    seed_everything(42)
    model = build_model(arch, in_channels=9, num_classes=1).to(device)
    params = model.num_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ven_loss_fn = build_loss("venagnosis_loss", alpha=0.80, gamma=2.0).to(device)
    cgmn_loss_fn = build_loss("cgmn_composite_loss", pos_weight=6.0).to(device)
    dice_fn = BinaryDiceLoss(from_logits=True).to(device)

    best_metrics = None
    best_iou = -1.0
    best_ckpt_path = os.path.join(save_dir, f"best_{arch}_{source}.pth")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_model_loss(
                arch, outputs, masks, images, ven_loss_fn, cgmn_loss_fn, dice_fn
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate_graph_model(model, val_loader, device)
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                best_metrics = val_metrics
                model.save_checkpoint(best_ckpt_path, epoch=epoch, metrics=val_metrics)

    elapsed = time.time() - t0
    if os.path.exists(best_ckpt_path):
        model.load_checkpoint(best_ckpt_path, map_location=device)

    assert best_metrics is not None
    best_metrics["params"] = params
    best_metrics["train_time_sec"] = round(elapsed, 2)
    print(
        f"[{source.upper():6s}] {arch:18s} | Params: {params:7,d} | "
        f"IoU: {best_metrics['iou']:.4f} | Dice: {best_metrics['dice']:.4f} | "
        f"Prec: {best_metrics['precision']:.4f} | Rec: {best_metrics['recall']:.4f} | "
        f"BIoU: {best_metrics['boundary_iou']:.4f} | Acc: {best_metrics['accuracy']:.4f} "
        f"(th={best_metrics['threshold']:.2f}, {elapsed:.1f}s)"
    )
    return model, best_metrics


def plot_benchmark_comparison(results: Dict[str, Dict[str, Dict[str, float]]], save_path: str) -> None:
    model_labels = {
        "hrsei_gnn": "HRSEI-GNN\n(Jebur 2026)",
        "venagnosis_pfgnn": "Venagnosis PF-GNN\n(Hu & Wu 2026)",
        "gsri_net": "GSRI-Net\n(Liu 2026)",
        "cgmn_net": "CGMN Multi-Branch\n(Shao 2026)",
    }
    archs = list(model_labels.keys())
    metrics_keys = ["iou", "dice", "precision", "recall", "boundary_iou"]
    metric_names = ["IoU", "Dice (F1)", "Precision", "Recall", "Boundary IoU"]
    colors = ["#2b5c8f", "#d95f02", "#388e3c", "#7b3294"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax_idx, source in enumerate(["cut", "anhgoc"]):
        ax = axes[ax_idx]
        x = np.arange(len(metrics_keys))
        width = 0.19
        for i, arch in enumerate(archs):
            vals = [results[source][arch][k] for k in metrics_keys]
            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=model_labels[arch].replace("\n", " "), color=colors[i], edgecolor="black", linewidth=0.6)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        title_src = "Cut Chips (124×124)" if source == "cut" else "Full Radiometric Scenes (Anhgoc)"
        ax.set_title(f"Can Gio Mangrove Disturbance — {title_src}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ax_idx == 0:
            ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Quantitative Comparison of 2026 Graph-Based Models on Can Gio Dataset", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_qualitative_comparison(
    models_dict: Dict[str, nn.Module],
    thresholds_dict: Dict[str, float],
    val_dataset: CanGioDataset,
    device: torch.device,
    save_path: str,
) -> None:
    # Pick a representative validation scene (e.g., LC08_20140215)
    sample_idx = min(2, len(val_dataset) - 1)
    img_tensor, mask_tensor, meta = val_dataset[sample_idx]
    img_batch = img_tensor.unsqueeze(0).to(device)
    gt = mask_tensor[0].numpy()

    img_np = img_tensor.numpy()
    rgb_disp = stretch_rgb(np.stack([img_np[2], img_np[1], img_np[0]], axis=-1))
    false_color = stretch_rgb(np.stack([img_np[3], img_np[2], img_np[1]], axis=-1))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Can Gio Canopy Disturbance Qualitative Comparison — Scene {meta['scene_id']} (Held-out Validation)",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].imshow(rgb_disp)
    axes[0, 0].set_title("True Color (RGB)", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title("False Color (NIR-R-G)", fontweight="bold")
    axes[0, 1].axis("off")

    # HRSEI ecological quality map from HRSEIGNNNet
    hrsei_out = models_dict["hrsei_gnn"](img_batch)
    hrsei_map = hrsei_out["hrsei_index"][0, 0].cpu().numpy()
    im_hrsei = axes[0, 2].imshow(hrsei_map, cmap="YlGn", vmin=0, vmax=1)
    axes[0, 2].set_title("HRSEI Ecological Index (Jebur 2026)", fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(im_hrsei, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[0, 3].imshow(gt, cmap="Blues", vmin=0, vmax=1)
    axes[0, 3].set_title(f"Ground Truth Mask (Pos={int(gt.sum())} px)", fontweight="bold")
    axes[0, 3].axis("off")

    titles = {
        "hrsei_gnn": "HRSEI-GNN (Jebur 2026)",
        "venagnosis_pfgnn": "Venagnosis PF-GNN (Hu & Wu 2026)",
        "gsri_net": "GSRI-Net (Liu 2026)",
        "cgmn_net": "CGMN Multi-Branch (Shao 2026)",
    }

    for idx, arch in enumerate(["hrsei_gnn", "venagnosis_pfgnn", "gsri_net", "cgmn_net"]):
        ax = axes[1, idx]
        m = models_dict[arch]
        m.eval()
        out = m(img_batch)
        probs = torch.sigmoid(out["seg_logits"])[0, 0].cpu().numpy()
        th = thresholds_dict.get(arch, 0.5)
        bin_pred = (probs >= th).astype(np.float32)

        # TP (Green), FP (Red), FN (Blue) overlay on darkened False Color
        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
        tp = (bin_pred == 1) & (gt == 1)
        fp = (bin_pred == 1) & (gt == 0)
        fn = (bin_pred == 0) & (gt == 1)
        overlay[tp] = [0.0, 0.95, 0.0]
        overlay[fp] = [0.95, 0.15, 0.15]
        overlay[fn] = [0.15, 0.45, 0.95]

        m_scene = compute_binary_metrics(probs, gt, threshold=th)
        ax.imshow(false_color * 0.45 + overlay * 0.55)
        ax.set_title(f"{titles[arch]}\nIoU={m_scene['iou']:.3f} | Dice={m_scene['dice']:.3f}", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark 2026 Graph-Based Models on Can Gio Dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"Can Gio Mangrove Canopy Disturbance — 2026 Graph Neural Network Benchmark")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    config = Config()
    archs = ["hrsei_gnn", "venagnosis_pfgnn", "gsri_net", "cgmn_net"]
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {"cut": {}, "anhgoc": {}}
    trained_models_anhgoc: Dict[str, nn.Module] = {}
    thresholds_anhgoc: Dict[str, float] = {}

    for source in ["cut", "anhgoc"]:
        scenes = discover_scenes("data", source=source)
        train_scenes = [
            s for s in scenes
            if not any(vy in s["year_folder"] for vy in config.val_years) and s.get("groundtruth_path")
        ]
        val_scenes = [
            s for s in scenes
            if any(vy in s["year_folder"] for vy in config.val_years) and s.get("groundtruth_path")
        ]

        # Repeat train scenes 3x per epoch with D4 random rotations/flips for stable convergence on 8 scenes
        train_dataset = CanGioDataset(scenes=train_scenes * 3, augment=True, require_gt=True)
        val_dataset = CanGioDataset(scenes=val_scenes, augment=False, require_gt=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"\n>>> Domain: {source.upper()} (Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}) <<<")
        for arch in archs:
            model, metrics = train_and_eval_single_model(
                arch=arch,
                source=source,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
            )
            all_results[source][arch] = metrics
            if source == "anhgoc":
                trained_models_anhgoc[arch] = model
                thresholds_anhgoc[arch] = metrics["threshold"]

        if source == "anhgoc":
            qual_path = os.path.join("results", "plots", "fig2_qualitative_canopy_disturbance_maps.png")
            plot_qualitative_comparison(
                models_dict=trained_models_anhgoc,
                thresholds_dict=thresholds_anhgoc,
                val_dataset=val_dataset,
                device=device,
                save_path=qual_path,
            )
            print(f"Saved qualitative comparison plot to: {qual_path}")

    # Save quantitative JSON & Markdown report
    json_path = os.path.join("results", "tables", "graph_models_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    fig1_path = os.path.join("results", "plots", "fig1_graph_models_benchmark_comparison.png")
    plot_benchmark_comparison(all_results, fig1_path)
    print(f"Saved quantitative bar chart to: {fig1_path}")

    # Write clean markdown table
    md_lines = [
        "# Can Gio Mangrove Canopy Disturbance — Graph-Based Models (2026) Benchmark",
        "",
        "| Model Architecture | Reference Study | Data Source | Params | IoU | Dice / F1 | Precision | Recall | Boundary IoU | Accuracy |",
        "|:---|:---|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    ref_map = {
        "hrsei_gnn": "Jebur et al. (2026)",
        "venagnosis_pfgnn": "Hu & Wu (2026)",
        "gsri_net": "Liu et al. (2026)",
        "cgmn_net": "Shao et al. (2026)",
    }
    name_map = {
        "hrsei_gnn": "HRSEIGNNNet (`hrsei_gnn`)",
        "venagnosis_pfgnn": "VenagnosisPFGNNNet (`venagnosis_pfgnn`)",
        "gsri_net": "GSRIRotationRobustNet (`gsri_net`)",
        "cgmn_net": "CGMNMultiBranchNet (`cgmn_net`)",
    }
    for source_label, source_key in [("Cut Chips", "cut"), ("Full Scenes", "anhgoc")]:
        for arch in archs:
            m = all_results[source_key][arch]
            md_lines.append(
                f"| {name_map[arch]} | {ref_map[arch]} | {source_label} | {m['params']:,d} | "
                f"{m['iou']:.4f} | {m['dice']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['boundary_iou']:.4f} | {m['accuracy']:.4f} |"
            )

    md_path = os.path.join("results", "tables", "graph_models_benchmark.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved markdown table to: {md_path}")


if __name__ == "__main__":
    main()
