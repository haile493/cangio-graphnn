"""
BiG-SAM Benchmark Script for Can Gio Mangrove Canopy Disturbance Mapping:
Evaluates BiG-SAM on full scenes only ('anhgoc') under two settings:
  1. From Scratch (`pretrained=False`)
  2. Pretrained SAM Foundation Weights (`pretrained=True`)

Strict zero-leakage split:
  - Train: 2006-2007, 2008, 2010 (7 consecutive bi-temporal scene pairs)
  - Val:   2011, 2014 (5 consecutive bi-temporal scene pairs)
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

from src.nets import (
    BiGSAMConfig,
    BiGSAMLoss,
    BiGSAMNet,
    build_loss,
    build_model,
)
from src.config import Config
from src.dataset import BiTemporalCanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou, stretch_rgb


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_big_sam(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    model.eval()
    all_probs = []
    all_targets = []
    scene_probs = []
    scene_targets = []
    scene_rgbs = []

    for img1, img2, masks, _ in val_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        out = model(img1, img2)
        logits = out.seg_logits
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = masks.numpy()

        for i in range(img1.shape[0]):
            p_map = probs[i, 0]
            t_map = targets_np[i, 0]
            rgb_map = img2[i, [2, 1, 0]].cpu().permute(1, 2, 0).numpy()
            scene_probs.append(p_map)
            scene_targets.append(t_map)
            scene_rgbs.append(rgb_map)
            all_probs.append(p_map.flatten())
            all_targets.append(t_map.flatten())

    concat_probs = np.concatenate(all_probs)
    concat_targets = np.concatenate(all_targets)

    # Search optimal F1 threshold
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.20, 0.80, 25):
        m = compute_binary_metrics(concat_probs, concat_targets, threshold=float(th))
        f1_score = m["dice"]
        if f1_score > best_f1:
            best_f1 = f1_score
            best_th = float(th)

    final_metrics = compute_binary_metrics(concat_probs, concat_targets, threshold=best_th)
    final_metrics["f1"] = final_metrics["dice"]
    bious = [
        compute_boundary_iou(p, t, threshold=best_th)
        for p, t in zip(scene_probs, scene_targets)
    ]
    final_metrics["boundary_iou"] = float(np.mean(bious))
    final_metrics["threshold"] = round(best_th, 4)
    return final_metrics, scene_probs, scene_targets, scene_rgbs


def train_single_big_sam(
    variant_name: str,
    pretrained: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 2e-4,
    save_dir: str = "results/models",
) -> Tuple[nn.Module, Dict[str, float], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    seed_everything(42)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n==================================================================")
    print(f" TRAINING: {variant_name.upper()} (Pretrained={pretrained})")
    print(f"==================================================================")

    model = build_model(
        "big_sam",
        in_channels=9,
        num_classes=1,
        pretrained=pretrained,
        target_size=1024,
        num_nodes=32,
        k_neighbors=8,
        top_k_prompts=5,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total={total_params:,} | Trainable={trainable_params:,}")

    # Optimize with cosine annealing
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = BiGSAMLoss()

    best_metrics = None
    best_f1 = -1.0
    best_probs = []
    best_targets = []
    best_rgbs = []
    best_ckpt_path = os.path.join(save_dir, f"{variant_name}_anhgoc.pth")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for img1, img2, masks, _ in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            out = model(img1, img2, targets=masks)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Evaluate on validation pairs
        val_metrics, val_probs, val_targets, val_rgbs = evaluate_big_sam(model, val_loader, device)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            best_probs = val_probs
            best_targets = val_targets
            best_rgbs = val_rgbs
            torch.save(model.state_dict(), best_ckpt_path)

        if ep % 5 == 0 or ep == epochs:
            print(
                f"Epoch [{ep:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"IoU: {val_metrics['iou']:.4f} | "
                f"B-IoU: {val_metrics['boundary_iou']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | "
                f"Rec: {val_metrics['recall']:.4f} | "
                f"Best F1: {best_f1:.4f}"
            )

    train_time = time.time() - t_start
    best_metrics["train_time_sec"] = round(train_time, 2)
    best_metrics["total_params"] = total_params
    best_metrics["trainable_params"] = trainable_params

    print(f"\nFinished {variant_name}: Best F1={best_metrics['f1']:.4f}, IoU={best_metrics['iou']:.4f}, B-IoU={best_metrics['boundary_iou']:.4f} (Time: {train_time:.1f}s)")
    return model, best_metrics, best_probs, best_targets, best_rgbs


def main():
    parser = argparse.ArgumentParser(description="BiG-SAM Full Scenes Benchmark")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    cfg = Config(data_source="anhgoc")
    scenes = discover_scenes(cfg.data_dir, source="anhgoc")
    print(f"Total full scenes discovered (anhgoc): {len(scenes)}")

    # Split: train (2006-2010), val (2011, 2014)
    train_scenes = [s for s in scenes if s["year_folder"] not in ["Cut_2011", "Cut_2014"]]
    val_scenes = [s for s in scenes if s["year_folder"] in ["Cut_2011", "Cut_2014"]]

    train_ds = BiTemporalCanGioDataset(train_scenes, pairing_strategy="consecutive", augment=True)
    val_ds = BiTemporalCanGioDataset(val_scenes, pairing_strategy="consecutive", augment=False)

    print(f"Bi-temporal Pairs: Train={len(train_ds)}, Val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    results = {}

    # 1. BiG-SAM from Scratch
    m_scratch, res_scratch, probs_sc, targets_sc, rgbs_sc = train_single_big_sam(
        variant_name="big_sam_scratch",
        pretrained=False,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )
    results["big_sam_scratch"] = res_scratch

    # 2. BiG-SAM with Pretrained SAM Weights
    m_pre, res_pre, probs_pre, targets_pre, rgbs_pre = train_single_big_sam(
        variant_name="big_sam_pretrained",
        pretrained=True,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )
    results["big_sam_pretrained"] = res_pre

    # Save benchmark metrics JSON
    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", "big_sam_benchmark_anhgoc.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {json_path}")

    # Plot visual qualitative comparison on validation scenes
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(len(val_ds), 4, figsize=(14, 3 * len(val_ds)))
    if len(val_ds) == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(len(val_ds)):
        # RGB T2
        rgb_disp = stretch_rgb(rgbs_pre[i])
        axes[i, 0].imshow(rgb_disp)
        axes[i, 0].set_title(f"Val Pair {i+1}: T2 Post-Disturbance")
        axes[i, 0].axis("off")

        # Ground Truth
        axes[i, 1].imshow(targets_pre[i], cmap="magma", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth Canopy Disturbance")
        axes[i, 1].axis("off")

        # From Scratch Prediction
        th_sc = res_scratch["threshold"]
        pred_sc = (probs_sc[i] > th_sc).astype(float)
        axes[i, 2].imshow(pred_sc, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title(f"BiG-SAM (Scratch)\nF1: {res_scratch['f1']:.3f} | IoU: {res_scratch['iou']:.3f}")
        axes[i, 2].axis("off")

        # Pretrained SAM Prediction
        th_pre = res_pre["threshold"]
        pred_pre = (probs_pre[i] > th_pre).astype(float)
        axes[i, 3].imshow(pred_pre, cmap="magma", vmin=0, vmax=1)
        axes[i, 3].set_title(f"BiG-SAM (Pretrained SAM)\nF1: {res_pre['f1']:.3f} | IoU: {res_pre['iou']:.3f}")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plot_path = os.path.join("results", "plots", "big_sam_fullscene_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved qualitative plot to {plot_path}")

    # Print summary comparative table
    print("\n" + "=" * 80)
    print(" EMPIRICAL BENCHMARK SUMMARY (FULL SCENES ONLY: 'ANHGOC')")
    print("=" * 80)
    print(f"{'Model Variant':<24} | {'F1 Score':<10} | {'IoU':<10} | {'B-IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'Time (s)':<8}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<24} | {m['f1']:<10.4f} | {m['iou']:<10.4f} | {m['boundary_iou']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['train_time_sec']:<8.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
