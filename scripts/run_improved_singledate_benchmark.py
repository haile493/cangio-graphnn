"""
Benchmark for Improved Proposed Model on Single-Date Radiometric Input:
Evaluates whether the improved single-date mechanisms:
  1. Non-local parcel self-attention & self-contrastive perturbation
  2. Pure 9-channel multispectral SPC & biophysical polarity (avoiding RGB compression bottleneck)
  3. Upgraded BiGSAMLossV2 (focal_alpha=0.80, false-positive background suppression lambda_fp=0.25, parcel_weight=0.30)
  4. Optimal learning rate (lr=1.5e-3 with CosineAnnealing)
can match or outperform the baseline graph models (especially Venagnosis PF-GNN 0.6105 IoU / 0.7582 Dice)
on Single-Date Radiometric input (15 full Landsat scenes `anhgoc`, zero-leakage split).
"""

import json
import os
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ag_models import build_model
from src.config import Config
from src.dataset import CanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_singledate_model(
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
        logits = out.seg_logits if hasattr(out, "seg_logits") else (out["seg_logits"] if isinstance(out, dict) else out)
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

    best_th = 0.5
    best_iou = -1.0
    for th in np.linspace(0.20, 0.80, 31):
        m = compute_binary_metrics(concat_probs, concat_targets, threshold=float(th))
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_th = float(th)

    final_metrics = compute_binary_metrics(concat_probs, concat_targets, threshold=best_th)
    final_metrics["f1"] = final_metrics["dice"]
    bious = [
        compute_boundary_iou(p, t, threshold=best_th)
        for p, t in zip(scene_probs, scene_targets)
    ]
    final_metrics["boundary_iou"] = float(np.mean(bious))
    final_metrics["threshold"] = round(best_th, 4)
    return final_metrics


def train_singledate_variant(
    name: str,
    use_sam_prior: bool,
    pretrained: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1.5e-3,
    save_dir: str = "results/models",
) -> Dict[str, float]:
    seed_everything(42)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" TRAINING ENHANCED SINGLE-DATE: {name.upper()}")
    print(f" (SAM Prior={use_sam_prior}, Pretrained={pretrained}, lr={lr})")
    print(f"{'='*70}")

    model = build_model(
        "big_sam_v2",
        in_channels=9,
        num_classes=1,
        base_channels=48,
        parcel_grid_size=31,
        num_gnn_layers=3,
        num_heads=4,
        k_semantic=4,
        k_rotation=4,
        use_sam_prior=use_sam_prior,
        pretrained=pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: Total={total_params:,} | Trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_metrics = None
    best_iou = -1.0
    best_ckpt_path = os.path.join(save_dir, f"{name}_best.pth")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            out = model(images, targets=masks)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += float(loss.item())

        scheduler.step()
        train_loss /= len(train_loader)

        val_metrics = evaluate_singledate_model(model, val_loader, device)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), best_ckpt_path)

        if ep % 5 == 0 or ep == epochs:
            print(
                f"Epoch [{ep:02d}/{epochs:02d}] "
                f"Loss: {train_loss:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | "
                f"Rec: {val_metrics['recall']:.4f} | "
                f"B-IoU: {val_metrics['boundary_iou']:.4f} | "
                f"Best IoU: {best_iou:.4f}"
            )

    train_time = time.time() - t_start
    best_metrics["train_time_sec"] = round(train_time, 2)
    best_metrics["total_params"] = total_params
    best_metrics["trainable_params"] = trainable_params

    print(
        f"\n[{name.upper()}] Final Best: IoU={best_metrics['iou']:.4f} | "
        f"F1={best_metrics['f1']:.4f} | Prec={best_metrics['precision']:.4f} | "
        f"Rec={best_metrics['recall']:.4f} | B-IoU={best_metrics['boundary_iou']:.4f} "
        f"(th={best_metrics['threshold']:.2f}, {train_time:.1f}s)"
    )
    return best_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Improved Single-Date Benchmark on device: {device}")

    cfg = Config(data_source="anhgoc")
    scenes = discover_scenes(cfg.data_dir, source="anhgoc")
    train_scenes = [s for s in scenes if s["year_folder"] not in ["Cut_2011", "Cut_2014"]]
    val_scenes = [s for s in scenes if s["year_folder"] in ["Cut_2011", "Cut_2014"]]

    train_ds = CanGioDataset(train_scenes, augment=True)
    val_ds = CanGioDataset(val_scenes, augment=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    results = {}

    # 1. Pure Multispectral Enhanced Model (No SAM ViT RGB compression bottleneck)
    results["big_sam_v2_singledate_pure_multispectral"] = train_singledate_variant(
        name="big_sam_v2_singledate_pure_multispectral",
        use_sam_prior=False,
        pretrained=False,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        lr=1.5e-3,
    )

    # 2. Pretrained SAM Prior Enhanced Model
    results["big_sam_v2_singledate_sam_enhanced"] = train_singledate_variant(
        name="big_sam_v2_singledate_sam_enhanced",
        use_sam_prior=True,
        pretrained=True,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        lr=8e-4,
    )

    out_file = os.path.join("results", "tables", "improved_singledate_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved improved single-date benchmark results to {out_file}")


if __name__ == "__main__":
    main()
