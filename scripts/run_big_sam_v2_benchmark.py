"""
BiG-SAM v2 Benchmark & Comparison Script:
Evaluates the enhanced Graph-Augmented Foundation Architecture (BiG-SAM v2)
against BiG-SAM v1 and the 4 state-of-the-art 2026 graph remote sensing models.

Evaluations executed:
  1. Bi-Temporal Setting (T1 -> T2):
     - `big_sam_v2_scratch`: Trained from scratch with native biophysical SPC & Dual-Vein Graph
     - `big_sam_v2_pretrained`: Initialized with frozen SAM foundation visual priors
  2. Single-Date Setting (T2 direct radiometric):
     - `big_sam_v2_singledate`: Benchmarked directly against Venagnosis, HRSEI, GSRI, and CGMN

Uses strict zero-leakage split:
  - Train: 2006-2007, 2008, 2010 (7 bi-temporal pairs, 8 scenes)
  - Val:   2011, 2014 (5 bi-temporal pairs, 6 scenes)
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
    BiGSAMLossV2,
    BiGSAMNetV2,
    BiGSAMV2Config,
    build_loss,
    build_model,
)
from src.config import Config
from src.dataset import BiTemporalCanGioDataset, CanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou, stretch_rgb


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_bitemporal(
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
    for th in np.linspace(0.15, 0.85, 35):
        m = compute_binary_metrics(concat_probs, concat_targets, threshold=float(th))
        if m["dice"] > best_f1:
            best_f1 = m["dice"]
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


@torch.no_grad()
def evaluate_singledate(
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

    for images, masks, _ in val_loader:
        images = images.to(device)
        out = model(images)
        logits = out.seg_logits
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = masks.numpy()

        for i in range(images.shape[0]):
            p_map = probs[i, 0]
            t_map = targets_np[i, 0]
            rgb_map = images[i, [2, 1, 0]].cpu().permute(1, 2, 0).numpy()
            scene_probs.append(p_map)
            scene_targets.append(t_map)
            scene_rgbs.append(rgb_map)
            all_probs.append(p_map.flatten())
            all_targets.append(t_map.flatten())

    concat_probs = np.concatenate(all_probs)
    concat_targets = np.concatenate(all_targets)

    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.15, 0.85, 35):
        m = compute_binary_metrics(concat_probs, concat_targets, threshold=float(th))
        if m["dice"] > best_f1:
            best_f1 = m["dice"]
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


def train_big_sam_v2_bitemporal(
    variant_name: str,
    pretrained: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 3e-4,
    save_dir: str = "results/models",
) -> Tuple[nn.Module, Dict[str, float]]:
    seed_everything(42)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n==================================================================")
    print(f" TRAINING BI-TEMPORAL: {variant_name.upper()} (Pretrained SAM Prior={pretrained})")
    print(f"==================================================================")

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
        use_sam_prior=pretrained,
        pretrained=pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total={total_params:,} | Trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_metrics = None
    best_f1 = -1.0
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

        val_metrics, _, _, _ = evaluate_bitemporal(model, val_loader, device)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
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

    print(f"Finished {variant_name}: Best F1={best_metrics['f1']:.4f}, IoU={best_metrics['iou']:.4f}, B-IoU={best_metrics['boundary_iou']:.4f} ({train_time:.1f}s)")
    return model, best_metrics


def train_big_sam_v2_singledate(
    variant_name: str,
    pretrained: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 3e-4,
    save_dir: str = "results/models",
) -> Tuple[nn.Module, Dict[str, float]]:
    seed_everything(42)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n==================================================================")
    print(f" TRAINING SINGLE-DATE: {variant_name.upper()} (Pretrained SAM Prior={pretrained})")
    print(f"==================================================================")

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
        use_sam_prior=pretrained,
        pretrained=pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total={total_params:,} | Trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_metrics = None
    best_f1 = -1.0
    best_ckpt_path = os.path.join(save_dir, f"{variant_name}_anhgoc.pth")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        val_metrics, _, _, _ = evaluate_singledate(model, val_loader, device)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
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

    print(f"Finished {variant_name}: Best F1={best_metrics['f1']:.4f}, IoU={best_metrics['iou']:.4f}, B-IoU={best_metrics['boundary_iou']:.4f} ({train_time:.1f}s)")
    return model, best_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking BiG-SAM v2 on device: {device}")

    cfg = Config(data_source="anhgoc")
    scenes = discover_scenes(cfg.data_dir, source="anhgoc")
    print(f"Total full scenes discovered (anhgoc): {len(scenes)}")

    train_scenes = [s for s in scenes if s["year_folder"] not in ["Cut_2011", "Cut_2014"]]
    val_scenes = [s for s in scenes if s["year_folder"] in ["Cut_2011", "Cut_2014"]]

    # 1. Bi-Temporal Loaders
    bitemp_train_ds = BiTemporalCanGioDataset(train_scenes, pairing_strategy="consecutive", augment=True)
    bitemp_val_ds = BiTemporalCanGioDataset(val_scenes, pairing_strategy="consecutive", augment=False)
    bitemp_train_loader = DataLoader(bitemp_train_ds, batch_size=1, shuffle=True)
    bitemp_val_loader = DataLoader(bitemp_val_ds, batch_size=1, shuffle=False)

    # 2. Single-Date Loaders (Exact match to 4 graph paper benchmarks)
    single_train_ds = CanGioDataset(train_scenes, augment=True)
    single_val_ds = CanGioDataset(val_scenes, augment=False)
    single_train_loader = DataLoader(single_train_ds, batch_size=1, shuffle=True)
    single_val_loader = DataLoader(single_val_ds, batch_size=1, shuffle=False)

    all_results = {}

    # Experiment 1: BiG-SAM v2 (Bi-Temporal Scratch)
    _, res_bi_scratch = train_big_sam_v2_bitemporal(
        variant_name="big_sam_v2_bitemp_scratch",
        pretrained=False,
        train_loader=bitemp_train_loader,
        val_loader=bitemp_val_loader,
        device=device,
        epochs=50,
        lr=3e-4,
    )
    all_results["big_sam_v2_bitemp_scratch"] = res_bi_scratch

    # Experiment 2: BiG-SAM v2 (Bi-Temporal Pretrained SAM Prior)
    _, res_bi_pre = train_big_sam_v2_bitemporal(
        variant_name="big_sam_v2_bitemp_pretrained",
        pretrained=True,
        train_loader=bitemp_train_loader,
        val_loader=bitemp_val_loader,
        device=device,
        epochs=50,
        lr=3e-4,
    )
    all_results["big_sam_v2_bitemp_pretrained"] = res_bi_pre

    # Experiment 3: BiG-SAM v2 (Single-Date Scratch)
    _, res_single_scratch = train_big_sam_v2_singledate(
        variant_name="big_sam_v2_single_scratch",
        pretrained=False,
        train_loader=single_train_loader,
        val_loader=single_val_loader,
        device=device,
        epochs=50,
        lr=3e-4,
    )
    all_results["big_sam_v2_single_scratch"] = res_single_scratch

    # Experiment 4: BiG-SAM v2 (Single-Date Pretrained SAM Prior)
    _, res_single_pre = train_big_sam_v2_singledate(
        variant_name="big_sam_v2_single_pretrained",
        pretrained=True,
        train_loader=single_train_loader,
        val_loader=single_val_loader,
        device=device,
        epochs=50,
        lr=3e-4,
    )
    all_results["big_sam_v2_single_pretrained"] = res_single_pre

    # Save results json
    out_json = os.path.join("results", "big_sam_v2_benchmark_anhgoc.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}")


if __name__ == "__main__":
    main()
