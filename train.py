"""
Unified Training Script for Can Gio Mangrove Canopy Disturbance Mapping.

Supports Single-Date and Bi-Temporal training for 6 Graph Neural Network Architectures:
  1. `hrsei_gnn`        (Jebur et al., J. Hazard. Mater. Adv. 2026)
  2. `venagnosis_pfgnn` (Hu & Wu, IEEE Access 2026)
  3. `gsri_net`         (Liu et al., ACM ICMR 2026)
  4. `cgmn_net`         (Shao et al., GIScience & Remote Sensing 2026)
  5. `big_sam`          (Bi-temporal Graph Segment Anything Foundation Model v1)
  6. `big_sam_v2`       (Graph-Augmented Foundation Architecture v2)

Default training schedule: 50 epochs with Cosine Annealing learning rate schedule.
"""

import argparse
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import Config
from src.dataset import CanGioDataset, BiTemporalCanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou
from src.models import (
    GRAPH_MODEL_NAMES,
    BiTemporalWrapper,
    build_loss,
    build_model,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_date(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1.5e-3,
    save_dir: str = "results/models",
) -> Dict[str, float]:
    os.makedirs(save_dir, exist_ok=True)
    model = build_model(model_name, in_channels=9, num_classes=1).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ven_loss_fn = build_loss("venagnosis_loss", alpha=0.80, gamma=2.0).to(device)
    cgmn_loss_fn = build_loss("cgmn_composite_loss", pos_weight=6.0).to(device)
    dice_fn = build_loss("binary_dice_loss", from_logits=True).to(device)

    best_f1 = -1.0
    best_metrics = None
    ckpt_path = os.path.join(save_dir, f"best_{model_name}_single.pth")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if model_name == "venagnosis_pfgnn":
                loss, _ = ven_loss_fn(outputs, masks)
            elif model_name == "cgmn_net":
                loss, _ = cgmn_loss_fn(outputs, masks, ref_image=images)
            else:
                logits = outputs["seg_logits"] if isinstance(outputs, dict) else (outputs.seg_logits if hasattr(outputs, "seg_logits") else outputs)
                loss = dice_fn(logits, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for v_img, v_mask, _ in val_loader:
                    v_img = v_img.to(device)
                    out = model(v_img)
                    logits = out["seg_logits"] if isinstance(out, dict) else (out.seg_logits if hasattr(out, "seg_logits") else out)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    all_preds.append(probs)
                    all_targets.append(v_mask.numpy().flatten())

            cat_preds = np.concatenate(all_preds)
            cat_targets = np.concatenate(all_targets)

            best_th = 0.5
            ep_f1 = -1.0
            for th in np.linspace(0.2, 0.8, 13):
                m = compute_binary_metrics(cat_preds, cat_targets, threshold=float(th))
                if m["dice"] > ep_f1:
                    ep_f1 = m["dice"]
                    best_th = float(th)

            cur_m = compute_binary_metrics(cat_preds, cat_targets, threshold=best_th)
            cur_m["threshold"] = round(best_th, 4)
            print(f"Epoch [{ep:02d}/{epochs:02d}] Loss: {train_loss / len(train_loader):.4f} | Val F1: {cur_m['dice']:.4f} | IoU: {cur_m['iou']:.4f}")

            if cur_m["dice"] > best_f1:
                best_f1 = cur_m["dice"]
                best_metrics = cur_m
                torch.save({"model_state": model.state_dict(), "metrics": cur_m, "epoch": ep}, ckpt_path)

    elapsed = time.time() - t0
    best_metrics["train_time_sec"] = round(elapsed, 2)
    best_metrics["params"] = params
    print(f"\nTraining Complete: {model_name} (Single-Date) -> Best F1: {best_metrics['dice']:.4f}, IoU: {best_metrics['iou']:.4f} ({elapsed:.1f}s)")
    return best_metrics


def train_bitemporal(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 3e-4,
    save_dir: str = "results/models",
) -> Dict[str, float]:
    from scripts.run_all_models_bitemporal_benchmark import train_bitemporal_model

    os.makedirs(save_dir, exist_ok=True)
    mode = "native_5d" if model_name == "cgmn_net" else "difference"
    base_m = build_model(model_name, in_channels=9, num_classes=1)
    wrapped_model = BiTemporalWrapper(base_m, mode=mode, in_channels=9)

    return train_bitemporal_model(
        name=f"{model_name}_bitemporal",
        model=wrapped_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Graph Neural Networks on Can Gio Canopy Disturbance")
    parser.add_argument("--model", type=str, default="big_sam_v2", choices=GRAPH_MODEL_NAMES, help="Model architecture")
    parser.add_argument("--mode", type=str, default="bitemporal", choices=["single", "bitemporal"], help="Single-date or Bi-temporal")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--source", type=str, default="anhgoc", choices=["anhgoc", "cut"], help="Data source mode")
    parser.add_argument("--save-dir", type=str, default="results/models", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TRAINING CAN GIO GRAPH NEURAL NETWORK")
    print(f"Architecture: {args.model.upper()} | Mode: {args.mode.upper()} | Epochs: {args.epochs} | Device: {device}")
    print("=" * 80)

    cfg = Config(data_dir=args.data_dir, data_source=args.source)
    scenes = discover_scenes(cfg.data_dir, source=args.source)
    train_scenes = [s for s in scenes if s["year_folder"] not in ["Cut_2011", "Cut_2014"]]
    val_scenes = [s for s in scenes if s["year_folder"] in ["Cut_2011", "Cut_2014"]]

    if args.mode == "single":
        train_ds = CanGioDataset(train_scenes, augment=True, require_gt=True)
        val_ds = CanGioDataset(val_scenes, augment=False, require_gt=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        metrics = train_single_date(
            model_name=args.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
        )
    else:
        train_ds = BiTemporalCanGioDataset(train_scenes, pairing_strategy="consecutive", augment=True)
        val_ds = BiTemporalCanGioDataset(val_scenes, pairing_strategy="consecutive", augment=False)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        metrics = train_bitemporal(
            model_name=args.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
        )

    out_json = os.path.join("results", f"{args.model}_{args.mode}_metrics.json")
    os.makedirs("results", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved evaluation metrics to: {out_json}")


if __name__ == "__main__":
    main()
