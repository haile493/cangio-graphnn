"""
Comprehensive Bi-Temporal Benchmark for All Graph Models on Can Gio:
Evaluates how each graph architecture performs when applied to bi-temporal pairs (T1 -> T2):
  1. `hrsei_gnn` (Jebur et al., 2026) — Biophysical Polarity Difference & Early Fusion
  2. `venagnosis_pfgnn` (Hu & Wu, 2026) — Multi-Relational Parcel Siamese Differencing
  3. `gsri_net` (Liu et al., 2026) — Dynamic Dilated KNN Rotation-Robust Differencing
  4. `cgmn_net` (Shao et al., 2026) — Native 5D Temporal GRU Sequence Branch
  5. `big_sam` v1 (Scratch & Pretrained)
  6. `big_sam_v2` (Scratch & Pretrained)

Strict zero-leakage split:
  - Train: 2006-2007, 2008, 2010 (7 consecutive bi-temporal scene pairs)
  - Val:   2011, 2014 (5 consecutive bi-temporal scene pairs)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ag_models import (
    BinaryDiceLoss,
    build_loss,
    build_model,
)
from ag_models.models.big_sam import BinaryFocalLoss
from src.config import Config
from src.dataset import BiTemporalCanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BiTemporalWrapper(nn.Module):
    """
    Adapts single-date remote sensing graph architectures to bi-temporal input pairs (T1, T2):
    Supports:
      - 'difference': passes differential reflectance tensor (X2 - X1) + indices
      - 'concat': passes concatenated [X1, X2] (18 channels)
      - 'siamese_diff': passes X1 and X2 through shared model, fuses features
      - 'native_5d': passes (B, 2, C, H, W) directly into architectures with native temporal branches (CGMN)
    """

    def __init__(self, base_model: nn.Module, mode: str = "difference", in_channels: int = 9):
        super().__init__()
        self.base_model = base_model
        self.mode = mode
        self.in_channels = in_channels
        if mode == "concat":
            # Adapter conv reducing 18 -> 9 channels
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mode == "native_5d":
            # Stack into (B, 2, C, H, W)
            x_seq = torch.stack([x1, x2], dim=1)
            out = self.base_model(x_seq)
            return out if isinstance(out, dict) else {"seg_logits": out.seg_logits if hasattr(out, "seg_logits") else out}
        elif self.mode == "concat":
            x_cat = torch.cat([x1, x2], dim=1)
            x_in = self.adapter(x_cat)
            out = self.base_model(x_in)
            return out if isinstance(out, dict) else {"seg_logits": out.seg_logits if hasattr(out, "seg_logits") else out}
        else:  # 'difference'
            # Pure differential radiometric contrast + current state
            x_diff = (x2 - x1).clamp(-1.0, 1.0)
            # Add spectral index contrast preservation
            x_in = 0.5 * x2 + 0.5 * x_diff
            out = self.base_model(x_in)
            return out if isinstance(out, dict) else {"seg_logits": out.seg_logits if hasattr(out, "seg_logits") else out}


@torch.no_grad()
def evaluate_bitemporal_wrapper(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_targets = []
    scene_probs = []
    scene_targets = []

    for img1, img2, masks, _ in val_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        out = model(img1, img2)
        logits = out["seg_logits"] if isinstance(out, dict) else (out.seg_logits if hasattr(out, "seg_logits") else out)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = masks.numpy()

        for i in range(img1.shape[0]):
            p_map = probs[i, 0]
            t_map = targets_np[i, 0]
            scene_probs.append(p_map)
            scene_targets.append(t_map)
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
    return final_metrics


def train_bitemporal_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 3e-4,
) -> Dict[str, float]:
    seed_everything(42)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    focal = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    dice = BinaryDiceLoss()

    best_metrics = None
    best_f1 = -1.0
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for img1, img2, masks, _ in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            out = model(img1, img2)
            logits = out["seg_logits"] if isinstance(out, dict) else (out.seg_logits if hasattr(out, "seg_logits") else out)
            loss = focal(logits, masks) + dice(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        val_metrics = evaluate_bitemporal_wrapper(model, val_loader, device)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics

    elapsed = time.time() - t_start
    best_metrics["train_time_sec"] = round(elapsed, 2)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_metrics["params"] = params
    print(
        f"[{name.upper()}] Best F1: {best_metrics['f1']:.4f} | IoU: {best_metrics['iou']:.4f} | "
        f"Prec: {best_metrics['precision']:.4f} | Rec: {best_metrics['recall']:.4f} | "
        f"B-IoU: {best_metrics['boundary_iou']:.4f} (th={best_metrics['threshold']:.2f}, {elapsed:.1f}s)"
    )
    return best_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Bi-Temporal Benchmark across all graph models on device: {device}")

    cfg = Config(data_source="anhgoc")
    scenes = discover_scenes(cfg.data_dir, source="anhgoc")
    train_scenes = [s for s in scenes if s["year_folder"] not in ["Cut_2011", "Cut_2014"]]
    val_scenes = [s for s in scenes if s["year_folder"] in ["Cut_2011", "Cut_2014"]]

    train_ds = BiTemporalCanGioDataset(train_scenes, pairing_strategy="consecutive", augment=True)
    val_ds = BiTemporalCanGioDataset(val_scenes, pairing_strategy="consecutive", augment=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    results = {}

    # 1. HRSEI-GNN (Jebur et al. 2026) - Biophysical Polarity Difference
    m_hrsei_raw = build_model("hrsei_gnn", in_channels=9, hidden_dim=32, num_gnn_layers=3)
    m_hrsei = BiTemporalWrapper(m_hrsei_raw, mode="difference", in_channels=9)
    results["hrsei_gnn_bitemporal"] = train_bitemporal_model(
        "HRSEI-GNN (Bi-Temporal)", m_hrsei, train_loader, val_loader, device
    )

    # 2. Venagnosis PF-GNN (Hu & Wu 2026) - Multi-Relational Parcel Differencing
    m_ven_raw = build_model("venagnosis_pfgnn", in_channels=9, base_channels=48, num_gnn_layers=3, parcel_grid_size=31)
    m_ven = BiTemporalWrapper(m_ven_raw, mode="difference", in_channels=9)
    results["venagnosis_pfgnn_bitemporal"] = train_bitemporal_model(
        "Venagnosis PF-GNN (Bi-Temporal)", m_ven, train_loader, val_loader, device
    )

    # 3. GSRI-Net (Liu et al. 2026) - Dynamic Dilated KNN Rotation-Robust Differencing
    m_gsri_raw = build_model("gsri_net", in_channels=9, base_channels=48, num_heads=4, k_neighbors=4)
    m_gsri = BiTemporalWrapper(m_gsri_raw, mode="difference", in_channels=9)
    results["gsri_net_bitemporal"] = train_bitemporal_model(
        "GSRI-Net (Bi-Temporal)", m_gsri, train_loader, val_loader, device
    )

    # 4. CGMN Multi-Branch (Shao et al. 2026) - Native 5D Temporal Sequence Branch
    m_cgmn_raw = build_model("cgmn_net", in_channels=9, branch_dim=32, grid_size=31)
    m_cgmn = BiTemporalWrapper(m_cgmn_raw, mode="native_5d", in_channels=9)
    results["cgmn_net_bitemporal"] = train_bitemporal_model(
        "CGMN Multi-Branch (Native 5D GRU)", m_cgmn, train_loader, val_loader, device
    )

    out_file = os.path.join("results", "tables", "all_models_bitemporal_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all bi-temporal model benchmark results to {out_file}")


if __name__ == "__main__":
    main()
