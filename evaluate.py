"""
Evaluation Script for Can Gio Mangrove Canopy Disturbance Mapping Models.

Evaluates trained model checkpoints across validation/test scenes:
  - Binary segmentation metrics: IoU, Dice (F1), Precision, Recall, Accuracy.
  - Boundary delineation metrics: Boundary IoU (Cheng et al., CVPR 2021).
  - Optimal threshold determination via validation sweep.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

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
from src.models import GRAPH_MODEL_NAMES, BiTemporalWrapper, build_model


@torch.no_grad()
def evaluate_singledate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_targets = []
    scene_probs = []
    scene_targets = []

    for images, masks, _ in val_loader:
        images = images.to(device)
        out = model(images)
        logits = out["seg_logits"] if isinstance(out, dict) else (out.seg_logits if hasattr(out, "seg_logits") else out)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets_np = masks.numpy()

        for i in range(images.shape[0]):
            p = probs[i, 0]
            t = targets_np[i, 0]
            scene_probs.append(p)
            scene_targets.append(t)
            all_probs.append(p.flatten())
            all_targets.append(t.flatten())

    cat_probs = np.concatenate(all_probs)
    cat_targets = np.concatenate(all_targets)

    if threshold is None:
        best_th = 0.5
        best_f1 = -1.0
        for th in np.linspace(0.15, 0.85, 35):
            m = compute_binary_metrics(cat_probs, cat_targets, threshold=float(th))
            if m["dice"] > best_f1:
                best_f1 = m["dice"]
                best_th = float(th)
        threshold = best_th

    final_metrics = compute_binary_metrics(cat_probs, cat_targets, threshold=threshold)
    final_metrics["f1"] = final_metrics["dice"]
    bious = [compute_boundary_iou(p, t, threshold=threshold) for p, t in zip(scene_probs, scene_targets)]
    final_metrics["boundary_iou"] = float(np.mean(bious))
    final_metrics["threshold"] = round(threshold, 4)
    return final_metrics


@torch.no_grad()
def evaluate_bitemporal_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    from scripts.run_all_models_bitemporal_benchmark import evaluate_bitemporal_wrapper
    return evaluate_bitemporal_wrapper(model, val_loader, device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Can Gio Canopy Disturbance Models")
    parser.add_argument("--model", type=str, default="big_sam_v2", choices=GRAPH_MODEL_NAMES, help="Model architecture")
    parser.add_argument("--mode", type=str, default="bitemporal", choices=["single", "bitemporal"], help="Single-date or Bi-temporal")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--source", type=str, default="anhgoc", choices=["anhgoc", "cut"], help="Data source mode")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (or None for auto-sweep)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 80)
    print(f"EVALUATING MODEL: {args.model.upper()} ({args.mode.upper()})")
    print(f"Device: {device} | Checkpoint: {args.checkpoint or 'Initialized weights'}")
    print("=" * 80)

    cfg = Config(data_dir=args.data_dir, data_source=args.source)
    scenes = discover_scenes(cfg.data_dir, source=args.source)
    val_scenes = [s for s in scenes if s["year_folder"] in ["Cut_2011", "Cut_2014"]]

    if not val_scenes:
        print("[!] No validation scenes discovered. Check data directory.")
        return

    # Build model
    base_m = build_model(args.model, in_channels=9, num_classes=1)
    if args.mode == "bitemporal":
        mode = "native_5d" if args.model == "cgmn_net" else "difference"
        eval_model = BiTemporalWrapper(base_m, mode=mode, in_channels=9)
    else:
        eval_model = base_m

    # Load weights if given
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[+] Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        eval_model.load_state_dict(state_dict, strict=False)

    eval_model.to(device)

    if args.mode == "single":
        val_ds = CanGioDataset(val_scenes, augment=False, require_gt=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        metrics = evaluate_singledate_model(eval_model, val_loader, device, threshold=args.threshold)
    else:
        val_ds = BiTemporalCanGioDataset(val_scenes, pairing_strategy="consecutive", augment=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        metrics = evaluate_bitemporal_model(eval_model, val_loader, device)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    print("=" * 50)
    print(f"  * Dice / F1 Score:  {metrics.get('f1', metrics.get('dice', 0.0)):.4f}")
    print(f"  * IoU:              {metrics.get('iou', 0.0):.4f}")
    print(f"  * Precision:        {metrics.get('precision', 0.0):.4f}")
    print(f"  * Recall:           {metrics.get('recall', 0.0):.4f}")
    print(f"  * Accuracy:         {metrics.get('accuracy', 0.0):.4f}")
    print(f"  * Boundary IoU:     {metrics.get('boundary_iou', 0.0):.4f}")
    print(f"  * Optimal Thresh:   {metrics.get('threshold', 0.5):.4f}")
    print("=" * 50)

    out_file = os.path.join("results", f"eval_{args.model}_{args.mode}.json")
    os.makedirs("results", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Results saved to: {out_file}")


if __name__ == "__main__":
    main()
