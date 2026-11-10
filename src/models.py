"""
Graph-Based Models for Can Gio Mangrove Canopy Disturbance Mapping.

Architectures:
  1. `hrsei_gnn`         -> `HRSEIGNNNet` (Jebur et al., J. Hazard. Mater. Adv. 2026)
  2. `venagnosis_pfgnn`  -> `VenagnosisPFGNNNet` (Hu & Wu, IEEE Access 2026)
  3. `gsri_net`          -> `GSRIRotationRobustNet` (Liu et al., ACM ICMR 2026)
  4. `cgmn_net`          -> `CGMNMultiBranchNet` (Shao et al., GIScience & Remote Sensing 2026)
  5. `big_sam`           -> `BiGSAMNet` (Bi-Temporal Graph Segment Anything v1)
  6. `big_sam_v2`        -> `BiGSAMNetV2` (Graph-Augmented Foundation Architecture v2)
"""

from typing import Any, Dict
import torch
import torch.nn as nn

try:
    from ag_models import (
        BiGSAMConfig,
        BiGSAMLoss,
        BiGSAMLossV2,
        BiGSAMNet,
        BiGSAMNetV2,
        BiGSAMV2Config,
        CGMNCompositeLoss,
        CGMNMultiBranchNet,
        GSRIRotationRobustNet,
        HRSEIGNNNet,
        VenagnosisLoss,
        VenagnosisPFGNNNet,
        BinaryDiceLoss,
        FocalLoss,
        build_loss,
        build_model,
    )
    from ag_models.models.big_sam import BinaryFocalLoss
except ImportError:
    from src.models_fallback import (
        BiGSAMConfig,
        BiGSAMLoss,
        BiGSAMLossV2,
        BiGSAMNet,
        BiGSAMNetV2,
        BiGSAMV2Config,
        CGMNCompositeLoss,
        CGMNMultiBranchNet,
        GSRIRotationRobustNet,
        HRSEIGNNNet,
        VenagnosisLoss,
        VenagnosisPFGNNNet,
        BinaryDiceLoss,
        BinaryFocalLoss,
        FocalLoss,
        build_loss,
        build_model,
    )

GRAPH_MODEL_NAMES = [
    "big_sam",
    "big_sam_v2",
    "hrsei_gnn",
    "venagnosis_pfgnn",
    "gsri_net",
    "cgmn_net",
]


class BiTemporalWrapper(nn.Module):
    """Adapts single-date remote sensing graph architectures to bi-temporal input pairs (T1, T2)."""

    def __init__(self, base_model: nn.Module, mode: str = "difference", in_channels: int = 9):
        super().__init__()
        self.base_model = base_model
        self.mode = mode
        self.in_channels = in_channels
        if mode == "concat":
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mode == "native_5d":
            x_seq = torch.stack([x1, x2], dim=1)
            out = self.base_model(x_seq)
            return out if isinstance(out, dict) else {"seg_logits": out.seg_logits if hasattr(out, "seg_logits") else out}
        elif self.mode == "concat":
            x_cat = torch.cat([x1, x2], dim=1)
            x_in = self.adapter(x_cat)
            out = self.base_model(x_in)
            return out if isinstance(out, dict) else {"seg_logits": out.seg_logits if hasattr(out, "seg_logits") else out}
        else:  # 'difference'
            x_diff = (x2 - x1).clamp(-1.0, 1.0)
            x_in = 0.5 * x2 + 0.5 * x_diff
            out = self.base_model(x_in)
            return out if isinstance(out, dict) else {"seg_logits": out.seg_logits if hasattr(out, "seg_logits") else out}


def create_graph_model(arch: str, in_channels: int = 9, num_classes: int = 1, **kwargs: Any) -> nn.Module:
    """Builds one of the graph-based remote sensing models."""
    return build_model(arch, in_channels=in_channels, num_classes=num_classes, **kwargs)


__all__ = [
    "GRAPH_MODEL_NAMES",
    "BiGSAMNet",
    "BiGSAMLoss",
    "BiGSAMNetV2",
    "BiGSAMLossV2",
    "BiGSAMConfig",
    "BiGSAMV2Config",
    "HRSEIGNNNet",
    "VenagnosisPFGNNNet",
    "GSRIRotationRobustNet",
    "CGMNMultiBranchNet",
    "BiTemporalWrapper",
    "BinaryDiceLoss",
    "BinaryFocalLoss",
    "FocalLoss",
    "VenagnosisLoss",
    "CGMNCompositeLoss",
    "create_graph_model",
    "build_model",
    "build_loss",
]
