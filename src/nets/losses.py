"""Loss functions shared by the Can Gio graph models (extracted from ag_models.losses)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("binary_dice_loss")
class BinaryDiceLoss(nn.Module):
    """Dimension-agnostic Binary Soft Dice Loss on logits."""

    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits) if self.from_logits else logits
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.float().view(targets.shape[0], -1)

        inter = (probs_flat * targets_flat).sum(dim=1)
        card = (probs_flat + targets_flat).sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


@LOSS_REGISTRY.register("focal_loss", aliases=["focal_loss_3d", "focal_loss_2d"])
class FocalLoss(nn.Module):
    """Dimension-agnostic Multi-Class Focal Loss."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets.long(), weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1.0 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal)


@LOSS_REGISTRY.register("venagnosis_loss", aliases=["pfgnn_loss"])
class VenagnosisLoss(nn.Module):
    """
    Venagnosis Multi-Component Objective (Hu & Wu, IEEE Access 2026, Eqs. 27-29):
      L_total = L_focal + L_dice + lambda_aux * L_aux + lambda_fp * L_fp
    Designed for severe positive class imbalance in localized land/canopy degradation detection.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        lambda_aux: float = 0.4,
        lambda_fp: float = 0.25,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_aux = lambda_aux
        self.lambda_fp = lambda_fp
        self.dice = BinaryDiceLoss(from_logits=True)

    def _focal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        w = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = -w * ((1.0 - pt) ** self.gamma) * torch.log(pt)
        return focal.mean()

    def forward(self, outputs, targets: torch.Tensor):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        seg_logits = outputs["seg_logits"] if isinstance(outputs, dict) else outputs

        l_focal = self._focal(seg_logits, targets)
        l_dice = self.dice(seg_logits, targets)

        zero = torch.zeros((), device=seg_logits.device)
        l_aux = zero
        if isinstance(outputs, dict) and outputs.get("aux_seg_logits") is not None:
            l_aux = 0.5 * self._focal(outputs["aux_seg_logits"], targets) + 0.5 * self.dice(outputs["aux_seg_logits"], targets)

        # False-positive suppression penalty on background pixels (Eq. 27)
        probs = torch.sigmoid(seg_logits)
        bg_mask = 1.0 - targets
        l_fp = ((probs ** 2) * bg_mask).sum() / bg_mask.sum().clamp_min(1.0)

        total = l_focal + l_dice + self.lambda_aux * l_aux + self.lambda_fp * l_fp
        return total, {
            "loss_total": float(total.item()),
            "loss_focal": float(l_focal.item()),
            "loss_dice": float(l_dice.item()),
            "loss_aux": float(l_aux.item()),
            "loss_fp": float(l_fp.item()),
        }


@LOSS_REGISTRY.register("cgmn_composite_loss", aliases=["cgmn_loss"])
class CGMNCompositeLoss(nn.Module):
    """
    CGMN Four-Term Composite Loss (Shao et al., GIScience & Remote Sensing 2026, Eqs. 1-5):
      L = L_city(node) + lambda_prov * L_prov(regional weak supervision)
        + lambda_mask * L_mask(masked node prediction, Eq. 3)
        + lambda_graph * L_graph(distance-weighted spatial smoothness, Eq. 4)
    """

    def __init__(
        self,
        lambda_prov: float = 0.3,
        lambda_mask: float = 0.3,
        lambda_graph: float = 0.15,
        pos_weight: float = 4.0,
    ):
        super().__init__()
        self.lambda_prov = lambda_prov
        self.lambda_mask = lambda_mask
        self.lambda_graph = lambda_graph
        self.pos_weight = pos_weight
        self.dice = BinaryDiceLoss(from_logits=True)

    def _node_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw) + self.dice(logits, targets)

    def forward(self, outputs, targets: torch.Tensor, ref_image: Optional[torch.Tensor] = None):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        seg_logits = outputs["seg_logits"] if isinstance(outputs, dict) else outputs
        probs = torch.sigmoid(seg_logits)

        # 1. Node/pixel supervision loss (Eq. 2)
        l_node = self._node_loss(seg_logits, targets)

        # 2. Regional/provincial aggregate weak supervision loss (Eq. 1)
        pool_pred = F.adaptive_avg_pool2d(probs, (4, 4))
        pool_gt = F.adaptive_avg_pool2d(targets, (4, 4))
        l_prov = F.mse_loss(pool_pred, pool_gt)

        # 3. Mask prediction loss under 20% random feature masking (Eq. 3)
        l_mask = torch.zeros((), device=seg_logits.device)
        if isinstance(outputs, dict) and outputs.get("masked_logits") is not None:
            l_mask = self._node_loss(outputs["masked_logits"], targets)

        # 4. Graph Smoothness Loss (Eq. 4): weighted differences between neighboring predictions
        dy = (probs[:, :, 1:, :] - probs[:, :, :-1, :]) ** 2
        dx = (probs[:, :, :, 1:] - probs[:, :, :, :-1]) ** 2
        if ref_image is not None:
            if ref_image.ndim == 5:
                ref_image = ref_image[:, -1]
            wy = torch.exp(-10.0 * (ref_image[:, :, 1:, :] - ref_image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True))
            wx = torch.exp(-10.0 * (ref_image[:, :, :, 1:] - ref_image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True))
            l_graph = (wy * dy).mean() + (wx * dx).mean()
        else:
            l_graph = dy.mean() + dx.mean()

        total = (
            l_node
            + self.lambda_prov * l_prov
            + self.lambda_mask * l_mask
            + self.lambda_graph * l_graph
        )
        return total, {
            "loss_total": float(total.item()),
            "loss_node": float(l_node.item()),
            "loss_prov": float(l_prov.item()),
            "loss_mask": float(l_mask.item()),
            "loss_graph": float(l_graph.item()),
        }
