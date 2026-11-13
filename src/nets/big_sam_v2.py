"""
BiG-SAM v2: Enhanced Graph-Augmented Foundation Architecture for Earth Observation
Canopy Disturbance Mapping and Remote Sensing Change Detection.

Registered as `big_sam_v2` in `MODEL_REGISTRY`.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.base import BaseModel, ModelConfig, ModelOutput
from src.nets.registry import LOSS_REGISTRY, MODEL_REGISTRY
from src.nets.losses import BinaryDiceLoss
from src.nets.big_sam import BinaryFocalLoss
from src.nets.big_sam_v2_modules import (
    BiTemporalBipartiteDifferencingV2,
    DenseBoundaryRefinementDecoderV2,
    DualVeinGraphReasoner,
    MultiSpectralFoundationAdapterV2,
)

DEFAULT_SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "results/models/sam_vit_b_01ec64.pth")


@dataclass
class BiGSAMV2Config(ModelConfig):
    """Configuration for BiG-SAM v2 Model."""
    in_channels: int = 9
    num_classes: int = 1
    base_channels: int = 48
    parcel_grid_size: int = 31
    num_gnn_layers: int = 3
    num_heads: int = 4
    k_semantic: int = 4
    k_rotation: int = 4
    corridor_span: int = 2
    use_sam_prior: bool = True
    pretrained_sam_path: str = DEFAULT_SAM_CHECKPOINT


@MODEL_REGISTRY.register("big_sam_v2", aliases=["big_sam_gnn_hybrid", "venagnosis_sam_hybrid"])
class BiGSAMNetV2(BaseModel):
    """
    BiG-SAM v2: Graph-Augmented Foundation Architecture for Canopy Disturbance Mapping.
    Fuses the 4 breakthrough 2026 remote sensing graph paradigms with foundation visual priors:
      1. Biophysical Polarity Normalization & Synergistic Parcel Characterization (Jebur et al. & Hu & Wu)
      2. Multi-Relational Knowledge Graph + Progressive Cross-Layer Fusion (Hu & Wu PF-GNN)
      3. Dynamic Dilated KNN Max-Relative Rotation-Robust Interaction (Liu et al. GSRI)
      4. Bitemporal Bipartite Cross-Attention & Differential Perturbation
      5. Dense Multi-Scale Boundary & Ecological Refinement Decoder
    """

    config_class = BiGSAMV2Config

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 1,
        base_channels: int = 48,
        parcel_grid_size: int = 31,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        k_semantic: int = 4,
        k_rotation: int = 4,
        corridor_span: int = 2,
        use_sam_prior: bool = True,
        pretrained: bool = True,
        pretrained_sam_path: str = DEFAULT_SAM_CHECKPOINT,
        **kwargs: Any,
    ):
        super().__init__(
            BiGSAMV2Config(
                arch="big_sam_v2",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                base_channels=base_channels,
                extra={
                    "parcel_grid_size": parcel_grid_size,
                    "num_gnn_layers": num_gnn_layers,
                    "num_heads": num_heads,
                    "k_semantic": k_semantic,
                    "k_rotation": k_rotation,
                    "corridor_span": corridor_span,
                    "use_sam_prior": use_sam_prior and pretrained,
                },
            )
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.parcel_grid_size = parcel_grid_size
        self.use_sam_prior = use_sam_prior and pretrained

        # 1. Multi-Spectral Foundation Adapter with Biophysical Polarity
        self.adapter = MultiSpectralFoundationAdapterV2(
            in_channels=in_channels,
            feat_dim=base_channels,
            use_sam_prior=self.use_sam_prior,
            pretrained_sam_path=pretrained_sam_path if self.use_sam_prior else None,
        )

        # 2. Dual-Vein Multi-Relational & Rotation-Robust Graph Reasoner
        self.graph_reasoner = DualVeinGraphReasoner(
            dim=base_channels,
            num_gnn_layers=num_gnn_layers,
            num_heads=num_heads,
            k_semantic=k_semantic,
            k_rotation=k_rotation,
            corridor_span=corridor_span,
        )

        # 3. Bitemporal Bipartite Cross-Attention & Differential Perturbation Block
        self.bitemporal_block = BiTemporalBipartiteDifferencingV2(
            dim=base_channels,
            num_heads=num_heads,
        )

        # 4. Dense Boundary & Ecological Refinement Decoder
        self.decoder = DenseBoundaryRefinementDecoderV2(
            feat_dim=base_channels,
            num_classes=num_classes,
        )

        self.loss_fn = BiGSAMLossV2()

    def _extract_and_reason_single(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int], torch.Tensor]:
        """
        Extracts features and performs dual-vein graph reasoning for a single timestamp.
        """
        B, C, H, W = x.shape
        f_spc, f_loc1, f_loc2, f_found = self.adapter(x)

        # Downsample to parcel grid
        hg = min(H, self.parcel_grid_size)
        wg = min(W, self.parcel_grid_size)

        h_parcel = F.adaptive_avg_pool2d(f_loc2, (hg, wg))
        if f_found is not None:
            f_found_pooled = F.adaptive_avg_pool2d(f_found, (hg, wg))
            h_parcel = h_parcel + 0.3 * f_found_pooled

        h_nodes = h_parcel.flatten(2).transpose(1, 2)  # (B, N, D)

        # Dual-vein graph reasoning
        fused_nodes, f_state = self.graph_reasoner(
            nodes=h_nodes,
            grid_hw=(hg, wg),
            feat_map=h_parcel,
        )
        return fused_nodes, f_loc1, f_loc2, (hg, wg), f_state

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """
        Forward pass supporting both Single-Date and Bi-Temporal modalities:
        - Single-date: forward(x1)
        - Bi-temporal: forward(x1, x2) or forward(x_5d)
        """
        if x2 is None:
            if x1.dim() == 5:
                x2 = x1[:, 1]
                x1 = x1[:, 0]
            elif x1.shape[1] == 2 * self.in_channels:
                x2 = x1[:, self.in_channels :]
                x1 = x1[:, : self.in_channels]
            # else: x2 remains None -> single date mode!

        orig_hw = (x1.shape[2], x1.shape[3])

        if x2 is not None:
            # Bi-temporal mode
            nodes1, _, _, _, _ = self._extract_and_reason_single(x1)
            nodes2, f_loc1_2, f_loc2_2, grid_hw, f_state2 = self._extract_and_reason_single(x2)
            delta_nodes = self.bitemporal_block(nodes2=nodes2, nodes1=nodes1)
            f_loc1_tgt, f_loc2_tgt = f_loc1_2, f_loc2_2
            ref_image = x2
        else:
            # Single-date mode
            nodes1, f_loc1_1, f_loc2_1, grid_hw, f_state1 = self._extract_and_reason_single(x1)
            delta_nodes = self.bitemporal_block(nodes2=nodes1, nodes1=None)
            f_loc1_tgt, f_loc2_tgt = f_loc1_1, f_loc2_1
            ref_image = x1

        # Multi-scale boundary and ecological refinement decoding
        seg_logits, edge_logits, hrsei_index, aux_seg_logits = self.decoder(
            f_loc1=f_loc1_tgt,
            f_loc2=f_loc2_tgt,
            delta_nodes=delta_nodes,
            grid_hw=grid_hw,
            orig_hw=orig_hw,
        )

        loss = None
        loss_dict = {}
        if targets is not None:
            loss, loss_dict = self.loss_fn(
                seg_logits=seg_logits,
                edge_logits=edge_logits,
                hrsei_index=hrsei_index,
                aux_seg_logits=aux_seg_logits,
                targets=targets,
                ref_image=ref_image,
            )

        return ModelOutput(
            seg_logits=seg_logits,
            edge_logits=edge_logits,
            hrsei_index=hrsei_index,
            aux_seg_logits=aux_seg_logits,
            latent_feat=delta_nodes,
            loss=loss,
            extra=loss_dict,
        )


# =============================================================================
# Compound Multi-Objective Loss for BiG-SAM v2
# =============================================================================


def _compute_gt_edge(mask: torch.Tensor) -> torch.Tensor:
    """Computes morphological boundary target from binary ground-truth mask."""
    max_p = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    min_p = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (max_p - min_p).clamp(0.0, 1.0)


@LOSS_REGISTRY.register("big_sam_v2_loss")
class BiGSAMLossV2(nn.Module):
    """
    Compound Multi-Objective Decoupled Loss for BiG-SAM v2:
      1. Focal + Binary Dice Loss on primary disturbance logits (seg_logits)
      2. Boundary Morphological BCE Loss on edge_logits (Liu et al. GSRI)
      3. Biophysical Ecological Consistency MSE Loss on hrsei_index (Jebur et al. HRSEI)
      4. Auxiliary Parcel Classification Loss on aux_seg_logits (Hu & Wu Venagnosis)
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.80,
        edge_weight: float = 0.30,
        eco_weight: float = 0.25,
        parcel_weight: float = 0.30,
        lambda_fp: float = 0.25,
    ):
        super().__init__()
        self.focal_fn = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice_fn = BinaryDiceLoss()
        self.edge_weight = edge_weight
        self.eco_weight = eco_weight
        self.parcel_weight = parcel_weight
        self.lambda_fp = lambda_fp

    def forward(
        self,
        outputs: Optional[Union[ModelOutput, Dict[str, torch.Tensor]]] = None,
        targets: Optional[torch.Tensor] = None,
        seg_logits: Optional[torch.Tensor] = None,
        edge_logits: Optional[torch.Tensor] = None,
        hrsei_index: Optional[torch.Tensor] = None,
        aux_seg_logits: Optional[torch.Tensor] = None,
        ref_image: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if outputs is not None:
            seg_logits = outputs.get("seg_logits") if isinstance(outputs, dict) else outputs.seg_logits
            edge_logits = outputs.get("edge_logits") if isinstance(outputs, dict) else getattr(outputs, "edge_logits", None)
            hrsei_index = outputs.get("hrsei_index") if isinstance(outputs, dict) else getattr(outputs, "hrsei_index", None)
            aux_seg_logits = outputs.get("aux_seg_logits") if isinstance(outputs, dict) else getattr(outputs, "aux_seg_logits", None)

        if targets is None:
            raise ValueError("Target masks must be provided for BiGSAMLossV2.")

        # 1. Main segmentation loss
        l_focal = self.focal_fn(seg_logits, targets)
        l_dice = self.dice_fn(seg_logits, targets)
        l_seg = l_focal + l_dice
        total_loss = l_seg

        # False-positive background suppression penalty (Hu & Wu 2026 Eq. 27)
        probs = torch.sigmoid(seg_logits)
        bg_mask = (1.0 - targets).clamp(0.0, 1.0)
        l_fp = ((probs ** 2) * bg_mask).sum() / bg_mask.sum().clamp_min(1.0)
        total_loss = total_loss + self.lambda_fp * l_fp

        loss_dict: Dict[str, float] = {
            "loss_seg": float(l_seg.item()),
            "loss_dice": float(l_dice.item()),
            "loss_focal": float(l_focal.item()),
            "loss_fp": float(l_fp.item()),
        }

        # 2. Morphological Edge Loss (Liu et al. GSRI)
        if edge_logits is not None:
            edge_gt = _compute_gt_edge(targets)
            l_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_gt)
            total_loss = total_loss + self.edge_weight * l_edge
            loss_dict["loss_edge"] = float(l_edge.item())

        # 3. Biophysical Ecological Target (Jebur et al. HRSEI)
        if hrsei_index is not None and ref_image is not None and ref_image.shape[1] >= 8:
            ndvi_ch = ref_image[:, 6:7]
            nbr_ch = ref_image[:, 7:8]
            eco_target = ((0.5 * ndvi_ch + 0.5 * nbr_ch) * (1.0 - targets)).clamp(0.0, 1.0)
            l_eco = F.mse_loss(hrsei_index, eco_target)
            total_loss = total_loss + self.eco_weight * l_eco
            loss_dict["loss_eco"] = float(l_eco.item())

        # 4. Parcel Auxiliary Loss (Hu & Wu Venagnosis)
        if aux_seg_logits is not None:
            l_aux = self.focal_fn(aux_seg_logits, targets) + self.dice_fn(aux_seg_logits, targets)
            total_loss = total_loss + self.parcel_weight * l_aux
            loss_dict["loss_aux"] = float(l_aux.item())

        loss_dict["loss_total"] = float(total_loss.item())
        return total_loss, loss_dict


__all__ = [
    "BiGSAMV2Config",
    "BiGSAMNetV2",
    "BiGSAMLossV2",
]
