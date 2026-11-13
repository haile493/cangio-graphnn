"""
BiG-SAM: Bi-temporal Graph Neural Network with Segment Anything (SAM) Foundation Priors
for Earth Observation Canopy Disturbance Mapping and Remote Sensing Change Detection.

Registered as `big_sam` in `MODEL_REGISTRY`.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.base import BaseModel, ModelConfig, ModelOutput
from src.nets.registry import LOSS_REGISTRY, MODEL_REGISTRY
from src.nets.losses import BinaryDiceLoss, FocalLoss
from src.nets.big_sam_modules import (
    BiTemporalBipartiteGraphBlock,
    PromptGuidedSAMDecoder,
    SAMTokenLearner,
    SpecSAMAdapter,
)

DEFAULT_SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "results/models/sam_vit_b_01ec64.pth")


@dataclass
class BiGSAMConfig(ModelConfig):
    """Configuration for BiG-SAM Model."""
    in_channels: int = 9
    num_classes: int = 1
    sam_type: str = "vit_b"
    pretrained: bool = True
    pretrained_sam_path: str = DEFAULT_SAM_CHECKPOINT
    freeze_sam_encoder: bool = False
    target_size: int = 1024
    num_nodes: int = 64
    k_neighbors: int = 8
    num_heads: int = 4
    top_k_prompts: int = 5
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    node_loss_weight: float = 0.3


@MODEL_REGISTRY.register("big_sam", aliases=["big_sam_cd", "sam_bitemporal_gnn"])
class BiGSAMNet(BaseModel):
    """
    BiG-SAM: Bi-temporal Graph Segment Anything Network.
    Combines:
      1. Multi-Spectral SAM Adapter (`SpecSAMAdapter`)
      2. SAM Siamese ViT Image Encoder (Scratch or Pretrained)
      3. SAM-guided Semantic Super-Tokenization (`SAMTokenLearner`)
      4. Spatio-Temporal Bipartite GNN Reasoning (`BiTemporalBipartiteGraphBlock`)
      5. Prompt-Guided Boundary Refinement (`PromptGuidedSAMDecoder`)
    """

    config_class = BiGSAMConfig

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 1,
        sam_type: str = "vit_b",
        pretrained: bool = True,
        pretrained_sam_path: str = DEFAULT_SAM_CHECKPOINT,
        freeze_sam_encoder: bool = False,
        target_size: int = 1024,
        num_nodes: int = 64,
        k_neighbors: int = 8,
        num_heads: int = 4,
        top_k_prompts: int = 5,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.pretrained_sam_path = pretrained_sam_path
        self.freeze_sam_encoder = freeze_sam_encoder
        self.target_size = target_size
        self.num_nodes = num_nodes

        # 1. Multi-Spectral Adapter
        self.adapter = SpecSAMAdapter(
            in_channels=in_channels,
            out_channels=3,
            target_size=target_size,
        )

        # 2. SAM ViT Foundation Model (from segment_anything)
        from segment_anything import sam_model_registry

        ckpt_to_load = pretrained_sam_path if (pretrained and os.path.exists(pretrained_sam_path)) else None
        self.sam = sam_model_registry[sam_type](checkpoint=ckpt_to_load)

        if self.freeze_sam_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False

        sam_dim = 256  # ViT-B neck outputs 256

        # 3. SAM-Guided Semantic Token Learner
        self.token_learner = SAMTokenLearner(
            in_dim=sam_dim,
            num_nodes=num_nodes,
            target_size=target_size,
        )

        # 4. Spatio-Temporal Bipartite Graph Block
        self.graph_reasoner = BiTemporalBipartiteGraphBlock(
            dim=sam_dim,
            num_heads=num_heads,
            k_neighbors=k_neighbors,
        )

        # 5. Prompt-Guided Boundary Decoder
        self.decoder = PromptGuidedSAMDecoder(
            dim=sam_dim,
            top_k_prompts=top_k_prompts,
            target_size=target_size,
        )

        self.loss_fn = BiGSAMLoss()

    def extract_sam_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Adapts multispectral image and passes through SAM image encoder."""
        x_sam, orig_hw = self.adapter(x)
        # Check if pos_embed matches target patch grid
        target_hw = (x_sam.shape[2] // 16, x_sam.shape[3] // 16)
        encoder = self.sam.image_encoder
        if (
            hasattr(encoder, "pos_embed")
            and encoder.pos_embed is not None
            and (encoder.pos_embed.shape[1], encoder.pos_embed.shape[2]) != target_hw
        ):
            orig_pe = encoder.pos_embed
            interp_pe = nn.Parameter(
                F.interpolate(
                    orig_pe.permute(0, 3, 1, 2),
                    size=target_hw,
                    mode="bicubic",
                    align_corners=False,
                ).permute(0, 2, 3, 1),
                requires_grad=orig_pe.requires_grad,
            )
            encoder.pos_embed = interp_pe
            try:
                feat = encoder(x_sam)
            finally:
                encoder.pos_embed = orig_pe
        else:
            feat = encoder(x_sam)
        return feat, orig_hw

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """
        Forward pass for bitemporal change detection.
        Args:
            x1: T1 image tensor (B, C, H, W) OR concatenated / 5D tensor (B, 2, C, H, W)
            x2: T2 image tensor (B, C, H, W)
            targets: Optional ground-truth change mask (B, 1, H, W)
        """
        if x2 is None:
            if x1.dim() == 5:
                x2 = x1[:, 1]
                x1 = x1[:, 0]
            elif x1.shape[1] == 2 * self.in_channels:
                x2 = x1[:, self.in_channels :]
                x1 = x1[:, : self.in_channels]
            else:
                raise ValueError("Expected bi-temporal inputs: provide (x1, x2) or 5D (B, 2, C, H, W)")

        # 1. Siamese feature extraction via SAM
        feat1, _ = self.extract_sam_features(x1)
        feat2, orig_hw = self.extract_sam_features(x2)

        # 2. SAM-guided token learning
        nodes1, centroids1, attn_weights1 = self.token_learner(feat1)
        nodes2, centroids2, attn_weights2 = self.token_learner(feat2)

        # 3. Spatio-Temporal Bipartite GNN Reasoning
        delta_nodes, node_logits, aligned_nodes1 = self.graph_reasoner(
            nodes1=nodes1,
            centroids1=centroids1,
            nodes2=nodes2,
            centroids2=centroids2,
        )

        # 4. Prompt-guided boundary decoding
        seg_logits, aux_logits, prompt_points = self.decoder(
            f1=feat1,
            f2=feat2,
            delta_nodes=delta_nodes,
            node_logits=node_logits,
            centroids2=centroids2,
            attn_weights2=attn_weights2,
            sam_prompt_encoder=self.sam.prompt_encoder,
            sam_mask_decoder=self.sam.mask_decoder,
            orig_hw=orig_hw,
        )

        loss = None
        if targets is not None:
            loss, _ = self.loss_fn(
                seg_logits=seg_logits,
                aux_logits=aux_logits,
                node_logits=node_logits,
                attn_weights2=attn_weights2,
                targets=targets,
            )

        return ModelOutput(
            seg_logits=seg_logits,
            aux_seg_logits=aux_logits,
            node_logits=node_logits,
            prompt_points=prompt_points,
            loss=loss,
        )


# =============================================================================
# Compound Loss for BiG-SAM
# =============================================================================


class BinaryFocalLoss(nn.Module):
    """Numerically stable Binary Focal Loss for binary change detection."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        focal_loss = alpha_factor * modulating_factor * bce
        return focal_loss.mean()


@LOSS_REGISTRY.register("big_sam_loss")
class BiGSAMLoss(nn.Module):
    """
    Compound Loss for BiG-SAM:
      1. Pixel Focal Loss on final refined change logits
      2. Pixel Binary Dice Loss on final logits
      3. Auxiliary Focal + Dice Loss on direct convolutional logits
      4. Node-level BCE loss supervised by pooled ground-truth disturbance density
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        aux_weight: float = 0.3,
        node_weight: float = 0.3,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.node_weight = node_weight

        self.focal_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)
        self.dice_fn = BinaryDiceLoss()

    def forward(
        self,
        outputs: Optional[Union[ModelOutput, Dict[str, torch.Tensor]]] = None,
        targets: Optional[torch.Tensor] = None,
        seg_logits: Optional[torch.Tensor] = None,
        aux_logits: Optional[torch.Tensor] = None,
        node_logits: Optional[torch.Tensor] = None,
        attn_weights2: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if outputs is not None:
            seg_logits = outputs.get("seg_logits") if isinstance(outputs, dict) else outputs.seg_logits
            aux_logits = outputs.get("aux_seg_logits") if isinstance(outputs, dict) else getattr(outputs, "aux_seg_logits", None)
            node_logits = outputs.get("node_logits") if isinstance(outputs, dict) else getattr(outputs, "node_logits", None)

        if targets is None:
            raise ValueError("Target masks must be provided for BiGSAMLoss.")

        # 1. Main pixel loss
        l_focal = self.focal_fn(seg_logits, targets)
        l_dice = self.dice_fn(seg_logits, targets)
        l_pixel = self.focal_weight * l_focal + self.dice_weight * l_dice

        total_loss = l_pixel
        loss_dict: Dict[str, float] = {
            "loss_pixel": float(l_pixel.item()),
            "loss_dice": float(l_dice.item()),
            "loss_focal": float(l_focal.item()),
        }

        # 2. Auxiliary pixel loss
        if aux_logits is not None:
            l_aux = 0.5 * self.focal_fn(aux_logits, targets) + 0.5 * self.dice_fn(aux_logits, targets)
            total_loss = total_loss + self.aux_weight * l_aux
            loss_dict["loss_aux"] = float(l_aux.item())

        # 3. Supervised Node Disturbance Loss
        if node_logits is not None and attn_weights2 is not None:
            # Downsample target to match HW of attn_weights2
            B, N, HW = attn_weights2.shape
            H_w = int(HW**0.5)
            targets_small = F.interpolate(targets, size=(H_w, H_w), mode="area")
            targets_flat = targets_small.view(B, 1, HW)  # (B, 1, HW)
            # Node target density: (B, N, 1)
            node_target = torch.bmm(attn_weights2, targets_flat.transpose(1, 2))  # (B, N, 1)
            node_target = (node_target > 0.05).float()
            l_node = F.binary_cross_entropy_with_logits(node_logits, node_target)
            total_loss = total_loss + self.node_weight * l_node
            loss_dict["loss_node"] = float(l_node.item())

        loss_dict["loss_total"] = float(total_loss.item())
        return total_loss, loss_dict


__all__ = [
    "BiGSAMConfig",
    "BiGSAMNet",
    "BiGSAMLoss",
]
