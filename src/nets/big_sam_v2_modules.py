"""
BiG-SAM v2 Building Blocks (2026):
Enhanced Graph-Augmented Foundation Architecture for Remote Sensing Canopy Disturbance.
Synthesizes the 4 state-of-the-art 2026 remote sensing graph paradigms:
  1. Biophysical Polarity Normalization (Jebur et al., J. Hazard. Mater. Adv. 2026)
  2. Synergistic Parcel Characterization & Multi-Relational Knowledge Graph (Hu & Wu, IEEE Access 2026)
  3. Dynamic Dilated KNN Max-Relative Rotation-Robust Graph (Liu et al., ACM ICMR 2026)
  4. Multi-Branch Foundation Prior Alignment & Composite Decoupled Supervision (Shao et al., GIScience 2026)
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.registry import ATTENTION_REGISTRY, BLOCK_REGISTRY
from src.nets.graph_modules import (
    BiophysicalPolarityNorm,
    DDKGGraphBlock,
    MultiRelationalGraphBuilder,
    PFGNNLayer,
    SynergisticParcelCharacterization,
)


def _conv_bn_gelu(in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


# =============================================================================
# 1. Multi-Spectral Foundation Adapter with Biophysical Polarity
# =============================================================================


@BLOCK_REGISTRY.register("multispectral_foundation_adapter_v2")
class MultiSpectralFoundationAdapterV2(nn.Module):
    """
    Multi-Spectral Foundation Adapter v2:
    Combines Jebur et al. Biophysical Polarity Normalization with a native-resolution
    channel-mixing stream and optional frozen SAM visual foundation priors.
    """

    def __init__(
        self,
        in_channels: int = 9,
        feat_dim: int = 48,
        use_sam_prior: bool = True,
        pretrained_sam_path: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.use_sam_prior = use_sam_prior
        self.has_sam_encoder = False

        # 1. Biophysical polarity calibration
        self.polarity_norm = BiophysicalPolarityNorm(in_channels)

        # 2. Synergistic Parcel Characterization (Hu & Wu 2026)
        self.spc = SynergisticParcelCharacterization(
            in_channels=in_channels * 2,  # raw + polarity
            hidden_dim=feat_dim,
        )

        # 3. Local high-resolution spatial feature stream
        self.local_res1 = _conv_bn_gelu(feat_dim, feat_dim, 3, 1)
        self.local_res2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            _conv_bn_gelu(feat_dim, feat_dim, 3, 1),
        )

        # 4. Optional frozen SAM Foundation Image Encoder
        self.sam_proj = None
        if use_sam_prior and pretrained_sam_path:
            try:
                from segment_anything import sam_model_registry
                import os
                if os.path.exists(pretrained_sam_path):
                    sam = sam_model_registry["vit_b"](checkpoint=pretrained_sam_path)
                    self.sam_encoder = sam.image_encoder
                    # Strictly freeze all SAM parameters
                    for p in self.sam_encoder.parameters():
                        p.requires_grad = False
                    self.has_sam_encoder = True
                    # Projection from SAM 256 neck to feat_dim
                    self.sam_proj = nn.Sequential(
                        nn.Conv2d(256, feat_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(feat_dim),
                        nn.GELU(),
                    )
                    # Channel mixer for SAM 3-channel input
                    self.rgb_mixer = nn.Conv2d(in_channels, 3, kernel_size=1)
            except Exception as e:
                self.has_sam_encoder = False
                warnings.warn(f"SAM encoder could not be built ({e}); using the scratch patch encoder instead.")
        if use_sam_prior and pretrained_sam_path and not self.has_sam_encoder:
            warnings.warn(
                f"SAM checkpoint not found at '{pretrained_sam_path}'. Set the SAM_CHECKPOINT environment variable "
                "or pass pretrained_sam_path; falling back to the scratch patch encoder (pretrained BiG-SAM "
                "checkpoints will NOT load)."
            )

        if not self.has_sam_encoder:
            # Standalone patch prior encoder (from scratch)
            self.scratch_patch_encoder = nn.Sequential(
                nn.Conv2d(in_channels, feat_dim, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.GELU(),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.GELU(),
            )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input Landsat tensor (B, C, H, W)
        Returns:
            f_spc: Native resolution SPC features (B, feat_dim, H, W)
            f_loc1: High-res local features (B, feat_dim, H, W)
            f_loc2: Mid-res local features (B, feat_dim, H/2, W/2)
            f_foundation: Foundation prior features (B, feat_dim, Hg, Wg) or None
        """
        B, C, H, W = x.shape
        x_pol = self.polarity_norm(x)
        x_comb = torch.cat([x, x_pol], dim=1)

        f_spc = self.spc(x_comb)
        f_loc1 = self.local_res1(f_spc)
        f_loc2 = self.local_res2(f_loc1)

        f_foundation = None
        if self.has_sam_encoder and self.sam_proj is not None:
            # Extract frozen SAM foundation priors
            with torch.no_grad():
                x_rgb = self.rgb_mixer(x)
                if (H, W) != (1024, 1024):
                    x_sam_in = F.interpolate(x_rgb, size=(1024, 1024), mode="bilinear", align_corners=False)
                else:
                    x_sam_in = x_rgb
                feat_sam = self.sam_encoder(x_sam_in)  # (B, 256, 64, 64)
            f_foundation = self.sam_proj(feat_sam)  # (B, feat_dim, 64, 64)
        elif hasattr(self, "scratch_patch_encoder"):
            f_foundation = self.scratch_patch_encoder(x)  # (B, feat_dim, H/4, W/4)

        return f_spc, f_loc1, f_loc2, f_foundation


# =============================================================================
# 2. Dual-Vein Multi-Relational & Rotation-Robust Parcel Graph Reasoner
# =============================================================================


@BLOCK_REGISTRY.register("dual_vein_graph_reasoner")
class DualVeinGraphReasoner(nn.Module):
    """
    Dual-Vein Graph Reasoning Engine:
    Combines:
      - Vein A: Knowledge-Guided Multi-Relational Graph (Hu & Wu 2026: Spatial + Semantic + Corridor)
                with Progressive Fusion Graph Network (PF-GNN) layers.
      - Vein B: Dynamic Dilated KNN Rotation-Robust Graph (Liu et al. 2026 GSRI)
                with Max-Relative Contrastive Convolution.
      - Adaptive Cross-Vein Gating: Dynamically blends relational topology and rotation-robust features.
    """

    def __init__(
        self,
        dim: int = 48,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        k_semantic: int = 4,
        k_rotation: int = 4,
        corridor_span: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.num_gnn_layers = num_gnn_layers

        # Vein A: Multi-Relational Graph Builder & PF-GNN stack
        self.graph_builder = MultiRelationalGraphBuilder(k_semantic=k_semantic, corridor_span=corridor_span)
        self.pf_gnn_layers = nn.ModuleList([
            PFGNNLayer(
                dim=dim,
                num_heads=num_heads,
                num_candidates=3,
                residual_beta=0.2,
            )
            for _ in range(num_gnn_layers)
        ])

        # Vein B: Dynamic Dilated KNN Max-Relative Graph Block (Liu et al. GSRI)
        self.ddkg_block = DDKGGraphBlock(
            channels=dim,
            k_neighbors=k_rotation,
            dilation=2,
        )

        # Cross-Vein Adaptive Fusion Gate
        self.vein_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        nodes: torch.Tensor,
        grid_hw: Tuple[int, int],
        feat_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            nodes: Parcel node embeddings (B, N, D)
            grid_hw: Grid dimensions (Hg, Wg) where N = Hg * Wg
            feat_map: Spatial parcel feature map (B, D, Hg, Wg)
        Returns:
            f_fused_nodes: Reasoning-enhanced parcel nodes (B, N, D)
            f_state: Progressive cross-layer state (B, N, D)
        """
        B, N, D = nodes.shape
        Hg, Wg = grid_hw

        # --- Vein A: Multi-Relational Topology Reasoning ---
        neighbor_idx = self.graph_builder(nodes, (Hg, Wg))
        h_curr = nodes
        f_state = nodes.clone()
        for pf_layer in self.pf_gnn_layers:
            h_curr, f_state = pf_layer(h_curr, f_state, neighbor_idx)

        # --- Vein B: Rotation-Robust Max-Relative Feature Interaction (GSRI) ---
        f_rot_map = self.ddkg_block(feat_map)  # (B, D, Hg, Wg)
        f_rot_nodes = f_rot_map.flatten(2).transpose(1, 2)  # (B, N, D)

        # --- Cross-Vein Adaptive Gating ---
        cat_veins = torch.cat([f_state, f_rot_nodes], dim=-1)
        gate = self.vein_gate(cat_veins)
        f_fused_nodes = self.out_norm(gate * f_state + (1.0 - gate) * f_rot_nodes)

        return f_fused_nodes, f_state


# =============================================================================
# 3. Bitemporal Bipartite Cross-Attention & Differential Perturbation Block
# =============================================================================


@ATTENTION_REGISTRY.register("bitemporal_bipartite_differencing_v2")
class BiTemporalBipartiteDifferencingV2(nn.Module):
    """
    Bitemporal Bipartite Cross-Attention & Differential Perturbation Block:
    - Bipartite Cross-Attention: T2 queries T1 to map seasonal/phenological illumination.
    - Differential Reasoning: Computes contrastive perturbation features
      delta_H = MLP([H2, H1_aligned, |H2 - H1_aligned|, H2 * H1_aligned]).
    - Works seamlessly in both bi-temporal mode (T1, T2) and single-date mode (T2 only).
    """

    def __init__(self, dim: int = 48, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.diff_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )
        self.diff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        nodes2: torch.Tensor,
        nodes1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            nodes2: T2 parcel nodes (B, N, D)
            nodes1: Optional T1 parcel nodes (B, N, D)
        Returns:
            delta_nodes: Disturbance-focused parcel representations (B, N, D)
        """
        if nodes1 is None:
            # Single-date mode: Non-local self-attention across parcels + self-contrastive perturbation
            B, N, D = nodes2.shape
            q2 = self.q_proj(nodes2).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k2 = self.k_proj(nodes2).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = self.v_proj(nodes2).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q2, k2.transpose(-2, -1)) / (self.head_dim**0.5)
            attn = F.softmax(scores, dim=-1)

            self_ctx = torch.matmul(attn, v2).transpose(1, 2).contiguous().view(B, N, D)
            self_ctx = self.norm(nodes2 + self.out_proj(self_ctx))

            # Quad-term self-contrastive disturbance perturbation
            diff_feat = torch.cat(
                [nodes2, self_ctx, torch.abs(nodes2 - self_ctx), nodes2 * self_ctx],
                dim=-1,
            )
            delta_nodes = self.diff_norm(nodes2 + self.diff_mlp(diff_feat))
            return delta_nodes

        B, N, D = nodes2.shape
        q2 = self.q_proj(nodes2).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_proj(nodes1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_proj(nodes1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q2, k1.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(scores, dim=-1)

        aligned_h1 = torch.matmul(attn, v1).transpose(1, 2).contiguous().view(B, N, D)
        aligned_h1 = self.norm(nodes2 + self.out_proj(aligned_h1))

        # Quad-term differential perturbation
        diff_feat = torch.cat(
            [nodes2, aligned_h1, torch.abs(nodes2 - aligned_h1), nodes2 * aligned_h1],
            dim=-1,
        )
        delta_nodes = self.diff_norm(nodes2 + self.diff_mlp(diff_feat))
        return delta_nodes


# =============================================================================
# 4. Dense Progressive Multi-Scale Boundary & Ecological Refinement Decoder
# =============================================================================


@BLOCK_REGISTRY.register("dense_boundary_refinement_decoder_v2")
class DenseBoundaryRefinementDecoderV2(nn.Module):
    """
    Dense Boundary & Ecological Refinement Decoder:
    Synthesizes multi-scale skip connections from native-resolution SPC (H x W),
    mid-resolution local features (H/2 x W/2), and the up-projected graph disturbance state.
    Generates multi-task predictions:
      1. Primary Canopy Disturbance Mask (`seg_logits`)
      2. Morphological Edge / Boundary Logits (`edge_logits`, GSRI)
      3. Biophysical Ecological Index (`hrsei_index`, HRSEI)
      4. Parcel-Level Auxiliary Disturbance Logits (`aux_seg_logits`, Venagnosis)
    """

    def __init__(self, feat_dim: int = 48, num_classes: int = 1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Parcel-level classification head (Hu & Wu 2026)
        self.parcel_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, num_classes),
        )

        # Multi-scale fusion block
        self.fusion = nn.Sequential(
            _conv_bn_gelu(feat_dim * 3, feat_dim * 2, 3, 1),
            _conv_bn_gelu(feat_dim * 2, feat_dim, 3, 1),
        )

        # Head 1: Main canopy disturbance segmentation head
        self.seg_head = nn.Sequential(
            _conv_bn_gelu(feat_dim, feat_dim // 2, 3, 1),
            nn.Conv2d(feat_dim // 2, num_classes, kernel_size=1),
        )

        # Head 2: Boundary / Edge delineation head (Liu et al. GSRI)
        self.edge_head = nn.Sequential(
            _conv_bn_gelu(feat_dim, feat_dim // 2, 3, 1),
            nn.Conv2d(feat_dim // 2, 1, kernel_size=1),
        )

        # Head 3: HRSEI Ecological Quality head (Jebur et al. HRSEI)
        self.hrsei_head = nn.Sequential(
            _conv_bn_gelu(feat_dim, 16, 3, 1),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self,
        f_loc1: torch.Tensor,
        f_loc2: torch.Tensor,
        delta_nodes: torch.Tensor,
        grid_hw: Tuple[int, int],
        orig_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_loc1: Native resolution features (B, D, H, W)
            f_loc2: Mid-resolution features (B, D, H/2, W/2)
            delta_nodes: Disturbance parcel nodes (B, N, D)
            grid_hw: (Hg, Wg)
            orig_hw: (H, W)
        Returns:
            seg_logits: Primary canopy disturbance logits (B, num_classes, H, W)
            edge_logits: Morphological boundary logits (B, 1, H, W)
            hrsei_index: Biophysical ecological index in (0, 1) (B, 1, H, W)
            aux_seg_logits: Parcel-level auxiliary disturbance logits (B, num_classes, H, W)
        """
        B, N, D = delta_nodes.shape
        Hg, Wg = grid_hw
        H, W = orig_hw

        # Parcel-level logits
        parcel_logits = self.parcel_head(delta_nodes).transpose(1, 2).view(B, self.num_classes, Hg, Wg)
        aux_seg_logits = F.interpolate(parcel_logits, size=(H, W), mode="bilinear", align_corners=False)

        # Project graph disturbance state to spatial map and upsample
        f_graph_map = delta_nodes.transpose(1, 2).view(B, D, Hg, Wg)
        f_graph_up = F.interpolate(f_graph_map, size=(H, W), mode="bilinear", align_corners=False)
        f_loc2_up = F.interpolate(f_loc2, size=(H, W), mode="bilinear", align_corners=False)

        # Multi-scale fusion
        f_fused = self.fusion(torch.cat([f_loc1, f_loc2_up, f_graph_up], dim=1))

        seg_logits = self.seg_head(f_fused)
        edge_logits = self.edge_head(f_fused)
        hrsei_raw = self.hrsei_head(f_fused)
        hrsei_index = torch.sigmoid(hrsei_raw)

        return seg_logits, edge_logits, hrsei_index, aux_seg_logits


__all__ = [
    "MultiSpectralFoundationAdapterV2",
    "DualVeinGraphReasoner",
    "BiTemporalBipartiteDifferencingV2",
    "DenseBoundaryRefinementDecoderV2",
]
