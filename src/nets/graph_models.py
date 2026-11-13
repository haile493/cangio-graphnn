"""
Graph-Based Remote Sensing Models (2026 Studies) Adapted for Multi-Spectral Earth Observation
and Mangrove Canopy Disturbance Segmentation:

1. `HRSEIGNNNet` (`hrsei_gnn`, Jebur et al., J. Hazard. Mater. Adv. 2026):
   - Positive/Negative Biophysical Indicator Normalization (Eqs. 10-11)
   - Multi-scale 4/8-Neighbor Spatial Grid GNN Message Passing (Eqs. 13-14)
   - Outputs `seg_logits` (disturbance/hazard risk prediction) and `hrsei_index` (`aux_seg_logits`, in [0, 1]).

2. `VenagnosisPFGNNNet` (`venagnosis_pfgnn`, Hu & Wu, IEEE Access 2026):
   - Stage 1: Synergistic Parcel Characterization (`SPCModule`, Eqs. 5-6, 16)
   - Stage 2: Multi-Relational Knowledge Graph Builder (`MultiRelationalGraphBuilder`: spatial, semantic, corridor veins)
   - Stage 3: Progressive Fusion Graph Network (`PFGNNLayer`: Multi-Head GATv2 + K-strategy Dynamic Gating + GRU Cross-Layer Fusion)
   - Outputs `seg_logits`, `aux_seg_logits`, and `latent_feat` (parcel embeddings for contrastive supervision).

3. `GSRIRotationRobustNet` (`gsri_net`, Liu et al., ACM ICMR 2026):
   - Multi-scale convolutional pyramid encoder
   - Stacked Graph-based Semantic Rerouting Interaction (`GSRIModule`: Dynamic Dilated KNN Max-Relative Graph
     + Dual-PE Sinusoidal/Relative Cross-Attention, Eqs. 1-5)
   - Rotation-robust multi-scale feature fusion head outputting `seg_logits` and `edge_logits`.

4. `CGMNMultiBranchNet` (`cgmn_net`, Shao et al., GIScience & Remote Sensing 2026):
   - Tri-branch architecture:
     (a) `CGMNSpatialGraphBranch`: Reciprocal distance-weighted spatial graph convolution branch
     (b) Static Branch: Pointwise nonlinear residual MLP branch
     (c) Temporal/Scale GRU Branch: Gated Recurrent Unit capturing multi-temporal or multi-scale contextual dynamics
   - Outputs `seg_logits`, `masked_logits` (under 20% random feature masking during training, Eq. 3), and `region_pred`.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.base import BaseModel, ModelConfig, ModelOutput
from src.nets.registry import MODEL_REGISTRY
from src.nets.graph_modules import (
    BiophysicalPolarityNorm,
    CGMNSpatialGraphBranch,
    GSRIModule,
    MultiRelationalGraphBuilder,
    PFGNNLayer,
    SpatialGridGNNLayer,
    SynergisticParcelCharacterization,
)


def _conv_bn_gelu(in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


# =============================================================================
# 1. HRSEI-GNN (Jebur et al., 2026)
# =============================================================================


@MODEL_REGISTRY.register("hrsei_gnn", aliases=["hrsei_gnn_net"])
class HRSEIGNNNet(BaseModel):
    """
    Hybrid Remote Sensing Socio-Ecological Index GNN (Jebur et al., 2026, Eqs. 9-14).
    Integrates biophysical polarity normalization with multi-hop 4/8-nearest-neighbor
    spatial graph message passing to jointly estimate ecological quality (HRSEI in [0, 1])
    and localized hazard/disturbance segmentation logits.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 1,
        hidden_dim: int = 32,
        num_gnn_layers: int = 3,
        include_diagonals: bool = True,
    ):
        super().__init__(
            ModelConfig(
                arch="hrsei_gnn",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                base_channels=hidden_dim,
                extra={
                    "num_gnn_layers": num_gnn_layers,
                    "include_diagonals": include_diagonals,
                },
            )
        )
        self.polarity_norm = BiophysicalPolarityNorm(in_channels)
        self.stem = _conv_bn_gelu(in_channels * 2, hidden_dim, kernel_size=3, padding=1)

        # Fine-grid GNN message passing layers (Eq. 13) with increasing dilation hops
        self.gnn_layers = nn.ModuleList([
            SpatialGridGNNLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                include_diagonals=include_diagonals,
                dilation=1 if i == 0 else (2 if i == 1 else 3),
                act="relu",
            )
            for i in range(num_gnn_layers)
        ])

        # Regional context GNN at 1/2 resolution to capture broader socio-ecological neighborhood
        self.coarse_gnn = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            SpatialGridGNNLayer(hidden_dim, hidden_dim, include_diagonals=True, dilation=1, act="relu"),
            SpatialGridGNNLayer(hidden_dim, hidden_dim, include_diagonals=True, dilation=2, act="relu"),
        )

        # HRSEI ecological quality head constrained to (0, 1) via Sigmoid (Eq. 14)
        self.hrsei_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Hazard / Canopy Disturbance risk prediction head conditioned on GNN features + HRSEI anomaly
        self.risk_head = nn.Sequential(
            _conv_bn_gelu(hidden_dim * 2 + 1, hidden_dim, kernel_size=3, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x: (B, C, H, W)
        hw = x.shape[-2:]
        x_pol = self.polarity_norm(x)
        h = self.stem(torch.cat([x, x_pol], dim=1))

        for gnn in self.gnn_layers:
            h = h + gnn(h)

        h_coarse = self.coarse_gnn(h)
        h_coarse_up = F.interpolate(h_coarse, size=hw, mode="bilinear", align_corners=False)

        # Eq. (14): HRSEI output constrained to [0, 1] via Sigmoid
        hrsei_logits = self.hrsei_head(h)
        hrsei_index = torch.sigmoid(hrsei_logits)

        # Combine fine GNN state, regional GNN state, and inverted ecological index (1 - HRSEI = degradation stress)
        fused = torch.cat([h, h_coarse_up, 1.0 - hrsei_index], dim=1)
        seg_logits = self.risk_head(fused)

        return ModelOutput(
            seg_logits=seg_logits,
            aux_seg_logits=hrsei_logits,
            hrsei_index=hrsei_index,
            latent_feat=h,
        )


# =============================================================================
# 2. Venagnosis PF-GNN (Hu & Wu, 2026)
# =============================================================================


@MODEL_REGISTRY.register("venagnosis_pfgnn", aliases=["venagnosis", "pf_gnn"])
class VenagnosisPFGNNNet(BaseModel):
    """
    Venagnosis Knowledge-Guided Graph Reasoning Framework (Hu & Wu, IEEE Access 2026):
      - Stage 1: `SynergisticParcelCharacterization` (SPC, Eqs. 5-6, 16)
      - Stage 2: `MultiRelationalGraphBuilder` (Spatial, Semantic, Corridor edge schemas)
      - Stage 3: `PFGNNLayer` stack (Multi-Head GATv2 + Hierarchical Dynamic Gating + Progressive Cross-Layer Fusion)
      - Stage 4: Boundary-preserving multi-scale skip refinement head
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 1,
        base_channels: int = 48,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        num_candidates: int = 3,
        k_semantic: int = 4,
        parcel_grid_size: int = 31,
    ):
        super().__init__(
            ModelConfig(
                arch="venagnosis_pfgnn",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                base_channels=base_channels,
                extra={
                    "num_gnn_layers": num_gnn_layers,
                    "num_heads": num_heads,
                    "num_candidates": num_candidates,
                    "k_semantic": k_semantic,
                    "parcel_grid_size": parcel_grid_size,
                },
            )
        )
        self.parcel_grid_size = parcel_grid_size

        # Stage 1: Synergistic Parcel Characterization (Eqs. 5-6, 16)
        self.spc = SynergisticParcelCharacterization(in_channels=in_channels, hidden_dim=base_channels)

        # Local high-resolution encoder branch for parcel boundary preservation
        self.local_enc1 = _conv_bn_gelu(base_channels, base_channels, 3, 1)
        self.local_enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            _conv_bn_gelu(base_channels, base_channels, 3, 1),
        )

        # Stage 2: Knowledge-Guided Multi-Relational Graph Builder
        self.graph_builder = MultiRelationalGraphBuilder(k_semantic=k_semantic, corridor_span=2)

        # Stage 3: Progressive Fusion Graph Network (PF-GNN, Eqs. 17-26)
        self.pf_gnn_layers = nn.ModuleList([
            PFGNNLayer(
                dim=base_channels,
                num_heads=num_heads,
                num_candidates=num_candidates,
                residual_beta=0.2,
            )
            for _ in range(num_gnn_layers)
        ])

        # Parcel node classification MLP head (Algorithm 1)
        self.parcel_head = nn.Sequential(
            nn.Linear(base_channels, base_channels // 2),
            nn.LayerNorm(base_channels // 2),
            nn.GELU(),
            nn.Linear(base_channels // 2, num_classes),
        )

        # Dense boundary refinement decoder combining progressive graph state F^(L) with SPC fine features
        self.decoder = nn.Sequential(
            _conv_bn_gelu(base_channels * 3, base_channels, 3, 1),
            _conv_bn_gelu(base_channels, base_channels // 2, 3, 1),
            nn.Conv2d(base_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x: (B, C, H, W)
        b, _, h, w = x.shape

        # Stage 1: SPC representation H^(0)
        h_spc = self.spc(x)  # (B, D, H, W)
        f_loc1 = self.local_enc1(h_spc)  # (B, D, H, W)
        f_loc2 = self.local_enc2(f_loc1)  # (B, D, H/2, W/2)

        # Downsample into parcel nodes (N = Hg * Wg)
        hg = min(h, self.parcel_grid_size)
        wg = min(w, self.parcel_grid_size)
        h_parcel = F.adaptive_avg_pool2d(f_loc2, (hg, wg))  # (B, D, Hg, Wg)
        d = h_parcel.shape[1]
        h_nodes = h_parcel.flatten(2).transpose(1, 2)  # (B, N, D)

        # Stage 2: Build Multi-Relational Graph (spatial + semantic + corridor veins)
        neighbor_idx = self.graph_builder(h_nodes, (hg, wg))  # (B, N, K_total)

        # Stage 3: PF-GNN Reasoning with Progressive Cross-Layer State F^(l)
        h_curr = h_nodes
        f_state = h_nodes.clone()
        for pf_layer in self.pf_gnn_layers:
            h_curr, f_state = pf_layer(h_curr, f_state, neighbor_idx)

        # Parcel-level auxiliary prediction from final progressive fusion state F^(L)
        parcel_logits = self.parcel_head(f_state).transpose(1, 2).view(b, -1, hg, wg)
        aux_seg_logits = F.interpolate(parcel_logits, size=(h, w), mode="bilinear", align_corners=False)

        # Project progressive graph state back to spatial map and fuse with fine SPC features
        f_graph_map = f_state.transpose(1, 2).view(b, d, hg, wg)
        f_graph_up = F.interpolate(f_graph_map, size=(h, w), mode="bilinear", align_corners=False)
        f_loc2_up = F.interpolate(f_loc2, size=(h, w), mode="bilinear", align_corners=False)

        seg_logits = self.decoder(torch.cat([f_loc1, f_loc2_up, f_graph_up], dim=1))

        return ModelOutput(
            seg_logits=seg_logits,
            aux_seg_logits=aux_seg_logits,
            latent_feat=f_state,  # (B, N, D) for contrastive supervision
        )


# =============================================================================
# 3. GSRI Rotation-Robust Semantic Modeling Network (Liu et al., 2026)
# =============================================================================


@MODEL_REGISTRY.register("gsri_net", aliases=["gsri_rotation_net", "gsri"])
class GSRIRotationRobustNet(BaseModel):
    """
    Graph-Based Rotation-Robust Semantic Modeling Network (Liu et al., ACM ICMR 2026):
      - Extracts multi-scale pyramidal features (F_1, F_2, F_3)
      - Applies stacked `GSRIModule` blocks (Dynamic Dilated KNN Max-Relative Graph Convolution
        + Dual-PE Sinusoidal/Relative Cross-Attention, Eqs. 1-5) across pyramid levels
      - Aligns rotation-invariant semantics with high-resolution spatial details for oriented
        canopy disturbance and blowdown corridor delineation.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 1,
        base_channels: int = 48,
        num_heads: int = 4,
        k_neighbors: int = 6,
        dilation: int = 2,
    ):
        super().__init__(
            ModelConfig(
                arch="gsri_net",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                base_channels=base_channels,
                extra={
                    "num_heads": num_heads,
                    "k_neighbors": k_neighbors,
                    "dilation": dilation,
                },
            )
        )
        # Multi-scale FPN-style backbone
        self.enc1 = nn.Sequential(
            _conv_bn_gelu(in_channels, base_channels, 3, 1),
            _conv_bn_gelu(base_channels, base_channels, 3, 1),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            _conv_bn_gelu(base_channels, base_channels * 2, 3, 1),
            _conv_bn_gelu(base_channels * 2, base_channels, 1, 0),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            _conv_bn_gelu(base_channels, base_channels * 2, 3, 1),
            _conv_bn_gelu(base_channels * 2, base_channels, 1, 0),
        )

        # Stacked GSRI modules across multi-scale hierarchy (Figure 3 of Liu et al., 2026)
        self.gsri_high = GSRIModule(base_channels, num_heads=num_heads, k_neighbors=k_neighbors, dilation=dilation)
        self.gsri_mid = GSRIModule(base_channels, num_heads=num_heads, k_neighbors=k_neighbors, dilation=dilation)
        self.gsri_low = GSRIModule(base_channels, num_heads=num_heads, k_neighbors=k_neighbors, dilation=1)

        # Orientation-aware boundary auxiliary head
        self.edge_head = nn.Sequential(
            _conv_bn_gelu(base_channels, base_channels // 2, 3, 1),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
        )

        # Multi-scale cascading fusion head
        self.fusion_head = nn.Sequential(
            _conv_bn_gelu(base_channels * 3, base_channels, 3, 1),
            _conv_bn_gelu(base_channels, base_channels // 2, 3, 1),
            nn.Conv2d(base_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x: (B, C, H, W)
        hw = x.shape[-2:]
        f1 = self.enc1(x)       # (B, D, H, W)
        f2 = self.enc2(f1)      # (B, D, H/2, W/2)
        f3 = self.enc3(f2)      # (B, D, H/4, W/4)

        # Multi-scale cascading GSRI (high-level rotational semantics rerouted to lower scales)
        t3 = self.gsri_high(f3, f3)
        t2 = self.gsri_mid(f2, t3)
        t1 = self.gsri_low(f1, t2)

        t3_up = F.interpolate(t3, size=hw, mode="bilinear", align_corners=False)
        t2_up = F.interpolate(t2, size=hw, mode="bilinear", align_corners=False)

        edge_logits = self.edge_head(t1)
        seg_logits = self.fusion_head(torch.cat([t1, t2_up, t3_up], dim=1))

        return ModelOutput(
            seg_logits=seg_logits,
            edge_logits=edge_logits,
            latent_feat=t1,
        )


# =============================================================================
# 4. CGMN Multi-Branch Network (Shao et al., 2026)
# =============================================================================


@MODEL_REGISTRY.register("cgmn_net", aliases=["cgmn", "cgmn_multibranch"])
class CGMNMultiBranchNet(BaseModel):
    """
    Carbon/Canopy Graph Multi-Branch Network (Shao et al., GIScience & Remote Sensing 2026):
      - Branch 1 (Graph Branch): `CGMNSpatialGraphBranch` with reciprocal spatial distance edge weights
        (w_ij = 1 / d_ij) and multi-layer graph convolutions.
      - Branch 2 (Static Branch): Pointwise nonlinear MLP residual layers modeling multi-source
        spectral & biophysical feature relationships.
      - Branch 3 (Temporal / Multi-Scale GRU Branch): Gated Recurrent Unit (`nn.GRU`) modeling
        sequential dynamics across either temporal acquisitions (B, T, C, H, W) or multi-scale
        contextual hierarchy (1x, 2x, 4x receptive fields) when a single timestamp (B, C, H, W) is given.
      - Supports random node feature masking (mask_ratio = 0.2, Eq. 3) during training to compute
        the Mask Prediction Loss.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 1,
        branch_dim: int = 32,
        num_gnn_layers: int = 2,
        grid_size: int = 16,
        mask_ratio: float = 0.2,
    ):
        super().__init__(
            ModelConfig(
                arch="cgmn_net",
                dims=2,
                in_channels=in_channels,
                num_seg_classes=num_classes,
                base_channels=branch_dim,
                extra={
                    "num_gnn_layers": num_gnn_layers,
                    "grid_size": grid_size,
                    "mask_ratio": mask_ratio,
                },
            )
        )
        self.in_channels = in_channels
        self.branch_dim = branch_dim
        self.mask_ratio = mask_ratio

        self.input_proj = _conv_bn_gelu(in_channels, branch_dim, kernel_size=3, padding=1)

        # Branch 1: Distance-weighted Spatial Graph Branch
        self.graph_branch = CGMNSpatialGraphBranch(
            channels=branch_dim,
            num_layers=num_gnn_layers,
            grid_size=grid_size,
        )

        # Branch 2: Static Nonlinear Feature Branch (Pointwise MLP with residual connection)
        self.static_branch = nn.Sequential(
            nn.Conv2d(branch_dim, branch_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim * 2),
            nn.GELU(),
            nn.Conv2d(branch_dim * 2, branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU(),
        )

        # Branch 3: Temporal / Multi-Scale Recurrent GRU Branch
        self.gru = nn.GRU(
            input_size=branch_dim,
            hidden_size=branch_dim,
            num_layers=1,
            batch_first=True,
        )
        self.scale_pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.scale_pool4 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # Tri-branch concatenation & prediction head (Section 3.1)
        self.fusion_head = nn.Sequential(
            _conv_bn_gelu(branch_dim * 3, branch_dim * 2, kernel_size=3, padding=1),
            _conv_bn_gelu(branch_dim * 2, branch_dim, kernel_size=3, padding=1),
            nn.Conv2d(branch_dim, num_classes, kernel_size=1),
        )

    def _run_gru_branch(self, feat_seq: List[torch.Tensor]) -> torch.Tensor:
        # feat_seq: list of T tensors each of shape (B, D, H, W)
        b, d, h, w = feat_seq[0].shape
        t_len = len(feat_seq)
        # Stack along time/scale: (B, T, D, H, W) -> (B*H*W, T, D)
        seq = torch.stack(feat_seq, dim=1).permute(0, 3, 4, 1, 2).reshape(b * h * w, t_len, d)
        gru_out, _ = self.gru(seq)
        last_state = gru_out[:, -1, :]  # (B*H*W, D)
        return last_state.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

    def _forward_single_stream(self, x_4d: torch.Tensor, temporal_seq: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        f0 = self.input_proj(x_4d)
        b_graph = self.graph_branch(f0)
        b_static = f0 + self.static_branch(f0)

        if temporal_seq is not None:
            b_temp = self._run_gru_branch(temporal_seq)
        else:
            # Construct 3-step multi-scale historical/contextual sequence (coarse -> medium -> fine)
            s1 = self.scale_pool4(f0)
            s2 = self.scale_pool2(f0)
            s3 = f0
            b_temp = self._run_gru_branch([s1, s2, s3])

        fused = torch.cat([b_graph, b_static, b_temp], dim=1)
        return self.fusion_head(fused)

    def forward(self, x: torch.Tensor, compute_masked_branch: Optional[bool] = None) -> ModelOutput:
        """
        Args:
            x: Either (B, C, H, W) single-date multi-spectral tensor
               or (B, T, C, H, W) multi-temporal sequence.
            compute_masked_branch: If True (or self.training), also runs a 20% randomly masked
               forward pass for `CGMNCompositeLoss` Eq. (3).
        """
        if compute_masked_branch is None:
            compute_masked_branch = self.training

        if x.ndim == 5:
            # (B, T, C, H, W)
            b, t_len, c, h, w = x.shape
            temporal_proj = [self.input_proj(x[:, t]) for t in range(t_len)]
            x_curr = x[:, -1]
            seg_logits = self._forward_single_stream(x_curr, temporal_seq=temporal_proj)
        else:
            x_curr = x
            seg_logits = self._forward_single_stream(x_curr, temporal_seq=None)

        masked_logits = None
        mask_indicator = None
        if compute_masked_branch and self.mask_ratio > 0.0:
            b, c, h, w = x_curr.shape
            # Randomly mask 20% of spatial nodes by setting their input features to zero (Eq. 3)
            mask_indicator = (torch.rand(b, 1, h, w, device=x_curr.device) < self.mask_ratio).float()
            x_masked = x_curr * (1.0 - mask_indicator)
            masked_logits = self._forward_single_stream(x_masked, temporal_seq=None)

        return ModelOutput(
            seg_logits=seg_logits,
            masked_logits=masked_logits,
            mask_indicator=mask_indicator,
        )


__all__ = [
    "HRSEIGNNNet",
    "VenagnosisPFGNNNet",
    "GSRIRotationRobustNet",
    "CGMNMultiBranchNet",
]
