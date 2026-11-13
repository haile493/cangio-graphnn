"""
BiG-SAM Building Blocks (2026):
Novel Bi-temporal Graph Neural Network with Segment Anything (SAM) Foundation Priors
for Remote Sensing Canopy Disturbance & Change Detection.

Modules:
1. `SpecSAMAdapter`: Multi-Spectral Spectral Adapter mapping C-channel Landsat/Sentinel
   imagery (RGB + NIR + SWIRs + NDVI + NBR + NDWI) to SAM-compatible 3-channel input space.
2. `SAMTokenLearner`: SAM-guided spatial soft super-tokenization extracting N topological
   semantic nodes and coordinate centroids from dense ViT features.
3. `BiTemporalBipartiteGraphBlock`: Dual-phase GNN performing intra-temporal spatial GATv2
   message passing and inter-temporal differential bipartite cross-attention.
4. `PromptGuidedSAMDecoder`: Generates top-K prompt points from node change likelihoods
   and applies SAM prompt encoder + mask decoder for sub-pixel boundary delineation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.registry import ATTENTION_REGISTRY, BLOCK_REGISTRY


# =============================================================================
# 1. Multi-Spectral SAM Adapter
# =============================================================================


@BLOCK_REGISTRY.register("spec_sam_adapter")
class SpecSAMAdapter(nn.Module):
    """
    Multi-Spectral Channel Adapter (MCA).
    Maps C input bands (e.g., 9 bands: blue, green, red, nir, swir1, swir2, ndvi, nbr, ndwi)
    to SAM-compatible 3-channel feature space while preserving high-frequency radiometric contrasts.
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 3, target_size: int = 1024):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size

        # Channel mixing projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=1, bias=True),
        )

        # Residual RGB pass-through (assuming bands 0, 1, 2 or normalized red/green/blue)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            x_sam: SAM-ready tensor (B, 3, target_size, target_size)
            orig_hw: Original spatial size (H, W)
        """
        orig_hw = (x.shape[2], x.shape[3])
        # Project multispectral channels
        out = self.proj(x)
        if self.in_channels >= 3:
            rgb_direct = x[:, :3]
            out = self.alpha * rgb_direct + (1.0 - self.alpha) * out

        # Scale to SAM ViT target resolution
        if (out.shape[2], out.shape[3]) != (self.target_size, self.target_size):
            out = F.interpolate(out, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)

        return out, orig_hw


# =============================================================================
# 2. SAM-Guided Semantic Super-Tokenization
# =============================================================================


@BLOCK_REGISTRY.register("sam_token_learner")
class SAMTokenLearner(nn.Module):
    """
    Adaptive Semantic Token Learner.
    Compresses dense SAM ViT feature maps (B, D, H', W') into N semantic entity nodes
    and extracts continuous spatial centroids (px, py) in [0, target_size] coordinates.
    """

    def __init__(self, in_dim: int = 256, num_nodes: int = 64, target_size: int = 1024):
        super().__init__()
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.target_size = target_size

        self.attn_net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, num_nodes, kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: Dense SAM image features (B, D, H', W')
        Returns:
            nodes: (B, num_nodes, D)
            centroids: (B, num_nodes, 2) normalized to [0, target_size]
            attn_weights: (B, num_nodes, H'*W') for transpose reconstruction
        """
        B, D, H, W = feat.shape
        # Spatial attention logits: (B, num_nodes, H, W)
        attn_logits = self.attn_net(feat)
        attn_weights = F.softmax(attn_logits.view(B, self.num_nodes, H * W), dim=-1)  # (B, N, HW)

        # Flat feature map: (B, HW, D)
        feat_flat = feat.view(B, D, H * W).permute(0, 2, 1)

        # Node features: (B, N, D)
        nodes = torch.bmm(attn_weights, feat_flat)

        # Coordinate grid: (H, W, 2) in [0, target_size]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, float(self.target_size), H, device=feat.device),
            torch.linspace(0.0, float(self.target_size), W, device=feat.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, H * W, 2)  # (1, HW, 2)
        grid = grid.expand(B, -1, -1)

        # Centroids: (B, N, 2)
        centroids = torch.bmm(attn_weights, grid)

        return nodes, centroids, attn_weights


# =============================================================================
# 3. Bi-Temporal Spatio-Temporal Bipartite Graph Block
# =============================================================================


@ATTENTION_REGISTRY.register("bitemporal_bipartite_graph")
class BiTemporalBipartiteGraphBlock(nn.Module):
    """
    Dual-Phase Bitemporal Graph Neural Network:
    - Intra-temporal Phase: GATv2-style spatial message passing among spatial neighbors.
    - Inter-temporal Phase: Bipartite cross-temporal attention aligning T1 and T2 nodes,
      followed by differential topological reasoning to produce pure disturbance features.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, k_neighbors: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k_neighbors = k_neighbors

        # Intra-temporal GATv2 projections
        self.q_intra = nn.Linear(dim, dim, bias=False)
        self.k_intra = nn.Linear(dim, dim, bias=False)
        self.v_intra = nn.Linear(dim, dim, bias=False)
        self.out_intra = nn.Linear(dim, dim)
        self.norm_intra = nn.LayerNorm(dim)

        # Inter-temporal Bipartite Cross-Attention
        self.q_inter = nn.Linear(dim, dim, bias=False)
        self.k_inter = nn.Linear(dim, dim, bias=False)
        self.v_inter = nn.Linear(dim, dim, bias=False)
        self.out_inter = nn.Linear(dim, dim)
        self.norm_inter = nn.LayerNorm(dim)

        # Differential Reasoning MLP
        self.diff_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
        self.norm_diff = nn.LayerNorm(dim)

        # Node change classifier head (outputs likelihood in [0, 1])
        self.node_classifier = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _intra_temporal_gnn(self, nodes: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Message passing over k-NN spatial graph with GATv2 attention.
        """
        B, N, D = nodes.shape
        # Spatial distance matrix: (B, N, N)
        dist = torch.cdist(centroids, centroids)
        # Top-k mask
        _, topk_idx = torch.topk(dist, k=min(self.k_neighbors, N), largest=False, dim=-1)
        mask = torch.zeros(B, N, N, device=nodes.device, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)

        # Multi-head attention
        q = self.q_intra(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = self.k_intra(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_intra(nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, H, N, N)
        mask_exp = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(~mask_exp, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.norm_intra(nodes + self.out_intra(out))
        return out

    def forward(
        self,
        nodes1: torch.Tensor,
        centroids1: torch.Tensor,
        nodes2: torch.Tensor,
        centroids2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            nodes1: T1 nodes (B, N, D)
            centroids1: T1 centroids (B, N, 2)
            nodes2: T2 nodes (B, N, D)
            centroids2: T2 centroids (B, N, 2)
        Returns:
            delta_nodes: Disturbance representation at T2 nodes (B, N, D)
            node_logits: Node-level disturbance logits (B, N, 1)
            aligned_nodes1: T1 nodes mapped to T2 coordinates (B, N, D)
        """
        B, N, D = nodes1.shape

        # 1. Intra-temporal context modeling
        h1 = self._intra_temporal_gnn(nodes1, centroids1)
        h2 = self._intra_temporal_gnn(nodes2, centroids2)

        # 2. Inter-temporal Bipartite Cross-Attention (T2 queries T1)
        q2 = self.q_inter(h2).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_inter(h1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_inter(h1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        cross_scores = torch.matmul(q2, k1.transpose(-2, -1)) / (self.head_dim**0.5)
        cross_attn = F.softmax(cross_scores, dim=-1)

        aligned_h1 = torch.matmul(cross_attn, v1).transpose(1, 2).contiguous().view(B, N, D)
        aligned_h1 = self.norm_inter(h2 + self.out_inter(aligned_h1))

        # 3. Differential Reasoning: [h2, aligned_h1, |h2 - aligned_h1|, h2 * aligned_h1]
        diff_feat = torch.cat([h2, aligned_h1, torch.abs(h2 - aligned_h1), h2 * aligned_h1], dim=-1)
        delta_nodes = self.norm_diff(h2 + self.diff_mlp(diff_feat))

        # 4. Node disturbance classification
        node_logits = self.node_classifier(delta_nodes)

        return delta_nodes, node_logits, aligned_h1


# =============================================================================
# 4. SAM Prompt-Guided Boundary Decoder
# =============================================================================


@BLOCK_REGISTRY.register("prompt_guided_sam_decoder")
class PromptGuidedSAMDecoder(nn.Module):
    """
    Synthesizes difference features, extracts top-K high-confidence disturbance centroids
    as positive point prompts, and drives SAM's Mask Decoder to generate sharp change masks.
    """

    def __init__(
        self,
        dim: int = 256,
        top_k_prompts: int = 5,
        target_size: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.top_k_prompts = top_k_prompts
        self.target_size = target_size

        # Difference feature fusion block
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Auxiliary direct convolutional segmentation head (for fast gradient flow & fallback)
        self.aux_head = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        delta_nodes: torch.Tensor,
        node_logits: torch.Tensor,
        centroids2: torch.Tensor,
        attn_weights2: torch.Tensor,
        sam_prompt_encoder: Optional[nn.Module] = None,
        sam_mask_decoder: Optional[nn.Module] = None,
        orig_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f1: SAM features T1 (B, D, H', W')
            f2: SAM features T2 (B, D, H', W')
            delta_nodes: Disturbance node representations (B, N, D)
            node_logits: Node disturbance likelihoods (B, N, 1)
            centroids2: Spatial coordinates of T2 nodes (B, N, 2) in [0, target_size]
            attn_weights2: Token learner attention weights (B, N, H'*W')
            sam_prompt_encoder: Pretrained/scratch SAM PromptEncoder
            sam_mask_decoder: Pretrained/scratch SAM MaskDecoder
            orig_hw: Target output spatial resolution (H, W)
        Returns:
            seg_logits: Final change detection logits (B, 1, H, W)
            aux_logits: Auxiliary pixel-wise logits (B, 1, H, W)
            prompt_points: Harvested coordinate prompts (B, K, 2)
        """
        B, D, H_prime, W_prime = f2.shape

        # 1. Project delta_nodes back to dense 2D feature grid: (B, D, H', W')
        # attn_weights2 is (B, N, HW). Transpose is (B, HW, N).
        f_diff_flat = torch.bmm(attn_weights2.permute(0, 2, 1), delta_nodes)  # (B, HW, D)
        f_diff = f_diff_flat.permute(0, 2, 1).contiguous().view(B, D, H_prime, W_prime)

        # 2. Multi-temporal difference fusion
        f_abs = torch.abs(f2 - f1)
        f_cat = torch.cat([f2, f1, f_abs, f_diff], dim=1)  # (B, 4D, H', W')
        f_fused = self.fusion(f_cat)  # (B, D, H', W')

        # Direct aux logits
        aux_raw = self.aux_head(f_fused)  # (B, 1, H', W')

        # 3. Dynamic top-K coordinate prompt harvesting
        node_scores = torch.sigmoid(node_logits.squeeze(-1))  # (B, N)
        k = min(self.top_k_prompts, node_scores.shape[1])
        top_scores, top_indices = torch.topk(node_scores, k=k, dim=1)  # (B, K)

        # Gather centroids for top-K nodes
        batch_idx = torch.arange(B, device=f2.device).unsqueeze(1).expand(-1, k)
        prompt_points = centroids2[batch_idx, top_indices]  # (B, K, 2)
        prompt_labels = torch.ones((B, k), dtype=torch.int, device=f2.device)

        # 4. SAM Mask Decoder Pass
        if sam_prompt_encoder is not None and sam_mask_decoder is not None:
            sparse_emb, dense_emb = sam_prompt_encoder(
                points=(prompt_points, prompt_labels),
                boxes=None,
                masks=None,
            )
            low_res_masks, _ = sam_mask_decoder(
                image_embeddings=f_fused,
                image_pe=sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )  # (B, 1, 256, 256)
            mask_out = low_res_masks
        else:
            mask_out = aux_raw

        # 5. Interpolate back to original input resolution
        target_hw = orig_hw if orig_hw is not None else (H_prime, W_prime)
        seg_logits = F.interpolate(mask_out, size=target_hw, mode="bilinear", align_corners=False)
        aux_logits = F.interpolate(aux_raw, size=target_hw, mode="bilinear", align_corners=False)

        return seg_logits, aux_logits, prompt_points


__all__ = [
    "SpecSAMAdapter",
    "SAMTokenLearner",
    "BiTemporalBipartiteGraphBlock",
    "PromptGuidedSAMDecoder",
]
