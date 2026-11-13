"""
Reusable Graph-Based Remote Sensing Building Blocks (2026 Studies):

1. `BiophysicalPolarityNorm` & `SpatialGridGNNLayer` (Jebur et al., J. Hazard. Mater. Adv. 2026):
   - Positive/negative indicator normalization (Eqs. 10-11) and 4/8-neighbor distance-weighted
     spatial graph message passing (Eq. 13).
2. `SynergisticParcelCharacterization`, `MultiRelationalGraphBuilder`, & `PFGNNLayer` (Hu & Wu, IEEE Access 2026):
   - Multi-modal spectral + morphological gradient characterization with LayerNorm MLPs and importance gating (Eqs. 5-6, 16).
   - Multi-relational graph topology (spatial adjacency, semantic context cosine KNN, directional corridor links).
   - Multi-head GATv2 aggregation (Eqs. 17-18) + K-candidate Hierarchical Dynamic Gating (Eqs. 19-22)
     + Progressive Cross-Layer GRU-style State Fusion (Eq. 26).
3. `DDKGGraphBlock` & `GSRIModule` (Liu et al., ACM ICMR 2026):
   - Dynamic Dilated KNN-Graph with max-relative graph convolution v_i' = v_i + max_{j in N(i)} phi(W(v_j - v_i))
     and Relative Positional Encoding (Eqs. 1-2).
   - Dual-PE (Sinusoidal + Relative) Supervised Cross-Attention for cross-scale rotation-invariant feature alignment (Eqs. 3-5).
4. `CGMNSpatialGraphBranch` (Shao et al., GIScience & Remote Sensing 2026):
   - Reciprocal spatial distance-weighted graph convolution branch (w_ij = 1 / d_ij) combined with spectral affinity.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.registry import ATTENTION_REGISTRY, BLOCK_REGISTRY


# =============================================================================
# 1. Jebur et al. (2026): HRSEI Biophysical Polarity & Spatial Grid GNN
# =============================================================================


@BLOCK_REGISTRY.register("biophysical_polarity_norm")
class BiophysicalPolarityNorm(nn.Module):
    """
    Positive & Negative Indicator Normalization (Jebur et al., 2026, Eqs. 10-11):
      Y_pos = (X - X_min) / (X_max - X_min + eps)
      Y_neg = (X_max - X) / (X_max - X_min + eps)
    Combined via learnable per-channel polarity gates gamma in (0, 1) and affine calibration.
    """

    def __init__(self, in_channels: int, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        # Learnable polarity logits per channel (initialized so vegetation/moisture bands favor positive)
        self.polarity_logits = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.affine_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        denom = (x_max - x_min).clamp_min(self.eps)
        y_pos = (x - x_min) / denom
        y_neg = (x_max - x) / denom
        gamma = torch.sigmoid(self.polarity_logits)
        y = gamma * y_pos + (1.0 - gamma) * y_neg
        return y * self.affine_scale + self.affine_bias


@BLOCK_REGISTRY.register("spatial_grid_gnn", aliases=["hrsei_gnn_layer"])
class SpatialGridGNNLayer(nn.Module):
    """
    Vectorized 4-Cardinal & 8-Neighbor Spatial Grid GNN Layer (Jebur et al., 2026, Eq. 13):
      h_i^{(l+1)} = sigma( W_self h_i^{(l)} + sum_{j in N(i)} (1 / c_ij) W_neigh h_j^{(l)} + b^{(l)} )
    Supports cardinal (N, S, E, W with c_ij = 4) and diagonal neighbors (c_ij = 4*sqrt(2))
    plus optional dilated spatial hops for wider receptive fields.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        include_diagonals: bool = True,
        dilation: int = 1,
        act: str = "relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_diagonals = include_diagonals
        self.dilation = dilation

        self.w_self = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.w_cardinal = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if include_diagonals:
            self.w_diag = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.register_parameter("w_diag", None)

        # Edge-affinity gate to modulate message passing across sharp canopy boundaries
        self.edge_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

        # Build fixed depthwise shift kernels for cardinal and diagonal aggregation
        k_card = torch.zeros(1, 1, 3, 3)
        k_card[0, 0, 0, 1] = 0.25  # North
        k_card[0, 0, 2, 1] = 0.25  # South
        k_card[0, 0, 1, 0] = 0.25  # West
        k_card[0, 0, 1, 2] = 0.25  # East
        self.register_buffer("kernel_cardinal", k_card)

        k_diag = torch.zeros(1, 1, 3, 3)
        inv_d = 0.25 / math.sqrt(2.0)
        k_diag[0, 0, 0, 0] = inv_d
        k_diag[0, 0, 0, 2] = inv_d
        k_diag[0, 0, 2, 0] = inv_d
        k_diag[0, 0, 2, 2] = inv_d
        self.register_buffer("kernel_diag", k_diag)

    def _aggregate_neighbors(self, feat: torch.Tensor, kernel_1x1: torch.Tensor) -> torch.Tensor:
        c = feat.shape[1]
        weight = kernel_1x1.expand(c, 1, 3, 3).contiguous()
        padded = F.pad(feat, (self.dilation, self.dilation, self.dilation, self.dilation), mode="replicate")
        return F.conv2d(padded, weight, dilation=self.dilation, groups=c)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C_in, H, W)
        gated_h = h * self.edge_gate(h)
        agg_card = self._aggregate_neighbors(gated_h, self.kernel_cardinal)
        out = self.w_self(h) + self.w_cardinal(agg_card)
        if self.include_diagonals and self.w_diag is not None:
            agg_diag = self._aggregate_neighbors(gated_h, self.kernel_diag)
            out = out + self.w_diag(agg_diag)
        out = self.norm(out)
        return self.act(out)


# =============================================================================
# 2. Hu & Wu (2026): Venagnosis SPC, Multi-Relational Graph, and PF-GNN
# =============================================================================


@BLOCK_REGISTRY.register("spc_module", aliases=["synergistic_parcel_characterization"])
class SynergisticParcelCharacterization(nn.Module):
    """
    Synergistic Parcel Characterization (SPC) Module (Hu & Wu, 2026, Eqs. 5-6, 16):
    1. Extracts morphological/topographic edge & curvature signatures (Sobel_x, Sobel_y, Laplacian)
       using a topography-focused CNN branch.
    2. Enhances spectral and morphological streams via LayerNorm-stabilized MLPs (Eq. 5):
         h_rs = sigma(W_2 * LayerNorm(sigma(W_1 * x_rs + b_1)) + b_2)
    3. Applies adaptive importance gating (Eq. 16):
         H^(0) = LayerNorm(H_proj + alpha * H_enh)
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        rs_dim = hidden_dim // 2
        morph_dim = hidden_dim - rs_dim

        # Fixed Sobel & Laplacian spatial derivative kernels for morphological characterization
        sobel_x = torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]) / 8.0
        sobel_y = torch.tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]) / 8.0
        laplacian = torch.tensor([[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]) / 4.0
        morph_kernels = torch.stack([sobel_x, sobel_y, laplacian], dim=0)  # (3, 1, 3, 3)
        self.register_buffer("morph_kernels", morph_kernels)

        # Topography/Morphology CNN extractor
        self.topo_cnn = nn.Sequential(
            nn.Conv2d(in_channels * 3, morph_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(morph_dim),
            nn.GELU(),
            nn.Conv2d(morph_dim, morph_dim, kernel_size=3, padding=1, groups=morph_dim, bias=False),
            nn.BatchNorm2d(morph_dim),
            nn.GELU(),
        )

        # Spectral Enhancer MLP (Eq. 5)
        self.rs_fc1 = nn.Linear(in_channels, rs_dim)
        self.rs_ln = nn.LayerNorm(rs_dim)
        self.rs_fc2 = nn.Linear(rs_dim, rs_dim)

        # Morphological Enhancer MLP (Eq. 5)
        self.morph_fc1 = nn.Linear(morph_dim, morph_dim)
        self.morph_ln = nn.LayerNorm(morph_dim)
        self.morph_fc2 = nn.Linear(morph_dim, morph_dim)

        # Physical & Semantic Fusion + Adaptive Input Enhancement (Eq. 6 & Eq. 16)
        self.proj_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.enh_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.imp_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_ln = nn.LayerNorm(hidden_dim)

    def _compute_morph_input(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        weight = self.morph_kernels.repeat(c, 1, 1, 1)  # (3*C, 1, 3, 3)
        grad_feats = F.conv2d(padded, weight, groups=c)  # (B, 3*C, H, W)
        return self.topo_cnn(grad_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> returns (B, hidden_dim, H, W)
        b, c, h, w = x.shape
        morph_map = self._compute_morph_input(x)  # (B, morph_dim, H, W)

        # Permute to token format (B, H, W, D) for LayerNorm-stabilized MLPs
        x_perm = x.permute(0, 2, 3, 1)
        m_perm = morph_map.permute(0, 2, 3, 1)

        h_rs = F.gelu(self.rs_fc2(self.rs_ln(F.gelu(self.rs_fc1(x_perm)))))
        h_morph = F.gelu(self.morph_fc2(self.morph_ln(F.gelu(self.morph_fc1(m_perm)))))

        h_cat = torch.cat([h_rs, h_morph], dim=-1)  # (B, H, W, hidden_dim)
        h_proj = self.proj_mlp(h_cat)
        h_enh = self.enh_mlp(h_proj)
        alpha = self.imp_gate(h_proj)
        h0 = self.out_ln(h_proj + alpha * h_enh)
        return h0.permute(0, 3, 1, 2).contiguous()


@BLOCK_REGISTRY.register("multi_relational_graph")
class MultiRelationalGraphBuilder(nn.Module):
    """
    Knowledge-Guided Multi-Relational Edge Builder (Hu & Wu, 2026, Section III.D):
    Constructs three complementary relation neighbor index tensors for N = H_g * W_g nodes:
      1. Spatial Adjacency (r_spatial): 8-connected local spatial neighborhood.
      2. Semantic Context (r_semantic): Top-k cosine-similar nodes in feature space.
      3. Corridor Association (r_corridor): Anisotropic directional corridor neighbors
         capturing elongated disturbance corridors.
    Returns stacked neighbor index tensor `neighbor_idx` of shape (B, N, K_total).
    """

    def __init__(self, k_semantic: int = 4, corridor_span: int = 2):
        super().__init__()
        self.k_semantic = k_semantic
        self.corridor_span = corridor_span

    @torch.no_grad()
    def forward(self, node_feats: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            node_feats: (B, N, D) where N = H_g * W_g
            grid_hw: (H_g, W_g)
        Returns:
            neighbor_idx: (B, N, K_total) long tensor of neighbor indices for each node.
        """
        b, n, d = node_feats.shape
        hg, wg = grid_hw
        device = node_feats.device

        # 1. Spatial 8-neighborhood + self (9 neighbors)
        rows = torch.arange(hg, device=device).view(hg, 1).expand(hg, wg)
        cols = torch.arange(wg, device=device).view(1, wg).expand(hg, wg)

        spatial_offsets = [
            (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        # 3. Directional corridor offsets (long-range axial & diagonal blowdown corridors)
        s = self.corridor_span
        corridor_offsets = [(-s, 0), (s, 0), (0, -s), (0, s), (-s, -s), (s, s)]

        idx_list = []
        for dr, dc in spatial_offsets + corridor_offsets:
            nr = (rows + dr).clamp(0, hg - 1)
            nc = (cols + dc).clamp(0, wg - 1)
            idx_list.append((nr * wg + nc).reshape(-1))
        fixed_idx = torch.stack(idx_list, dim=-1).unsqueeze(0).expand(b, n, -1)  # (B, N, 15)

        # 2. Semantic Context top-k cosine similarity neighbors
        k_sem = min(self.k_semantic, n)
        norm_f = F.normalize(node_feats.detach(), p=2, dim=-1, eps=1e-6)
        sim = torch.bmm(norm_f, norm_f.transpose(1, 2))  # (B, N, N)
        _, sem_idx = torch.topk(sim, k=k_sem, dim=-1, largest=True)  # (B, N, k_sem)

        return torch.cat([fixed_idx, sem_idx], dim=-1)  # (B, N, 15 + k_sem)


@BLOCK_REGISTRY.register("pf_gnn_layer")
class PFGNNLayer(nn.Module):
    """
    Progressive Fusion Graph Network (PF-GNN) Layer (Hu & Wu, 2026, Eqs. 17-26):
      1. Multi-Head GATv2 Attention Aggregation over multi-relational neighbors (Eqs. 17-18):
           e_ij^(h) = (a^(h))^T LeakyReLU(W_l^(h) h_i + W_r^(h) h_j)
           alpha_ij^(h) = softmax_{j in N(i)}(e_ij^(h))
      2. Hierarchical Dynamic Gating with K parallel candidate update strategies (Eqs. 19-22):
           G_k^(l) = sigma(MLP_{gate,k}([A^(l) || H^(l-1)]))
           O_k^(l) = G_k^(l) * A^(l) + (1 - G_k^(l)) * H^(l-1)
           w^(l)   = softmax(MLP_sel(A^(l)))
           H^(l)   = LayerNorm(sum_k w_k^(l) O_k^(l) + beta * H^(l-1))
      3. Progressive Cross-Layer GRU-style State Fusion (Eqs. 23-26):
           F^(l)   = (1 - Z^(l)) * F^(l-1) + Z^(l) * F_tilde^(l)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_candidates: int = 3,
        residual_beta: float = 0.2,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_candidates = num_candidates
        self.residual_beta = residual_beta

        # GATv2 projections (Eq. 17)
        self.w_left = nn.Linear(dim, dim, bias=False)
        self.w_right = nn.Linear(dim, dim, bias=False)
        self.attn_vec = nn.Parameter(torch.randn(1, 1, 1, num_heads, self.head_dim) * 0.02)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # K parallel candidate gating networks (Eq. 19)
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
            for _ in range(num_candidates)
        ])
        # Selector network (Eq. 21)
        self.selector_mlp = nn.Sequential(
            nn.Linear(dim, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), num_candidates),
        )
        self.layer_norm = nn.LayerNorm(dim)

        # Progressive Cross-Layer State Fusion gates (Eqs. 23-26)
        self.z_gate = nn.Linear(dim * 2, dim)
        self.r_gate = nn.Linear(dim * 2, dim)
        self.c_proj = nn.Linear(dim * 2, dim)

    def forward(
        self,
        h_prev: torch.Tensor,
        f_prev: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_prev: (B, N, D) previous layer node embeddings H^(l-1)
            f_prev: (B, N, D) progressive cross-layer memory state F^(l-1)
            neighbor_idx: (B, N, K_nbr) neighbor indices
        Returns:
            h_new: (B, N, D) updated node features H^(l)
            f_new: (B, N, D) updated progressive fusion state F^(l)
        """
        b, n, d = h_prev.shape
        k_nbr = neighbor_idx.shape[-1]

        # 1. GATv2 Multi-Head Aggregation (Eqs. 17-18)
        hl = self.w_left(h_prev).view(b, n, 1, self.num_heads, self.head_dim)  # (B, N, 1, H, Dh)
        hr_all = self.w_right(h_prev)  # (B, N, D)

        # Gather neighbor features vectorially
        flat_idx = neighbor_idx.reshape(b, n * k_nbr, 1).expand(b, n * k_nbr, d)
        hr_nbr = torch.gather(hr_all, dim=1, index=flat_idx).view(b, n, k_nbr, self.num_heads, self.head_dim)

        # GATv2: a^T LeakyReLU(W_l h_i + W_r h_j)
        e_ij = (self.leaky_relu(hl + hr_nbr) * self.attn_vec).sum(dim=-1)  # (B, N, K_nbr, H)
        alpha_ij = F.softmax(e_ij, dim=2).unsqueeze(-1)  # (B, N, K_nbr, H, 1)

        a_curr = (alpha_ij * hr_nbr).sum(dim=2).reshape(b, n, d)  # (B, N, D)

        # 2. Hierarchical Dynamic Gating (Eqs. 19-22)
        c_cat = torch.cat([a_curr, h_prev], dim=-1)  # (B, N, 2D)
        candidates = []
        for gate_mlp in self.gate_mlps:
            g_k = gate_mlp(c_cat)  # (B, N, D)
            o_k = g_k * a_curr + (1.0 - g_k) * h_prev
            candidates.append(o_k)
        o_stack = torch.stack(candidates, dim=2)  # (B, N, K, D)
        w_sel = F.softmax(self.selector_mlp(a_curr), dim=-1).unsqueeze(-1)  # (B, N, K, 1)

        h_new = self.layer_norm((w_sel * o_stack).sum(dim=2) + self.residual_beta * h_prev)

        # 3. Progressive Cross-Layer Fusion (Eqs. 23-26)
        hf_cat = torch.cat([h_new, f_prev], dim=-1)
        z_l = torch.sigmoid(self.z_gate(hf_cat))
        r_l = torch.sigmoid(self.r_gate(hf_cat))
        f_tilde = torch.tanh(self.c_proj(torch.cat([h_new, r_l * f_prev], dim=-1)))
        f_new = (1.0 - z_l) * f_prev + z_l * f_tilde

        return h_new, f_new


# =============================================================================
# 3. Liu et al. (2026): GSRI Dynamic Dilated KNN-Graph & Dual-PE Cross-Attention
# =============================================================================


def _sinusoidal_pe_2d(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generates 2D Sinusoidal Positional Encoding (1, dim, h, w)."""
    y = torch.arange(h, device=device, dtype=dtype).unsqueeze(1).expand(h, w)
    x = torch.arange(w, device=device, dtype=dtype).unsqueeze(0).expand(h, w)
    half = dim // 2
    quarter = max(half // 2, 1)
    omega = torch.arange(quarter, device=device, dtype=dtype) / quarter
    omega = 1.0 / (10000.0 ** omega)

    out_y = y.reshape(-1, 1) * omega.view(1, -1)
    out_x = x.reshape(-1, 1) * omega.view(1, -1)
    pe = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
    if pe.shape[-1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[-1]))
    else:
        pe = pe[:, :dim]
    return pe.transpose(0, 1).view(1, dim, h, w)


@BLOCK_REGISTRY.register("ddkg_graph_block")
class DDKGGraphBlock(nn.Module):
    """
    Dynamic Dilated KNN-Graph (DDKG) with Max-Relative Graph Convolution (Liu et al., 2026, Eqs. 1-2):
      v_i' = v_i + max_{j in N_dilated_knn(i)} phi(W(v_j - v_i))
      F_O  = CBN(DDKG(CBN(F_i), RPE)) + F_i
    """

    def __init__(self, channels: int, k_neighbors: int = 6, dilation: int = 2):
        super().__init__()
        self.channels = channels
        self.k = k_neighbors
        self.dilation = dilation

        self.cbn1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # Relative Positional Encoding generator via depthwise convolution
        self.rpe = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)

        # Max-Relative Graph Convolution update phi(W([v_i || max(v_j - v_i)]))
        self.rel_proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.cbn2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        f_in = self.cbn1(x)
        f_rpe = f_in + self.rpe(f_in)

        # Pool if grid is large to keep pairwise KNN fast and memory-bounded
        pool_stride = 1
        if h * w > 1024:
            pool_stride = 2 if h * w <= 4096 else 4
            f_graph = F.adaptive_avg_pool2d(f_rpe, (max(h // pool_stride, 4), max(w // pool_stride, 4)))
        else:
            f_graph = f_rpe

        gh, gw = f_graph.shape[-2:]
        n = gh * gw
        tokens = f_graph.flatten(2).transpose(1, 2)  # (B, N, C)

        # Dynamic Dilated KNN in normalized embedding space
        norm_t = F.normalize(tokens.detach(), p=2, dim=-1, eps=1e-6)
        sim = torch.bmm(norm_t, norm_t.transpose(1, 2))  # (B, N, N)

        k_dilated = min(self.k * self.dilation, n)
        _, top_idx = torch.topk(sim, k=k_dilated, dim=-1, largest=True)
        dilated_idx = top_idx[:, :, :: self.dilation]  # (B, N, k_eff)
        k_eff = dilated_idx.shape[-1]

        # Gather neighbor tokens and compute max-relative difference (Eq. 2)
        flat_idx = dilated_idx.reshape(b, n * k_eff, 1).expand(b, n * k_eff, c)
        nbr_tokens = torch.gather(tokens, dim=1, index=flat_idx).view(b, n, k_eff, c)
        max_rel_diff, _ = (nbr_tokens - tokens.unsqueeze(2)).max(dim=2)  # (B, N, C)

        updated = tokens + self.rel_proj(torch.cat([tokens, max_rel_diff], dim=-1))  # (B, N, C)
        f_out = updated.transpose(1, 2).view(b, c, gh, gw)
        if (gh, gw) != (h, w):
            f_out = F.interpolate(f_out, size=(h, w), mode="bilinear", align_corners=False)

        return self.cbn2(f_out) + x


@ATTENTION_REGISTRY.register("gsri_module", aliases=["gsri_interaction"])
class GSRIModule(nn.Module):
    """
    Graph-based Semantic Rerouting Interaction (GSRI) Module (Liu et al., 2026, Eqs. 1-5):
    1. Rotational Semantic Graph Representation via `DDKGGraphBlock` on semantic feature map F_i.
    2. Dual-PE Supervised Cross-Attention (Eqs. 3-5) aligning graph-modeled features F_O
       with pyramidal spatial feature map F_P using Sinusoidal + Relative Positional Encoding.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        k_neighbors: int = 6,
        dilation: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.ddkg = DDKGGraphBlock(channels, k_neighbors=k_neighbors, dilation=dilation)

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, f_spatial: torch.Tensor, f_semantic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            f_spatial: Pyramidal feature map F_P of shape (B, C, H, W)
            f_semantic: High-level feature map F_i of shape (B, C, H_s, W_s). Defaults to f_spatial.
        Returns:
            Enhanced rotation-robust feature map F_T of shape (B, C, H, W)
        """
        if f_semantic is None:
            f_semantic = f_spatial

        b, c, h, w = f_spatial.shape
        # 1. Rotational Semantic Graph Representation (Eqs. 1-2)
        f_o = self.ddkg(f_semantic)

        # Pool keys/values to compact attention grid (e.g., <= 16x16) for cross-scale efficiency
        attn_h, attn_w = min(h, 16), min(w, 16)
        f_p_sub = F.adaptive_avg_pool2d(f_spatial, (attn_h, attn_w))
        f_o_sub = F.adaptive_avg_pool2d(f_o, (attn_h, attn_w))

        # 2. Dual-PE Supervised Cross-Attention (Eqs. 3-5)
        pe = _sinusoidal_pe_2d(attn_h, attn_w, c, f_spatial.device, f_spatial.dtype)
        seq_p = self.norm_q((f_p_sub + pe).flatten(2).transpose(1, 2))  # (B, L, C)
        seq_o = self.norm_kv((f_o_sub + pe).flatten(2).transpose(1, 2))  # (B, L, C)

        l_seq = seq_p.shape[1]
        q = self.q_proj(seq_p).view(b, l_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(seq_o).view(b, l_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(seq_o).view(b, l_seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, l_seq, c)
        ctx_map = self.out_proj(ctx).transpose(1, 2).view(b, c, attn_h, attn_w)

        if (attn_h, attn_w) != (h, w):
            ctx_map = F.interpolate(ctx_map, size=(h, w), mode="bilinear", align_corners=False)

        out = f_spatial + ctx_map
        return out + self.ffn(out)


# =============================================================================
# 4. Shao et al. (2026): CGMN Distance-Weighted Spatial Graph Branch
# =============================================================================


@BLOCK_REGISTRY.register("cgmn_graph_branch")
class CGMNSpatialGraphBranch(nn.Module):
    """
    CGMN Spatial Graph Convolution Branch (Shao et al., 2026, Section 3.1):
    Constructs a spatial proximity graph over regional nodes with edge weights
    defined as the reciprocal of spatial distance w_ij = 1 / (d_ij + eps) modulated
    by learned feature affinity, and applies multi-layer graph convolutions.
    """

    def __init__(self, channels: int, num_layers: int = 2, grid_size: int = 16):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.grid_size = grid_size

        self.gcn_self = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_layers)])
        self.gcn_neigh = nn.ModuleList([nn.Linear(channels, channels, bias=False) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(num_layers)])
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def _build_distance_adj(self, gh: int, gw: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.arange(gh, device=device, dtype=dtype)
        x = torch.arange(gw, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # (N, 2)
        dist = torch.cdist(coords, coords, p=2.0)  # (N, N)
        # Reciprocal distance weights w_ij = 1 / d_ij for non-self edges within radius
        inv_dist = torch.where(dist > 0, 1.0 / (dist + 1e-5), torch.zeros_like(dist))
        # Row-normalize adjacency matrix
        deg = inv_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return inv_dist / deg  # (N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        gh, gw = min(h, self.grid_size), min(w, self.grid_size)
        x_pool = F.adaptive_avg_pool2d(x, (gh, gw))
        nodes = x_pool.flatten(2).transpose(1, 2)  # (B, N, C)

        adj_dist = self._build_distance_adj(gh, gw, x.device, x.dtype).unsqueeze(0)  # (1, N, N)
        # Modulate reciprocal distance with cosine feature similarity
        norm_n = F.normalize(nodes, p=2, dim=-1, eps=1e-6)
        aff = torch.relu(torch.bmm(norm_n, norm_n.transpose(1, 2)))  # (B, N, N)
        adj = adj_dist * (0.5 + 0.5 * aff)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        h_node = nodes
        for w_s, w_n, ln in zip(self.gcn_self, self.gcn_neigh, self.norms):
            agg = torch.bmm(adj, h_node)
            h_node = F.gelu(ln(w_s(h_node) + w_n(agg))) + h_node

        out_map = h_node.transpose(1, 2).view(b, c, gh, gw)
        if (gh, gw) != (h, w):
            out_map = F.interpolate(out_map, size=(h, w), mode="bilinear", align_corners=False)
        return self.refine_conv(out_map + x)


__all__ = [
    "BiophysicalPolarityNorm",
    "SpatialGridGNNLayer",
    "SynergisticParcelCharacterization",
    "MultiRelationalGraphBuilder",
    "PFGNNLayer",
    "DDKGGraphBlock",
    "GSRIModule",
    "CGMNSpatialGraphBranch",
]
