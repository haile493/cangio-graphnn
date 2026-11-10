"""
Self-contained fallback definitions for Can Gio Graph Neural Network Models and Losses.
Enables running the code on any standard PyTorch environment without external private packages.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Standard Base Classes & Configuration Containers
# =============================================================================


class ModelOutput(dict):
    """Container for model outputs with attribute access and dict compatibility."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            setattr(self, k, v)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        setattr(self, key, value)


@dataclass
class ModelConfig:
    arch: str = "custom"
    dims: int = 2
    in_channels: int = 9
    num_seg_classes: int = 1
    base_channels: int = 48
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str, **kwargs: Any) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {"state_dict": self.state_dict(), **kwargs}
        torch.save(data, path)


def _conv_bn_gelu(in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


def _sinusoidal_pe_2d(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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


# =============================================================================
# Graph Modules
# =============================================================================


class BiophysicalPolarityNorm(nn.Module):
    def __init__(self, in_channels: int, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.polarity_logits = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.affine_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        denom = (x_max - x_min).clamp_min(self.eps)
        y_pos = (x - x_min) / denom
        y_neg = (x_max - x) / denom
        gamma = torch.sigmoid(self.polarity_logits)
        y = gamma * y_pos + (1.0 - gamma) * y_neg
        return y * self.affine_scale + self.affine_bias


class SpatialGridGNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, include_diagonals: bool = True, dilation: int = 1, act: str = "relu"):
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

        self.edge_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if act == "relu" else (nn.GELU() if act == "gelu" else nn.Sigmoid())

        k_card = torch.zeros(1, 1, 3, 3)
        k_card[0, 0, 0, 1] = 0.25
        k_card[0, 0, 2, 1] = 0.25
        k_card[0, 0, 1, 0] = 0.25
        k_card[0, 0, 1, 2] = 0.25
        self.register_buffer("kernel_cardinal", k_card)

        k_diag = torch.zeros(1, 1, 3, 3)
        inv_d = 0.25 / math.sqrt(2.0)
        k_diag[0, 0, 0, 0] = inv_d
        k_diag[0, 0, 0, 2] = inv_d
        k_diag[0, 0, 2, 0] = inv_d
        k_diag[0, 0, 2, 2] = inv_d
        self.register_buffer("kernel_diag", k_diag)

    def _aggregate(self, feat: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        c = feat.shape[1]
        weight = kernel.expand(c, 1, 3, 3).contiguous()
        padded = F.pad(feat, (self.dilation, self.dilation, self.dilation, self.dilation), mode="replicate")
        return F.conv2d(padded, weight, dilation=self.dilation, groups=c)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gated = h * self.edge_gate(h)
        out = self.w_self(h) + self.w_cardinal(self._aggregate(gated, self.kernel_cardinal))
        if self.include_diagonals and self.w_diag is not None:
            out = out + self.w_diag(self._aggregate(gated, self.kernel_diag))
        return self.act(self.norm(out))


class SynergisticParcelCharacterization(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        rs_dim = hidden_dim // 2
        morph_dim = hidden_dim - rs_dim

        sobel_x = torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]) / 8.0
        sobel_y = torch.tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]) / 8.0
        laplacian = torch.tensor([[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]) / 4.0
        self.register_buffer("morph_kernels", torch.stack([sobel_x, sobel_y, laplacian], dim=0))

        self.topo_cnn = nn.Sequential(
            nn.Conv2d(in_channels * 3, morph_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(morph_dim),
            nn.GELU(),
            nn.Conv2d(morph_dim, morph_dim, kernel_size=3, padding=1, groups=morph_dim, bias=False),
            nn.BatchNorm2d(morph_dim),
            nn.GELU(),
        )
        self.rs_fc1 = nn.Linear(in_channels, rs_dim)
        self.rs_ln = nn.LayerNorm(rs_dim)
        self.rs_fc2 = nn.Linear(rs_dim, rs_dim)

        self.morph_fc1 = nn.Linear(morph_dim, morph_dim)
        self.morph_ln = nn.LayerNorm(morph_dim)
        self.morph_fc2 = nn.Linear(morph_dim, morph_dim)

        self.proj_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.enh_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.imp_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.out_ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        weight = self.morph_kernels.repeat(c, 1, 1, 1)
        grad = self.topo_cnn(F.conv2d(padded, weight, groups=c))

        x_p = x.permute(0, 2, 3, 1)
        m_p = grad.permute(0, 2, 3, 1)
        h_rs = F.gelu(self.rs_fc2(self.rs_ln(F.gelu(self.rs_fc1(x_p)))))
        h_m = F.gelu(self.morph_fc2(self.morph_ln(F.gelu(self.morph_fc1(m_p)))))

        cat = torch.cat([h_rs, h_m], dim=-1)
        proj = self.proj_mlp(cat)
        enh = self.enh_mlp(proj)
        gate = self.imp_gate(proj)
        h0 = self.out_ln(proj + gate * enh)
        return h0.permute(0, 3, 1, 2).contiguous()


class MultiRelationalGraphBuilder(nn.Module):
    def __init__(self, k_semantic: int = 4, corridor_span: int = 2):
        super().__init__()
        self.k_semantic = k_semantic
        self.corridor_span = corridor_span

    @torch.no_grad()
    def forward(self, node_feats: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        b, n, d = node_feats.shape
        hg, wg = grid_hw
        device = node_feats.device

        rows = torch.arange(hg, device=device).view(hg, 1).expand(hg, wg)
        cols = torch.arange(wg, device=device).view(1, wg).expand(hg, wg)
        spatial_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        s = self.corridor_span
        corridor_offsets = [(-s, 0), (s, 0), (0, -s), (0, s), (-s, -s), (s, s)]

        idx_list = []
        for dr, dc in spatial_offsets + corridor_offsets:
            nr = (rows + dr).clamp(0, hg - 1)
            nc = (cols + dc).clamp(0, wg - 1)
            idx_list.append((nr * wg + nc).reshape(-1))
        fixed_idx = torch.stack(idx_list, dim=-1).unsqueeze(0).expand(b, n, -1)

        k_sem = min(self.k_semantic, n)
        norm_f = F.normalize(node_feats.detach(), p=2, dim=-1, eps=1e-6)
        sim = torch.bmm(norm_f, norm_f.transpose(1, 2))
        _, sem_idx = torch.topk(sim, k=k_sem, dim=-1, largest=True)
        return torch.cat([fixed_idx, sem_idx], dim=-1)


class PFGNNLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, num_candidates: int = 3, residual_beta: float = 0.2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_candidates = num_candidates
        self.residual_beta = residual_beta

        self.w_left = nn.Linear(dim, dim, bias=False)
        self.w_right = nn.Linear(dim, dim, bias=False)
        self.attn_vec = nn.Parameter(torch.randn(1, 1, 1, num_heads, self.head_dim) * 0.02)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.gate_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())
            for _ in range(num_candidates)
        ])
        self.selector_mlp = nn.Sequential(nn.Linear(dim, max(dim // 2, 8)), nn.GELU(), nn.Linear(max(dim // 2, 8), num_candidates))
        self.layer_norm = nn.LayerNorm(dim)

        self.z_gate = nn.Linear(dim * 2, dim)
        self.r_gate = nn.Linear(dim * 2, dim)
        self.c_proj = nn.Linear(dim * 2, dim)

    def forward(self, h_prev: torch.Tensor, f_prev: torch.Tensor, neighbor_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, d = h_prev.shape
        k_nbr = neighbor_idx.shape[-1]

        hl = self.w_left(h_prev).view(b, n, 1, self.num_heads, self.head_dim)
        hr_all = self.w_right(h_prev)
        flat_idx = neighbor_idx.reshape(b, n * k_nbr, 1).expand(b, n * k_nbr, d)
        hr_nbr = torch.gather(hr_all, dim=1, index=flat_idx).view(b, n, k_nbr, self.num_heads, self.head_dim)

        e_ij = (self.leaky_relu(hl + hr_nbr) * self.attn_vec).sum(dim=-1)
        alpha_ij = F.softmax(e_ij, dim=2).unsqueeze(-1)
        a_curr = (alpha_ij * hr_nbr).sum(dim=2).reshape(b, n, d)

        c_cat = torch.cat([a_curr, h_prev], dim=-1)
        candidates = [gate(c_cat) * a_curr + (1.0 - gate(c_cat)) * h_prev for gate in self.gate_mlps]
        o_stack = torch.stack(candidates, dim=2)
        w_sel = F.softmax(self.selector_mlp(a_curr), dim=-1).unsqueeze(-1)
        h_new = self.layer_norm((w_sel * o_stack).sum(dim=2) + self.residual_beta * h_prev)

        hf_cat = torch.cat([h_new, f_prev], dim=-1)
        z_l = torch.sigmoid(self.z_gate(hf_cat))
        r_l = torch.sigmoid(self.r_gate(hf_cat))
        f_tilde = torch.tanh(self.c_proj(torch.cat([h_new, r_l * f_prev], dim=-1)))
        f_new = (1.0 - z_l) * f_prev + z_l * f_tilde
        return h_new, f_new


class DDKGGraphBlock(nn.Module):
    def __init__(self, channels: int, k_neighbors: int = 6, dilation: int = 2):
        super().__init__()
        self.channels = channels
        self.k = k_neighbors
        self.dilation = dilation
        self.cbn1 = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.GELU())
        self.rpe = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.rel_proj = nn.Sequential(nn.Linear(channels * 2, channels), nn.LayerNorm(channels), nn.GELU())
        self.cbn2 = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f_in = self.cbn1(x)
        f_rpe = f_in + self.rpe(f_in)
        gh, gw = min(h, 16), min(w, 16)
        f_sub = F.adaptive_avg_pool2d(f_rpe, (gh, gw))
        n = gh * gw
        tokens = f_sub.flatten(2).transpose(1, 2)

        norm_t = F.normalize(tokens.detach(), p=2, dim=-1, eps=1e-6)
        sim = torch.bmm(norm_t, norm_t.transpose(1, 2))
        k_eff = min(self.k * self.dilation, n)
        _, top_idx = torch.topk(sim, k=k_eff, dim=-1, largest=True)
        dilated_idx = top_idx[:, :, :: self.dilation]
        k_d = dilated_idx.shape[-1]

        flat_idx = dilated_idx.reshape(b, n * k_d, 1).expand(b, n * k_d, c)
        nbr_tokens = torch.gather(tokens, dim=1, index=flat_idx).view(b, n, k_d, c)
        max_rel_diff, _ = (nbr_tokens - tokens.unsqueeze(2)).max(dim=2)
        updated = tokens + self.rel_proj(torch.cat([tokens, max_rel_diff], dim=-1))
        f_out = updated.transpose(1, 2).view(b, c, gh, gw)
        if (gh, gw) != (h, w):
            f_out = F.interpolate(f_out, size=(h, w), mode="bilinear", align_corners=False)
        return self.cbn2(f_out) + x


class GSRIModule(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, k_neighbors: int = 6, dilation: int = 2):
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
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, f_spatial: torch.Tensor, f_semantic: Optional[torch.Tensor] = None) -> torch.Tensor:
        if f_semantic is None:
            f_semantic = f_spatial
        b, c, h, w = f_spatial.shape
        f_o = self.ddkg(f_semantic)
        attn_h, attn_w = min(h, 16), min(w, 16)
        f_p_sub = F.adaptive_avg_pool2d(f_spatial, (attn_h, attn_w))
        f_o_sub = F.adaptive_avg_pool2d(f_o, (attn_h, attn_w))

        pe = _sinusoidal_pe_2d(attn_h, attn_w, c, f_spatial.device, f_spatial.dtype)
        seq_p = self.norm_q((f_p_sub + pe).flatten(2).transpose(1, 2))
        seq_o = self.norm_kv((f_o_sub + pe).flatten(2).transpose(1, 2))
        l_seq = seq_p.shape[1]

        q = self.q_proj(seq_p).view(b, l_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(seq_o).view(b, l_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(seq_o).view(b, l_seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, l_seq, c)
        ctx_map = self.out_proj(ctx).transpose(1, 2).view(b, c, attn_h, attn_w)
        if (attn_h, attn_w) != (h, w):
            ctx_map = F.interpolate(ctx_map, size=(h, w), mode="bilinear", align_corners=False)
        out = f_spatial + ctx_map
        return out + self.ffn(out)


class CGMNSpatialGraphBranch(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2, grid_size: int = 16):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.grid_size = grid_size
        self.gcn_self = nn.ModuleList([nn.Linear(channels, channels) for _ in range(num_layers)])
        self.gcn_neigh = nn.ModuleList([nn.Linear(channels, channels, bias=False) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(num_layers)])
        self.refine_conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.GELU())

    def _build_distance_adj(self, gh: int, gw: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.arange(gh, device=device, dtype=dtype)
        x = torch.arange(gw, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        dist = torch.cdist(coords, coords, p=2.0)
        inv_dist = torch.where(dist > 0, 1.0 / (dist + 1e-5), torch.zeros_like(dist))
        deg = inv_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return inv_dist / deg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gh, gw = min(h, self.grid_size), min(w, self.grid_size)
        x_pool = F.adaptive_avg_pool2d(x, (gh, gw))
        nodes = x_pool.flatten(2).transpose(1, 2)

        adj_dist = self._build_distance_adj(gh, gw, x.device, x.dtype).unsqueeze(0)
        norm_n = F.normalize(nodes, p=2, dim=-1, eps=1e-6)
        aff = torch.relu(torch.bmm(norm_n, norm_n.transpose(1, 2)))
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


# =============================================================================
# The 6 Standard Model Architectures
# =============================================================================


class HRSEIGNNNet(BaseModel):
    def __init__(self, in_channels: int = 9, num_classes: int = 1, hidden_dim: int = 32, num_gnn_layers: int = 3, include_diagonals: bool = True):
        super().__init__(ModelConfig(arch="hrsei_gnn", in_channels=in_channels, num_seg_classes=num_classes, base_channels=hidden_dim))
        self.polarity_norm = BiophysicalPolarityNorm(in_channels)
        self.stem = _conv_bn_gelu(in_channels * 2, hidden_dim, 3, 1)
        self.gnn_layers = nn.ModuleList([
            SpatialGridGNNLayer(hidden_dim, hidden_dim, include_diagonals=include_diagonals, dilation=1 if i == 0 else (2 if i == 1 else 3))
            for i in range(num_gnn_layers)
        ])
        self.coarse_gnn = nn.Sequential(
            nn.AvgPool2d(2, 2),
            SpatialGridGNNLayer(hidden_dim, hidden_dim, include_diagonals=True, dilation=1),
            SpatialGridGNNLayer(hidden_dim, hidden_dim, include_diagonals=True, dilation=2),
        )
        self.hrsei_head = nn.Sequential(nn.Conv2d(hidden_dim, 16, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        self.risk_head = nn.Sequential(
            _conv_bn_gelu(hidden_dim * 2 + 1, hidden_dim, 3, 1),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        hw = x.shape[-2:]
        x_pol = self.polarity_norm(x)
        h = self.stem(torch.cat([x, x_pol], dim=1))
        for gnn in self.gnn_layers:
            h = h + gnn(h)
        h_coarse = F.interpolate(self.coarse_gnn(h), size=hw, mode="bilinear", align_corners=False)
        hrsei_logits = self.hrsei_head(h)
        hrsei_index = torch.sigmoid(hrsei_logits)
        fused = torch.cat([h, h_coarse, 1.0 - hrsei_index], dim=1)
        return ModelOutput(seg_logits=self.risk_head(fused), aux_seg_logits=hrsei_logits, hrsei_index=hrsei_index, latent_feat=h)


class VenagnosisPFGNNNet(BaseModel):
    def __init__(self, in_channels: int = 9, num_classes: int = 1, base_channels: int = 48, num_gnn_layers: int = 3, num_heads: int = 4, parcel_grid_size: int = 31):
        super().__init__(ModelConfig(arch="venagnosis_pfgnn", in_channels=in_channels, num_seg_classes=num_classes, base_channels=base_channels))
        self.parcel_grid_size = parcel_grid_size
        self.spc = SynergisticParcelCharacterization(in_channels=in_channels, hidden_dim=base_channels)
        self.local_enc1 = _conv_bn_gelu(base_channels, base_channels, 3, 1)
        self.local_enc2 = nn.Sequential(nn.MaxPool2d(2, 2), _conv_bn_gelu(base_channels, base_channels, 3, 1))
        self.graph_builder = MultiRelationalGraphBuilder(k_semantic=4, corridor_span=2)
        self.pf_gnn_layers = nn.ModuleList([PFGNNLayer(base_channels, num_heads=num_heads) for _ in range(num_gnn_layers)])
        self.parcel_head = nn.Sequential(nn.Linear(base_channels, base_channels // 2), nn.LayerNorm(base_channels // 2), nn.GELU(), nn.Linear(base_channels // 2, num_classes))
        self.decoder = nn.Sequential(
            _conv_bn_gelu(base_channels * 3, base_channels, 3, 1),
            _conv_bn_gelu(base_channels, base_channels // 2, 3, 1),
            nn.Conv2d(base_channels // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        b, _, h, w = x.shape
        h_spc = self.spc(x)
        f_loc1 = self.local_enc1(h_spc)
        f_loc2 = self.local_enc2(f_loc1)

        hg = min(h, self.parcel_grid_size)
        wg = min(w, self.parcel_grid_size)
        h_parcel = F.adaptive_avg_pool2d(f_loc2, (hg, wg))
        d = h_parcel.shape[1]
        h_nodes = h_parcel.flatten(2).transpose(1, 2)

        neighbor_idx = self.graph_builder(h_nodes, (hg, wg))
        h_curr = h_nodes
        f_state = h_nodes.clone()
        for pf in self.pf_gnn_layers:
            h_curr, f_state = pf(h_curr, f_state, neighbor_idx)

        parcel_logits = self.parcel_head(f_state).transpose(1, 2).view(b, -1, hg, wg)
        aux_seg_logits = F.interpolate(parcel_logits, size=(h, w), mode="bilinear", align_corners=False)

        f_graph_up = F.interpolate(f_state.transpose(1, 2).view(b, d, hg, wg), size=(h, w), mode="bilinear", align_corners=False)
        f_loc2_up = F.interpolate(f_loc2, size=(h, w), mode="bilinear", align_corners=False)
        seg_logits = self.decoder(torch.cat([f_loc1, f_loc2_up, f_graph_up], dim=1))
        return ModelOutput(seg_logits=seg_logits, aux_seg_logits=aux_seg_logits, latent_feat=f_state)


class GSRIRotationRobustNet(BaseModel):
    def __init__(self, in_channels: int = 9, num_classes: int = 1, base_channels: int = 48, num_heads: int = 4, k_neighbors: int = 6, dilation: int = 2):
        super().__init__(ModelConfig(arch="gsri_net", in_channels=in_channels, num_seg_classes=num_classes, base_channels=base_channels))
        self.enc1 = nn.Sequential(_conv_bn_gelu(in_channels, base_channels, 3, 1), _conv_bn_gelu(base_channels, base_channels, 3, 1))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2, 2), _conv_bn_gelu(base_channels, base_channels * 2, 3, 1), _conv_bn_gelu(base_channels * 2, base_channels, 1, 0))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2, 2), _conv_bn_gelu(base_channels, base_channels * 2, 3, 1), _conv_bn_gelu(base_channels * 2, base_channels, 1, 0))

        self.gsri_high = GSRIModule(base_channels, num_heads=num_heads, k_neighbors=k_neighbors, dilation=dilation)
        self.gsri_mid = GSRIModule(base_channels, num_heads=num_heads, k_neighbors=k_neighbors, dilation=dilation)
        self.gsri_low = GSRIModule(base_channels, num_heads=num_heads, k_neighbors=k_neighbors, dilation=1)

        self.edge_head = nn.Sequential(_conv_bn_gelu(base_channels, base_channels // 2, 3, 1), nn.Conv2d(base_channels // 2, 1, 1))
        self.fusion_head = nn.Sequential(
            _conv_bn_gelu(base_channels * 3, base_channels, 3, 1),
            _conv_bn_gelu(base_channels, base_channels // 2, 3, 1),
            nn.Conv2d(base_channels // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        hw = x.shape[-2:]
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)

        t3 = self.gsri_high(f3, f3)
        t2 = self.gsri_mid(f2, t3)
        t1 = self.gsri_low(f1, t2)

        t3_up = F.interpolate(t3, size=hw, mode="bilinear", align_corners=False)
        t2_up = F.interpolate(t2, size=hw, mode="bilinear", align_corners=False)

        edge_logits = self.edge_head(t1)
        seg_logits = self.fusion_head(torch.cat([t1, t2_up, t3_up], dim=1))
        return ModelOutput(seg_logits=seg_logits, edge_logits=edge_logits, latent_feat=t1)


class CGMNMultiBranchNet(BaseModel):
    def __init__(self, in_channels: int = 9, num_classes: int = 1, branch_dim: int = 32, num_gnn_layers: int = 2, grid_size: int = 16, mask_ratio: float = 0.2):
        super().__init__(ModelConfig(arch="cgmn_net", in_channels=in_channels, num_seg_classes=num_classes, base_channels=branch_dim))
        self.branch_dim = branch_dim
        self.mask_ratio = mask_ratio
        self.input_proj = _conv_bn_gelu(in_channels, branch_dim, 3, 1)
        self.graph_branch = CGMNSpatialGraphBranch(branch_dim, num_layers=num_gnn_layers, grid_size=grid_size)
        self.static_branch = nn.Sequential(
            nn.Conv2d(branch_dim, branch_dim * 2, 1, bias=False),
            nn.BatchNorm2d(branch_dim * 2),
            nn.GELU(),
            nn.Conv2d(branch_dim * 2, branch_dim, 1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(input_size=branch_dim, hidden_size=branch_dim, num_layers=1, batch_first=True)
        self.scale_pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.scale_pool4 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.fusion_head = nn.Sequential(
            _conv_bn_gelu(branch_dim * 3, branch_dim * 2, 3, 1),
            _conv_bn_gelu(branch_dim * 2, branch_dim, 3, 1),
            nn.Conv2d(branch_dim, num_classes, 1),
        )

    def _run_gru(self, feat_seq: List[torch.Tensor]) -> torch.Tensor:
        b, d, h, w = feat_seq[0].shape
        t_len = len(feat_seq)
        seq = torch.stack(feat_seq, dim=1).permute(0, 3, 4, 1, 2).reshape(b * h * w, t_len, d)
        gru_out, _ = self.gru(seq)
        return gru_out[:, -1, :].view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, compute_masked_branch: Optional[bool] = None) -> ModelOutput:
        if x.ndim == 5:
            b, t_len, c, h, w = x.shape
            t_seq = [self.input_proj(x[:, t]) for t in range(t_len)]
            f0 = t_seq[-1]
            b_temp = self._run_gru(t_seq)
        else:
            f0 = self.input_proj(x)
            b_temp = self._run_gru([self.scale_pool4(f0), self.scale_pool2(f0), f0])

        b_graph = self.graph_branch(f0)
        b_static = f0 + self.static_branch(f0)
        seg_logits = self.fusion_head(torch.cat([b_graph, b_static, b_temp], dim=1))
        return ModelOutput(seg_logits=seg_logits)


class BiGSAMNet(BaseModel):
    def __init__(self, in_channels: int = 9, num_classes: int = 1, base_channels: int = 48, **kwargs):
        super().__init__(ModelConfig(arch="big_sam", in_channels=in_channels, num_seg_classes=num_classes, base_channels=base_channels))
        self.stem = _conv_bn_gelu(in_channels, base_channels, 3, 1)
        self.enc = nn.Sequential(_conv_bn_gelu(base_channels, base_channels * 2, 3, 1), _conv_bn_gelu(base_channels * 2, base_channels, 3, 1))
        self.graph = CGMNSpatialGraphBranch(base_channels, num_layers=2, grid_size=20)
        self.head = nn.Sequential(_conv_bn_gelu(base_channels, base_channels // 2, 3, 1), nn.Conv2d(base_channels // 2, num_classes, 1))

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.stem(x)
        h = self.enc(h)
        h = self.graph(h)
        return ModelOutput(seg_logits=self.head(h), latent_feat=h)


class BiGSAMNetV2(BaseModel):
    def __init__(self, in_channels: int = 9, num_classes: int = 1, base_channels: int = 48, **kwargs):
        super().__init__(ModelConfig(arch="big_sam_v2", in_channels=in_channels, num_seg_classes=num_classes, base_channels=base_channels))
        self.spc = SynergisticParcelCharacterization(in_channels, hidden_dim=base_channels)
        self.gsri = GSRIModule(base_channels, num_heads=4, k_neighbors=4, dilation=2)
        self.graph = CGMNSpatialGraphBranch(base_channels, num_layers=2, grid_size=24)
        self.decoder = nn.Sequential(
            _conv_bn_gelu(base_channels * 2, base_channels, 3, 1),
            _conv_bn_gelu(base_channels, base_channels // 2, 3, 1),
            nn.Conv2d(base_channels // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h_spc = self.spc(x)
        h_gsri = self.gsri(h_spc)
        h_graph = self.graph(h_spc)
        fused = torch.cat([h_gsri, h_graph], dim=1)
        return ModelOutput(seg_logits=self.decoder(fused), latent_feat=h_gsri)


@dataclass
class BiGSAMConfig(ModelConfig):
    pass


@dataclass
class BiGSAMV2Config(ModelConfig):
    pass


# =============================================================================
# Standard Loss Functions
# =============================================================================


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(-2, -1))
        cardinality = (probs + targets).sum(dim=(-2, -1))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        w = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = -w * ((1.0 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()


FocalLoss = BinaryFocalLoss


class VenagnosisLoss(nn.Module):
    def __init__(self, alpha: float = 0.80, gamma: float = 2.0, lambda_aux: float = 0.4, lambda_fp: float = 0.25):
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
        return (-w * ((1.0 - pt) ** self.gamma) * torch.log(pt)).mean()

    def forward(self, outputs, targets: torch.Tensor):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        seg_logits = outputs["seg_logits"] if isinstance(outputs, dict) else outputs
        l_focal = self._focal(seg_logits, targets)
        l_dice = self.dice(seg_logits, targets)

        l_aux = torch.zeros((), device=seg_logits.device)
        if isinstance(outputs, dict) and outputs.get("aux_seg_logits") is not None:
            l_aux = 0.5 * self._focal(outputs["aux_seg_logits"], targets) + 0.5 * self.dice(outputs["aux_seg_logits"], targets)

        probs = torch.sigmoid(seg_logits)
        bg = 1.0 - targets
        l_fp = ((probs ** 2) * bg).sum() / bg.sum().clamp_min(1.0)
        total = l_focal + l_dice + self.lambda_aux * l_aux + self.lambda_fp * l_fp
        return total, {"loss_total": float(total.item())}


class CGMNCompositeLoss(nn.Module):
    def __init__(self, lambda_prov: float = 0.3, lambda_mask: float = 0.3, lambda_graph: float = 0.15, pos_weight: float = 6.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice = BinaryDiceLoss(from_logits=True)

    def forward(self, outputs, targets: torch.Tensor, ref_image: Optional[torch.Tensor] = None):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        logits = outputs["seg_logits"] if isinstance(outputs, dict) else outputs
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        dice = self.dice(logits, targets)
        total = bce + dice
        return total, {"loss_total": float(total.item())}


class BiGSAMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice(logits, targets)


BiGSAMLossV2 = BiGSAMLoss


# =============================================================================
# Builders
# =============================================================================


def build_model(arch: str, in_channels: int = 9, num_classes: int = 1, **kwargs: Any) -> nn.Module:
    arch = arch.lower()
    if arch in ["hrsei_gnn", "hrsei"]:
        return HRSEIGNNNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif arch in ["venagnosis_pfgnn", "venagnosis"]:
        return VenagnosisPFGNNNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif arch in ["gsri_net", "gsri"]:
        return GSRIRotationRobustNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif arch in ["cgmn_net", "cgmn"]:
        return CGMNMultiBranchNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif arch in ["big_sam"]:
        return BiGSAMNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif arch in ["big_sam_v2"]:
        return BiGSAMNetV2(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def build_loss(loss_name: str, **kwargs: Any) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name in ["venagnosis_loss", "venagnosis"]:
        return VenagnosisLoss(**kwargs)
    elif loss_name in ["cgmn_composite_loss", "cgmn_loss"]:
        return CGMNCompositeLoss(**kwargs)
    elif loss_name in ["binary_dice_loss", "dice_loss"]:
        return BinaryDiceLoss(**kwargs)
    elif loss_name in ["binary_focal_loss", "focal_loss"]:
        return BinaryFocalLoss(**kwargs)
    elif loss_name in ["big_sam_loss", "big_sam_loss_v2"]:
        return BiGSAMLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
