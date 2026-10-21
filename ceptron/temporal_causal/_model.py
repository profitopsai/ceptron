"""The TCPFN network (generated; discovery serving only)."""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE = 'joint_panel'
STANDARDIZATION = 'per_column_zscore_clip10'
_CLIP = 10.0

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight, self.eps).to(x.dtype)

def _rope_cache(n: int, dim: int, device, base: float=10000.0):
    inv = 1.0 / base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    t = torch.arange(n, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos(), emb.sin())

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)

class _FeedForward(nn.Module):

    def __init__(self, d: int, ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, ff)
        self.w2 = nn.Linear(ff, d)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class _Attention(nn.Module):

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads:
            raise ValueError(f'd_model {d} not divisible by n_heads {n_heads}')
        self.hd = d // n_heads
        if self.hd % 2:
            raise ValueError('head dim must be even for rotary embedding')
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, *, causal: bool, rope=None, key_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        N, S, d = x.shape
        q, k, v = self.qkv(x).view(N, S, 3, self.n_heads, self.hd).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            cos, sin = rope
            q = _apply_rope(q, cos[:S], sin[:S])
            k = _apply_rope(k, cos[:S], sin[:S])
        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
        return self.out(y.transpose(1, 2).reshape(N, S, d))

class _Block(nn.Module):

    def __init__(self, d: int, n_heads: int, ff: int, axis: str):
        super().__init__()
        if axis not in ('time', 'variate'):
            raise ValueError(axis)
        self.axis = axis
        self.n1 = RMSNorm(d)
        self.attn = _Attention(d, n_heads)
        self.n2 = RMSNorm(d)
        self.ff = _FeedForward(d, ff)

    def forward(self, h: torch.Tensor, var_mask: torch.Tensor, rope) -> torch.Tensor:
        B, V, L, d = h.shape
        if self.axis == 'time':
            x = h.reshape(B * V, L, d)
            x = x + self.attn(self.n1(x), causal=True, rope=rope)
            x = x + self.ff(self.n2(x))
            return x.view(B, V, L, d)
        x = h.transpose(1, 2).reshape(B * L, V, d)
        km = var_mask[:, None, :].expand(B, L, V).reshape(B * L, V)
        x = x + self.attn(self.n1(x), causal=False, key_mask=km)
        x = x + self.ff(self.n2(x))
        return x.view(B, L, V, d).transpose(1, 2)

def combine_link_direction(link: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    lp = (F.logsigmoid(link) + F.logsigmoid(direction)).clamp(max=-1e-07)
    log1m = torch.where(lp > -0.6931, torch.log(-torch.expm1(lp)), torch.log1p(-torch.exp(lp)))
    return lp - log1m

class JointPanelScorer(nn.Module):

    def __init__(self, d_model: int=256, n_heads: int=8, n_layers: int=8, ff_factor: int=4, patch_len: int=8, max_lag: int=10, max_patches: int=512, pair_head: str='pooled', cross_dim: int=32, cross_groups: int=8, cross_shifts: int=2, edge_head: str='joint', time_gap_input: bool=False, categorical_input: bool=False, n_col_types: int=4, graph_head: bool=False):
        super().__init__()
        if edge_head not in ('joint', 'factorized'):
            raise ValueError(f"edge_head must be 'joint' or 'factorized', got {edge_head!r}")
        if pair_head not in ('pooled', 'cross_time'):
            raise ValueError(f"pair_head must be 'pooled' or 'cross_time', got {pair_head!r}")
        if pair_head == 'cross_time' and cross_dim % cross_groups:
            raise ValueError('cross_dim must be divisible by cross_groups')
        if n_layers % 2:
            raise ValueError('n_layers must be even: layers alternate time / variate')
        self.patch_len, self.max_patches = (patch_len, max_patches)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.time_gap_input = bool(time_gap_input)
        self.categorical_input = bool(categorical_input)
        self.n_col_types = int(n_col_types)
        if self.time_gap_input:
            self.gap_embed = nn.Linear(1, d_model)
        if self.categorical_input:
            self.coltype_embed = nn.Embedding(self.n_col_types, d_model)
        self.layers = nn.ModuleList((_Block(d_model, n_heads, ff_factor * d_model, 'time' if i % 2 == 0 else 'variate') for i in range(n_layers)))
        self.final_norm = RMSNorm(d_model)
        self.pool = nn.Linear(2 * d_model, d_model)
        self.src_proj = nn.Linear(d_model, d_model)
        self.dst_proj = nn.Linear(d_model, d_model)
        self.pair_head = pair_head
        self.cross_shifts, self.cross_groups = (int(cross_shifts), int(cross_groups))
        n_cross = 0
        if pair_head == 'cross_time':
            self.cross_q = nn.Linear(d_model, cross_dim, bias=False)
            self.cross_k = nn.Linear(d_model, cross_dim, bias=False)
            n_cross = (2 * self.cross_shifts + 1) * self.cross_groups
            self.cross_norm = nn.LayerNorm(n_cross)
        self.n_cross = n_cross
        self.edge_mlp = nn.Sequential(nn.Linear(3 * d_model + n_cross, d_model), nn.GELU(), nn.Linear(d_model, 1 + max_lag + 1))
        self.graph_head = bool(graph_head)
        if self.graph_head:
            self.any_edge_out = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.edge_head = edge_head
        if edge_head == 'factorized':
            self.link_out = nn.Linear(d_model, 1)
        self.model_config = {'model_type': MODEL_TYPE, 'model': {'d_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers, 'ff_factor': ff_factor, 'patch_len': patch_len, 'max_lag': max_lag, 'max_patches': max_patches, 'pair_head': pair_head, **({'cross_dim': cross_dim, 'cross_groups': cross_groups, 'cross_shifts': cross_shifts} if pair_head == 'cross_time' else {}), **({'edge_head': edge_head} if edge_head != 'joint' else {}), **({'time_gap_input': True} if self.time_gap_input else {}), **({'categorical_input': True, 'n_col_types': self.n_col_types} if self.categorical_input else {}), **({'graph_head': True} if self.graph_head else {})}, 'standardization': STANDARDIZATION}
        self._rope: dict = {}

    @staticmethod
    def standardize_panel(x: torch.Tensor) -> torch.Tensor:
        xf = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = xf.mean(dim=1, keepdim=True)
        std = xf.std(dim=1, keepdim=True, unbiased=False)
        z = (xf - mean) / (std + 1e-06)
        z = torch.where(std > 1e-08, z, torch.zeros_like(z))
        return z.clamp_(-_CLIP, _CLIP)

    def _rope_for(self, L: int, device):
        key = (L, str(device))
        if key not in self._rope:
            self._rope[key] = _rope_cache(max(L, 1), self.layers[0].attn.hd, device)
        return self._rope[key]

    def encode(self, x: torch.Tensor, var_mask: Optional[torch.Tensor]=None, time_gap: Optional[torch.Tensor]=None, col_type: Optional[torch.Tensor]=None) -> torch.Tensor:
        B, T, V = x.shape
        P = self.patch_len
        if T % P:
            raise ValueError(f'T={T} is not a multiple of patch_len={P}; call fit_time first')
        L = T // P
        if L > self.max_patches:
            raise ValueError(f'{L} patches exceed max_patches={self.max_patches}')
        if var_mask is None:
            var_mask = torch.ones(B, V, dtype=torch.bool, device=x.device)
        if col_type is None and self.categorical_input:
            col_type = torch.zeros(B, V, dtype=torch.long, device=x.device)
        h = x.transpose(1, 2).reshape(B, V, L, P)
        h = self.patch_embed(h)
        if time_gap is not None:
            if not self.time_gap_input:
                raise ValueError('time_gap passed but the model was built with time_gap_input=False')
            g = time_gap
            if g.dim() == 2:
                g = g.unsqueeze(-1).expand(B, T, V)
            gp = g.transpose(1, 2).reshape(B, V, L, P).mean(dim=-1, keepdim=True)
            h = h + self.gap_embed(gp)
        if col_type is not None:
            if not self.categorical_input:
                raise ValueError('col_type passed but the model was built with categorical_input=False')
            ct = self.coltype_embed(col_type.long())
            h = h + ct.unsqueeze(2)
        rope = self._rope_for(L, x.device)
        for layer in self.layers:
            h = layer(h, var_mask, rope)
        return self.final_norm(h)

    def pool_variates(self, h: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.cat([h.mean(dim=2), h[:, :, -1, :]], dim=-1))

    def _cross_features(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, V, L, G, r = q.shape
        S = self.cross_shifts
        kp = F.pad(k, (0, 0, 0, 0, S, S))
        kw = kp.unfold(2, L, 1)
        prod = torch.einsum('bitgr,bjsgrt->bijsg', q, kw)
        shifts = torch.arange(-S, S + 1, device=q.device)
        n = (L - shifts.abs()).clamp(min=1).to(prod.dtype)
        prod = prod / (n[:, None] * r ** 0.5)
        return self.cross_norm(prod.reshape(B, V, V, -1).float()).to(q.dtype)

    def _pair_terms(self, z: torch.Tensor, h: Optional[torch.Tensor]):
        B, V, d = z.shape
        u = self.src_proj(z)
        v = self.dst_proj(z)
        cross = None
        if self.pair_head == 'cross_time':
            if h is None:
                raise ValueError('the cross_time pair head needs the per-patch representations h')
            Bh, Vh, L, _ = h.shape
            G = self.cross_groups
            q, k = (self.cross_q(h).view(Bh, Vh, L, G, -1), self.cross_k(h).view(Bh, Vh, L, G, -1))
            cross = self._cross_features(q, k)
        first = self.edge_mlp[0]
        w = first.weight
        w_u, w_v, w_uv = (w[:, :d], w[:, d:2 * d], w[:, 2 * d:3 * d])
        w_c = w[:, 3 * d:3 * d + self.n_cross]
        row = F.linear(u, w_u, first.bias)
        col = F.linear(v, w_v)
        return (u, v, row, col, cross, w_uv, w_c)

    def _pair_hidden(self, u_rows, v, row_rows, col, cross_rows, w_uv, w_c):
        pre = F.linear(u_rows[:, :, None, :] * v[:, None, :, :], w_uv)
        pre = pre + row_rows[:, :, None, :] + col[:, None, :, :]
        if cross_rows is not None:
            pre = pre + F.linear(cross_rows.to(u_rows.dtype), w_c)
        return self.edge_mlp[1](pre)

    def score_pairs_all(self, z: torch.Tensor, pair_chunk: Optional[int]=None, h: Optional[torch.Tensor]=None, var_mask: Optional[torch.Tensor]=None) -> dict:
        B, V, d = z.shape
        u, v, row, col, cross, w_uv, w_c = self._pair_terms(z, h)
        if pair_chunk is None:
            if torch.is_grad_enabled() and z.requires_grad:
                pair_chunk = V
            else:
                pair_chunk = max(1, int(64000000 // max(1, B * V * (3 * d + self.n_cross))))
        last = self.edge_mlp[2]
        edge_parts, lag_parts, link_parts = ([], [], [])
        if self.graph_head:
            vm = var_mask.bool() if var_mask is not None else torch.ones(B, V, dtype=torch.bool, device=z.device)
            g_sum = torch.zeros(B, d, dtype=torch.float32, device=z.device)
            g_cnt = torch.zeros(B, 1, dtype=torch.float32, device=z.device)
            g_max = torch.full((B, d), float('-inf'), dtype=torch.float32, device=z.device)
            idx = torch.arange(V, device=z.device)
        for s in range(0, V, pair_chunk):
            e = min(V, s + pair_chunk)
            hid = self._pair_hidden(u[:, s:e], v, row[:, s:e], col, cross[:, s:e] if cross is not None else None, w_uv, w_c)
            out = last(hid)
            edge_parts.append(out[..., 0])
            lag = out[..., 1:]
            lag_parts.append(lag)
            if self.edge_head == 'factorized':
                link_parts.append(self.link_out(hid)[..., 0])
            if self.graph_head:
                m = vm[:, s:e, None] & vm[:, None, :] & (idx[s:e, None] != idx[None, :])[None]
                hf = hid.float()
                g_sum = g_sum + (hf * m[..., None]).sum(dim=(1, 2))
                g_cnt = g_cnt + m.sum(dim=(1, 2)).float()[:, None]
                g_max = torch.maximum(g_max, hf.masked_fill(~m[..., None], float('-inf')).amax(dim=(1, 2)))
        edge = torch.cat(edge_parts, dim=1)
        lag = torch.cat(lag_parts, dim=1)
        eye = torch.eye(V, dtype=torch.bool, device=z.device)
        res = {'lag_logits': lag}
        if self.graph_head:
            g_max = torch.where(torch.isfinite(g_max), g_max, torch.zeros_like(g_max))
            feat = torch.cat([g_sum / g_cnt.clamp(min=1.0), g_max], dim=-1).to(z.dtype)
            res['any_edge_logit'] = self.any_edge_out(feat)[..., 0]
        if self.edge_head == 'factorized':
            a = torch.cat(link_parts, dim=1)
            link = 0.5 * (a + a.transpose(1, 2))
            res['link_logits'] = link.masked_fill(eye[None], -10000.0)
            res['direction_logits'] = edge.masked_fill(eye[None], -10000.0)
            edge = combine_link_direction(link.float(), edge.float()).to(edge.dtype)
        res['edge_logits'] = edge.masked_fill(eye[None], -10000.0)
        return res

    def encode_pool(self, x: torch.Tensor, var_mask: Optional[torch.Tensor]=None, time_gap: Optional[torch.Tensor]=None, col_type: Optional[torch.Tensor]=None):
        h = self.encode(x, var_mask, time_gap=time_gap, col_type=col_type)
        return (self.pool_variates(h), h)

    @classmethod
    def from_config(cls, model_config: dict) -> 'JointPanelScorer':
        if model_config.get('model_type') != MODEL_TYPE:
            raise ValueError(f"not a {MODEL_TYPE} checkpoint: model_type={model_config.get('model_type')!r}")
        m = model_config['model']
        return cls(d_model=m['d_model'], n_heads=m['n_heads'], n_layers=m['n_layers'], ff_factor=m['ff_factor'], patch_len=m['patch_len'], max_lag=m['max_lag'], max_patches=m.get('max_patches', 512), pair_head=m.get('pair_head', 'pooled'), cross_dim=m.get('cross_dim', 32), cross_groups=m.get('cross_groups', 8), cross_shifts=m.get('cross_shifts', 2), edge_head=m.get('edge_head', 'joint'), time_gap_input=bool(m.get('time_gap_input', False)), categorical_input=bool(m.get('categorical_input', False)), n_col_types=int(m.get('n_col_types', 4)), graph_head=bool(m.get('graph_head', False)))

    @classmethod
    def load(cls, state_dict: dict, model_config: dict) -> 'JointPanelScorer':
        model = cls.from_config(model_config)
        clean = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean, strict=True)
        return model
