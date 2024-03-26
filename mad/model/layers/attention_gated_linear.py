# based on the recurrent GLA implementation provided at:
# https://github.com/sustcsonglin/flash-linear-attention

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN

from mad.model.layers.ops.norm.fused_norm_gate import FusedRMSNormSwishGate
from mad.model.layers.ops.norm.rmsnorm import RMSNorm
from mad.model.layers.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


class GatedLinearAttention(nn.Module):

    def __init__(
        self,
        dim: int = 1024,
        expand_v: float = 2.0,
        expand_k: float = 1.0,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        layernorm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        mode: str = 'fused_chunk',
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        *args, **kwargs
    ) -> GatedLinearAttention:
        super().__init__()
        self.d_model = dim

        self.mode = mode
        self.value_dim = int(self.d_model * expand_v)
        self.key_dim = int(self.d_model * expand_k)
        self.clamp_min = clamp_min

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.num_heads = num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_fn = ACT2FN[gate_fn]

        self.q_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.value_dim, bias=False)

        self.gk_proj = nn.Sequential(nn.Linear(self.d_model,  gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim, bias=True))
        self.o_proj = nn.Linear(self.value_dim, self.d_model, bias=False)

        if (gate_fn == 'swish') and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, eps=layernorm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)

        self.gate_logit_normalizer = gate_logit_normalizer

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[1].weight, gain=2 ** -2.5)

    def forward(self, x):
        mode = self.mode

        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        gk = rearrange(self.gk_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        gk = (F.logsigmoid(gk) / self.gate_logit_normalizer)

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)
        if mode == 'fused_recurrent':
            o = fused_recurrent_gla(q, k, v, gk, None)
        elif mode == 'fused_chunk':
            o = fused_chunk_gla(q, k, v, gk)
        elif mode == 'chunk':
            o = chunk_gla(q, k, v, gk)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = rearrange(o, 'b h l d -> b l h d')
        g = self.g_proj(x)

        if self.fuse_norm_and_gate:
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            o = self.g_norm(o)
            o = rearrange(o, 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o