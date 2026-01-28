# Gated Delta Network (GDN) implementation for MAD-Lab benchmark
# Based on: https://github.com/sustcsonglin/flash-linear-attention
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Implements the gated delta rule: S_t = g*S_{t-1} + beta*k*v

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

# Lazy imports for fla - only loaded when GatedDeltaNet is instantiated
FusedRMSNormGated = None
RMSNorm = None
ShortConvolution = None
chunk_gated_delta_rule = None
fused_recurrent_gated_delta_rule = None

def _ensure_fla_imports():
    """Lazily import fla modules to avoid import errors when fla is not installed."""
    global FusedRMSNormGated, RMSNorm, ShortConvolution
    global chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    if FusedRMSNormGated is None:
        try:
            from fla.modules import FusedRMSNormGated as _FusedRMSNormGated
            from fla.modules import RMSNorm as _RMSNorm
            from fla.modules import ShortConvolution as _ShortConvolution
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _chunk_gated_delta_rule
            from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as _fused_recurrent_gated_delta_rule
            FusedRMSNormGated = _FusedRMSNormGated
            RMSNorm = _RMSNorm
            ShortConvolution = _ShortConvolution
            chunk_gated_delta_rule = _chunk_gated_delta_rule
            fused_recurrent_gated_delta_rule = _fused_recurrent_gated_delta_rule
        except ImportError as e:
            raise ImportError(
                "GatedDeltaNet requires the 'flash-linear-attention' (fla) package. "
                "Install it with: pip install flash-linear-attention"
            ) from e


class GatedDeltaNet(nn.Module):
    """
    Gated Delta Network layer adapted for MAD-Lab benchmark.
    
    Based on: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
    https://arxiv.org/abs/2412.06464
    
    Implements the gated delta rule: S_t = g * S_{t-1} + beta * k_t * v_t^T
    where g is a learnable decay and beta controls the update strength.
    
    Args:
        dim (int): Model dimension (hidden_size). Required by MAD-Lab.
        max_length (int): Maximum sequence length (unused but required by MAD-Lab).
        expand_v (float): Expansion ratio for value dimension. Default: 2.0.
        head_dim (int): Dimension of each head for Q/K. Default: 32.
        num_heads (int): Number of attention heads. Default: 4.
        num_v_heads (int): Number of value heads (for GVA). Default: None (same as num_heads).
        mode (str): Kernel mode - 'chunk' or 'fused_recurrent'. Default: 'chunk'.
        use_gate (bool): Whether to use output gate. Default: True.
        use_short_conv (bool): Whether to use short convolutions. Default: True.
        conv_size (int): Short convolution kernel size. Default: 4.
        conv_bias (bool): Use bias in short convolution. Default: False.
        allow_neg_eigval (bool): Allow negative eigenvalues (beta * 2). Default: False.
        norm_eps (float): Epsilon for normalization layers. Default: 1e-5.
    """

    def __init__(
        self,
        dim: int = 128,
        max_length: int = 1024,
        expand_v: float = 2.0,
        head_dim: int = 32,
        num_heads: int = 4,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        *args, **kwargs
    ) -> GatedDeltaNet:
        super().__init__()
        
        # Ensure fla imports are available
        _ensure_fla_imports()

        self.hidden_size = dim
        self.mode = mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.expand_v = expand_v

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        # Consistency checks
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        # Projections
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.a_proj = nn.Linear(dim, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(dim, self.num_v_heads, bias=False)

        # Learnable decay parameters
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # dt (timestep) initialization
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Short convolutions for local context
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=conv_bias, activation='silu')
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off unless you know what you are doing."
            )

        # Output gating and normalization
        if use_gate:
            self.g_proj = nn.Linear(dim, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, seq_len, dim) -> (batch, seq_len, dim)"""

        # Apply short convolutions or SiLU
        if self.use_short_conv:
            q, _ = self.q_conv1d(self.q_proj(x), cache=None, output_final_state=False)
            k, _ = self.k_conv1d(self.k_proj(x), cache=None, output_final_state=False)
            v, _ = self.v_conv1d(self.v_proj(x), cache=None, output_final_state=False)
        else:
            q = F.silu(self.q_proj(x))
            k = F.silu(self.k_proj(x))
            v = F.silu(self.v_proj(x))

        # Reshape: (B, T, H*D) -> (B, T, H, D)
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        # Handle GVA: repeat q, k if num_v_heads > num_heads
        if self.num_v_heads > self.num_heads:
            q = repeat(q, 'b t h d -> b t (h g) d', g=self.num_v_heads // self.num_heads)
            k = repeat(k, 'b t h d -> b t (h g) d', g=self.num_v_heads // self.num_heads)

        # Beta: update strength for delta rule
        beta = self.b_proj(x).sigmoid()  # (B, T, num_v_heads)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Decay gate: g = -A * softplus(a_proj(x) + dt_bias)
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)

        # Apply gated delta rule kernel
        if self.mode == 'chunk':
            o, _ = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            o, _ = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        # Output gating and projection
        if self.use_gate:
            g_out = rearrange(self.g_proj(x), 'b t (h d) -> b t h d', d=self.head_v_dim)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')
        return self.o_proj(o)
