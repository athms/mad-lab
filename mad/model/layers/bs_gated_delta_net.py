# Basis-Subspace Gated Delta Network (BS-GDN) implementation for MAD-Lab benchmark
# Based on: https://github.com/sustcsonglin/flash-linear-attention
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Extends GDN with basis transformations (Hadamard, DCT) and subhead decomposition

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from torch.nn import functional as F

# Lazy imports for fla - only loaded when BSGatedDeltaNet is instantiated
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
                "BSGatedDeltaNet requires the 'flash-linear-attention' (fla) package. "
                "Install it with: pip install flash-linear-attention"
            ) from e


class BSGatedDeltaNet(nn.Module):
    """
    Basis-Subspace Gated Delta Network (BS-GDN) layer for MAD-Lab benchmark.
    
    The key innovation is splitting each head into multiple subheads (subspaces),
    applying basis transformations (Hadamard, DCT) to Q and K, then summing
    the outputs from each subhead.
    
    Args:
        dim (int): Model dimension (hidden_size).
        max_length (int): Maximum sequence length (unused but required by MAD-Lab interface).
        expand_v (float): Expansion ratio for value dimension. Default: 2.0.
        head_dim (int): Dimension of each head for Q/K. Default: 32.
        num_heads (int): Number of attention heads. Default: 4.
        mode (str): Kernel mode - 'chunk' or 'fused_recurrent'. Default: 'chunk'.
        use_gate (bool): Whether to use output gate. Default: True.
        use_short_conv (bool): Whether to use short convolutions. Default: True.
        conv_size (int): Short convolution kernel size. Default: 4.
        conv_bias (bool): Use bias in short convolution. Default: False.
        norm_eps (float): Epsilon for normalization layers. Default: 1e-5.
        bs_basis (str): Basis type - 'hadamard', 'dct', 'identity', 'random', 'learned'. Default: 'hadamard'.
        bs_subheads (int): Number of subheads per head. Default: 1.
        allow_neg_eigval (bool): Allow negative eigenvalues (multiplies beta by 2). Default: False.
    """

    def __init__(
        self,
        dim: int = 128,
        max_length: int = 1024,
        expand_v: float = 2.0,
        head_dim: int = 32,
        num_heads: int = 4,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        bs_basis: str = 'hadamard',
        bs_subheads: int = 1,
        allow_neg_eigval: bool = False,
        *args, **kwargs
    ) -> BSGatedDeltaNet:
        super().__init__()
        
        # Ensure fla imports are available
        _ensure_fla_imports()

        self.hidden_size = dim
        self.mode = mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.bs_basis = bs_basis
        self.bs_subheads = bs_subheads
        self.allow_neg_eigval = allow_neg_eigval

        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim

        # Subhead dimension
        if self.head_k_dim % bs_subheads != 0:
            raise ValueError(
                f"head_k_dim={self.head_k_dim} must be divisible by bs_subheads={bs_subheads}."
            )
        self.subhead_k_dim = self.head_k_dim // bs_subheads

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        # Projections
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        
        # Beta projection - outputs for each head * subhead
        self.b_proj = nn.Linear(dim, num_heads * bs_subheads, bias=False)
        
        # Alpha projection for input-dependent decay
        self.a_proj = nn.Linear(dim, num_heads * bs_subheads, bias=False)

        # Initialize basis transformation matrix
        self._init_basis()

        # Learnable decay parameters (per head * subhead)
        A = torch.empty(num_heads * bs_subheads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(num_heads * bs_subheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # Short convolutions
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
        self.reset_parameters()

    def _init_basis(self):
        """Initialize the basis transformation matrix."""
        if self.bs_basis == 'hadamard':
            from scipy.linalg import hadamard
            B = torch.from_numpy(hadamard(self.head_k_dim).astype('float32'))
            B = B / (self.head_k_dim ** 0.5)
            self.register_buffer('B', B.unsqueeze(0).unsqueeze(0))  # (1, 1, Dk, Dk)
        elif self.bs_basis == 'dct':
            # DCT will be initialized lazily in forward to avoid meta tensor issues
            self.register_buffer('B', None)
        elif self.bs_basis == 'identity':
            B = torch.eye(self.head_k_dim)
            self.register_buffer('B', B.unsqueeze(0).unsqueeze(0))
        elif self.bs_basis == 'random':
            B = torch.randn(self.head_k_dim, self.head_k_dim) / (self.head_k_dim ** 0.5)
            self.register_buffer('B', B.unsqueeze(0).unsqueeze(0))
        elif self.bs_basis == 'learned':
            # Learnable basis - register as parameter, not buffer
            B = nn.Parameter(torch.randn(self.head_k_dim, self.head_k_dim) / (self.head_k_dim ** 0.5))
            B._no_weight_decay = True
            self.B = B.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported bs_basis: {self.bs_basis}")

    def _build_dct_basis(self, device, dtype):
        """Build DCT basis matrix lazily."""
        n = self.head_k_dim
        dct_mtx = np.zeros((n, n), dtype=np.float32)
        for k in range(n):
            for i in range(n):
                coeff = (1 / math.sqrt(n)) if k == 0 else math.sqrt(2 / n)
                dct_mtx[k, i] = coeff * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
        B = torch.from_numpy(dct_mtx).to(device=device, dtype=dtype) / (self.head_k_dim ** 0.5)
        return B.unsqueeze(0).unsqueeze(0)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.b_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.a_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)
        if self.use_gate:
            nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)

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
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)  # (B, T, H, Dk)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_k_dim)  # (B, T, H, Dk)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)  # (B, T, H, Dv)

        # Get basis matrix (lazy init for DCT)
        if self.bs_basis == 'dct' and (self.B is None or self.B.is_meta):
            self.B = self._build_dct_basis(k.device, k.dtype)
        
        B = self.B.to(dtype=k.dtype, device=k.device)
        
        # Apply basis transformation: k @ B, q @ B.T (as per reference)
        k = k @ B  # (B, T, H, Dk)
        q = q @ B.transpose(-1, -2)  # (B, T, H, Dk)

        # Handle subheads: split head dimension into subheads
        if self.subhead_k_dim < 64:
            # Avoid Triton autotune kernels that are unstable for very small K
            # Repeat q, k along head dimension
            q = repeat(q, 'b t h d -> b t (h g) d', g=self.bs_subheads)  # (B, T, H*G, Dk)
            k = repeat(k, 'b t h d -> b t (h g) d', g=self.bs_subheads)  # (B, T, H*G, Dk)
        else:
            # View as subheads
            q = q.view(*q.shape[:-2], self.num_heads * self.bs_subheads, self.subhead_k_dim)
            k = k.view(*k.shape[:-2], self.num_heads * self.bs_subheads, self.subhead_k_dim)
        
        # Replicate v for each subhead
        v = v.unsqueeze(-2).repeat(*([1] * (v.dim() - 1)), self.bs_subheads, 1)  # (B, T, H, G, Dv)
        v = v.reshape(*v.shape[:-3], self.num_heads * self.bs_subheads, self.head_v_dim)  # (B, T, H*G, Dv)

        # Beta: update strength for delta rule (per head * subhead)
        beta = self.b_proj(x).sigmoid()  # (B, T, H*G)
        
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Decay gate (per head * subhead)
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

        # Output gating with subhead handling
        if self.use_gate:
            g_out = rearrange(self.g_proj(x), 'b t (h d) -> b t h d', d=self.head_v_dim)  # (B, T, H, Dv)
            # Replicate gate for subheads
            g_out = g_out.unsqueeze(-2).repeat(1, 1, 1, self.bs_subheads, 1)  # (B, T, H, G, Dv)
            g_out = g_out.reshape(*g_out.shape[:-3], self.num_heads * self.bs_subheads, self.head_v_dim)
            o = self.o_norm(o, g_out)  # (B, T, H*G, Dv)
        else:
            o = self.o_norm(o)

        # Sum over subheads
        o = o.view(*o.shape[:-2], self.num_heads, self.bs_subheads, self.head_v_dim)  # (B, T, H, G, Dv)
        o = o.sum(-2)  # (B, T, H, Dv)

        o = rearrange(o, 'b t h d -> b t (h d)')
        return self.o_proj(o)
