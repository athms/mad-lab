import typing as tp
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_ampere_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


_WARNED = set()
def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        warnings.warn(msg)
        _WARNED.add(key)


if _is_ampere_or_newer():
    # On Ampere or newer, use FlashAttention MHA directly.
    from flash_attn.modules.mha import MHA as _BackendMHA

    class Attention(_BackendMHA):
        """Wrapper for FlashAttention MHA (kept for interface compatibility)."""

        def __init__(
            self,
            dim: int,
            causal: bool = True,
            n_heads: int = 16,
            rotary_emb_dim: float = 0.0,
            dropout: float = 0.0,
            window_size: tp.Tuple[int, int] = (-1, -1),
            num_heads_kv: int = None,
            cross_attn: bool = False,
            qkv_proj_bias: bool = True,
            out_proj_bias: bool = True,
            softmax_scale: float = None,
            dwconv: bool = False,
            rotary_emb_base: float = 10000.0,
            rotary_emb_scale_base: float = None,
            rotary_emb_interleaved: bool = False,
            use_alibi: bool = False,
            fused_bias_fc: bool = False,
            use_flash_attn: bool = True,
            return_residual: bool = False,
            device=None,
            dtype=None,
            *args,
            **kwargs,
        ) -> None:
            super().__init__(
                embed_dim=dim,
                num_heads=n_heads,
                rotary_emb_dim=rotary_emb_dim,
                dropout=dropout,
                causal=causal,
                window_size=window_size,
                use_flash_attn=use_flash_attn,
                num_heads_kv=num_heads_kv,
                cross_attn=cross_attn,
                qkv_proj_bias=qkv_proj_bias,
                out_proj_bias=out_proj_bias,
                softmax_scale=softmax_scale,
                dwconv=dwconv,
                rotary_emb_base=rotary_emb_base,
                rotary_emb_scale_base=rotary_emb_scale_base,
                rotary_emb_interleaved=rotary_emb_interleaved,
                use_alibi=use_alibi,
                fused_bias_fc=fused_bias_fc,
                return_residual=return_residual,
                device=device,
                dtype=dtype,
            )


else:
    class Attention(nn.Module):
        """
        Fallback attention for pre-Ampere GPUs.

        Uses torch scaled_dot_product_attention (SDPA) and ignores FlashAttention-specific
        features like rotary embeddings and sliding window attention.

        Expected input: x of shape (B, T, D). Returns (B, T, D).
        """

        def __init__(
            self,
            dim: int,
            causal: bool = True,
            n_heads: int = 16,
            rotary_emb_dim: float = 0.0,
            dropout: float = 0.0,
            window_size: tp.Tuple[int, int] = (-1, -1),
            num_heads_kv: int = None,
            cross_attn: bool = False,
            qkv_proj_bias: bool = True,
            out_proj_bias: bool = True,
            softmax_scale: float = None,
            dwconv: bool = False,
            rotary_emb_base: float = 10000.0,
            rotary_emb_scale_base: float = None,
            rotary_emb_interleaved: bool = False,
            use_alibi: bool = False,
            fused_bias_fc: bool = False,
            use_flash_attn: bool = True,
            return_residual: bool = False,
            device=None,
            dtype=None,
            *args,
            **kwargs,
        ) -> None:
            super().__init__()
            if dim % n_heads != 0:
                raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")

            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.causal = causal
            self.dropout = float(dropout)

            # Warn about unsupported features on this backend
            if rotary_emb_dim and rotary_emb_dim > 0:
                _warn_once(
                    "rotary",
                    "Attention fallback is using torch SDPA and ignores rotary embeddings on pre-Ampere GPUs.",
                )
            if window_size != (-1, -1):
                _warn_once(
                    "window",
                    "Attention fallback ignores window_size (no sliding-window attention) on pre-Ampere GPUs.",
                )
            if cross_attn:
                _warn_once(
                    "cross",
                    "Attention fallback currently ignores cross_attn and runs self-attention only.",
                )
            if use_alibi:
                _warn_once(
                    "alibi",
                    "Attention fallback ignores ALiBi on pre-Ampere GPUs.",
                )
            if softmax_scale is not None:
                _warn_once(
                    "scale",
                    "Attention fallback ignores softmax_scale (uses default scaling).",
                )

            # Simple QKV projections + output projection
            factory_kwargs = {}
            if device is not None:
                factory_kwargs["device"] = device
            if dtype is not None:
                factory_kwargs["dtype"] = dtype

            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_proj_bias, **factory_kwargs)
            self.out = nn.Linear(dim, dim, bias=out_proj_bias, **factory_kwargs)

            # Keep signature compatibility even if unused
            self.return_residual = return_residual

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # x: (B, T, D)
            b, t, d = x.shape
            qkv = self.qkv(x)  # (B, T, 3D)
            q, k, v = qkv.chunk(3, dim=-1)

            # reshape to (B, H, T, Hd)
            q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

            # SDPA: returns (B, H, T, Hd)
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )

            # back to (B, T, D)
            attn = attn.transpose(1, 2).contiguous().view(b, t, d)
            y = self.out(attn)

            # Some codepaths in FA MHA support return_residual; MAD seems to call layer(x) only.
            # If your code expects a tuple in some places, adapt here.
            return y
