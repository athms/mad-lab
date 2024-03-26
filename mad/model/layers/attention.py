import typing as tp
from flash_attn.modules.mha import MHA


class Attention(MHA):
    """Wrapper for the Multi-Head Attention module from the `flash_attn` package."""
    def __init__(self,
        dim: int,
        causal: bool = True,
        n_heads: int = 16,
        rotary_emb_dim: float = 0.,
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
        *args, **kwargs
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


if __name__ == '__main__':
    import torch
    x = torch.rand(2,128,128).to(torch.bfloat16).cuda()

    # single headed:
    attn = Attention(dim=128, n_heads=1, dtype=torch.bfloat16).cuda()
    y = attn(x)
    assert x.shape==y.shape

    # multi headed:
    m_attn = Attention(dim=128, n_heads=16, dtype=torch.bfloat16).cuda()
    m_y = m_attn(x)
    assert x.shape==m_y.shape

    # sliding: 
    s_attn = Attention(dim=128, n_heads=16, window_size=(6,6), dtype=torch.bfloat16).cuda()
    s_y = s_attn(x)
    assert x.shape==s_y.shape