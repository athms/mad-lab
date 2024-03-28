import torch
import torch.nn.functional as F
import math
import typing as tp
import torch.nn as nn
from einops import rearrange

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

from mad.model.layers.featurization.hyena_filter import HyenaFilter, OptimModule
from mad.model.layers.featurization.rtf import RTF
from mad.model.layers.ops.fftconv import fftconv, fftconv_heads


# utils:

@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


# Hyena convolution operators:

class HyenaConv(OptimModule):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        filter_cls: str='rtf',
        filter_cfg: tp.Dict=None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.bias = nn.Parameter(torch.randn(self.d_model)) if self.use_bias else None

        filter_cfg = filter_cfg or {}
        if filter_cls=='implicit':
            filter = HyenaFilter
            filter_cfg['d_model'] = d_model
            filter_cfg['seq_len'] = seq_len
            self.filter = filter(**filter_cfg)
        elif filter_cls=='rtf':
            filter = RTF
            filter_cfg['d_model'] = d_model
            filter_cfg['trunc_len'] = seq_len
            self.filter = filter(**filter_cfg).get_k
        else:
            raise NotImplementedError(f'filter {filter_cls} not valid.')

    def forward(self, x, L, k=None, bias=None, k_rev=None, *args, **kwargs):
        if k is None:
            # Currently does not work if k is None as the filter
            # comes in L, D instead of D, L
            k = self.filter(L=L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        bias = self.bias if bias is None else bias
        bias = bias if self.use_bias else 0 * bias
        k = k.to(x.device)
        y = fftconv(x, k, bias)

        return y.to(dtype=x.dtype)
    
class MultiHeadHyenaConv(HyenaConv):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        num_heads: int,
        filter_cls: str='rtf',
        filter_cfg: tp.Dict=None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            seq_len=seq_len,
            filter_cls=filter_cls,
            filter_cfg=filter_cfg,
            bias=bias,
            **kwargs,
        )
        self.num_heads = num_heads

    def forward(self, v, x1, x2, L, k=None, bias=None, k_rev=None, *args, **kwargs):
        if k is None:
            # Currently does not work if k is None as the filter
            # comes in L, D instead of D, L
            k = self.filter(L=L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        bias = self.bias if bias is None else bias
        bias = bias if self.use_bias else 0 * bias
        y = fftconv_heads(
            v,
            k,
            bias,
            v=x2,
            head_dim=self.num_heads,
            q=x1,
        )

        return y.to(dtype=v.dtype)


# Hyena operators:

class HyenaOperator(nn.Module):
    def __init__(
        self,
        dim: int,
        max_length: int,
        order: int = 2,
        long_conv_cfg: tp.Dict = None,
        num_heads: int = 1,
        inner_factor: int = 2,
        proj_groups: int = 4,
        num_blocks: int = 1,
        fused_bias_fc: bool = False,
        outer_mixing: bool = False,
        nested: bool = False,
        dropout: float = 0.0,
        post_order_ffn: bool = False,
        short_filter_order: int = 3,
        activation: str = "id",
        *args, **kwargs
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            dim (int): Dimension of the input and output embeddings (width of the layer)
            max_length: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            long_conv_cfg: (dict): Long Conv config
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            proj_groups: (int): Number of tied projection groups. Defaults to 4
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            fused_bias_fc: (bool): Whether to use fused bias FC. Defaults to False
            dropout: (float): Dropout probability. Defaults to 0.0
            post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation: (str): type of act between kernel output and FF (default identity)
        """
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"Model dimension {dim} must be divisible by num heads {num_heads}"
        assert (
            max_length % num_blocks == 0
        ), f"Maximum signal length {max_length} must be divisible by block dimension {num_blocks}"
        self.inner_dim = inner_factor * dim
        self.head_dim = self.inner_dim // num_heads
        self.dim = dim
        self.max_length = max_length
        self.order = order
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.outer_mixing = outer_mixing
        self.post_order_ffn = post_order_ffn
        self.proj_groups = proj_groups
        long_conv_cfg = long_conv_cfg or {}

        # setup activation.
        if activation in [ None, 'id', 'identity', 'linear' ]:
            self.activation = nn.Identity()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation in ['swish', 'silu']:
            self.activation = nn.SiLU()
        elif activation == 'glu':
            self.activation = nn.GLU(dim=dim)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

        # setup short filters
        assert order >= 2, f"Order must be at least 2, (got {order})"
        total_width = self.inner_dim * (order + 1)
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=short_filter_order,
            groups=total_width,
            padding=short_filter_order - 1,
        )

        # setup long conv
        long_conv_cfg = long_conv_cfg or {}
        self.long_conv = HyenaConv(
            self.head_dim * (order - 1),
            max_length,
            nested=nested,
            **long_conv_cfg
        )

        # setup projections
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.out_proj = linear_cls(self.inner_dim, self.dim)
        self.in_proj = linear_cls(self.dim, self.inner_dim * (order + 1) // proj_groups)


    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.max_length)

        u = u @ self.in_proj.weight.t() + self.in_proj.bias
        if self.proj_groups > 1:
            u = u.repeat(1, 1, self.proj_groups)
        u = rearrange(u, "b l d -> b d l")

        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.inner_dim, dim=1)
        k = self.long_conv.filter(l_filter, device=u.device)

        k = k[0] if type(k) is tuple else k
        v = self.dropout(v * x[1])
        bias = self.long_conv.bias

        v = self.long_conv(v, l_filter, k=k, bias=bias, k_rev=None)
        y = v * x[0]
        y  = rearrange(y, "b d l -> b l d")
        y = self.out_proj(y)
        return y

    @property
    def d_output(self):
        return self.dim


class MultiHeadHyenaOperator(HyenaOperator):
    def __init__(
        self,
        dim: int,
        max_length: int,
        order: int = 2,
        long_conv_cfg: tp.Dict = None,
        num_heads: int = 1,
        inner_factor: int = 1,
        num_blocks: int = 1,
        fused_bias_fc: bool = False,
        outer_mixing: bool = False,
        dropout: float = 0.0,
        post_order_ffn: bool = False,
        short_filter_order: int = 3,
        activation: str = "id",
        layer_idx: int = None,
        *args, **kwargs
    ):
        super().__init__(
            dim=dim,
            max_length=max_length,
            order=order,
            long_conv_cfg=long_conv_cfg,
            num_heads=num_heads,
            inner_factor=inner_factor,
            num_blocks=num_blocks,
            fused_bias_fc=fused_bias_fc,
            outer_mixing=outer_mixing,
            dropout=dropout,
            post_order_ffn=post_order_ffn,
            short_filter_order=short_filter_order,
            activation=activation,
            *args, **kwargs
        )
        self.layer_idx = layer_idx

        # this double assigns as there is another call in super().__init__
        long_conv_cfg = long_conv_cfg or {}
        self.long_conv = MultiHeadHyenaConv(
            self.head_dim * (order - 1),
            max_length,
            **long_conv_cfg
        )

    def _update_kv_cache(self, u, inference_params):
        assert self.layer_idx is not None
        l = u.size(-2)
        l_filter = min(l, self.max_length)
        if self.layer_idx not in inference_params.key_value_memory_dict:
            u = self.in_proj(u)
            u = rearrange(u, "b l d -> b d l")
            if l >= l_filter:
                k = self.long_conv.filter(l_filter, device=u.device)
                k = k[0] if type(k) is tuple else k
                # `c` is always 1 by default
                k = rearrange(k, "c l v -> c v l", v=self.head_dim)[0].contiguous()
            else:
                k = None
            inference_params.key_value_memory_dict[self.layer_idx] = (u, k)
        else:
            u = self.in_proj(u)
            u = rearrange(u, "b 1 d -> b d 1")
            u_, k = inference_params.key_value_memory_dict[self.layer_idx]
            u = torch.cat((u_, u), dim=-1)
            if k is not None:
                k = self.long_conv.filter(l_filter, device=u.device)
                k = k[0] if type(k) is tuple else k
                # `c` is always 1 by default
                k = rearrange(k, "c l v -> c v l", v=self.head_dim)[0].contiguous()

        return u, k

    def forward(self, u, inference_params=None, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.max_length)

        if inference_params is not None:
            # if inference_params is passed then we expect u to have just a single element
            u, k = self._update_kv_cache(u, inference_params)
        else:
            u = u @ self.in_proj.weight.t() + self.in_proj.bias
            if self.proj_groups > 1:
                u = u.repeat(1, 1, self.proj_groups)
            u = rearrange(u, "b l d -> b d l")
            k = self.long_conv.filter(l_filter, device=u.device)
            k = k[0] if type(k) is tuple else k
        
        uc = self.short_filter(u)[..., :l_filter]
    
        x1, x2, v = uc.split(self.inner_dim, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        v = v.contiguous()

        y = self.long_conv(
            v,
            x1,
            x2,
            l_filter,
            k=k,
            bias=self.long_conv.bias,
        )
        
        y = rearrange(y, "b d l -> b l d")
        y = self.out_proj(y)
        return y


class HyenaExpertsOperator(HyenaOperator):
    def __init__(
        self,
        dim: int,
        dim_inner: int,
        max_length: int,
        order: int = 2,
        long_conv_cfg: tp.Dict = None,
        num_heads: int = 1,
        inner_factor: int = 1,
        num_blocks: int = 1,
        num_experts: int = 8,
        active_experts: int = 2,
        fused_bias_fc: bool = False,
        outer_mixing: bool = False,
        dropout: float = 0.0,
        post_order_ffn: bool = False,
        short_filter_order: int = 3,
        activation: str = "id",
        *args, **kwargs
    ):
        super().__init__(
            dim=dim,
            max_length=max_length,
            order=order,
            long_conv_cfg=long_conv_cfg,
            num_heads=num_heads,
            inner_factor=inner_factor,
            num_blocks=num_blocks,
            fused_bias_fc=fused_bias_fc,
            outer_mixing=outer_mixing,
            dropout=dropout,
            post_order_ffn=post_order_ffn,
            short_filter_order=short_filter_order,
            activation=activation,
            *args, **kwargs
        )

        self.num_experts = num_experts
        self.active_experts = active_experts
        self.dim_inner = dim_inner

        # replace in_proj, out_proj and short_filter
        self.in_proj = nn.Linear(self.dim, (order + 1) * self.dim_inner * self.num_experts)
        self.out_proj = nn.Linear(self.dim_inner, self.dim)
        self.short_filter = nn.Conv1d(
            in_channels=(order + 1) * self.dim_inner * self.num_experts,
            out_channels=(order + 1) * self.dim_inner * self.num_experts,
            kernel_size=short_filter_order,
            groups=(order + 1) * self.dim_inner * self.num_experts,
            padding=short_filter_order - 1,
        )

        self.router = nn.Linear(self.dim, self.num_experts)

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.max_length)
        u_pre = u
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")

        uc = self.short_filter(u)[..., :l_filter]

        uc = rearrange(
            uc,
            "b (ho v) (z l) -> b ho v z l",
            z=self.num_blocks,
            ho=self.num_heads,
            v=self.head_dim * (self.order + 1),
        )

        *x, v = uc.split(self.dim, dim=2)
        k = self.long_conv.filter(l_filter, device=u.device)
        k = k[0] if type(k) is tuple else k
        # `c` is always 1 by default
        k = rearrange(k, "c (v o) l -> c o v l", v=self.head_dim, o=self.order - 1)[0]
        
        bias = rearrange(
            self.long_conv.bias, "(v o) -> o v", v=self.head_dim, o=self.order - 1
        )

        for o, x_i in enumerate(reversed(x[1:])):
            if self.outer_mixing:
                v = rearrange(v, "b h v z l -> b h 1 v z l")
                v = self.dropout(v * rearrange(x_i, "b h v z l -> b h v 1 z l"))
                v = v.sum(dim=2)
            else:
                v = self.dropout(v * x_i)

            # the bias term is broadcasted. Last dimension (l) is handled by fftconv
            v = self.long_conv(v, l_filter, k=k[o], bias=bias[o, None, :, None], k_rev=None)
            if self.post_order_ffn:
                w = self.ord_proj_w[o]
                v = mul_sum(
                    rearrange(w, "h1 h2 -> 1 h1 h2 1 1 1"),
                    rearrange(v, "b h v z l -> b h 1 v z l"),
                )

        y = self.activation(
            rearrange(
                v * x[0],
                "b h v z l -> b (z l) (h v)",
                z=self.num_blocks,
                h=self.num_heads,
            )
        )
        scores = self.router(u_pre)
        # zero out non-topk experts
        topk_scores = scores.topk(self.active_experts, dim=-1)[0]
        min_score = topk_scores.min(dim=-1, keepdim=True)[0]
        scores = torch.where(scores < min_score, torch.zeros_like(scores), scores)
        scores = F.softmax(scores, dim=-1)
        y = y.reshape(y.shape[0], y.shape[1], self.dim_inner, self.num_experts)
        y = (y * scores.unsqueeze(-2)).sum(dim=-1)
        y = self.out_proj(y)
        return y