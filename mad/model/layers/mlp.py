import torch
import typing as tp
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """
    Multi-layer perceptron (MLP).
    
    Args:
        dim (int): Outer width.
        dim_inner (int, optional): Inner width.
        drop_rate (float, optional): Dropout rate.
        act (tp.Callable, optional): Activation function.
        bias (bool, optional): If True, bias is included in linear projections.
    """
    def __init__(self,
        dim: int,
        dim_inner: int = None,
        drop_rate: float = 0.,
        act: tp.Callable = nn.GELU(approximate='tanh'),
        bias: bool=True,
        *args, **kwargs
    ) -> None:
        super().__init__()

        dim_inner = dim*4 if dim_inner is None else dim_inner
        self.fc = nn.Linear(dim, dim_inner, bias=bias)
        self.proj = nn.Linear(dim_inner, dim, bias=bias)
        self.act = act
        self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.proj(x)
        x = self.drop2(x)
        return x


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU).

    Args:
        dim (int): Width of the model.
        drop_rate (float, optional): Dropout rate.
        act (tp.Callable, optional): Activation function for the gate.
        bias (bool, optional): If True, bias is included in linear projections.
        multiple_of (int, optional): Make suer inner width is multiple of this.
    """
    def __init__(
        self,
        dim,
        drop_rate = 0.,
        act = nn.Sigmoid(),
        bias = False,
        multiple_of = 16,
        *args, **kwargs
    ):
        super().__init__()

        self.act = act
        self.multiple_of = multiple_of

        dim_inner = int(2 * dim * 4 / 3)
        dim_inner = self.multiple_of * ((dim_inner + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, dim_inner, bias=bias)
        self.w2 = nn.Linear(dim, dim_inner, bias=bias)
        self.w3 = nn.Linear(dim_inner, dim, bias=bias)

        self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop3 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        w1_out = self.w1(x)
        w2_out = self.w2(x)
        return self.drop3( self.w3( self.drop1( self.act(w1_out) ) * self.drop2(w2_out) ) )


class SwiGLU(GLU):
    """
    Swish-Gated Linear Unit (SwiGLU).

    Args:
        dim (int): Width of the model.
        drop_rate (float, optional): Dropout rate.
        bias (bool, optional): If True, bias is included in linear projections.
        multiple_of (int, optional): Make suer inner width is multiple of this.
    """
    def __init__(self,
        dim,
        drop_rate = 0.,
        bias = False,
        multiple_of = 16,
        *args, **kwargs
    ) -> None:
        super().__init__(
            dim=dim,
            drop_rate=drop_rate,
            act=nn.SiLU(),
            bias=bias,
            multiple_of=multiple_of,
        )


class MoeMlp(nn.Module):
    """
    Mixture of Experts (MoE) GLU.
    
    Args:
        dim (int): Outter width.
        num_experts (int): Number of experts.
        active_experts (int): Number of active experts.
        dim_inner (int, optional): Inner width.
        drop_rate (float, optional): Dropout rate.
        act (tp.Callable, optional): Activation function.
        bias (bool, optional): If True, bias is included in linear projections.
    """
    def __init__(self,
        dim: int,
        num_experts: int,
        active_experts: int,
        dim_inner: int = None,
        drop_rate: float = 0.,
        act: tp.Callable = nn.GELU(approximate='tanh'),
        bias: bool=True,
        *args, **kwargs
    ) -> None:
        super().__init__()

        dim_inner = dim if dim_inner is None else dim_inner
        self.dim_inner = dim_inner
        self.dim = dim

        self.up1 = nn.Linear(dim, dim_inner * num_experts, bias=bias)
        self.up2 = nn.Linear(dim, dim_inner * num_experts, bias=bias)
        self.down1 = nn.Linear(dim_inner * num_experts, dim * num_experts, bias=bias)
        self.act = act
        self.num_experts = num_experts
        self.active_experts = active_experts

        self.router = nn.Linear(dim, num_experts)
        self.gates = nn.Parameter(torch.randn(num_experts, dim_inner))

        self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop3 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x1, x2 = self.drop1(self.up1(x)), self.up2(x)
        z = self.drop3(self.down1(x1 * self.drop2(self.act(x2))))

        # b, l, num_experts
        scores = self.router(x)
        
        # zero out non-topk experts
        topk_scores = scores.topk(self.active_experts, dim=-1)[0]
        min_score = topk_scores.min(dim=-1, keepdim=True)[0]
        scores = torch.where(scores < min_score, torch.zeros_like(scores), scores)
        scores = F.softmax(scores, dim=-1)
        
        z = z.reshape(z.shape[0], z.shape[1], self.dim, self.num_experts)

        return (z * scores.unsqueeze(-2)).sum(dim=-1)