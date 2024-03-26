import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OptimModule(nn.Module):
    """ Interface for nn.Module that allows registering buffers/parameters
    with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable
        learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    def __init__(self, dim, w=10, w_mod=1, train_freq=True):
        super().__init__()

        init_tensor = torch.ones(1, dim)
        self.freq = (
            nn.Parameter(w * init_tensor)
            if train_freq
            else w * torch.ones(1, dim)
        )
        self.w_mod = w_mod

    def forward(self, x):
        return torch.sin(self.w_mod * self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
        self,
        d_model: int,
        emb_dim: int = 3,
        order: int = 16,
        seq_len: int = 1024,
        lr: float = 1e-3,
        lr_pos_emb: float = 1e-5,
        w: float = 1.,
        wd: float = 0.,
        depth_implicit: int = 2,
        nested: bool = False,
        modulate: bool = True,
        normalized: bool = False,
        *args, **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.modulate = modulate

        act = Sin(dim=order, w=w)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        if nested:
            self.implicit_filter = nn.Sequential(
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=order,
                    kernel_size=3,
                    groups=emb_dim,
                    padding=2,
                ),
                act,
                nn.Conv1d(
                    in_channels=order,
                    out_channels=d_model,
                    kernel_size=order,
                    groups=order,
                    padding=2,
                ),
            )
        else:                
            self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, order), act,)
            for i in range(depth_implicit):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)
            # final linear layer
            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        
        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        if normalized:
            self.post_modulation_norm = nn.LayerNorm(d_model)
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def forward(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            h = self.post_modulation_norm(h)

        h = rearrange(h, '1 d l -> 1 l d')
        return h