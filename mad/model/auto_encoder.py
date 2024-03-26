import torch
import typing as tp
from torch import nn
from einops import rearrange

from mad.model.layers.featurization.posemb import posemb_sincos_1d
from mad.model.layers.ops.norm.rmsnorm import RMSNorm


class AutoEncoder(nn.Module):
    """
    AutoEncoder model backbone.

    Args:
        dim (int): Width of the model.
        layers (list): List of layer modules.
        layer_cfgs (list): List of layer configs.
        vocab_size (int): Size of the vocabulary.
        global_pool (str, optional): Global pooling strategy (one of 'last', 'cls', 'avg').
        max_length (int, optional): Maximum length of the input sequence.
        norm (nn.Module, optional): Normalization layer.
        position_embeds (tp.Callable, optional): Positional embeddings.
        embed_drop_rate (float, optional): Dropout rate for the token embeddings.
    """
    def __init__(self,
        layers: list,
        layer_cfgs: list,
        vocab_size: int,
        dim: int = 128,
        global_pool: str = 'last',
        max_length: int = 1280,
        norm: nn.Module = RMSNorm,
        position_embeds: tp.Callable = None,
        embed_drop_rate: float = 0.0,
        *args, **kwargs
    ) -> None:
        super().__init__()

        assert global_pool in {'avg', 'cls', 'last'}, 'global_pool must be one of "avg", "cls" or "last"'
        assert len(layer_cfgs)==len(layers), 'number of layer configs must be equal to number of layers'
        assert all(cfg['dim']==dim for cfg in layer_cfgs), 'all layer configs must have the same dimensionality'
        assert all(cfg['max_length']==max_length for cfg in layer_cfgs), 'all layer configs must have the same max_length'
                
        self.vocab_size = vocab_size
        self.token_embeds = nn.Embedding(vocab_size, dim)
        enc_posembs = position_embeds(max_length, dim) if position_embeds is not None else None
        self.enc_position_embeds = enc_posembs.weight if isinstance(enc_posembs, nn.Embedding) else enc_posembs
        assert self.enc_position_embeds is None or self.enc_position_embeds.shape == (max_length, dim),\
              'position embeddings must have shape (max_length, dim)'
        self.dec_position_embeds = posemb_sincos_1d(max_length, dim)
        self.drop_embed = nn.Dropout(embed_drop_rate)

        self.encoder = nn.ModuleList([])
        for layer, layer_cfg in zip(layers, layer_cfgs):
            layer_cfg['max_length'] = max_length + 1 if global_pool == 'cls' else max_length
            self.encoder.append(nn.Sequential(norm(layer_cfg['dim']), layer(**layer_cfg)))
        
        self.global_pool = global_pool
        if global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # hard-coded non-expanding MLP decoder
        self.decoder = nn.Sequential(
            norm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            norm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        
        self.unembed = nn.Sequential(norm(dim), nn.Linear(dim, vocab_size))
        self.apply(self._init_weights)

    def embed(self,
        inputs_ids: torch.Tensor,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seqlen = inputs_ids.shape
        input_embeds = self.token_embeds(inputs_ids)
        if self.enc_position_embeds is not None:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long)
            posembs = self.enc_position_embeds[position_ids]
            input_embeds = input_embeds + posembs.to(input_embeds.device)
        return self.drop_embed(input_embeds)
    
    def encode(self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        
        batch_size, seqlen = input_ids.shape
        x = self.embed(input_ids)

        if self.global_pool == 'cls':
            x = torch.cat([x, self.cls_token.expand(batch_size, -1, -1)], dim=1)
        
        for layer in self.encoder:
            x = x + layer(x)
        
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.global_pool in {'last', 'cls'}:
            x = x[:,-1]
        else:
            raise NotImplementedError(f'global pool {self.global_pool} not implemented')
        
        return x, position_ids
    
    def decode(self,
        encoding: torch.Tensor,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        
        batch_size, dim = encoding.shape
        x = encoding.unsqueeze(1) + self.dec_position_embeds.to(encoding.device)[position_ids]
        x = rearrange(x, 'b n d -> (b n) d')

        for layer in self.decoder:
            x = layer(x)

        token_logits = self.unembed(x)
        token_logits = rearrange(token_logits, '(b n) v -> b n v', b=batch_size)
        return token_logits
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoding, position_ids = self.encode(input_ids)
        token_logits = self.decode(encoding, position_ids)
        return token_logits

    def _init_weights(self, m, initializer_range=0.02) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=initializer_range)
