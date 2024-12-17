from collections import OrderedDict
import torch
import typing as tp
from torch import nn

from mad.model.layers.ops.norm.rmsnorm import RMSNorm


class LanguageModel(nn.Module):
    """
    Language model backbone.
    
    Args:
        dim (int): Width of the model.
        vocab_size (int): Size of the vocabulary.
        layers (list): List of layer modules.
        layer_cfgs (list): List of layer configs.
        max_length (int, optional): Maximum length of the input sequence.
        norm (nn.Module, optional): Normalization layer.
        position_embeds (tp.Callable, optional): Positional embeddings.
        embed_drop_rate (float, optional): Dropout rate for the token embeddings.
    """
    def __init__(self,
        vocab_size: int,
        layers: list,
        layer_cfgs: list,
        dim: int = 128,
        max_length: int = 1280,
        norm: nn.Module = RMSNorm,
        position_embeds: tp.Callable = None,
        embed_drop_rate: float = 0.0,
        *args, **kwargs
    ) -> None:
        super().__init__()

        assert len(layer_cfgs)==len(layers), 'number of layer configs must be equal number of layers'
        assert all(cfg['dim']==dim for cfg in layer_cfgs), 'all layer configs must specify the same dimensionality'
        assert all(cfg['max_length']==max_length for cfg in layer_cfgs), 'all layer configs must have the same max_length'
                
        self.vocab_size = vocab_size
        self.token_embeds = nn.Embedding(vocab_size, dim)
        position_embeds = position_embeds(max_length, dim) if position_embeds is not None else None
        self.position_embeds = position_embeds.weight if isinstance(position_embeds, nn.Embedding) else position_embeds
        assert self.position_embeds is None or self.position_embeds.shape == (max_length, dim),\
              'position embeddings must have shape (max_length, dim)'
        self.drop_embed = nn.Dropout(embed_drop_rate)
        
        self.model = nn.ModuleList([])
        for layer, layer_cfg in zip(layers, layer_cfgs):
            self.model.append(nn.Sequential(OrderedDict([
                ('norm', norm(layer_cfg['dim'])),
                ('layer', layer(**layer_cfg))
            ])))

        self.unembed = nn.Sequential(OrderedDict([
            ('norm', norm(layer_cfg['dim'])), 
            ('lm_head', nn.Linear(dim, vocab_size))
        ]))

        self.apply(self._init_weights)
        
    def embed(self,
        inputs_ids: torch.Tensor,
        position_ids: torch.Tensor=None
    ) -> torch.Tensor:
        batch_size, seqlen = inputs_ids.shape
        input_embeds = self.token_embeds(inputs_ids)
        if self.position_embeds is not None:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long)
            posembs = self.position_embeds[position_ids]
            input_embeds = input_embeds + posembs.to(input_embeds.device)
        return self.drop_embed(input_embeds)
    
    def forward(self, inputs_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(inputs_ids)
        for layer in self.model:
            x = x + layer(x)
        return self.unembed(x)

    def _init_weights(self, m, initializer_range=0.02) -> None:
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                if not getattr(m.bias, "_no_reinit", False):
                    nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=initializer_range)
