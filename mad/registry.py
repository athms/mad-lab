from mad.data.instances import (
    generate_in_context_recall_instance,
    generate_noisy_in_context_recall_instance,
    generate_fuzzy_in_context_recall_instance,
    generate_memorization_instance,
    generate_compression_instance,
    generate_selective_copying_instance
)
from mad.model import layers
from mad import model


task_registry = {
    'in-context-recall': {
        'instance_fn': generate_in_context_recall_instance,
        'cfg': 'configs/tasks/in-context-recall.yml',
        'shorthand': 'CR'
    },
    'noisy-in-context-recall': {
        'instance_fn': generate_noisy_in_context_recall_instance,
        'cfg': 'configs/tasks/noisy-in-context-recall.yml',
        'shorthand': 'NR'
    },
    'fuzzy-in-context-recall': {
        'instance_fn': generate_fuzzy_in_context_recall_instance,
        'cfg': 'configs/tasks/fuzzy-in-context-recall.yml',
        'shorthand': 'FR'
    },
    'memorization': {
        'instance_fn': generate_memorization_instance,
        'cfg': 'configs/tasks/memorization.yml',
        'shorthand': 'M'
    },
    'compression': {
        'instance_fn': generate_compression_instance,
        'cfg': 'configs/tasks/compression.yml',
        'shorthand': 'C'
    },
    'selective-copying': {
        'instance_fn': generate_selective_copying_instance,
        'cfg': 'configs/tasks/selective-copying.yml',
        'shorthand': 'SC'
    },
}


layer_registry = {
    # channel mixers:
    'mlp': {
        'module': layers.Mlp,
        'cfg': 'configs/layers/mlp.yml',
        'shorthand': 'M'
    },
    'moe-mlp': {
        'module': layers.MoeMlp,
        'cfg': 'configs/layers/moe-mlp.yml',
        'shorthand': 'MoE'
    },
    'swiglu': {
        'module': layers.SwiGLU,
        'cfg': 'configs/layers/swiglu.yml',
        'shorthand': 'Sg'
    },
    # sequence mixers:
    'attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/attention.yml',
        'shorthand': 'A'
    },
    'sliding-attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/sliding-attention.yml',
        'shorthand': 'As'
    },
    'linear-attention': {
        'module': layers.LinearAttention,
        'cfg': 'configs/layers/linear-attention.yml',
        'shorthand': 'Al'
    },
    'gated-linear-attention': {
        'module': layers.GatedLinearAttention,
        'cfg': 'configs/layers/gated-linear-attention.yml',
        'shorthand': 'Alg'
    },
    'hyena': {
        'module': layers.HyenaOperator,
        'cfg': 'configs/layers/hyena.yml',
        'shorthand': 'H'
    },
    'hyena-experts': {
        'module': layers.HyenaExpertsOperator,
        'cfg': 'configs/layers/hyena-experts.yml',
        'shorthand': 'He'
    },
    'mamba': {
        'module':layers. Mamba,
        'cfg': 'configs/layers/mamba.yml',
        'shorthand': 'Mb'
    },
    'mh-attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/mh-attention.yml',
        'shorthand': 'mA'
    },
    'mh-sliding-attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/mh-sliding-attention.yml',
        'shorthand': 'mAs'
    },
    'mh-linear-attention': {
        'module': layers.LinearAttention,
        'cfg': 'configs/layers/mh-linear-attention.yml',
        'shorthand': 'mAl'
    },
    'mh-gated-linear-attention': {
        'module': layers.GatedLinearAttention,
        'cfg': 'configs/layers/mh-gated-linear-attention.yml',
        'shorthand': 'mAlg'
    },
    'mh-hyena': {
        'module': layers.MultiHeadHyenaOperator,
        'cfg': 'configs/layers/mh-hyena.yml',
        'shorthand': 'mH'
    },
    'rwkv5-time-mixer': {
        'module': layers.time_mixer_rwkv5_wrapped_bf16,
        'cfg': 'configs/layers/rwkv5-time-mixer.yml',
        'shorthand': 'R5t'
    },
    'rwkv5-channel-mixer': {
        'module': layers.channel_mixer_rwkv5_wrapped,
        'cfg': 'configs/layers/rwkv5-channel-mixer.yml',
        'shorthand': 'R5c'
    },
    'rwkv6-time-mixer': {
        'module': layers.time_mixer_rwkv6_wrapped_bf16,
        'cfg': 'configs/layers/rwkv6-time-mixer.yml',
        'shorthand': 'R6t'
    },
    'rwkv6-channel-mixer': {
        'module': layers.channel_mixer_rwkv6_wrapped,
        'cfg': 'configs/layers/rwkv6-channel-mixer.yml',
        'shorthand': 'R6c'
    }
}


model_registry = {
    'language-model': model.LanguageModel,
    'autoencoder': model.AutoEncoder
}
