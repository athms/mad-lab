import os
import yaml
import typing as tp
import numpy as np
from dataclasses import dataclass, fields
from torch import nn

from mad.paths import get_base_path, make_dataset_path
from mad.data.instances import generate_kv_map
from mad.registry import task_registry, layer_registry, model_registry


def load_yml(path):
    """Helper function to load a yaml file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class BaseConfig:
    def update_from_kwargs(self, kwargs):
        """Update fields of the config with kwargs."""
        valid_keys = {field.name for field in fields(self)}
        for key, value in kwargs.items():
            if key in valid_keys:
                setattr(self, key, value)


@dataclass
class MADConfig(BaseConfig):
    """MAD configuration."""

    # task settings:
    task: str = 'in-context-recall'
    vocab_size: int = 16
    seq_len: int = 128
    frac_noise: float = 0.0
    noise_vocab_size: int = 0
    num_tokens_to_copy: int = 0
    k_motif_size: int = 1
    v_motif_size: int = 1
    multi_query: bool = True
    num_train_examples: int = 12_800
    num_test_examples: int = 1_280

    # data settings:
    data_path: str = './data'
    num_data_workers: int = 0
    persistent_data_workers: bool = True

    # training settings:
    batch_size: int = 128
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 0.
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    min_lr: float = 1e-6
    early_stop: bool = False
    stop_patience: int = 20
    plateau_patience: int = 5
    plateau_factor: float = 0.9
    accelerator: str = 'cuda'
    devices: int = 1
    save_checkpoints: bool = True 
    precision: str = 'bf16'

    # misc:
    seed: int = 12345
    target_ignore_index: int = -100

    @property
    def instance_fn(self) -> tp.Callable:
        """returns function from registry used to generate an instance of the task"""
        if self.task in task_registry:
            return task_registry[self.task]['instance_fn']
        else:
            return None

    @property
    def instance_fn_kwargs(self) -> tp.Dict:
        """returns dict of all kwargs required to create an instance with self.instance_fn"""
        if self.task == 'memorization':
            # We need to generate a kv_map for the memorization task.
            # As this mapping is fixed, we can generate it here,
            # avoiding that it is recreated every time a new task instance is created.
            if self.k_motif_size>1 or self.v_motif_size>1:
                print('/!\ setting {k,v}_motif_size to 1, as motifs>1 are not supported for the memorization task.')
            kv_map = generate_kv_map(
                vocab_size=self.vocab_size - 1, # also account for insert tokens
                k_motif_size=1,
                v_motif_size=1,
                seed=self.seed
            )
        else:
            kv_map = None
        return dict(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            k_motif_size=self.k_motif_size,
            v_motif_size=self.v_motif_size,
            frac_noise=self.frac_noise,
            noise_vocab_size=self.noise_vocab_size,
            rng=np.random.default_rng(self.seed),
            multi_query=self.multi_query,
            kv_map=kv_map
        )

    @property
    def dataset_path(self):
        return make_dataset_path(self)

    @property
    def train_dataset_path(self) -> str:
        return os.path.join(self.dataset_path, 'train')

    @property
    def test_dataset_path(self) -> str:
        return os.path.join(self.dataset_path, 'test')


@dataclass
class MADModelConfig(BaseConfig):
    """Model configuration for models built from architecture
    components provided in this repository."""
    layers: tp.List[str] = None
    backbone: str = 'language-model'
    dim: int = 128
    max_length: int = 1_280
    vocab_size: int = 16
    norm: nn.Module = nn.LayerNorm
    position_embeds: tp.Callable = None
    embed_drop_rate: float = 0.0

    def build_model_from_registry(self):
        """build a model from components registered in MAD"""
        layer_configs = []
        for layer in self.layers:
            _cfg = load_yml(os.path.join(get_base_path(), layer_registry[layer]['cfg']))
            _cfg['dim'] = self.dim
            _cfg['max_length'] = self.max_length
            layer_configs.append(_cfg)
        model = model_registry[self.backbone](
            dim=self.dim,
            vocab_size=self.vocab_size,
            layers=[layer_registry[l]['module'] for l in self.layers],
            layer_cfgs=layer_configs,
            max_length=self.max_length,
            norm=self.norm,
            position_embeds=self.position_embeds,
            embed_drop_rate=self.embed_drop_rate,
        )
        return model


def make_benchmark_mad_configs(**kwargs):
    """Returns a list containing all MADConfigs of the MAD benchmark."""

    lrs = [1e-4, 5e-4, 1e-3]
    wds = [0.0, 0.1]
    mad_configs = []
    for task in task_registry.keys():
        task_cfg = load_yml(task_registry[task]['cfg'])
        baseline = task_cfg['baseline']
        baseline['task'] = task
        for k,v in kwargs.items():
            baseline[k] = v
        changes = task_cfg['changes']

        for lr in lrs:
            for wd in wds:
                # baseline task setting:
                mad_config = MADConfig(lr=lr, weight_decay=wd)
                mad_config.update_from_kwargs(baseline)
                mad_configs.append(mad_config)
                # changes to baseline setting, varying task difficulty:
                for change_key in changes:
                    change_cfg = dict(baseline)
                    for change_value in changes[change_key]:
                        change_cfg[change_key] = change_value
                        mad_config = MADConfig(lr=lr, weight_decay=wd)
                        mad_config.update_from_kwargs(change_cfg)
                        mad_configs.append(mad_config)

    return mad_configs