import os
import argparse
import yaml
import torch
import ray

import typing as tp
import numpy as np
import ray.util.multiprocessing as mp

from train import train
from mad.registry import layer_registry, model_registry
from mad.configs import MADConfig, make_benchmark_mad_configs
from mad.paths import make_log_path, get_base_path
from mad.analysis import compute_model_mad_scores


def get_args():
    parser = argparse.ArgumentParser()

    # model settings:
    parser.add_argument('--layers', nargs='+', default=['mh-attention', 'swiglu', 'mh-attention', 'swiglu'], help='layers to use in the model')
    parser.add_argument('--dim', type=int, default=128, help='width of model (applied to all layers)')
    
    # training settings:
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use for training')
    parser.add_argument('--cpus', type=int, default=16, help='number of cpus to use for training')
    parser.add_argument('--num-trials-gpu', type=int, default=1, help='number of trials to run per gpu')
    parser.add_argument('--num-cpus-trial', type=int, default=2, help='number of cpus to allocate to each trial')
    parser.add_argument('--num-data-workers', type=int, default=0, help='number of workers used for data generation and loading')
    parser.add_argument('--precision', type=str, default='bf16', help='precision of model (see PyTorch Lightning Trainer docs for deatils)')
    parser.add_argument('--persistent-workers', action=argparse.BooleanOptionalAction, default=True, help='if True, data workers are kept alive between training epochs')

    # logging: 
    parser.add_argument('--logs-path', type=str, default='./benchmark/logs', help='path where logs are stored')
    parser.add_argument('--log-to-csv', action=argparse.BooleanOptionalAction, default=True, help='if True, training metrics are locally saved to csv')
    parser.add_argument('--log-to-wandb', action=argparse.BooleanOptionalAction, default=False, help='if True, training metrics are logged to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='MAD', help='project name to use when logging to Weights & Biases')
    parser.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True, help='if True, last and best model checkpoints of each training run are saved in the log directory')

    # data:
    parser.add_argument('--data-path', type=str, default='./benchmark/data', help='path where benchmark data are stored')

    # misc:
    parser.add_argument('--ray-tmp-path', type=str, default='./tmp/ray/', help='tmp path to be used by ray')

    return vars(parser.parse_args())


def check_benchmark_data_present(mad_configs):
    """Make sure benchmark data are present."""
    for mad_config in mad_configs:
        assert os.path.isdir(mad_config.train_dataset_path)
        assert os.path.isdir(mad_config.test_dataset_path)


def benchmark(
    make_model_fn: tp.Callable,
    model_id: str,
    gpus: int = 1,
    cpus: int = 12,
    num_trials_gpu: int = 1,
    num_cpus_trial: int = 2,
    data_path: str = './benchmark/data',
    logs_path: str = './benchmark/logs',
    log_to_csv: bool = True,
    log_to_wandb: bool = False,
    wandb_project: str = 'MAD',
    save_checkpoints: bool = True,
    precision: str = 'bf16',
    persistent_workers: bool = True,
    ray_tmp_path: str = '/tmp/ray'
):
    """
    Benchmark a model on MAD.
    
    Args:
        make_model_fn (callable): function that returns a PyTorch model
        model_id (str): unique identifier for the model
        mad_configs (list): list of MADConfig objects
        gpus (int): number of gpus to use for training
        cpus (int): number of cpus to use for training
        num_trials_gpu (int): number of trials to run per gpu
        num_cpus_trial (int): number of cpus to allocate to each trial
        logs_path (str): path where logs are stored
        log_to_csv (bool): if True, training metrics are locally saved to csv
        log_to_wandb (bool): if True, training metrics are logged to Weights & Biases
        wandb_project (str): project name to use when logging to Weights & Biases
        save_checkpoints (bool): if True, last and best model checkpoints of each training run are saved in the log directory
        ray_tmp_path (str): tmp path to be used by ray
        
    Returns:
        MAD scores for the model
    """

    # create all MAD configs for benchmark:
    mad_configs = make_benchmark_mad_configs(
        data_path=data_path,
        precision=precision,
        persistent_workers=persistent_workers
    )
    check_benchmark_data_present(mad_configs)

    def setup_model_and_train(mad_config):
        """Helper to setup model and train it according to MAD config."""
        log_path = make_log_path(
            base_path=logs_path,
            mad_config=mad_config,
            model_id=model_id,
        )
        model = make_model_fn(
            task=mad_config.task,
            vocab_size=mad_config.vocab_size,
            max_length=mad_config.seq_len
        )
        results = train(
            model=model,
            mad_config=mad_config,
            log_path=log_path,
            log_to_csv=log_to_csv,
            log_to_wandb=log_to_wandb,
            save_checkpoints=save_checkpoints,
            wandb_project=wandb_project
        )
        return results

    if gpus > 1:

        @ray.remote(num_gpus=1./num_trials_gpu, num_cpus=num_cpus_trial)
        def select_gpu_and_train(args):
            """Helper to select a gpu and train a model; used in multiprocessing pool."""
            job_id, mad_config = args
            gpu_id = job_id % gpus
            torch.cuda.device(gpu_id)
            return setup_model_and_train(mad_config)
        
        if not ray.is_initialized(): # set this so we can easily benchmark multiple architectures in sequence
            ray.init(num_gpus=gpus, num_cpus=cpus, _temp_dir=ray_tmp_path)
        pool = mp.Pool(gpus*num_trials_gpu)
        instances = pool.map(select_gpu_and_train.remote, enumerate(mad_configs))
        ray.get(instances);

    else:
        for mad_config in mad_configs:
            setup_model_and_train(mad_config);

    mad_scores = compute_model_mad_scores(
        model_id=model_id,
        logs_path=logs_path
    )
    print('\n----')
    print('MAD scores for each synthetic task:')
    for task, score in zip(mad_scores.index, mad_scores.values):
        print(f'  {task}: {score}')
    print(f'Mean across Tasks: {np.mean(mad_scores.values)}')

    return mad_scores


if __name__ == '__main__':

    # get cli args:

    args = get_args()

    # load layer modules and their configs:

    def load_yml(path):
        """Load a yaml file from a given path."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    layers = [layer_registry[l]['module'] for l in args['layers']]
    layer_configs = []
    for layer in args['layers']:
            layer_configs.append( load_yml(os.path.join(get_base_path(), layer_registry[layer]['cfg'])) )
   
    # define identifier for model used for logging:

    model_id = '-'.join(layer_registry[l]['shorthand'] for l in args['layers'])

    # define function to create model during benchmark:
    # (this is necessary because the model's backbone, 
    # vocab size, and max_length change during the benchmark)

    def make_model_fn(
        task: str,
        vocab_size: int,
        max_length: int,
        dim: int = args['dim'],
        layers: tp.Tuple[tp.Callable] = layers,
        layer_configs: tp.Tuple[dict] = layer_configs,
    ) -> torch.nn.Module:
        """
        Function to create the model that is to be benchmarked.
        
        Args:
            task (str): MAD task for which the model is trained
            vocab_size (int): size of the model's vocabulary
            max_length (int): maximum length of the input sequences
            dim (int): width of the model
            layers (list): list of layer modules
            layer_configs (list): list of layer configs
            
        Returns:
            PyTorch model
        """
        # set max_length and dim in layer configs:
        for lc in layer_configs:
            lc['max_length'] = max_length
            lc['dim'] = dim
        # select backbone based on task:
        backbone = 'language-model' if task not in {'compression'} else 'autoencoder'
        return model_registry[backbone](
            dim=dim,
            vocab_size=vocab_size,
            layers=layers,
            layer_cfgs=layer_configs,
            max_length=max_length,
        )
    
    # run benchmark:

    mad_scores = benchmark(
        make_model_fn=make_model_fn,
        model_id=model_id,
        gpus=args['gpus'],
        cpus=args['cpus'],
        num_trials_gpu=args['num_trials_gpu'],
        num_cpus_trial=args['num_cpus_trial'],
        logs_path=args['logs_path'],
        log_to_csv=args['log_to_csv'],
        log_to_wandb=args['log_to_wandb'],
        wandb_project=args['wandb_project'],
        save_checkpoints=args['save_checkpoints'],
        ray_tmp_path=args['ray_tmp_path']
    )