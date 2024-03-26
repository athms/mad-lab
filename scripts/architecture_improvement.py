# Example usage:
# python -m scripts.architecture_improvement \
#     --gpus 1 \
#     --cpus 16 \
#     --num-trials-gpu 1 \
#     --num-cpus-trial 2 \
#     --logs-path ./logs/hyena-improvement \
#     --log-to-csv \
#     --log-to-wandb \
#     --wandb-project MAD \
#     --save-checkpoints

import os
import argparse
import yaml

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from mad.model import LanguageModel, AutoEncoder
from mad.registry import layer_registry
from benchmark import benchmark


def get_args():
    parser = argparse.ArgumentParser()

    # training settings:
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--cpus', type=int, default=16, help='number of cpus to use')
    parser.add_argument('--num-trials-gpu', type=int, default=1, help='number of trials to run per gpu')
    parser.add_argument('--num-cpus-trial', type=int, default=2, help='number of cpus to allocate to each trial')
    parser.add_argument('--num-data-workers', type=int, default=0, help='number of workers for data generation and dataloader')
    parser.add_argument('--precision', type=str, default='bf16', help='precision of model during training')
    parser.add_argument('--persistent-workers', action=argparse.BooleanOptionalAction, default=True, help='whether to keep workers alive between tasks')

    # logging: 
    parser.add_argument('--logs-path', type=str, default='./logs/hyena-improvement', help='path where individual training run logs are stored')
    parser.add_argument('--log-to-csv', action=argparse.BooleanOptionalAction, default=True, help='whether to log individual training run results to csv')
    parser.add_argument('--log-to-wandb', action=argparse.BooleanOptionalAction, default=False, help='whether to log individual training runs to wandb')
    parser.add_argument('--wandb-project', type=str, default='MAD', help='wandb project name')
    parser.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True, help='whether to save final and best model checkpoints of each run')

    # output paths:
    parser.add_argument('--results-path', type=str, default='./results', help='path where results are stored')
    parser.add_argument('--figures-path', type=str, default='./figures', help='path where figures are stored')

    # misc:
    parser.add_argument('--ray-tmp-path', type=str, default='/tmp/ray/', help='path where ray will store temporary files')

    return vars(parser.parse_args())


def load_yml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_sweep(
    gpus: int = 1, 
    cpus: int = 16,
    num_trials_gpu: int = 1,
    num_cpus_trial: int = 2,
    logs_path: str = './logs/hyena-improvement',
    log_to_csv: bool = True,
    log_to_wandb: bool = False,
    wandb_project: str = 'MAD',
    save_checkpoints: bool = True,
    ray_tmp_path: str = '/tmp/ray/'
) -> None:

    mad_scores = []
    for model_label, model_layers in {
        # Hyena base model:
        'Hyena + SwiGLU': ['hyena', 'swiglu', 'hyena', 'swiglu'],
        # + Heads
        'MH Hyena + SwiGLU': ['mh-hyena', 'swiglu', 'mh-hyena', 'swiglu'],
        # ++ Attention
        'Striped MH Hyena + SwiGLU': ['mh-hyena', 'swiglu', 'mh-attention', 'swiglu'],
        # +++ MoE
        'Striped MH Hyena + MoE': ['mh-hyena', 'moe-mlp', 'mh-attention', 'moe-mlp']
    }.items():

        # Define function that creates model.

        def make_model_fn(
            task: str,
            vocab_size: int,
            max_length: int,
        ) -> nn.Module:
            
            layers = [layer_registry[l]['module'] for l in model_layers]
            layer_configs = [ load_yml(os.path.join(layer_registry[l]['cfg'])) for l in model_layers ]
            for layer_config in layer_configs:
                layer_config['max_length'] = max_length
            # layer registry has 3 entries per layer:
            # - module: torch.nn.Module to create layer
            # - cfg: path to config specifying default setting of layer
            # - shorthand: abbreviation of layer name
            
            backbone = LanguageModel if task not in {'compression'} else AutoEncoder
            return backbone(
                vocab_size=vocab_size,
                max_length=max_length,
                layers=layers,
                layer_cfgs=layer_configs,
            )
        
        # Create an identifier for the model.

        model_id = '-'.join(layer_registry[l]['shorthand'] for l in model_layers)
        
        # Run model through benchmark.

        model_mad_scores = benchmark(
            make_model_fn=make_model_fn,
            model_id=model_id,
            gpus=gpus,
            cpus=cpus,
            num_trials_gpu=num_trials_gpu,
            num_cpus_trial=num_cpus_trial,
            logs_path=logs_path,
            log_to_csv=log_to_csv,
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            save_checkpoints=save_checkpoints,
            ray_tmp_path=ray_tmp_path,
        )

        # Collect results.

        model_mad_scores['total'] = model_mad_scores.mean()
        model_mad_scores['model'] = model_label
        mad_scores.append(model_mad_scores.to_frame().T)
        
    return pd.concat(mad_scores)
    

def plot_mad_scores(mad_scores: pd.DataFrame):

    # Define some key variables.

    tasks = [
        'in-context-recall',
        'fuzzy-in-context-recall',
        'noisy-in-context-recall',
        'selective-copying',
        'compression',
        'memorization',
    ]
    task_labels = [
        'Context Recall',
        'Fuzzy Recall',
        'Noisy Recall',
        'Selective Copy',
        'Compress',
        'Memorize',
    ]
    n_tasks = 6
    task_offsets = [-0.3125, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125]
    task_colors = sns.color_palette("Set2", n_tasks)
    model_sorting_idx = np.argsort(mad_scores['total'].values)
    n_models = len(model_sorting_idx)

    # Make Plot.

    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    # plot total scores:
    X = np.arange(1, n_models+1)
    Y = mad_scores['total'].values[model_sorting_idx]
    for x, y in zip(X, Y):
        axs[0].bar(
            x, y,
            color='lightgray',
            linewidth=0.5,
            edgecolor='k',
            width=0.5,
        )
    axs[0].set_ylabel(f'Mean\nEval. Acc.')
    axs[0].set_ylim(0, 1)

    # plot per-task scores:
    for ti, (task, task_label, offset) in enumerate(zip(tasks, task_labels, task_offsets)):
        Y = mad_scores[task].values[model_sorting_idx]
        for i, (x, y) in enumerate(zip(list(X), list(Y))):
            axs[1].bar(
                x+offset, y,
                color=task_colors[ti],
                width=0.11,
                align='center',
                label=task_label if i == len(Y)-1 else None
            )
    axs[1].set_ylabel('Per Task\nEval. Acc.')
    axs[1].set_xticks(X)
    axs[1].set_xticklabels(mad_scores['model'].values[model_sorting_idx], rotation=90)
    axs[1].set_ylim(0, 1.5)
    axs[1].legend(
        ncols=n_tasks,
        frameon=False,
        fontsize='small' if n_models>5 else 'x-small',
        loc='upper center'
    )

    return fig, axs


if __name__ == '__main__':

    # Get cli arguments and create output directories.
    
    args = get_args()
    os.makedirs(args['results_path'], exist_ok=True)
    os.makedirs(args['figures_path'], exist_ok=True)
    
    # Run sweep.

    mad_scores = run_sweep(
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
    mad_scores.to_csv(
        os.path.join(
            args['results_path'],
            'mad_scores_hyena_improvement.csv'
        ),
        index=False
    )

    # Plot MAD scores.

    fig, axs = plot_mad_scores(mad_scores)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            args['figures_path'],
            'mad_scores_hyena_improvement.png'
        ),
        dpi=330
    )