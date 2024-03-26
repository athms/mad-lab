import os
import argparse
import shutil
import torch
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from mad.configs import MADConfig, MADModelConfig
from mad.paths import make_log_path
from mad.data import generate_data
from mad.model import PLModelWrap
from mad.registry import task_registry, layer_registry


def get_args():
    parser = argparse.ArgumentParser()

    # task settings:
    parser.add_argument('--task', type=str, default='in-context-recall', choices=list(task_registry.keys()), help='task to train model on')
    parser.add_argument('--vocab-size', type=int, default=16, help='size of token vocabulary')
    parser.add_argument('--seq-len', type=int, default=128, help='length of input sequences')
    parser.add_argument('--num-train-examples', type=int, default=12_800, help='number of training examples')
    parser.add_argument('--num-test-examples', type=int, default=1_280, help='number of test examples')
    parser.add_argument('--frac-noise', type=float, default=0., help='fraction of input sequence that is noise')
    parser.add_argument('--noise-vocab-size', type=int, default=0, help='size of noise token vocabulary')
    parser.add_argument('--num-tokens-to-copy', type=int, default=0, help='number of tokens to copy in selective-copying')
    parser.add_argument('--k-motif-size', type=int, default=1, help='number of adjacent tokens that together form a key in fuzzy in-context recall')
    parser.add_argument('--v-motif-size', type=int, default=1, help='number of adjacent tokens that together form a value in fuzzy in-context recall')
    parser.add_argument('--multi-query', action=argparse.BooleanOptionalAction, default=True, help='if True, multi-query variant of in-context recall tasks is used')
    
    # model settings:
    parser.add_argument('--layers', nargs='+', default=['mh-attention', 'swiglu', 'mh-attention', 'swiglu'], help='layers of model')
    parser.add_argument('--backbone', type=str, default='language-model', help='model backbone used for layers')
    parser.add_argument('--dim', type=int, default=128, help='width of the model (will be enforced for all layers)')
    
    # training settings:
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='optimizer used for training')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'cosine', 'none'], help='learning rate scheduler used for training')
    parser.add_argument('--early-stop', action=argparse.BooleanOptionalAction, default=False, help='if True, training is stopped when validation accuracy does not improve anymore for training epochs specified by stop-patience')
    parser.add_argument('--stop-patience', type=int, default=20, help='patience for early stopping (in training epochs)')
    parser.add_argument('--plateau-patience', type=int, default=5, help='patience for plateau scheduler (in training epochs)')
    parser.add_argument('--plateau-factor', type=float, default=0.9, help='learning rate reduce factor for plateau scheduler')
    parser.add_argument('--accelerator', type=str, default='cuda', help='accelerator used for training')
    parser.add_argument('--devices', type=int, default=1, help='number of devices to use for training')
    parser.add_argument('--precision', type=str, default='bf16', help='precision of the model (see PyTorch Lightning Trainer docs for details)')

    # optimizer settings:
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for optimizer')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='minimum learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=0., help='weight decay for optimizer')

    # logging: 
    parser.add_argument('--log-base-path', type=str, default='./logs', help='path where training logs are stored')
    parser.add_argument('--log-to-csv', action=argparse.BooleanOptionalAction, default=True, help='if True, metrics are stored locallly in a csv file in the log directory')
    parser.add_argument('--log-to-wandb', action=argparse.BooleanOptionalAction, default=False, help='if True, metrics are logged to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='MAD', help='name of the Weights & Biases project to log to')
    parser.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True, help='if True, final and best model checkpoints are saved in the log directory')

    # data:
    parser.add_argument('--data-path', type=str, default='./data', help='path where generated data are stored')
    parser.add_argument('--num-data-workers', type=int, default=0, help='number of workers for data generation and data loading')
    parser.add_argument('--persistent-data-workers', action=argparse.BooleanOptionalAction, default=True, help='if True, data workers are kept alive between epochs')

    # misc:
    parser.add_argument('--seed', type=int, default=12345, help='random seed for reproducibility')
    parser.add_argument('--target-ignore-index', type=int, default=-100, help='ignore index for target in loss function')

    args = vars(parser.parse_args())

    # make sure we select the correct model backbone!
    if args['task'] in {'compression'} and args['backbone'] != 'autoencoder':
        print(f'Setting model backbone to "autoencoder", which is required for the compression task!')
        args['backbone'] = 'autoencoder'

    return args


# train model according to mad_config:

def train(
    model: torch.nn.Module,
    mad_config: MADConfig,
    log_path: str,
    log_to_csv: bool = True,
    log_to_wandb: bool = False,
    wandb_project: str = 'MAD',
    save_checkpoints: bool = True
) -> pd.DataFrame:
    """
    Train a model with given configuration and log results.

    Args:
        model (nn.Module): model to train
        mad_config (MADConfig): MAD configuration
        log_path (str): path to logs
        log_to_csv (bool): if True, log results to csv in log_path
        log_to_wandb (bool): if True, log results to Weights & Biases
        wandb_project (str): name of Weights & Biases project to log to
        save_checkpoints (bool): if True, save model checkpoints

    Returns:
        results_df (pd.DataFrame): results of training
    """

    # Set random seed.

    random.seed(mad_config.seed)
    np.random.seed(mad_config.seed)
    torch.manual_seed(mad_config.seed)

    # Check if results exist already.

    if os.path.exists(log_path):
        path_results_df = os.path.join(log_path, 'results.csv')
        if os.path.exists(path_results_df):
            results_df = pd.read_csv(path_results_df)
            print(f'Log path "{log_path}" exists, retrieved results from there...')
            return results_df
        else:
            shutil.rmtree(log_path)

    # PyTorch Lightning Model Wrap.

    model_wrapped = PLModelWrap(model=model, mad_config=mad_config)

    # Make Data.

    data = generate_data(
        instance_fn=mad_config.instance_fn,
        instance_fn_kwargs=mad_config.instance_fn_kwargs,
        train_data_path=mad_config.train_dataset_path,
        test_data_path=mad_config.test_dataset_path,
        num_train_examples=mad_config.num_train_examples,
        num_test_examples=mad_config.num_test_examples,
        num_workers=mad_config.num_data_workers
    )

    # Make Dataloaders.
        
    train_dl = DataLoader(
        dataset=data['train'],
        batch_size=mad_config.batch_size,
        shuffle=True,
        num_workers=mad_config.num_data_workers,
        persistent_workers=mad_config.persistent_data_workers and mad_config.num_data_workers>0
    )

    test_dl = DataLoader(
        dataset=data['test'],
        batch_size=mad_config.batch_size,
        shuffle=False,
        num_workers=mad_config.num_data_workers,
        persistent_workers=mad_config.persistent_data_workers and mad_config.num_data_workers>0
    )

    # Make Loggers & Callbacks.
    
    early_stop = pl.callbacks.EarlyStopping(
        monitor='test/Accuracy_epoch',
        min_delta=0.00,
        stopping_threshold=0.999,
        patience=mad_config.stop_patience if mad_config.early_stop else mad_config.epochs,
        verbose=True,
        mode='max'
    )
    callbacks = [early_stop]

    if save_checkpoints and log_path is not None:
        checkpoint_best = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor="test/Perplexity_epoch",
            mode="min",
            dirpath=os.path.join(log_path, 'checkpoints'),
            filename="best",
        )
        checkpoint_last = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor="epoch",
            mode="max",
            dirpath=os.path.join(log_path, 'checkpoints'),
            filename="last",
        )
        callbacks += [checkpoint_best, checkpoint_last]

    loggers = []
    if log_to_csv and log_path is not None:
        loggers.append(
            pl.loggers.CSVLogger(
                save_dir=log_path,
                name='logs',
                version=''
            )
        )

    if log_to_wandb:
        # We import wandb here so it doesn't create any random directories in /tmp
        # when not used
        import wandb 
        wandb.init(
            project=wandb_project,
            name=os.path.basename(log_path) if log_path is not None else None
        )
        loggers.append(pl.loggers.WandbLogger())

    # set default precision of float32 matrix multiplications:
    torch.set_float32_matmul_precision('high')

    # Make Trainer.

    trainer = pl.Trainer(
        max_epochs=mad_config.epochs,
        accelerator=mad_config.accelerator if torch.cuda.is_available() else 'cpu',
        devices=mad_config.devices,
        logger=loggers,
        enable_checkpointing=mad_config.save_checkpoints,
        callbacks=callbacks,
        precision=mad_config.precision,
    )

    # Train.

    trainer.fit(model_wrapped, train_dl, test_dl)

    # Evaluate Final Performance.

    results_train = trainer.validate(dataloaders=train_dl)[0]
    results_test = trainer.validate(dataloaders=test_dl)[0]
    results_df = pd.DataFrame({
        # training data:
        'train_acc': results_train['test/Accuracy_epoch'], # its called "test/..." because we compute results with trainer.validate
        'train_ppl': results_train['test/Perplexity_epoch'],
        'train_loss': results_train['test/Loss_epoch'],
        # test data:
        'test_acc': results_test['test/Accuracy_epoch'],
        'test_ppl': results_test['test/Perplexity_epoch'],
        'test_loss': results_test['test/Loss_epoch'],
    }, index=[0])
    results_df.to_csv(os.path.join(log_path, 'results.csv'), index=False)

    # Done!

    return results_df


if __name__ == '__main__':

    # get cli args:

    args = get_args()

    # create MAD config:

    mad_config = MADConfig()
    mad_config.update_from_kwargs(args)
    
    # create model config:

    model_config = MADModelConfig()
    model_config.update_from_kwargs(args)
    model = model_config.build_model_from_registry()    
    model_id = '-'.join(layer_registry[l]['shorthand'] for l in model_config.layers)

    # train model:

    log_path = make_log_path(
        base_path=args['log_base_path'],
        mad_config=mad_config,
        model_id=model_id
    )
    train(
        model=model,
        mad_config=mad_config,
        log_path=log_path,
        log_to_csv=args['log_to_csv'],
        log_to_wandb=args['log_to_wandb'],
        wandb_project=args['wandb_project'],
        save_checkpoints=args['save_checkpoints']
    )