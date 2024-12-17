import torch
import typing as tp
import pytorch_lightning as pl
import torchmetrics as met
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from mad.metrics import Accuracy


class PLModelWrap(pl.LightningModule):
    """
    PyTorch Lightning model wrapper.
    
    Args:
        model (nn.Module): Model to wrap.
        mad_config (MADConfig): MAD configuration.
        metrics (list, optional): List of metrics to use.
    """

    def __init__(self, model, mad_config, metrics: list=['acc', 'ppl']):
        super().__init__()
        self.model = model
        self.mad_config = mad_config
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.mad_config.target_ignore_index)
        self.instantiate_metrics(metrics=metrics)
        self.save_hyperparameters('mad_config')

    def instantiate_metrics(self, metrics: list) -> None:
        mets = []
        for m in metrics:
            if m=='acc':
                mets.append(
                    Accuracy(
                        num_classes=self.model.vocab_size,
                        ignore_index=self.mad_config.target_ignore_index
                    )
                )
            elif m=='ppl':
                mets.append(met.text.Perplexity(ignore_index=self.mad_config.target_ignore_index))
            elif isinstance(m, met.Metric):
                mets.append(m)
            else:
                raise ValueError(f"invalid metric: {m}, must be one of 'acc', 'ppl' or a torchmetrics metric instance")

        mets = met.MetricCollection(mets)
        self.train_metrics = mets.clone(prefix='train/')
        self.test_metrics = mets.clone(prefix='test/')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
        return loss, outputs, targets
    
    def phase_step(self,
        batch: tuple,
        batch_idx: int,
        phase: str='train'
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        loss, outputs, targets = self.step(batch, batch_idx)
        self.log(f'{phase}/Loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        metrics = getattr(self, f'{phase}_metrics')(outputs, targets)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, "outputs": outputs, "targets": targets}
    
    def training_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        return self.phase_step(batch, batch_idx, phase='train')
    
    def validation_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        # We currently do not use any validation data, only train/test
        return self.phase_step(batch, batch_idx, phase='test')

    def test_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        return self.phase_step(batch, batch_idx, phase='test')

    def configure_optimizers(self) -> tp.Union[torch.optim.Optimizer, tp.Dict[str, tp.Any]]:
        # param groups
        decay_params, no_decay_params = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if not getattr(p, '_no_weight_decay', False) and ("bias" not in n) and ("norm" not in n):
                    decay_params.append(p)
                else:
                    no_decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": self.mad_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # optimizer:
        if self.mad_config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.mad_config.lr
            )
        elif self.mad_config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.mad_config.lr
            )
        else:
            raise ValueError(f"invalid optimizer: {self.mad_config.optimizer}")
        
        # scheduler:
        if self.mad_config.scheduler == 'none':
            return optimizer
        elif self.mad_config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.mad_config.epochs,
                eta_min=self.mad_config.min_lr,
                last_epoch=-1
            )
            return {'optimizer': optimizer, 'scheduler': scheduler}
        elif self.mad_config.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.mad_config.plateau_patience,
                factor=self.mad_config.plateau_factor,
                min_lr=self.mad_config.min_lr,
                verbose=True
            )
            return {'optimizer': optimizer, 'scheduler': scheduler, 'monitor': "test/Loss_epoch"}
        else:
            raise ValueError(f"invalid scheduler: {self.mad_config.scheduler}")