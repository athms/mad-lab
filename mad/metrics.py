import torch
from torchmetrics.classification import MulticlassAccuracy


class Accuracy(MulticlassAccuracy):
    """Subclass of `torchmetrics.classification.MulticlassAccuracy` that flattens inputs."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_flat = preds.view(-1, preds.size(-1))
        preds_flat = preds_flat.argmax(axis=-1)
        target = target.view(-1)
        super().update(preds=preds_flat, target=target)