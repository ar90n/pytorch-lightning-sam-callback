# this file is derived from https://github.com/davda54/sam/blob/main/sam.py
from typing import Any, Dict, Iterator, List, Optional, cast

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _get_params(optimizer: Optimizer) -> Iterator[Tensor]:
    for param_group in cast(List[Dict[Any, Any]], optimizer.param_groups):
        for param in param_group["params"]:
            if not isinstance(param, Tensor):
                raise TypeError(f"expected Tensor, but got: {type(param)}")
            yield param


def _get_loss(step_output: STEP_OUTPUT) -> Optional[Tensor]:
    if step_output is None:
        return None
    if isinstance(step_output, Tensor):
        return step_output
    return step_output.get("loss")


class SAM(Callback):
    _rho: float
    _adaptive: bool
    _batch: Any
    _batch_idx: int

    def __init__(self, rho: float = 0.05, adaptive: bool = False) -> None:
        super().__init__()
        self._rho = rho
        self._adaptive = adaptive

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self._batch = batch
        self._batch_idx = batch_idx

    @torch.no_grad()
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:

        org_weights = self._first_step(optimizer)
        with torch.enable_grad():
            step_output = pl_module.training_step(self._batch, self._batch_idx)
            loss = _get_loss(step_output)
            if loss is not None:
                trainer.strategy.backward(
                    loss, optimizer=optimizer, optimizer_idx=opt_idx
                )
        self._second_step(optimizer, org_weights)

    def _norm_weights(self, p: torch.Tensor) -> torch.Tensor:
        return torch.abs(p) if self._adaptive else torch.ones_like(p)

    def _grad_norm(self, optimizer: Optimizer) -> torch.Tensor:
        param_norms = torch.stack(
            [
                (self._norm_weights(p) * p.grad).norm()
                for p in _get_params(optimizer)
                if isinstance(p.grad, Tensor)
            ]
        )
        return param_norms.norm()

    def _first_step(self, optimizer: Optimizer) -> Dict[Tensor, Tensor]:
        scale = self._rho / (self._grad_norm(optimizer) + 1e-4)
        org_weights: Dict[Tensor, Tensor] = {}
        for p in _get_params(optimizer):
            if p.grad is None:
                continue
            org_weights[p] = p.data.clone()
            e_w = (torch.pow(p, 2) if self._adaptive else 1.0) * p.grad * scale.to(p)
            p.add_(e_w)  # climb to the local maximum "w + e(w)"
        optimizer.zero_grad()
        return org_weights

    def _second_step(
        self, optimizer: Optimizer, org_weights: Dict[Tensor, Tensor]
    ) -> None:
        for p in _get_params(optimizer):
            if p.grad is None:
                continue
            p.data = org_weights[p]
