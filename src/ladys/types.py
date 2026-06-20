"""Shared typed outputs for models, losses, and training strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass
class ModelOutput:
    """Standard model output container.

    All fields are optional because methods expose different scientific
    quantities. Downstream metrics can consume whichever fields a model provides.
    """

    rates: Tensor | None = None
    latents: Tensor | None = None
    reconstruction: Tensor | None = None
    distribution: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossOutput:
    """Standard loss container returned by model-specific objectives."""

    total: Tensor
    named_terms: Mapping[str, Tensor | float] = field(default_factory=dict)
    objective: str = "loss"


@dataclass
class StepResult:
    """Reporting contract returned by optimization strategies."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    batch_size: int = 0
    objective: str = "loss"

    @classmethod
    def from_loss(cls, loss: LossOutput, batch_size: int) -> "StepResult":
        metrics: dict[str, float] = {}
        for key, value in loss.named_terms.items():
            if isinstance(value, Tensor):
                metrics[key] = float(value.detach().cpu())
            else:
                metrics[key] = float(value)
        return cls(
            loss=float(loss.total.detach().cpu()),
            metrics=metrics,
            batch_size=batch_size,
            objective=loss.objective,
        )


def observations_from_batch(batch: Tensor | Mapping[str, Tensor], key: str = "spikes") -> Tensor:
    """Extract `(batch, time, neurons)` observations from a trainer batch."""

    if isinstance(batch, Tensor):
        return batch
    if key not in batch:
        raise KeyError(f"Batch is missing observation key '{key}'.")
    return batch[key]


def move_batch_to_device(batch: Any, device: torch.device | str) -> Any:
    """Move tensors in a nested batch to a device."""

    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(value, device) for value in batch)
    return batch

