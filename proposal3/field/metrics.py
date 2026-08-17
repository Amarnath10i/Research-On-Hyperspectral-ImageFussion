"""Evaluation metrics for P3: relative observation error and the
hold-out-sensor gap delta_sensor = E_unseen - E_seen."""

from __future__ import annotations

import torch


def relative_error(y_hat: torch.Tensor, y: torch.Tensor,
                   eps: float = 1e-12) -> float:
    return float((y_hat - y).norm().item()
                 / (y.norm().item() + eps))


def delta_sensor(e_unseen: float, e_seen: float) -> float:
    return e_unseen - e_seen