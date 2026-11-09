"""Cosine learning-rate schedule (paper Table A.5): decay to `final_ratio` over `decay_fraction` of training, then constant."""

from __future__ import annotations

import math
from typing import Literal


def cosine_lr(step: int, total_steps: int, base_lr: float, final_ratio: float = 0.03, decay_fraction: float = 250_000 / 320_000) -> float:
    decay_steps = max(1, int(total_steps * decay_fraction))
    if step >= decay_steps:
        return base_lr * final_ratio
    cos = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cos)


def scheduled_lr(
    mode: Literal["cosine", "constant"],
    step: int,
    total_steps: int,
    base_lr: float,
    final_ratio: float = 0.03,
    decay_fraction: float = 250_000 / 320_000,
) -> float:
    """Select the configured LR schedule; max_hours never changes total_steps."""
    if mode == "cosine":
        return cosine_lr(step, total_steps, base_lr, final_ratio, decay_fraction)
    if mode == "constant":
        return base_lr
    raise ValueError(f"unknown learning-rate schedule {mode!r}; use 'cosine' or 'constant'")


def actor_ramp_scale(actor_updates_completed: int, ramp_updates: int) -> float:
    """Linear Actor LR ramp indexed by actual optimizer updates (1-based progress)."""
    if ramp_updates <= 0:
        return 1.0
    return min(1.0, max(0.0, actor_updates_completed / ramp_updates))
