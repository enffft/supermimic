"""Explicit, collection-only running statistics for the world model.

Statistics are updated by the replay recorder once for each real simulator
transition. Forward calls never mutate them, so replay reuse and imagination do
not bias the moments. Inputs are the existing fixed-scale canonical features;
outputs are the dynamics head's pre-``out_scale`` delta coordinates.
"""

from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, min_std: float = 1e-2):
        super().__init__()
        if dim < 1 or min_std <= 0:
            raise ValueError("RunningNorm needs positive dim and min_std")
        self.dim = int(dim)
        self.min_std = float(min_std)
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, samples: torch.Tensor) -> None:
        if samples.shape[-1] != self.dim:
            raise ValueError(f"RunningNorm expected {self.dim} features, got {samples.shape[-1]}")
        x = samples.detach().reshape(-1, self.dim).to(device=self.mean.device, dtype=torch.float32)
        if x.shape[0] == 0:
            return
        n = int(x.shape[0])
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        old_n = self.count.to(dtype=torch.float32)
        total = old_n + n
        delta = batch_mean - self.mean
        self.mean.add_(delta * (n / total))
        m2 = self.var * old_n + batch_var * n + delta.square() * (old_n * n / total)
        self.var.copy_(m2 / total)
        self.count.add_(n)

    def _std(self, x: torch.Tensor) -> torch.Tensor:
        return self.var.to(dtype=x.dtype, device=x.device).clamp_min(self.min_std**2).sqrt()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(dtype=x.dtype, device=x.device)) / self._std(x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._std(x) + self.mean.to(dtype=x.dtype, device=x.device)


class WorldModelNormalizer(nn.Module):
    """Normalizer for canonical state features and raw dynamics increments."""

    def __init__(self, input_dim: int, delta_dim: int, min_std: float = 1e-2):
        super().__init__()
        self.input = RunningNorm(input_dim, min_std)
        self.delta = RunningNorm(delta_dim, min_std)

    @torch.no_grad()
    def update_real(self, features: torch.Tensor, delta_target: torch.Tensor) -> None:
        self.input.update(features)
        self.delta.update(delta_target)
