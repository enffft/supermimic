"""Explicit, bounded M1 collection-time action exploration."""

from __future__ import annotations

import torch

from supermimic.robots.spec import M1_JOINTS


class M1BoundedGaussianExplorer:
    """Group-scaled normalized Action22 noise, applied only to real collection."""

    def __init__(self, *, leg_std: float, arm_std: float, wheel_std: float,
                 env_fraction: float, decay_end_fraction: float = 0.75):
        if not 0.0 <= env_fraction <= 1.0:
            raise ValueError("exploration environment fraction must lie in [0,1]")
        if min(leg_std, arm_std, wheel_std) < 0.0:
            raise ValueError("exploration standard deviations must be nonnegative")
        if not 0.0 < decay_end_fraction <= 1.0:
            raise ValueError("exploration decay end fraction must lie in (0,1]")
        self.leg_std = float(leg_std)
        self.arm_std = float(arm_std)
        self.wheel_std = float(wheel_std)
        self.env_fraction = float(env_fraction)
        self.decay_end_fraction = float(decay_end_fraction)
        self.std_by_action = torch.tensor([
            self.wheel_std if name.endswith("FOOT_JOINT") else
            self.arm_std if name.startswith("piper_x_joint") else self.leg_std
            for name in M1_JOINTS
        ])

    @torch.no_grad()
    def sample(self, action: torch.Tensor, *, iteration: int, total_iterations: int):
        if action.ndim != 2 or action.shape[-1] != 22:
            raise ValueError("M1 exploration expects normalized Action22 [B,22]")
        if total_iterations <= 0:
            raise ValueError("total_iterations must be positive")
        progress = min(max(float(iteration) / total_iterations, 0.0), self.decay_end_fraction)
        decay = max(0.25, 1.0 - 0.75 * progress / self.decay_end_fraction)
        std = self.std_by_action.to(action) * decay
        mask = torch.rand(action.shape[0], device=action.device) < self.env_fraction
        noise = torch.randn_like(action) * std
        return noise, mask, std
