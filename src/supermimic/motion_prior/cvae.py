"""Device-native AMP CVAE implementation with a strict checkpoint-compatible layout."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

HISTORY = 4
STATE_DIM = 53
GOAL_DIM = 8
CONDITION_DIM = 220
HORIZON = 10
FUTURE_DIM = 31
FEATURE_VERSION = "m1_cvae_motion_prior_v1_state53_goal8_future31"


class _CVAECore(nn.Module):
    """Layer names/shapes match AMP ``ConditionalVAE`` exactly."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, layers: int = 3):
        super().__init__()
        if min(latent_dim, hidden_dim, layers) < 1:
            raise ValueError("CVAE dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.future_dim = HORIZON * FUTURE_DIM
        self.posterior_backbone = self._mlp(CONDITION_DIM + self.future_dim, hidden_dim, layers)
        self.posterior_stats = nn.Linear(hidden_dim, 2 * latent_dim)
        self.prior_backbone = self._mlp(CONDITION_DIM, hidden_dim, layers)
        self.prior_stats = nn.Linear(hidden_dim, 2 * latent_dim)
        self.decoder = nn.Sequential(
            self._mlp(CONDITION_DIM + latent_dim, hidden_dim, layers),
            nn.Linear(hidden_dim, self.future_dim),
        )

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
        blocks: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for _ in range(layers - 1):
            blocks.extend((nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
        return nn.Sequential(*blocks)

    @staticmethod
    def _bounded_stats(layer: nn.Module, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = layer(hidden).chunk(2, dim=-1)
        return mu, torch.clamp(logvar, min=-10.0, max=5.0)

    def prior(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bounded_stats(self.prior_stats, self.prior_backbone(condition))

    def decode(self, condition: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder[0](torch.cat((condition, latent), dim=-1))
        return self.decoder[1](hidden).reshape(-1, HORIZON, FUTURE_DIM)


class M1MotionPrior(nn.Module):
    """AMP CVAE with differentiable ``[B,4,53]``/``[B,8]`` inference.

    The public forward returns physical-unit Future31 values. It intentionally
    never exposes posterior inference: online planning only uses current/past
    state and the task Goal.
    """

    feature_version = FEATURE_VERSION

    def __init__(self, x_mean, x_std, y_mean, y_std, latent_dim: int = 32,
                 hidden_dim: int = 256, layers: int = 3):
        super().__init__()
        self.model = _CVAECore(latent_dim=latent_dim, hidden_dim=hidden_dim, layers=layers)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32).reshape(CONDITION_DIM))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32).reshape(CONDITION_DIM))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32).reshape(FUTURE_DIM))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32).reshape(FUTURE_DIM))
        if (not torch.isfinite(self.x_mean).all() or not torch.isfinite(self.x_std).all()
                or not torch.isfinite(self.y_mean).all() or not torch.isfinite(self.y_std).all()
                or torch.any(self.x_std <= 0) or torch.any(self.y_std <= 0)):
            raise ValueError("invalid AMP CVAE normalization")

    def forward(self, state_history: torch.Tensor, task_goal: torch.Tensor, *,
                mode: Literal["prior_mean", "prior_sample"] = "prior_mean",
                latent_eps: torch.Tensor | None = None) -> torch.Tensor:
        from supermimic.motion_prior.features import state_history_to_cvae53

        state = state_history_to_cvae53(state_history)
        if task_goal.shape != (state.shape[0], GOAL_DIM):
            raise ValueError(f"task_goal must have shape [B,8], got {tuple(task_goal.shape)}")
        if not task_goal.is_floating_point() or task_goal.device != state.device:
            task_goal = task_goal.to(device=state.device, dtype=state.dtype)
        condition = torch.cat((state.flatten(1), task_goal), dim=-1)
        condition = (condition - self.x_mean) / self.x_std
        mu, logvar = self.model.prior(condition)
        if mode == "prior_mean":
            latent = mu
        elif mode == "prior_sample":
            if latent_eps is None:
                latent_eps = torch.randn_like(mu)
            if latent_eps.shape != mu.shape:
                raise ValueError(f"latent_eps must have shape {tuple(mu.shape)}")
            latent = mu + torch.exp(0.5 * logvar) * latent_eps
        else:
            raise ValueError("mode must be 'prior_mean' or 'prior_sample'")
        normalized = self.model.decode(condition, latent)
        result = normalized * self.y_std + self.y_mean
        if result.shape != (state.shape[0], HORIZON, FUTURE_DIM):
            raise RuntimeError("AMP CVAE output violated Future31 contract")
        return result
