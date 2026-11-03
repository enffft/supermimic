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
PRIOR_INPUT_COMPATIBILITY = ("legacy_v1", "train_constant_compat_v1")
_Q6_CONDITION_INDICES = tuple(i * STATE_DIM + 30 for i in range(HISTORY))
_QD6_CONDITION_INDICES = tuple(i * STATE_DIM + 48 for i in range(HISTORY))
_TRAIN_CONSTANT_CONDITION_INDICES = _Q6_CONDITION_INDICES + _QD6_CONDITION_INDICES
_TRAIN_CONSTANT_MAX_STD = 1.0e-3


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
                 hidden_dim: int = 256, layers: int = 3,
                 input_compatibility: str = "legacy_v1"):
        super().__init__()
        self.model = _CVAECore(latent_dim=latent_dim, hidden_dim=hidden_dim, layers=layers)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32).reshape(CONDITION_DIM))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32).reshape(CONDITION_DIM))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32).reshape(FUTURE_DIM))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32).reshape(FUTURE_DIM))
        if input_compatibility not in PRIOR_INPUT_COMPATIBILITY:
            raise ValueError(f"input_compatibility must be one of {PRIOR_INPUT_COMPATIBILITY}")
        self.input_compatibility = input_compatibility
        if (not torch.isfinite(self.x_mean).all() or not torch.isfinite(self.x_std).all()
                or not torch.isfinite(self.y_mean).all() or not torch.isfinite(self.y_std).all()
                or torch.any(self.x_std <= 0) or torch.any(self.y_std <= 0)):
            raise ValueError("invalid AMP CVAE normalization")
        self._validate_input_compatibility()

    def _validate_input_compatibility(self) -> None:
        """Keep mean projection exclusive to checkpoints trained on constants.

        V3.3 has real q6/qd6 variance. Projecting those dimensions to their
        training means silently discards motion, so fail at load and inference
        if that compatibility policy is paired with non-constant statistics.
        """
        if self.input_compatibility != "train_constant_compat_v1":
            return
        index = torch.tensor(_TRAIN_CONSTANT_CONDITION_INDICES, device=self.x_std.device)
        std = self.x_std.index_select(0, index)
        if bool((std > _TRAIN_CONSTANT_MAX_STD).any()):
            raise ValueError(
                "train_constant_compat_v1 is only valid for tiny-std q6/qd6 checkpoints; "
                f"maximum selected std is {float(std.max()):.6g}"
            )

    def forward_features(self, state_features: torch.Tensor, task_goal: torch.Tensor, *,
                         mode: Literal["prior_mean", "prior_sample"] = "prior_mean",
                         latent_eps: torch.Tensor | None = None) -> torch.Tensor:
        """Run the CVAE from already encoded AMP State53 features on-device."""
        if state_features.ndim != 3 or state_features.shape[1:] != (HISTORY, STATE_DIM):
            raise ValueError(f"state_features must be [B,4,53], got {tuple(state_features.shape)}")
        if task_goal.shape != (state_features.shape[0], GOAL_DIM):
            raise ValueError(f"task_goal must have shape [B,8], got {tuple(task_goal.shape)}")
        if not state_features.is_floating_point() or not torch.isfinite(state_features).all():
            raise ValueError("state_features must be finite floating point")
        if task_goal.device != state_features.device or task_goal.dtype != state_features.dtype:
            task_goal = task_goal.to(device=state_features.device, dtype=state_features.dtype)
        condition = torch.cat((state_features.flatten(1), task_goal), dim=-1)
        if self.input_compatibility == "train_constant_compat_v1":
            self._validate_input_compatibility()
            index = torch.tensor(_TRAIN_CONSTANT_CONDITION_INDICES, device=condition.device)
            target = self.x_mean.index_select(0, index).expand(condition.shape[0], -1)
            condition = condition.clone()
            condition[:, index] = target
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
        if result.shape != (state_features.shape[0], HORIZON, FUTURE_DIM):
            raise RuntimeError("AMP CVAE output violated Future31 contract")
        return result

    def forward(self, state_history: torch.Tensor, task_goal: torch.Tensor, *,
                mode: Literal["prior_mean", "prior_sample"] = "prior_mean",
                latent_eps: torch.Tensor | None = None) -> torch.Tensor:
        from supermimic.motion_prior.features import state_history_to_cvae53

        state = state_history_to_cvae53(state_history)
        return self.forward_features(state, task_goal, mode=mode, latent_eps=latent_eps)

    @torch.no_grad()
    def input_diagnostics(self, state_history: torch.Tensor, task_goal: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return device-native diagnostics; callers choose when to synchronize."""
        from supermimic.motion_prior.features import state_history_to_cvae53

        state_features = state_history_to_cvae53(state_history)
        if state_features.ndim != 3 or state_features.shape[1:] != (HISTORY, STATE_DIM):
            raise ValueError("state_features must be [B,4,53]")
        condition = torch.cat((state_features.flatten(1), task_goal.to(state_features)), dim=-1)
        q_idx = torch.tensor(_Q6_CONDITION_INDICES, device=condition.device)
        qd_idx = torch.tensor(_QD6_CONDITION_INDICES, device=condition.device)
        constant_idx = torch.tensor(_TRAIN_CONSTANT_CONDITION_INDICES, device=condition.device)
        delta = (condition.index_select(-1, constant_idx) - self.x_mean.index_select(0, constant_idx))
        projected_delta = delta if self.input_compatibility == "train_constant_compat_v1" else torch.zeros_like(delta)
        normalized = (condition - self.x_mean) / self.x_std
        if self.input_compatibility == "train_constant_compat_v1":
            normalized = normalized.clone()
            normalized[:, constant_idx] = 0.0
        abs_z = normalized.abs().flatten()
        raw_q6 = state_features[..., 30]
        raw_qd6 = state_features[..., 48]
        tolerance = torch.maximum(self.x_std.index_select(0, constant_idx) * 10.0,
                                  torch.full_like(self.x_std.index_select(0, constant_idx), 1.0e-5))
        ood = delta.abs() > tolerance
        return {
            "prior/input/max_abs_zscore": abs_z.max(),
            "prior/input/p99_abs_zscore": torch.quantile(abs_z, 0.99),
            "prior/input/constant_feature_ood_fraction": ood.to(state_features.dtype).mean(),
            "prior/input/q6_raw": raw_q6.abs().amax(),
            "prior/input/qd6_raw": raw_qd6.abs().amax(),
            "prior/input/q6_normalized": normalized[:, q_idx].abs().amax(),
            "prior/input/qd6_normalized": normalized[:, qd_idx].abs().amax(),
            "prior/input/compat_projection_count": (delta.abs() > 0).sum().to(state_features.dtype)
                if self.input_compatibility == "train_constant_compat_v1" else torch.zeros((), device=condition.device),
            "prior/input/compat_projection_magnitude": projected_delta.abs().mean(),
        }
