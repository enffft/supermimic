"""Joint-position action with Gaussian / OU execution noise (plan D9).

The *clean* policy action stays in ``raw_action`` (so ``last_action`` observations and
the action-rate penalty see it); Gaussian noise ``eps ~ N(0, std^2)`` is added to the
normalized action only when computing the executed set-point. ``executed_raw`` exposes
``a + eps`` for the replay buffer (the world model is fit on executed actions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from supermimic.tasks.action_contract import clamp_executed_action

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self._applied_noise = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)
        self.ou_env_fraction = float(cfg.ou_env_fraction)
        self.ou_noise_std = float(cfg.ou_noise_std)
        self.ou_theta = float(cfg.ou_theta)
        self.action_clip = float(cfg.action_clip)
        self.dt = float(env.step_dt)
        self._ou_state = torch.zeros_like(self._raw_actions)
        self._ou_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    @property
    def policy_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def executed_action(self) -> torch.Tensor:
        return self._executed_raw

    @property
    def physical_setpoint(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def previous_policy_action(self) -> torch.Tensor:
        return self._env.action_manager.prev_action

    @property
    def applied_noise(self) -> torch.Tensor:
        """Selected Gaussian or OU perturbation before the final action clip."""
        return self._applied_noise

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        gaussian = self.noise_std * torch.randn_like(self._raw_actions) if self.noise_std > 0.0 else torch.zeros_like(self._raw_actions)
        if self.ou_noise_std > 0.0 and self.ou_env_fraction > 0.0:
            self._ou_state += (
                -self.ou_theta * self._ou_state * self.dt
                + self.ou_noise_std * (self.dt**0.5) * torch.randn_like(self._ou_state)
            )
            noise = torch.where(self._ou_mask[:, None], self._ou_state, gaussian)
        else:
            noise = gaussian
        self._applied_noise = noise
        self._executed_raw = clamp_executed_action(self._raw_actions + noise, self.action_clip)
        self._processed_actions = self._executed_raw * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0
        self._ou_state[env_ids] = 0.0
        count = int(self._ou_mask[env_ids].numel())
        if self.ou_env_fraction > 0.0:
            self._ou_mask[env_ids] = torch.rand(count, device=self.device) < self.ou_env_fraction
        else:
            self._ou_mask[env_ids] = False

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"ou_state": self._ou_state.clone(), "ou_mask": self._ou_mask.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "ou_state" in state and state["ou_state"].shape == self._ou_state.shape:
            self._ou_state.copy_(state["ou_state"].to(self._ou_state))
        if "ou_mask" in state and state["ou_mask"].shape == self._ou_mask.shape:
            self._ou_mask.copy_(state["ou_mask"].to(self._ou_mask))


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1
    ou_env_fraction: float = 0.0
    ou_noise_std: float = 0.0
    ou_theta: float = 2.0
    action_clip: float = float("inf")

    def build(self, env: ManagerBasedRlEnv) -> NoisyJointPositionAction:
        return NoisyJointPositionAction(self, env)


class MixedM1Action(ActionTerm):
    """M1's policy-order action: position targets for 18 joints and velocity for 4 wheels."""

    cfg: MixedM1ActionCfg

    def __init__(self, cfg: MixedM1ActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        ids, names = self._entity.find_joints(cfg.joint_names, preserve_order=True)
        if tuple(names) != tuple(cfg.joint_names):
            raise ValueError(f"M1 action joint order mismatch: got {names}")
        self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        wheel_names = set(cfg.wheel_joint_names)
        self._wheel_action_idx = torch.tensor([i for i, n in enumerate(names) if n in wheel_names], device=self.device, dtype=torch.long)
        self._position_action_idx = torch.tensor([i for i, n in enumerate(names) if n not in wheel_names], device=self.device, dtype=torch.long)
        self._wheel_joint_ids = self._joint_ids[self._wheel_action_idx]
        self._position_joint_ids = self._joint_ids[self._position_action_idx]
        self._action_dim = len(ids)
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self._physical_setpoint = torch.zeros_like(self._raw_actions)
        self._scale = torch.tensor(cfg.action_scales, dtype=torch.float32, device=self.device).expand(self.num_envs, -1)
        self._noise_std = float(cfg.noise_std)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    @property
    def policy_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def executed_action(self) -> torch.Tensor:
        return self._executed_raw

    @property
    def physical_setpoint(self) -> torch.Tensor:
        return self._physical_setpoint

    @property
    def previous_policy_action(self) -> torch.Tensor:
        return self._env.action_manager.prev_action

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        noisy = actions + self._noise_std * torch.randn_like(actions) if self._noise_std > 0 else actions
        self._executed_raw = clamp_executed_action(noisy, self.cfg.action_clip)

    def apply_actions(self) -> None:
        q0 = self._entity.data.default_joint_pos[:, self._joint_ids]
        position_target = q0[:, self._position_action_idx] + self._executed_raw[:, self._position_action_idx] * self._scale[:, self._position_action_idx]
        limits = self._entity.data.soft_joint_pos_limits[:, self._position_joint_ids]
        position_target = torch.maximum(torch.minimum(position_target, limits[..., 1]), limits[..., 0])
        wheel_target = self._executed_raw[:, self._wheel_action_idx] * self._scale[:, self._wheel_action_idx]
        self._physical_setpoint.zero_()
        self._physical_setpoint[:, self._position_action_idx] = position_target
        self._physical_setpoint[:, self._wheel_action_idx] = wheel_target
        self._entity.set_joint_position_target(position_target, joint_ids=self._position_joint_ids)
        self._entity.set_joint_velocity_target(wheel_target, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids=None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._executed_raw[env_ids] = 0.0
        self._physical_setpoint[env_ids] = 0.0


@dataclass(kw_only=True)
class MixedM1ActionCfg(ActionTermCfg):
    joint_names: tuple[str, ...]
    wheel_joint_names: tuple[str, ...]
    action_scales: tuple[float, ...]
    noise_std: float = 0.1
    action_clip: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> MixedM1Action:
        return MixedM1Action(self, env)
