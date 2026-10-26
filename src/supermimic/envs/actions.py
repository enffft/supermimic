"""Joint-position action with execution noise (plan D9).

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

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self._executed_raw = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        self._processed_actions = self._executed_raw * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1

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

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        noisy = actions + self._noise_std * torch.randn_like(actions) if self._noise_std > 0 else actions
        self._executed_raw = noisy.clamp(-self.cfg.action_clip, self.cfg.action_clip)

    def apply_actions(self) -> None:
        q0 = self._entity.data.default_joint_pos[:, self._joint_ids]
        position_target = q0[:, self._position_action_idx] + self._executed_raw[:, self._position_action_idx] * self._scale[:, self._position_action_idx]
        limits = self._entity.data.soft_joint_pos_limits[:, self._position_joint_ids]
        position_target = torch.maximum(torch.minimum(position_target, limits[..., 1]), limits[..., 0])
        wheel_target = self._executed_raw[:, self._wheel_action_idx] * self._scale[:, self._wheel_action_idx]
        self._entity.set_joint_position_target(position_target, joint_ids=self._position_joint_ids)
        self._entity.set_joint_velocity_target(wheel_target, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids=None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._executed_raw[env_ids] = 0.0


@dataclass(kw_only=True)
class MixedM1ActionCfg(ActionTermCfg):
    joint_names: tuple[str, ...]
    wheel_joint_names: tuple[str, ...]
    action_scales: tuple[float, ...]
    noise_std: float = 0.1
    action_clip: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> MixedM1Action:
        return MixedM1Action(self, env)
