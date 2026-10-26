"""Narrow task interfaces consumed by the shared imagination / BPTT code."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.tasks.velocity_reward import (
    VELOCITY_REWARD_NAMES,
    VelocityRewardCfg,
    VelocityRobotParams,
    velocity_reward_terms,
    velocity_reward_total,
)


@dataclass
class MotionTrackingTask:
    library: MotionLibrary
    future_frames: int = 10
    anchor_obs: bool = False
    goal_interface: str = "motion"
    debug_objective: bool = False

    @property
    def goal_dim(self) -> int:
        return self.library.goal_feat_dim * self.future_frames + (9 if self.anchor_obs else 0)


@dataclass
class VelocityCommandTask:
    """Go2 body-frame velocity task with a shared analytic reward core."""

    params: VelocityRobotParams | None = None
    dt: float = 0.02
    reward_cfg: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)
    goal_interface: str = "velocity"
    reward_term_names: tuple[str, ...] = VELOCITY_REWARD_NAMES

    @property
    def default_joint_pos(self) -> torch.Tensor | None:
        return None if self.params is None else self.params.default_joint_pos

    def reward_terms(
        self,
        s_prev: torch.Tensor,
        s_next: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        *,
        measured_torque: torch.Tensor | None = None,
        measured_joint_acceleration: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.params is None:
            raise RuntimeError("VelocityCommandTask requires Go2 reward parameters")
        terms = velocity_reward_terms(
            s_prev, s_next, goal, action, previous_action, self.params,
            measured_torque=measured_torque,
            measured_joint_acceleration=measured_joint_acceleration,
        )
        return {name: value * float(self.dt) for name, value in terms.items()}

    def reward(
        self,
        s_prev: torch.Tensor,
        s_next: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        return velocity_reward_total(self.reward_terms(s_prev, s_next, goal, action, previous_action))

    @property
    def goal_dim(self) -> int:
        return 3
