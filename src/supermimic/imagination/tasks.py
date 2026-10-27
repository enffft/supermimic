"""Narrow task interfaces consumed by the shared imagination / BPTT code."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.tasks.velocity_reward import (
    VELOCITY_REWARD_NAMES,
    VelocityRewardCfg,
    VelocityRobotParams,
    velocity_reward_terms,
    velocity_reward_total,
)
from supermimic.tasks.world_point_reward import (
    WORLD_POINT_REWARD_NAMES,
    WorldPointRobotParams,
    world_point_reward_terms,
)
from supermimic.kinematics.rotations import quat_apply_inverse


@dataclass
class MotionTrackingTask:
    library: MotionLibrary
    future_frames: int = 10
    anchor_obs: bool = False
    anchor_orientation_obs: bool = False
    goal_interface: str = "motion"
    debug_objective: bool = False

    @property
    def goal_dim(self) -> int:
        if self.anchor_obs and self.anchor_orientation_obs:
            raise ValueError("choose position+orientation Anchor or orientation-only Anchor")
        return self.library.goal_feat_dim * self.future_frames + (9 if self.anchor_obs else 6 if self.anchor_orientation_obs else 0)


@dataclass
class VelocityCommandTask:
    """Go2 body-frame velocity task with a shared analytic reward core."""

    params: VelocityRobotParams | None = None
    dt: float = 0.02
    reward_cfg: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)
    fall_height: float = 0.17
    tilt_feature_threshold: float = 1.0 - math.cos(math.radians(70.0))
    soft_term_temperature: float = 0.015
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
        executed_action: torch.Tensor | None = None,
        measured_torque: torch.Tensor | None = None,
        measured_joint_acceleration: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.params is None:
            raise RuntimeError("VelocityCommandTask requires Go2 reward parameters")
        terms = velocity_reward_terms(
            s_prev, s_next, goal, action, previous_action, self.params,
            executed_action=executed_action,
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

    def termination_features(self, state: torch.Tensor) -> torch.Tensor:
        """Margins to the same absolute-height and gravity-axis hard-failure limits."""
        from supermimic.kinematics.rotations import quat_apply_inverse

        gravity = state.new_zeros(state.shape[:-1] + (3,))
        gravity[..., 2] = -1.0
        gravity_local = quat_apply_inverse(state[..., 3:7], gravity)
        tilt_feature = (1.0 + gravity_local[..., 2]).abs()
        return torch.stack([self.fall_height - state[..., 2], tilt_feature - self.tilt_feature_threshold], dim=-1)

    def termination_probability(self, state: torch.Tensor) -> torch.Tensor:
        margins = self.termination_features(state)
        p_height = torch.sigmoid(margins[..., 0] / self.soft_term_temperature)
        p_tilt = torch.sigmoid(margins[..., 1] / self.soft_term_temperature)
        return 1.0 - (1.0 - p_height) * (1.0 - p_tilt)

    def termination_logits(self, state: torch.Tensor) -> torch.Tensor:
        p = self.termination_probability(state).clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(p)

    @property
    def goal_dim(self) -> int:
        return 3


@dataclass
class WorldPointReachTask:
    """M1 goal-conditioned task with an episode-fixed env-local world target."""

    params: WorldPointRobotParams
    fk: object
    tcp_body_indices: tuple[int, int]
    dt: float = 0.02
    goal_interface: str = "world_point_reach"
    reward_term_names: tuple[str, ...] = WORLD_POINT_REWARD_NAMES

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self.params.default_joint_pos

    @property
    def goal_dim(self) -> int:
        return 3

    def actor_goal(self, state: torch.Tensor, target_position_world: torch.Tensor) -> torch.Tensor:
        """Re-express the fixed local-world target in the predicted/current base frame."""
        target_body_world = target_position_world - state[..., 0:3]
        return quat_apply_inverse(state[..., 3:7], target_body_world)

    def reward_terms(
        self,
        s_prev: torch.Tensor,
        s_next: torch.Tensor,
        target_position_world: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        *,
        executed_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # The shared imagination stepper supplies the physical action mapping.
        # Point-task action-rate regularization intentionally follows policy
        # action history, matching the real actuator/history contract.
        del executed_action
        terms = world_point_reward_terms(
            s_prev, s_next, target_position_world, action, previous_action,
            self.params, self.fk, self.tcp_body_indices,
        )
        return {name: value * float(self.dt) for name, value in terms.items()}

    def termination_features(self, state: torch.Tensor) -> torch.Tensor:
        from supermimic.kinematics.rotations import quat_apply_inverse

        gravity = state.new_zeros(state.shape[:-1] + (3,))
        gravity[..., 2] = -1.0
        gravity_local = quat_apply_inverse(state[..., 3:7], gravity)
        tilt_feature = (1.0 + gravity_local[..., 2]).abs()
        return torch.stack([
            self.params.fall_height - state[..., 2],
            tilt_feature - self.params.tilt_feature_threshold,
        ], dim=-1)

    def termination_probability(self, state: torch.Tensor) -> torch.Tensor:
        features = self.termination_features(state)
        p_height = torch.sigmoid(features[..., 0] / self.params.soft_term_temperature)
        p_tilt = torch.sigmoid(features[..., 1] / self.params.soft_term_temperature)
        return 1.0 - (1.0 - p_height) * (1.0 - p_tilt)

    def termination_logits(self, state: torch.Tensor) -> torch.Tensor:
        return torch.logit(self.termination_probability(state).clamp(1e-6, 1.0 - 1e-6))
