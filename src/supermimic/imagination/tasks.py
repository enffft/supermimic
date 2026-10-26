"""Narrow task interfaces consumed by the shared imagination / BPTT code."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.envs.mdp.motion_library import MotionLibrary


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
    """Velocity command semantics. A diagnostic reward is opt-in for gradient smoke tests only."""

    debug_objective: bool = False
    default_joint_pos: torch.Tensor | None = None
    dt: float = 0.02
    goal_interface: str = "velocity"
    reward_term_names: tuple[str, ...] = ("debug_velocity_tracking",)

    def reward(self, s_next: torch.Tensor, goal: torch.Tensor, num_joints: int, dt: float) -> torch.Tensor:
        if not self.debug_objective:
            raise RuntimeError(
                "Go2 has no selected task reward yet; policy optimization is disabled. "
                "Enable the diagnostic objective only for a gradient smoke test."
            )
        from supermimic.kinematics.rotations import quat_apply_inverse

        n = num_joints
        root_quat = s_next[..., 3:7]
        lin_world = s_next[..., 7 + n:10 + n]
        ang_body = s_next[..., 10 + n:13 + n]
        lin_body = quat_apply_inverse(root_quat, lin_world)
        velocity = torch.cat([lin_body[..., :2], ang_body[..., 2:3]], dim=-1)
        # Deliberately diagnostic-only: smooth bounded tracking signal for checking BPTT.
        error_sq = ((velocity - goal) ** 2).sum(-1)
        return float(dt) * torch.exp(-error_sq / 0.25)

    @property
    def goal_dim(self) -> int:
        return 3
