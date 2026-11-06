"""State-based flat-ground gait shaping, shared by real and imagined rewards."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.kinematics.go2_fk import GO2_FOOT_RADIUS, go2_feet_world


@dataclass
class Go2GaitCfg:
    # Rates before dt; deliberately smaller than velocity tracking (3.0).
    feet_clearance: float = 0.0  # Opt-in; keep height shaping disabled by default.
    diagonal_gait: float = -0.15
    feet_slip: float = -0.20
    clearance_target: float = 0.05  # sphere bottom above flat ground, metres
    clearance_sigma: float = 0.025

    def __post_init__(self) -> None:
        if self.clearance_target <= 0 or self.clearance_sigma <= 0:
            raise ValueError("Go2 clearance target and sigma must be positive")


GO2_GAIT_REWARD_NAMES = ("feet_clearance", "diagonal_gait", "feet_slip")


def go2_gait_reward_terms(
    state: torch.Tensor, goal: torch.Tensor, cfg: Go2GaitCfg,
) -> dict[str, torch.Tensor]:
    """Encourage swing clearance, diagonal support, and stationary stance feet.

    No phase clock or contact history is required: the WM predicts the same
    joint/root state used by real rewards. Contact is a smooth geometric proxy,
    NOT a measured contact force. Ground is z=0 in the current velocity task.
    """
    zero = state[..., 0] * 0.0
    if cfg.feet_clearance == cfg.diagonal_gait == cfg.feet_slip == 0.0:
        return dict.fromkeys(GO2_GAIT_REWARD_NAMES, zero)
    pos, vel = go2_feet_world(state)
    height = pos[..., 2] - GO2_FOOT_RADIUS
    speed_sq = vel[..., :2].square().sum(-1)
    # Includes pure yaw commands (0.25 m is a representative foot lever arm).
    command_speed = (
        goal[..., :2].square().sum(-1) + (0.25 * goal[..., 2]).square() + 1e-8
    ).sqrt() - 1e-4
    moving = ((command_speed - 0.03) / 0.12).clamp(0.0, 1.0)
    swing_speed = torch.tanh((torch.sqrt(speed_sq + 1e-6) - 0.001) / 0.5)
    clearance = torch.exp(-((height - cfg.clearance_target) / cfg.clearance_sigma).square())
    swing = torch.sigmoid((height - 0.025) / 0.008)
    # FL/RR and FR/RL move together; both diagonals airborne or grounded cost
    # more than two-foot support. This is a soft coordination bias, not a timer.
    diagonal_error = 0.5 * (
        (swing[..., 0] - swing[..., 3]).square()
        + (swing[..., 1] - swing[..., 2]).square()
    )
    support_error = ((swing.sum(-1) - 2.0) / 2.0).square()
    stance = torch.sigmoid((0.015 - height) / 0.005)
    return {
        "feet_clearance": cfg.feet_clearance * moving * (clearance * swing_speed).mean(-1),
        "diagonal_gait": cfg.diagonal_gait * moving * (diagonal_error + support_error),
        "feet_slip": cfg.feet_slip * (stance * speed_sq).mean(-1),
    }
