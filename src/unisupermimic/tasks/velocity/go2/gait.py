# Adapted from enffft/supermimic; original MIT source + hashes in migration/source_reference.
"""State-based flat-ground gait shaping, shared by real and imagined rewards."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .feet import GO2_FOOT_RADIUS, go2_feet_world


@dataclass
class Go2GaitCfg:
    # Rates before dt; deliberately smaller than velocity tracking (3.0).
    feet_clearance: float = 0.0  # Opt-in; keep height shaping disabled by default.
    diagonal_gait: float = -0.15
    feet_slip: float = -0.20
    clearance_target: float = 0.05  # sphere bottom above flat ground, metres
    clearance_sigma: float = 0.025

    clearance_mode: str = "gaussian"
    clearance_tau: float = 0.02
    clearance_region: str = "all"

    def __post_init__(self) -> None:
        if self.clearance_target <= 0 or self.clearance_sigma <= 0:
            raise ValueError("Go2 clearance target and sigma must be positive")
        if self.clearance_mode not in ("gaussian", "soft_deficit"):
            raise ValueError("Unknown clearance mode")
        if self.clearance_region not in ("all", "stairs") or self.clearance_tau <= 0:
            raise ValueError("Invalid clearance region/tau")


GO2_GAIT_REWARD_NAMES = ("feet_clearance", "diagonal_gait", "feet_slip")


def go2_gait_reward_terms(
    state: torch.Tensor,
    goal: torch.Tensor,
    cfg: Go2GaitCfg,
    *,
    foot_ground: torch.Tensor | None = None,
    clearance_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Encourage swing clearance, diagonal support, and stationary stance feet.

    No phase clock or contact history is required: the WM predicts the same
    joint/root state used by real rewards. Contact is a smooth geometric proxy,
    NOT a measured contact force. Optional per-foot world terrain heights share
    the same geometric proxy in real transitions and imagination.
    """
    zero = state[..., 0] * 0.0
    if cfg.feet_clearance == cfg.diagonal_gait == cfg.feet_slip == 0.0:
        return dict.fromkeys(GO2_GAIT_REWARD_NAMES, zero)
    pos, vel = go2_feet_world(state)
    height = pos[..., 2] - GO2_FOOT_RADIUS
    if foot_ground is not None:
        height = height - foot_ground
    speed_sq = vel[..., :2].square().sum(-1)
    # Includes pure yaw commands (0.25 m is a representative foot lever arm).
    command_speed = (
        goal[..., :2].square().sum(-1) + (0.25 * goal[..., 2]).square() + 1e-8
    ).sqrt() - 1e-4
    moving = ((command_speed - 0.03) / 0.12).clamp(0.0, 1.0)
    swing_speed = torch.tanh((torch.sqrt(speed_sq + 1e-6) - 0.001) / 0.5)
    clearance = clearance_shape(height, cfg)
    if clearance_weight is not None:
        clearance = clearance * clearance_weight
    swing = torch.sigmoid((height - 0.025) / 0.008)
    # FL/RR and FR/RL move together; both diagonals airborne or grounded cost
    # more than two-foot support. This is a soft coordination bias, not a timer.
    diagonal_error = 0.5 * (
        (swing[..., 0] - swing[..., 3]).square() + (swing[..., 1] - swing[..., 2]).square()
    )
    support_error = ((swing.sum(-1) - 2.0) / 2.0).square()
    stance = torch.sigmoid((0.015 - height) / 0.005)
    return {
        "feet_clearance": cfg.feet_clearance * moving * (clearance * swing_speed).mean(-1),
        "diagonal_gait": cfg.diagonal_gait * moving * (diagonal_error + support_error),
        "feet_slip": cfg.feet_slip * (stance * speed_sq).mean(-1),
    }


def clearance_shape(height: torch.Tensor, cfg: Go2GaitCfg) -> torch.Tensor:
    """C1 single-sided deficit, or the exact legacy Gaussian.

    A2's squared positive part has zero value AND derivative at/above target.
    The speed gate is intentionally independent of height: otherwise increasing
    swing admission can make low feet more costly and reverse the height gradient.
    This remains a kinematic swing proxy, not measured-contact supervision.
    """
    if cfg.clearance_mode == "gaussian":
        return torch.exp(-((height - cfg.clearance_target) / cfg.clearance_sigma).square())
    deficit = (
        cfg.clearance_tau * F.softplus((cfg.clearance_target - height) / cfg.clearance_tau)
        - cfg.clearance_tau * math.log(2)
    ).clamp_min(0)
    return -(deficit / cfg.clearance_target).square()


def clearance_region_weight(xy: torch.Tensor, boxes: torch.Tensor | None) -> torch.Tensor:
    """Per predicted foot position; never use the environment's reset domain ID.

    Boxes cover compiled staircase tiles. A 10cm transition at their outer edge
    is C1, while risers/treads within a tile have the same weight. Flat tiles get
    exactly zero; no region metadata is passed to the Actor.
    """
    if boxes is None or boxes.numel() == 0:
        return torch.zeros_like(xy[..., 0])
    distance = torch.minimum(xy[..., None, :] - boxes[:, :2], boxes[:, 2:] - xy[..., None, :])
    inside = (distance.amin(-1) / 0.1).clamp(0, 1)
    smooth = inside.square() * (3 - 2 * inside)
    return smooth.amax(-1)
