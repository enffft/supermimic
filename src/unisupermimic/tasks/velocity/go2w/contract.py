"""Versioned Go2W ordering and configurable mixed motor control."""

from dataclasses import dataclass
from pathlib import Path

import torch

from unisupermimic.rl.contracts.layout import DirectPolicySpec, StateLayout

LEGS = ("FL", "FR", "RL", "RR")
LEG_JOINTS = tuple(f"robot/{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
WHEEL_JOINTS = tuple(f"robot/{leg}_wheel_joint" for leg in LEGS)
# Logical order differs from MJCF traversal order; every sim index is resolved by name.
JOINTS = LEG_JOINTS + WHEEL_JOINTS
DEFAULT_Q = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5)
EFFORT = (23.7, 23.7, 45.43) * 4 + (15.0,) * 4
LAYOUT = StateLayout(12, 16, 16, version="go2w_compact_state_v1")
POLICY_SPEC = DirectPolicySpec((12, 12, 4, 3, 3, 16), (1.0, 0.2, 0.05, 0.5, 1.0, 1.0), action_dim=16)
MODEL = Path(__file__).resolve().parents[3] / "assets/robots/go2w/flat.xml"


@dataclass
class MixedControlCfg:
    dt: float = 0.02
    leg_position_scale: float = 0.25
    leg_kp: float = 40.0
    leg_kd: float = 1.0
    wheel_kp: float = 0.0
    wheel_kd: float = 0.5
    wheel_velocity_scale: float = 15.0
    leg_effort_limits: tuple[float, ...] = (23.7, 23.7, 45.43) * 4
    wheel_effort_limits: tuple[float, ...] = (15.0,) * 4

    def validate(self):
        import math

        values = (self.dt, self.leg_position_scale, self.leg_kp, self.leg_kd,
                  self.wheel_kd, self.wheel_velocity_scale)
        if any(not math.isfinite(x) or x <= 0 for x in values) or self.wheel_kp != 0:
            raise ValueError("Mixed control requires positive finite gains/scales and wheel_kp=0")
        limits = tuple(self.leg_effort_limits) + tuple(self.wheel_effort_limits)
        if len(self.leg_effort_limits) != 12 or len(self.wheel_effort_limits) != 4:
            raise ValueError("Expected twelve leg and four wheel effort limits")
        if any(not math.isfinite(x) or not 0 < x <= maximum for x, maximum in zip(limits, EFFORT)):
            raise ValueError("Controller effort limits exceed official motor ctrlrange")


def soft_limits(device="cpu"):
    bounds = torch.tensor([
        (-1.0472, 1.0472), (-1.5708, 3.4907), (-2.7227, -0.83776),
        (-1.0472, 1.0472), (-1.5708, 3.4907), (-2.7227, -0.83776),
        (-1.0472, 1.0472), (-0.5236, 4.5379), (-2.7227, -0.83776),
        (-1.0472, 1.0472), (-0.5236, 4.5379), (-2.7227, -0.83776),
    ], device=device)
    mid, radius = bounds.mean(-1), (bounds[:, 1] - bounds[:, 0]) * 0.45
    return torch.stack((mid - radius, mid + radius), -1)


def mixed_torque(q_leg, qd, action, control, default_q, limits):
    """Same differentiable law for substep effort control and imagined torque."""
    a = action.clamp(-1, 1)
    target = (default_q + control.leg_position_scale * a[..., :12]).clamp(limits[:, 0], limits[:, 1])
    leg = control.leg_kp * (target - q_leg) - control.leg_kd * qd[..., :12]
    wheel = control.wheel_kd * (control.wheel_velocity_scale * a[..., 12:] - qd[..., 12:])
    effort = a.new_tensor(tuple(control.leg_effort_limits) + tuple(control.wheel_effort_limits))
    return torch.cat((leg, wheel), -1).clamp(-effort, effort)
