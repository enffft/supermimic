"""Versioned Go2 contract, independent of the upstream locomotion defaults."""

from pathlib import Path

import torch

JOINTS = tuple(
    f"robot/{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)
DEFAULT_Q = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5)
EFFORT = (23.7, 23.7, 35.55) * 4
DT = 0.02
MODEL = Path(__file__).resolve().parents[3] / "assets/robots/go2_velocity/flat.xml"


def position_setpoint(action, default, scale):
    return default + action * scale


def soft_limits(device="cpu"):
    # Source mjlab soft_joint_pos_limit_factor=0.9, URDF joint limits.
    bounds = torch.tensor(
        [(-1.0472, 1.0472), (-0.5236, 4.5379), (-2.7227, -0.83776)] * 4, device=device
    )
    mid = bounds.mean(-1)
    radius = (bounds[:, 1] - bounds[:, 0]) * 0.45
    return torch.stack((mid - radius, mid + radius), -1)
