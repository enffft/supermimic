"""Shared physical-state predicates used by M1 Reach evaluation."""

from __future__ import annotations

import torch

from supermimic.kinematics.rotations import quat_apply_inverse


def root_tilt_and_stability(
    root_quaternion_wxyz: torch.Tensor,
    root_height_m: torch.Tensor,
    *,
    minimum_height_m: float,
    maximum_tilt_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upright tilt angle and a mask for a physically stable base."""
    if root_quaternion_wxyz.shape[-1] != 4:
        raise ValueError("root quaternion must have shape [...,4] in wxyz order")
    if root_height_m.shape != root_quaternion_wxyz.shape[:-1]:
        raise ValueError("root height shape must match quaternion leading dimensions")
    gravity_world = torch.zeros_like(root_quaternion_wxyz[..., :3])
    gravity_world[..., 2] = -1.0
    gravity_body = quat_apply_inverse(root_quaternion_wxyz, gravity_world)
    tilt = torch.acos((-gravity_body[..., 2]).clamp(-1.0, 1.0))
    stable = (root_height_m >= minimum_height_m) & (tilt <= maximum_tilt_rad)
    return tilt, stable
