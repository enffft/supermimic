"""Differentiable Go2 foot collision centers and velocities in RobotSpec order.

Geometry matches the Go2 URDF: FL, FR, RL, RR; hip X, thigh/calf Y.
The foot sphere has radius 0.022 m and its center is -0.002 m along calf X.
"""

from __future__ import annotations

import torch

from supermimic.kinematics.rotations import quat_apply

GO2_FOOT_RADIUS = 0.022


def go2_feet_world(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (..., 4, 3) sphere center positions and world linear velocities.

    Uses joint rates and rigid-body velocity propagation, avoiding finite
    differences across resets and remaining valid at arbitrary leading dimensions.
    Root linear velocity is world-frame; root angular velocity is body-frame.
    """
    q = state[..., 7:19].unflatten(-1, (4, 3))
    qd = state[..., 25:37].unflatten(-1, (4, 3))
    hip, thigh, calf = q.unbind(-1)
    hip_d, thigh_d, calf_d = qd.unbind(-1)
    sh, ch = hip.sin(), hip.cos()
    st, ct = thigh.sin(), thigh.cos()
    sk, ck = (thigh + calf).sin(), (thigh + calf).cos()
    knee_d = thigh_d + calf_d
    side = state.new_tensor((1.0, -1.0, 1.0, -1.0))
    x = -0.213 * (st + sk) - 0.002 * ck
    y = 0.0955 * side
    z = -0.213 * (ct + ck) + 0.002 * sk
    rotated_y = ch * y - sh * z
    rotated_z = sh * y + ch * z
    anchor = state.new_tensor((
        (0.1934, 0.0465, 0.0), (0.1934, -0.0465, 0.0),
        (-0.1934, 0.0465, 0.0), (-0.1934, -0.0465, 0.0),
    ))
    pos_b = torch.stack((x, rotated_y, rotated_z), dim=-1) + anchor
    xd = -0.213 * (ct * thigh_d + ck * knee_d) + 0.002 * sk * knee_d
    zd = 0.213 * (st * thigh_d + sk * knee_d) + 0.002 * ck * knee_d
    vel_b = torch.stack((
        xd, -sh * zd - hip_d * rotated_z, ch * zd + hip_d * rotated_y,
    ), dim=-1)
    omega_b = state[..., None, 22:25].expand_as(pos_b)
    quat = state[..., None, 3:7].expand(*pos_b.shape[:-1], 4)
    pos_w = state[..., None, :3] + quat_apply(quat, pos_b)
    vel_w = state[..., None, 19:22] + quat_apply(
        quat, vel_b + torch.cross(omega_b, pos_b, dim=-1),
    )
    return pos_w, vel_w
