"""Strict, device-native conversion between M1 State57 and AMP CVAE features."""

from __future__ import annotations

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    matrix_from_quat,
    quat_apply_inverse,
    quat_mul,
)
from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS, M1_LEG_JOINTS

HISTORY = 4
STATE57_DIM = 57
STATE53_DIM = 53
GOAL8_DIM = 8
FUTURE31_DIM = 31
WHEEL_NAMES = tuple(n for n in M1_LEG_JOINTS if n.endswith("FOOT_JOINT"))
_AMP_Q_INDEX = torch.tensor([M1_JOINTS.index(n) for n in M1_AMP_JOINTS], dtype=torch.long)
_AMP_WHEEL_INDEX = torch.tensor([M1_JOINTS.index(n) for n in WHEEL_NAMES], dtype=torch.long)


def _check_state(state: torch.Tensor) -> None:
    if state.ndim < 2 or state.shape[-1] != STATE57_DIM:
        raise ValueError(f"M1 physical state must end in 57, got {tuple(state.shape)}")
    if not state.is_floating_point() or not torch.isfinite(state).all():
        raise ValueError("M1 state must be finite floating point")


def state_history_to_cvae53(state_history: torch.Tensor) -> torch.Tensor:
    """Map ``[B,4,57]`` physical history to AMP ``[B,4,53]``.

    Position and quaternion are env-local ``wxyz``; root linear velocity in
    State57 is world-frame and is rotated into the full root body frame because
    AMP raw Frames store body-frame linear and angular velocity. Heading-relative
    rotation6d remains yaw-only and is referenced to the newest observed frame.
    Joint and wheel ordering is explicitly name based.
    """
    _check_state(state_history)
    if state_history.ndim != 3 or state_history.shape[1] != HISTORY:
        raise ValueError(f"state_history must be [B,4,57], got {tuple(state_history.shape)}")
    root_q = state_history[..., 3:7]
    root_q = root_q / root_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    heading_now = heading_quat(root_q[:, -1])
    heading_inv = heading_now.clone()
    heading_inv[..., 1:] *= -1
    relative_q = quat_mul(heading_inv[:, None].expand_as(root_q), root_q)
    # State57 is [root qpos 7, joints 22, root qvel 6, joint qvel 22].
    # Keep these offsets explicit: 36:39/39:42 would only be correct for a
    # different robot with 29 joints and silently read M1 joint velocities.
    body_velocity = quat_apply_inverse(root_q, state_history[..., 29:32])
    root_ang_vel_body = state_history[..., 32:35]
    joint_pos = state_history[..., 7:29].index_select(-1, _AMP_Q_INDEX.to(state_history.device))
    joint_vel = state_history[..., 35:57].index_select(-1, _AMP_Q_INDEX.to(state_history.device))
    wheel_rate = state_history[..., 35:57].index_select(-1, _AMP_WHEEL_INDEX.to(state_history.device))
    relative_matrix = matrix_from_quat(relative_q)
    # AMP stores the first matrix column followed by the second matrix column.
    # A row-major reshape of [..., :, :2] interleaves columns and changes the
    # learned CVAE input schema even though it still has six values.
    relative_rot6d = torch.cat((relative_matrix[..., :, 0], relative_matrix[..., :, 1]), dim=-1)
    result = torch.cat((
        state_history[..., 2:3], relative_rot6d, body_velocity,
        root_ang_vel_body, joint_pos, joint_vel, wheel_rate,
    ), dim=-1)
    if result.shape[-1] != STATE53_DIM:
        raise AssertionError(f"adapter produced {result.shape[-1]} features, expected 53")
    return result


def task_goal8(state: torch.Tensor, task_type: torch.Tensor,
               velocity_command_body: torch.Tensor,
               fixed_world_target: torch.Tensor) -> torch.Tensor:
    """Encode L/M Goal8; M target stays fixed in env-local world coordinates.

    ``task_type`` is integer 0=L, 1=M. LM is deliberately rejected in v1.
    """
    _check_state(state)
    batch = state.shape[0]
    task_type = task_type.reshape(batch).to(device=state.device, dtype=torch.long)
    velocity_command_body = velocity_command_body.to(device=state.device, dtype=state.dtype)
    fixed_world_target = fixed_world_target.to(device=state.device, dtype=state.dtype)
    if velocity_command_body.shape != (batch, 3) or fixed_world_target.shape != (batch, 3):
        raise ValueError("velocity command and fixed world target must both be [B,3]")
    if bool(((task_type < 0) | (task_type > 1)).any()):
        raise ValueError("v1 supports only task_type 0=L or 1=M")
    ee_delta_heading = quat_apply_inverse(heading_quat(state[..., 3:7]), fixed_world_target - state[..., :3])
    is_l = task_type == 0
    velocity = torch.where(is_l[:, None], velocity_command_body, torch.zeros_like(velocity_command_body))
    ee_delta = torch.where(is_l[:, None], torch.zeros_like(ee_delta_heading), ee_delta_heading)
    masks = torch.stack((is_l.to(state.dtype), (~is_l).to(state.dtype)), dim=-1)
    goal = torch.cat((velocity, ee_delta, masks), dim=-1)
    if goal.shape != (batch, GOAL8_DIM) or not torch.isfinite(goal).all():
        raise ValueError("invalid Goal8 tensor")
    return goal
