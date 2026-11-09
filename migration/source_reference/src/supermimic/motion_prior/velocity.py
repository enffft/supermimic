"""Shared M1 body-velocity error semantics for planning and real evaluation."""

from __future__ import annotations

import torch

from supermimic.kinematics.rotations import quat_apply_inverse


def velocity_command_errors(
    state: torch.Tensor, command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return body XY error, its squared norm, yaw-rate error, and body vz.

    ``command`` is ``[vx, vy, wz]``: its third component is a yaw-rate
    command in rad/s, never a vertical linear-velocity command. State is the
    physical M1 State57 and root linear velocity is stored in world axes.
    """
    if state.shape[-1] != 57 or command.shape[-1] != 3:
        raise ValueError("expected M1 State57 and [vx, vy, wz] command")
    try:
        torch.broadcast_shapes(state.shape[:-1], command.shape[:-1])
    except RuntimeError as exc:
        raise ValueError("state and velocity command batch dimensions do not broadcast") from exc
    return velocity_command_errors_from_components(
        state[..., 3:7], state[..., 29:32], state[..., 34], command,
    )


def velocity_command_errors_from_components(
    root_quaternion_wxyz: torch.Tensor,
    root_linear_velocity_world: torch.Tensor,
    yaw_rate_body: torch.Tensor,
    command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Component form used by live MuJoCo command diagnostics."""
    if root_quaternion_wxyz.shape[-1] != 4 or root_linear_velocity_world.shape[-1] != 3:
        raise ValueError("expected root quaternion [...,4] and root linear velocity [...,3]")
    command = command.to(device=root_quaternion_wxyz.device, dtype=root_quaternion_wxyz.dtype)
    body_velocity = quat_apply_inverse(root_quaternion_wxyz, root_linear_velocity_world)
    yaw_rate_body = yaw_rate_body.to(device=body_velocity.device, dtype=body_velocity.dtype)
    xy_error = body_velocity[..., :2] - command[..., :2]
    xy_squared_error = xy_error.square().sum(-1)
    yaw_rate_error = yaw_rate_body - command[..., 2]
    return xy_error, xy_squared_error, yaw_rate_error, body_velocity[..., 2]


def task_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """Mean only valid task rows; return ``None`` when this task has no rows."""
    if values.shape[:mask.ndim] != mask.shape:
        raise ValueError("task mask must match the leading value dimensions")
    selected = values[mask.to(device=values.device, dtype=torch.bool)]
    if selected.numel() == 0:
        return None
    return selected.mean()
