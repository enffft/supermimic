"""Decode Future31 to physical M1 states without CPU/NumPy detours."""

from __future__ import annotations

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_canonical,
    quat_inv,
    quat_mul,
    quat_normalize,
    rotvec_from_quat,
)
from supermimic.motion_prior.features import FUTURE31_DIM
from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS, M1_LEG_JOINTS


def future31_to_m1_action22(
    future: torch.Tensor,
    *,
    robot_spec,
    return_diagnostics: bool = False,
):
    """Convert a V3.3 Future31 plan to normalized, interleaved M1 Action22.

    Future31 stores 18 position targets in ``M1_AMP_JOINTS`` order followed
    by four wheel angular-rate targets in ``robot_spec.wheel_joint_names``
    order. Action22 instead follows ``robot_spec.joint_names`` (the M1
    interleaved leg/wheel order followed by the Piper arm). This is a
    candidate actuator setpoint conversion, not an inverse-dynamics action.

    The returned action is bounded for execution. With ``return_diagnostics``
    the unbounded inverse-map output is also returned, so losses can retain a
    useful gradient when a prior predicts outside the V3 action envelope.
    """
    from supermimic.tasks.action_contract import inverse_m1_setpoint_action

    if future.ndim != 3 or future.shape[1:] != (10, FUTURE31_DIM):
        raise ValueError("Future31 must have shape [B,10,31]")
    if robot_spec.name != "m1_piper" or robot_spec.action_contract_version != "m1_mixed_action_v3":
        raise ValueError("Future31 Action22 conversion requires the M1 V3 action contract")
    if not torch.isfinite(future).all():
        raise ValueError("Future31 contains non-finite values")
    if tuple(robot_spec.joint_names) != tuple(M1_JOINTS):
        raise ValueError("M1 V3 Action22 requires the canonical interleaved M1_JOINTS order")
    if tuple(robot_spec.amp_joint_names) != tuple(M1_AMP_JOINTS):
        raise ValueError("M1 V3 Action22 requires the canonical M1_AMP_JOINTS order")
    expected_wheels = tuple(name for name in M1_LEG_JOINTS if name.endswith("FOOT_JOINT"))
    if tuple(robot_spec.wheel_joint_names) != expected_wheels:
        raise ValueError("M1 V3 wheel order must match FAR,FBL,RAR,RBL Future31 order")

    batch = future.shape[0]
    default = future.new_tensor(robot_spec.default_joint_pos)
    joint_target = default.expand(batch, 10, -1).clone()
    amp_ids = torch.tensor(
        [robot_spec.joint_names.index(name) for name in robot_spec.amp_joint_names],
        dtype=torch.long, device=future.device,
    )
    # CopySlices keeps the reference path differentiable with respect to all
    # 18 Future31 position targets while leaving wheel-position placeholders
    # at the default pose (the inverse map overwrites their velocity branch).
    joint_target[..., amp_ids] = future[..., 9:27]
    wheel_rate = future[..., 27:31]
    raw_action = inverse_m1_setpoint_action(
        joint_target, wheel_rate, robot_spec=robot_spec, clip=False,
    )
    action = raw_action.clamp(-1.0, 1.0)

    wheel_mask = torch.tensor(
        [name in robot_spec.wheel_joint_names for name in robot_spec.joint_names],
        dtype=torch.bool, device=future.device,
    )
    bounds = future.new_tensor(robot_spec.position_target_bounds)
    q = joint_target[..., ~wheel_mask]
    q_bounds = bounds[~wheel_mask]
    q_below = (q_bounds[:, 0] - q).clamp_min(0.0)
    q_above = (q - q_bounds[:, 1]).clamp_min(0.0)
    wheel_scale = future.new_tensor(robot_spec.action_scales)[
        torch.tensor(
            [robot_spec.joint_names.index(name) for name in robot_spec.wheel_joint_names],
            dtype=torch.long, device=future.device,
        )
    ]
    wheel_excess = (wheel_rate.abs() - wheel_scale).clamp_min(0.0)
    preclip_excess = (raw_action.abs() - 1.0).clamp_min(0.0)
    # Smooth, scale-free penalty used by Prior optimization. The clamped
    # output alone has zero gradient beyond the envelope; this term does not.
    diagnostics = {
        "out_of_action_envelope_penalty": preclip_excess.square().mean(),
        "out_of_action_envelope_fraction": (preclip_excess > 0).to(future.dtype).mean(),
        "position_envelope_violation_fraction": ((q_below > 0) | (q_above > 0)).to(future.dtype).mean(),
        "position_envelope_violation_max_rad": torch.maximum(q_below, q_above).amax(),
        "wheel_rate_violation_fraction": (wheel_excess > 0).to(future.dtype).mean(),
        "wheel_rate_violation_max_rad_s": wheel_excess.amax(),
        "position_action_saturation_fraction": (preclip_excess[..., ~wheel_mask] > 0).to(future.dtype).mean(),
        "wheel_action_saturation_fraction": (preclip_excess[..., wheel_mask] > 0).to(future.dtype).mean(),
    }
    if return_diagnostics:
        return action, raw_action, diagnostics
    return action


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """AMP-compatible first-two-column Gram-Schmidt rotation decoder."""
    if value.shape[-1] != 6:
        raise ValueError("rotation6d input must have six features")
    a, b = value[..., :3], value[..., 3:6]
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    b = b - (a * b).sum(-1, keepdim=True) * a
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    c = torch.cross(a, b, dim=-1)
    return torch.stack((a, b, c), dim=-1)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Stable differentiable matrix→wxyz quaternion conversion."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    # Keep the unused candidate branches differentiable too: sqrt(0) can
    # produce infinite derivatives that become NaN after gather's zero-gradient
    # masking on an unselected branch.
    q_abs = torch.sqrt(torch.clamp(torch.stack((
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ), dim=-1), min=1e-12))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
        torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
        torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
    ), dim=-2)
    candidates = candidates / (2.0 * q_abs[..., :, None].clamp_min(1.0e-8))
    choice = q_abs.argmax(dim=-1)
    gather = choice[..., None, None].expand(*choice.shape, 1, 4)
    q = candidates.gather(-2, gather).squeeze(-2)
    return quat_canonical(quat_normalize(q))


def decode_future31(state: torch.Tensor, future: torch.Tensor, *, dt: float = 0.02) -> torch.Tensor:
    """Decode ``[B,10,31]`` to physical-state references ``[B,10,57]``.

    Root translation/rotation are relative to the initial base's yaw heading.
    Leg, Piper and wheel fields are mapped by canonical names (wheel joints are
    interleaved in the physical State57 layout).
    """
    if state.ndim != 2 or state.shape[-1] != 57:
        raise ValueError("initial M1 state must be [B,57]")
    if future.ndim != 3 or future.shape[0] != state.shape[0] or future.shape[1:] != (10, FUTURE31_DIM):
        raise ValueError("Future31 must be [B,10,31]")
    if state.device != future.device or state.dtype != future.dtype:
        raise ValueError("state and Future31 must share device and dtype")
    if not torch.isfinite(state).all() or not torch.isfinite(future).all():
        raise ValueError("state and Future31 must be finite")
    root_heading = heading_quat(state[..., 3:7])
    heading_seq = root_heading[:, None].expand(-1, 10, -1)
    root_pos = state[..., None, :3] + quat_apply(heading_seq, future[..., 0:3])
    relative_rotation = matrix_to_quaternion(rotation_6d_to_matrix(future[..., 3:9]))
    root_quat = quat_normalize(quat_mul(heading_seq, relative_rotation))

    joint_pos = state[..., None, 7:29].expand(-1, 10, -1).clone()
    amp_names = list(M1_AMP_JOINTS)
    for i, name in enumerate(amp_names):
        joint_pos[..., M1_JOINTS.index(name)] = future[..., 9 + i] if i < 12 else future[..., 21 + i - 12]
    joint_vel = torch.zeros_like(joint_pos)
    prev_q = torch.cat((state[:, None, 7:29], joint_pos[:, :-1]), dim=1)
    joint_vel = (joint_pos - prev_q) / float(dt)
    wheel_names = [n for n in M1_LEG_JOINTS if n.endswith("FOOT_JOINT")]
    for i, name in enumerate(wheel_names):
        joint_vel[..., M1_JOINTS.index(name)] = future[..., 27 + i]

    root_lin_vel_w = torch.zeros_like(root_pos)
    prev_pos = torch.cat((state[:, None, :3], root_pos[:, :-1]), dim=1)
    root_lin_vel_w = (root_pos - prev_pos) / float(dt)
    prev_quat = torch.cat((state[:, None, 3:7], root_quat[:, :-1]), dim=1)
    delta_quat = quat_mul(quat_inv(prev_quat), root_quat)
    root_ang_vel_b = rotvec_from_quat(delta_quat) / float(dt)
    qpos = torch.cat((root_pos, root_quat, joint_pos), dim=-1)
    return torch.cat((qpos, root_lin_vel_w, root_ang_vel_b, joint_vel), dim=-1)


def reference_tracking_errors(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-component, unit-normalized tracking losses over aligned horizon rows."""
    from supermimic.kinematics.rotations import quat_error_magnitude
    from supermimic.robots.spec import M1_JOINTS

    if predicted.shape != reference.shape or predicted.shape[-1] != 57:
        raise ValueError("predicted and reference state sequences must both be [...,57]")
    leg_ids = [M1_JOINTS.index(n) for n in M1_AMP_JOINTS[:12]]
    arm_ids = [M1_JOINTS.index(n) for n in M1_AMP_JOINTS[12:]]
    wheel_ids = [M1_JOINTS.index(n) for n in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")]
    def mean_rows(value: torch.Tensor) -> torch.Tensor:
        if valid_mask is None:
            return value.mean()
        mask = valid_mask.to(device=value.device, dtype=value.dtype)
        if mask.shape != value.shape[:mask.ndim]:
            raise ValueError("reference tracking valid_mask must match leading sample dimensions")
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    root_pos = mean_rows(((predicted[..., :3] - reference[..., :3]) / 0.5).square().mean(-1))
    root_rot = mean_rows((quat_error_magnitude(predicted[..., 3:7], reference[..., 3:7]) / 0.5).square())
    q_pred, q_ref = predicted[..., 7:29], reference[..., 7:29]
    qd_pred, qd_ref = predicted[..., 35:57], reference[..., 35:57]
    return {
        "root_position": root_pos,
        "root_rotation": root_rot,
        "leg_position": mean_rows(((q_pred[..., leg_ids] - q_ref[..., leg_ids]) / 2.0).square().mean(-1)),
        "arm_position": mean_rows(((q_pred[..., arm_ids] - q_ref[..., arm_ids]) / 2.0).square().mean(-1)),
        "wheel_velocity": mean_rows(((qd_pred[..., wheel_ids] - qd_ref[..., wheel_ids]) / 10.0).square().mean(-1)),
    }
