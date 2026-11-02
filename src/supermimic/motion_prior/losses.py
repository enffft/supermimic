"""Independent Planner and Tracking Actor objectives for M1 joint learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from supermimic.kinematics.rotations import (
    heading_yaw,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    rotvec_from_quat,
    wrap_to_pi,
)
from supermimic.motion_prior.reference import reference_tracking_errors
from supermimic.motion_prior.velocity import velocity_command_errors
from supermimic.robots.spec import M1_JOINTS, M1_LEG_JOINTS


@dataclass
class M1JointLossCfg:
    # Planner task objective
    l_linear_velocity_weight: float = 1.0
    l_yaw_rate_weight: float = 0.5
    m_terminal_weight: float = 1.0
    m_path_weight: float = 0.1
    m_progress_weight: float = 1.0
    m_base_drift_weight: float = 0.1
    # Regularizers. Energy remains disabled until a differentiable torque model is validated.
    energy_weight: float = 0.0
    control_effort_surrogate_weight: float = 0.0
    feasibility_weight: float = 1.0
    # Calibrated on reset-state CVAE prior_mean: at 0.001, direct reference
    # smoothness gradients were 5.6x the L task gradient; 0.0001 reduced this
    # proxy ratio to 0.56. Recheck after task-specific WM/Actor calibration.
    smoothness_weight: float = 0.0001
    anchor_weight: float = 0.02
    supervised_weight: float = 0.0
    # Actor tracking weights (losses are already normalized by physical scale).
    actor_root_position_weight: float = 0.5
    actor_root_rotation_weight: float = 0.25
    actor_leg_weight: float = 0.25
    actor_arm_weight: float = 1.0
    actor_wheel_weight: float = 0.5
    actor_tcp_weight: float = 0.0  # reserved; TCP reference loss is not implemented in v1
    actor_stability_weight: float = 1.0
    actor_control_weight: float = 0.01
    # Explicitly named physical thresholds used by soft feasibility/smoothness terms.
    minimum_base_height_m: float = 0.22
    maximum_tilt_rad: float = 0.9
    maximum_wheel_rate_rad_s: float = 5.0
    maximum_leg_velocity_rad_s: float = 16.75  # M1 URDF limit for the 12 leg position joints
    maximum_arm_velocity_rad_s: float = 10.0  # Piper URDF limit for the six arm position joints
    maximum_joint_acceleration_rad_s2: float = 80.0
    maximum_root_jerk_m_s3: float = 75.0
    maximum_root_angular_jerk_rad_s3: float = 600.0
    maximum_joint_jerk_rad_s3: float = 2000.0
    maximum_wheel_jerk_rad_s3: float = 4000.0
    maximum_root_acceleration_m_s2: float = 12.0
    maximum_root_angular_acceleration_rad_s2: float = 30.0
    control_dt: float = 0.02


def _joint_mask(device: torch.device) -> torch.Tensor:
    mask = torch.ones(len(M1_JOINTS), dtype=torch.bool, device=device)
    for name in M1_LEG_JOINTS:
        if name.endswith("FOOT_JOINT"):
            mask[M1_JOINTS.index(name)] = False
    return mask


def _joint_velocity_limits(device: torch.device, cfg: M1JointLossCfg) -> torch.Tensor:
    limits = torch.full((len(M1_JOINTS),), cfg.maximum_leg_velocity_rad_s, device=device)
    for name in M1_JOINTS:
        if name.startswith("piper_x_joint"):
            limits[M1_JOINTS.index(name)] = cfg.maximum_arm_velocity_rad_s
        elif name.endswith("FOOT_JOINT"):
            # Conservative planner operating cap below the M1 asset's 110 rad/s limit.
            limits[M1_JOINTS.index(name)] = cfg.maximum_wheel_rate_rad_s
    return limits


def _task_planner_terms(states: torch.Tensor, task_type: torch.Tensor,
                        velocity_command: torch.Tensor, fixed_world_target: torch.Tensor,
                        fk, tcp_index: int, cfg: M1JointLossCfg) -> dict[str, torch.Tensor]:
    # states [H+1,B,57], task labels and task contexts are episode-constant.
    state = states[1:]
    batch = states.shape[1]
    task_type = task_type.reshape(batch).to(device=states.device, dtype=torch.long)
    l_mask = (task_type == 0).to(states.dtype)[None, :]
    m_mask = (task_type == 1).to(states.dtype)[None, :]
    velocity_command = velocity_command.to(device=states.device, dtype=states.dtype)
    fixed_world_target = fixed_world_target.to(device=states.device, dtype=states.dtype)
    linear_error, _, yaw_rate_error, _ = velocity_command_errors(
        state, velocity_command[None].expand(state.shape[0], -1, -1),
    )
    l_linear = F.smooth_l1_loss(
        linear_error, torch.zeros_like(linear_error), reduction="none",
    ).div(0.5).mean(-1)
    l_yaw = F.smooth_l1_loss(
        yaw_rate_error, torch.zeros_like(yaw_rate_error), reduction="none",
    )
    l_per_step = cfg.l_linear_velocity_weight * l_linear + cfg.l_yaw_rate_weight * l_yaw
    l_task = (l_per_step * l_mask).sum() / (l_mask.sum() * state.shape[0]).clamp_min(1.0)

    if fk is None:
        raise ValueError("M planner objective requires the validated M1 differentiable FK model")
    flat = state.reshape(-1, 57)
    qpos_dim = 29
    bodies = fk(flat[:, :qpos_dim], flat[:, qpos_dim:])
    tcp = bodies["pos"][:, tcp_index].reshape(state.shape[0], batch, 3)
    distance = torch.linalg.vector_norm(tcp - fixed_world_target[None], dim=-1)
    d0 = torch.linalg.vector_norm(
        fk(states[0, :, :qpos_dim], states[0, :, qpos_dim:])["pos"][:, tcp_index] - fixed_world_target,
        dim=-1,
    )
    d_final = distance[-1]
    m_path = distance.mean(0)
    m_progress = d0 - d_final
    m_task = (
        cfg.m_terminal_weight * (d_final / 0.10)
        + cfg.m_path_weight * (m_path / 0.10)
        - cfg.m_progress_weight * (m_progress / 0.10)
    )
    m_task = (m_task * m_mask[0]).sum() / m_mask[0].sum().clamp_min(1.0)

    initial_xy = states[0, :, :2]
    base_xy = state[..., :2]
    base_xy_drift = torch.linalg.vector_norm(base_xy - initial_xy[None], dim=-1) / 0.25
    yaw = heading_yaw(state[..., 3:7])
    yaw0 = heading_yaw(states[0, :, 3:7])
    base_yaw_drift = wrap_to_pi(yaw - yaw0[None]).abs() / 0.5
    m_drift = ((base_xy_drift.square() + base_yaw_drift.square()) * m_mask).sum() / (
        m_mask.sum() * state.shape[0]
    ).clamp_min(1.0)
    return {
        "task_l": l_task,
        "task_l_linear_velocity": (l_linear * l_mask).sum() / (l_mask.sum() * state.shape[0]).clamp_min(1.0),
        "task_l_yaw_rate": (l_yaw * l_mask).sum() / (l_mask.sum() * state.shape[0]).clamp_min(1.0),
        "task_m": m_task,
        "task_progress_m": (m_progress * m_mask[0]).sum() / m_mask[0].sum().clamp_min(1.0),
        "terminal_error_m": (d_final * m_mask[0]).sum() / m_mask[0].sum().clamp_min(1.0),
        "base_drift_m": (
            (torch.linalg.vector_norm(base_xy - initial_xy[None], dim=-1) * m_mask).sum()
            / (m_mask.sum() * state.shape[0]).clamp_min(1.0)
        ),
        "m_base_drift": cfg.m_base_drift_weight * m_drift,
        "tcp_distance_sequence": distance,
    }


def _feasibility(states: torch.Tensor, soft_limits: torch.Tensor, cfg: M1JointLossCfg) -> torch.Tensor:
    state = states[1:]
    low = F.relu(cfg.minimum_base_height_m - state[..., 2]) / 0.05
    gravity = state.new_zeros(state.shape[:-1] + (3,))
    gravity[..., 2] = -1.0
    gravity_body = quat_apply_inverse(state[..., 3:7], gravity)
    tilt = torch.acos((-gravity_body[..., 2]).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    tilt_cost = F.relu(tilt - cfg.maximum_tilt_rad) / 0.25
    q = state[..., 7:29]
    mask = _joint_mask(state.device)
    q_limits = soft_limits[mask]
    below = F.relu(q_limits[None, None, :, 0] - q[..., mask]) / 0.25
    above = F.relu(q[..., mask] - q_limits[None, None, :, 1]) / 0.25
    qlim = (below.square() + above.square()).mean()
    velocity_limits = _joint_velocity_limits(state.device, cfg)
    velocity_violation = F.relu(state[..., 35:57].abs() - velocity_limits) / velocity_limits
    wheel_ids = [M1_JOINTS.index(n) for n in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")]
    wheel_speed = F.relu(state[..., 35:57][..., wheel_ids].abs() - cfg.maximum_wheel_rate_rad_s) / cfg.maximum_wheel_rate_rad_s
    return (
        low.square().mean() + tilt_cost.square().mean() + qlim
        + velocity_violation.square().mean() + wheel_speed.square().mean()
    )


def _reference_smoothness_terms(
    initial: torch.Tensor, reference: torch.Tensor, cfg: M1JointLossCfg,
) -> dict[str, torch.Tensor]:
    """Unit-scaled acceleration and jerk regularizers with current state as t=0."""
    from supermimic.motion_prior.reference import decode_future31

    decoded = decode_future31(initial, reference, dt=cfg.control_dt)
    root_path = torch.cat((initial[:, None, :3], decoded[..., :3]), dim=1)
    root_vel = torch.diff(root_path, dim=1) / cfg.control_dt
    root_acc = torch.diff(root_vel, dim=1) / cfg.control_dt
    root_term = (F.relu(root_acc.norm(dim=-1) - cfg.maximum_root_acceleration_m_s2) / cfg.maximum_root_acceleration_m_s2).square().mean()
    root_jerk = torch.diff(root_acc, dim=1) / cfg.control_dt
    root_jerk_term = (F.relu(root_jerk.norm(dim=-1) - cfg.maximum_root_jerk_m_s3) / cfg.maximum_root_jerk_m_s3).square().mean()
    quat_path = torch.cat((initial[:, None, 3:7], decoded[..., 3:7]), dim=1)
    root_delta = quat_mul(quat_inv(quat_path[:, :-1]), quat_path[:, 1:])
    root_omega = rotvec_from_quat(root_delta) / cfg.control_dt
    root_alpha = torch.diff(root_omega, dim=1) / cfg.control_dt
    root_rot_term = (F.relu(root_alpha.norm(dim=-1) - cfg.maximum_root_angular_acceleration_rad_s2)
                     / cfg.maximum_root_angular_acceleration_rad_s2).square().mean()
    root_angular_jerk = torch.diff(root_alpha, dim=1) / cfg.control_dt
    root_angular_jerk_term = (
        F.relu(root_angular_jerk.norm(dim=-1) - cfg.maximum_root_angular_jerk_rad_s3)
        / cfg.maximum_root_angular_jerk_rad_s3
    ).square().mean()
    q_path = torch.cat((initial[:, None, 7:29], decoded[..., 7:29]), dim=1)
    q_vel = torch.diff(q_path, dim=1) / cfg.control_dt
    q_acc = torch.diff(q_vel, dim=1) / cfg.control_dt
    joint_mask = _joint_mask(q_path.device)
    q_term = (F.relu(q_acc[..., joint_mask].abs() - cfg.maximum_joint_acceleration_rad_s2)
              / cfg.maximum_joint_acceleration_rad_s2).square().mean()
    wheel_ids = [M1_JOINTS.index(n) for n in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")]
    wheel_rate = reference[..., 27:31]
    wheel_path = torch.cat((initial[:, None, 35:57][:, :, wheel_ids], wheel_rate), dim=1)
    wheel_acc = torch.diff(wheel_path, dim=1) / cfg.control_dt
    wheel_term = (F.relu(wheel_acc.abs() - cfg.maximum_joint_acceleration_rad_s2) / cfg.maximum_joint_acceleration_rad_s2).square().mean()
    wheel_jerk = torch.diff(wheel_acc, dim=1) / cfg.control_dt
    wheel_jerk_term = (F.relu(wheel_jerk.abs() - cfg.maximum_wheel_jerk_rad_s3)
                       / cfg.maximum_wheel_jerk_rad_s3).square().mean()
    joint_jerk = torch.diff(q_acc[..., joint_mask], dim=1) / cfg.control_dt
    joint_jerk_term = (F.relu(joint_jerk.abs() - cfg.maximum_joint_jerk_rad_s3)
                       / cfg.maximum_joint_jerk_rad_s3).square().mean()
    return {
        "smooth_root_acceleration": root_term,
        "smooth_root_jerk": root_jerk_term,
        "smooth_root_angular_acceleration": root_rot_term,
        "smooth_root_angular_jerk": root_angular_jerk_term,
        "smooth_joint_acceleration": q_term,
        "smooth_joint_jerk": joint_jerk_term,
        "smooth_wheel_acceleration": wheel_term,
        "smooth_wheel_jerk": wheel_jerk_term,
    }


def _reference_smoothness(initial: torch.Tensor, reference: torch.Tensor, cfg: M1JointLossCfg) -> torch.Tensor:
    return torch.stack(tuple(_reference_smoothness_terms(initial, reference, cfg).values())).sum()


def _anchor_loss(reference: torch.Tensor, initial_reference: torch.Tensor) -> torch.Tensor:
    if reference.shape != initial_reference.shape:
        raise ValueError("initial Prior anchor output shape mismatch")
    scales = reference.new_tensor([0.5] * 3 + [1.0] * 6 + [2.0] * 12 + [2.0] * 6 + [10.0] * 4)
    return ((reference - initial_reference) / scales).square().mean()


def planner_losses(
    states: torch.Tensor,
    actions: torch.Tensor,
    initial: torch.Tensor,
    reference: torch.Tensor,
    initial_reference: torch.Tensor,
    task_type: torch.Tensor,
    velocity_command: torch.Tensor,
    fixed_world_target: torch.Tensor,
    *,
    fk,
    tcp_index: int,
    soft_limits: torch.Tensor,
    cfg: M1JointLossCfg,
) -> dict[str, torch.Tensor]:
    """Motion Prior objective; task error is independent of Actor trajectory error."""
    task = _task_planner_terms(states, task_type, velocity_command, fixed_world_target, fk, tcp_index, cfg)
    task_loss = task["task_l"] + task["task_m"] + task["m_base_drift"]
    feasibility = _feasibility(states, soft_limits, cfg)
    smooth_terms = _reference_smoothness_terms(initial, reference, cfg)
    smoothness = torch.stack(tuple(smooth_terms.values())).sum()
    anchor = _anchor_loss(reference, initial_reference)
    # This is an explicitly named control-effort surrogate. It is not a torque,
    # electrical-energy or mechanical-power estimate and defaults to disabled.
    effort_surrogate = actions.square().mean() + 0.25 * torch.diff(actions, dim=0).square().mean()
    energy = actions.new_zeros(())  # no validated differentiable M1 actuator model is available yet
    total = (
        task_loss + cfg.feasibility_weight * feasibility + cfg.smoothness_weight * smoothness
        + cfg.anchor_weight * anchor + cfg.energy_weight * energy
        + cfg.control_effort_surrogate_weight * effort_surrogate
    )
    return {
        "total": total,
        "task": task_loss,
        "task_l": task["task_l"],
        "task_m": task["task_m"],
        "task_progress": task["task_progress_m"],
        "terminal_error": task["terminal_error_m"],
        "base_drift": task["base_drift_m"],
        "energy": energy,
        "control_effort_surrogate": effort_surrogate,
        "feasibility": feasibility,
        "smoothness": smoothness,
        **smooth_terms,
        "anchor": anchor,
        "supervised": total.new_zeros(()),
    }


def actor_losses(
    states: torch.Tensor,
    reference_states: torch.Tensor,
    actions: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    soft_limits: torch.Tensor,
    cfg: M1JointLossCfg,
) -> dict[str, torch.Tensor]:
    """Actor objective: track fixed Prior reference while retaining safety/control terms."""
    if states.shape != reference_states.shape:
        raise ValueError("Actor state trajectory and fixed reference must align exactly")
    parts = reference_tracking_errors(states, reference_states)
    track = (
        cfg.actor_root_position_weight * parts["root_position"]
        + cfg.actor_root_rotation_weight * parts["root_rotation"]
        + cfg.actor_leg_weight * parts["leg_position"]
        + cfg.actor_arm_weight * parts["arm_position"]
        + cfg.actor_wheel_weight * parts["wheel_velocity"]
    )
    state = states
    low = F.relu(cfg.minimum_base_height_m - state[..., 2]) / 0.05
    gravity = state.new_zeros(state.shape[:-1] + (3,))
    gravity[..., 2] = -1.0
    gravity_body = quat_apply_inverse(state[..., 3:7], gravity)
    tilt = torch.acos((-gravity_body[..., 2]).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    tilt_cost = F.relu(tilt - cfg.maximum_tilt_rad) / 0.25
    q = state[..., 7:29]
    mask = _joint_mask(state.device)
    limits = soft_limits[mask]
    limit = (
        F.relu(limits[:, 0] - q[..., mask]).square()
        + F.relu(q[..., mask] - limits[:, 1]).square()
    ).mean()
    stability = low.square().mean() + tilt_cost.square().mean() + limit
    action_path = torch.cat((previous_action[None], actions), dim=0)
    action_rate = torch.diff(action_path, dim=0).square().mean()
    qd = state[..., 35:57]
    joint_speed = (qd / _joint_velocity_limits(state.device, cfg)).square().mean()
    joint_acc = (torch.diff(qd, dim=0) / cfg.control_dt / 80.0).square().mean() if qd.shape[0] > 1 else qd.new_zeros(())
    saturation = F.relu(actions.abs() - 0.95).square().mean()
    control = action_rate + 0.1 * joint_speed + 0.01 * joint_acc + 0.1 * saturation
    total = track + cfg.actor_stability_weight * stability + cfg.actor_control_weight * control
    return {
        "total": total, "tracking": track, **parts,
        "stability": stability, "control": control,
        "action_rate": action_rate, "joint_speed": joint_speed,
        "joint_acceleration": joint_acc, "action_saturation": saturation,
    }
