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
    planner_task_weight: float = 1.0
    planner_actor_task_weight: float = 0.25
    planner_dynamics_weight: float = 0.5
    termination_failure_weight: float = 1.0
    out_of_envelope_weight: float = 0.1
    actor_task_weight: float = 1.0
    actor_bc_weight: float = 0.05
    # Task errors are normalized by configurable physical tolerances.  A unit
    # loss therefore means approximately one stated tolerance violation.
    l_vx_tolerance_m_s: float = 0.05
    l_vy_tolerance_m_s: float = 0.05
    l_wz_tolerance_rad_s: float = 0.10
    m_tcp_distance_tolerance_m: float = 0.08
    m_tcp_speed_tolerance_m_s: float = 0.05
    m_near_goal_radius_m: float = 0.16
    m_near_goal_transition_m: float = 0.02
    m_base_translation_tolerance_m: float = 0.08
    m_base_yaw_tolerance_rad: float = 0.12
    m_base_velocity_tolerance_m_s: float = 0.05
    m_base_yaw_rate_tolerance_rad_s: float = 0.10
    # Planner task objective.  The M process term excludes the terminal frame;
    # terminal distance and terminal velocity are then each evaluated once.
    l_linear_velocity_weight: float = 1.0
    l_yaw_rate_weight: float = 0.5
    m_terminal_weight: float = 1.0
    m_path_weight: float = 1.0
    m_near_goal_speed_weight: float = 1.0
    m_base_drift_weight: float = 0.1
    m_base_velocity_weight: float = 0.25
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

    def __post_init__(self) -> None:
        positive = (
            "l_vx_tolerance_m_s", "l_vy_tolerance_m_s", "l_wz_tolerance_rad_s",
            "m_tcp_distance_tolerance_m", "m_tcp_speed_tolerance_m_s",
            "m_near_goal_transition_m", "m_base_translation_tolerance_m",
            "m_base_yaw_tolerance_rad", "m_base_velocity_tolerance_m_s",
            "m_base_yaw_rate_tolerance_rad_s", "control_dt",
        )
        invalid = {name: getattr(self, name) for name in positive if getattr(self, name) <= 0}
        if invalid:
            raise ValueError(f"M1 task physical tolerances and control_dt must be positive: {invalid}")
        if self.m_near_goal_radius_m < 0:
            raise ValueError("m_near_goal_radius_m must be nonnegative")


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


def expected_termination_failure_cost(survival_probability: torch.Tensor | None,
                                      *, reference: torch.Tensor) -> torch.Tensor:
    """Expected lost-horizon cost in normalized task-tolerance units.

    ``survival_probability[k]`` is the cumulative probability of reaching the
    successor state at step k. Each lost control interval costs one unit, where
    one unit is one configured task tolerance (for example 0.05 m/s on L or
    0.08 m on M). This is a physical-scale cost, not an uncalibrated large
    penalty. Task costs themselves are separately continued through an
    absorbing state by ``_task_planner_terms``.
    """
    if survival_probability is None:
        return reference.new_zeros(())
    if survival_probability.ndim != 2 or survival_probability.shape[0] < 1:
        raise ValueError("survival_probability must have shape [H,B] with H >= 1")
    survival = survival_probability.clamp(0.0, 1.0)
    return (1.0 - survival).mean()


def _absorbing_state_paths(
    states: torch.Tensor, valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return live and last-safe states for each transition in a fixed horizon."""
    horizon, batch = states.shape[0] - 1, states.shape[1]
    if valid_mask is None:
        valid = torch.ones(horizon, batch, dtype=torch.bool, device=states.device)
    else:
        if valid_mask.shape != (horizon, batch):
            raise ValueError("task valid_mask must have shape [H,B]")
        valid = valid_mask.to(device=states.device, dtype=torch.bool)
    held_before: list[torch.Tensor] = []
    live: list[torch.Tensor] = []
    last_safe = states[0]
    alive_before = torch.ones(batch, dtype=torch.bool, device=states.device)
    for step in range(horizon):
        # The transition that first terminates is excluded from task state
        # tracking; it remains covered by the separate dense safety objective.
        before = torch.where(alive_before[:, None], states[step], last_safe)
        after = states[step + 1]
        after_task = torch.where(valid[step, :, None], after, before)
        held_before.append(before)
        live.append(after_task)
        last_safe = after_task
        alive_before = alive_before & valid[step]
    return torch.stack(live), torch.stack(held_before)


def _expected_with_absorbing(
    live_loss: torch.Tensor, held_loss: torch.Tensor,
    survival_probability: torch.Tensor | None,
) -> torch.Tensor:
    if survival_probability is None:
        return live_loss
    if survival_probability.shape != live_loss.shape:
        raise ValueError("survival_probability must align with per-step task losses [H,B]")
    survival = survival_probability.to(device=live_loss.device, dtype=live_loss.dtype).clamp(0.0, 1.0)
    return survival * live_loss + (1.0 - survival) * held_loss


def _task_planner_terms(states: torch.Tensor, task_type: torch.Tensor,
                        velocity_command: torch.Tensor, fixed_world_target: torch.Tensor,
                        fk, tcp_index: int, cfg: M1JointLossCfg,
                        valid_mask: torch.Tensor | None = None,
                        survival_probability: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    # states [H+1,B,57], task labels and task contexts are episode-constant.
    if states.ndim != 3 or states.shape[-1] != 57 or states.shape[0] < 2:
        raise ValueError("task states must have shape [H+1,B,57]")
    state, held_state = _absorbing_state_paths(states, valid_mask)
    batch = states.shape[1]
    task_type = task_type.reshape(batch).to(device=states.device, dtype=torch.long)
    l_mask = (task_type == 0).to(states.dtype)
    m_mask = (task_type == 1).to(states.dtype)

    def task_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask[None].expand(value.shape[0], -1)
        return (value * expanded).sum() / expanded.sum().clamp_min(1.0)

    def tolerance_loss(error: torch.Tensor, tolerance: float) -> torch.Tensor:
        scaled = error / float(tolerance)
        return F.smooth_l1_loss(scaled, torch.zeros_like(scaled), reduction="none")

    velocity_command = velocity_command.to(device=states.device, dtype=states.dtype)
    fixed_world_target = fixed_world_target.to(device=states.device, dtype=states.dtype)
    command = velocity_command[None].expand(state.shape[0], -1, -1)
    live_xy, _, live_yaw, _ = velocity_command_errors(state, command)
    held_xy, _, held_yaw, _ = velocity_command_errors(held_state, command)
    live_linear = (
        tolerance_loss(live_xy[..., 0], cfg.l_vx_tolerance_m_s)
        + tolerance_loss(live_xy[..., 1], cfg.l_vy_tolerance_m_s)
    ) * 0.5
    held_linear = (
        tolerance_loss(held_xy[..., 0], cfg.l_vx_tolerance_m_s)
        + tolerance_loss(held_xy[..., 1], cfg.l_vy_tolerance_m_s)
    ) * 0.5
    live_yaw_loss = tolerance_loss(live_yaw, cfg.l_wz_tolerance_rad_s)
    held_yaw_loss = tolerance_loss(held_yaw, cfg.l_wz_tolerance_rad_s)
    l_linear = _expected_with_absorbing(live_linear, held_linear, survival_probability)
    l_yaw = _expected_with_absorbing(live_yaw_loss, held_yaw_loss, survival_probability)
    l_per_step = cfg.l_linear_velocity_weight * l_linear + cfg.l_yaw_rate_weight * l_yaw
    l_task = task_mean(l_per_step, l_mask)

    if fk is None:
        raise ValueError("M planner objective requires the validated M1 differentiable FK model")
    flat = torch.cat((state, held_state), dim=0).reshape(-1, 57)
    qpos_dim = 29
    bodies = fk(flat[:, :qpos_dim], flat[:, qpos_dim:])
    horizon = state.shape[0]
    tcp_live, tcp_held = bodies["pos"][:, tcp_index].reshape(2, horizon, batch, 3)
    distance_live = torch.linalg.vector_norm(tcp_live - fixed_world_target[None], dim=-1)
    distance_held = torch.linalg.vector_norm(tcp_held - fixed_world_target[None], dim=-1)
    distance = _expected_with_absorbing(distance_live, distance_held, survival_probability)
    process_live_loss = tolerance_loss(distance_live, cfg.m_tcp_distance_tolerance_m)
    process_held_loss = tolerance_loss(distance_held, cfg.m_tcp_distance_tolerance_m)
    process_expected = _expected_with_absorbing(
        process_live_loss, process_held_loss, survival_probability,
    )
    if horizon > 1:
        process_distance = process_expected[:-1]
        # The path term is averaged over nonterminal intervals, with no
        # duplicated progress or terminal-distance shaping.
        m_path = task_mean(process_distance, m_mask)
    else:
        m_path = states.new_zeros(())
    terminal_live = tolerance_loss(distance_live[-1], cfg.m_tcp_distance_tolerance_m)
    terminal_held = tolerance_loss(distance_held[-1], cfg.m_tcp_distance_tolerance_m)
    terminal_survival = survival_probability[-1] if survival_probability is not None else None
    terminal_loss = _expected_with_absorbing(
        terminal_live[None], terminal_held[None],
        terminal_survival[None] if terminal_survival is not None else None,
    )[0]
    m_terminal = task_mean(terminal_loss[None], m_mask)

    tcp_before = tcp_held
    # For the first row the previous TCP is the actual initial state. For later
    # rows tcp_held is the last safe endpoint, so failure absorption has zero
    # artificial TCP velocity.
    tcp_speed_live = torch.linalg.vector_norm((tcp_live - tcp_before) / cfg.control_dt, dim=-1)
    near_live = torch.sigmoid(
        (cfg.m_near_goal_radius_m - distance_live) / max(cfg.m_near_goal_transition_m, 1.0e-6)
    ).detach()
    near_held = torch.sigmoid(
        (cfg.m_near_goal_radius_m - distance_held) / max(cfg.m_near_goal_transition_m, 1.0e-6)
    ).detach()
    stop_live = near_live * tolerance_loss(tcp_speed_live, cfg.m_tcp_speed_tolerance_m_s)
    stop_held = torch.zeros_like(stop_live)
    stop_expected = _expected_with_absorbing(stop_live, stop_held, survival_probability)
    m_stop = task_mean(stop_expected, m_mask)

    initial_xy = states[0, :, :2]
    base_xy = state[..., :2]
    base_translation = torch.linalg.vector_norm(base_xy - initial_xy[None], dim=-1)
    yaw = heading_yaw(state[..., 3:7])
    yaw0 = heading_yaw(states[0, :, 3:7])
    base_yaw_drift = wrap_to_pi(yaw - yaw0[None]).abs()
    drift_loss = (
        F.relu(base_translation - cfg.m_base_translation_tolerance_m)
        / cfg.m_base_translation_tolerance_m
    ).square() + (
        F.relu(base_yaw_drift - cfg.m_base_yaw_tolerance_rad)
        / cfg.m_base_yaw_tolerance_rad
    ).square()
    zero_command = torch.zeros_like(command)
    live_base_velocity, _, live_base_yaw, _ = velocity_command_errors(state, zero_command)
    held_base_velocity, _, held_base_yaw, _ = velocity_command_errors(held_state, zero_command)
    base_velocity = torch.linalg.vector_norm(live_base_velocity, dim=-1)
    held_base_velocity_norm = torch.linalg.vector_norm(held_base_velocity, dim=-1)
    base_motion_live = (
        tolerance_loss(F.relu(base_velocity - cfg.m_base_velocity_tolerance_m_s), cfg.m_base_velocity_tolerance_m_s)
        + tolerance_loss(F.relu(live_base_yaw.abs() - cfg.m_base_yaw_rate_tolerance_rad_s), cfg.m_base_yaw_rate_tolerance_rad_s)
    )
    base_motion_held = (
        tolerance_loss(F.relu(held_base_velocity_norm - cfg.m_base_velocity_tolerance_m_s), cfg.m_base_velocity_tolerance_m_s)
        + tolerance_loss(F.relu(held_base_yaw.abs() - cfg.m_base_yaw_rate_tolerance_rad_s), cfg.m_base_yaw_rate_tolerance_rad_s)
    )
    held_translation = torch.linalg.vector_norm(held_state[..., :2] - initial_xy[None], dim=-1)
    held_yaw = heading_yaw(held_state[..., 3:7])
    held_yaw_drift = wrap_to_pi(held_yaw - yaw0[None]).abs()
    drift_held = (
        F.relu(held_translation - cfg.m_base_translation_tolerance_m)
        / cfg.m_base_translation_tolerance_m
    ).square() + (
        F.relu(held_yaw_drift - cfg.m_base_yaw_tolerance_rad)
        / cfg.m_base_yaw_tolerance_rad
    ).square()
    m_stability = _expected_with_absorbing(drift_loss, drift_held, survival_probability)
    m_base_motion = _expected_with_absorbing(base_motion_live, base_motion_held, survival_probability)
    m_drift = task_mean(m_stability, m_mask)
    m_motion = task_mean(m_base_motion, m_mask)
    m_task = (
        cfg.m_path_weight * m_path
        + cfg.m_terminal_weight * m_terminal
        + cfg.m_near_goal_speed_weight * m_stop
    )
    m_env_mask = m_mask
    m_progress = torch.linalg.vector_norm(
        fk(states[0, :, :qpos_dim], states[0, :, qpos_dim:])["pos"][:, tcp_index] - fixed_world_target,
        dim=-1,
    ) - torch.linalg.vector_norm(tcp_live[-1] - fixed_world_target, dim=-1)
    m_progress = (m_progress * m_env_mask).sum() / m_env_mask.sum().clamp_min(1.0)
    return {
        "task_l": l_task,
        "task_l_linear_velocity": task_mean(l_linear, l_mask),
        "task_l_yaw_rate": task_mean(l_yaw, l_mask),
        "task_m": m_task,
        "task_progress_m": m_progress,
        "terminal_error_m": (distance_live[-1] * m_env_mask).sum() / m_env_mask.sum().clamp_min(1.0),
        "m_tcp_distance_mean_m": task_mean(distance, m_mask),
        "m_tcp_terminal_distance_m": (distance_live[-1] * m_env_mask).sum() / m_env_mask.sum().clamp_min(1.0),
        "m_tcp_speed_mean_m_s": task_mean(tcp_speed_live, m_mask),
        "m_task_process": m_path,
        "m_task_terminal": m_terminal,
        "m_task_near_goal_speed": m_stop,
        "base_drift_m": task_mean(base_translation, m_mask),
        "m_base_drift": cfg.m_base_drift_weight * (m_drift + cfg.m_base_velocity_weight * m_motion),
        "m_base_stability": m_drift,
        "m_base_motion": m_motion,
        "tcp_distance_sequence": distance,
    }


def _feasibility(states: torch.Tensor, soft_limits: torch.Tensor, cfg: M1JointLossCfg,
                 valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    state = states[1:]
    if valid_mask is None:
        valid = torch.ones(state.shape[:-1], dtype=state.dtype, device=state.device)
    else:
        if valid_mask.shape != state.shape[:-1]:
            raise ValueError("feasibility valid_mask must have shape [H,B]")
        valid = valid_mask.to(device=state.device, dtype=state.dtype)

    def masked_mean(value: torch.Tensor) -> torch.Tensor:
        mask = valid
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

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
    qlim = masked_mean(below.square() + above.square())
    velocity_limits = _joint_velocity_limits(state.device, cfg)
    velocity_violation = F.relu(state[..., 35:57].abs() - velocity_limits) / velocity_limits
    wheel_ids = [M1_JOINTS.index(n) for n in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")]
    wheel_speed = F.relu(state[..., 35:57][..., wheel_ids].abs() - cfg.maximum_wheel_rate_rad_s) / cfg.maximum_wheel_rate_rad_s
    return (masked_mean(low.square()) + masked_mean(tilt_cost.square()) + qlim
            + masked_mean(velocity_violation.square()) + masked_mean(wheel_speed.square()))


def _reference_smoothness_terms(
    initial: torch.Tensor, reference: torch.Tensor, cfg: M1JointLossCfg,
) -> dict[str, torch.Tensor]:
    """Unit-scaled acceleration and jerk regularizers with current state as t=0."""
    from supermimic.motion_prior.reference import decode_future31

    decoded = decode_future31(initial, reference, dt=cfg.control_dt)
    root_path = torch.cat((initial[:, None, :3], decoded[..., :3]), dim=1)
    root_segment_velocity = torch.diff(root_path, dim=1) / cfg.control_dt
    root_vel = torch.cat((initial[:, None, 29:32], root_segment_velocity), dim=1)
    root_acc = torch.diff(root_vel, dim=1) / cfg.control_dt
    root_term = (F.relu(root_acc.norm(dim=-1) - cfg.maximum_root_acceleration_m_s2) / cfg.maximum_root_acceleration_m_s2).square().mean()
    root_jerk = torch.diff(root_acc, dim=1) / cfg.control_dt
    root_jerk_term = (F.relu(root_jerk.norm(dim=-1) - cfg.maximum_root_jerk_m_s3) / cfg.maximum_root_jerk_m_s3).square().mean()
    quat_path = torch.cat((initial[:, None, 3:7], decoded[..., 3:7]), dim=1)
    root_delta = quat_mul(quat_inv(quat_path[:, :-1]), quat_path[:, 1:])
    root_segment_omega = rotvec_from_quat(root_delta) / cfg.control_dt
    root_omega = torch.cat((initial[:, None, 32:35], root_segment_omega), dim=1)
    root_alpha = torch.diff(root_omega, dim=1) / cfg.control_dt
    root_rot_term = (F.relu(root_alpha.norm(dim=-1) - cfg.maximum_root_angular_acceleration_rad_s2)
                     / cfg.maximum_root_angular_acceleration_rad_s2).square().mean()
    root_angular_jerk = torch.diff(root_alpha, dim=1) / cfg.control_dt
    root_angular_jerk_term = (
        F.relu(root_angular_jerk.norm(dim=-1) - cfg.maximum_root_angular_jerk_rad_s3)
        / cfg.maximum_root_angular_jerk_rad_s3
    ).square().mean()
    q_path = torch.cat((initial[:, None, 7:29], decoded[..., 7:29]), dim=1)
    q_segment_vel = torch.diff(q_path, dim=1) / cfg.control_dt
    q_vel = torch.cat((initial[:, None, 35:57], q_segment_vel), dim=1)
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


def _anchor_group_terms(reference: torch.Tensor, initial_reference: torch.Tensor) -> dict[str, torch.Tensor]:
    if reference.shape != initial_reference.shape or reference.shape[-1] != 31:
        raise ValueError("initial Prior anchor must align with Future31")
    return {
        "anchor_root": ((reference[..., :9] - initial_reference[..., :9])
                        / reference.new_tensor([0.5] * 3 + [1.0] * 6)).square().mean(),
        "anchor_leg": ((reference[..., 9:21] - initial_reference[..., 9:21]) / 2.0).square().mean(),
        "anchor_arm": ((reference[..., 21:27] - initial_reference[..., 21:27]) / 2.0).square().mean(),
        "anchor_wheel": ((reference[..., 27:31] - initial_reference[..., 27:31]) / 10.0).square().mean(),
    }


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
    actor_states: torch.Tensor | None = None,
    raw_reference_actions: torch.Tensor | None = None,
    action_diagnostics: dict[str, torch.Tensor] | None = None,
    direct_valid_mask: torch.Tensor | None = None,
    actor_valid_mask: torch.Tensor | None = None,
    direct_safety_mask: torch.Tensor | None = None,
    actor_safety_mask: torch.Tensor | None = None,
    direct_survival_probability: torch.Tensor | None = None,
    actor_survival_probability: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Planner objective with separate direct-WM and Actor task paths."""
    task = _task_planner_terms(
        states, task_type, velocity_command, fixed_world_target, fk, tcp_index,
        cfg, direct_valid_mask, direct_survival_probability,
    )
    task_direct = task["task_l"] + task["task_m"] + task["m_base_drift"]
    if actor_states is None:
        actor_task_terms = {key: states.new_zeros(()) for key in ("task_l", "task_m", "m_base_drift")}
        task_actor = states.new_zeros(())
    else:
        actor_task_terms = _task_planner_terms(
            actor_states, task_type, velocity_command, fixed_world_target, fk, tcp_index,
            cfg, actor_valid_mask, actor_survival_probability,
        )
        task_actor = actor_task_terms["task_l"] + actor_task_terms["task_m"] + actor_task_terms["m_base_drift"]

    from supermimic.motion_prior.reference import decode_future31
    reference_states = decode_future31(initial, reference)
    aligned_reference = reference_states[:, :states.shape[0] - 1]
    physical = reference_tracking_errors(
        states[1:].transpose(0, 1), aligned_reference,
        direct_valid_mask.transpose(0, 1) if direct_valid_mask is not None else None,
    )
    dynamics = (
        0.5 * physical["root_position"]
        + 0.25 * physical["root_rotation"]
        + 0.25 * physical["leg_position"]
        + physical["arm_position"]
        + 0.5 * physical["wheel_velocity"]
    )
    direct_safety_mask = direct_valid_mask if direct_safety_mask is None else direct_safety_mask
    actor_safety_mask = actor_valid_mask if actor_safety_mask is None else actor_safety_mask
    feasibility_direct = _feasibility(states, soft_limits, cfg, direct_safety_mask)
    feasibility_actor = (
        _feasibility(actor_states, soft_limits, cfg, actor_safety_mask)
        if actor_states is not None else feasibility_direct
    )
    feasibility = 0.5 * (feasibility_direct + feasibility_actor)
    smooth_terms = _reference_smoothness_terms(initial, reference, cfg)
    smoothness = torch.stack(tuple(smooth_terms.values())).sum()
    anchor_terms = _anchor_group_terms(reference, initial_reference)
    anchor = torch.stack(tuple(anchor_terms.values())).mean()
    # This is an explicitly named control-effort surrogate. It is not a torque,
    # electrical-energy or mechanical-power estimate and defaults to disabled.
    effort_surrogate = actions.square().mean()
    if actions.shape[0] > 1:
        effort_surrogate = effort_surrogate + 0.25 * torch.diff(actions, dim=0).square().mean()
    energy = actions.new_zeros(())  # no validated differentiable M1 actuator model is available yet
    failure_direct = expected_termination_failure_cost(direct_survival_probability, reference=states)
    failure_actor = expected_termination_failure_cost(actor_survival_probability, reference=states)
    failure = 0.5 * (failure_direct + failure_actor) if actor_states is not None else failure_direct
    out_of_envelope = (
        F.relu(raw_reference_actions.abs() - 1.0).square().mean()
        if raw_reference_actions is not None else actions.new_zeros(())
    )
    total = (
        cfg.planner_task_weight * task_direct
        + cfg.planner_actor_task_weight * task_actor
        + cfg.planner_dynamics_weight * dynamics
        + cfg.feasibility_weight * feasibility
        + cfg.out_of_envelope_weight * out_of_envelope
        + cfg.smoothness_weight * smoothness
        + cfg.anchor_weight * anchor + cfg.energy_weight * energy
        + cfg.control_effort_surrogate_weight * effort_surrogate
        + cfg.termination_failure_weight * failure
    )
    return {
        "total": total,
        "task": task_direct,
        "task_direct": task_direct,
        "task_actor": task_actor,
        "failure_loss": failure,
        "wm_termination_failure_loss": failure,
        "failure_direct": failure_direct,
        "failure_actor": failure_actor,
        "task_l": task["task_l"],
        "task_m": task["task_m"],
        "task_l_actor": actor_task_terms["task_l"],
        "task_m_actor": actor_task_terms["task_m"],
        "task_progress": task["task_progress_m"],
        "terminal_error": task["terminal_error_m"],
        "base_drift": task["base_drift_m"],
        "dynamics": dynamics,
        "feasibility_direct": feasibility_direct,
        "feasibility_actor": feasibility_actor,
        **{f"dynamics_{key}": value for key, value in physical.items()},
        "energy": energy,
        "control_effort_surrogate": effort_surrogate,
        "feasibility": feasibility,
        "smoothness": smoothness,
        **smooth_terms,
        "anchor": anchor,
        **anchor_terms,
        "out_of_envelope": out_of_envelope,
        **(action_diagnostics or {}),
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
    task_type: torch.Tensor | None = None,
    velocity_command: torch.Tensor | None = None,
    fixed_world_target: torch.Tensor | None = None,
    fk=None,
    tcp_index: int = 0,
    reference_actions: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    safety_mask: torch.Tensor | None = None,
    survival_probability: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Actor objective: Task + grouped reference style + small candidate-action BC."""
    if states.shape != reference_states.shape:
        raise ValueError("Actor state trajectory and fixed reference must align exactly")
    parts = reference_tracking_errors(
        states, reference_states,
        valid_mask,
    )
    if valid_mask is None:
        valid = torch.ones(states.shape[:2], dtype=states.dtype, device=states.device)
    else:
        if valid_mask.shape != states.shape[:2]:
            raise ValueError("Actor valid_mask must have shape [H,B]")
        valid = valid_mask.to(device=states.device, dtype=states.dtype)

    if safety_mask is None:
        safety = valid
    else:
        if safety_mask.shape != states.shape[:2]:
            raise ValueError("Actor safety_mask must have shape [H,B]")
        safety = safety_mask.to(device=states.device, dtype=states.dtype)

    def masked_mean(value: torch.Tensor, mask: torch.Tensor = valid) -> torch.Tensor:
        expanded = mask
        while expanded.ndim < value.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(value)
        return (value * expanded).sum() / expanded.sum().clamp_min(1.0)
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
    def safety_mean(value: torch.Tensor) -> torch.Tensor:
        expanded = safety
        while expanded.ndim < value.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(value)
        return (value * expanded).sum() / expanded.sum().clamp_min(1.0)

    stability = safety_mean(low.square()) + safety_mean(tilt_cost.square()) + safety_mean(
        F.relu(limits[:, 0] - q[..., mask]).square()
        + F.relu(q[..., mask] - limits[:, 1]).square()
    )
    action_path = torch.cat((previous_action[None], actions), dim=0)
    action_rate = masked_mean(torch.diff(action_path, dim=0).square())
    qd = state[..., 35:57]
    joint_speed = masked_mean((qd / _joint_velocity_limits(state.device, cfg)).square())
    if qd.shape[0] > 1:
        pair_valid = valid[1:] * valid[:-1]
        joint_acc = masked_mean(
            (torch.diff(qd, dim=0) / cfg.control_dt / 80.0).square(), pair_valid,
        )
    else:
        joint_acc = qd.new_zeros(())
    saturation = masked_mean(F.relu(actions.abs() - 0.95).square())
    control = action_rate + 0.1 * joint_speed + 0.01 * joint_acc + 0.1 * saturation
    bc = masked_mean((actions - reference_actions).square()) if reference_actions is not None else actions.new_zeros(())
    task_terms: dict[str, torch.Tensor] = {}
    if task_type is not None:
        if velocity_command is None or fixed_world_target is None:
            raise ValueError("Actor task loss requires velocity command and fixed world target")
        initial_state = states[0] if initial_state is None else initial_state
        task_states = torch.cat((initial_state.unsqueeze(0), states), dim=0)
        task_terms = _task_planner_terms(
            task_states, task_type, velocity_command, fixed_world_target, fk, tcp_index,
            cfg, valid_mask, survival_probability,
        )
        task = task_terms["task_l"] + task_terms["task_m"] + task_terms["m_base_drift"]
    else:
        task = actions.new_zeros(())
    failure = expected_termination_failure_cost(survival_probability, reference=states)
    task_weighted = cfg.actor_task_weight * task
    bc_weighted = cfg.actor_bc_weight * bc
    stability_weighted = cfg.actor_stability_weight * stability
    control_weighted = cfg.actor_control_weight * control
    failure_weighted = cfg.termination_failure_weight * failure
    total = track + bc_weighted + task_weighted + stability_weighted + control_weighted + failure_weighted
    task_l = task_terms.get("task_l", actions.new_zeros(()))
    task_m = task_terms.get("task_m", actions.new_zeros(()))
    m_base_drift = task_terms.get("m_base_drift", actions.new_zeros(()))
    return {
        "total": total, "tracking": track, "bc": bc, "task": task, **parts,
        "task_weighted": task_weighted,
        "bc_weighted": bc_weighted,
        "safety_weighted": stability_weighted,
        "control_weighted": control_weighted,
        "failure_weighted": failure_weighted,
        "task_L": task_l,
        "task_M": task_m + m_base_drift,
        "task_M_reach": task_m,
        "m_base_drift": m_base_drift,
        "task_L_weighted": cfg.actor_task_weight * task_l,
        "task_M_weighted": cfg.actor_task_weight * (task_m + m_base_drift),
        "safety": stability,
        "stability": stability, "control": control, "failure_loss": failure,
        "wm_termination_failure_loss": failure,
        "action_rate": action_rate, "joint_speed": joint_speed,
        "joint_acceleration": joint_acc, "action_saturation": saturation,
        **{f"{key}_task": value for key, value in task_terms.items() if value.ndim == 0},
    }
