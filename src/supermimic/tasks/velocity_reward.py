"""Shared Go2 velocity reward math for MuJoCo and differentiable imagination."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.tasks.action_contract import position_setpoint


@dataclass
class VelocityRewardCfg:
    """First-pass reward coefficients; values are reward rates before the env dt."""

    linear_tracking: float = 3.0
    command_direction: float = 0.5
    yaw_tracking: float = 1.2
    upright: float = 0.5
    base_lin_vel_z: float = -1.0
    base_ang_vel_xy: float = -0.05
    action_rate: float = -0.03
    joint_torque: float = -1.0e-5
    joint_acceleration: float = -2.5e-7
    joint_deviation: float = -0.02
    joint_limit: float = -1.0
    base_height_low: float = -5.0
    tracking_sigma_sq: float = 0.25
    upright_sigma_sq: float = 0.16
    min_safe_height: float = 0.22
    action_clip: float = 1.0


@dataclass
class VelocityRobotParams:
    default_joint_pos: torch.Tensor
    soft_joint_limits: torch.Tensor
    action_scales: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    effort_limits: torch.Tensor
    dt: float = 0.02
    cfg: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)


VELOCITY_REWARD_NAMES = (
    "track_lin_vel_xy",
    "command_direction",
    "track_yaw_rate",
    "upright",
    "base_lin_vel_z",
    "base_ang_vel_xy",
    "action_rate",
    "joint_torque",
    "joint_acceleration",
    "joint_deviation",
    "joint_limit",
    "base_height_low",
)


def _joint_torque_proxy(s_next: torch.Tensor, executed_action: torch.Tensor, params: VelocityRobotParams) -> torch.Tensor:
    """Approximate Go2 PD actuator force for imagined states.

    The real reward uses MuJoCo's measured actuator-space force. Imagination has no
    MuJoCo actuator state, so it reconstructs the unclipped position target and applies
    the configured PD law and effort limit. This proxy is differentiable with respect
    to the predicted joint state and actor action.
    """
    return go2_joint_torque_proxy(
        s_next, executed_action, params.default_joint_pos, params.action_scales,
        params.kp, params.kd, params.effort_limits, params.soft_joint_limits,
    )


def go2_joint_torque_proxy(
    state_next: torch.Tensor,
    executed_action: torch.Tensor,
    default_joint_pos: torch.Tensor,
    action_scales: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    effort_limits: torch.Tensor,
    soft_joint_limits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable Go2 PD torque using the same mapping as the real action term."""
    nj = executed_action.shape[-1]
    q = state_next[..., 7:7 + nj]
    qd = state_next[..., 13 + nj:13 + 2 * nj]
    target = position_setpoint(executed_action, default_joint_pos, action_scales)
    if soft_joint_limits is not None:
        target = torch.maximum(torch.minimum(target, soft_joint_limits[..., 1]), soft_joint_limits[..., 0])
    return (kp * (target - q) - kd * qd).clamp(-effort_limits, effort_limits)


def joint_acceleration_from_states(
    state_prev: torch.Tensor, state_next: torch.Tensor, dt: float, num_joints: int | None = None
) -> torch.Tensor:
    """Finite-difference joint acceleration at the control period (never a physics substep)."""
    nj = int(num_joints or ((state_next.shape[-1] - 13) // 2))
    qd_prev = state_prev[..., 13 + nj:13 + 2 * nj]
    qd_next = state_next[..., 13 + nj:13 + 2 * nj]
    return (qd_next - qd_prev) / float(dt)


def velocity_reward_terms(
    s_prev: torch.Tensor,
    s_next: torch.Tensor,
    goal: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    params: VelocityRobotParams,
    *,
    executed_action: torch.Tensor | None = None,
    measured_torque: torch.Tensor | None = None,
    measured_joint_acceleration: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return weighted reward-rate terms, without multiplying by control ``dt``.

    State layout is RobotSpec order: ``[root xyz, quat wxyz, q, lin_vel_world,
    ang_vel_body, qd]``. The real simulator passes measured torque and acceleration;
    imagination supplies neither and uses the explicit PD/finite-difference proxies.
    """
    nj = action.shape[-1]
    q = s_next[..., 7:7 + nj]
    root_quat = s_next[..., 3:7]
    lin_vel_w = s_next[..., 7 + nj:10 + nj]
    ang_vel_b = s_next[..., 10 + nj:13 + nj]
    joint_vel = s_next[..., 13 + nj:13 + 2 * nj]
    lin_vel_b = quat_apply_inverse(root_quat, lin_vel_w)
    goal_lin_error_sq = ((lin_vel_b[..., :2] - goal[..., :2]) ** 2).sum(-1)
    yaw_error_sq = (ang_vel_b[..., 2] - goal[..., 2]).square()
    gravity_w = torch.zeros_like(root_quat[..., :3])
    gravity_w[..., 2] = -1.0
    gravity_b = quat_apply_inverse(root_quat, gravity_w)

    if executed_action is None:
        executed_action = action.clamp(-params.cfg.action_clip, params.cfg.action_clip)
    torque = measured_torque if measured_torque is not None else _joint_torque_proxy(s_next, executed_action, params)
    if measured_joint_acceleration is not None:
        joint_acc = measured_joint_acceleration
    else:
        qd_prev = s_prev[..., 13 + nj:13 + 2 * nj]
        joint_acc = (joint_vel - qd_prev) / float(params.dt)

    limits = params.soft_joint_limits
    below = F.relu(limits[..., 0] - q)
    above = F.relu(q - limits[..., 1])
    q_error = q - params.default_joint_pos
    terms = {
        "track_lin_vel_xy": params.cfg.linear_tracking * torch.exp(-goal_lin_error_sq / params.cfg.tracking_sigma_sq),
        "command_direction": params.cfg.command_direction
        * (torch.linalg.vector_norm(goal[..., :2], dim=-1) / (torch.linalg.vector_norm(goal[..., :2], dim=-1) + 0.1))
        * torch.tanh(
            ((lin_vel_b[..., :2] * goal[..., :2]).sum(-1)
             / torch.linalg.vector_norm(goal[..., :2], dim=-1).clamp_min(1e-4)) / 0.3
        ),
        "track_yaw_rate": params.cfg.yaw_tracking * torch.exp(-yaw_error_sq / params.cfg.tracking_sigma_sq),
        "upright": params.cfg.upright * torch.exp(-gravity_b[..., :2].square().sum(-1) / params.cfg.upright_sigma_sq),
        "base_lin_vel_z": params.cfg.base_lin_vel_z * lin_vel_w[..., 2].square(),
        "base_ang_vel_xy": params.cfg.base_ang_vel_xy * ang_vel_b[..., :2].square().sum(-1),
        "action_rate": params.cfg.action_rate * (action - previous_action).square().sum(-1),
        "joint_torque": params.cfg.joint_torque * torque.square().sum(-1),
        "joint_acceleration": params.cfg.joint_acceleration * joint_acc.square().sum(-1),
        "joint_deviation": params.cfg.joint_deviation * q_error.square().sum(-1),
        "joint_limit": params.cfg.joint_limit * (below.square() + above.square()).sum(-1),
        "base_height_low": params.cfg.base_height_low * F.relu(params.cfg.min_safe_height - s_next[..., 2]).square(),
    }
    return terms


def velocity_reward_total(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([terms[name] for name in VELOCITY_REWARD_NAMES], dim=0).sum(0)


def go2_velocity_reward_term(
    env,
    term_name: str,
    linear_tracking: float = 3.0,
    yaw_tracking: float = 1.2,
    command_direction: float = 0.5,
    min_safe_height: float = 0.22,
) -> torch.Tensor:
    """RewardManager adapter; caches the shared core and records real command metrics."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_supermimic_go2_reward_cache", None)
    if cache is None or cache["step"] != step:
        from supermimic.envs.state_io import StateIO
        from supermimic.envs.mdp.commands import VelocityCommand
        from supermimic.robots.spec import get_robot_spec

        robot = env.scene["robot"]
        spec = get_robot_spec("go2")
        state_io = getattr(env, "_supermimic_go2_state_io", None)
        if state_io is None:
            state_io = StateIO(env)
            env._supermimic_go2_state_io = state_io
        state = state_io.read()
        limits = robot.data.soft_joint_pos_limits[0]
        default = robot.data.default_joint_pos[0]
        params = VelocityRobotParams(
            default_joint_pos=default,
            soft_joint_limits=limits,
            action_scales=default.new_tensor(spec.action_scales),
            kp=default.new_tensor(spec.kp),
            kd=default.new_tensor(spec.kd),
            effort_limits=default.new_tensor(spec.effort_limits),
            dt=float(env.step_dt),
            cfg=VelocityRewardCfg(
                linear_tracking=float(linear_tracking),
                yaw_tracking=float(yaw_tracking),
                command_direction=float(command_direction),
                min_safe_height=float(min_safe_height),
            ),
        )
        command = env.command_manager.get_term("velocity")
        if not isinstance(command, VelocityCommand):
            raise TypeError("Go2 reward requires the local VelocityCommand term")
        action = env.action_manager.action
        action_term = env.action_manager.get_term("joint_pos")
        executed_action = action_term.executed_raw
        previous_action = env.action_manager.prev_action
        torque = robot.data.qfrc_actuator.clone()
        joint_acceleration = robot.data.joint_acc.clone()
        if torque.shape[-1] != len(spec.joint_names) or joint_acceleration.shape[-1] != len(spec.joint_names):
            raise RuntimeError(
                "Go2 actuator diagnostics must expose exactly the 12 RobotSpec joints; "
                f"got torque={tuple(torque.shape)}, acceleration={tuple(joint_acceleration.shape)}"
            )
        goal = command.command.clone()
        action = action.clone()
        executed_action = executed_action.clone()
        previous_action = previous_action.clone()
        state_for_reward = state.clone()
        state_prev = state
        recorder_manager = getattr(env, "recorder_manager", None)
        if recorder_manager is not None and "replay" in getattr(recorder_manager, "active_terms", []):
            state_prev = recorder_manager.get_term("replay").prev_s.clone()
        terms = velocity_reward_terms(
            state_prev, state_for_reward, goal, action, previous_action, params,
            executed_action=executed_action,
            measured_torque=torque,
            measured_joint_acceleration=joint_acceleration,
        )
        torque_proxy = _joint_torque_proxy(state_for_reward, executed_action, params)
        acceleration_fd = joint_acceleration_from_states(state_prev, state_for_reward, params.dt, len(spec.joint_names))
        nj = len(spec.joint_names)
        lin_body = quat_apply_inverse(state[:, 3:7], state[:, 7 + nj:10 + nj])
        command.record_real_step(
            command.command, lin_body[:, :2], state[:, 10 + nj:13 + nj][:, 2], state[:, 2],
            env.reset_terminated,
        )
        cache = {
            "step": step,
            "terms": terms,
            "dt": float(env.step_dt),
            "inputs": {
                "state": state_for_reward,
                "state_prev": state_prev,
                "goal": goal,
                "action": action,
                "executed_action": executed_action,
                "previous_action": previous_action,
                "torque": torque,
                "joint_acceleration": joint_acceleration,
                "torque_proxy": torque_proxy,
                "acceleration_fd": acceleration_fd,
                "params": params,
            },
        }
        env._supermimic_go2_reward_cache = cache
    if term_name not in cache["terms"]:
        raise KeyError(f"unknown Go2 reward term {term_name!r}")
    # RewardManager applies its standard dt scaling exactly once.
    return cache["terms"][term_name]
