"""Goal-conditioned Go2 chase-ball reward and coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from supermimic.kinematics.rotations import heading_yaw, quat_apply_inverse


@dataclass
class ChaseBallRewardCfg:
    approach_weight: float = 1.5
    approach_speed_clip: float = 1.5
    near_weight: float = 3.0
    tag_weight: float = 15.0
    tag_radius: float = 0.20
    tag_sigma: float = 0.30
    facing_weight: float = 0.5
    upright_weight: float = 0.35
    angular_velocity_weight: float = -0.02
    action_rate_weight: float = -0.02
    joint_deviation_weight: float = -0.01
    minimum_height: float = 0.22
    base_height_weight: float = -5.0
    tag_softness: float = 0.06
    joint_torque_weight: float = -2.0e-5
    joint_acceleration_weight: float = -2.5e-7


def chase_goal_yaw_frame(state: torch.Tensor, ball_state: torch.Tensor) -> torch.Tensor:
    """Return ``[relative ball xyz, ball velocity xyz]`` in the robot yaw frame.

    Inputs use environment-local world coordinates and ``ball_state=[p(3),v(3)]``.
    Roll and pitch intentionally do not rotate the navigation goal.
    """
    return chase_goal_from_pose(state[..., :3], state[..., 3:7], ball_state)


def chase_goal_from_pose(root_pos: torch.Tensor, root_quat: torch.Tensor, ball_state: torch.Tensor) -> torch.Tensor:
    yaw = heading_yaw(root_quat)
    yaw_quat = torch.stack((torch.cos(yaw * 0.5), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw * 0.5)), -1)
    rel = quat_apply_inverse(yaw_quat, ball_state[..., :3] - root_pos)
    vel = quat_apply_inverse(yaw_quat, ball_state[..., 3:6])
    return torch.cat((rel, vel), -1)


def chase_front_position(state: torch.Tensor, front_offset: float = 0.20) -> torch.Tensor:
    """Go2 front reference point in env-local world coordinates."""
    root = state[..., :3]
    yaw = heading_yaw(state[..., 3:7])
    forward = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), -1)
    return root + float(front_offset) * forward


def chase_ball_reward_terms(
    state_next: torch.Tensor,
    ball_state_next: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    default_joint_pos: torch.Tensor,
    cfg: ChaseBallRewardCfg,
    *,
    tag_event: torch.Tensor | None = None,
    state_prev: torch.Tensor | None = None,
    executed_action: torch.Tensor | None = None,
    torque: torch.Tensor | None = None,
    joint_acceleration: torch.Tensor | None = None,
    robot_params=None,
) -> dict[str, torch.Tensor]:
    """Weighted reward rates (the simulator and imagination apply ``dt`` once).

    ``tag_event`` is a one-step hard success event in the real simulator.  When it
    is omitted the differentiable tag approximation is used in imagination.
    """
    ball_pos = ball_state_next[..., :3]
    front = chase_front_position(state_next)
    delta_xy = ball_pos[..., :2] - front[..., :2]
    distance = torch.linalg.vector_norm(delta_xy, dim=-1)
    direction = delta_xy / distance[..., None].clamp_min(1.0e-6)
    nj = action.shape[-1]
    velocity_xy = state_next[..., 7 + nj:9 + nj]
    approach = (velocity_xy * direction).sum(-1).clamp(-cfg.approach_speed_clip, cfg.approach_speed_clip)
    yaw = heading_yaw(state_next[..., 3:7])
    forward_xy = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
    facing = 0.5 * (1.0 + (forward_xy * direction).sum(-1).clamp(-1.0, 1.0))

    quat = state_next[..., 3:7]
    gravity_w = torch.zeros_like(quat[..., :3])
    gravity_w[..., 2] = -1.0
    gravity_b = quat_apply_inverse(quat, gravity_w)
    q = state_next[..., 7:7 + nj]
    qerr = q - default_joint_pos
    near = cfg.near_weight * torch.exp(-distance.square() / (cfg.tag_sigma**2))
    tag = (
        cfg.tag_weight * torch.sigmoid((cfg.tag_radius - distance) / cfg.tag_softness)
        if tag_event is None
        else cfg.tag_weight * tag_event.to(state_next.dtype)
    )
    if torque is None:
        if robot_params is not None:
            from supermimic.tasks.velocity_reward import go2_joint_torque_proxy
            torque = go2_joint_torque_proxy(
                state_next, action if executed_action is None else executed_action,
                robot_params.default_joint_pos, robot_params.action_scales,
                robot_params.kp, robot_params.kd, robot_params.effort_limits,
                robot_params.soft_joint_limits,
            )
        else:
            torque = state_next.new_zeros(state_next.shape[:-1] + (nj,))
    if joint_acceleration is None:
        if state_prev is None:
            joint_acceleration = state_next.new_zeros(state_next.shape[:-1] + (nj,))
        else:
            from supermimic.tasks.velocity_reward import joint_acceleration_from_states
            dt = float(robot_params.dt) if robot_params is not None else 0.02
            joint_acceleration = joint_acceleration_from_states(state_prev, state_next, dt, nj)
    return {
        "approach": cfg.approach_weight * approach,
        "near": near,
        "tag_success": tag,
        "facing": cfg.facing_weight * facing,
        "upright": cfg.upright_weight * torch.exp(-gravity_b[..., :2].square().sum(-1) / 0.16),
        "angular_velocity": cfg.angular_velocity_weight * state_next[..., 10 + nj:12 + nj].square().sum(-1),
        "action_rate": cfg.action_rate_weight * (action - previous_action).square().sum(-1),
        "joint_deviation": cfg.joint_deviation_weight * qerr.square().sum(-1),
        "base_height_low": cfg.base_height_weight * torch.relu(cfg.minimum_height - state_next[..., 2]).square(),
        "joint_torque": cfg.joint_torque_weight * torque.square().sum(-1),
        "joint_acceleration": cfg.joint_acceleration_weight * joint_acceleration.square().sum(-1),
    }


def chase_reward_total(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple(terms.values()), dim=0).sum(0)


def chase_tag_probability(state: torch.Tensor, ball_state: torch.Tensor, cfg: ChaseBallRewardCfg) -> torch.Tensor:
    distance = torch.linalg.vector_norm(chase_front_position(state)[..., :2] - ball_state[..., :2], dim=-1)
    return torch.sigmoid((cfg.tag_radius - distance) / cfg.tag_softness)


def chase_fall_probability(state: torch.Tensor, height: float, tilt_deg: float, softness: float = 0.015) -> torch.Tensor:
    gravity = state.new_zeros(state.shape[:-1] + (3,))
    gravity[..., 2] = -1.0
    from supermimic.kinematics.rotations import quat_apply_inverse

    gravity_local = quat_apply_inverse(state[..., 3:7], gravity)
    tilt = (1.0 + gravity_local[..., 2]).abs()
    tilt_thr = 1.0 - math.cos(math.radians(tilt_deg))
    p_h = torch.sigmoid((float(height) - state[..., 2]) / softness)
    p_t = torch.sigmoid((tilt - tilt_thr) / softness)
    return 1.0 - (1.0 - p_h) * (1.0 - p_t)


def go2_chase_reward_term(
    env,
    term_name: str,
    approach_weight: float = 1.5,
    near_weight: float = 3.0,
    tag_weight: float = 15.0,
    tag_radius: float = 0.20,
    tag_sigma: float = 0.30,
    facing_weight: float = 0.5,
    minimum_height: float = 0.22,
    joint_torque_weight: float = -2.0e-5,
    joint_acceleration_weight: float = -2.5e-7,
) -> torch.Tensor:
    """RewardManager adapter; obtains the same fixed world goal stored by Replay."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_supermimic_chase_reward_cache", None)
    if cache is None or cache["step"] != step:
        from supermimic.envs.mdp.commands import ChaseBallCommand
        from supermimic.envs.state_io import StateIO

        command = env.command_manager.get_term("chase_ball")
        if not isinstance(command, ChaseBallCommand):
            raise TypeError("chase_ball reward requires ChaseBallCommand")
        state_io = getattr(env, "_supermimic_chase_state_io", None)
        if state_io is None:
            state_io = StateIO(env)
            env._supermimic_chase_state_io = state_io
        state_next = state_io.read()
        robot = env.scene["robot"]
        # The command advances the scripted ball by exactly one env dt. Use that
        # same post-step state for reward and the next observation snapshot.
        goal_next = command.goal.clone()
        action = env.action_manager.action.clone()
        previous_action = env.action_manager.prev_action.clone()
        default = env.scene["robot"].data.default_joint_pos[0]
        cfg = ChaseBallRewardCfg(
            approach_weight=float(approach_weight), near_weight=float(near_weight),
            tag_weight=float(tag_weight), tag_radius=float(tag_radius),
            tag_sigma=float(tag_sigma), facing_weight=float(facing_weight),
            minimum_height=float(minimum_height),
            joint_torque_weight=float(joint_torque_weight),
            joint_acceleration_weight=float(joint_acceleration_weight),
        )
        tag_event = command.record_real_step(state_next, goal_next)
        from supermimic.robots.spec import get_robot_spec
        spec = get_robot_spec("go2")
        limits = robot.data.soft_joint_pos_limits[0]
        from supermimic.tasks.velocity_reward import VelocityRobotParams
        robot_params = VelocityRobotParams(
            default_joint_pos=default, soft_joint_limits=limits,
            action_scales=default.new_tensor(spec.action_scales),
            kp=default.new_tensor(spec.kp), kd=default.new_tensor(spec.kd),
            effort_limits=default.new_tensor(spec.effort_limits), dt=float(env.step_dt),
        )
        state_prev = state_next
        recorder_manager = getattr(env, "recorder_manager", None)
        if recorder_manager is not None and "replay" in getattr(recorder_manager, "active_terms", []):
            state_prev = recorder_manager.get_term("replay").prev_s.clone()
        else:
            previous_state = getattr(env, "_supermimic_chase_previous_state", None)
            if previous_state is not None:
                done = getattr(env, "reset_terminated", torch.zeros(state_next.shape[0], dtype=torch.bool, device=state_next.device))
                done = done | getattr(env, "reset_time_outs", torch.zeros_like(done))
                state_prev = torch.where(done[..., None], state_next, previous_state)
        action_term = env.action_manager.get_term("joint_pos")
        executed_action = getattr(action_term, "executed_raw", action).clone()
        measured_torque = robot.data.qfrc_actuator.clone()
        measured_acceleration = robot.data.joint_acc.clone()
        if measured_torque.shape[-1] != len(spec.joint_names) or measured_acceleration.shape[-1] != len(spec.joint_names):
            raise RuntimeError(
                "Go2 actuator diagnostics must expose exactly the 12 RobotSpec joints; "
                f"got torque={tuple(measured_torque.shape)}, acceleration={tuple(measured_acceleration.shape)}"
            )
        terms = chase_ball_reward_terms(
            state_next, goal_next, action, previous_action, default, cfg,
            tag_event=tag_event, state_prev=state_prev,
            executed_action=executed_action, torque=measured_torque,
            joint_acceleration=measured_acceleration, robot_params=robot_params,
        )
        from supermimic.tasks.velocity_reward import go2_joint_torque_proxy, joint_acceleration_from_states
        torque_proxy = go2_joint_torque_proxy(state_next, executed_action, default, robot_params.action_scales,
                                              robot_params.kp, robot_params.kd, robot_params.effort_limits, limits)
        acceleration_fd = joint_acceleration_from_states(state_prev, state_next, robot_params.dt, len(spec.joint_names))
        env._supermimic_chase_previous_state = state_next.clone()
        cache = {
            "step": step, "terms": terms,
            "inputs": {"torque": measured_torque, "joint_acceleration": measured_acceleration,
                        "torque_proxy": torque_proxy, "acceleration_fd": acceleration_fd},
        }
        env._supermimic_chase_reward_cache = cache
    if term_name not in cache["terms"]:
        raise KeyError(f"unknown chase_ball reward term {term_name!r}")
    return cache["terms"][term_name]


CHASE_REWARD_NAMES = (
    "approach", "near", "tag_success", "facing", "upright",
    "angular_velocity", "action_rate", "joint_deviation", "base_height_low",
    "joint_torque", "joint_acceleration",
)
